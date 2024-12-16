import os
import sys
import time
import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.dataset import CharacterDataset
from utils.function import plot_sample
from model.generator import SynthesisGenerator
from model.loss import SupConLoss, EdgeLoss
from model.discriminator import MultiScaleDiscriminatorV2
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from hccr.hccr_v3 import HandwritingRecognitionModel
from utils.function import find_folders


class NCELoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(NCELoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        # 计算anchor和positive之间的相似度
        pos_sim = F.cosine_similarity(anchor, positive)
        # 计算anchor和negatives之间的相似度
        neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives).mean(1)
        # 计算NCE损失
        losses = F.relu(self.margin + neg_sim - pos_sim).mean()
        return losses


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    world_size = 2  # number of GPUs for your case
    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)


def main_worker(index, world_size, config):
    checkpoint_max_iter_path = find_folders()
    torch.cuda.set_device(index)
    device = torch.device("cuda:{}".format(index))

    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=index)

    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    device = torch.device(f'cuda:{index}')
    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    reference_count = int(config.parameter.reference_count)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)
    ocr_path = config.parameter.ocr_path


    # create logger

    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)

    # load dataset
    dataset = CharacterDataset(dataset_path, reference_count=reference_count)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            drop_last=True, sampler=sampler)

    # the rest of stuff up until model creation...

    generator_model = SynthesisGenerator(reference_count=reference_count, batch_size=batch_size).to(device)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    generator_model.load_state_dict(
        torch.load(os.path.join(checkpoint_max_iter_path, "generator.pth"), map_location=map_location))
    generator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator_model)

    generator_model = DistributedDataParallel(generator_model, device_ids=[index], find_unused_parameters=True)
    generator_model.train()

    discriminator_model = MultiScaleDiscriminatorV2().to(device)
    discriminator_model.load_state_dict(
        torch.load(os.path.join(checkpoint_max_iter_path, "discriminator.pth"), map_location=map_location))
    discriminator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator_model)
    discriminator_model = DistributedDataParallel(discriminator_model, device_ids=[index], find_unused_parameters=True)
    discriminator_model.train()

    ocr_model = HandwritingRecognitionModel()
    for param in ocr_model.parameters():
        param.requires_grad = False
    checkpoint = torch.load(ocr_path, map_location=map_location)
    ocr_model.load_state_dict(checkpoint['model_state_dict'])
    ocr_model = ocr_model.cuda()
    ocr_model.eval()
    if index == 0:
        os.makedirs(checkpoint_path, exist_ok=True)

        sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
        print(OmegaConf.to_yaml(config))
        print('image number: {}\n'.format(len(dataset)))
        print("checkpoint_max_iter_path", checkpoint_max_iter_path)

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=config.parameter.generator.learning_rate,
                               betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=config.parameter.discriminator.learning_rate,
                                   betas=(0, 0.999), weight_decay=1e-4)
    adam_dict = torch.load(os.path.join(checkpoint_max_iter_path, "adam.pth"), map_location=map_location)
    generator_optimizer.load_state_dict(adam_dict["generator_adam"])
    discriminator_optimizer.load_state_dict(adam_dict["dim_adam"])
    # start training
    current_iteration = 0
    current_time = time.time()
    nce_loss = SupConLoss()
    edge_loss = EdgeLoss(device=index).cuda(device=device)
    for reference_image, writer_label, template_image, character_label, script_image, _ in dataloader:

        reference_image, writer_label, template_image, character_label, script_image = reference_image.cuda(
            index), writer_label.cuda(index), template_image.cuda(index), character_label.cuda(
            index), script_image.cuda(index)

        # generator
        generator_optimizer.zero_grad()

        result_image, template_structure, nce_emb, nce_emb_patch = generator_model(reference_image, template_image)

        dis_fake = discriminator_model(result_image)
        loss_generator_adversarial = F.binary_cross_entropy(dis_fake, torch.ones_like(dis_fake))

        prediction_character = ocr_model(result_image)

        loss_generator_classification = F.cross_entropy(prediction_character, character_label)

        result_structure = generator_model.module.structure(result_image)
        loss_generator_structure = 0
        for i in range(len(result_structure)):
            loss_generator_structure += 0.5 * torch.mean(torch.square(template_structure[i] - result_structure[i]))
        nce_writer = nce_loss(nce_emb, writer_label)
        nce_glyph = nce_loss(nce_emb_patch)
        # loss_style = style_loss(script_image,result_image)
        loss_generator_style = nce_writer + nce_glyph

        eg_loss = edge_loss(result_image, script_image)
        # loss_generator_reconstruction =  eg_loss + F.l1_loss(result_image, script_image) * 0.2
        loss_generator_reconstruction = eg_loss

        loss_generator = config.parameter.generator.loss_function.weight_adversarial * loss_generator_adversarial + config.parameter.generator.loss_function.weight_classification * loss_generator_classification + config.parameter.generator.loss_function.weight_structure * loss_generator_structure + config.parameter.generator.loss_function.weight_style * loss_generator_style + config.parameter.generator.loss_function.weight_reconstruction * loss_generator_reconstruction
        loss_generator.backward()
        generator_optimizer.step()
        reduce_loss(loss_generator, index, world_size)

        discriminator_optimizer.zero_grad()
        dis_real = discriminator_model(script_image)
        dis_fake = discriminator_model(result_image.detach())
        loss_discriminator_adversarial = F.binary_cross_entropy(dis_fake,
                                                                torch.zeros_like(dis_fake)) + F.binary_cross_entropy(
            dis_real, torch.ones_like(dis_real))

        loss_discriminator = config.parameter.discriminator.loss_function.weight_adversarial * loss_discriminator_adversarial
        loss_discriminator.backward()
        discriminator_optimizer.step()
        reduce_loss(loss_discriminator, index, world_size)

        # report
        if current_iteration % report_interval == 0 and index == 0:
            last_time = current_time
            current_time = time.time()
            iteration_time = (current_time - last_time) / report_interval

            print('iteration {} / {}:'.format(current_iteration, num_iterations))
            print('time: {:.6f} seconds per iteration'.format(iteration_time))
            print(
                'generator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}, structure loss: {:.6f}, style loss1: {:.6f}, style loss2: {:.6f}, reconstruction loss: {:.6f}'.format(
                    loss_generator.item(), loss_generator_adversarial.item(), loss_generator_classification.item(),
                    loss_generator_structure.item(), nce_writer.item(), nce_glyph.item(),
                    loss_generator_reconstruction.item()))
            print('discriminator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}\n'.format(
                loss_discriminator.item(), loss_discriminator_adversarial.item(), 0))

        # save
        if current_iteration % save_interval == 0 and index == 0 and current_iteration > 0:
            save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))

            image_path = os.path.join(checkpoint_path, 'sample_{}.png'.format(current_iteration))
            generator_path = os.path.join(save_path, 'generator.pth')
            discriminator_path = os.path.join(save_path, 'discriminator.pth')
            adam_path = os.path.join(save_path, 'adam.pth')

            image = plot_sample(reference_image, template_image, script_image, result_image)[0]
            Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
            if current_iteration % (save_interval * 4) == 0 and index == 0 and current_iteration > 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(generator_model.module.state_dict(), generator_path)
                torch.save(discriminator_model.module.state_dict(), discriminator_path)
                torch.save({
                    "generator_adam": generator_optimizer.state_dict(),
                    "dim_adam": discriminator_optimizer.state_dict()
                }, adam_path)

                print('save sample image in: {}'.format(image_path))
                print('save generator model in: {}'.format(generator_path))
                print('save discriminator model in: {}\n'.format(discriminator_path))

        if current_iteration >= num_iterations:
            break
        current_iteration += 1


if __name__ == '__main__':
    main()
