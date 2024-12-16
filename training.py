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
from hccr.hccr_v3 import HandwritingRecognitionModel
import torch.nn as nn


def gram_matrix(input):
    a, b = input.size()  # a=batch size(=1), b=512 (feature dimension)
    features = input.view(a, b)  # reshape the tensor
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b)  # normalize the gram matrix by dividing by the total number of elements


class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()

    def forward(self, generate_feature, target_feature):
        target = gram_matrix(target_feature).to(device="cuda")
        G = gram_matrix(generate_feature).to(device="cuda")
        loss = nn.MSELoss()(G, target)
        return loss


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    reference_count = int(config.parameter.reference_count)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)
    device_ids = config.parameter.device_ids
    ocr_path = config.parameter.ocr_path
    is_continue = config.parameter.is_continue
    continue_pth = config.parameter.continue_pth

    device = torch.device(f'cuda:{device_ids[0]}')
    torch.cuda.set_device(device)

    # create logger

    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)

    # checkpoint_max_iter_path = None
    os.makedirs(checkpoint_path, exist_ok=True)

    # load dataset
    dataset = CharacterDataset(dataset_path, reference_count=reference_count)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # the rest of stuff up until model creation...

    generator_model = SynthesisGenerator(reference_count=reference_count, batch_size=batch_size).to(device)
    generator_model.train()

    discriminator_model = MultiScaleDiscriminatorV2().to(device)
    discriminator_model.train()

    ocr_model = HandwritingRecognitionModel()
    for param in ocr_model.parameters():
        param.requires_grad = False

    checkpoint = torch.load(ocr_path)
    ocr_model.load_state_dict(checkpoint['model_state_dict'])
    ocr_model = ocr_model.cuda()
    ocr_model.eval()

    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
    print(OmegaConf.to_yaml(config))
    print('image number: {}\n'.format(len(dataset)))

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=config.parameter.generator.learning_rate,
                               betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=config.parameter.discriminator.learning_rate,
                                   betas=(0, 0.999), weight_decay=1e-4)
    # continue training
    if is_continue:
        adam_dict = torch.load(os.path.join(continue_pth, "adam.pth"))
        generator_optimizer.load_state_dict(adam_dict["generator_adam"])
        discriminator_optimizer.load_state_dict(adam_dict["dim_adam"])
        discriminator_model.load_state_dict(torch.load(os.path.join(continue_pth, "discriminator.pth")))
        generator_model.load_state_dict(torch.load(os.path.join(continue_pth, "generator.pth")))

    generator_model = torch.nn.DataParallel(generator_model, device_ids=device_ids, output_device=device)
    discriminator_model = torch.nn.DataParallel(discriminator_model, device_ids=device_ids, output_device=device)

    current_iteration = 0
    current_time = time.time()
    nce_loss = SupConLoss()
    edge_loss = EdgeLoss(device=device).cuda(device=device)
    gram_style = GramStyleLoss()

    for reference_image, writer_label, template_image, character_label, script_image, _ in dataloader:
        reference_image, writer_label, template_image, character_label, script_image = reference_image.cuda(
            device), writer_label.cuda(device), template_image.cuda(device), character_label.cuda(
            device), script_image.cuda(device)

        # generator
        generator_optimizer.zero_grad()

        result_image, template_structure, reference_style, nce_emb, writer_style = generator_model(reference_image,
                                                                                                   template_image)

        dis_fake = discriminator_model(result_image)
        loss_generator_adversarial = F.binary_cross_entropy(dis_fake, torch.ones_like(dis_fake))

        if current_iteration <= 100000:
            prediction_character = ocr_model(result_image)
            loss_generator_classification = F.cross_entropy(prediction_character, character_label)

            result_structure = generator_model.module.structure(result_image)
            loss_generator_structure = 0
            for i in range(len(result_structure)):
                loss_generator_structure += 0.5 * torch.mean(torch.square(template_structure[i] - result_structure[i]))
            nce_writer = nce_loss(nce_emb, writer_label)
            result_style = generator_model.module.style(result_image)
            cus_style = gram_style(result_style, reference_style) * 1e4

            eg_loss = edge_loss(result_image, script_image)
            loss_generator_reconstruction = eg_loss

            loss_generator = config.parameter.generator.loss_function.weight_adversarial * loss_generator_adversarial + config.parameter.generator.loss_function.weight_classification * loss_generator_classification + config.parameter.generator.loss_function.weight_structure * loss_generator_structure + config.parameter.generator.loss_function.weight_style * nce_writer + config.parameter.generator.loss_function.weight_reconstruction * loss_generator_reconstruction + cus_style + style_class_loss_cal
        else:
            loss_generator = config.parameter.generator.loss_function.weight_adversarial * loss_generator_adversarial

        loss_generator.backward()
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        dis_real = discriminator_model(script_image)
        dis_fake = discriminator_model(result_image.detach())
        loss_discriminator_adversarial = F.binary_cross_entropy(dis_fake,
                                                                torch.zeros_like(dis_fake)) + F.binary_cross_entropy(
            dis_real, torch.ones_like(dis_real))

        loss_discriminator = config.parameter.discriminator.loss_function.weight_adversarial * loss_discriminator_adversarial
        loss_discriminator.backward()
        discriminator_optimizer.step()

        if current_iteration % report_interval == 0:
            last_time = current_time
            current_time = time.time()
            iteration_time = (current_time - last_time) / report_interval

            print('iteration {} / {}:'.format(current_iteration, num_iterations))
            print('time: {:.6f} seconds per iteration'.format(iteration_time))

            if current_iteration <= 100000:
                print(
                    'generator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}, structure loss: {:.6f}, style loss1: {:.6f}, style loss2: {:.6f}, reconstruction loss: {:.6f}'.format(
                        loss_generator.item(), loss_generator_adversarial.item(), loss_generator_classification.item(),
                        loss_generator_structure.item(), nce_writer.item(), cus_style.item(),
                        loss_generator_reconstruction.item()))
            else:
                print('generator loss: {:.6f}, adversarial loss: {:.6f}'.format(loss_generator.item(),
                                                                                loss_generator_adversarial.item()))

            print('discriminator loss: {:.6f}, adversarial loss: {:.6f}\n'.format(loss_discriminator.item(), 0))
            # save
        if current_iteration % save_interval == 0 and current_iteration > 0:
            save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))

            image_path = os.path.join(checkpoint_path, 'sample_{}.png'.format(current_iteration))
            generator_path = os.path.join(save_path, 'generator.pth')
            discriminator_path = os.path.join(save_path, 'discriminator.pth')
            adam_path = os.path.join(save_path, 'adam.pth')
            style_path = os.path.join(save_path, 'style.pth')

            image = plot_sample(reference_image, template_image, script_image, result_image)[0]
            Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
            if current_iteration % (save_interval * 10) == 0 and current_iteration > 0:
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
