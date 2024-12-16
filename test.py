import os
import sys
import time

import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from utils.function import ColorReverse, RecoverNormalize
import pickle
from utils.logger import Logger
from utils.dataset import CharacterDataset
from model.generator import SynthesisGenerator
import tqdm


@hydra.main(version_base=None, config_path='./config', config_name='test')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    reference_count = int(config.parameter.reference_count)
    pre = bool(config.parameter.pre)
    dataset_path = str(config.parameter.dataset_path)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'test.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # load dataset
    dataset = CharacterDataset(dataset_path, reference_count=reference_count,
                               is_train=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    print('image number: {}\n'.format(len(dataset)))

    # create model
    generator_model = SynthesisGenerator().to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    with open('./assert/character.pkl', 'rb') as file:
        character_map = pickle.load(file)

    data_iter = iter(dataloader)

    output_transform = transforms.Compose([
        RecoverNormalize(),
        transforms.Resize((64, 64)),
        ColorReverse(),
        transforms.ToPILImage()
    ])
    with torch.no_grad():
        for _ in tqdm.tqdm(range(len(dataset) // 64)):
            reference_image, writer_label, template_image, character_label, script_image, chinese_char = next(data_iter)
            reference_image, writer_label, template_image, character_label, script_image = reference_image.to(
                device), writer_label.to(device), template_image.to(device), character_label.to(
                device), script_image.to(device)
            result, template_structure, reference_style, nce_emb, _ = generator_model(reference_image, template_image)

            writer_label = writer_label.detach().cpu()
            character_label = character_label.detach().cpu()

            for i in range(len(script_image)):
                # im = result[i]
                # im = output_transform(im)
                path = os.path.join(checkpoint_path, str(writer_label[i].item()))
                if not os.path.exists(path):
                    # 如果目录不存在，则创建它
                    os.makedirs(path)
                try:
                    cr = character_map[int(chinese_char[i])]
                except Exception as e:
                    print(e)
                    sys.exit(0)
                if pre:
                    new_img = Image.new('RGB', (128, 64)).to(device)
                    new_img.paste(im, (0, 0))
                    new_img.paste(output_transform(script_image[i]), (64, 0))
                    im.save(os.path.join(path, "{}.png".format(cr)))

                else:
                    im = result[i]
                    im = output_transform(im)
                    im.save(os.path.join(path, "{}.png".format(cr)))


if __name__ == '__main__':
    main()
