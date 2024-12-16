import os
import sys
import glob
import pickle
import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torchvision import transforms

from utils.logger import Logger
from utils.function import SquarePad, ColorReverse, RecoverNormalize
from model.generator import SynthesisGenerator


@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    reference_path = str(config.parameter.reference_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    reference_count = int(config.parameter.reference_count)
    target_text = str(config.parameter.target_text)
    template_path = str(config.parameter.template_path)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'inference.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    # create transform
    input_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        ColorReverse(),
        SquarePad(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    output_transform = transforms.Compose([
        RecoverNormalize(),
        transforms.Resize((64, 64)),
        ColorReverse(),
        transforms.ToPILImage()
    ])
    align_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
    ])

    # fetch reference
    reference_list = []
    file_list = glob.glob('{}/*.png'.format(reference_path))
    for file in tqdm(file_list, desc='fetching reference'):
        image = Image.open(file)
        reference_list.append(image)
    while len(reference_list) < reference_count:
        reference_list.extend(reference_list)
    reference_list = reference_list[720:720 + reference_count]
    reference_image = [np.array(align_transform(image)) for image in reference_list]
    reference_image = np.concatenate(reference_image, axis=1)
    Image.fromarray(reference_image).save(os.path.join(checkpoint_path, 'reference.png'))
    reference = [input_transform(image) for image in reference_list]
    reference = torch.cat(reference, dim=0).unsqueeze(0).to(device)
    print('fetch {} reference images\n'.format(reference_count))

    # load dictionary
    with open('./assert/character.pkl', 'rb') as file:
        character_map = pickle.load(file)
    character_remap = {value: key for key, value in character_map.items()}
    with open('./assert/punctuation.pkl', 'rb') as file:
        punctuation_map = pickle.load(file)
    punctuation_remap = {value: key for key, value in punctuation_map.items()}
    print('load dictionary from archive\n')

    # generate script
    images = []
    for word in tqdm(target_text, desc='generating script'):
        if word in character_remap.keys():
            image = Image.open(os.path.join(template_path,
                                            '{}.png'.format(character_remap[word])))
            template = input_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                result, _, _, _ = generator_model(reference, template)
                images.append(output_transform(result.view(1, 64, 64)))
    for i, img in enumerate(images):
        img.save("./figure/plt_{}.png".format(i))


if __name__ == '__main__':
    main()
