import os
import glob

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import pickle
import torch.nn.functional as F


class SquarePad(object):
    def __call__(self, image):
        _, width, height = image.shape
        target_size = max(width, height)
        pad_width = (target_size - width) // 2 + 10
        pad_height = (target_size - height) // 2 + 10
        return F.pad(image, (pad_width, pad_height, pad_width, pad_height), 'constant', 0)


class ColorReverse(object):
    def __call__(self, image):
        image = 1 - image
        image /= image.max()
        return image


class CharacterDataset(TensorDataset):
    def __init__(self, data_root, type="script"):
        self.script_root = os.path.join(data_root, type)
        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        for writer in tqdm(writer_list, desc='loading dataset'):
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list = self.script_list + character_list
        index = 0
        self.index_dict = {}
        self.char_dict = {}

        with open("../assert/character.pkl", "rb") as f:
            charmap = pickle.load(f)
        self.charmap = charmap

        self.swapped_dict = {v: k for k, v in self.charmap.items()}

        for item in charmap:
            self.index_dict[item] = index
            self.char_dict[index] = charmap[item]
            index += 1
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            ColorReverse(),
            SquarePad(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        char_path = self.script_list[index]

        character_name = os.path.basename(char_path).split(".")[0]
        index1 = self.swapped_dict[character_name]
        script_image = self.transforms(Image.open(char_path))
        return script_image, torch.tensor(self.index_dict[int(index1)]), character_name

    def __len__(self):
        return len(self.script_list)
