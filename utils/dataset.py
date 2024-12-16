import os
import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
import pickle
from utils.function import SquarePad, ColorReverse
import re


class CharacterDataset(TensorDataset):
    def __init__(self, data_root, reference_count: int = 1, is_train: bool = True):
        self.reference_count = reference_count
        self.template_root = os.path.join(data_root, 'template')
        self.is_train = is_train
        if is_train:
            self.script_root = os.path.join(data_root, 'script')
        else:
            self.script_root = os.path.join(data_root, 'test')

        self.template_list = glob.glob('{}/*.png'.format(self.template_root))
        self.character_map = {os.path.basename(path): i for i, path in enumerate(self.template_list)}

        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        for writer in tqdm(writer_list, desc='loading dataset'):
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list.append(character_list)

        with open("../assert/character.pkl", "rb") as f:
            charmap = pickle.load(f)
        self.index_dict = {}
        index = 0
        for item in charmap:
            self.index_dict[item] = index
            index += 1
        self.remap_list = []
        for writer in range(len(self.script_list)):
            for character in range(len(self.script_list[writer])):
                self.remap_list.append((writer, character))

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            ColorReverse(),
            SquarePad(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        writer, character = self.remap_list[index]
        reference_path = np.random.choice(self.script_list[writer], self.reference_count, replace=True)
        script_path = self.script_list[writer][character]
        character_name = os.path.basename(script_path)
        template_path = os.path.join(self.template_root, character_name)
        writer_label = torch.tensor(writer)
        character_label = self.index_dict[int(character_name[:-4])]
        character_label = torch.tensor(character_label)
        if not self.is_train:
            match = re.search(r'C(\d+)-f-f', script_path)
            number = match.group(1)
            writer_label = torch.tensor(int(number))
        reference_image = torch.concat([self.transforms(Image.open(path)) for path in reference_path])
        template_image = self.transforms(Image.open(template_path))
        script_image = self.transforms(Image.open(script_path))
        pattern = r"(\d+)\.png$"
        match = re.search(pattern, character_name)
        chinese_char = match.group(1)

        return reference_image, writer_label, template_image, character_label, script_image, chinese_char

    def __len__(self):
        return len(self.remap_list)

    @property
    def writer_count(self):
        return len(self.script_list)

    @property
    def character_count(self):
        return len(self.template_list)
