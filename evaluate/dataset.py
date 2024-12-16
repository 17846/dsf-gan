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
    def __init__(self, data_root, type="script", size=64):
        # 300 个作者的手写
        self.script_root = os.path.join(data_root, type)

        # 二维数组，每一项存储一个作家所有图片路径
        self.script_list = []
        # 获取所有的作家文件夹
        writer_list = glob.glob('{}/*'.format(self.script_root))
        # tqdm为可视化进度条库
        for writer in tqdm(writer_list, desc='loading dataset'):
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list = self.script_list + character_list
        index = 0
        self.index_dict = {}
        self.char_dict = {}

        with open("/public/home/ncu3/fjl/metascript-main/assert/character.pkl", "rb") as f:
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
            transforms.Resize((size, size)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        # writer 作者编号(数组下标) 对应 script_list 的一维， character 该作者某个字符的下标，对应 script_list 的二维
        char_path = self.script_list[index]

        character_name = os.path.basename(char_path).split(".")[0]
        try:
            index1 = self.swapped_dict[character_name]  # 中文命名
        except KeyError as e:
            index1 = character_name  # 数字文件名
            character_name = self.char_dict[self.index_dict[int(index1)]]  # 数字命名 真实字符

        # 处理需要生成的图片
        script_image = self.transforms(Image.open(char_path))
        # 参考图 作者在script_list的下标 模板图 模板字符在template_list的数组下标 目标图
        # return script_image,torch.tensor(self.index_dict[int(character_name)]),self.charmap[int(character_name)],None
        return script_image, torch.tensor(self.index_dict[int(index1)]), character_name, char_path

    def __len__(self):
        return len(self.script_list)
