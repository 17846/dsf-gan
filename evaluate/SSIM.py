from utils.ssim import SSIM
from torchvision import transforms
from utils.function import *
import pickle
from torch.utils.data import TensorDataset
import os
import glob
from tqdm import tqdm

from torch.utils.data import DataLoader


class CharacterDataset(TensorDataset):
    def __init__(self, data_root, pre_root):
        self.script_root = data_root
        self.pre_root = pre_root
        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        print(len(writer_list))
        for writer in writer_list:
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list = self.script_list + character_list
        index = 0

        with open("../assert/character.pkl", "rb") as f:
            charmap = pickle.load(f)
        self.charmap = charmap

        self.swapped_dict = {v: k for k, v in self.charmap.items()}
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
        dir_ = char_path.split("/")[-2]
        pre_char_path = os.path.join(self.pre_root,
                                     "C{}-f-f/{}.png".format(dir_.zfill(3), self.swapped_dict[character_name]))
        img1 = self.transforms(Image.open(char_path))
        img2 = self.transforms(Image.open(pre_char_path))
        return img1, img2

    def __len__(self):
        return len(self.script_list)


class SDTCharacterDataset(TensorDataset):
    def __init__(self, data_root, pre_root):
        self.script_root = data_root
        self.pre_root = pre_root
        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        for writer in writer_list:
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list = self.script_list + character_list

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            # ColorReverse(),
            # SquarePad(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        char_path = self.script_list[index]
        character_name = os.path.basename(char_path).split(".")[0]

        dir_ = char_path.split("/")[-2]
        dir_ = int(dir_) + 1
        pre_char_pkl = os.path.join(self.pre_root, "C{}-f.pkl".format(str(dir_).zfill(3)))
        with open(pre_char_pkl, "rb") as f:
            char_pkl = pickle.load(f)
        image = None
        for item in char_pkl:
            if item['label'] == character_name:
                image = item['img']
                break

        img1 = self.transforms(Image.open(char_path))
        img2 = self.transforms(Image.fromarray(image))
        return img1, img2

    def __len__(self):
        return len(self.script_list)

# generate sample path
gen_root = ""
# test file path
dataset_root = "../dataset/test"
dataset = CharacterDataset(gen_root, dataset_root)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)
ssim_loss = SSIM(window_size=11)

ssim_loss_list = []
for img1, img2 in tqdm(dataloader, desc="计算SSIM"):
    img1 = img1.cuda()
    img2 = img2.cuda()
    ssim_loss_list.append(ssim_loss(img1, img2))

print("平均SSIM为：{}".format(sum(ssim_loss_list) / len(ssim_loss_list)))
