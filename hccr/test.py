import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apex import amp
from dataset import CharacterDataset
from model import GaborGoogLeNet
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import torch.distributed as dist
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
# Load your dataset
dataset = CharacterDataset("/public/home/ncu3/fjl/metascript-main/v2/checkpoint/test/test_20240317122850","")
char_dict = dataset.char_dict
data_loader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=16)

# Define the model, loss function an optimizer
model = GaborGoogLeNet(charset_size=3755)

re_train = "/public/home/ncu3/fjl/metascript-main/hccr/checkpoint/train_20240313_163018/checkpoint_epoch_1_acc_96.74.pth"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

# Use AMP for mixed precision training
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
if re_train is not None:
    checkpoint = torch.load(re_train,map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Training loop

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels,chars in tqdm(data_loader, desc='测试进度'):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        index = predicted[0].detach().to("cpu").item()
        print("TRUE",chars[0])
        print("PRED:",char_dict[index])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            

    print(f"Validation Accuracy: {100.0*correct/total:.2f}%")