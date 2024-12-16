import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apex import amp
from utils.dataset import CharacterDataset
from torchvision import models
import os
import pickle
from datetime import datetime
from tqdm import tqdm

device_ids = [0]
torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
root_path = "./densenet/checkpoint"

# 创建文件夹
now = datetime.now()
dir_name = now.strftime('train_%Y%m%d_%H%M%S')
save_path = os.path.join(root_path, dir_name)
os.makedirs(save_path, exist_ok=True)

device = torch.device('cuda')
# Load your dataset
dataset = CharacterDataset("../dataset")
testset = CharacterDataset("../dataset", type="test")

train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=16)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=16)

# Define the model, loss function an optimizer
# model = GaborGoogLeNet(charset_size=3755)
model = models.densenet121(pretrained=True)
original_first_layer = model.features.conv0
model.features.conv0 = nn.Conv2d(1,
                                 original_first_layer.out_channels,
                                 kernel_size=original_first_layer.kernel_size,
                                 stride=original_first_layer.stride,
                                 padding=original_first_layer.padding,
                                 bias=False)
num_classes = 3755
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model = model.cuda(device=device_ids[0])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Use AMP for mixed precision training
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

model = torch.nn.DataParallel(model, device_ids=device_ids)

num_epochs = 1000
loss_list = []
# Training loop
for epoch in range(num_epochs):
    model.train()
    index = 0
    batch_iter = tqdm(train_loader, desc="batch train")
    batch_iter.set_description('Epoch %i' % epoch)
    for data in batch_iter:
        images, labels = data[0].cuda(device=device_ids[0]), data[1].cuda(device=device_ids[0])
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        batch_iter.set_postfix(loss=loss.item())

        if index % 100 == 0:
            loss_list.append(loss.item())
            with open(os.path.join(save_path, 'losses.pkl'), 'wb') as f:
                pickle.dump(loss_list, f)

        index += 1
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        index = 0
        for images, labels, _, _ in test_loader:
            images = images.cuda(device=device_ids[0])
            labels = labels.cuda(device=device_ids[0])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            index += 1
            if index % 100 == 0:
                print('Epoch [{}/{}], Step [{}], Accuracy: {:.2f}'
                      .format(epoch + 1, num_epochs, index + 1, 100.0 * correct / total))

    print("\n")
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          #   f"Train Loss: {loss.item():.4f}, "
          f"Validation Accuracy: {100.0 * correct / total:.2f}%")
    print("\n")
    torch.save({
        'model_state_dict': model.module.state_dict(),
    }, os.path.join(save_path, f"checkpoint_epoch_{epoch}_acc_{100.0 * correct / total:.2f}.pth"))
