import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apex import amp
from dataset import CharacterDataset
from hccr_v3 import HandwritingRecognitionModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import pickle
from datetime import datetime
from tqdm import tqdm

device_ids = [0, 1]

torch.cuda.set_device('cuda:{}'.format(device_ids[0]))

# 创建文件夹
now = datetime.now()
dir_name = now.strftime('train_%Y%m%d_%H%M%S')
dataset_path = ""
save_path = os.path.join("./checkpoint", dir_name)
os.makedirs(save_path, exist_ok=True)

device = torch.device('cuda')
# Load your dataset
dataset = CharacterDataset(dataset_path)
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2)
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

# Define the model, loss function an optimizer
model = HandwritingRecognitionModel(num_classes=3755)

model = model.cuda(device=device_ids[0])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Use AMP for mixed precision training
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = torch.nn.DataParallel(model, device_ids=device_ids)

num_epochs = 10000
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
        if index % 100 == 0:
            loss_list.append(loss.item())
            with open(os.path.join(save_path, 'losses.pkl'), 'wb') as f:
                pickle.dump(loss_list, f)
            batch_iter.set_postfix(loss=loss.item())

        index += 1
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        index = 0
        for images, labels, _ in test_loader:
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
          f"Train Loss: {loss.item():.4f}, "
          f"Validation Accuracy: {100.0 * correct / total:.2f}%")
    print("\n")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_state_dict': amp.state_dict()
    }, os.path.join(save_path, f"checkpoint_epoch_{epoch}_acc_{100.0 * correct / total:.2f}.pth"))
