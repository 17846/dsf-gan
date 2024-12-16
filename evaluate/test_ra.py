import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apex import amp
from torchvision import models
from dataset import CharacterDataset
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda')
# Load your dataset
generate_path = ""
checkpoint_path = ""
dataset = CharacterDataset(generate_path, "",
                           size=64)
char_dict = dataset.char_dict
data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)

# Define the model, loss function an optimizer
# model = HandwritingRecognitionModel()
model = models.densenet121(pretrained=False)
original_first_layer = model.features.conv0
model.features.conv0 = nn.Conv2d(1,
                                 original_first_layer.out_channels,
                                 kernel_size=original_first_layer.kernel_size,
                                 stride=original_first_layer.stride,
                                 padding=original_first_layer.padding,
                                 bias=False)
num_classes = 3755
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Use AMP for mixed precision training
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
checkpoint = torch.load(checkpoint_path, map_location="cuda")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Training loop
error_list = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, chars, char_path in tqdm(data_loader, desc='测试进度'):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        index = predicted[0].detach().to("cpu").item()
        print("TRUE", chars[0])
        print("PRED:", char_dict[index])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100.0 * correct / total:.2f}%")
