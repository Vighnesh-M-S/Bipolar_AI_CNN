import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.ssd_resnet import SSDResNet
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
import torchvision
from utils import compute_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor()
])

dataset = VOCDetection(root="./data", year="2007", image_set="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = SSDResNet(num_classes=21).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = images.to(device)

        class_preds, box_preds = model(images)
        gt_classes = torch.randint(0, 21, (images.size(0), class_preds.size(1))).to(device)
        gt_boxes = torch.rand(images.size(0), box_preds.size(1), 4).to(device)

        cls_loss = F.cross_entropy(class_preds.view(-1, 21), gt_classes.view(-1))
        box_loss = F.smooth_l1_loss(box_preds, gt_boxes)

        loss = cls_loss + box_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), f"ssd_epoch{epoch+1}.pth")
