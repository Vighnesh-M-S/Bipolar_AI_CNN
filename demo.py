from models.ssd_resnet import SSDResNet
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

model = SSDResNet(num_classes=21)
model.load_state_dict(torch.load("ssd_epoch10.pth"))
model.eval()

img = Image.open("demo.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    class_preds, box_preds = model(input_tensor)

# Visualize one prediction
fig, ax = plt.subplots(1)
ax.imshow(img)
# Dummy box
box = [50, 50, 150, 150]
rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                         linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
