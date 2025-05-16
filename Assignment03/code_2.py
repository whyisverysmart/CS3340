import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw


train_dir = 'data/images/train'
train_ann = 'data/annotations/instances_train.json'
test_dir = 'data/images/test'
test_ann = 'data/annotations/instances_test.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = CocoDetection(train_dir, train_ann, transforms=CocoTransform())
test_dataset = CocoDetection(test_dir, test_ann, transforms=CocoTransform())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(num_classes=2)  # heart + background
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)


def train_epoch(model, loader):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(loader):
        images = list(img.to(device) for img in images)
        new_targets = []
        for i, t in enumerate(targets):
            boxes = torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]  # [x,y,w,h] → [x1,y1,x2,y2]
            labels = torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64)
            new_targets.append({'boxes': boxes.to(device), 'labels': labels.to(device)})
        loss_dict = model(images, new_targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    return epoch_loss / len(loader)

epochs = 10
loss_list = []
for epoch in range(epochs):
    epoch_loss = train_epoch(model, train_loader)
    loss_list.append(epoch_loss)
    print(f"Epoch {epoch+1} done.")

plt.figure()
plt.plot(range(1, epochs + 1), loss_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.xticks(range(1, epochs + 1))
plt.savefig("output_imgs/loss_curve1.png")

def evaluate(model, data_loader):
    model.eval()
    coco_gt = data_loader.dataset.coco
    coco_results = []
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(data_loader)):
            image = images[0].to(device)
            img_id = targets[0][0]['image_id']
            outputs = model([image])[0]
            for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                box = box.cpu().numpy()
                x1, y1, x2, y2 = box
                coco_results.append({
                    'image_id': img_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score)
                })

    with open('results.json', 'w') as f:
        json.dump(coco_results, f, indent=4)

    coco_dt = coco_gt.loadRes('results.json')
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


evaluate(model, test_loader)

def visualize_and_save_predictions(model, dataset, save_dir='output_imgs'):
    model.eval()
    indices = [10, 20, 30]

    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        image_tensor = image.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)[0]

        image_np = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        image_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(image_pil)

        for box, score in zip(output['boxes'], output['scores']):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1), f"{score:.2f}", fill='red')

        save_path = os.path.join(save_dir, f"prediction1_{i+1}.jpg")
        image_pil.save(save_path)
        print(f"Saved: {save_path}")

visualize_and_save_predictions(model, test_dataset)
