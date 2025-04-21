import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


class LicensePlateDataset(Dataset):
    def __init__(self, data, label_map=None, transform=None):
        self.data = data
        self.transform = transform

        if label_map is None:
            chars = sorted({item['label'] for item in data})
            self.label_map = {char: idx for idx, char in enumerate(chars)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]['image']  # [1, 40, 20]
        label_str = self.data[idx]['label']
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LeNetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)        # [1, 40, 20] -> [6, 36, 16]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # [6, 36, 16] -> [6, 18, 8]
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)       # [6, 18, 8] -> [16, 14, 4]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # [16, 14, 4] -> [16, 7, 2]
        self.fc1 = nn.Linear(16 * 7 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):           # [1, 40, 20]
        x = F.relu(self.conv1(x))   # -> [6, 36, 16]
        x = self.pool1(x)           # -> [6, 18, 8]
        x = F.relu(self.conv2(x))   # -> [16, 14, 4]
        x = self.pool2(x)           # -> [16, 7, 2]
        x = x.view(-1, 16 * 7 * 2)  # Flatten
        x = F.relu(self.fc1(x))     # -> 120
        x = F.relu(self.fc2(x))     # -> 84
        x = self.fc3(x)             # -> num_classes (34)
        return x

def plot_train_loss(train_losses, save_path="train_loss_curve.png"):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy(train_accuracies, val_accuracies, save_path="accuracy_curve.png"):
    epochs = np.arange(1, len(train_accuracies) + 1)
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    model = model.to(device)
    best_val_acc = 0.0

    train_losses, train_accuracies, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved (Val Acc = {best_val_acc:.4f})")

    plot_train_loss(train_losses)
    plot_accuracy(train_accuracies, val_accuracies)

def compute_metrics(y_true, y_pred, num_classes):
    report = classification_report(y_true, y_pred, output_dict=True)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(num_classes))
    mAP = average_precision_score(y_true_bin, y_pred_bin, average='macro')

    return report, mAP

def plot_class_precision(report, label_map, save_path="class_precision.png"):
    class_names = list(label_map.keys())
    precisions = [report[str(label)]['precision'] for label in range(len(class_names))]

    plt.figure(figsize=(12, 6))
    plt.bar(class_names, precisions, color='skyblue')
    plt.ylabel("Precision")
    plt.xlabel("Char")
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def final_evaluate_and_plot(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    report, mAP = compute_metrics(all_labels, all_preds, len(data_loader.dataset.label_map))
    plot_class_precision(report, data_loader.dataset.label_map, save_path="class_precision.png")

    print(f"Test Accuracy: {correct/total:.4f}")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")


if __name__ == "__main__":
    # Load the data
    data = torch.load('data.pt')
    train_data = data['train'] # 1440
    test_data = data['test'] # 360

    # Split into training and validation sets
    train_split, val_split = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True) # 1152, 288

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    train_dataset = LicensePlateDataset(train_split, transform=train_transform) # Number of classes: 34
    val_dataset = LicensePlateDataset(val_split, label_map=train_dataset.label_map, transform=val_transform)
    test_dataset = LicensePlateDataset(test_data, label_map=train_dataset.label_map, transform=val_transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the model
    num_classes = len(train_dataset.label_map)
    model = LeNetClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    final_evaluate_and_plot(model, test_loader, device)
    