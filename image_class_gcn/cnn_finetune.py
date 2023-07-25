import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, resnet18
from torchvision import transforms, datasets
from utils import FitAndPad,  classification_dir, datasets_dir, code_dir, results_dir, PILResize
from torch.utils.data import DataLoader
import os

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations

transform = transforms.Compose([
    FitAndPad(),
    transforms.ToTensor()
])

# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Load the pre-trained VGG16 model
# model = vgg16(pretrained=True)
model = resnet18(pretrained=True)

# Freeze all the layers in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for the new classification task
num_classes = 10  # Number of classes in your new dataset
if model._get_name() == 'VGG':
    model.classifier[6] = nn.Linear(4096, num_classes)
elif model._get_name() == 'ResNet':
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

# Move the model to the device
model = model.to(device)

# Load your dataset
img_treat_dir = '/samples_autocontrast1'
original_dir = classification_dir+img_treat_dir

train_dataset = datasets.ImageFolder(original_dir+'/Training', transform=transform)
test_dataset = datasets.ImageFolder(original_dir+'/Test_1', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 100
max_val_acc = - np.inf

if __name__ == '__main__':
    print('+++ Beginning Training +++')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if accuracy > max_val_acc:
            max_val_acc = accuracy
            print("Best model! Saving...")
            with open(results_dir + '/'+model._get_name()+'results.txt', 'w') as f:
                f.write('epoch: ' + str(epoch) + '; test acc: ' + str(max_val_acc))
            torch.save(model.state_dict(), code_dir+"/models/"+model._get_name()+"_best_model.pth")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Fine-tuning completed.")
