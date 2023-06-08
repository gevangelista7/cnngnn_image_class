import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision import transforms, datasets
import os

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained VGG16 model
model = vgg16(pretrained=True)

# Freeze all the layers in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for the new classification task
num_classes = 10  # Number of classes in your new dataset
model.classifier[6] = nn.Linear(4096, num_classes)

# Move the model to the device
model = model.to(device)

# Load your dataset
img_treat = 'autocontrast1'
dataset_dir = '../datasets' if os.name == 'nt' else '/home/gabriel/thesis/dataset'

original_dir = dataset_dir + '/UATD_classification/samples_' + img_treat

train_dataset = datasets.ImageFolder("path_to_train_data", transform=transform)
test_dataset = datasets.ImageFolder("path_to_test_data", transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
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

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Fine-tuning completed.")
