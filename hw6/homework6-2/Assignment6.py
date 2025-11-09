import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


class GAPNet(nn.Module):
    def __init__(self):
        super(GAPNet, self).__init__()

        # first convolution layer: input 3 (RGB), output 6 feature maps, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)

        # max pooling layer: downsample the feature map by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # second convolution layer: input 6, output 10 feature maps
        self.conv2 = nn.Conv2d(6, 10, kernel_size=5, stride=1)

        # global average pooling layer: reduce each 10x10 feature map to a single value
        self.gap = nn.AvgPool2d(kernel_size=10, stride=10)

        # fully connected layer: 10 features → 10 output classes
        self.fc = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        # conv1 → ReLU activation → max pooling
        x = self.pool(F.relu(self.conv1(x)))

        # conv2 → ReLU activation
        x = F.relu(self.conv2(x))

        # global average pooling
        x = self.gap(x)

        # flatten (N,10,1,1) → (N,10)
        x = torch.flatten(x, 1)

        # fully connected layer to output class scores
        x = self.fc(x)

        return x


def train_GAPNet():
    """Train GAPNet on CIFAR-10 for 10 epochs with SGD optimizer"""
    # Reuse CIFAR10_dataset_a() for dataset
    images, labels = CIFAR10_dataset_a()  # just to show function usage

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GAPNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/2000:.3f}")
                running_loss = 0.0

    print("Finished Training")
    PATH = "./Gap_net_10epoch.pth"
    torch.save(net.state_dict(), PATH)
    print(f"Saved model to {PATH}")


def eval_GAPNet():
    """Evaluate GAPNet on CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GAPNet().to(device)
    net.load_state_dict(torch.load("./Gap_net_10epoch.pth", map_location=device))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on 10000 test images: {100 * correct / total:.2f}%")

def backbone():
    """Q3: Load pretrained ResNet18 and extract features from cat_eye.jpg"""
    from torchvision import models
    from PIL import Image

    # Load pretrained ResNet18 backbone
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.eval()

    # Remove the final fully connected layer (classifier)
    backbone = nn.Sequential(*list(resnet18.children())[:-1])

    # Load input image
    img_path = "cat_eye.jpg"
    img = Image.open(img_path).convert("RGB")

    # Define preprocessing transforms (same as ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transform and add batch dimension
    input_tensor = transform(img).unsqueeze(0)

    # Forward pass through the backbone to get feature maps
    with torch.no_grad():
        features = backbone(input_tensor)

    print("Feature shape:", features.shape)
    return features

def transfer_learning():
    """Q4: Fine-tune only the last layer of pretrained ResNet18 on CIFAR-10"""
    from torchvision import models

    # 1. Load pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Freeze all layers (do not update gradients)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Modify the last fully connected layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # 4. Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 6. Prepare CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    # 7. Train for 10 epochs
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/2000:.3f}")
                running_loss = 0.0

    print("Finished fine-tuning last layer.")

    # 8. Save model weights
    PATH = "./Res_net_10epoch.pth"
    torch.save(model.state_dict(), PATH)
    print(f"Saved model to {PATH}")

    # 9. Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on 10000 test images: {100 * correct / total:.2f}%")


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block used in MobileNetV1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    """Q5: Implement MobileNetV1 architecture"""
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            # initial conv layer
            nn.Conv2d(ch_in, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # depthwise separable blocks
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),

            # 5 repeated blocks with 512 channels
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),

            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),

            nn.AvgPool2d(kernel_size=7)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)  # flatten
        x = self.fc(x)
        return x

    
if __name__ == '__main__':
    #Q1
    # from torchvision import models
    # resnet34 = models.resnet34(pretrained=True)
    # num_para = compute_num_parameters(resnet34)
    # print(num_para)

    # Q2
    print("\n=== Training GAPNet (Q2) ===")
    train_GAPNet()

    print("\n=== Evaluating GAPNet (Q2) ===")
    eval_GAPNet()

    # Q3
    # print("\n=== Extracting features using ResNet18 backbone (Q3) ===")
    # features = backbone()
    # print("Returned feature tensor shape:", features.shape)

    # Q4
    # print("\n=== Training ResNet18 with transfer learning (Q4) ===")
    # transfer_learning()

    # Q5
    # print("\n=== Testing MobileNetV1 structure (Q5) ===")
    # ch_in = 3
    # n_classes = 1000
    # model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)

    # # create a dummy input image (batch_size=1, 3x224x224)
    # x = torch.randn(1, 3, 224, 224)

    # # forward pass
    # out = model(x)
    # print("Output shape:", out.shape)
