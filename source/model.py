import torch
import torch.nn as nn
from torchvision import models

class ObjectDetectionCNN(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(ObjectDetectionCNN, self).__init__()
        self.name = "Custom CNN Model"
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 8, kernel_size=(3,3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=input_channels * 8, out_channels=input_channels * 16, kernel_size=(3,3), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2))
        self.conv3 = nn.Conv2d(in_channels=input_channels * 16, out_channels=input_channels * 32, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=input_channels * 32, out_channels=input_channels * 64, kernel_size=(3,3), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2))
        self.conv5 = nn.Conv2d(in_channels=input_channels * 64, out_channels=input_channels * 128, kernel_size=(3,3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=input_channels * 128, out_channels=input_channels * 256, kernel_size=(3,3), padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2))
        self.conv7 = nn.Conv2d(in_channels=input_channels * 256, out_channels=input_channels * 512, kernel_size=(3,3), padding="same")
        self.conv8 = nn.Conv2d(in_channels=input_channels * 512, out_channels=input_channels * 1024, kernel_size=(3,3), padding="same")
        self.pool4 = nn.MaxPool2d(kernel_size = (2,2))
        self.fc1 = nn.Linear(1024 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_cls = nn.Linear(256, n_classes)
        self.fc3 = nn.Linear(256, 64)
        self.fc_bbox = nn.Linear(64, 4)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
    def forward(self, image):
        image = nn.Dropout(p=0.2)(self.relu(self.conv1(image)))
        image = nn.Dropout(p=0.2)(self.relu(self.conv2(image)))
        image = self.pool1(image)

        image = nn.Dropout(p=0.2)(self.relu(self.conv3(image)))
        image = nn.Dropout(p=0.2)(self.relu(self.conv4(image)))
        image = self.pool2(image)

        image = nn.Dropout(p=0.2)(self.relu(self.conv5(image)))
        image = nn.Dropout(p=0.2)(self.relu(self.conv6(image)))
        image = self.pool3(image)

        image = nn.Dropout(p=0.2)(self.relu(self.conv7(image)))
        image = nn.Dropout(p=0.2)(self.relu(self.conv8(image)))
        image = self.pool4(image)
        
        image = self.flatten(image)

        image = nn.Dropout(p=0.2)(self.relu(self.fc1(image)))
        image = nn.Dropout(p=0.2)(self.relu(self.fc2(image)))
        category = self.softmax(self.fc_cls(image))

        image = nn.Dropout(p=0.2)(self.relu(self.fc3(image)))
        bbox = self.relu(self.fc_bbox(image))

        return bbox, category
    
class ObjectDetectionVGG(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(ObjectDetectionVGG, self).__init__()
        self.name = "VGG-19 Model"
        self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=(3,3), padding="same")
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.classifier[6] = nn.Linear(self.vgg19.classifier[6].in_features, n_classes)
        
        for param in self.vgg19.features.parameters():
            param.requires_grad = False

        for param in self.vgg19.classifier.parameters():
            param.requires_grad = True

        self.fc_bbox = nn.Linear(n_classes, 4)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        image = self.input_proj(image)
        image = self.vgg19(image)
        category = self.softmax(image)
        bbox = self.relu(self.fc_bbox(image))

        return bbox, category

class ObjectDetectionViT(nn.Module):
    def __init__(self, num_classes, img_size=128, patch_size=16, embed_dim=256, num_heads=8, num_layers=6, ff_dim=512):
        super(ObjectDetectionViT, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim) for _ in range(num_layers)
        ])

        # Output head
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc_bbox = nn.Linear(embed_dim, 4)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)  # Shape: (batch_size, embed_dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        x += self.position_embedding  # Add positional encoding

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Global average pooling and final classification
        x = self.global_avg_pool(x.transpose(1, 2))  # Shape: (batch_size, embed_dim, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, embed_dim)
        category = self.softmax(self.fc(x))  # Shape: (batch_size, num_classes)
        bbox = self.relu(self.fc_bbox(x))
        return bbox, category


