import torch
import torch.nn as nn
from torchvision import models
import runtime_parameters
from torchvision.models import vit_b_16
from torchvision import models
from torchvision.ops import RoIAlign

class ObjectDetectionCNN(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(ObjectDetectionCNN, self).__init__()
        self.name = "Custom CNN Model"
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels * 8, kernel_size=(3,3), padding="same")
        self.conv2 = nn.Conv2d(in_channels=input_channels * 8, out_channels=input_channels * 16, kernel_size=(3,3), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2)) # 224 X 224 -> 112 X 112
        self.conv3 = nn.Conv2d(in_channels=input_channels * 16, out_channels=input_channels * 32, kernel_size=(3,3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=input_channels * 32, out_channels=input_channels * 64, kernel_size=(3,3), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2)) # 112 X 112 -> 56 X 56
        self.conv5 = nn.Conv2d(in_channels=input_channels * 64, out_channels=input_channels * 128, kernel_size=(3,3), padding="same")
        self.conv6 = nn.Conv2d(in_channels=input_channels * 128, out_channels=input_channels * 256, kernel_size=(3,3), padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2)) # 56 X 56 -> 28 X 28
        self.conv7 = nn.Conv2d(in_channels=input_channels * 256, out_channels=input_channels * 512, kernel_size=(3,3), padding="same")
        self.conv8 = nn.Conv2d(in_channels=input_channels * 512, out_channels=input_channels * 1024, kernel_size=(3,3), padding="same")
        self.pool4 = nn.MaxPool2d(kernel_size = (2,2)) # 28 X 28 -> 14 X 14
        self.fc1 = nn.Linear(input_channels * 1024 * 14 * 14, 1024)
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
        if runtime_parameters.image_channels != 3:
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
        if runtime_parameters.image_channels != 3:
            image = self.input_proj(image)
        image = self.vgg19(image)
        category = self.softmax(image)
        bbox = self.relu(self.fc_bbox(image))

        return bbox, category

class ObjectDetectionViT(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ObjectDetectionViT, self).__init__()
        self.name = "ViT Model"
        if runtime_parameters.image_channels != 3:
            self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=(3,3), padding="same")
        self.model = vit_b_16(pretrained=True)
        # Replace the head for category probabilities
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        # Additional head for bounding box predictions
        self.bbox_head = nn.Linear(num_classes, 4)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if runtime_parameters.image_channels != 3:
            x = self.input_proj(x)
        # Extract features
        features = self.model(x)
        # Category probabilities
        category_probabilities = self.softmax(features)
        # Bounding box predictions
        bounding_boxes = self.relu(self.bbox_head(features))
        return bounding_boxes, category_probabilities

class FastRCNN(nn.Module):
    def __init__(self, input_channels, n_classes, roi_output_size=(7, 7)):
        super(FastRCNN, self).__init__()
        self.name = "Fast R-CNN Model"
        if runtime_parameters.image_channels != 3:
            self.input_proj = nn.Conv2d(input_channels, 3, kernel_size=(3,3), padding="same")
        self.backbone = models.resnet50(pretrained=True)  # Pretrained backbone for feature extraction
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.n_classes = n_classes
        self.roi_pool = RoIAlign(roi_output_size, spatial_scale=1.0, sampling_ratio=2)
        
        # Fully connected layers for classification and bounding box regression
        feature_dim = self._get_feature_dim(roi_output_size, self.backbone)
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_cls = nn.Linear(1024, n_classes)
        self.fc_bbox = nn.Linear(1024, 4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def _get_feature_dim(self, roi_output_size, backbone):
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_features = backbone(dummy_input)
        roi_output = self.roi_pool(dummy_features, [torch.tensor([[0, 0, 224, 224]], dtype = torch.float32)])
        return roi_output.numel()

    def forward(self, images, rois):
        if runtime_parameters.image_channels != 3:
            images = self.input_proj(images)
        # Feature extraction using the backbone
        features = self.backbone(images)
        
        # Apply RoI pooling
        pooled_features = self.roi_pool(features, rois)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification and bounding box regression
        x = self.relu(self.fc1(pooled_features))
        x = self.relu(self.fc2(x))
        cls_logits = self.softmax(self.fc_cls(x))
        bbox_regression = self.fc_bbox(x)

        return bbox_regression, cls_logits
