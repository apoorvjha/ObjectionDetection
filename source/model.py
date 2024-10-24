import torch
import torch.nn as nn

class ObjectDetectionCNN(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(ObjectDetectionCNN, self).__init__()
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
        image = self.relu(self.conv1(image))
        image = self.relu(self.conv2(image))
        image = self.pool1(image)

        image = self.relu(self.conv3(image))
        image = self.relu(self.conv4(image))
        image = self.pool2(image)

        image = self.relu(self.conv5(image))
        image = self.relu(self.conv6(image))
        image = self.pool3(image)

        image = self.relu(self.conv7(image))
        image = self.relu(self.conv8(image))
        image = self.pool4(image)
        
        image = self.flatten(image)

        image = self.relu(self.fc1(image))
        image = self.relu(self.fc2(image))
        cls = self.softmax(self.fc_cls(image))

        image = self.relu(self.fc3(image))
        bbox = self.relu(self.fc_bbox(image))

        return bbox, cls
