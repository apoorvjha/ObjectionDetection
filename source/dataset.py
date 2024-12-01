from utility import *
import torch

class Dataset:
    def __init__(self, data, image_path_column, annotation_path_column, label_column, onehot_encoder):
        self.data = data
        self.image_path_column = image_path_column
        self.annotation_path_column = annotation_path_column
        self.label_column = label_column
        self.onehot_encoder = onehot_encoder
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image_path = self.data[self.image_path_column].iloc[idx]
        annotation_path = self.data[self.annotation_path_column].iloc[idx]
        image, bbox = read_image(image_path, annotation_path, preprocess_image = True)
        image = image / 255
        label = self.data[self.label_column].iloc[idx]
        return {
            "image" : torch.tensor(image, dtype = torch.float32),
            "bbox" : torch.tensor(bbox, dtype = torch.float32),
            "label" : torch.tensor(self.onehot_encoder.transform(np.array([label]).reshape(-1,1)).toarray(), dtype = torch.float32).view(-1)
        }