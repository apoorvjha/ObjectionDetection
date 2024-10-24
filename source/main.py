from utility import *
from dataset import *
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

if __name__ == "__main__":
    image_annotation_mapping = get_image_annot_mapping()
    train_data, validation_data, test_data = train_test_split(image_annotation_mapping, stratify = runtime_parameters.label_column_name)
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    onehot_encoder.fit(np.array(train_data[runtime_parameters.label_column_name].tolist()).reshape(-1,1))
    train_dataset = Dataset(
        train_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataset = Dataset(
        validation_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_dataset = Dataset(
        test_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)