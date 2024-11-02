from utility import *
from dataset import *
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from model_trainer import *
from model import *
from model_evaluator import *

if __name__ == "__main__":
    set_seed(42)
    image_annotation_mapping = get_image_annot_mapping()
    train_data, validation_data, test_data = train_test_split(image_annotation_mapping, stratify = runtime_parameters.label_column_name)
    print("Data Size : ", image_annotation_mapping.shape)
    print("Train Data Size : ", train_data.shape)
    print("Validation Data Size : ", validation_data.shape)
    print("Test Data Size : ", test_data.shape)
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    onehot_encoder.fit(np.array(train_data[runtime_parameters.label_column_name].tolist()).reshape(-1,1))
    train_dataset = Dataset(
        train_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    train_dataloader = DataLoader(train_dataset, batch_size=runtime_parameters.batch_size, shuffle=True)
    validation_dataset = Dataset(
        validation_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=runtime_parameters.batch_size, shuffle=False)
    test_dataset = Dataset(
        test_data, 
        runtime_parameters.images_file_path_column,
        runtime_parameters.annotation_file_path_column,
        runtime_parameters.label_column_name, 
        onehot_encoder
    )
    test_dataloader = DataLoader(test_dataset, batch_size=runtime_parameters.batch_size, shuffle=False)
    print("Catgories : ", len(onehot_encoder.categories_[0]))
    
    object_detection_model = ObjectDetectionCNN(runtime_parameters.image_channels, len(onehot_encoder.categories_[0]))
    object_detection_model, history = train(object_detection_model, train_dataloader, validation_dataloader, len(onehot_encoder.categories_[0]))
    plot_loss(history, "CNN_Loss_Plt.jpg")
    evaluate(object_detection_model,test_dataloader,len(onehot_encoder.categories_[0]))

    object_detection_model = ObjectDetectionVGG(runtime_parameters.image_channels, len(onehot_encoder.categories_[0]))
    object_detection_model, history = train(object_detection_model, train_dataloader, validation_dataloader, len(onehot_encoder.categories_[0]))
    plot_loss(history, "VGG_Loss_Plt.jpg")
    evaluate(object_detection_model,test_dataloader,len(onehot_encoder.categories_[0]))

    object_detection_model = ObjectDetectionViT(runtime_parameters.image_channels, len(onehot_encoder.categories_[0]))
    object_detection_model, history = train(object_detection_model, train_dataloader, validation_dataloader, len(onehot_encoder.categories_[0]))
    plot_loss(history, "ViT_Loss_Plt.jpg")
    evaluate(object_detection_model,test_dataloader,len(onehot_encoder.categories_[0]))