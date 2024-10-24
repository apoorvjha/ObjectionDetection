import torch

resize_shape = (128,128)
convert2gray = True
apply_smoothing = True
contrast_streching = True
gaussian_kernel_size = (5, 5)
gaussian_sigma = 0
annotations_data_path = "../caltech-101/caltech-101/Annotations/"
annotation_file_path_column = "file_path_annotations"
images_data_path = "../caltech-101/caltech-101/101_ObjectCategories/"
images_file_path_column = "file_path_categories"
label_column_name = "label"
train_size = 0.8
val_size = 0.1
test_size = 0.1
batch_size = 64
device = ['cuda' if torch.cuda.is_available() else 'cpu'][0]
learning_rate = 1e-3
epochs=120
patience = 10
t_max = 30
eta_min = 1e-1