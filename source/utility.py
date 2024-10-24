import scipy.io
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
import numpy as np
import runtime_parameters 

def read_mat_file(file_name):
    mat = scipy.io.loadmat(file_name)
    return mat

def get_box_coordinates(mat_obj):
    return mat_obj["box_coord"]

def get_metadata(root_directory):
    traversal_result = []
    for root, dirs, files in os.walk(root_directory):
        if len(dirs) == 0:
            category = root.split('/')[-1]
            processed_result = list(map(lambda file : os.path.join(root, file), files))
            traversal_result.append(pd.DataFrame({
                "file_path" : processed_result,
                "file_index" : [file.split('_')[-1].split('.')[0] for file in files],
                runtime_parameters.label_column_name : [category for _ in range(len(processed_result))]
            }))
    if len(traversal_result):
        return pd.concat(traversal_result, axis = 0, ignore_index = True)
    else:
        return pd.DataFrame(columns = ["file_path", runtime_parameters.label_column_name])

def visualize(image_path, bbox, filename = "./debug.jpg"):
    image = mpimg.imread(image_path)
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    # Create a rectangle patch for the bounding box
    bbox = patches.Rectangle(
        (float(bbox[0]), float(bbox[2])), 
        float(bbox[3] - bbox[0]), 
        float(bbox[1] - bbox[2]), 
        linewidth=2, 
        edgecolor='r', 
        facecolor='none'
    )
    # Add the bounding box to the plot
    ax.add_patch(bbox)
    # Optionally set limits and labels
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.title('Image with Bounding Box')
    plt.axis('off')  # Hide axes
    # Show the plot
    # plt.show()
    plt.savefig(filename)

def contrast_stretch(image):
    if len(image.shape) == 3:
        # Split the image into its color channels
        channels = cv2.split(image)
        stretched_channels = []

        for channel in channels:
            min_val = np.min(channel)
            max_val = np.max(channel)
            stretched = (channel - min_val) * (255 / (max_val - min_val))
            stretched = stretched.astype(np.uint8)
            stretched_channels.append(stretched)

        # Merge the stretched channels back together
        stretched_image = cv2.merge(stretched_channels)
        return stretched_image
    else:
        # Find the minimum and maximum pixel values in the image
        min_val = np.min(image)
        max_val = np.max(image)

        # Apply contrast stretching formula
        stretched = (image - min_val) * (255 / (max_val - min_val))
        stretched = stretched.astype(np.uint8)  # Convert back to uint8

        return stretched

def preprocess(
        image,
        bbox,
        resize_shape = None,
        convert2gray = True,
        apply_smoothing = True,
        contrast_streching = True
    ):
    if resize_shape is not None:
        scale_x = resize_shape[0] / image.shape[0]
        scale_y = resize_shape[1] / image.shape[1]

        bbox[0] = bbox[0] * scale_x
        bbox[2] = bbox[2] * scale_y
        bbox[3] = bbox[3] * scale_x
        bbox[1] = bbox[1] * scale_y

        image = cv2.resize(image, resize_shape)
    if convert2gray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if apply_smoothing:
        # Apply Gaussian smoothing
        kernel_size = runtime_parameters.gaussian_kernel_size  # Choose an odd number for the kernel size
        sigma = runtime_parameters.gaussian_sigma  # Let OpenCV calculate sigma
        image = cv2.GaussianBlur(image, kernel_size, sigma)
    
    if contrast_streching:
        image = contrast_stretch(image)

    return image, bbox

def read_image(image_path, annotations_path, preprocess_image = True):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    annotations = read_mat_file(annotations_path)
    bbox = get_box_coordinates(annotations)[0]
    if preprocess_image:
        image, bbox = preprocess(
            image,
            bbox,
            resize_shape = runtime_parameters.resize_shape,
            convert2gray = runtime_parameters.convert2gray,
            apply_smoothing = runtime_parameters.apply_smoothing,
            contrast_streching = runtime_parameters.contrast_streching
        )
    image = image.reshape(-1, runtime_parameters.resize_shape[0], runtime_parameters.resize_shape[1])
    return image, bbox

def get_image_annot_mapping():
    annotations_data = get_metadata(runtime_parameters.annotations_data_path).rename(columns = {"file_path" : runtime_parameters.annotation_file_path_column})
    categories_data = get_metadata(runtime_parameters.images_data_path).rename(columns = {"file_path" : runtime_parameters.images_file_path_column})
    data = categories_data.merge(annotations_data, how = "left", on = ["file_index", "label"]).dropna()
    return data

def train_test_split(image_annotation_mapping, stratify = None):
    train_data = []
    validation_data = []
    test_data = []
    if stratify is not None:
        for category in image_annotation_mapping[stratify].unique().tolist():
            category_data = image_annotation_mapping[image_annotation_mapping[stratify] == category].reset_index(drop = True)
            index = category_data.index.tolist()
            train_index = np.random.choice(
                index,
                size = int(len(index) * runtime_parameters.train_size),
                replace = False
            )
            remaining_index = [idx for idx in index if idx not in train_index]
            validation_index = np.random.choice(
                remaining_index,
                size = int(len(remaining_index) * runtime_parameters.val_size),
                replace = False
            )
            test_index = np.array([idx for idx in remaining_index if idx not in validation_index])
            train_data.append(category_data.iloc[train_index])
            validation_data.append(category_data.iloc[validation_index])
            test_data.append(category_data.iloc[test_index])
    else:
        index = image_annotation_mapping.index.tolist()
        train_index = np.random.choice(
            index,
            size = int(len(index) * runtime_parameters.train_size),
            replace = False
        )
        remaining_index = [idx for idx in index if idx not in train_index]
        validation_index = np.random.choice(
            remaining_index,
            size = int(len(remaining_index) * runtime_parameters.val_size),
            replace = False
        )
        test_index = np.array([idx for idx in remaining_index if idx not in validation_index])
        
        train_data.append(image_annotation_mapping.iloc[train_index])
        validation_data.append(image_annotation_mapping.iloc[validation_index])
        test_data.append(image_annotation_mapping.iloc[test_index])
    
    train_data = pd.concat(train_data, axis = 0, ignore_index=True)
    validation_data = pd.concat(validation_data, axis = 0, ignore_index=True)
    test_data = pd.concat(test_data, axis = 0, ignore_index=True)

    return train_data, validation_data, test_data
        
