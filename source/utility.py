import scipy.io
import os
import pandas as pd

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
                "label" : [category for _ in range(len(processed_result))]
            }))
    if len(traversal_result):
        return pd.concat(traversal_result, axis = 0, ignore_index = True)
    else:
        return pd.DataFrame(columns = ["file_path", "label"])

def visualize(image, bbox):
    pass

if __name__ == '__main__':
    # print(get_box_coordinates(read_mat_file("./caltech-101/caltech-101/Annotations/anchor/annotation_0001.mat") ))
    annotations_data = get_metadata("./caltech-101/caltech-101/Annotations/")
    categories_data = get_metadata("./caltech-101/caltech-101/101_ObjectCategories/")

    annotation_labels = set(annotations_data["label"].unique().tolist())
    categories_labels = set(categories_data["label"].unique().tolist())
    print(
        "Check : ",
        "\n Annotation labels Not in Category Labels : ",
        annotation_labels.difference(categories_labels),
        "\n Category labels Not in Annotation Labels : ",
        categories_labels.difference(annotation_labels)
    )