import torch
import runtime_parameters
from loss_function import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, test_data, num_classes):
    loss_fn = ObjectDetectionLoss(num_classes, runtime_parameters.lambda_box, runtime_parameters.lambda_cls)
    model_name = model.name
    model = model.to(device = runtime_parameters.device)
    running_loss_test = []
    actual_bboxes = []
    actual_labels = []
    predicted_bboxes = []
    predicted_labels = []
    count = 0
    
    for idx, data in enumerate(test_data):
        image = data["image"].to(device = runtime_parameters.device)
        bbox = data["bbox"].to(device = runtime_parameters.device)
        label = data["label"].to(device = runtime_parameters.device)

        with torch.no_grad():
            bbox_prediction, label_prediction = model(image)

            actual_label = torch.argmax(label, dim=1).cpu().numpy()
            predicted_label = torch.argmax(label_prediction, dim = 1).cpu().numpy()

            actual_bbox = bbox.cpu().numpy()
            predicted_bbox = bbox_prediction.cpu().numpy()

            actual_labels.extend(actual_label.tolist())
            predicted_labels.extend(predicted_label.tolist())

            actual_bboxes.extend(actual_bbox.tolist())
            predicted_bboxes.extend(predicted_bbox.tolist())

            loss = loss_fn(bbox_prediction, bbox, label_prediction, label)
            running_loss_test.append(loss.item())
            if idx%10 == 0:
                print(f"\t############# [Test {idx + 1}] : Loss = {loss.item()}")
        count += 1
    # Evaluation Metrics ...
    average_loss = sum(running_loss_test) / count
    # Calculate precision, recall, and F1 score for each class
    precision_per_class = precision_score(actual_labels, predicted_labels, average=None)
    recall_per_class = recall_score(actual_labels, predicted_labels, average=None)
    f1_per_class = f1_score(actual_labels, predicted_labels, average=None)
    
    # Calculate micro and macro averages
    precision_macro = precision_score(actual_labels, predicted_labels, average='macro')
    recall_macro = recall_score(actual_labels, predicted_labels, average='macro')
    f1_macro = f1_score(actual_labels, predicted_labels, average='macro')
    
    precision_micro = precision_score(actual_labels, predicted_labels, average='micro')
    recall_micro = recall_score(actual_labels, predicted_labels, average='micro')
    f1_micro = f1_score(actual_labels, predicted_labels, average='micro')

    print(f"\n\n========================== EVALUATION OF {model_name} ===========================\n")
    print("Average Loss : ", average_loss)
    print("Precision Per Class : ", precision_per_class)
    print("Recall Per Class : ", recall_per_class)
    print("F1 Score Per Class : ", f1_per_class)

    print("Precision Macro : ", precision_macro)
    print("Recall Macro : ", recall_macro)
    print("F1 Score Macro : ", f1_macro)

    print("Precision Micro : ", precision_micro)
    print("Recall Micro : ", recall_micro)
    print("F1 Score Micro : ", f1_micro)
    print("\n=======================================================================================\n\n") 

