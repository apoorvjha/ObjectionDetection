import torch
import runtime_parameters
from loss_function import *

def train(model, train_data, validation_data, num_classes):
    optimizer = torch.optim.Adam(model.parameters(), lr = runtime_parameters.learning_rate)
    loss_fn = ObjectDetectionLoss(num_classes)
    loss_history = {"train_loss" : [], "val_loss" : []}
    model = model.to(device = runtime_parameters.device)
    for epoch in range(runtime_parameters.epochs):
        running_loss_train = 0.0
        count = 0
        for idx, data in enumerate(train_data):
            image = data["image"].to(device = runtime_parameters.device)
            bbox = data["bbox"].to(device = runtime_parameters.device)
            label = data["label"].to(device = runtime_parameters.device)

            optimizer.zero_grad()

            bbox_prediction, label_prediction = model(image)

            loss = loss_fn(bbox_prediction, bbox, label_prediction, label)
            running_loss_train += loss.item()
            
            if idx%100 == 0:
                print(f"\t############# [TRAIN {idx + 1}] : Loss = {loss.item()}")

            loss.backward()
            optimizer.step()
            count += 1

        loss_history["train_loss"].append(running_loss_train / count)

        running_loss_val = 0.0
        count = 0
        for idx, data in enumerate(validation_data):
            image = data["image"].to(device = runtime_parameters.device)
            bbox = data["bbox"].to(device = runtime_parameters.device)
            label = data["label"].to(device = runtime_parameters.device)

            with torch.no_grad():
                bbox_prediction, label_prediction = model(image)

                loss = loss_fn(bbox_prediction, bbox, label_prediction, label)
                running_loss_val += loss.item()
                if idx%100 == 0:
                    print(f"\t############# [Validation {idx + 1}] : Loss = {loss.item()}")
            count += 1

        loss_history["val_loss"].append(running_loss_val / count)

        print(f"[EPOCH = {epoch + 1}] => Train Loss : {loss_history['train_loss'][-1]} | Validation Loss : {loss_history['val_loss'][-1]}")
    return model, loss_history
