import gc
import json
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DroneDataset import add_padding_fn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class TransferLearningModel(nn.Module):
    """
    A fully convolutional detection model that uses ResNet18 CNN layers for transfer learning
    detection_layer format [layer_0, layer_1, layer_2, ..., output_layer]
    layer_x = [in_channel, out_channel, kernel_size, stride, padding, padding_mode]
    """

    def __init__(self, detection_layers, activation_func, dropout_rate):

        # take resnet18 with learned weights
        resnet = resnet18(weights="DEFAULT")

        super().__init__()

        self.detection_layers = detection_layers
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate

        # transfer learning (ResNet18 CNN layers)
        self.resnetLayers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

            

        detection_head = []

        # add dropout layer before detection head 
        # if dropout rate is higher than 0 
        if dropout_rate != 0.0:
            dropout_layer = nn.Dropout2d(p=dropout_rate)
            detection_head.append(dropout_layer)

        # Detection head
        for layer in detection_layers:
                
            # hidden layer
            conv_layer = nn.Sequential(
                nn.Conv2d(layer[0], layer[1], layer[2], stride=layer[3], padding=layer[4], padding_mode=layer[5]),
                nn.BatchNorm2d(layer[1]),
                activation_func()
            )
            
            detection_head.append(conv_layer)


        # output layer
        output_layer_in_channel = detection_layers[-1][1]

        output_layer = nn.Conv2d(output_layer_in_channel, 5, 1, 1, 0, padding_mode="zeros")

        detection_head.append(output_layer)

        self.detection_network = nn.Sequential(*detection_head)

    
    def forward(self,x):

        out = self.resnetLayers(x)

        out = self.detection_network(out)

        # class probability, x and y is between 0-1 is between 0-1 
        objectness_and_bbox_center_x_y = torch.sigmoid(out[:, 0:3, :, :])
        
        # width and height > 0
        bbox_center_sizes = F.softplus(out[:, 3:5, :, :])

        out = torch.cat([objectness_and_bbox_center_x_y, bbox_center_sizes], dim=1)

        return out


    def freeze_resnet_layers(self):
        for param in self.resnetLayers.parameters():
            param.requires_grad = False

        self.resnetLayers.eval()


    def unfreeze_last_2_resnet_layers(self):
        last_2_layer = [self.resnetLayers[-2], self.resnetLayers[-1]]

        for layer in last_2_layer:
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()


    def unfreeze_last_resnet_layer(self):
        last_layer = self.resnetLayers[-1]

        for param in last_layer.parameters():
            param.requires_grad = True

        last_layer.train()



class DroneLoss(nn.Module):
    """
        Loss function for Drone detection
    """
    def __init__(self, positive_class_bce_coefficient, localization_loss_coefficent, bbox_loss_func):
        super().__init__()

        self.class_loss_func = nn.BCELoss(reduction="none")

        self.bbox_loss_func = bbox_loss_func(reduction="none")
        self.localization_loss_coefficent = localization_loss_coefficent
        self.positive_class_bce_coefficient = positive_class_bce_coefficient

    def forward(self, predictions, labels):

        pred_cls = predictions[:, 0, :, :]
        label_cls = labels[:, 0, :, :]


        pred_bbox = predictions[:, 1:5, :, :]
        label_bbox = labels[:, 1:5, :, :]

        valid_mask = labels[:,5, :, :]

        # class loss
        cls_loss = self.class_loss_func(pred_cls, label_cls)
        # apply weightining
        class_weights = torch.where(
            label_cls == 1,
            torch.tensor(
                self.positive_class_bce_coefficient,
                device=label_cls.device
            ),
            torch.tensor(
                (1.0 - self.positive_class_bce_coefficient),
                device=label_cls.device
            )
            )
        cls_loss = cls_loss * class_weights
        # ignore paddings
        cls_loss = cls_loss * valid_mask
        # divide the total class loss by total number of valid cells (if no valid cell divide by 1 instead)
        cls_loss = cls_loss.sum() / (valid_mask.sum().clamp(min=1))

        # localization loss
        # calculate localization loss only if there is an object
        object_mask = (label_cls == 1)
        object_mask = object_mask.unsqueeze(1).expand_as(pred_bbox)

        localization_loss = self.bbox_loss_func(pred_bbox, label_bbox)

        if object_mask.any():
            localization_loss = (localization_loss[object_mask].mean())
        else:
            localization_loss = predictions.new_tensor(0.0)

        total_loss = (cls_loss + (self.localization_loss_coefficent * localization_loss))

        return (total_loss, cls_loss, localization_loss)



def train(model, fine_tune_layers_num, dataset_loader, criterion, optimizer, max_num_epochs, early_stopping, patience):
    """
        fine_tune_layers_num decides which resnet layers to fine tune
        0 --> none
        1 --> last resnet layer
        2 --> last 2 resnet layer
    """

    if(max_num_epochs == 0):
        return None, None, None, None, None, None

    if(torch.cuda.is_available() == True):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.train()
    model.freeze_resnet_layers()
    if fine_tune_layers_num == 0:
        pass
    elif fine_tune_layers_num == 1:
        model.unfreeze_last_resnet_layer()
    elif fine_tune_layers_num == 2:
        model.unfreeze_last_2_resnet_layers()

    model = model.to(device)

    best_loss = float("inf")
    patience_counter = 0
    stop_flag = False

    total_loss_list = []
    class_loss_list = []
    localization_loss_list = []

    for epoch in range(max_num_epochs):
        # if early_stopping == False then never stop until reach max epoch
        if stop_flag:
            break

        
        total_train_loss = 0.0
        total_classification_loss = 0.0
        total_localization_loss = 0.0
        total_samples = 0

        for inputs, labels, _ in dataset_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss, classification_loss, localization_loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # multiply with batch size
            total_train_loss += loss.item() * inputs.size(0)
            total_classification_loss += classification_loss.item() * inputs.size(0)
            total_localization_loss += localization_loss.item() * inputs.size(0)

            total_samples += inputs.size(0)

        total_train_loss = total_train_loss / total_samples
        total_classification_loss = total_classification_loss / total_samples
        total_localization_loss = total_localization_loss / total_samples

        total_loss_list.append(total_train_loss)
        class_loss_list.append(total_classification_loss)
        localization_loss_list.append(total_localization_loss)

        if(best_loss > total_train_loss):
            best_loss = total_train_loss
            patience_counter = 0
        elif early_stopping:
            patience_counter += 1
            if patience_counter >= patience:
                stop_flag = True
                break

    loss_list = [total_train_loss, total_classification_loss, total_localization_loss]

    train_bbox_preds, train_bbox_labels = predict(model, dataset_loader)
    
    return train_bbox_preds, train_bbox_labels, loss_list, total_loss_list, class_loss_list, localization_loss_list


def calculate_IoU(bbox1, bbox2):
    """
        bbox = [center_x, center_y, width, height]
    """

    intersection_x_min = max((bbox1[0] - bbox1[2]/2), (bbox2[0] - bbox2[2]/2))
    intersection_x_max = min((bbox1[0] + bbox1[2]/2), (bbox2[0] + bbox2[2]/2))
    intersection_y_min = max((bbox1[1] - bbox1[3]/2), (bbox2[1] - bbox2[3]/2))
    intersection_y_max = min((bbox1[1] + bbox1[3]/2), (bbox2[1] + bbox2[3]/2))

    intersection_x = max(0, (intersection_x_max - intersection_x_min))
    intersection_y = max(0, (intersection_y_max - intersection_y_min))

    intersection = intersection_x * intersection_y

    union = ((bbox1[2]*bbox1[3]) + (bbox2[2]*bbox2[3])) - intersection

    if union == 0:
        return 0.0
    
    return intersection / union


def predict(model, dataset_loader):
    """
        Only predict, model is not trained
        returns predictions [[{"bbox": [x,y,w,h], "confidence": %ab.c}] , ...] and bbox_predictions_results [[image0_bboxes], ...]
    """

    if(torch.cuda.is_available() == True):
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    pred_bboxes = []
    label_bboxes = []

    # for finding bbox location

    with torch.no_grad():
        for inputs, labels, _ in dataset_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            # inference
            predictions = model(inputs)

            objectness = predictions[:, 0]

            x = predictions[:, 1]
            y = predictions[:, 2]

            width = predictions[:, 3]
            height = predictions[:, 4]

            if labels.shape[1] == 6:
                # true if it is not padding
                is_valid = labels[:, 5]
            else:
                is_valid = torch.ones_like(labels[:, 0])
        
            cell_size = 32
            batch_size = objectness.shape[0]

            # iterate batches
            for batch in range(batch_size):
                
                # ---------- prediction bboxes ----------
                rows , cols = objectness[batch].shape

                image_pred_bboxes = []

                for row in range(rows):
                    for col in range(cols):
            
                        if is_valid[batch, row, col].item() != 1.0:
                            continue
                        
                        confidence = objectness[batch, row, col].item()
                
                        center_x = (col + x[batch, row, col].item()) * cell_size
                        center_y = (row + y[batch, row, col].item()) * cell_size
                
                        bbox_width = width[batch, row, col].item() * cell_size
                        bbox_height = height[batch, row, col].item() * cell_size

                        image_pred_bboxes.append({
                            "bbox": [center_x, center_y, bbox_width, bbox_height],
                            "confidence": confidence
                        })
                        # ground truth bboxes

                pred_bboxes.append(image_pred_bboxes)

                # --------------- ground truth bboxes ---------------
                    
                label_object_cells = torch.where(labels[batch, 0] == 1)
                label_rows, label_cols = label_object_cells[0], label_object_cells[1]

                image_label_bboxes = []
                for row, col in zip(label_rows, label_cols):
                        row, col = row.item(), col.item()

                        if is_valid[batch, row, col].item() != 1.0:
                            continue

                        label_x = (col + labels[batch, 1, row, col].item()) * cell_size
                        label_y = (row + labels[batch, 2, row, col].item()) * cell_size
                
                        label_width = labels[batch, 3, row, col].item() * cell_size
                        label_height = labels[batch, 4, row, col].item() * cell_size

                        image_label_bboxes.append({"bbox": [label_x, label_y, label_width, label_height]})

                label_bboxes.append(image_label_bboxes)

    return pred_bboxes, label_bboxes


# for drawing bboxes, no ground truth
def predict_locations(model, image_tensor, objectness_threshold):
    """
        Only predict, model is not trained
        returns bboxes' localtions
    """

    if(torch.cuda.is_available() == True):
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        objectness = output[:, 0]

        x = output[:, 1]
        y = output[:, 2]

        width = output[:, 3]
        height = output[:, 4]

    # remove batch dimension
    objectness = objectness[0]
    x = x[0]
    y = y[0]
    width = width[0]
    height = height[0]

    object_cells = torch.where(objectness >= objectness_threshold)
    # object_cells ---> (tensor([row1, row2, ...]), tensor([col1, col2, ...]))

    rows = object_cells[0]
    cols = object_cells[1]

    predictions = []
    cell_size = 32

    for row, col in zip(rows, cols):

        row = row.item()
        col = col.item()

        confidence = objectness[row, col].item()

        center_x = (col + x[row, col].item()) * cell_size
        center_y = (row + y[row, col].item()) * cell_size

        bbox_width = width[row, col].item() * cell_size
        bbox_height = height[row, col].item() * cell_size

        # center to upper left corner

        corner_x = center_x - (bbox_width / 2)
        corner_y = center_y - (bbox_height / 2)

        predictions.append({
            "bbox": [corner_x, corner_y, bbox_width, bbox_height],
            "confidence": confidence
        })

    return predictions


def calculate_detection_metrics(predictions_bboxes, label_bboxes, IoU_threshold, objectness_threshold):

    TP = 0
    FP = 0
    FN = 0

    # for every image
    for image_pred_bboxes, image_label_bboxes in zip(predictions_bboxes, label_bboxes):

        # sort the predictions so pred with hight confidence priotorized
        image_pred_bboxes.sort(key= lambda x: x["confidence"], reverse=True)

        # if more than one predictions has the same bbox as highest IoU 
        # than one of them is TP and others are FP
        matched_bbox_indexes = []

        # for every prediction in one image
        for image_pred_bbox in image_pred_bboxes:

            if image_pred_bbox["confidence"] <= objectness_threshold:
                continue
            
            image_pred_bbox_value = image_pred_bbox["bbox"]
            max_iou = 0
            max_label_index = None


            for index, image_label_bbox in enumerate(image_label_bboxes):

                # do not allow more than one predictions to 
                # match with same ground truth bbox
                if index in matched_bbox_indexes:
                    continue

                image_label_bbox_value = image_label_bbox["bbox"]
                iou = calculate_IoU(image_pred_bbox_value, image_label_bbox_value)
                
                if IoU_threshold <= iou:
                    if max_iou <= iou:
                        max_iou = iou
                        max_label_index = index
        
            if (max_label_index is not None):
                TP += 1
                matched_bbox_indexes.append(max_label_index) 
            else:
                FP += 1

        # If Ground truth is not found, than it is a False Negative
        FN += (len(image_label_bboxes) - len(matched_bbox_indexes))

    return TP, FP, FN


def calculate_metrics(TP, FP, FN):

    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return precision, recall, f1


def bbox_mAP_input_format(bbox):
    x = bbox[0] 
    y = bbox[1] 
    w = bbox[2] 
    h = bbox[3]

    x1 = x - (w / 2)
    y1 = y - (h / 2) 

    x2 = x + (w / 2)
    y2 = y + (h / 2)

    return [x1, y1, x2, y2] 


def convert_preds_to_mAP_input_format(pred_bboxes):
    """
        Convert predict() output to MeanAveragePrecision() method input format
    """
    preds = []

    for image_bboxes in pred_bboxes:

        bboxes = []
        scores = []
        labels = []

        if len(image_bboxes) == 0:

            preds.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            })

            continue
        
        for prediction in image_bboxes:

            bboxes.append(bbox_mAP_input_format(prediction["bbox"]))

            scores.append(prediction["confidence"])

            # only drone class
            labels.append(1)

        preds.append({
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            })

    return preds


def convert_ground_truth_to_mAP_input_format(label_bboxes):
    """
        Convert predict() output to MeanAveragePrecision() method input format
    """
    targets = []

    for image_bboxes in label_bboxes:

        bboxes = []
        labels = []

        if len(image_bboxes) == 0:

            targets.append({
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.int64)
            })

            continue

        for ground_truth in image_bboxes:

            bboxes.append(bbox_mAP_input_format(ground_truth["bbox"]))
            labels.append(1)

        targets.append({
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })

    return targets


def validate(train_set, validation_set, hyperparameter_combinations, fine_tune, output_file, save_models=False, model_save_dir="saved_models", use_different_learning_rates=False):

    detection_layers_list = hyperparameter_combinations["detection_layers"]
    activation_func_list =  hyperparameter_combinations["activation_func"]
    dropout_rate_list = hyperparameter_combinations["dropout_rate"]

    positive_class_bce_coefficient_list = hyperparameter_combinations["positive_class_bce_coefficient"]
    localization_loss_coefficent_list =  hyperparameter_combinations["localization_loss_coefficent"]
    bbox_loss_func_list = hyperparameter_combinations["bbox_loss_func"]

    fine_tune_layers_num_list = hyperparameter_combinations["fine_tune_layers_num"]

    optimizer_list = hyperparameter_combinations["optimizer"]
    optimizer_kwargs_list = hyperparameter_combinations["optimizer_kwargs"]

    first_max_num_epochs_list = hyperparameter_combinations["first_max_num_epochs"]
    second_max_num_epochs_list = hyperparameter_combinations["second_max_num_epochs"]
    patience_list = hyperparameter_combinations["patience"]

    objectness_threshold_list = hyperparameter_combinations["objectness_threshold"]

    batch_size_list = hyperparameter_combinations["batch_size"]

    early_stop_step1_list = hyperparameter_combinations["early_stop_step1"]

    early_stop_step2_list = hyperparameter_combinations["early_stop_step2"]

    if use_different_learning_rates:
        lr2 = hyperparameter_combinations["lr2"]

    result_rows = []

    if save_models:
        os.makedirs(model_save_dir, exist_ok=True)

    model_combination_index = 0

    for detection_layers in detection_layers_list:
        for activation_func in activation_func_list:
            for dropout_rate in dropout_rate_list:
                for positive_class_bce_coefficient in positive_class_bce_coefficient_list:
                    for localization_loss_coefficent in localization_loss_coefficent_list:
                        for bbox_loss_func in bbox_loss_func_list:
                            for fine_tune_layers_num in fine_tune_layers_num_list:
                                for optimizer_class in optimizer_list:
                                    for optimizer_kwargs in optimizer_kwargs_list:
                                        for first_max_num_epochs in first_max_num_epochs_list:
                                            for second_max_num_epochs in second_max_num_epochs_list:
                                                for early_stop_step1 in early_stop_step1_list:
                                                    for early_stop_step2 in early_stop_step2_list:
                                                        for patience in patience_list:
                                                            for batch_size in batch_size_list:
                                                                
                                                                # build the model
                                                                model = TransferLearningModel(
                                                                    detection_layers,
                                                                    activation_func,
                                                                    dropout_rate
                                                                )

                                                                criterion = DroneLoss(
                                                                    positive_class_bce_coefficient,
                                                                    localization_loss_coefficent,
                                                                    bbox_loss_func
                                                                )

                                                                if use_different_learning_rates:
                                                                    optimizer = optimizer_class([
                                                                                {"params": model.detection_network.parameters(), "lr": optimizer_kwargs["lr"]},
                                                                                {"params": model.resnetLayers.parameters(), "lr": lr2},
                                                                                    ])
                                                                else:
                                                                    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

                                                                # prepare DatasetLoader
                                                                train_set_loader = DataLoader(
                                                                                            train_set,
                                                                                            batch_size=batch_size,
                                                                                            shuffle=True,
                                                                                            num_workers=4,
                                                                                            pin_memory=True,
                                                                                            collate_fn=add_padding_fn
                                                                                            )

                                                                validation_set_loader = DataLoader(
                                                                                            validation_set,
                                                                                            batch_size=batch_size,
                                                                                            shuffle=False,
                                                                                            num_workers=4,
                                                                                            pin_memory=True,
                                                                                            collate_fn=add_padding_fn
                                                                                            )

                                                                # train model
                                                                
                                                                # first train, do not apply early stopping
                                                                train_bbox_preds, train_bbox_labels, loss_list, total_loss_list1, class_loss_list1, localization_loss_list1 = train(model, 0, train_set_loader, criterion, optimizer, first_max_num_epochs, early_stop_step1, patience)

                                                                # second train, apply early stopping
                                                                if fine_tune:
                                                                    train_bbox_preds, train_bbox_labels, loss_list, total_loss_list2, class_loss_list2, localization_loss_list2 = train(model, fine_tune_layers_num, train_set_loader, criterion, optimizer, second_max_num_epochs, early_stop_step2, patience)
                                                                else:
                                                                    total_loss_list2 = []
                                                                    class_loss_list2 = []
                                                                    localization_loss_list2 = []

                                                                # validate model
                                                                val_bbox_preds, val_bbox_labels = predict(model, validation_set_loader)

                                                                # ------------------------------------------------------------------------------
                                                                # calculate mAP50 for train results
                                                                train_metric_pred = convert_preds_to_mAP_input_format(train_bbox_preds)
                                                                train_metric_ground_truth = convert_ground_truth_to_mAP_input_format(train_bbox_labels)
                    
                                                                train_metric = MeanAveragePrecision(iou_thresholds=[0.5])
                                                                train_metric.update(train_metric_pred, train_metric_ground_truth)
                                                                train_result = train_metric.compute()
                                                                # ------------------------------------------------------------------------------

                                                                # calculate mAP50 for validation results
                                                                val_metric_pred = convert_preds_to_mAP_input_format(val_bbox_preds)
                                                                val_metric_ground_truth = convert_ground_truth_to_mAP_input_format(val_bbox_labels)

                                                                val_metric = MeanAveragePrecision(iou_thresholds=[0.5])
                                                                val_metric.update(val_metric_pred, val_metric_ground_truth)
                                                                val_result = val_metric.compute()
                                                                # ------------------------------------------------------------------------------
                                                                
                                                                for objectness_threshold in objectness_threshold_list:
                                                                    # choose optimal threshold
                                                                    
                                                                    # ------------------------------------------------------------------------------
                                                                    train_TP, train_FP, train_FN = calculate_detection_metrics(train_bbox_preds, train_bbox_labels, 0.5, objectness_threshold)
                                                                    train_precision, train_recall, train_f1 = calculate_metrics(train_TP, train_FP, train_FN)

                                                                    val_TP, val_FP, val_FN = calculate_detection_metrics(val_bbox_preds, val_bbox_labels, 0.5, objectness_threshold)
                                                                    val_precision, val_recall, val_f1 = calculate_metrics(val_TP, val_FP, val_FN)


                                                                    row = {

                                                                        # ---------------- hyperparameters ---------------- 
                                                                        "detection_layers": model.detection_layers,
                                                                        "activation_func": model.activation_func.__name__,
                                                                        "dropout_rate": model.dropout_rate,

                                                                        "positive_class_bce_coefficient": positive_class_bce_coefficient,
                                                                        "localization_loss_coefficent": localization_loss_coefficent, 
                                                                        "bbox_loss_func": bbox_loss_func.__name__,

                                                                        "fine_tune": fine_tune,
                                                                        "fine_tune_layers_num": fine_tune_layers_num,
                                                                        "optimizer": optimizer_class.__name__,
                                                                        "optimizer_kwargs": optimizer_kwargs,
                                                                        "first_max_num_epochs": first_max_num_epochs,
                                                                        "second_max_num_epochs": second_max_num_epochs,
                                                                        "patience": patience,
                                                                        "objectness_threshold": objectness_threshold,

                                                                        "batch_size": batch_size,

                                                                        "early_stop_step1": early_stop_step1,
                                                                        "early_stop_step2": early_stop_step2,

                                                                        "total_train_loss": loss_list[0],
                                                                        "train_classification_loss": loss_list[1],
                                                                        "total_localization_loss": loss_list[2],

                                                                        # ---------- how loss changed through epochs -------
                                                                        
                                                                        # -------- train 1 -------
                                                                        "total_loss_list1": total_loss_list1, 
                                                                        "class_loss_list1": class_loss_list1, 
                                                                        "localization_loss_list1": localization_loss_list1,

                                                                        # -------- train 2 -------
                                                                        "total_loss_list2": total_loss_list2, 
                                                                        "class_loss_list2": class_loss_list2, 
                                                                        "localization_loss_list2": localization_loss_list2,

                                                                        # ---------------------------------------------------

                                                                        
                                                                        
                                                                        # -------------------- Metrics ------------------
                                                                        "train_TP": train_TP,
                                                                        "train_FP": train_FP,
                                                                        "train_FN": train_FN,
                                                                        "train_precision": train_precision,
                                                                        "train_recall": train_recall,
                                                                        "train_f1": train_f1,
                                                                        "train_mAP50": train_result["map_50"].item(),

                                                                        "val_TP": val_TP,
                                                                        "val_FP": val_FP,
                                                                        "val_FN": val_FN,
                                                                        "val_precision": val_precision,
                                                                        "val_recall": val_recall,
                                                                        "val_f1": val_f1,
                                                                        "val_mAP50": val_result["map_50"].item(),

                                                                        "model_combination_index": model_combination_index
                                                                        
                                                                    }

                                                                    result_rows.append(row)

                                                                # if save_models == True than save the model
                                                                model_path = None
                                                                if save_models:
                                                                    model_path = os.path.join(model_save_dir, f"model_{model_combination_index}.pt")
                                                                    torch.save({
                                                                        "model_state_dict": model.state_dict(),
                                                                        "detection_layers": model.detection_layers,
                                                                        "activation_func": model.activation_func,
                                                                        "dropout_rate": model.dropout_rate
                                                                    }, model_path)

                                                                model_combination_index += 1
                                                                #-----------------------------------------------
                                                                
                                                                # clear memory
                                                                del model
                                                                del optimizer
                                                                del criterion
                                                                del train_set_loader
                                                                del validation_set_loader
                                                                gc.collect()
                                                                torch.cuda.empty_cache()
                                                                print("combination validated")

                                                                with open(output_file, "w", encoding="utf-8") as f:
                                                                    json.dump(result_rows, f, indent=4, ensure_ascii=False)
