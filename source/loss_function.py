import torch
import torch.nn as nn

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes, lambda_box = 1.0, lambda_cls = 1.0):
        super(ObjectDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def iou_loss(self, boxes_pred, boxes_target):
        # Calculate Intersection over Union (IoU)
        x1 = torch.max(boxes_pred[:, 0], boxes_target[:, 0])
        y1 = torch.max(boxes_pred[:, 2], boxes_target[:, 2])
        x2 = torch.min(boxes_pred[:, 3], boxes_target[:, 3])
        y2 = torch.min(boxes_pred[:, 1], boxes_target[:, 1])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box_area_pred = (boxes_pred[:, 3] - boxes_pred[:, 0]) * (boxes_pred[:, 1] - boxes_pred[:, 2])
        box_area_target = (boxes_target[:, 3] - boxes_target[:, 0]) * (boxes_target[:, 1] - boxes_target[:, 2])

        union_area = box_area_pred + box_area_target - inter_area
        iou = inter_area / (union_area + 1e-6)  # Avoid division by zero

        # IoU loss
        return 1 - iou
    def forward(self, boxes_pred, boxes_target, labels_pred, labels_target):
        # Calculate losses
        box_loss = self.iou_loss(boxes_pred, boxes_target).mean()
        cls_loss_raw = self.cross_entropy_loss(labels_pred, labels_target)
        cls_loss = 1 / (1 + torch.exp(-1 * cls_loss_raw))
        # print("[Loss Module] DEBUG -----------> ", box_loss, cls_loss)

        # Combine losses
        total_loss = self.lambda_box * box_loss + self.lambda_cls * cls_loss
        return total_loss
