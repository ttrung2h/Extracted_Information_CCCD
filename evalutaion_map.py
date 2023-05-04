import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import pandas as pd

def convert_from_yolov7_format(file_path,img_width = 2560,img_height=1620):
    result = []
    with open(file_path) as f:
        data = f.readlines()
        for line in data:
            cls_name, x, y, w, h = line.split()
            x,y,w,h = float(x),float(y),float(w),float(h)
            x_min = int((x - w/2)*img_width)
            y_min = int((y - h/2)*img_height)
            x_max = int((x + w/2)*img_width)
            y_max = int((y + h/2)*img_height)
            result.append([x_min,y_min,x_max,y_max,cls_name])
    return result

def compute_iou(gt_box, pred_box):
    """Compute IoU between two bounding boxes."""
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union_area = gt_area + pred_area - inter_area
    iou = inter_area / union_area
    return iou

def calculate_map(ground_truth, predictions, num_classes=3, iou_threshold=0.5):
    """
    Calculate mAP for object detection.
    """
    aps = []
    for cls in range(num_classes):
        gt_boxes = [box for box in ground_truth if box[4] == cls]
        pred_boxes = [box for box in predictions if box[4] == cls]

        num_gt_boxes = len(gt_boxes)
        num_pred_boxes = len(pred_boxes)

        if num_gt_boxes == 0 or num_pred_boxes == 0:
            continue

        tp = np.zeros(num_pred_boxes)
        fp = np.zeros(num_pred_boxes)

        # Sort predicted boxes by confidence score in descending order
        sorted_indices = np.argsort([-box[4] for box in pred_boxes])
        pred_boxes = [pred_boxes[i] for i in sorted_indices]

        for i in range(num_pred_boxes):
            pred_box = pred_boxes[i]

            ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou_index = np.argmax(ious)
            max_iou = ious[max_iou_index]

            if max_iou >= iou_threshold:
                if not gt_boxes[max_iou_index] in pred_boxes[:i]:
                    tp[i] = 1
                    gt_boxes[max_iou_index] = None
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / num_gt_boxes
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))

        # Calculate AP by computing the area under the precision-recall curve
        ap = 0
        for j in range(len(precision) - 1):
            ap += (recall[j + 1] - recall[j]) * precision[j + 1]
        aps.append(ap)

    # Compute mAP by taking the average of the APs across all classes
    mAP = sum(aps) / len(aps)
    return mAP


result = pd.DataFrame(columns=["img_file","mAp"])
labels_folder = 'Data_label/train/labels'
true_labels_file = os.listdir(labels_folder)

#convert from yolov7 format to xmin,ymin,xmax,ymax
predict = convert_from_yolov7_format("annotation.txt")
predict = np.array(predict).astype(np.float32)

#traverse all file in labels folder
for file in true_labels_file:
    file_path = os.path.join(labels_folder,file)
    #convert from yolov7 format to xmin,ymin,xmax,ymax in true label
    true = convert_from_yolov7_format(file_path)
    true = np.array(true).astype(np.float32)
    
    data = {"img_file":file,"mAp":np.round(calculate_map(true,predict),2)}
    df = pd.DataFrame(data,index=[0])
    result = pd.concat([result,df],ignore_index=True)

result.sort_values(by=["img_file"],ascending=True,inplace=True)
result.to_csv("result_mAp.csv",index=False)