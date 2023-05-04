import json
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

def convert_from_yolov7_format(file_path,img_width = 2560,img_height=1620):
    """
    params:
        file_path: str
        img_width: int
    return:
        result: list
    """
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
            result.append({"class":cls_name,"x_min":x_min,"y_min":y_min,"x_max":x_max,"y_max":y_max})
    return result


def caculate_iou(box1,box2):
    """
    params:
        box1: dict
        box2: dict
    return:
        iou: float
    """
    xmin1,ymin1,xmax1,ymax1 = box1["x_min"],box1["y_min"],box1["x_max"],box1["y_max"]
    xmin2,ymin2,xmax2,ymax2 = box2["x_min"],box2["y_min"],box2["x_max"],box2["y_max"]

    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2

    area1 = w1*h1
    area2 = w2*h2
    
    xmin_inter = max(xmin1, xmin2)
    ymin_inter = max(ymin1, ymin2)
    xmax_inter = min(xmax1, xmax2)
    ymax_inter = min(ymax1, ymax2)
    if xmin_inter >= xmax_inter or ymin_inter >= ymax_inter:
        return 0.0
    area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    area_union = area1 + area2 - area_inter
    iou = area_inter / area_union
    return iou

classes = ["DOB","Id","Name"]
labels_folder = 'Data_label/train/labels'
true_labels_file = os.listdir(labels_folder)
predict = convert_from_yolov7_format("annotation.txt")
result = pd.DataFrame(columns=["img_file","class","iou"])


for file in true_labels_file:
    if file == ".DS_Store":
        continue
    true_labels = convert_from_yolov7_format(os.path.join(labels_folder,file))
    for i in range(len(true_labels)):
        iou = caculate_iou(true_labels[i],predict[i])
        data = {"img_file":file,"class":classes[int(true_labels[i]["class"])],"iou":np.round(iou,2)}
        data_df = pd.DataFrame(data, index=[0])

        # Concatenate the data DataFrame to the original DataFrame
        result = pd.concat([result, data_df], ignore_index=True)

result.to_csv("result_iou.csv",index=False)


