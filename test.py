import cv2
import os
import numpy as np
import motmetrics as mm
import torch
import torchvision
from scipy import ndimage , misc
import csv


def tracktor():
     
        #path declaration
        path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img1/000001.jpg'
        out_path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img2'
        path_box=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/gt/gt.txt'

        #predicting bounding boxes for each video frame
        box=[]
        #saving box locations
        with open(path_box) as bb_list:
                data = csv.reader(bb_list , delimiter = ',')
                for row in data:
                   if int(row[0]) == 1:
                       box.append([int(row[2]),int(row[3]),int(row[4]),int(row[5])])
        image = cv2.imread(path_image)
        for i in range(len(box)):
              cv2.rectangle(image,(box[i][0],box[i][1]),(box[i][0]+box[i][2],box[i][1]+box[i][3]),(255,0,0),2)
        #writing output images to output folder
        os.chdir(out_path_image)
        out = os.path.join(out_path_image,'predicted_'+path_image+'.jpg')
        cv2.imshow('x',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(out,image)

if __name__ == '__main__':
    tracktor()
