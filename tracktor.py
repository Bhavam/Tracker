import cv2
import os
import numpy as np
import motmetrics as mm
import torch
import torchvision
from scipy import ndimage , misc
import csv
class box:
    def __init__(self,box_id,left,top,width,depth,height,x,y,z):
            self.bb_left = left
            self.bb_top = top
            self.bb_width = width
            self.bb_depth = depth
class frame:
    def __init__(self,obj,frame_id):
            self.id = 0
            self.pedestrians[frame_id] = obj #each id has a list of boxes
            

def tracktor():
     
        #path declaration
        path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img1'
        out_path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img2'
        path_box=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/gt/gt.txt'
        
        #list of frames
        training_data = []
        #predicting bounding boxes for each video frame
        
        #saving box locations
        with open(path_box) as bb_list:
                data = csv.reader(bb_list , delimiter = ',')
                for row in data:
                   frame(box(row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]),row[0])
        frame_count = 1
        images = []
        for img_path in os.listdir(path_image):
            
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            training_data.append(image)
            idx = frame_count

            predicted_image = cv2.rectangle(image,
            (frame[idx].bb_top,
            frame[idx].bb_left),
            (frame[idx].bb_top + frame[idx].bb_depth,
            frame[idx].bb_left + frame[idx].bb_width),(2,2,2) , 2)
            frame_count = frame_count + 1

            #writing output images to output folder
            os.chdir(out_path_image)
            out = os.path.join(out_path_image,'predicted_'+img_path)
            cv2.imwrite(out,predicted_image)

if __name__ == '__main__':
    tracktor()
