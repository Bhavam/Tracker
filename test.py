import cv2
import os
import numpy as np
import motmetrics as mm
import torch
import torchvision
import csv

#path declarations
path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img1'
out_path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img2'
path_box=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/gt/gt.txt'

class box:
    def __init__(self,frame_id,box_id,left,top,width,depth,x,y,z):
            self.frame_id = frame_id
            self.box_id = box_id
            self.bb_left = left
            self.bb_top = top
            self.bb_width = width
            self.bb_depth = depth

class tracktor():
     def __init__(self,object_detector,reid_network,config):

        
        #list of frames
        training_data = []

        #predicting bounding boxes for each video frame
        
          
        frame_count = 0
        images = []
        for img_path in os.listdir(path_image):
            
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            training_data.append(image)
            idx = frame_count
            predicted_image = image

            for i in range(box_count):
                   if training_data[i].frame_id == idx:
                           cv2.rectangle(predicted_image,
                                (int(training_data[i].bb_left),int(training_data[i].bb_top)),
                                (int(training_data[i].bb_left + training_data[i].bb_width),int(training_data[i].bb_top + training_data[i].bb_depth)),
                                (255,0,0) ,2)

            frame_count = frame_count + 1
            """ cv2.imshow('x',predicted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()   """  
            #writing output images to output folder
            os.chdir(out_path_image)
            out = os.path.join(out_path_image,'predicted_'+img_path)
            cv2.imwrite(out,predicted_image)

     def read_detections(self):
        #reading bounding box information
        with open(path_box) as bb_list:
                data = csv.reader(bb_list , delimiter = ',')
                for row in data:
                   #make this a 2D dictionary to decrease loop complexity
                   training_data.append(box(float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])))
        box_count = len(training_data) 
if __name__ == '__main__':
    tracktor()