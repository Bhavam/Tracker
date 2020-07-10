import cv2
import os
import numpy as np
import motmetrics as mm
import torch
import torchvision
import csv

#path declarations hardcoded for the moment
path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img1'
out_path_image=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/img2'
path_box=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/gt/gt.txt'

#Class to save box details
class box:
    def __init__(self,frame_id,box_id,left,top,width,depth,x,y,z):
            self.frame_id = frame_id
            self.box_id = box_id
            self.bb_left = left
            self.bb_top = top
            self.bb_width = width
            self.bb_depth = depth

#main tracktor class
class tracktor():
     def __init__(self,object_detector,reid_network,config):
         self.object_detector = object_detector
         self.reid_network = reid_network
         #figure out parameters to be minimally extracted from config file
         self.read_detections()
         self.predict_boxes()
         self.write_boxes()
     def predict_boxes(self):
        self.boxes = []
        self.score = []
        #figure out output of boxes, scores to fit to custom bounding box class
        for img_path in os.listdir(path_image): 
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            self.object_detector.load_image(image)
            boxes,scores = self.object_detector.detect(image)
     def write_boxes(self):
        #list of frames
        self.training_data = []
        #predicting bounding boxes for each video frame  
        self.frame_count = 0
        self.images = []
        for img_path in os.listdir(path_image):
            
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            training_data.append(image)
            idx = self.frame_count
            predicted_image = image

            for i in range(self.box_count):
                   if self.training_data[i].frame_id == idx:
                           cv2.rectangle(predicted_image,
                                (int(self.training_data[i].bb_left),int(self.training_data[i].bb_top)),
                                (int(self.training_data[i].bb_left + self.training_data[i].bb_width),int(self.training_data[i].bb_top + self.training_data[i].bb_depth)),
                                (255,0,0) ,2)

            frame_count = frame_count + 1

            #Uncomment to see predicted video frames one by one 
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
                   self.training_data.append(box(float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])))
        self.box_count = len(self.training_data) 

if __name__ == '__main__':
    tracktor()