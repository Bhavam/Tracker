import cv2
import os
import numpy as np
import motmetrics as mm
import torch
import torchvision
import csv
import matplotlib.pyplot as plt

#path declarations hardcoded for the moment
path_image=r'/home/sushant/Desktop/bhavam-code/Tracker/images/'
out_path_image=r'/home/sushant/Desktop/bhavam-code/Tracker/output/'
# path_box=r'C:/Users/HP.HP-PC/tracking_wo_bnw/data/MOT17Det/train/MOT17-02/gt/gt.txt'
img_path_ex = r'/home/sushant/Desktop/bhavam-code/Tracker/images/000004.jpg'

#Class to save box details
class box:
    def __init__(self,frame_id,box_id,left,top,width,depth,x,y,z):
            self.frame_id = frame_id
            self.box_id = box_id
            self.bb_left = left
            self.bb_top = top
            self.bb_width = width
            self.bb_depth = depth
 
    def create_frame_list(self,boxes):
        frame_map = {}
        for bx in boxes:
                if bx.frame_id in frame_map:
                    frame_map[bx.frame_id].append(bx.box_id)
                else:
                    frame_map[bx.frame_id] = [bx.box_id]
        
        return frame_map      

#main tracktor class
class tracktor:
     def __init__(self,object_detector): #,reid_network
         #self.object_detector = object_detector
         #self.reid_network = reid_network
         #figure out parameters to be minimally extracted from config file
        self.category = ['person']
        self.training_data = []
        self.boxes = []
        self.score = []
        self.threshold = 0.5
        self.object_detector = object_detector
        self.run()
        self.imge = None
         
     def run(self):
         #self.read_detections()
         self.plot_boxes()
         #self.write_boxes()
        #  self.predict_one(img_path_ex)
         
     def predict_boxes(self,img_path):
        #figure out output of boxes, scores to fit to custom bounding box class 
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            self.imge = image
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
            image = transform(image)
            pred = self.object_detector([image]) 
            #pred_class = [self.category[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
            boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] 
            self.score = list(pred[0]['scores'].detach().numpy())
            pred_t = [self.score.index(x) for x in self.score if x > self.threshold][-1]
            self.boxes = boxes[:pred_t+1]
            #pred_class = pred_class[:pred_t+1]
            return self.boxes,self.score

     def plot_boxes(self):
         i = 0;
         for img_path in os.listdir(path_image):
             i += 1
             totalImagePath = os.path.join(path_image,img_path)
             bb,s = self.predict_boxes(img_path)
             
             img = cv2.imread(totalImagePath)
            
             for i in range(len(bb)):
                 cv2.rectangle(img, bb[i][0], bb[i][1],(255 , 0 , 0),2)
             cv2.imwrite(out_path_image+"pred00"+str(i)+".jpg",img)
             cv2.waitKey(0)
             cv2.destroyAllWindows()

         
     def write_boxes(self):
        #predicting bounding boxes for each video frame  
        self.frame_count = 0
        self.images = []
        for img_path in os.listdir(path_image):
            
            total_img_path = os.path.join(path_image , img_path)
            image = cv2.imread(total_img_path)
            self.training_data.append(image)
            idx = self.frame_count
            predicted_image = image

            for i in range(self.box_count):
                   if self.training_data[i].frame_id == idx:
                           cv2.rectangle(predicted_image,
                                (int(self.training_data[i].bb_left),int(self.training_data[i].bb_top)),
                                (int(self.training_data[i].bb_left + self.training_data[i].bb_width),int(self.training_data[i].bb_top + self.training_data[i].bb_depth)),
                                (255,0,0) ,2)

            self.frame_count = self.frame_count + 1

            #Uncomment to see predicted video frames one by one 
            """ cv2.imshow('x',predicted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()   """  
            #writing output images to output folder
            os.chdir(out_path_image)
            out = os.path.join(out_path_image,'predicted_'+img_path)
            cv2.imwrite(out,predicted_image)
   
     def read_detections1(self):
        #reading bounding box information
        with open(path_box) as bb_list:
                data = csv.reader(bb_list , delimiter = ',')
                for row in data:
                   #make this a 2D dictionary to decrease loop complexity
                   self.training_data.append(box(float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])))
        self.box_count = len(self.training_data)   

     def predict_one(self,img_path):
             bb,s = self.predict_boxes(img_path)
             img = cv2.imread(img_path)
             for i in range(len(bb)):
                 cv2.rectangle(img, bb[i][0], bb[i][1],(255 , 0 , 0),2)
             cv2.imwrite(out_path_image,img)
             cv2.waitKey(0)
             cv2.destroyAllWindows()