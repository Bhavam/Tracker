import os
from os import path as osp
import numpy as np
import motmetrics as mm
import torch
import torchvision
from frcnn_fpn import FRCNN_FPN
#from resnet import resnet50
from tracktor import tracktor
import yaml

def control():
    #Config file upload
    with open(r'C:/Users/HP.HP-PC/Tracker/config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    #default seed assumed

    #Object Detector 
    obj_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    obj_detect.eval()
    #obj_detect.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/tracking_wo_bnw/output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model"), map_location={'cuda:0': 'cpu'})

    #Re-identification Network
    """  reid_network = resnet50(pretrained=True)
    reid_network.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/tracking_wo_bnw/output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth")) """
    
    t = tracktor(obj_detect,config)

if(__name__ == "__main__"):
    control()