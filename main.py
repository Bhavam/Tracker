import os
from os import path as osp
import numpy as np
import motmetrics as mm
import torch
import torchvision
import data
import FRCNN_FPN
import resnet50
import tracktor
import yaml

def control(tracktor,reid):
   def __init__(self):
    #Config file upload
    with open(r'"C:/Users/HP.HP-PC/Tracker/config.yaml"') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    #default seed assumed

    #Object Detector 
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/Tracker/model_epoch_27.model"))

    #Re-identification Network
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/Tracker/ResNet_iter_25245.pth"))
    

    tracktor = tracktor(obj_detect, reid_network,config)

if(__name__ == "__main__"):
    control()