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

device = torch.device("cpu") #change to gpu if u have

def control(tracktor,reid):

    #default seed assumed

    #Object Detector 
    obj_detect = FRCNN_FPN(num_classes=3)
    obj_detect.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/Tracker/model_epoch_27.model"))

    #Re-identification Network
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(r"C:/Users/HP.HP-PC/Tracker/ResNet_iter_25245.pth"))
    
    tracktor = Tracktor(obj_detect, reid_network,config) #figure out how to use config.yaml as an object


if(__name__ == "__main__"):