import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime
import math
import test
import cv2

import time

def getFrames(file):
    vidcap=cv2.VideoCapture(file) #capture and read the video file
    success,image=vidcap.read()
    c=0
    while success:
        cv2.imwrite("frames/%d.png" % c, image) #write the images into frames directory
        success,image=vidcap.read()
        c+=1


video = cv2.VideoCapture('video.mp4')
#write fps in fps.txt
fps = video.get(cv2.CAP_PROP_FPS)
file1 = open("fps.txt","w")
file1.write(str(int(fps)))