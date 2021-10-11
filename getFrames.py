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
    vidcap=cv2.VideoCapture(file)
    success,image=vidcap.read()
    c=0
    while success:
        cv2.imwrite("frames/%d.png" % c, image)
        success,image=vidcap.read()
        c+=1

# print("Started..")
# start=time.time()
# getFrames('video.mp4')
# end=time.time()

# print(f"Time taken is {end-start}")