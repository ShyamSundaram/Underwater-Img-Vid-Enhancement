'''
python test.py --checkpoint CHECKPOINTS_PATH
'''
import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime
import math
import multiprocessing
from functools import partial


def enhance(img_name, result_path="results", checkpoint="checkpoints/model_best_2842.pth.tar"):
    # Check for GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    #print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    #print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    testtransform = transforms.Compose([transforms.ToTensor(),])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()

    
    img = Image.open(img_name)
    img_name = (img_name.split('/')[-1]).split('.')[0]
    # print(img_name)
    if(img.mode=='RGBA'):
        img=img.convert('RGB')
    inp = testtransform(img).unsqueeze(0)
    inp = inp.to(device)
    out = model(inp)

    corrected = unloader(out.cpu().squeeze(0))
    dir = result_path
    if not os.path.exists(dir):
        os.makedirs(dir)
    corrected.save(dir+'/{}.png'.format(img_name))
    # print(dir+'/{}.png'.format(img_name))

    endtime = datetime.datetime.now()
    # print(endtime-starttime)


def EnahanceInParallel(imgs_path="test_img"):
    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))
    
    p=multiprocessing.Pool(2)
    starttime = datetime.datetime.now()
    p.map(enhance,ori_dirs)
    endtime = datetime.datetime.now()
    print(endtime-starttime)
    #enhance("./test_img/img.png",result_path="r2")