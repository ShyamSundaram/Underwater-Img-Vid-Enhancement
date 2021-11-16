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


def main(img_name, checkpoint="checkpoints/model_best_2842.pth.tar", result_path="results"):
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


if __name__ == '__main__':
    main("./test_img/img.png",result_path="r2")