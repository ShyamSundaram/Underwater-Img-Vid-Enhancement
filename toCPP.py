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


def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    # Check for GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    
    img_name = (imgdir.split('/')[-1]).split('.')[0]
    img = Image.open(imgdir)
    if(img.mode=='RGBA'):
        img=img.convert('RGB')
    inp = testtransform(img).unsqueeze(0)
    inp = inp.to(device)
    print(inp.size())
    out = model(inp)

    corrected = unloader(out.cpu().squeeze(0))
    dir = '{}/results_{}'.format(result_path, checkpoint['epoch'])
    if not os.path.exists(dir):
        os.makedirs(dir)
    corrected.save(dir+'/{}corrected.png'.format(img_name))
    print(dir+'/{}corrected.png'.format(img_name))

def trace():
    traced_net=torch.jit.trace(PhysicalNN(),torch.randn(1,3,750,831))
    torch.jit.save(traced_net,"model.pt")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    parser.add_argument(
            '--images', help='test images folder', default='./test_img/')
    parser.add_argument(
            '--result', help='results folder', default='./results/')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    imgs = args.images
    result_path = args.result
    main(checkpoint=checkpoint, imgs_path=imgs, result_path=result_path)