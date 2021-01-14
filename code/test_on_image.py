import sys, os, argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import pickle 

import datasets, hopenet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', default='hopenet_robust_alpha1.pkl', type=str, help='Path of model snapshot.')
    parser.add_argument('--image_folder', default='../Downloads/Celeba/full_data/images/training/', type=str, help='Root folder to all images. Do not support recursively searching')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    assert not os.path.exists('result.pkl')
    
    cudnn.enabled = True

    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict( torch.load(args.snapshot) )
    model.cuda()
    model.eval()  

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    all_images = os.listdir(args.image_folder)
    all_images.sort()

    file_output = [ ]
    yaw_output = [] 
    pitch_output = []
    roll_output = [] 


    for i, image_file in enumerate(all_images):
        print(i)

        img = Image.open( os.path.join( args.image_folder, image_file )   )
        img = transformations(img).unsqueeze(0).cuda()      
        yaw, pitch, roll = model( img )

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        file_output.append( image_file )
        yaw_output.append( float(yaw_predicted) )
        pitch_output.append( float(pitch_predicted) )
        roll_output.append( float(roll_predicted) )


    output = { 'file_name': file_output, 'yaw':yaw_output, 'pitch':pitch_output, 'roll':roll_output }

    with open('result.pkl', "wb") as fp:  
        pickle.dump(output, fp)