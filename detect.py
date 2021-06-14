import os

import cv2

import torch
import torch.nn as nn

from nets.nn import resnet50
from utils.util import predict
import argparse

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)
          }


def detect(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50().to(device)

    print('LOADING MODEL...')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # if you have single gpu then please modify model loading process
    model.load_state_dict(torch.load('yolo.pth')['state_dict'])
    model.eval()
    image_name = args.image
    image = cv2.imread(image_name)

    print('\nPREDICTING...')
    result = predict(model, image_name)

    for x1y1, x2y2, class_name, _, prob in result:
        color = COLORS[class_name]
        cv2.rectangle(image, x1y1, x2y2, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        p1 = (x1y1[0], x1y1[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    if args.save_img:
        cv2.imwrite('result.jpg', image)

    cv2.imshow('Prediction', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', default='assets/person.jpg', required=False, help='Path to Image file')
    parser.add_argument('--save_img', action='store_true', help='Save the Image after detection')
    parser.add_argument('--video', default='', required=False, help='Path to Video file')  # maybe later
    args = parser.parse_args()

    detect(args)
