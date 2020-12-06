import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
# importing libraries
import numpy as np
import torch
import math

# Constants
DEBUG = True
IMAGE_LENGTH=1280
IMAGE_WIDTH = 1280*2

def colculateRotatedCorners(deg,x,y,width,length):
    f"""
    :param deg: the rotated degree
    :param x: the bottom left x coordinate
    :param y: the bottom left y coordinate
    :param width: of the x coordinate 
    :param length: of the y coordinate
    :return: the after rotation bounding rectangular - bottomLeft, newWidth,newLength
    calculate the new bounding rectangular input(90,0,0,10,100) => return((-100,0),100,10)
    """
    deg = math.radians(deg) # to radians
    rotateMatrix = torch.tensor([[math.cos(deg), -math.sin(deg)], [math.sin(deg), math.cos(deg)]])
    corners = torch.tensor([[x,y], [x+width,y],[x,y+length],[x+width,y+length]])
    rotateCorners = torch.mm(rotateMatrix,corners.T)
    x_min = int(torch.min(rotateCorners[0]))
    x_max = int(torch.max(rotateCorners[0]))
    y_min = int(torch.min(rotateCorners[1]))
    y_max = int(torch.max(rotateCorners[1]))
    bottomLeft = (x_min,y_min)
    newWidth = abs(x_min-x_max)
    newLength = abs(y_max-y_min)
    return  bottomLeft,newWidth,newLength

def check_boundingRectangular_inImage(bottomLeft,newWidth,newLength):
    """
    Check if the ne bounding Rectangular is inside image boundaries
    :param bottomLeft:
    :param newWidth:
    :param newLength:
    :return: True - inside boundaries \False - outside
    """

    # check bottom left coordinates inside the image boundaries
    if not(0 <= bottomLeft[0] <= IMAGE_WIDTH or  0<=bottomLeft[1]<=IMAGE_LENGTH):
        return  False
    # check if corners are inside the image boundaries
    if not(0 <= newWidth + bottomLeft[0] <= IMAGE_WIDTH or  0<=newLength + bottomLeft[1]<=IMAGE_LENGTH):
        return  False
    # else everything is OK
    return  True

def RotateArray(image,degreeDelta,x_bottomLeft,y_bottomLeft,width,length):
    """
    Return an  array with rotated images and their new bounding rectangular
    :param image:
    :param degreeDelta:
    :param x_bottomLeft:
    :param y_bottomLeft:
    :param width:
    :param length:
    :return:
    """

    #TODO: try to change this rotation function to work with GPU for faster implementation
    if DEBUG:
        print('Rotated Image')

    imageRtatedArray = []
    for i in range(int(360/degreeDelta)):
        if DEBUG:
            print(f"Rotate image by {i*degreeDelta} degree")

        # calculate the new bounding rectangular and verify them:
        bottomLeft,newWidth,newLength = colculateRotatedCorners(i*degreeDelta,x_bottomLeft,y_bottomLeft,width,length)
        isInsideboundaries = check_boundingRectangular_inImage(bottomLeft,newWidth,newLength )
        if not(isInsideboundaries):
            continue

        # add rotated image to images list with the new bounding rectangular
        rotated = rotate(image, angle=i*degreeDelta, mode='wrap') # rotate image
        imageRtatedArray.append(rotated) # add it
        # TODO: add an object format for returnnig image with rectangular
    return  imageRtatedArray



if __name__ == "__main__":
    #TODO: Hoe do I read image with its rectangular data

    colculateRotatedCorners(90.,0.,0.,10.,100.)
    image = io.imread('../Images_original/test.JPG')
    array = RotateArray(image,60)
    plt.imshow(image)


    # displaying the image
    io.imshow(image)

