from mrcnn import visualize, utils, model, config, parallel_model
from PIL import Image
import numpy as np
import cv2
conf=config.Config()
conf.NUM_CLASSES=81

print(conf.BATCH_SIZE)
mode="training"
nn=model.MaskRCNN(mode,conf,"C:/Users/lomiag/PycharmProjects/Mask_RCNN_new\My_Project")

nn.build(mode,conf)
nn.load_weights("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/mask_rcnn_coco.h5")

im1=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN/images/2383514521_1fc8d7b0de_z.jpg"))
im2=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN/images/2502287818_41e4b0c4fb_z.jpg"))
im_arr=np.array([im1,im2])
nn.mode="inference"
nn.detect(im_arr)
#