from mrcnn import visualize, utils, model, config, parallel_model
from PIL import Image
import numpy as np
import cv2
conf=config.Config()
conf.NUM_CLASSES=81
conf.BATCH_SIZE=6
print(conf.BATCH_SIZE)
mode="training"
nn=model.MaskRCNN(mode,conf,"C:/Users/lomiag/PycharmProjects/Mask_RCNN_new\My_Project")

nn.build(mode,conf)
nn.load_weights("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/mask_rcnn_coco.h5")

im1=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN/images/2383514521_1fc8d7b0de_z.jpg"))/255
im2=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN/images/2502287818_41e4b0c4fb_z.jpg"))/255
im3=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/7933423348_c30bd9bd4e_z.jpg"))/255
im4=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/9118579087_f9ffa19e63_z.jpg"))/255
im5=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/8053677163_d4c8f416be_z.jpg"))/255
im6=np.array(Image.open("C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/9247489789_132c0d534a_z.jpg"))/255
im_arr=np.array([im1,im2,im3,im4,im5,im6])
nn.mode="inference"
print(im1)
nn.detect(im1)
