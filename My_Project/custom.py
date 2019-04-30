from matplotlib import pyplot as plt
import gluon
from gluoncv import model_zoo, data, utils
from PIL import Image, ImageTk
import numpy as np
import scipy.misc
import imageio
import mxnet
import tkinter as tk


class Segmentor:
    def __init__(self, mask=True, bboxes=False, model_name='mask_rcnn_resnet50_v1b_coco'):
        self.mask = mask
        self.bboxes = bboxes
        self.net = model_zoo.get_model(model_name, pretrained=True)
        self.mask = None
        self.im = None

    def read_im(self, path):
        x, orig_img = data.transforms.presets.rcnn.load_test(path)

        return x, orig_img

    def segment(self, x, orig_img):
        """
        Segments the picture.
        """
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in self.net(x)]
        # paint segmentation mask on images directly
        width, height = orig_img.shape[1], orig_img.shape[0]
        # Expand the mask onto the image
        masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)
        # Plot the mask onto the image
        orig_img = utils.viz.plot_mask(orig_img, masks)
        self.mask = masks
        self.im = orig_img

    def create_im_with_mask(self):
        """
        Plots the segmented image.
        """

        ax = utils.viz.plot_mask(self.im, self.mask)
        return ax

    def plot_mask(self,ax):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(ax)
        plt.show()

class GUI:
    def __init__(self):
        pass

    def buildGUI(self,root):
        """

        :param root: The root of the tkinter
        :return: None
        """
        self.root = tk.Tk()
        im = Image.open("images/logo.png")
        photo = ImageTk.PhotoImage(im)

        label = tk.Label(root, image=photo)
        label.image = photo  # keep a reference!
        label.pack()
        label.place(x=600, y=5)

        self.ModelVizFrame = tk.Frame(root, width=950, height=500, background="bisque")
        self.ModelVizFrame.pack()
        self.ModelVizFrame.place(x=40, y=160)


im_dir_path="C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/"
im_name="3132016470_c27baa00e8_z.jpg"

full_im_path=im_dir_path+im_name
seg=Segmentor()

x,im=seg.read_im(full_im_path)
seg.segment(x,im)
masked_im=seg.create_im_with_mask()


fig2 = plt.figure(figsize=(15, 15))
plt.imshow(Image.open(full_im_path))
plt.show()