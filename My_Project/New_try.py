from matplotlib import pyplot as plt
import gluon
from gluoncv import model_zoo, data, utils
from PIL import Image
import numpy as np
import scipy.misc
import imageio
im_name="9247489789_132c0d534a_z.jpg"
net = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
im_fname = utils.download("",path="C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/images/"+im_name)

plt.show()
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)


ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]

# paint segmentation mask on images directly
width, height = orig_img.shape[1], orig_img.shape[0]
masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)

orig_img = utils.viz.plot_mask(orig_img, masks)

# identical to Faster RCNN object detection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax = utils.viz.plot_mask(orig_img, masks)
print(len(masks))
print(ax)
imageio.imsave('C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/Output_Files/mask_'+im_name, masks)
imageio.imsave('C:/Users/lomiag/PycharmProjects/Mask_RCNN_new/Output_Files/outfile_'+im_name, ax)



