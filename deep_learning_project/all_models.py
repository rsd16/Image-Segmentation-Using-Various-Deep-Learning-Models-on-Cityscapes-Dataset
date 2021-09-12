from . import pspnet
from . import unet
from . import segnet
from . import fcn


model_names = {}

model_names['fcn_8'] = fcn.fcn_8
model_names['fcn_32'] = fcn.fcn_32
model_names['fcn_8_vgg'] = fcn.fcn_8_vgg
model_names['fcn_32_vgg'] = fcn.fcn_32_vgg
model_names['fcn_8_resnet50'] = fcn.fcn_8_resnet50
model_names['fcn_32_resnet50'] = fcn.fcn_32_resnet50
model_names['fcn_8_mobilenet'] = fcn.fcn_8_mobilenet
model_names['fcn_32_mobilenet'] = fcn.fcn_32_mobilenet

model_names['pspnet'] = pspnet.pspnet
model_names['vgg_pspnet'] = pspnet.vgg_pspnet
model_names['resnet50_pspnet'] = pspnet.resnet50_pspnet
model_names['mobilenet_pspnet'] = pspnet.mobilenet_pspnet

model_names['segnet'] = segnet.segnet
model_names['vgg_segnet'] = segnet.vgg_segnet
model_names['resnet50_segnet'] = segnet.resnet50_segnet
model_names['mobilenet_segnet'] = segnet.mobilenet_segnet

model_names['unet_mini'] = unet.unet_mini
model_names['unet'] = unet.unet
model_names['vgg_unet'] = unet.vgg_unet
model_names['resnet50_unet'] = unet.resnet50_unet
model_names['mobilenet_unet'] = unet.mobilenet_unet
