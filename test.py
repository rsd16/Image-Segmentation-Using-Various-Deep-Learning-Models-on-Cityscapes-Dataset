from deep_learning_project.unet import unet
from deep_learning_project.pspnet import vgg_pspnet
import matplotlib.pyplot as plt


n_classes = 100
epochs = 5
trainset_images = 'Dataset/Training Images/'
trainset_masks = 'Dataset/Training Masks/'
testset_images = 'Dataset/Test Images/'
testset_masks = 'Dataset/Test Masks/'

models = {}

# model['model name'] = [input height, input width]

models['fcn_8'] = [416, 608]
models['fcn_32'] = [416, 608]
models['fcn_8_vgg'] = [416, 608]
models['fcn_32_vgg'] = [416, 608]
models['fcn_8_resnet50'] = [416, 608]
models['fcn_32_resnet50'] = [416, 608]
models['fcn_8_mobilenet'] = [224, 224]
models['fcn_32_mobilenet'] = [224, 224]

models['pspnet'] = [384, 576]
models['vgg_pspnet'] = [384, 576]
models['resnet50_pspnet'] = [384, 576]
models['mobilenet_pspnet'] = [224, 224]

models['segnet'] = [416, 608]
models['vgg_segnet'] = [416, 608]
models['resnet50_segnet'] = [416, 608]
models['mobilenet_segnet'] = [224, 224]

models['unet_mini'] = [360, 480]
models['unet'] = [416, 608]
models['vgg_unet'] = [416, 608]
models['resnet50_unet'] = [416, 608]
models['mobilenet_unet'] = [224, 224]

model = vgg_pspnet(n_classes=n_classes, input_height=models['vgg_pspnet'][0], input_width=models['vgg_pspnet'][1])

model.train(train_images=trainset_images, train_annotations=trainset_masks, epochs=epochs)

print(model.evaluate_segmentation(inp_images_dir=testset_images, annotations_dir=testset_masks))

segmentation_array = model.predict_segmentation(inp=r'D:\safe\university\arshad\deep\project\Dataset\Test Images\0016E5_07959.png', out_fname='0016E5_07959 - segmentation result.png')
print(segmentation_array)

segmentation_array = model.predict_segmentation(inp=r'D:\safe\university\arshad\deep\project\Dataset\Test Images\0016E5_08033.png', out_fname='0016E5_08033 - segmentation result.png')
#print(segmentation_array)

segmentation_array = model.predict_segmentation(inp=r'D:\safe\university\arshad\deep\project\Dataset\Test Images\0016E5_08073.png', out_fname='0016E5_08073 - segmentation result.png')
#print(segmentation_array)

segmentation_array = model.predict_segmentation(inp=r'D:\safe\university\arshad\deep\project\Dataset\Test Images\0016E5_08119.png', out_fname='0016E5_08119 - segmentation result.png')
#print(segmentation_array)

segmentation_array = model.predict_segmentation(inp=r'D:\safe\university\arshad\deep\project\Dataset\Test Images\0016E5_07965.png', out_fname='0016E5_07965 - segmentation result.png')
#print(segmentation_array)
