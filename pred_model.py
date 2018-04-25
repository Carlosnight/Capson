def pred_model(test_generator):

    from keras.applications.xception import Xception, preprocess_input as xception_process
    from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_process
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet2_process
    from keras.models import Model
    from keras.optimizers import * # SGD
    from IPython.display import SVG
    from keras.layers import *
    import h5py as h5py
    import cv2 
    import numpy as np
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from keras.preprocessing import image
    from PIL import Image
    import shutil
    from IPython.display import display
    from keras.utils import np_utils
    from sklearn.cross_validation import train_test_split

    # model 1
    from keras.applications import xception
    tensor = Input(shape=(299,299,3))
    x_tensor = Lambda(xception_process)(tensor)
    model_xception = Xception(input_tensor=x_tensor, include_top=False, weights='imagenet') #input_shape=(299,299,3)
    x = GlobalAveragePooling2D()(model_xception.output)
    xception_model = Model(model_xception.input, x)

    xception_test = xception_model.predict_generator(test_generator, verbose=1)

    # model 2
    from keras.applications import inception_v3
    y_tensor = Lambda(inception_v3_process)(tensor)
    model_inception_v3 = InceptionV3(input_tensor=y_tensor, include_top=False, weights='imagenet') # input_shape=(299,299,3)
    y = GlobalAveragePooling2D()(model_inception_v3.output)
    inceptionV3_model = Model(inputs=tensor, outputs=y)

    inceptionV3_test = inceptionV3_model.predict_generator(test_generator, verbose=1)

    # model 3
    from keras.applications import inception_resnet_v2
    z_tensor = Lambda(inception_resnet2_process)(tensor)
    model_inceptionresnetv2 = InceptionResNetV2(input_tensor=z_tensor, include_top=False, weights='imagenet') # input_shape=(299,299,3)
    z = GlobalAveragePooling2D()(model_inceptionresnetv2.output)
    inceptionresnetv2_model = Model(model_inceptionresnetv2.input, z)

    inceptionresnetv2_test = inceptionresnetv2_model.predict_generator(test_generator, verbose=1)

    test_data = []
    test_data.append(np.array(xception_test))
    test_data.append(np.array(inceptionV3_test))
    test_data.append(np.array(inceptionresnetv2_test))

    test_data = np.concatenate(test_data, axis=1)

    return test_data

def get_generator(path):
    from keras.preprocessing import image
    gendata = image.ImageDataGenerator()
    test_generator = gendata.flow_from_directory(path, (299, 299), shuffle=False, 
                                         batch_size=16, class_mode=None)

    return test_generator