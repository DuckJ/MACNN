from __future__ import absolute_import
import keras
import numpy as np 
import cPickle as p
import os
import warnings
import tensorflow as tf
from keras import optimizers
from keras import applications
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense,Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Convolution2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions,preprocess_input,_obtain_input_shape

from sklearn.cluster import KMeans

nb_train_samples = 5994
nb_val_samples = 5794
batch_size = 18
epoch = 100
path = 'vgg19.h5'

gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape


    img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    model = Model(img_input, x, name='vgg19')

    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
             weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)

    return model



train_data_dir = '/home/zhangjin/train'

train_datagen = ImageDataGenerator(rescale=1. / 255)

# # this is the augmentation configuration we will use for testing:
# # only rescaling

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(448, 448),
    batch_size=batch_size,
    shuffle=False)


input_tensor = Input(shape=(448,448,3))
model = VGG19(weights='imagenet', include_top=False,input_tensor = input_tensor)


def change(x): 
	x = tf.argmax(x,axis=2)
	# x = tf.reshape(x,[-1,512,1])
	# x1 = x % 28 
	# y1 = x / 28
	# x = tf.Variable([x1, y1])
	# x = tf.transpose(x,[1,2,0])
	return x

def trans(x):
	return tf.transpose(x,[0,2,1])

x = model.layers[20].output
x = keras.layers.core.Reshape((28*28,512))(x)
x = keras.layers.core.Lambda(trans)(x)
x = keras.layers.core.Lambda(change)(x)


# x = Lambda(change)(x)

final_model = Model(input_tensor, x)
# print final_model.output

for i,layer in enumerate(final_model.layers):
  print(i, layer.name, layer.output_shape)

# opt = keras.optimizers.Adam(lr=0.001)
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# final_model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# model_checkpoint=ModelCheckpoint(path,monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
# final_model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     # validation_data=val_generator,
#     # validation_steps=nb_val_samples // batch_size,
#     verbose = 1,
#     callbacks=[model_checkpoint])

features = final_model.predict_generator(train_generator,nb_train_samples // batch_size)
np.save(open('features_train.npy', 'w'), features)

print features.shape