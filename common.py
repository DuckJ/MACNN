import os
import keras
import numpy as np 
import cPickle as p
from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense,Input, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Convolution2D,AveragePooling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions,preprocess_input,_obtain_input_shape
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from PIL import Image as pil_image
import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--batchsize', type=int, default=18)
# parser.add_argument('--epochs', type =int, default=100)
# args = parser.parse_args()

# batchsize = args.batchsize
# epochs = args.epochs
# print batchsize
# print epochs

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

batch_size = 6
epochs = 100
num_classes = 200
nb_train_samples = 5994
nb_val_samples = 5794

# nb_train_samples = 3916
# nb_val_samples = 692

# data_path = "/mnt/zhangjin/visual_genome_python_driver/input"
train_filename = '/home/zhangjin/bird_data/train_list.txt'
val_filename = '/home/zhangjin/bird_data/test_list.txt'
train_data_dir = '/home/zhangjin/train'
val_data_dir = '/home/zhangjin/val' 


# train_data_dir = "/mnt/zhangjin/visual_genome_python_driver/data_scene/train"
# val_data_dir = "/mnt/zhangjin/visual_genome_python_driver/data_scene/vali"

path = 'vgg19_channelgroup.h5'

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




train_datagen = ImageDataGenerator(rescale=1. / 255)
# train_datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#         
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# # this is the augmentation configuration we will use for testing:
# # only rescaling

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(448, 448),
    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(448, 448),
    batch_size=batch_size)

input_tensor = Input(shape=(448,448,3))
model = VGG19(weights='imagenet', include_top=False,input_tensor = input_tensor)
# model = new_VGG16(weights='imagenet', include_top=False,input_tensor = input_tensor)
print('Model loaded.')
# print model.output.shape
# model.pop()
for layer in model.layers:
  layer.trainable = False
# for layer in model.layers:
#   print(layer.name,layer.output_shape)
def trans(x):
	return tf.transpose(x,[0,2,1])

def change(x): 
	x = tf.argmax(x,axis=2)
	# x = tf.reshape(x,[-1,512,1])
	# x1 = x % 28 
	# y1 = x / 28
	# x = tf.Variable([x1, y1])
	# x = tf.transpose(x,[1,2,0])
	return x


x_conv5_4 = model.output
# x = keras.layers.core.Reshape((28*28,512))(x_conv5_4)
# x = keras.layers.core.Lambda(trans)(x)
# y1 = keras.layers.core.Lambda(change)(x)

x = AveragePooling2D((28,28))(x_conv5_4)
x = Flatten()(x)
# x = keras.layers.core.Reshape((-1,512))(x)
# x = Flatten(name='flatten')(x)
x = Dense(512, activation='tanh', name='fc1')(x)
# x = Dropout(0.5)(x)
y2 = Dense(512, activation='sigmoid', name='fc2')(x)

# x = Dropout(0.5)(x)
# x = Dense(30, activation='softmax', name='predictions')(x)

final_model = Model(input_tensor,y2)

def generator(filename,data_dir,batch_size):
	label = np.loadtxt('p0_channel')
	with open(filename, 'r') as f:
		line = f.readlines()
		indexs = list(range(len(line)))
		while True:
			np.random.shuffle(indexs)
			label = label[indexs]
			lines = []
			for i in indexs:
				lines.append(line[i])
		    # for i in range(0,100,batchsize):
			for i in range(0,len(indexs), batch_size):
				if((i+batch_size)>len(indexs)):
					batch_x = lines[i:]
					batch_y = label[i:]
					result = np.empty(((len(indexs)-i),448,448,3))
				else:
					batch_x = lines[i:(i+batch_size)]
					batch_y = label[i:(i+batch_size)]
					result = np.empty((batch_size,448,448,3))
				for t,x in enumerate(batch_x):
					addr = x[:3]+'/'+x[4:]
					addr = addr.rstrip('\r\n')
					addr = addr.split(' ')[0]
					path = data_dir + '/'+ addr
					img = pil_image.open(path)
					img = img.convert('RGB')
					img = img.resize((448,448), pil_image.BILINEAR)
					x = np.asarray(img, dtype=K.floatx())
					result[t,:,:,:] = x
				result = result/255
				yield result,batch_y
		# ii = ii + 1





for i,layer in enumerate(final_model.layers):
  print(i, layer.name, layer.output_shape)

# features = final_model.predict_generator(generator(filename,batch_size),nb_train_samples // batch_size)
# np.save(open('features_train.npy', 'w'), features)

# print features.shape

# a = generator(filename,10)
# print type(a)
# # print dir(a)
# print(train_generator.next()[0].shape)

# print (a.next())
# print (a.next()[0].shape)
# print (a.next()[1].shape)
# final_model = Sequential()
# for l in model.layers:
#     final_model.add(l)

# final_model.add(top_model)

# for layer in final_model.layers:
#     print(layer.name,layer.output_shape)
#     # layer.trainable = False


opt = optimizers.SGD(lr=1e-3, momentum=0.9)
opt = keras.optimizers.Adam(lr=0.001)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
final_model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model_checkpoint=ModelCheckpoint(path,monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
final_model.fit_generator(
    generator(train_filename,train_data_dir,batch_size),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=generator(val_filename,val_data_dir,batch_size),
    validation_steps=nb_val_samples // batch_size,
    verbose = 1,
    callbacks=[model_checkpoint])
# filename = os.path.join(data_path,'d1.bin')
# with open(filename, mode='rb') as f:
#   x_train = p.load(f)
# for i in range(2,3):
#   filenames = os.path.join(data_path,'d%d.bin'%i)
#   with open(filenames, mode='rb') as f:
#     x = p.load(f)
#   x_train = np.concatenate((x_train,x))
  
# x_train = x_train.astype('float32')
# x_train = x_train/255

# train_row = x_train.shape[0]

# y_train = np.asarray([3]*170+[13]*51+[14]*102+[5]*136+[2]*153+[12]*170+[9]*170+[6]*67+[1]*60
# 	+[7]*170+[10]*119+[8]*43+[0]*187+[11]*170+[4]*170+[16]*153+[27]*170+[24]*65+[29]*81+[26]*170+[15]*170
# 	+[18]*85+[23]*170+[17]*170+[22]*170+[21]*170+[25]*67+[20]*170+[19]*48+[28]*119)
# # y_train = np.array([0]*199+[1]*396+[2]*239+[3]*889+[4]*167)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# # print x_train.shape
# # print y_train.shape
# c = list(zip(x_train, y_train))
# np.random.shuffle(c)
# x_train[:], y_train[:] = zip(*c)

# x_train = x_train.reshape(train_row, 224, 224, 3)

# print x_train.shape
# print y_train.shape

# filename = os.path.join(data_path,'v.bin')
# with open(filename, mode='rb') as f:
#   x_vali = p.load(f)

# x_vali = x_vali.astype('float32')
# x_vali = x_vali/255
# vali_row = x_vali.shape[0]
# y_vali = np.asarray([16]*27+[3]*30+[27]*30+[24]*10+[29]*14+[26]*30+[13]*9+[14]*18+[15]*30
#   +[5]*24+[18]*15+[23]*30+[2]*27+[17]*30+[12]*30+[22]*30+[21]*30+[25]*11+[20]*30+[9]*30+[6]*15
#   +[1]*10+[7]*30+[10]*21+[8]*10+[0]*33+[19]*7+[11]*30+[28]*21+[4]*30)
# y_vali = keras.utils.to_categorical(y_vali, num_classes)

# d = list(zip(x_vali, y_vali))
# np.random.shuffle(d)
# x_vali[:], y_vali[:] = zip(*d)

# x_vali = x_vali.reshape(vali_row, 224, 224, 3)
# print x_vali.shape
# print y_vali.shape

# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# final_model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_vali, y_vali))
#                     # callbacks=[early_stopping])
# # final_model.fit_generator(
# #     train_generator,
# #     steps_per_epoch=nb_train_samples // batch_size,
# #     epochs=epochs)
# score = final_model.evaluate(x_vali, y_vali, verbose=1)   
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# final_model.save_weights(global_path)
# '''
# input_a = Input(shape=(224, 224, 3))
# center_model = Model(input_tensor,model.outputs)
# out = center_model(input_a)
# print out.shape
# out = Flatten()(out)
# out = Dense(1024, activation='relu')(out)
# out = Dropout(0.5)(out)
# out = Dense(256, activation='relu')(out)
# out = Dropout(0.5)(out)
# out = Dense(5, activation='softmax')(out)

# final_model = Model(input_a,out)
# for layer in final_model.layers:
# 	print layer.name
# opt = keras.optimizers.Adam(lr=0.01)
# # opt = optimizers.SGD(lr=1e-3, momentum=0.9)
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# final_model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])

# final_model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)

# final_model.save_weights(top_model_part_path)
# '''