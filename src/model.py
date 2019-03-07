from keras.applications.densenet import DenseNet169, DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, image
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K #selecing tensorflow as a backend for keras
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

#model parameters for training
K.set_learning_phase(1)

img_width, img_height = 224, 224
nb_train_samples = 36808
nb_validation_samples = 3197
epochs = 5
batch_size = 4
n_classes = 2

def getImagesInArray(train_dataframe):
    images =[]
    labels = []
    # images = list(train_dataframe['Path'].apply(load_img))
    # labels = list(train_dataframe['Label'])
    for i, data in tqdm(train_dataframe.iterrows()):
        img = load_img(data['Path'], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)
        labels.append(data['Label'])
    images = np.concatenate(images, axis=0)
    return {'images': images, 'labels': labels}

#Keras ImageDataGenerator to load, transform the images of the dataset
BASE_DATA_DIR = '../data/'
IMG_DATA_DIR = 'MURA-v1.1/'

train_data_dir = BASE_DATA_DIR + IMG_DATA_DIR + 'train'
valid_data_dir = BASE_DATA_DIR + IMG_DATA_DIR + 'valid'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range=5,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_image_df = pd.read_csv('../data/train_image_data.csv', names=['Path', 'Label'])
valid_image_df = pd.read_csv('../data/valid_image_data.csv', names=['Path', 'Label'])

valid_dict = getImagesInArray(valid_image_df)
train_dict = getImagesInArray(train_image_df)

validation_generator = test_datagen.flow(
    x=valid_dict['images'],
    y=valid_dict['labels']
)

train_generator = train_datagen.flow(
    x=train_dict['images'],
    y=train_dict['labels']
)
np.concatenate


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

#define a model (densenet 121)
def build_model():
    base_model = DenseNet121(blocks = [6, 12, 24, 16],input_shape=(img_width, img_height,3),
                             weights='imagenet',
                             include_top=False,
                             pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes,activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#Now let's build a model
model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

# #plotting the model
# from keras.utils.vis_utils import plot_model
# import pydot
# plot_model(model, to_file='images/densenet_archi.png', show_shapes=True)

#callbacks for early stopping incase of reduced learning rate, loss unimprovement
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

#train the model
model_history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size,
    callbacks=callbacks_list
)

from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops

def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return ops.convert_to_tensor(x, dtype=dtype)

def binary_crossentropy(target, output, from_logits=False):
  """Binary crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  """
  # Note: nn.sigmoid_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # transform back to logits
    epsilon_ = _to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output / (1 - output))
  return sigmoid_cross_entropy_with_logits(labels=target, logits=output)

def sigmoid_cross_entropy_with_logits(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    name=None):
  """Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `labels` must have the same type and shape.

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `logits` and `labels` do not have the same shape.
  """
  # pylint: disable=protected-access
  nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
                           labels, logits)
  # pylint: enable=protected-access

  with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    return math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)


#Now we evaluate the trained model with the validation dataset and make a prediction.
#The class predicted will be the class with maximum value for each image.
model.evaluate_generator(validation_generator, steps=nb_validation_samples //batch_size, max_queue_size=10, workers=0, use_multiprocessing=False)

pred = model.predict_generator(validation_generator, steps=nb_validation_samples //batch_size, max_queue_size=10, workers=0, use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)



# print('Confusion Matrix')
#
# cm = confusion_matrix(validation_generator.labels, np.argmax(pred, axis=1))
# plt.figure(figsize = (30,20))
# sn.set(font_scale=1.4) #for label size
# sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
# plt.show()
# print()
# print('Classification Report')
# print(classification_report(validation_generator.labels, predicted, target_names=class_names))
