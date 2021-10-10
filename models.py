import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Input, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras import layers as Layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet import ResNet101

def ConvBlockNoResidual(channels, depth, size, initial_stride = 1, pad_input=False, name="ConvBlock"):
  inputs = x = Layers.Input(shape=(size, size, channels if pad_input == 0 else (channels // 2)))
  
  for i in range(depth):
    x = Layers.Conv2D(filters=channels, kernel_size=3, strides=initial_stride if i == 0 else 1, padding='same')(x)
    x = Layers.BatchNormalization()(x)
    x = Layers.ReLU()(x)
    x = Layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')(x)
    x = Layers.BatchNormalization()(x)

    x = Layers.ReLU()(x)

  return tensorflow.keras.Model(inputs=inputs, outputs=x, name=name)


def ConvBlock(channels, depth, size, initial_stride = 1, pad_input=False, name="ConvBlock"):
  inputs = last_layer_output = Layers.Input(shape=(size, size, channels if pad_input == 0 else (channels // 2)))
  
  for i in range(depth):
    new_x = Layers.Conv2D(filters=channels, kernel_size=3, strides=initial_stride if i == 0 else 1, padding='same')(last_layer_output)
    new_x = Layers.BatchNormalization()(new_x)
    new_x = Layers.ReLU()(new_x)
    new_x = Layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')(new_x)
    new_x = Layers.BatchNormalization()(new_x)

    if i == 0 and pad_input:
      last_layer_output = Layers.Conv2D(filters=channels, kernel_size=1, strides=2, padding='same')(last_layer_output)
      last_layer_output = Layers.BatchNormalization()(last_layer_output)

    new_x = Layers.add([new_x, last_layer_output])
    last_layer_output = Layers.ReLU()(new_x)

  return tensorflow.keras.Model(inputs=inputs, outputs=last_layer_output, name=name)

def Custom34Model(classes, image_size, dropout_rate):
  model = Sequential(name="Custom34")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlockNoResidual(64, 3, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(128, 4, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(256, 6, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(512, 3, image_size // 16, initial_stride = 2, pad_input = True, name="Conv5"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def Custom8Model(classes, image_size, dropout_rate):
  model = Sequential(name="Custom18")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlockNoResidual(64, 1, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(128, 1, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(256, 1, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(512, 1, image_size // 16, initial_stride = 2, pad_input = True, name="Conv5"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def Custom6Model(classes, image_size, dropout_rate):
  model = Sequential(name="Custom6")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlockNoResidual(64, 1, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(128, 1, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(256, 1, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def Custom4Model(classes, image_size, dropout_rate):
  model = Sequential(name="Custom6")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlockNoResidual(64, 1, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlockNoResidual(128, 1, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def ResNet6Model(classes, image_size, dropout_rate):
  model = Sequential(name="Custom6")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlock(64, 1, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(128, 1, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(256, 1, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def ResNet34Model(classes, image_size, dropout_rate):
  model = Sequential(name="ResNet34")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlock(64, 3, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(128, 4, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(256, 6, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(512, 3, image_size // 16, initial_stride = 2, pad_input = True, name="Conv5"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def ResNet18Model(classes, image_size, dropout_rate):
  model = Sequential(name="ResNet34")
  model.add(Input(shape=(image_size, image_size, 3)))
  model.add(Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name="Conv1"))   
  model.add(MaxPool2D(3, strides=2, padding='same', name="Conv2_MaxPool"))
  model.add(ConvBlock(64, 2, image_size // 4, initial_stride = 1, pad_input = False, name="Conv2"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(128, 2, image_size // 4, initial_stride = 2, pad_input = True, name="Conv3"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(256, 2, image_size // 8, initial_stride = 2, pad_input = True, name="Conv4"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(ConvBlock(512, 2, image_size // 16, initial_stride = 2, pad_input = True, name="Conv5"))
  if dropout_rate != 0:
    model.add(Dropout(dropout_rate))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(classes, activation='softmax'))
  return model

def create_Custom4_model(classes, image_size, dropout_rate):
  return Custom4Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_Custom6_model(classes, image_size, dropout_rate):
  return Custom6Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_Custom8_model(classes, image_size, dropout_rate):
  return Custom8Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_ResNet6_model(classes, image_size, dropout_rate):
  return ResNet6Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_ResNet18_model(classes, image_size, dropout_rate):
  return ResNet18Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_ResNet34_model(classes, image_size, dropout_rate):
  return ResNet34Model(classes=classes, image_size= image_size, dropout_rate= dropout_rate)

def create_ResNet50_model(classes, image_size, dropout_rate):
  if image_size == 224:
    return ResNet50(
            include_top = True,
            classes=classes,
            weights = None,
            pooling = 'avg')
  else:
    model = Sequential()
    model.add(ResNet50(
        input_shape = (image_size, image_size, 3),
        include_top = False,
        weights = None,
        pooling = 'avg'))
    if dropout_rate != 0:
      model.add(Dropout(dropout_rate))
    model.add(Dense(classes, activation='softmax'))
    return model
  
def create_ResNet101_model(classes, image_size, dropout_rate):
  return ResNet101(
          include_top = True,
          classes=classes,
          weights = None,
          pooling = 'avg')
  
def create_InceptionV3_model(classes, image_size, dropout_rate):
    model = Sequential()
    model.add(InceptionV3(
        input_shape = (image_size, image_size, 3),
        include_top = False,
        weights = None,
        pooling = 'avg'))
    if dropout_rate != 0:
      model.add(Dropout(dropout_rate))
    model.add(Dense(classes, activation='softmax'))
    return model

def create_InceptionResNetV2_model(classes, image_size, dropout_rate):
    model = Sequential()
    model.add(InceptionResNetV2(
        input_shape = (image_size,image_size,3),
        include_top = False,
        weights = None,
        pooling = 'avg'))
    if dropout_rate != 0:
      model.add(Dropout(dropout_rate))
    model.add(Dense(classes, activation='softmax'))
    return model