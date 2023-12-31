import tensorflow
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.applications.vgg16 import VGG16
import torch.nn as nn
from torchvision.models.vgg import vgg16, VGG16_Weights


def vgg16_model():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    return model


'''
def VGG16Model(input_shape):
    # Define the model as a sequential sequence of layers
    vgg16_custom = Sequential()

    # Define the convolutional layers
    vgg16_custom.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    #vgg16_custom.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    vgg16_custom.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(MaxPooling2D((2, 2)))

    vgg16_custom.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(MaxPooling2D((2, 2)))

    vgg16_custom.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(MaxPooling2D((2, 2)))

    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(MaxPooling2D((2, 2)))

    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    vgg16_custom.add(MaxPooling2D((2, 2)))

    # Define classification Layers
    vgg16_custom.add(Flatten())
    vgg16_custom.add(Dense(4096, activation='relu'))
    vgg16_custom.add(Dropout(0.5))
    vgg16_custom.add(Dense(4096, activation='relu'))
    vgg16_custom.add(Dropout(0.5))
    vgg16_custom.add(Dense(3, activation='softmax'))

    # Print a summary of the model architecture
    vgg16_custom.summary()

print("here")



class VGG16_ModelSetup(nn.Module):
    def __init__(self, input_width, input_height):
        super().__init__()
        self.input_shape = (input_width, input_height, 3)
        self.classes = 3
        self.model = None
        self.build_model()

    def build_model(self):
        vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, classes=self.classes, include_top=False)

        for layer in vgg16.layers:
            layer.trainable = False

        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        self.model = model(inputs=vgg16.input, outputs=predictions)
        #self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


def vgg16_model(INPUT_WIDTH, INPUT_HEIGHT):
    vgg16 = VGG16_ModelSetup(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model
    return model
'''







