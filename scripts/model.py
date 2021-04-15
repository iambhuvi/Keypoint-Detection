from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, Dense, Dropout, Flatten, LeakyReLU
from keras.models import Model


class KeyPointModel:
    def __init__(self):
        self.model = self.getModel()

    def getModel(self):
        mobilenet = MobileNet(input_shape=(224,224,3),include_top=False, weights='imagenet') #using mobilenet as backbone

        last_layer = mobilenet.get_layer(index=-1).output #get the last layer
        for layer in mobilenet.layers: 
            layer.trainable = True # set all mobilenet layers trainable to True

        c1 = Conv2D(filters=1024, kernel_size=(3,3))(last_layer)
        l1 = LeakyReLU(alpha=0.3)(c1)
        d1 = Dropout(0.3)(l1)

        c2 = Conv2D(filters=512, kernel_size=(3,3))(d1)
        l2 = LeakyReLU(alpha=0.6)(c2)
        d2 = Dropout(0.5)(l2)

        f1 = Flatten()(d2)

        dense1 = Dense(24)(f1)
        o = LeakyReLU(alpha=0.3)(dense1)

        return Model(inputs=mobilenet.input, outputs=o)
