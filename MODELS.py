from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
# create the base pre-trained model
def inceptionv3(height,width,channels):
    base_model = InceptionV3(weights='imagenet',
                             include_top=False,
                             input_shape=(height,width,channels),
                             classes = 2)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def simple_model(height,width,channels):
    model = Sequential()
    model.add(Conv2D(filters = 32,kernel_size=5,padding='same',activation='relu',input_shape=(75,75,3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 32,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 32,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))
    return model
