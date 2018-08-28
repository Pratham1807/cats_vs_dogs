import sys
import config

from PIL import Image
from keras.preprocessing.image import img_to_array,load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D

height = config.standard_vals.height
width = config.standard_vals.width
channels = config.standard_vals.channels

model = Sequential()
model.add(Conv2D(filters = 32,kernel_size=5,padding='same',activation='relu',input_shape=(height,width,channels)))
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('cnnbest.hdf5')

img = load_img(sys.argv[1])

img.show()

img = img.resize((width, height), Image.ANTIALIAS)
x = img_to_array(img)
x = x.reshape(1,height,width,channels)

pred = model.predict(x)[0]
print(pred)
if pred[1]>0.65:
    print('IT IS A DOG!!!')
elif pred[0]>0.65:
    print('IT IS A CAT!!!')
else:
    print('CAN\'T SAY ANYTHING!!!')
