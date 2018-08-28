import os
import numpy as np
import pandas as pd

import config
import create_dataset

import MODELS

from keras.callbacks import EarlyStopping, ModelCheckpoint


train_dir = config.dir.train_dir

height = config.standard_vals.height
width = config.standard_vals.width
channels = config.standard_vals.channels

(X_train,X_val,y_train,y_val) = create_dataset.create_dataset(train_dir,height,width,channels)

# creating one hot encoded labels
targets_series = pd.Series(y_train)
one_hot = pd.get_dummies(targets_series, sparse=True)
y_train = np.asarray(one_hot)

targets_series = pd.Series(y_val)
one_hot = pd.get_dummies(targets_series, sparse=True)
y_val = np.asarray(one_hot)

#model = MODEL.inceptionv3(height,width,channels)
model = MODEL.simple_model(height,width,channels)

early_stop = EarlyStopping(monitor='val_loss',patience = 4,verbose=1)
checkpointer = ModelCheckpoint(filepath = 'cnnbest.hdf5',verbose=1,save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn = model.fit(X_train,y_train, batch_size=100, epochs=20, validation_data=(X_val,y_val), callbacks = [checkpointer,early_stop], verbose=1, shuffle=True)
