import numpy as np # linear algebra
import os
from datetime import datetime
import argparse

list_classes = ['nonsense','people']
SIZE=139

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, Activation, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import applications

input_shape = (SIZE, SIZE, 3)

def create_model(nclass):
    base_model = applications.InceptionV3(weights='imagenet', 
                                    include_top=False, 
                                    input_shape=input_shape)
    base_model.trainable = False

    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(BatchNormalization())
    add_model.add(GlobalAveragePooling2D())
    add_model.add(Dense(1024, activation='relu'))
    add_model.add(BatchNormalization())
    add_model.add(Dense(512, activation='relu'))
    add_model.add(BatchNormalization())
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dropout(0.5))
    add_model.add(BatchNormalization())
    add_model.add(Dense(nclass,activation="softmax"))

    model = add_model
    return model

def main(fpath=None,epoch=20):
    with tf.device('/DML:1'):
        data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_gen = data_gen.flow_from_directory(
            './train/',
            target_size=(SIZE, SIZE),
            class_mode='categorical',
        )

        test_gen = data_gen.flow_from_directory(
            './test/',
            target_size=(SIZE, SIZE),
            class_mode='categorical',
        )
        nclass = len(train_gen.class_indices)

        model = create_model(nclass)
        model.compile(loss='binary_crossentropy', 
                    optimizer=optimizers.Adam(lr=1e-5),
                    metrics=['accuracy'])
        model.summary()

        if fpath is None:
            file_path="train_model/model_chkpt.h5"
        else:
            file_path = fpath
            model.load_weights(file_path)

        checkpoint = ModelCheckpoint(file_path,monitor='val_acc',
        mode='max', verbose=2,save_weights_only=True,save_best_only=True)

        early = EarlyStopping(monitor="val_acc", mode="max", patience=15)

        callbacks_list = [checkpoint, early] #early
        model.fit(test_gen, 
                epochs=epoch,
                shuffle=True,
                verbose=True,
                validation_data=test_gen,
                callbacks=callbacks_list)
        model.save('model.h5',include_optimizer=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--chkpt",type=str, help="model checkpoint")
    parser.add_argument("-e","--epoch",type=int, help="epoch steps")
    args = parser.parse_args()
    print("GPUs Available: ", tf.config.experimental.list_physical_devices('DML'))
    main(args.chkpt,args.epoch)
