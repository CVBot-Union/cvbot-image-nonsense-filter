import numpy as np # linear algebra
import os
from PIL import Image
from skimage.transform import resize
from random import shuffle
from datetime import datetime
import argparse

list_classes = ['nonsense','people']
SIZE=224

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications

input_shape = (SIZE, SIZE, 3)

def create_model(nclass):
    base_model = applications.ResNet152V2(weights='imagenet', 
                                    include_top=False, 
                                    input_shape=input_shape)
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(nclass, 
                        activation='softmax'))
    return model

def main(fpath=None,epoch=20):
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
                optimizer=optimizers.SGD(lr=1e-5, 
                                        momentum=0.9),
                metrics=['accuracy'])
    model.summary()
    print(nclass)
    print(model.output.op.name)

    if fpath is None:
        file_path="train_model/model_chkpt_resnet.h5"
    else:
        file_path = fpath
        model.load_weights(file_path)

    checkpoint = ModelCheckpoint(file_path,monitor='val_accuracy',
    mode='max', verbose=2,save_weights_only=True,save_best_only=True)

    early = EarlyStopping(monitor="accuracy", mode="max", patience=15)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir,write_images=True)


    callbacks_list = [checkpoint, early,tensorboard_callback] #early

    history = model.fit(train_gen, 
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
    main(args.chkpt,args.epoch)
