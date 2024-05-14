# -*- coding: utf-8 -*-
"""
Created on 20221116

@author: dafei
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from models.cnn import mini_XCEPTION, big_XCEPTION
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input
import argparse
import torch.utils.data as Data


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

sys.setrecursionlimit(6000) # 设置最大递归深度为3000



import numpy as np
import build_model20221116
import sys
import keras

if __name__ == '__main__':

    # load dataset set, in this way, the loading process will be very quick
    train_X = np.load("../../dataset/fer2013/train_X.npy")
    # print (X_train.shape)
    train_y = np.load("../../dataset/fer2013/train_y.npy")
    # print (y_train.shape)
    validation_X = np.load("../../dataset/fer2013/validation_X.npy")
    validation_y = np.load("../../dataset/fer2013/validation_y.npy")

    mean_X = np.load("../../dataset/fer2013/X_mean.npy")
    train_X -= mean_X
    train_X = train_X.reshape(train_X.shape[0], 48, 48, 1)
    train_y = keras.utils.to_categorical(train_y, num_classes=7)

    validation_X -= mean_X
    validation_X = validation_X.reshape(validation_X.shape[0], 48, 48, 1)
    validation_y = keras.utils.to_categorical(validation_y, num_classes=7)

    model = build_model20221116.big_XCEPTION()
    # parameters
    epochs = 1000#16
    batch_size = 32#128
    validation_split = .2
    verbose = 1
    num_classes = 7
    patience = 50
    base_path = './'

    # callbacks
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    trained_models_path = base_path + '_big_XCEPTION'  # mini_XCEPTION
    model_names = trained_models_path + '20221116v1.h5'  # .{epoch:02d}-{val_acc:.2f}.hdf5
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint,  early_stop, reduce_lr]

    history = model.fit(train_X, train_y,steps_per_epoch=len(train_X) / batch_size, epochs=epochs, verbose=1, callbacks=callbacks, batch_size=batch_size,
                        validation_data=(validation_X, validation_y))
    build_model20221116.plot_training(history, "base")

    # fsock.close()
    model.save('../../model/fer2013_big_XCEPTION20221116v1.h5')
    print("finish")


