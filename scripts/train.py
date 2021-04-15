"""
This script is used to train and export the ML model according to the config

Usage:
    python3 ./scripts/train.py
"""
import logging
from pathlib import Path
#from cloudpickle import pickle
from pickle import dump

import click
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from Car import Car
from model import KeyPointModel
from utility import (get_keypoints, load_data, parse_config, set_logger,
                     train_test_split)


@click.command()
@click.argument("config_file", type=str, default="config.yml")
def train(config_file):
    """
    Main function that train and persists model based on training set/

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """
    ################
    # config logger
    ################
    logger = set_logger("../log/train.log")

    ###############################
    # Load config from config file
    ###############################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    keypoints_csv = Path(config['common']['labels_csv_path'])
    val_split = config['common']['val_split']
    train_img_scr_path = config['common']['img_source_path']
    test_img_scr_path = config['common']['img_source_path']
    image_width = config['common']['in_image_width']
    image_height = config['common']['in_image_height']
    
    epochs = config['train']['epochs']
    train_batch_size = config['train']['batch_size']
    weight_path = config['common']['weight_path']
    no_of_aug = config['train']['no_of_aug']
    test_batch_size = config['test']['batch_size']

    ############
    # Load Data
    ############
    logger.info(f"----------------Load the data----------------")
    selected_img, keypoint_df = load_data(keypoints_csv)
    logger.info(f"Number of selected images are {selected_img.shape}")
    logger.info(f"Few of the selected images are {selected_img[0:5]}")


    ####################################
    # Get train and test data generators
    ####################################

    X_train, y_train, X_test, y_test = train_test_split(selected_img, keypoint_df, val_split)

    train_gen = Car(x_set = X_train,
                    y_set = y_train,
                    mode = 'Train',
                    data_path = train_img_scr_path,
                    image_width= image_width,
                    image_height= image_height,
                    batch_size=train_batch_size,
                    augmentations= 'Self',
                    no_of_aug=no_of_aug)
    test_gen = Car( x_set = X_test,
                    y_set = y_test,
                    mode= 'Test',
                    data_path = test_img_scr_path,
                    image_width= image_width,
                    image_height= image_height,
                    batch_size=test_batch_size,
                    )

    #####################
    # Set and train model
    #####################

    logger.info(f"-------------------------Initiate Model---------------------")
    model = KeyPointModel().getModel()

    logger.info(f"--------------------Model Summary---------------------------")
    logger.info(f"{model.summary}")

    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_absolute_error'])

    # modelCheckPoint = ModelCheckpoint('car-{val_loss:.2f}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    earlyS = EarlyStopping(monitor='val_loss', min_delta=1, patience=3,restore_best_weights=True)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7)

    history = model.fit(x=train_gen, validation_data=test_gen, callbacks=[earlyS, reducelr], epochs=epochs)
    logger.info(history)
    logger.info("------------Saving Weights--------------")
    model.save_weights(weight_path)

if __name__ == "__main__":
    train()


