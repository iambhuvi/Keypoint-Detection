# -*- coding: utf-8 -*-

"""
This script is used to do prediction based on trained model
Usage:
    python3 ./scripts/predict.py
"""
import getopt
import logging
import os
import sys
from os.path import join
# from cloudpickle import load
from pickle import load

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import cv2
from tensorflow.keras.applications.mobilenet import preprocess_input

from model import KeyPointModel
from utility import get_keypoints, load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="config.yml")
def predict(config_file):
    """
    Main function that runs predictions
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("../log/predict.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)
    image_width = config['common']['in_image_width']
    image_height = config['common']['in_image_height']
    predict_img = config['predict']['folder_path']
    weights_path = config['common']['weights_path']

    X,img_names = preprocess(predict_img, image_width, image_height)

    model = KeyPointModel().getModel()
    logger.info(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)
    # logger.info("-----------Model Summary------------")
    # logger.info(model.summary())

    predicted_keypoints = model.predict(X)
    logger.info("Prediction Completed. Writing output to predicted.csv")
    write_output(predicted_keypoints, img_names)
    
def preprocess(predict_img, image_width, image_height):
    files_path = None
    for (root, _, files) in os.walk(predict_img, topdown=True):
        files_path = [join(root, file) for file in files]
    X = np.zeros((len(files_path), image_width, image_height, 3))

    for index, file_path in enumerate(files_path):
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(image_height, image_width))
        img = preprocess_input(np.array(img, dtype=np.float32))
        X[index] = img.copy()
    
    return X, files

def write_output(predicted_keypoints, img_files_name):
    df = pd.DataFrame(columns=['img', 'x', 'y'])
    no_of_points = int(predicted_keypoints.shape[1]/2)
    in_index_low = 0
    in_index_high = no_of_points
    for row_index in range(len(img_files_name)):
        key_point_index = 0
        for in_index in range(in_index_low, in_index_high):           
            df.loc[row_index + in_index, 'img'] = img_files_name[row_index]
            df.loc[row_index + in_index, 'x'] = int(predicted_keypoints[row_index][key_point_index])
            df.loc[row_index + in_index, 'y'] = int(predicted_keypoints[row_index][key_point_index+1])
            key_point_index = key_point_index + 2
        in_index_low = in_index_high
        in_index_high = in_index_high + no_of_points
    
    df.to_csv('predicted.csv', index=False)
if __name__ == "__main__":
    predict()
