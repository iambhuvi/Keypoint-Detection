import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/
    Args:
        log_path [str]: eg: "../log/train.log"
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger


def load_data(csv_path):
    """
    This method loads car keypoint information from the csv file.
    This method also take care in preprocessing that only images havig windshield and light with all 12 coordinate are selected
    """
    dataframe = pd.read_csv(csv_path)

    """# Preprocessing"""

    # get only labels whoes value is either windshield or light.
    df = dataframe[(dataframe['label'] == 'windshield')
                   | (dataframe['label'] == 'light')]

    # get image which have exactly 12 records 4-4-4 for windshield-light-light
    df_bool = pd.Series(df.groupby('name')['name'].count() == 12)
    df_bool = df_bool.reset_index(drop=False, name='val')
    selected_img = np.array(df_bool[df_bool['val'] == True]['name'])

    return selected_img, df


"""# Helper Functions"""


def train_test_split(selected_img, dataframe, val_split=0.20):
    """## Getting Train And Test sets."""

    total_images = len(selected_img)
    split_index = total_images - int(total_images*val_split)
    X_train = selected_img[:split_index]
    y_train = [get_keypoints(img_name, dataframe)
               for img_name in selected_img[:split_index]]

    X_test = selected_img[split_index:]
    y_test = [get_keypoints(img_name, dataframe)
              for img_name in selected_img[split_index:]]

    return X_train, y_train, X_test, y_test


def get_keypoints(imgName,df):
    t = df[df['name'] == imgName]
    anno = []
    for in_index in range(len(t)):
        x = t.iloc[in_index, [1]]['x']
        y = t.iloc[in_index, [2]]['y']
        anno.append((x, y))
    return anno

