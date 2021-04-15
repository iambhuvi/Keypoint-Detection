import albumentations as A
import numpy as np
from cv2 import cv2
from keras.applications.mobilenet import preprocess_input
from keras.utils.data_utils import Sequence


class Car(Sequence):
    def __init__(self, x_set, y_set, mode, data_path, image_width, image_height,  batch_size, augmentations = None, no_of_aug = None):
        self.x, self.y = x_set, y_set
        self.mode = mode
        self.path = data_path
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        if mode == 'Train':
            self.no_of_aug = no_of_aug    
            if augmentations == 'Self':
                self.augment = self.get_aug()
            else:
                self.augment = augmentations

    def __len__(self):
        # return number of batches for original dataset
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx+1) * self.batch_size]

        if self.mode == 'Train':

            # print(batch_x)
            X = np.zeros((len(batch_x) * self.no_of_aug,
                          self.image_height, self.image_width, 3))
            y = np.zeros((len(batch_x) * self.no_of_aug, 24))

            index = 0
            for img_index, img_name in enumerate(batch_x):
                imgPath = self.path+'/'+img_name
                # print(img_index, imgPath)
                img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
                keypoints = batch_y[img_index]

                for aug_index in range(self.no_of_aug):
                    # print(index,aug_index, name, img.shape)
                    # print( p_keypoints)
                    augment_dict = self.augment(image=img, keypoints=keypoints)
                    p_img, p_keypoints = augment_dict['image'], augment_dict['keypoints']
                    # print( p_keypoints)
                    image_height, image_width, _ = p_img.shape
                    p_img = cv2.resize(p_img, dsize=(
                        self.image_height, self.image_width,), interpolation=cv2.INTER_CUBIC)

                    X[index+aug_index] = preprocess_input(
                        np.array(p_img, dtype=np.float32))
                    # plt.imshow(X[index+aug_index])
                    # plt.show()
                    in_index = 0
                    for val in p_keypoints:
                        y[index+aug_index][in_index: in_index+2] = [int(val[0] * (
                            self.image_width/image_width)), int(val[1] * (self.image_height/image_height))]
                        in_index = in_index + 2
                index = index+self.no_of_aug

            # randomize the image postion in training data
            randomize_index = np.arange(len(X))
            np.random.shuffle(randomize_index)
            X = X[randomize_index]
            y = y[randomize_index]
            return X, y
        else:
            X = np.zeros(
                (self.batch_size, self.image_width, self.image_height, 3))
            y = np.zeros((self.batch_size, 24))

            for index, img_name in enumerate(batch_x):
                imgPath = self.path+'/'+img_name
                # print(val, imgPath)
                img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = img.shape

                img = cv2.resize(img, dsize=(
                    self.image_height, self.image_width,), interpolation=cv2.INTER_CUBIC)

                keypoints = batch_y[index]
                X[index] = preprocess_input(np.array(img, dtype=np.float32))

                in_index = 0
                for val in keypoints:
                    y[index][in_index: in_index+2] = [int(val[0] * (
                        self.image_width/image_width)), int(val[1] * (self.image_height/image_height))]
                    in_index = in_index + 2
            return X, y


    def get_aug(self):
        """## Defining Augmentations"""

        MUL_AUGMENTATION = A.Compose([
            A.HorizontalFlip(p=0.7),
            A.OneOf([
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.7),
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(rotate_limit=10, shift_limit=0, p=0.5),
            ], p=1),
            A.Blur(p=0.3),
            A.ChannelShuffle(p=0.3),
            A.RandomBrightnessContrast(p=0.5)
        ], keypoint_params=A.KeypointParams(format='xy'))

        return MUL_AUGMENTATION
