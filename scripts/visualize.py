"""
TODO: Use these methods.
This is script is used to visualize various stages of the project output.
"""
import click
import numpy as np
from cv2 import cv2

def plot_points(img, points, resize_keypoints, og_widht, og_height):
    # resize fo the orginal image
    image_height, image_width, _ = img.shape
    xmin, ymin, xmax, ymax = np.min(points[::2]), np.min(
        points[1::2]), np.max(points[::2]), np.max(points[1::2])
    if resize_keypoints == True:
        center_point = (int((xmin+(xmax-xmin)/2) * (image_width/og_widht)),
                        int((ymin+(ymax-ymin)/2) * (image_height/og_height)))
    else:
        center_point = (int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2))

    in_index = 0
    for i in range(4):
        # print(points)
        x, y = points[in_index: in_index+2]
        if resize_keypoints == True:
            x = int(x * (image_width/og_widht))
            y = int(y * (image_height/og_height))
        else:
            x = int(x)
            y = int(y)
        # print(x,y)
        img = cv2.circle(img, (x, y), radius=0, color=(0, 0, 255), thickness=8)
        in_index = in_index + 2
    img = cv2.circle(img, center_point, radius=0,
                     color=(255, 0, 0), thickness=8)
    return img


# def visualize_og_pred(img_name, annotation, pred):
#     index = 0
#     img_path = train_data_path + '/'+img_name
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     org_img = np.copy(img)
#     pred_img = np.copy(img)
#     for val in range(3):
#         org_points = annotation[index: index+8]
#         pred_points = pred[index: index+8]
#         # print(org_points,pred_points)
#         plot_points(org_img, org_points, False)
#         plot_points(pred_img, pred_points, True)
#         index = index+8

#     fig, axis = plt.subplots(1, 2, figsize=(20, 30))
#     axis[0].imshow(org_img)
#     axis[1].imshow(pred_img)
#     axis[0].set_title('Original')
#     axis[1].set_title('Predicted')
#     axis[0].axis('off')
#     axis[1].axis('off')

"""# Visualizing model metrics"""

    # history = model.his

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # epochs = range(len(history['lr']))
    # axes[0].plot(epochs, history['val_mean_absolute_error'], label='val')
    # axes[0].plot(epochs, history['mean_absolute_error'], label='training')
    # axes[0].set_ylabel('Mean Absolute Error')
    # axes[0].set_xlabel('Epoch')
    # axes[0].legend(loc='upper right')

    # axes[1].plot(epochs, history['lr'], label='lr')
    # axes[1].set_ylabel('Learning Rate')
    # axes[1].set_xlabel('Epoch')
    # axes[1].legend(loc='upper right')