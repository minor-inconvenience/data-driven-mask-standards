"""
Written by Elizabeth Hawke 20.11.22

Script with functions for getting 3D facial landmarks from 2D images using the face_alignment library
Based on a variation of the examples script - https://github.com/1adrianb/face-alignment/blob/master/examples/detect_landmarks_in_image.py

View the library license here: https://github.com/1adrianb/face-alignment/blob/master/LICENSE
"""

import face_alignment  # NOTE! This is a library! don't call any of your files face_alignment!
import matplotlib.pyplot as plt
import torch.cuda
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import pandas as pd
import numpy as np

# Make the labels list
labels_list = np.full((1, 17), 0)  # face
labels_list = np.concatenate((labels_list, np.full((1, 5), 1)), axis=None)  # add the eyebrows
labels_list = np.concatenate((labels_list, np.full((1, 5), 2)), axis=None)  # add the eyebrows2
labels_list = np.concatenate((labels_list, np.full((1, 4), 3)), axis=None)  # add nose
labels_list = np.concatenate((labels_list, np.full((1, 5), 4)), axis=None)  # add nostrils
labels_list = np.concatenate((labels_list, np.full((1, 6), 5)), axis=None)  # add eyes
labels_list = np.concatenate((labels_list, np.full((1, 6), 6)), axis=None)  # add eyes2
labels_list = np.concatenate((labels_list, np.full((1, 12), 7)), axis=None) # add lips
labels_list = np.concatenate((labels_list, np.full((1, 8), 8)), axis=None)  # add teeth


def save_data_to_csv(csv_path, file_name, data_to_save, column_names, mode_to_save = "w"):
    """ Saves data to csv, (overwrites whatever is in the .csv file unless mode_to_save is set to "a")"""
    # Save csv in designated place
    csv_file_name_path = csv_path + file_name

    x_coord = []
    y_coord = []
    z_coord = []

    # Split up coordinate data into columns
    for point in range(0, 68):
        try:
            x_coord.append(data_to_save[point][0])
            y_coord.append(data_to_save[point][1])
            z_coord.append(data_to_save[point][2])
        except:
            print("unable to add point " + str(point))

    # Put it into a pandas dataframe and convert to csv
    data = pd.DataFrame(data=[labels_list, x_coord, y_coord, z_coord])
    data = data.T
    print(data)
    data = data.rename(columns=column_names)
    data.to_csv(csv_file_name_path, mode=mode_to_save, index=False)


# Ok! Time for face stuff!

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'   # uses S3FD instead of BlazeFace (BlazeFace is apparently faster)
face_detector_kwargs = {"filter_threshold": 0.8}

# Run the 3D face alignment on a test image, with CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)


def read_a_face_file(face_file_path):
    """Read in the face using scikit-image"""
    try:
        input_face_img = io.imread(face_file_path)
        return input_face_img
    except FileNotFoundError:
        print("file not found")


def get_predictions(input_face_img):
    """Get predictions using face_alignment library"""
    try:
        preds = fa.get_landmarks(input_face_img)[-1]
        return preds
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory")
        try:
            torch.cuda.empty_cache()  # lots of arguments to if this actually helps at all or not
            preds = fa.get_landmarks(input_face_img)[-1]
            return preds
        except:
            print("emptied cache, but still unable to get predictions")


def read_face_and_get_predictions(face_file_path):
    """Read in face image file and get predictions"""
    input_face_img = read_a_face_file(face_file_path)
    preds = get_predictions(input_face_img)
    return input_face_img, preds


# 2D-Plotting parameters
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }


def plot_2D(input_face_img, preds, save_image_name):
    """Plots original image with points overlaid
    and also plots just the points on a grid """
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(input_face_img)

    for pred_type in pred_types.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:, 0] * 1.2,
                      preds[:, 1],
                      preds[:, 2],
                      c='cyan',
                      alpha=1.0,
                      edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                  preds[pred_type.slice, 1],
                  preds[pred_type.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])

    # Save the figure in the specified location so we can look at it again later
    plt.savefig(save_image_name)

    plt.show()
