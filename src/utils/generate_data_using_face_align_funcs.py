import matplotlib.pyplot as plt

import face_alignment_functions

# csv columns
column_names = {0: "face part", 1: "x_coordinate", 2: "y_coordinate", 3: "z_coordinate"}

# csv labels to make telling which points are which easier (a reminder for what's in the csv, not actually used in the code)
labels_map = {
    0: 'face',
    1: 'eyebrow1',
    2: 'eyebrow2',
    3: 'nose',
    4: 'nostril',
    5: 'eye1',
    6: 'eye2',
    7: 'lips',
    8: 'teeth'
}

# where to save the csv
file_path = "C:/Users/bazbo/Documents/Uni/IIA/Hackx/data-driven-mask-standards/src/utils/"

# where to find the images
face_file_path = 'C:/Users/bazbo/Downloads/'

# list of images we're going to get data for
faces_list = []

for face in range(0, len(faces_list)):

    input_face_file_path = face_file_path + faces_list[face] + ".jpg"
    print(input_face_file_path)
    input_face_img, preds = face_alignment_functions.read_face_and_get_predictions(input_face_file_path)

    # save the data
    csv_name = "face_points_" + faces_list[face] + ".csv"
    face_alignment_functions.save_data_to_csv(file_path, csv_name, preds, column_names, mode_to_save = "a")

    # plot data
    save_image_name = file_path + faces_list[face] + ".png"
    face_alignment_functions.plot_2D(input_face_img, preds, save_image_name)


