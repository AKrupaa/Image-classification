from glob import glob

import skimage
from skimage.feature import greycomatrix, greycoprops
import numpy as np
from pandas import DataFrame
from os.path import sep, join, splitext
import os
import cv2
from os import walk
from itertools import product

feature_names = ('dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM')
distances = (1, 3, 5)
angles = (0, np.pi / 4, np.pi / 2, np.pi / 4 * 3)


def cat_pic_into_pieces(base_path_to_all_categories, folder_to_save_to, h, w):
    # _, _, filenames = next(walk(path))

    # those dirs doesn't work
    for subdir, dirs, files in os.walk(base_path_to_all_categories):
        for file in files:

            # here we go
            _, category_folder = os.path.split(subdir)

            # open a image
            frame = cv2.imread(os.path.join(subdir, file))
            # get dimensions
            f_height, f_width, f_channels = frame.shape
            # get steps
            steps_h = f_height // h
            steps_w = f_width // w
            # ...
            name_and_extension = os.path.splitext(file)
            i = 0
            # into pieces!
            for _h in range(steps_h):
                for _w in range(steps_w):
                    box = frame[_h * h:(_h + 1) * h, _w * w:(_w + 1) * w]

                    save_to_path = os.path.join(base_path_to_all_categories,
                                                folder_to_save_to)  # , category_folder)  # , f'{name_and_extension[
                    # 0]}_{i}.jpg')

                    # first of all... category_folder!
                    try:
                        os.mkdir(save_to_path)
                    except OSError as error:
                        pass

                    try:
                        save_to_path = os.path.join(base_path_to_all_categories, folder_to_save_to, category_folder)
                        os.mkdir(save_to_path)
                    except OSError as error:
                        pass

                    save_to_path = os.path.join(base_path_to_all_categories, folder_to_save_to, category_folder,
                                                f'{name_and_extension[0]}_{i:03d}.jpg')
                    # then save image
                    cv2.imwrite(str(save_to_path), box)
                    cv2.waitKey(0)
                    i = i + 1


def get_features(array):
    patch_64 = (array / np.max(array) * 63)
    patch_64 = patch_64.astype(np.uint8)
    # patch_64 = np.clip(patch_64, 0, 63)
    glcm = skimage.feature.texture.greycomatrix(patch_64, distances, angles, 64, True, True)
    # skimage.feature.texture.greycomatrix

    features_vector = []
    for feature in feature_names:
        features_vector.extend(list(greycoprops(glcm, feature).flatten()))

    return features_vector


def read_pics_and_get_features(base_path_to_cropped_pics):
    features_array = []
    for subdir, dirs, files in os.walk(base_path_to_cropped_pics):
        for file in files:
            # here we go
            _, category_folder = os.path.split(subdir)
            # open a image
            frame = cv2.imread(os.path.join(subdir, file))
            # grey
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # for every image, get features, lol
            features_array.append(get_features(np.array(grey)))

    return features_array


def get_full_names():
    dist_str = ('1', '2', '3')
    angles_str = '0deg,45def,90deg,135deg'.split(',')
    return ['_'.join(f) for f in product(feature_names, dist_str, angles_str)]


if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.realpath(__file__))
    path = 'base_pictures'
    save_folder = 'cropped'

    base_pic_folder_path = os.path.join(base_folder, path)
    # print(base_pic_folder_path)
    print('Disassembling photos...')
    # it will take some time, be patient
    cat_pic_into_pieces(base_pic_folder_path, save_folder, 128, 128)
    print('Disassembling done!')
    print("-" * 120)
    print("Getting features...")
    features = read_pics_and_get_features(os.path.join(base_pic_folder_path, save_folder))
    print("GOT IT")
    print("-" * 120)
    #  ???
    full_feature_names = get_full_names()
    # full_feature_names.append("Category")
    print("Something with pandas")

    # ???
    df = DataFrame(data=features, columns=full_feature_names)
    df.to_csv('textures_data.csv', sep=',', index=False)
    print("saved")
    print("-" * 120)
    # CLASSIFICATION

    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import preprocessing
    from sklearn import utils

    classifier = svm.SVC()
    data = np.array(features)
    # X of shape (n_samples, n_features) holding the training samples
    X = data[:, :-1]
    # an array y of class labels (strings or integers), of shape (n_samples)
    Y = data[:, -1]
    print("CLASSIFICATION")
    print("Train set")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    print("classifier fit")
    lab_enc = preprocessing.LabelEncoder()
    encoded = lab_enc.fit_transform(y_train)
    # y_train = y_train.astype('uint8')
    # ValueError: Unknown label type: 'continuous'
    classifier.fit(X_train, encoded)
    print("classifier predict")
    y_predict = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_predict)
    print(f'accuracy score = {acc}')
    print("-" * 120)
    print("DONE")
