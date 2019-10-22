
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util import Util
import math
import scipy.ndimage as ndimage
import cv2
import random

pixels = []
sample_points = []


def draw_circle(event, x, y, flags, param):
    global pixels, sample_points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(pixels, (x, y), 5, (255, 0, 0), 1)
        # print('x = %d, y = %d'%(x, y))
        sample_points.append(x)
        sample_points.append(y)
        # print(point)


def extract_features_by_random_2points_from_four_grids(img_pixels, class_label):
    print("extracting features for {} ... ...", (class_label))
    points = []
    row = 0
    while row != 100:
        row = row + 25
        x_low = row - 25
        x_high = row
        y_low = 0
        y_high = 100
        while True:
            x1, y1 = np.random.randint(
                low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
            if img_pixels[x1, y1]:
                got_first_point = True
                break

        while True:
            x2, y2 = np.random.randint(
                low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
            if img_pixels[x2, y2]:
                got_second_point = True
                break
        points.append(x1)
        points.append(y1)
        points.append(x2)
        points.append(y2)
    points.append(class_label)
    print(points)
    return points


def extract_features_by_random_distance_maximization(img_pixels, class_label):
    # print(img_pixels.shape)
    print("extracting features for {} ... ...", (class_label))
    points = []
    row = 0
    while row != 100:
        row = row + 25
        x_low = row - 25
        x_high = row
        y_low = 0
        y_high = 100
        pre_dist = 0.0
        # my_x1, my_y1, my_x2, my_y2
        search_limit = 5000
        got_two_points = False
        i = 1
        while i != 10:
            i = i + 1
            search_count = 0
            got_first_point = False
            got_second_point = False
            while True:
                search_count = search_count + 1
                x1, y1 = np.random.randint(
                    low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
                if img_pixels[x1, y1]:
                    got_first_point = True
                    break
                if search_count == search_limit:
                    break

            search_count = 0
            while True:
                search_count = search_count + 1
                x2, y2 = np.random.randint(
                    low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
                if img_pixels[x2, y2]:
                    got_second_point = True
                    break
                if search_count == search_limit:
                    break

            if(got_first_point and got_second_point):
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if dist > pre_dist:
                    pre_dist = dist
                    my_x1 = x1
                    my_y1 = y1
                    my_x2 = x2
                    my_y2 = y2
                    got_two_points = True

        if got_two_points:
            print("got two points...")
            if len(points) != 16:
                points.append(my_x1)
                points.append(my_y1)
            if len(points) != 16:
                points.append(my_x2)
                points.append(my_y2)
        else:
            print("didn't found two points in ", (y_low), (y_high))
            if y_low == 75 and y_high == 100:
                row = 0

        print(len(points), row)
        if len(points) != 16 and row == 100:
            row = 0
        elif len(points) == 16:
            break

    points.append(class_label)
    print(points)
    return points


def extract_features_by_linear_search_distance_maximization(img_pixels, class_label):
    print("extracting features for {} ... ...".format(class_label))
    points = []
    row = 0
    while row != 100:
        row = row + 25
        x_low = row - 25
        x_high = row
        y_low = 0
        y_high = 100
        # print("doing sequential search ...")
        x1, y1, x2, y2, got_two_points, best_dist = do_linear_search(
            img_pixels, x_low, x_high, y_low, y_high)
        # print(x1, y1, x2, y2, got_two_points, best_dist)
        points.append(x1)
        points.append(y1)
        points.append(x2)
        points.append(y2)
    points.append(class_label)
    print(points)
    return points


def do_linear_search(img_pixels, x_low, x_high, y_low, y_high):
    best_x1 = -1
    best_y1 = -1
    best_x2 = -1
    best_y2 = -1
    got_two_points = False
    best_dist = 0
    print(x_low, x_high, y_low, y_high)
    for x1 in range(x_low, x_high):
        for y1 in range(y_low, y_high):
            if img_pixels[x1, y1] != 0:
                for x2 in range(x_low, x_high):
                    for y2 in range(y_low, y_high):
                        if img_pixels[x2, y2] != 0:
                            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            if dist > best_dist:
                                best_dist = dist
                                best_x1 = x1
                                best_y1 = y1
                                best_x2 = x2
                                best_y2 = y2
                                got_two_points = True

    return best_x1, best_y1, best_x2, best_y2, got_two_points, best_dist


def sample_50_data_each_class_bismillah():
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    df_samples = pd.DataFrame()
    for cls_lbl in range(0, 10):
        df_samples = df_samples.append(
            df_train.loc[df_train['label'] == cls_lbl].sample(n=50), ignore_index=True)

    train_labels = df_samples['label']
    # print(df_samples.shape)
    df_samples = df_samples.drop(columns='label')
    print("Creating 100x100 image from 28x28 mnist image by bilinear interpolation {}... ...".format(cls_lbl))
    samples = []
    for irow in range(df_samples.shape[0]):
        img_pixels = np.array(
            df_samples.iloc[irow, :], dtype=np.uint8).reshape((28, 28))  # 28x28
        img_pixels = img_pixels[4:24, 4:24]  # 20x20
        # bilinear interpolation 100x100
        img_pixels = ndimage.zoom(img_pixels, 5, order=1)
        # plt.imshow(img_pixels)
        # plt.show()
        print("sampling for: " + str(irow))
        # sample = extract_features_by_linear_search_distance_maximization(img_pixels, class_label=train_labels.iloc[irow])
        # sample = extract_features_by_random_distance_maximization(img_pixels, class_label=train_labels.iloc[irow])
        sample = extract_features_by_random_2points_from_four_grids(
            img_pixels, class_label=train_labels.iloc[irow])
        # print(sample)
        samples.append(sample)
    print("saving samples ... ...")
    samples = np.array(samples)
    np.savetxt("../saved_data/random_2points_from_four_grids.csv",
               samples, fmt='%i', delimiter=",")


def sample_hand_crafted_data():
    global pixels, sample_points
    print("Sampling 50 data per class ... ...")
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_samples = pd.DataFrame()
    for cls_lbl in range(10):
        df_samples = df_samples.append(
            df_train.loc[df_train['label'] == cls_lbl].sample(n=1), ignore_index=True)

    train_labels = df_samples['label']
    df_samples = df_samples.drop(columns='label')

    print("Creating 100x100 image from 28x28 mnist image by bilinear interpolation ... ...")
    samples = []
    for irow in range(df_samples.shape[0]):
        pixels = np.array(df_samples.iloc[irow, :], dtype=np.uint8).reshape(
            (28, 28))  # 28x28
        pixels = pixels[4:24, 4:24]  # 20x20
        # plt.imshow(pixels, cmap='gray')
        # bilinear interpolation 100x100
        pixels = ndimage.zoom(pixels, 5, order=1)
        window_name = 'img'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 500, 500)
        cv2.setMouseCallback(window_name, draw_circle)
        while(1):
            cv2.imshow(window_name, pixels)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # esc to stop
                cv2.destroyAllWindows()
                break
            elif (k == ord('x')):  # restart
                cv2.destroyAllWindows()
                window_name = 'img' + str(random.randint(1, 101))
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 500, 500)
                cv2.setMouseCallback(window_name, draw_circle)
                sample_points = []

        sample_points.append(train_labels.iloc[irow])
        if(len(sample_points) == 17):
            samples.append(sample_points)
        sample_points = []
        print(samples)
        # plt.imshow(pixels, cmap='gray')
        # plt.savefig('../saved_images/img_' + str(irow) + ".png")
        # plt.show()
    samples = np.array(samples)
    np.savetxt("../saved_data/50_samples_each_class_hand_made_features.csv",
               samples, fmt='%i', delimiter=",")


def sample_50_data_each_class():
    print("Sampling 50 data per class ... ...")
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_samples = pd.DataFrame()
    for cls_lbl in range(10):
        df_samples = df_samples.append(
            df_train.loc[df_train['label'] == cls_lbl].sample(n=50), ignore_index=True)

    train_labels = df_samples['label']
    df_samples = df_samples.drop(columns='label')

    print("extracting features ... ...")
    samples = []
    for irow in range(df_samples.shape[0]):  # df_samples.shape[0]
        pixels = np.array(df_samples.iloc[irow, :]).reshape((28, 28))
        pixels = pixels[4:25, 4:25]
        pixels = ndimage.zoom(pixels, 5, order=1)
        # print(pixels)
        # plt.imshow(pixels, cmap='gray')
        # plt.show()
        samples.append(get_n_features(
            pixels, class_label=train_labels.iloc[irow]))
    print("saving samples 50 per class ... ...")
    samples = np.array(samples)
    np.savetxt("../saved_data/50_samples_each_class_random_selection_without_grid_division.csv",
               samples.astype(int), fmt='%i', delimiter=",")


def get_n_features(pixels, class_label, n=8):
    i = 0
    points = []
    while(i != n):
        i = i + 1
        point = []
        while True:
            x, y = np.random.randint(
                low=0, high=100), np.random.randint(low=0, high=100)
            if pixels[x, y]:
                point.append(x)
                point.append(y)
                point.append(pixels[x, y])
                points.append(point)
                break
    # print(points)
    points = np.array(points)
    # ind = np.lexsort((points[:,1], points[:,0]))
    # sorted = points[ind]
    # features = sorted[:, 0:2]
    features = points[:, 0:2]
    # features = (features*100)/20
    # print(sorted)
    # print(features)
    features = np.append(features, class_label)
    return features


def get_data(fileName):
    df = pd.read_csv(fileName, delimiter=',', header=None)
    labels = df.iloc[:, -1]
    data = df.iloc[:, 0:16]
    return data, labels


def normalize_into_range(df):
    print("Normalizing pixel intensities into 0-100... ...")
    result = df.copy()
    for col in df.columns:
        max = df[col].max()
        result[col] = 100 * (df[col] / max)
    result = result.apply(np.floor)
    print(result.head())
    return result


def run(model_path, data, labels):
    print("Running saved model on df... ....")
    util = Util()
    model = util.load_model(model_path)
    util.predict(model=model, x_data=data, y_data=labels)


def initial_and_normalized_image_viewing():
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    df_samples = pd.DataFrame()
    for cls_lbl in range(0, 10):
        df_samples = df_samples.append(
            df_train.loc[df_train['label'] == cls_lbl].sample(n=1), ignore_index=True)
    df_samples = df_samples.drop(columns='label')
    imgs = []
    normalized_imgs = []
    for irow in range(df_samples.shape[0]):
        img_pixels = np.array(
            df_samples.iloc[irow, :], dtype=np.uint8).reshape((28, 28))  # 28x28
        imgs.append(img_pixels)
        img_pixels_container_only = img_pixels[4:24, 4:24]  # 20x20
        # bilinear interpolation 100x100
        img_pixels_normalized = ndimage.zoom(
            img_pixels_container_only, 5, order=1)
        normalized_imgs.append(img_pixels_normalized)
    n_row = 2
    n_col = 5
    _, axs = plt.subplots(n_row, n_col)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap='gray')
    plt.show()

    _, axs = plt.subplots(n_row, n_col)
    axs = axs.flatten()
    for img, ax in zip(normalized_imgs, axs):
        ax.imshow(img, cmap='gray')
    plt.show()


def main():
    # initial_and_normalized_image_viewing()
    # sample_50_data_each_class()
    # sample_50_data_each_class_bismillah()
    # data, labels = get_data('../saved_data/2_random_points_each_4of_grids_distance_maximization.csv')

    sm1 = '../saved_data/random_selection_without_grid_division.csv'
    sm2 = '../saved_data/random_2points_from_four_grids.csv'
    sm3 = '../saved_data/2_random_points_each_4of_grids_distance_maximization.csv'
    sm4 = '../saved_data/50_feature_vector_for_all_sequential_search_distance_maximization.csv'

    gini_best_model = "../saved_models/gini_bestsplitter_model.sav"
    gini_random_model = '../saved_models/gini_randomsplitter_model.sav'
    entropy_best = '../saved_models/entropy_bestsplitter_model.sav'
    entropy_random = '../saved_models/entropy_randomsplitter_model.sav'

    model_paths = [gini_best_model, gini_random_model,
                   entropy_best, entropy_random]
    SM_paths = [sm1, sm2, sm3, sm4]

    # for model_path in model_paths:
    #     for sm_path in SM_paths:
    #         print(model_path, sm_path)
    #         data, labels = get_data(sm_path)
    #         run(model_path, data, labels)

    my_best = '../saved_data/my_best.csv'
    for model_path in model_paths:
        print(model_path)
        data, labels = get_data(my_best)
        run(model_path, data, labels)


main()
