
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def sample_50_data_each_class():
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_samples = pd.DataFrame()
    for cls_lbl in range(10):
        df_samples = df_samples.append(df_train.loc[df_train['label'] == cls_lbl].sample(n=50), ignore_index = True)

    train_labels = df_samples['label']
    df_samples = df_samples.drop(columns='label')

    print("extracting features ... ...")
    samples = []
    for irow in range(df_samples.shape[0]):
        pixels = np.array(df_samples.iloc[irow, :]).reshape((28, 28))
        # plt.imshow(pixels, cmap='gray')
        # plt.show()
        samples.append(get_n_features(pixels, class_label=train_labels.iloc[irow]))
    print("saving samples 50 per class ... ...")
    samples = np.array(samples)
    print(samples.dtype)
    np.savetxt("../saved_data/50_samples_each_class.csv", samples, delimiter=",")

def get_n_features(pixels, class_label, n=16):
    i=0
    points = []
    while(i!=n):
        i = i+1
        point = []
        while True:
            x, y = np.random.randint(low=0, high=27), np.random.randint(low=0, high=27)
            if pixels[x, y]:
                point.append(x)
                point.append(y)
                point.append(pixels[x, y])
                points.append(point)
                break
    # print(points)
    points = np.array(points)
    ind = np.lexsort((points[:,1], points[:,0]))
    sorted = points[ind]
    features = sorted[:, 2]
    # print(sorted)
    # print(features)
    features = np.append(features, class_label)
    return features

def main():
    sample_50_data_each_class()

main()
