
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data():
    df_train = pd.read_csv('../mnist_in_csv/mnist_train.csv', delimiter=',')
    train_labels = df_train['label']
    df_train = df_train.drop(columns='label')
    print(train_labels.shape)
    print(df_train.shape)
    pixels = np.array(df_train.iloc[0, :]).reshape((28, 28))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()
    x = np.linspace(start=0, stop=27, num=8, endpoint=False, dtype='int16')
    trajectory_coordinates = np.array([x, x])
    features = []
    # random_coordinates = np.random.randint(low=0, high=27, size=(2, 8))
    for i in range(8):
        x, y = trajectory_coordinates[0, i], trajectory_coordinates[1, i]
        features.append(pixels[x, y])
    #     x, y = random_coordinates[0, i], random_coordinates[1, i]
    #     print(pixels[x, y])

    i=0
    while(i!=8):
        i = i+1
        while True:
            x, y = np.random.randint(low=0, high=27), np.random.randint(low=0, high=27)
            # print(i, " ", x, " ", y, " ", pixels[x, y])
            if pixels[x, y]:
                features.append(pixels[x, y])
                break
    features.append(train_labels.iloc[0])
    print(features)

def main():
    load_data()
    # x = np.linspace(start=0, stop=27, num=8, endpoint=False, dtype='int16')
    # trajectory_coordinates = np.array([x, x])
    # print(np.array([x, x]))
    # print(np.linspace(start=0, stop=27, num=8, endpoint=False, dtype='int16'))
    # print(np.random.randint(low=0, high=27, size=(2, 8)))

main()
