import sklearn
import numpy as np
from utils import sampling



if __name__ == "__main__":
    light = np.load("lights.npy")
    xdata = np.load("X_data.npy")
    tenper = len(xdata) // 10
    unlabeled, labeled = sampling(len(xdata), tenper)

    labeled_data = xdata[labeled, :]
    label = light[labeled, :]

    for i in range(4):
        label_feature = np.zeros(4)
        label_feature[i] = 1
        label_index = np.where(label == label_feature)
        print(labeled_data[label_index])


