import os
import cv2
from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot

def load(path, num_subjects):
    filenames = os.listdir(path)
    pixel = []
    label = [[i]*num_subjects for i in range(15)]
    for filename in filenames:
        image = cv2.imread(path+filename, -1)
        pixel.append(list(image.reshape(-1)))
    return np.array(pixel), np.array(label).reshape(-1)

def PCA(train_data, train_label, test_data, test_label, k, mode):
    mean = np.mean(train_data, axis=0)
    center = train_data - mean

    S = kernel(mode, train_data)

    eigenValue, eigenVector = np.linalg.eig(S)
    index = np.argsort(-eigenValue) # inverse -> max sort
    eigenValue = eigenValue[index]
    eigenVector = eigenVector[:,index]

    # remove negtive eigenValue
    for i in range(len(eigenValue)):
        if (eigenValue[i] <= 0):
            eigenValue = eigenValue[:i].real
            eigenVector = eigenVector[:, :i].real
            break

    transform = center.T@eigenVector
    show_transform(transform)

    z = transform.T @ center.T
    reconstruct = transform @ z + mean.reshape(-1, 1)
    index = np.random.choice(135, 10, replace=False)
    show_reconstruct(train_data[index], reconstruct[:, index])

    # test
    test("PCA", transform, z, train_data, train_label, test_data, test_label, mean)

    return transform, z

def LDA(pca_transform, pca_z, train_data, train_label, test_data, test_label, k, mode):
    mean = np.mean(pca_z, axis=1)
    N = pca_z.shape[0] # (134, 135)

    S_within = np.zeros((N, N))
    for i in range(15):
        S_within += np.cov(pca_z[:, i*9:i*9+9], bias=True)

    S_between = np.zeros((N, N))
    for i in range(15):
        class_mean = np.mean(pca_z[:, i*9:i*9+9], axis=1).T
        S_between += 9 * (class_mean - mean) @ (class_mean - mean).T

    S = np.linalg.inv(S_within) @ S_between
    eigenValue, eigenVector = np.linalg.eig(S)
    index = np.argsort(-eigenValue) # inverse -> max sort
    eigenValue = eigenValue[index]
    eigenVector = eigenVector[:, index]

    # remove negtive eigenValue
    for i in range(len(eigenValue)):
        if (eigenValue[i] <= 0):
            eigenValue = eigenValue[:i].real
            eigenVector = eigenVector[:, :i].real
            break

    transform = pca_transform @ eigenVector
    show_transform(transform)

    mean = np.mean(train_data, axis=0)
    center = train_data - mean
    z = transform.T @ center.T
    reconstruct = transform @ z + mean.reshape(-1, 1)
    show_reconstruct(train_data, reconstruct)

    # test
    test("LDA", transform, z, train_data, train_label, test_data, test_label, mean)

def kernel(mode, data):
    if mode == "none":
        S = np.cov(data, bias=True)
    else:
        if mode == "linear":
            S = data @ data.T
        elif mode == "poly":
            S = (0.01 * data @ data.T)**3
        elif mode == "RBF":
            S = np.exp(-0.01*cdist(data, data, 'sqeuclidean'))
        N = data.shape[0]
        one_N = np.ones((N, N))/N
        S = S - one_N @ S - S @ one_N + one_N @ S @ one_N
    return S

def test(testItem, transform, z, train_data, train_label, test_data, test_label, mean):
    test_z = transform.T @ (test_data - mean).T
    dist = np.zeros(train_data.shape[0])
    acc = 0
    for i in range(test_data.shape[0]):
        for j in range(train_data.shape[0]):
            dist[j] = cdist(test_z[:, i].reshape(1, -1), z[:, j].reshape(1, -1), 'sqeuclidean')
        knn = train_label[np.argsort(dist)[:k]]
        uniq_knn, uniq_knn_count = np.unique(knn, return_counts=True)
        predict = uniq_knn[np.argmax(uniq_knn_count)]

        # print(f"{predict}, {test_label[i]}")
        if predict == test_label[i]:
            acc += 1

    print(testItem+f" acc: {100*acc/test_data.shape[0]:.2f}%")

def show_transform(data):
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.axis("off")
        pyplot.imshow(data[:, i].reshape(231, 195), cmap="gray")
    pyplot.show()

def show_reconstruct(origin, reconstruct):
    _, axes = pyplot.subplots(2, 10)
    for i in range(10):
        axes[0][i].axis("off")
        axes[1][i].axis("off")
        axes[0][i].imshow(origin[i].reshape(231, 195), cmap="gray")
        axes[1][i].imshow(reconstruct[:, i].reshape(231, 195), cmap="gray")
    pyplot.show()

if __name__ == "__main__":
    train_data, train_label = load("./Yale_Face_Database/Training/", 9)
    test_data, test_label = load("./Yale_Face_Database/Testing/", 2)

    k = 3
    for kernel_mode in ["none", "linear", "poly", "RBF"]:
        print("kernel: " + kernel_mode)
        pca_transform, pca_z = PCA(train_data, train_label, test_data, test_label, k, kernel_mode)
        LDA(pca_transform, pca_z, train_data, train_label, test_data, test_label, k, kernel_mode)