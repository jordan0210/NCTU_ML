from K_means import load, kernel, kmeans # reuse function in k-means

import numpy as np
from scipy.spatial.distance import cdist

from PIL import Image, ImageColor
from matplotlib import pyplot
import os

def Laplacian(mode_s, Gram, filename):
    if os.path.exists(f"Laplacian_modeS{mode_s}_"+filename+".npy"):
        L = np.load(f"Laplacian_modeS{mode_s}_"+filename+".npy")
    else:
        W = Gram
        D = np.diag(np.sum(W, axis=1))
        L = D - W # ratio cut
        if mode_s == 0: # normalized cut
            # L_sym = D^(-1/2)*L*D^(-1/2)
            D_sqrt_inv = np.diag(1/np.diag(np.sqrt(D)))
            L = D_sqrt_inv @ L @ D_sqrt_inv
        np.save(f"Laplacian_modeS{mode_s}_"+filename+".npy", L)
    return L

def cal_eigen(mode_s, L, k, filename):
    if (os.path.exists(f"eigenValue_modeS{mode_s}_"+filename+".npy") and
        os.path.exists(f"eigenVector_modeS{mode_s}_"+filename+".npy")):
        eigenValue = np.load(f"eigenValue_modeS{mode_s}_"+filename+".npy")
        eigenVector = np.load(f"eigenVector_modeS{mode_s}_"+filename+".npy")
    else:
        eigenValue, eigenVector = np.linalg.eig(L)
        np.save(f"eigenValue_modeS{mode_s}_"+filename+".npy", eigenValue)
        np.save(f"eigenVector_modeS{mode_s}_"+filename+".npy", eigenVector)

    index = eigenValue.argsort()
    # select the smallest eigenValue except 0
    U = eigenVector[:, index[1: k+1]] # unnormalized
    if mode_s == 0: # normalized
        # T_ij = u_ij / (\sigma_k(u_ik**2))^(1/2)
        U /= np.sqrt(np.sum(np.power(U, 2), axis=1)).reshape(-1,1)
    return U

def visualize(history, image_size, filename, k, mode_s, mode_k):
    gif = []
    color = [ImageColor.getrgb('Red'), ImageColor.getrgb('Green'), ImageColor.getrgb('Blue'), ImageColor.getrgb('Yellow')]

    iteration = len(history)
    for i in range(iteration):
        gif.append(Image.new("RGB", image_size))
        for y in range(image_size[0]):
            for x in range(image_size[1]):
                gif[i].putpixel((x, y), color[history[i][y*image_size[0]+x]])
    try:
        os.mkdir("./spectral_clustering_gif")
        os.mkdir("./spectral_clustering_final")
    except OSError:
        print("dir already exist")
    gif[0].save("./spectral_clustering_gif/" + filename + f"_modes{mode_s}_modek{mode_k}_{k}.gif",
                format='GIF',
                save_all=True,
                append_images=gif[1:],
                duration=400, loop=0)
    gif[-1].save("./spectral_clustering_final/" + filename + f"_modek{mode}_{k}.jpg", format='JPEG')

def drawplot2D(data, result):
    pyplot.figure()
    x = data[:, 0]
    y = data[:, 1]
    pyplot.set_xlabel("1st dim")
    pyplot.set_ylabel("2nd dim")
    pyplot.title("coordinates in the eigenspace of graph Laplacian")
    for i in range(k):
        pyplot.scatter(x[result==i], y[result==i], marker='.')

def drawplot3D(data, result):
    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    ax.set_xlabel("1st dim")
    ax.set_ylabel("2nd dim")
    ax.set_zlabel("3rd dim")
    pyplot.title("coordinates in the eigenspace of graph Laplacian")
    for i in range(k):
        ax.scatter(x[result==i], y[result==i], z[result==i], '.')
    pyplot.show()

if __name__ == "__main__":
    # global parameter
    gamma_C = 1e-3
    gamma_S = 1e-3

    filename = input("Filename: ")
    k = int(input("number of clusters: "))
    # normalized -> normalized cut
    # unnormalized -> ratio cut
    mode_s = int(input("0(normalized) or 1(unnormalized): "))
    mode_k = int(input("0(k-means) or 1(k-means++): "))

    print("loading...")
    dataC, dataS, image_size = load(filename)
    print("Calculate Gram Matrix...")
    Gram = kernel(gamma_S, gamma_C, dataS, dataC)
    print("Calculate L, eigenValue and eigenVector...")
    L = Laplacian(mode_s, Gram, filename)
    U = cal_eigen(mode_s, L, k, filename)

    print("do k-means...")
    history = kmeans(U, k, mode_k)
    # visualize
    visualize(history, image_size, filename, k, mode_s, mode_k)
    if k == 2:
        drawplot2D(U, history[-1])
    elif k == 3:
        drawplot3D(U, history[-1])