import numpy as np
import random
from scipy.spatial.distance import cdist

from PIL import Image, ImageColor
import os

def load(filename):
    image = Image.open("./data/"+filename+".png",'r')
    data = np.array(image)
    # color data: RGB for each pixel (10000, 3)
    dataC = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # spatial data: coordinate for each pixel
    dataS = np.array([(i,j) for i in range(data.shape[0]) for j in range(data.shape[1])])
    return dataC, dataS, image.size

def kernel(gamma_S, gamma_C, S, C):
    result = np.exp(-gamma_S*cdist(S, S, 'sqeuclidean'))
    result *= np.exp(-gamma_C*cdist(C, C, 'sqeuclidean'))
    return result

def kmeans(Gram, k, mode):
    history = []

    mean = initial(Gram, k, mode)
    old_mean = np.zeros(mean.shape, dtype=Gram.dtype)
    while np.linalg.norm(mean - old_mean) > 1e-10:
        # E-step: classify all samples
        clusters = np.zeros(Gram.shape[0], dtype=int)
        for i in range(Gram.shape[0]):
            J = []
            for j in range(k):
                J.append(np.linalg.norm(Gram[i] - mean[j]))
            clusters[i] = np.argmin(J)
        history.append(clusters)

        # M-step: Update center mean
        old_mean = mean
        mean = np.zeros(mean.shape, dtype=Gram.dtype)
        counters = np.zeros(k)
        for i in range(Gram.shape[0]):
            mean[clusters[i]] += Gram[i]
            counters[clusters[i]] += 1
        for i in range(k):
            if counters[i] == 0:
                counters[i] = 1
            mean[i] /= counters[i]
    print("Total No. of iteration(s):", len(history))
    return history

def initial(Gram, k, mode):
    mean = np.zeros((k, Gram.shape[1]), dtype=Gram.dtype) # mark
    if mode == 0: # normal k-means -> random center
        center =  np.array(random.sample(range(0, 10000), k))
        mean = Gram[center,:]
    elif mode == 1: # k-means++
        mean[0] = Gram[np.random.randint(Gram.shape[0], size=1), :]
        for cluste_id in range(1, k):
            temp_dist = np.zeros((len(Gram), cluste_id))
            for i in range(len(Gram)):
                for j in range(cluste_id):
                    temp_dist[i][j] = np.linalg.norm(Gram[i]-mean[j])
            dist = np.min(temp_dist, axis=1)
            sum = np.sum(dist) * np.random.rand()
            for i in range(len(Gram)):
                sum -= dist[i]
                if sum <= 0:
                    mean[cluste_id] = Gram[i]
                    break
    return mean

def visualize(history, image_size, filename, k, mode):
    gif = []
    color = [ImageColor.getrgb('Red'), ImageColor.getrgb('Green'), ImageColor.getrgb('Blue'), ImageColor.getrgb('Yellow')]

    iteration = len(history)
    for i in range(iteration):
        gif.append(Image.new("RGB", image_size))
        for y in range(image_size[0]):
            for x in range(image_size[1]):
                gif[i].putpixel((x, y), color[history[i][y*image_size[0]+x]])
    try:
        os.mkdir("./k_means_final")
        os.mkdir("./k_means_gif")
    except OSError:
        print("dir already exist")
    gif[0].save("./k_means_gif/" + filename + f"_modek{mode}_{k}.gif",
                format='GIF',
                save_all=True,
                append_images=gif[1:],
                duration=400, loop=0)
    gif[-1].save("./k_means_final/" + filename + f"_modek{mode}_{k}.jpg", format='JPEG')

if __name__ == "__main__":
    gamma_C = 1e-3
    gamma_S = 1e-3

    filename = input("Filename: ")
    k = int(input("number of clusters: "))
    mode = int(input("0(k-means) or 1(k-means++): "))

    dataC, dataS, image_size = load(filename)
    Gram = kernel(gamma_S, gamma_C, dataS, dataC)

    history = kmeans(Gram, k, mode)
    visualize(history, image_size, filename, k, mode)