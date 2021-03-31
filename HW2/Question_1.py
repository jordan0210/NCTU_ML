import numpy as np
import math

def Discrete():
    prior = np.zeros(10)
    likelihood = np.ones((10, row, column, 32))

    for i in range(number):
        label = label_file.read(1)[0]
        prior[label] += 1
        for j in range(row):
            for k in range(column):
                pixel = image_file.read(1)[0]
                likelihood[label][j][k][int(pixel/8)] += 1

    for i in range(10):
        for j in range(row):
            for k in range(column):
                for l in range(32):
                    likelihood[i][j][k][l] = likelihood[i][j][k][l] / prior[i]
    return prior/60000, likelihood

def Continue():
    prior = np.zeros(10)
    Gaussian = np.zeros((10, row, column, 2)) # 2 means mean and variance

    for i in range(number):
        label = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                pixel = image_file.read(1)[0]
                Gaussian[label][j][k][0] = (prior[label]/(prior[label]+1)) * Gaussian[label][j][k][0] + pixel / (prior[label]+1)
                Gaussian[label][j][k][1] = (prior[label]/(prior[label]+1)) * Gaussian[label][j][k][1] + (pixel**2) / (prior[label]+1)
        prior[label] += 1

    for i in range(10):
        for j in range(row):
            for k in range(column):
                Gaussian[i][j][k][1] -= Gaussian[i][j][k][0]**2

    return prior/60000, Gaussian

if __name__ == "__main__":
    mode = int(input("Toggle option is (0 or 1): "))
    resultFile = open("result.txt", 'w')

    # training
    label_file = open("train-labels.idx1-ubyte", "rb")
    image_file = open("train-images.idx3-ubyte", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    if mode == 0:
        prior, likelihood = Discrete()
    elif mode == 1:
        prior, Gaussian = Continue()

    # testing
    label_file = open("t10k-labels.idx1-ubyte", "rb")
    image_file = open("t10k-images.idx3-ubyte", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    err = 0

    for image_index in range(number):
        resultFile.write("Postirior (in log scale):\n")

        # read testdata
        answer = label_file.read(1)[0]
        testdata = np.zeros((row, column), dtype=int)
        for i in range(row):
            for j in range(column):
                testdata[i][j] = image_file.read(1)[0]
                if mode == 0:
                    testdata[i][j] /= 8

        # Calculate postirior
        postirior = np.zeros(10)
        for label in range(10):
            postirior += np.log10(prior[label])
            for i in range(row):
                for j in range(column):
                    if mode == 0:
                        postirior[label] += np.log10(likelihood[label][i][j][testdata[i][j]])
                    elif mode == 1:
                        mean = Gaussian[label][i][j][0]
                        variance = Gaussian[label][i][j][1]
                        if variance != 0:
                            postirior[label] += -0.5 * math.log10(2 * math.pi * variance) - math.log10(math.exp(1)) * ((testdata[i][j] - mean) ** 2) / (2 * variance)
        prediction = np.argmax(postirior)
        for label in range(10):
            resultFile.write(f"{label}: {postirior[label]/np.sum(postirior)}\n")
        if prediction != answer:
            err += 1
        resultFile.write(f"Prediction: {prediction}, Ans: {answer}\n\n")

    # Print Bayesian classifier
    resultFile.write("Imagination of numbers in Bayesian classifier:")
    for label in range(10):
        resultFile.write(f"\n{label}:\n")
        for i in range(row):
            for j in range(column):
                if mode == 0:
                    classifier_value = np.argmax(likelihood[label][i][j])
                    resultFile.write(f"{int(classifier_value/16)} ")
                elif mode == 1:
                    classifier_value = Gaussian[label][i][j][0]   # The MLE of Gaussian is mean
                    resultFile.write(f"{int(classifier_value/128)} ")
            resultFile.write("\n\n")

    resultFile.write(f"Error rate: {err/number}")