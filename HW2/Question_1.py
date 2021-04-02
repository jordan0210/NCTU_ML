import numba as nb
import numpy as np
import math

def Load():
    label_file = open("train-labels.idx1-ubyte", "rb")
    image_file = open("train-images.idx3-ubyte", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    trainingLabel = np.zeros(number, dtype=int)
    trainingData = np.zeros((number, row, column), dtype=int)

    for i in range(number):
        trainingLabel[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                trainingData[i][j][k] = image_file.read(1)[0]

    label_file.close()
    image_file.close()

    label_file = open("t10k-labels.idx1-ubyte", "rb")
    image_file = open("t10k-images.idx3-ubyte", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    testLabel = np.zeros(number, dtype=int)
    testData = np.zeros((number, row, column), dtype=int)
    for i in range(number):
        testLabel[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                testData[i][j][k] = image_file.read(1)[0]

    label_file.close()
    image_file.close()

    return trainingLabel, trainingData, testLabel, testData

@nb.njit()
def Discrete(labels, data):
    prior = np.zeros(10)
    likelihood = np.ones((10, 28, 28, 32))

    for i in range(len(labels)):
        prior[labels[i]] += 1
        for j in range(28):
            for k in range(28):
                pixel = data[i][j][k]
                likelihood[labels[i]][j][k][int(pixel/8)] += 1

    for i in range(10):
        for j in range(28):
            for k in range(28):
                for l in range(32):
                    likelihood[i][j][k][l] = likelihood[i][j][k][l] / prior[i]
    return prior/60000, likelihood

@nb.njit()
def Continue(labels, data):
    prior = np.zeros(10)
    Gaussian = np.zeros((10, 28, 28, 2)) # 2 means mean and variance

    for i in range(len(labels)):
        label = labels[i]
        for j in range(28):
            for k in range(28):
                pixel = data[i][j][k]
                Gaussian[label][j][k][0] = (prior[label]/(prior[label]+1)) * Gaussian[label][j][k][0] + pixel / (prior[label]+1)
                Gaussian[label][j][k][1] = (prior[label]/(prior[label]+1)) * Gaussian[label][j][k][1] + (pixel**2) / (prior[label]+1)
        prior[label] += 1

    for i in range(10):
        for j in range(28):
            for k in range(28):
                Gaussian[i][j][k][1] -= Gaussian[i][j][k][0]**2

    return prior/60000, Gaussian

@nb.njit()
def test(mode, labels, data):
    err = 0.
    # result = ""

    posterior = np.zeros((len(labels), 10))
    predictions = np.zeros(len(labels))
    answers = np.zeros(len(labels))
    for image_index in range(len(labels)):
        # Calculate posterior
        for label in range(10):
            posterior[image_index] += np.log10(prior[label])
            for i in range(28):
                for j in range(28):
                    if mode == 0:
                        posterior[image_index][label] += np.log10(likelihood[label][i][j][int(data[image_index][i][j]/8)])
                    elif mode == 1:
                        mean = likelihood[label][i][j][0]
                        variance = likelihood[label][i][j][1]
                        if variance != 0:
                            posterior[image_index][label] += -0.5 * math.log10(2 * math.pi * variance) - math.log10(math.exp(1)) * ((data[image_index][i][j] - mean) ** 2) / (2 * variance)
        predictions[image_index] = np.argmax(posterior[image_index])
        answers[image_index] = labels[image_index]
        if predictions[image_index] != answers[image_index]:
            err += 1

    return posterior, predictions, answers, err/len(labels)

def printResult(likelihood, posterior, predictions, answers, err):
    result = ""

    for image_index in range(len(predections)):
        result += "Posterior (in log scale):\n"
        for label in range(10):
            result += f"{label}: {posterior[image_index][label]/np.sum(posterior[image_index])}\n"
        result += f"Prediction: {predictions[image_index]}, Ans: {answers[image_index]}\n\n"

    # Print Bayesian classifier
    result += "Imagination of numbers in Bayesian classifier:"
    for label in range(10):
        result += f"\n{label}:\n"
        for i in range(28):
            for j in range(28):
                if mode == 0:
                    classifier_value = np.argmax(likelihood[label][i][j])
                    result += f"{int(classifier_value/16)} "
                elif mode == 1:
                    classifier_value = likelihood[label][i][j][0]   # The MLE of likelihood is mean
                    result += f"{int(classifier_value/128)} "
            result += "\n\n"

    result += f"Error rate: {err}"

    return result

if __name__ == "__main__":
    mode = int(input("Toggle option is (0 or 1): "))

    # loading
    print("Loading...")
    trainingLabel, trainingData, testLabel, testData = Load()

    # training
    print("Training...")
    if mode == 0:
        prior, likelihood = Discrete(trainingLabel, trainingData)
    elif mode == 1:
        prior, likelihood = Continue(trainingLabel, trainingData)

    # testing
    print("Testing...")
    posterior, predections, answers, err = test(mode, testLabel, testData)

    resultFile = open("result.txt", 'w')
    resultFile.write(printResult(likelihood, posterior, predections, answers, err))