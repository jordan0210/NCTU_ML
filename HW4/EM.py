import numpy as np
from numba import njit

def Load():
    label_file = open("train-labels.idx1-ubyte", "rb")
    image_file = open("train-images.idx3-ubyte", "rb")

    label_file.read(8)
    image_file.read(4)
    number = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    column = int.from_bytes(image_file.read(4), byteorder='big')

    label = np.zeros(number, dtype=int)
    data = np.zeros((number, row * column), dtype=int)

    for i in range(number):
        label[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(column):
                data[i][column*j + k] = image_file.read(1)[0]/128

    label_file.close()
    image_file.close()

    return number, label, data.astype(float)

def EM(N, data):
    C, P = init() # C: chance to choise 0~9, P_(i,pixel): the probability for each pixel in 0~9
    result = ""
    dC = 100
    dP = 100
    count = 0
    while (dC >= 0.01) & (dP >= 0.01):
        count += 1
        w = E_step(N, C, P, data)
        C_new, P_new = M_step(N, C, P, w, data)
        dC = np.linalg.norm(C - C_new)
        dP = np.linalg.norm(P - P_new)
        C = C_new
        P = P_new
        for i in range(10):
            result += SetResult(P, i, False)
        result += f"No. of Iteration: {count}, Difference: {dC+dP:<15.10f}\n\n"
        result += "------------------------------------------------------------\n\n"
        if count == 20:
            break

    return C, P, result, count

@njit()
def init():
    C = np.full((10),0.1)
    P = np.random.rand(10, 784)
    return C, P

@njit()
def E_step(N, C, P, data):
    w = np.zeros((N, 10))
    for n in range(N):
        for i in range(10):
            w[n][i] = np.log(C[i]) + np.sum(data[n] * np.log(P[i]+1e-10)) + np.sum((1 - data[n]) * np.log(1-P[i]+1e-10))
        w[n] = np.exp(w[n] - max(w[n]))
        w[n] = w[n] / np.sum(w[n])
    return w

# @njit()
def M_step(N, C, P, w, data):
    C_new = np.zeros(10)
    for i in range(10):
        C_new[i] = np.sum(w.T[i]) / N
    P_new = w.T @ data
    for i in range(10):
        P_new[i] /= np.sum(w.T[i])

    return C_new, P_new

def SetResult(P, i, labeled):
    result = ""
    if labeled:
        result += "labeled "
    result += f"class {i}:\n"
    for row in range(28):
        for column in range(28):
            if (P[i][row*28+column] >= 0.5):
                result += "1 "
            else:
                result += "0 "
        result += "\n"
    result += "\n"
    return result

def SetCMResult(confusion_matrix, count, N):
    result = ""
    currect = 0
    for i in range(10):
        currect += confusion_matrix[i][0][0]
        result += f"Confusion Matrix {i}:\n"
        result += f"                Predict number {i} Predict not number {i}\n"
        result += "{0:<15} {1:^16} {2:^20}\n".format(f"Is number {i}", f"{confusion_matrix[i][0][0]}", f"{confusion_matrix[i][0][1]}")
        result += "{0:<15} {1:^16} {2:^20}\n".format(f"Isn't number {i}", f"{confusion_matrix[i][1][0]}", f"{confusion_matrix[i][1][1]}")
        result += "\n"
        result += "Sensitivity (Successfully predict number {0}):     {1}\n".format(f"{i}", f"{confusion_matrix[i][0][0]/sum(confusion_matrix[i][0])}")
        result += "Sensitivity (Successfully predict not number {0}): {1}\n".format(f"{i}", f"{confusion_matrix[i][1][1]/sum(confusion_matrix[i][1])}")
        result += "------------------------------------------------------------\n\n"

    result += f"Total iteration to converge: {count}\n"
    result += f"Total error rate: {1 - currect/N}"
    return result

def mappingLabel(predict, label):
    table = np.full((10), -1)
    index = 0
    while np.any(table < 0):
        if (table[label[index]] == -1) & (inarray(table, predict[index]) == -1):
            table[label[index]] = predict[index]
        index += 1
        if (index >= 60000):
            for i in range(10):
                if (inarray(table, i) == -1):
                    table[inarray(table, -1)] = i
            break
    return table

def inarray(table, value):
    for i in range(len(table)):
        if (value == table[i]):
            return i
    return -1

def Test(N, C, P, data, label, count):
    w = E_step(N, C, P, data)
    predict = np.zeros(N, dtype=int)
    for i in range(N):
        predict[i] = np.argmax(w[i])
    mappingTable = mappingLabel(predict, label)

    result = ""
    for i in range(10):
        result += SetResult(P, mappingTable[i], True)

    confusion_matrix = np.zeros((10, 2, 2), dtype=int)
    for n in range(N):
        T = inarray(mappingTable, predict[n])
        if (label[n] == T):
            confusion_matrix[label[n]][0][0] += 1
            for i in range(10):
                if (i != label[n]):
                    confusion_matrix[i][1][1] += 1
        if (label[n] != T):
            confusion_matrix[label[n]][0][1] += 1
            for i in range(10):
                if (i != label[n]) & (T == i):
                    confusion_matrix[i][1][0] += 1
                elif (i != label[n]) & (T != i):
                    confusion_matrix[i][1][1] += 1

    result += SetCMResult(confusion_matrix, count, N)
    return result

if __name__ == "__main__":
    # loading
    print("Loading...")
    N, label, data = Load()

    # EM algorithm
    print("EM...")
    C, P, result, count= EM(N, data)

    # Test
    print("Testing...")
    result += Test(N, C, P, data, label, count)

    resultFile = open(f"result.txt", 'w')
    resultFile.write(result)
