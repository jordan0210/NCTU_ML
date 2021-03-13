import LSE

from matplotlib import pyplot
import numpy
import math

N = int(input("Enter the number of polynomial bases: "))
Lambda = float(input("Enter lambda for LSE: "))

# Read test file
testFile = "test.txt"
file = open(testFile, 'r')
line = file.readline()
Xs = []
Ys = []
while line:
    x, y = list(map(float, line.split(',')))
    Xs.append(x)
    Ys.append(y)
    line = file.readline()

LSEparameters = LSE.doLSE(N, Lambda, Xs, Ys)
x_LSE = numpy.linspace(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1, 1000)
y_LSE = [0]*len(x_LSE)
for i in range(N):
    y_LSE = y_LSE + LSEparameters[i]*x_LSE**(N-1-i)

# Show Fitting line and Total error
print("LSE:")
line = "Fitting line: "
for i in range(N):
    line += str(LSEparameters[i]) + (("X^" + str(N-1-i) + " ") if i != N-1 else "")
    if (i < N-1) & (LSEparameters[i] >= 0):
        line += "+ "
print(line)
error = 0
for i in range(len(Xs)):
    y_err = 0
    for j in range(N):
        y_err = y_err + LSEparameters[j]*Xs[i]**(N-1-j)
    error += (y_err - Ys[i])**2
print("Total error:", error)

# Show plot
pyplot.subplot(2, 1, 1)
pyplot.title("LSE")
pyplot.scatter(Xs, Ys, color='red', s=10)
pyplot.xlim(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1)
pyplot.ylim(math.floor(min(Ys)) - 10, math.floor(max(Ys)) + 10)
pyplot.plot(x_LSE, y_LSE)
pyplot.subplot(2, 1, 2)
pyplot.title("Newton's method")
pyplot.scatter(Xs, Ys, color='red', s=10)
pyplot.xlim(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1)
pyplot.ylim(math.floor(min(Ys)) - 10, math.floor(max(Ys)) + 10)
pyplot.show()