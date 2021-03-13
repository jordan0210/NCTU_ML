import LSE
import Newton

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

LSE_parameters = LSE.doLSE(N, Lambda, Xs, Ys)[0]
Newton_parameters = Newton.doNewtonsMethod(N, Xs, Ys)[0]

x = numpy.linspace(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1, 1000)
y_LSE = [0]*len(x)
y_Newton = [0]*len(x)
for i in range(N):
    y_LSE += LSE_parameters[i] * x ** (N-1-i)
    y_Newton += Newton_parameters[i] * x ** (N-1-i)

# Show Fitting line and Total error
print("LSE:")
line = "Fitting line: "
for i in range(N):
    if (i != 0) & (LSE_parameters[i] >= 0):
        line += "+ "
    line += str(LSE_parameters[i]) + (("X^" + str(N-1-i) + " ") if i != N-1 else "")
print(line)
error = 0
for i in range(len(Xs)):
    y_err = 0
    for j in range(N):
        y_err = y_err + LSE_parameters[j] * Xs[i]**(N-1-j)
    error += (y_err - Ys[i])**2
print("Total error:", error, "\n")

print("Newton's Method:")
line = "Fitting line: "
for i in range(N):
    if (i != 0) & (Newton_parameters[i] >= 0):
        line += "+ "
    line += str(Newton_parameters[i]) + (("X^" + str(N-1-i) + " ") if i != N-1 else "")
print(line)
error = 0
for i in range(len(Xs)):
    y_err = 0
    for j in range(N):
        y_err = y_err + Newton_parameters[j] * Xs[i]**(N-1-j)
    error += (y_err - Ys[i])**2
print("Total error:", error)

# Show plot
pyplot.subplot(2, 1, 1)
pyplot.title("LSE: n=" + str(N) + ", lambda=" + str(Lambda))
pyplot.scatter(Xs, Ys, color='red', s=10)
pyplot.xlim(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1)
pyplot.ylim(math.floor(min(Ys)) - 10, math.floor(max(Ys)) + 10)
pyplot.plot(x, y_LSE)
pyplot.subplot(2, 1, 2)
pyplot.title("Newton's method: n=" + str(N))
pyplot.scatter(Xs, Ys, color='red', s=10)
pyplot.xlim(math.floor(min(Xs)) - 1, math.floor(max(Xs)) + 1)
pyplot.ylim(math.floor(min(Ys)) - 10, math.floor(max(Ys)) + 10)
pyplot.plot(x, y_Newton)
pyplot.show()