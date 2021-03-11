from matplotlib import pyplot as plt

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

plt.subplot(2, 1, 1)
plt.title("LSE")
plt.scatter(Xs, Ys, color='red', s=10)
plt.subplot(2, 1, 2)
plt.title("Newton's method")
plt.scatter(Xs, Ys, color='red', s=10)
plt.show()