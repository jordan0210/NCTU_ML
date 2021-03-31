import math
import numpy as np
from matplotlib import pyplot

if __name__ == "__main__":
    testFile = open("testcase.txt", 'r')
    line = testFile.readline().replace('\n','')
    print("Input initial parameter:")
    a_prior = int(input("a = "))
    b_prior = int(input("b = "))
    a_posterior = 0
    b_posterior = 0
    caseCount = 1

    Xs = np.linspace(0, 1, 1000)
    f, axs = pyplot.subplots(1, 3, figsize=(9,3))
    axs[0].set_title("prior")
    axs[0].set_xlim(0, 1)
    axs[1].set_title("likelihood")
    axs[1].set_xlim(0, 1)
    axs[2].set_title("posterior")
    axs[2].set_xlim(0, 1)
    pyplot.ion()
    pyplot.show()

    while line:
        f.suptitle(f"Step: {caseCount}")
        count_1 = line.count("1")
        count_0 = line.count("0")
        a_posterior = a_prior + count_1
        b_posterior = b_prior + count_0

        p_MLE = count_1 / (count_1 + count_0)
        likelihood = math.factorial(count_0 + count_1)/(math.factorial(count_0) * math.factorial(count_1))
        likelihood = likelihood * (p_MLE ** count_1) * ((1 - p_MLE) ** count_0)

        print(f"\nCase {caseCount}:", line)
        print("Likelihood:", likelihood)
        print(f"Beta Prior:     a = {a_prior}, b = {b_prior}")
        print(f"Beta posterior: a = {a_posterior}, b = {b_posterior}")

        # Draw plot
        if (a_prior * b_prior) != 0:
            Beta = math.factorial(a_prior + b_prior - 1)/(math.factorial(a_prior - 1) * math.factorial(b_prior - 1))
            Ys = Beta * (Xs ** (a_prior - 1)) * ((1 - Xs) ** (b_prior - 1))
            plot1, = axs[0].plot(Xs, Ys, color='r')

        Beta = math.factorial(count_0 + count_1)/(math.factorial(count_0) * math.factorial(count_1))
        Ys = Beta * (Xs ** (count_1)) * ((1 - Xs) ** (count_0))
        plot2, = axs[1].plot(Xs, Ys, color='b')

        if (a_posterior * b_posterior) != 0:
            Beta = math.factorial(a_posterior + b_posterior - 1)/(math.factorial(a_posterior - 1) * math.factorial(b_posterior - 1))
            Ys = Beta * (Xs ** (a_posterior - 1)) * ((1 - Xs) ** (b_posterior - 1))
            plot3, = axs[2].plot(Xs, Ys, color='r')

        pyplot.draw()

        input()
        if (a_prior * b_prior) != 0:
           plot1.remove()
        plot2.remove()
        if (a_posterior * b_posterior) != 0:
            plot3.remove()
        a_prior = a_posterior
        b_prior = b_posterior
        line = testFile.readline().replace('\n','')
        caseCount += 1