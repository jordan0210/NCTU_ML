import Gen

if __name__ == "__main__":
    mean_gen = float(input("Input mean: "))
    var_gen = float(input("Input variance: "))

    print(f"Data point source function: N({mean_gen}, {var_gen})\n")

    mean = Gen.Gaussian(mean_gen, var_gen)
    var = 0
    N = 1
    err_mean = 100
    err_var = 100

    while ((abs(err_mean) > 1e-4) | (abs(err_var) > 1e-4)):
        newPoint = Gen.Gaussian(mean_gen, var_gen)
        N += 1
        print("Add data point:", newPoint)
        err_mean = (newPoint - mean) / N
        err_var = ((newPoint - mean) ** 2) / N - var / (N - 1)
        mean += err_mean
        var += err_var
        print(f"Mean = {mean} Variance = {var}")

    print("\nN =", N)
    print("err_mean =", err_mean)
    print("err_var =", err_var)