from load import *

from libsvm.svmutil import *

if __name__ == "__main__":
    linear = '0'
    polynomial = '1'
    RBF = '2'

    # load data
    X_train = load_data("./data/X_train.csv")
    Y_train = load_label("./data/Y_train.csv")
    X_test = load_data("./data/X_test.csv")
    Y_test = load_label("./data/Y_test.csv")

    # training
    train = svm_problem(Y_train, X_train)
    linear_param = svm_parameter('-q -t ' + linear)
    polynomail_param = svm_parameter('-q -t ' + polynomial)
    RBF_param = svm_parameter('-q -t ' + RBF)
    linear_model = svm_train(train, linear_param)
    polynomial_model = svm_train(train, polynomail_param)
    RBF_model = svm_train(train, RBF_param)

    # testing
    print("linear kernel:")
    svm_predict(Y_test, X_test, linear_model)
    print("\npolynomial kernel:")
    svm_predict(Y_test, X_test, polynomial_model)
    print("\nRBF kernel:")
    svm_predict(Y_test, X_test, RBF_model)
