#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split


def LGR(c):
    # TODO: Load the Iris dataset using sklearn.
    X, y = load_iris(return_X_y=True)
    # Split train/test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=3)
    # TODO: Initialize a logistic regression model for the Iris dataset.
    # Here, you only need to tune the inverse regularization parameter `C`.
    # Please set `random_state` to 3.
    lr = LogisticRegression(random_state=3, C=c, max_iter=1000)
    # Start training.
    lr.fit(X_train, y_train)
    # Print the training error.
    return (lr.score(X_train, y_train), lr.score(X_test, y_test))
    # TODO: Print the testing error


def plot_graph():
    C = []
    for i in [0.001, 0.01, 0.1, 1, 10]:
        for j in range(1, 10):
            C.append(i * j)
    Train_Error = [1 - LGR(i)[0] for i in C]
    Test_Error = [1 - LGR(i)[1] for i in C]
    for i in range(len(C)):
        print(f"C={C[i]:.3f}, Train Error={Train_Error[i]}, Test Error={Test_Error[i]}")
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(C, Test_Error, label='Test')
    ax1.plot(C, Train_Error, label='Train')
    ax1.set_xscale('log')
    plt.legend()
    plt.xlabel('C_values')
    plt.ylabel('Error')
    plt.title('Accuracy of classification under different C values')
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.9, wspace=0.2, hspace=0)
    plt.show()


plot_graph()

# TODO: Load the Wine dataset.
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)

def LGR(c):
# TODO: Load the Iris dataset using sklearn.
    X, y = load_wine(return_X_y=True)
    # Split train/test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)
        # TODO: Initialize a logistic regression model for the Iris dataset.
    # Here, you only need to tune the inverse regularization parameter `C`.
    # Please set `random_state` to 3.
    lr = LogisticRegression(random_state=3,C=c, max_iter=5000)
        # Start training.
    lr.fit(X_train, y_train)
     # Print the training error.
    return(lr.score(X_train, y_train),lr.score(X_test,y_test))
    # TODO: Print the testing error

def plot_graph():
    C=[]
    for i in [0.001,0.01,0.1,1,10]:
        for j in range(1,10):
            C.append(i*j)
    Train_Error=[1-LGR(i)[0] for i in C]
    Test_Error=[1-LGR(i)[1] for i in C]
    for i in range(len(C)):
        print(f"C={C[i]:.3f}, Train Error={Train_Error[i]}, Test Error={Test_Error[i]}")
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1),(0,0))
    ax1.plot(C,Test_Error,label='Test')
    ax1.plot(C,Train_Error, label='Train')
    ax1.set_xscale('log')
    plt.legend()
    plt.xlabel('C_values')
    plt.ylabel('Error')
    plt.title('Accuracy of classification under different C values')
    plt.subplots_adjust(left=0.09,bottom=0.16,right=0.94,top=0.9,wspace=0.2,hspace=0)
    plt.show()
plot_graph()


# Initialize a random forest model using sklearn.
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting.
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!
# Please set `random_state` to 3 and feel free to set the value of `n_estimators`.
def LGR(c):
    # Load the Iris dataset for training a random forest model.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

    rf_d = RandomForestClassifier(criterion='gini', n_estimators=200, random_state=3, n_jobs=2, max_depth=c,
                                  max_samples=50)
    rf_d.fit(X_train, y_train)

    c *= 10
    rf_s = RandomForestClassifier(criterion='gini', n_estimators=200, random_state=3, n_jobs=2, max_depth=5,
                                  max_samples=c)
    rf_s.fit(X_train, y_train)

    return (
    rf_d.score(X_train, y_train), rf_d.score(X_test, y_test), rf_s.score(X_train, y_train), rf_s.score(X_test, y_test))


def plot_graph():
    C = [i for i in range(1, 11)]
    Train_Error_d = []
    Test_Error_d = []
    Train_Error_s = []
    Test_Error_s = []
    for i in C:
        result = LGR(i)
        Train_Error_d.append(1 - result[0])
        Test_Error_d.append(1 - result[1])
        Train_Error_s.append(1 - result[2])
        Test_Error_s.append(1 - result[3])

    for i in range(len(C)):
        print(f"Max Depth={C[i]:.3f}, Train Error Depth={Train_Error_d[i]}, Test Error Depth={Test_Error_d[i]}")
    for i in range(len(C)):
        print(
            f"Max Samples={C[i] * 10:.3f}, Train Error Sample={Train_Error_s[i]}, Test Error Sample={Test_Error_s[i]}")
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(C, Test_Error_d, label='Test')
    ax1.plot(C, Train_Error_d, label='Train')

    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.title('Accuracy of Iris classification under different max depth')
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.9, wspace=0.2, hspace=0)
    plt.show()
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot([10 * c for c in C], Test_Error_s, label='Test')
    ax1.plot([10 * c for c in C], Train_Error_s, label='Train')

    plt.legend()
    plt.xlabel('Max Sample')
    plt.ylabel('Error')
    plt.title('Accuracy of Iris classification under different max sample')
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.9, wspace=0.2, hspace=0)
    plt.show()


plot_graph()


# Initialize a random forest model using sklearn.
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting.
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!
# Please set `random_state` to 3 and feel free to set the value of `n_estimators`.
def LGR(c):
    # Load the Breast Cancer dataset for training a random forest model.
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

    rf_d = RandomForestClassifier(criterion='gini', n_estimators=200, random_state=3, n_jobs=2, max_depth=c,
                                  max_samples=100)
    rf_d.fit(X_train, y_train)

    c *= 10
    rf_s = RandomForestClassifier(criterion='gini', n_estimators=200, random_state=3, n_jobs=2, max_depth=10,
                                  max_samples=c)
    rf_s.fit(X_train, y_train)

    return (
    rf_d.score(X_train, y_train), rf_d.score(X_test, y_test), rf_s.score(X_train, y_train), rf_s.score(X_test, y_test))


def plot_graph():
    C = [i for i in range(1, 11)]
    Train_Error_d = []
    Test_Error_d = []
    Train_Error_s = []
    Test_Error_s = []
    for i in C:
        result = LGR(i)
        Train_Error_d.append(1 - result[0])
        Test_Error_d.append(1 - result[1])
        Train_Error_s.append(1 - result[2])
        Test_Error_s.append(1 - result[3])

    for i in range(len(C)):
        print(f"Max Depth={C[i]:.3f}, Train Error Depth={Train_Error_d[i]}, Test Error Depth={Test_Error_d[i]}")
    for i in range(len(C)):
        print(
            f"Max Samples={C[i] * 10:.3f}, Train Error Sample={Train_Error_s[i]}, Test Error Sample={Test_Error_s[i]}")
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(C, Test_Error_d, label='Test')
    ax1.plot(C, Train_Error_d, label='Train')

    plt.legend()
    plt.xlabel('Max Depth')
    plt.ylabel('Error')
    plt.title('Accuracy of Breast Cancer classification under different max depth')
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.9, wspace=0.2, hspace=0)
    plt.show()
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot([10 * c for c in C], Test_Error_s, label='Test')
    ax1.plot([10 * c for c in C], Train_Error_s, label='Train')

    plt.legend()
    plt.xlabel('Max Sample')
    plt.ylabel('Error')
    plt.title('Accuracy of Breast Cancer classification under different max sample')
    plt.subplots_adjust(left=0.09, bottom=0.16, right=0.94, top=0.9, wspace=0.2, hspace=0)
    plt.show()


plot_graph()