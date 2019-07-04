import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.metrics import accuracy_score


def activity_3_1():
    # obtiene dataset flor de iris
    iris = datasets.load_iris()

    # toma como valores solo el sepalo para simplificar
    X = iris.data[:, :2]
    y = iris.target

    # define la maquina de soporte vectorial y la entrena
    C = 1000.0
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y)

    # grafica la separacion de clases (recta) y el margen
    for index in range(len(clf.coef_)):
        if index != 1:
            w = clf.coef_[index]
            a = -w[0] / w[1]
            xx = np.linspace(3, 8)
            yy = a * xx - (clf.intercept_[index]) / w[1]

            margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
            yy_down = yy - np.sqrt(1 + a ** 2) * margin
            yy_up = yy + np.sqrt(1 + a ** 2) * margin

            # plt.clf()
            plt.plot(xx, yy, 'k-')
            plt.plot(xx, yy_down, 'k--')
            plt.plot(xx, yy_up, 'k--')

    # grafica delimitaciones de clase como contorno
    X0, X1 = X[:, 0], X[:, 1]
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # grafica puntos del dataset (sepalos)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(3.5, 8.3)
    plt.ylim(1.5, 5)
    plt.show()


def activity_3_2():
    # obtiene dataset flor de iris
    iris = datasets.load_iris()

    # toma como valores solo el sepalo para simplificarr
    X = iris.data[:, :2]
    y = iris.target

    # define la maquina de soporte vectorial y la entrena
    clf = svm.NuSVC(nu=0.5, kernel='poly')
    clf.fit(X, y)

    xx, yy = np.meshgrid(np.linspace(0, 8, 1000), np.linspace(0, 8, 1000))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0, 1, 2, 3, 4, 5, 6], colors='black')

    # grafica puntos del dataset (sepalos)
    X0, X1 = X[:, 0], X[:, 1]
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(3.5, 8.3)
    plt.ylim(1.5, 5)
    plt.show()


def activity_3_3():
    X, y = datasets.load_iris(return_X_y=True)
    kernel = 1.0 * kernels.RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    gpc.fit(X, y)
    score = gpc.score(X, y)
    print(score)

    y_pred = gpc.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % ('GPC', accuracy * 100))

    xx = np.linspace(3, 9, 100)
    yy = np.linspace(1, 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    probas = gpc.predict_proba(Xfull)
    imshow_handle = plt.imshow(probas.reshape((100, 100)),
                               extent=(3, 9, 1, 5), origin='lower')

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()

# activity_3_1()
# activity_3_2()
activity_3_3()
# activity_3_4()
