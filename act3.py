import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm, datasets
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from GPyOpt.methods import BayesianOptimization
from scipy.interpolate import griddata


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
    clf.fit(X, y)

    X0, X1 = X[:, 0], X[:, 1]
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # grafica puntos del dataset (sepalos)
    X0, X1 = X[:, 0], X[:, 1]
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(3.5, 8.3)
    plt.ylim(1.5, 5)
    plt.show()


def activity_3_3():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = np.array(iris.target, dtype=int)

    h = .02

    # crea una malla para realizar la grafica
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    kernel = 1.0 * kernels.RBF([1.0])
    gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    Z = gpc_rbf_isotropic.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # coloca el resultado en colores
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Grafica
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def activity_3_4():
    moons = datasets.make_moons(n_samples=1000, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.2)

    plt.scatter(X_train[:,0], X_train[:,1], c=np.array(["r", "g"])[y_train], edgecolors=(0, 0, 0))
    plt.title('Train dataset - Moons generator')
    plt.show()
    plt.scatter(X_test[:,0], X_test[:,1], c=np.array(["r", "g"])[y_test], edgecolors=(0, 0, 0))
    plt.title('Validation dataset - Moons generator')
    plt.show()

    # define el kernel de base radial, la clasificador con GP, entrena y predice
    def custom_RBF(params):
        kernel = params[0][0] ** 2 * kernels.RBF(length_scale=params[0][1])
        gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel, optimizer=None).fit(X_train, y_train)
        y_pred = gpc_rbf_isotropic.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(params, acc)
        return acc

    bds = [
        {'name': 'p', 'type': 'continuous', 'domain': (1, 1000)},
        {'name': 'ls', 'type': 'continuous', 'domain': (1, 10)}
    ]

    # define el optimizador
    optimizer = BayesianOptimization(f=custom_RBF,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     verbosity=True,
                                     maximize=True)

    # realiza las 30 iteraciones de la optimizacion
    optimizer.run_optimization(max_iter=30)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = optimizer.X[:, 0].reshape(len(optimizer.X[:, 0]), 1).reshape(-1)
    yy = optimizer.X[:, 1].reshape(len(optimizer.X[:, 1]), 1).reshape(-1)
    zz = -optimizer.Y.reshape(-1)

    surf = ax.plot_trisurf(xx, yy, zz, cmap='viridis')
    fig.colorbar(surf)
    plt.xlabel('p')
    plt.ylabel('length scale')
    plt.title('RBF Accuracy')
    plt.show()


activity_3_1()
activity_3_2()
activity_3_3()
activity_3_4()
