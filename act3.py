import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from GPyOpt.methods import BayesianOptimization


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
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = np.array(iris.target, dtype=int)

    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    kernel = 1.0 * kernels.RBF([1.0])
    gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

    Z = gpc_rbf_isotropic.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def activity_3_4():
    moons = datasets.make_moons(n_samples=1000, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.2)

    # define el kernel de base radial, la clasificador con GP, entrena y predice
    def custom_RBF(params):
        print(params)
        kernel = params[0][0] ** 2 * kernels.RBF(length_scale=params[0][1])
        gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        y_pred = gpc_rbf_isotropic.predict(X_test)
        return accuracy_score(y_test, y_pred)

    bds = [
        {'name': 'p', 'type': 'continuous', 'domain': (0.1, 1000)},
        {'name': 'ls', 'type': 'continuous', 'domain': (0.01, 10)}
    ]

    # define el optimizador
    optimizer = BayesianOptimization(f=custom_RBF,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     verbosity=True,
                                     maximize=True)

    # realiza las 20 iteraciones de la optimizacion
    optimizer.run_optimization(max_iter=5)

    print(optimizer.Y)

    plt.contour(optimizer.X[:, 0], optimizer.X[:, 0], optimizer.Y)


# activity_3_1()
# activity_3_2()
# activity_3_3()
activity_3_4()
