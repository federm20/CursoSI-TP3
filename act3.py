import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# obtiene dataset flor de iris
iris = datasets.load_iris()

# toma como valores solo el sepalo para simplificar
X = iris.data[:, :2]
y = iris.target

# define la maquina de soporte vectorial y la entrena
C = 1.0
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X, y)

# grafica la separacion de clases y el margen

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(3, 8)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# plot the line, the points, and the nearest vectors to the plane
plt.clf()
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
plt.show()
