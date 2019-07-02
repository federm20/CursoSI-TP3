from GPyOpt.methods import BayesianOptimization
import numpy as np
import sklearn.datasets as skdataset
from sklearn.model_selection import KFold
import tensorflow as tf
from matplotlib import pyplot


# definicion del modelo de red neuronal
def getModel(params):
    # crea el modelo a usar en Keras (capas secuenciales sin recurrencias)
    model = tf.keras.Sequential()

    # arquitectura ConvNet
    model.add(tf.keras.layers.Conv2D(params[0], (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(params[1], (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # input como vector de 784 valores
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

    # Define las capas con la cantidad de neuronas de cada una y su funcion de activacion
    model.add(tf.keras.layers.Dense(params[2], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(params[3], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(params[4], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(params[5], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # probabilidad de cada clase
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    # Compila el modelo definiendo optimizador, funcion objetivo y metrica a evaluar
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Define la funcion a optimizar, en este caso la red ConvNet
def CNN(params):
    # casteo de cada variable a entero para que no arroje error la red neuronal
    values = [int(param) for param in params[0]]

    # implementa la validacion cruzada mediante KFold
    kf = KFold(n_splits=5, shuffle=False)

    # guarda la performance de cada KFold para luego promediar
    KF_history = []

    for train_index, test_index in kf.split(bunch_dataset["image_tensor"]):
        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = np.array(bunch_dataset["image_tensor"])[train_index], \
                          np.array(bunch_dataset["image_tensor"])[test_index]
        y_train, y_test = bunch_dataset["target"][train_index], bunch_dataset["target"][test_index]

        model = getModel([64, 32] + values)

        # Entrena el modelo
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), epochs=200)

        if np.max(history.history['val_acc']) < 0.7:
            return np.max(history.history['val_acc'])

        KF_history.append(history)

        # grafica el valor de precision durante el entrenamiento y la validacion de cada epoca
        # pyplot.plot(history.history['acc'], label='train')
        # pyplot.plot(history.history['val_acc'], label='test-shapes')
        # pyplot.legend()
        # pyplot.show()

    # retorna como medida de rendimiento, el promedio de la mejor validacion de cada Fold
    return np.average([np.max(history.history['val_acc']) for history in KF_history])


# obtiene las imagenes y su clasificacion
bunch_dataset = skdataset.load_files("shapes/")

# inicializa array de tensores
bunch_dataset["image_tensor"] = []

# realiza una decodificacion de la imagen en PNG a un tensor normalizado
with tf.Session() as sess:
    for image_png in bunch_dataset['data']:
        bunch_dataset["image_tensor"].append(sess.run(tf.image.decode_png(image_png, channels=1)) / 255.0)
        print('Loading image ', len(bunch_dataset["image_tensor"]))

# define hyperparametros a optimizar
bds = [
    # {'name': 'conv1', 'type': 'discrete', 'domain': (30, 60)},
    # {'name': 'conv2', 'type': 'discrete', 'domain': (30, 60)},
    {'name': 'nn1', 'type': 'discrete', 'domain': (100, 1000)},
    {'name': 'nn2', 'type': 'discrete', 'domain': (100, 1000)},
    {'name': 'nn3', 'type': 'discrete', 'domain': (100, 1000)},
    {'name': 'nn4', 'type': 'discrete', 'domain': (100, 1000)}
]

# define el optimizador
optimizer = BayesianOptimization(f=CNN,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type='EI',
                                 acquisition_jitter=0.05,
                                 maximize=True)

# realiza las 20 iteraciones de la optimizacion
optimizer.run_optimization(max_iter=20)

y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

print(y_bo)

pyplot.plot(y_bo, 'bo-', label='Bayesian optimization')
pyplot.xlabel('Iteration')
pyplot.ylabel('Performance')
# pyplot.ylim(-5000, -3000)
pyplot.title('Value of the best sampled CV score')
pyplot.legend()
pyplot.show()
