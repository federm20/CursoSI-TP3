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
    print("Inicia entrenamiento")

    # casteo de cada variable a entero para que no arroje error la red neuronal
    values = [int(param) for param in params[0]]

    # implementa la validacion cruzada mediante KFold
    kf = KFold(n_splits=4, shuffle=False)

    # guarda la performance de cada KFold para luego promediar
    KF_history = []

    for train_index, test_index in kf.split(bunch_dataset["image_tensor"]):
        X_train, X_test = np.array(bunch_dataset["image_tensor"])[train_index], \
                          np.array(bunch_dataset["image_tensor"])[test_index]
        y_train, y_test = bunch_dataset["target"][train_index], bunch_dataset["target"][test_index]

        # obtiene el modelo de red
        model = getModel(values)

        # Entrena el modelo
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), epochs=40,
                            verbose=0)

        if np.max(history.history['val_acc']) < 0.7:
            performance = np.max(history.history['val_acc'])
            print(params, performance)
            return performance

        KF_history.append(history)

    # retorna como medida de rendimiento, el promedio de la mejor validacion de cada Fold
    performance = np.average([np.max(history.history['val_acc']) for history in KF_history])
    print(params, performance)
    return performance


# obtiene las imagenes y su clasificacion
bunch_dataset = skdataset.load_files("shapes/")

# inicializa array de tensores
bunch_dataset["image_tensor"] = []

# realiza una decodificacion de la imagen en PNG a un tensor normalizado
with tf.Session() as sess:
    for image_png in bunch_dataset['data']:
        bunch_dataset["image_tensor"].append(sess.run(tf.image.decode_png(image_png, channels=1)) / 255.0)
        print('Loading image ', len(bunch_dataset["image_tensor"]))

# define hyperparametros a optimizar con sus fronteras
bds = [
    {'name': 'conv1', 'type': 'discrete', 'domain': np.arange(10, 100, 1)},
    {'name': 'conv2', 'type': 'discrete', 'domain': np.arange(10, 100, 1)},
    {'name': 'nn1', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'nn2', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'nn3', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'nn4', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)}
]

# prueba dos diferentes funciones de adquisicion
for adq in ['EI', 'MPI']:

    print('Optimizador con: ' + adq)

    # define el optimizador
    optimizer = BayesianOptimization(f=CNN,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type=adq,
                                     acquisition_jitter=0.05,
                                     verbosity=True,
                                     maximize=True)

    # realiza las 20 iteraciones de la optimizacion
    optimizer.run_optimization(max_iter=20)

    # obtiene los resultados y grafica
    y_bo = np.maximum.accumulate(-optimizer.Y).ravel()
    pyplot.plot(y_bo, 'bo-', label='Optimización Bayesiana')
    pyplot.xlabel('Iteración')
    pyplot.ylabel('Rendimiento')
    pyplot.title('20 iteraciones')
    pyplot.legend()
    pyplot.show()
