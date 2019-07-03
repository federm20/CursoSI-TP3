import tensorflow as tf
import numpy as np
import sklearn.datasets as skdataset
from sklearn.model_selection import KFold
from matplotlib import pyplot
from bayes_opt import BayesianOptimization


# definicion del modelo de red neuronal
def getModel(conv1, conv2, nn1, nn2, nn3, nn4):
    # crea el modelo a usar en Keras (capas secuenciales sin recurrencias)
    model = tf.keras.Sequential()

    # arquitectura ConvNet
    model.add(tf.keras.layers.Conv2D(conv1, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(conv2, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # input como vector de 784 valores
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

    # Define las capas con la cantidad de neuronas de cada una y su funcion de activacion
    model.add(tf.keras.layers.Dense(nn1, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(nn2, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(nn3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(nn4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    # probabilidad de cada clase
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    # Compila el modelo definiendo optimizador, funcion objetivo y metrica a evaluar
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Define la funcion a optimizar, en este caso la red ConvNet
# def CNN(conv1, conv2, nn1, nn2, nn3, nn4):
def CNN(nn1, nn2, nn3, nn4):
    conv1, conv2 = 64, 32
    conv1, conv2, nn1, nn2, nn3, nn4 = int(conv1), int(conv2), int(nn1), int(nn2), int(nn3), int(nn4)

    # Implementa la validacion cruzada mediante KFold
    kf = KFold(n_splits=4, shuffle=False)

    KF_history = []

    for train_index, test_index in kf.split(bunch_dataset["image_tensor"]):
        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = np.array(bunch_dataset["image_tensor"])[train_index], np.array(bunch_dataset["image_tensor"])[
            test_index]
        y_train, y_test = bunch_dataset["target"][train_index], bunch_dataset["target"][test_index]

        model = getModel(conv1, conv2, nn1, nn2, nn3, nn4)

        # Entrena el modelo
        history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), epochs=10)

        if np.max(history.history['val_acc']) < 0.5:
            return np.max(history.history['val_acc'])

        KF_history.append(history)

        # grafica el valor de precision durante el entrenamiento y la validacion de cada epoca
        # pyplot.plot(history.history['acc'], label='train')
        # pyplot.plot(history.history['val_acc'], label='test-shapes')
        # pyplot.legend()
        # pyplot.show()

    # retorna como medida de rendimiento, el promedio de la mejor validacion de cada Fold
    return np.average([np.max(history.history['val_acc']) for history in KF_history])


if __name__ == '__main__':

    # obtiene las imagenes y su clasificacion
    bunch_dataset = skdataset.load_files("shapes/")

    # inicializa array de tensores
    bunch_dataset["image_tensor"] = []

    # realiza una decodificacion de la imagen en PNG a un tensor normalizado
    with tf.Session() as sess:
        for image_png in bunch_dataset['data']:
            bunch_dataset["image_tensor"].append(sess.run(tf.image.decode_png(image_png, channels=1)) / 255.0)
            print('Loading image ', len(bunch_dataset["image_tensor"]))

    KAPPA = 5

    params = {
        # 'conv1': (10, 150),
        # 'conv2': (10, 150),
        'nn1': (100, 1500),
        'nn2': (100, 1500),
        'nn3': (100, 1000),
        'nn4': (100, 1000)
    }

    bo = BayesianOptimization(CNN, params)
    gp_params = {'corr': 'cubic'}
    # bo.maximize(init_points=2, n_iter=0)
    bo.maximize(init_points=5, n_iter=0, acq='ucb', kappa=KAPPA)
    bo.maximize(init_points=0, n_iter=20, kappa=KAPPA)
    # bo.maximize(init_points=0, n_iter=10, kappa=KAPPA)
