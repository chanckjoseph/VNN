import os
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
import pprint

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import contextlib
import datetime

csv_path = 'csv/input_crime.csv'
save_path = "saves/crimes/save_crime.h5"
save_best_path = "saves/crimes/best_model.h5"
log_path = "log/crimes/training_log_" + datetime.datetime.now().strftime("%Y%m%d%H%M") + ".csv"
stdout_path = "stdout/stdout_" + datetime.datetime.now().strftime("%Y%m%d%H%M") + ".txt"

base_year = 2000
base_x = 388373
base_y = 5435925

learning_rate = 0.9
dropout_rate = 0.4
# model_loss=[tf.keras.losses.categorical_crossentropy, tf.keras.losses.mean_squared_logarithmic_error]
model_loss = tf.keras.losses.mean_absolute_percentage_error
model_optimizer = tf.keras.optimizers.Adam()
# model_optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-9, momentum=0.6,nesterov=True)
# model_optimizer=tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

'''Google Colab TPU Setup Code Begins********************************************************************'''
'''
from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)
home_dir = "/content/gdrive/My Drive/Colab Notebooks/"
csv_path = home_dir + csv_path
save_path = home_dir + save_path
save_best_path = home_dir + save_best_path
log_path = home_dir + log_path
stdout_path = home_dir + stdout_path


if 'COLAB_TPU_ADDR' not in os.environ:
    print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
else:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    print('TPU address is', tpu_address)

# This address identifies the TPU we'll use when configuring TensorFlow.
TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
'''

'''Google Colab TPU Setup Code Ends********************************************************************'''

'''columns'''
'''TYPE,YEAR,MONTH,DAY,HOUR,MINUTE,NEIGHBOURHOOD,X,Y'''

'''Break and Enter Commercial	1
Break and Enter Residential/Other	2
Mischief	3
Other Theft	4
Theft from Vehicle	5
Theft of Bicycle	6
Theft of Vehicle	7
Vehicle Collision or Pedestrian Struck (with Fatality)	8
Vehicle Collision or Pedestrian Struck (with Injury)	9
'''
'''Arbutus Ridge	1
Central Business District	2
Dunbar-Southlands	3
Fairview	4
Grandview-Woodland	5
Hastings-Sunrise	6
Kensington-Cedar Cottage	7
Kerrisdale	8
Killarney	9
Kitsilano	10
Marpole	11
Mount Pleasant	12
Musqueam	13
Oakridge	14
Renfrew-Collingwood	15
Riley Park	16
Shaughnessy	17
South Cambie	18
Stanley Park	19
Strathcona	20
Sunset	21
Victoria-Fraserview	22
West End	23
West Point Grey	24
'''

model = None
batch_size = 300000

weight_scale = 0.75


def prep_data(batch_number=0):
    data = np.genfromtxt(csv_path, delimiter=",", filling_values=0)

    # Remove Type zeros
    data = np.delete(data, np.where(data[:, 0] == 0)[0], axis=0)

    # Remove NEIGHBOURHOOD zeros
    data = np.delete(data, np.where(data[:, 6] == 0)[0], axis=0)

    type_data = tf.keras.utils.to_categorical(data[:, 0])
    neighbourhood_data = tf.keras.utils.to_categorical(data[:, 6])

    input_data01 = type_data
    input_data02 = data[:, 1:6]
    output_data_01 = neighbourhood_data
    output_data_02 = data[:, 7:10]

    idx = np.random.randint(len(data), size=batch_number)

    return input_data01[idx, :], input_data02[idx, :], output_data_01[idx, :], output_data_02[idx, :]


def customLoss():
    def relative_loss(y_true, y_pred):
        return tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)

    return relative_loss


def build_model():
    inputA00 = tf.keras.Input(shape=[10, ])
    denseA01 = tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        inputA00)
    dropoutA01 = tf.keras.layers.Dropout(rate=dropout_rate)(denseA01)
    denseA02 = tf.keras.layers.Dense(units=32, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        dropoutA01)
    dropoutA02 = tf.keras.layers.Dropout(rate=dropout_rate)(denseA02)
    denseA03 = tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        dropoutA02)
    dropoutA03 = tf.keras.layers.Dropout(rate=dropout_rate)(denseA03)

    inputB00 = tf.keras.Input(shape=[5, ])
    denseB01 = tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        inputB00)
    dropoutB01 = tf.keras.layers.Dropout(rate=dropout_rate)(denseB01)
    denseB02 = tf.keras.layers.Dense(units=32, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        dropoutB01)
    dropoutB02 = tf.keras.layers.Dropout(rate=dropout_rate)(denseB02)
    denseB03 = tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer='random_uniform', trainable=True)(
        dropoutB02)
    dropoutB03 = tf.keras.layers.Dropout(rate=dropout_rate)(denseB03)

    merge01 = tf.keras.layers.concatenate([dropoutA03, dropoutB03], axis=-1)
    normal01 = tf.keras.layers.BatchNormalization()(merge01)

    denseMain01 = tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(normal01)
    dropoutMain01 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain01)
    denseMain02 = tf.keras.layers.Dense(units=512, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain01)
    dropoutMain02 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain02)
    denseMain03 = tf.keras.layers.Dense(units=1024, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain02)
    dropoutMain03 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain03)
    '''denseMain04 = tf.keras.layers.Dense(units=1024, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain03)
    dropoutMain04 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain04)
    denseMain05 = tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain04)
    dropoutMain05 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain05)
    denseMain06 = tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain05)
    dropoutMain06 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain06)
    denseMain07 = tf.keras.layers.Dense(units=512, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain06)
    dropoutMain07 = tf.keras.layers.Dropout(rate=dropout_rate)(denseMain07)'''

    '''denseOutB01 = tf.keras.layers.Dense(units=48, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain07)
    dropoutOutB01 = tf.keras.layers.Dropout(rate=dropout_rate)(denseOutB01)
    denseOutB02 = tf.keras.layers.Dense(units=24, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutOutB01)
    dropoutOutB02 = tf.keras.layers.Dropout(rate=dropout_rate)(denseOutB02)'''
    denseOutB03 = tf.keras.layers.Dense(units=2, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain03)

    '''denseOutA01 = tf.keras.layers.Dense(units=8, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(denseOutB03)
    dropoutOutA01 = tf.keras.layers.Dropout(rate=dropout_rate)(denseOutA01)
    denseOutA02 = tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer='random_uniform',
                                        trainable=True)(dropoutOutA01)
    dropoutOutA02 = tf.keras.layers.Dropout(rate=dropout_rate)(denseOutA02)'''
    denseOutA03 = tf.keras.layers.Dense(units=25, activation=tf.nn.softmax, kernel_initializer='random_uniform',
                                        trainable=True)(dropoutMain02)

    # model = tf.keras.Model(inputs=[inputA00,inputB00], outputs=[denseOutA03,denseOutB03])
    # model.compile(loss=model_loss, loss_weights=[weight_scale,1-weight_scale], optimizer=model_optimizer, metrics=['acc'])

    model = tf.keras.Model(inputs=[inputA00, inputB00], outputs=denseOutB03)
    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=['acc'])

    model.summary()
    return model


def retrain(epochs=5, model=None, input_data_01=None, input_data_02=None, output_data_01=None, output_data_02=None):
    print("retrain model")
    print(model)

    in01 = input("Redirect training output? ")

    if input_data_01 is None or input_data_02 is None or output_data_01 is None or output_data_01 is None:
        raise Exception('ReTrain Failed - Training Data no available')
    if model is None:
        model = build_model()
        model.load_weights(save_path)
    history = None

    "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000),
                 tf.keras.callbacks.ModelCheckpoint(filepath=save_best_path, monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.CSVLogger(log_path)]
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1e-3, patience=50, verbose=1,
    #                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)]

    if epochs == -1:
        history = model.fit_generator(generator=prep_data(25), max_queue_size=4, epochs=1, steps_per_epoch=4, workers=1,
                                      validation_split=0.1)
    else:
        if in01 == "y":
            with open(home_dir + "stdout/stdout_" + datetime.datetime.now().strftime("%Y%m%d%H%M") + ".txt", 'w') as f:
                with contextlib.redirect_stdout(f):
                    # history= model.fit([input_data_01, input_data_02], [output_data_01,output_data_02], epochs=epochs, batch_size=batch_size, use_multiprocessing=True, validation_split=0.2, callbacks=callbacks)
                    history = model.fit([input_data_01, input_data_02], output_data_02, epochs=epochs,
                                        batch_size=batch_size, use_multiprocessing=True, validation_split=0.2,
                                        callbacks=callbacks)
        else:
            # history= model.fit([input_data_01, input_data_02], [output_data_01,output_data_02], epochs=epochs, batch_size=batch_size, use_multiprocessing=True, validation_split=0.2, callbacks=callbacks)
            history = model.fit([input_data_01, input_data_02], output_data_02, epochs=epochs, batch_size=batch_size,
                                use_multiprocessing=True, validation_split=0.2, callbacks=callbacks)

    print(history.history.keys())
    print(history.history)
    print(history.history["val_loss"][len(history.history['val_loss']) - 1])
    print("Min val_loss")
    print(min(history.history["val_loss"]))


def predict():
    model = build_model()
    model.load_weights(save_path)
    input01 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    input02 = np.array([[3, 5, 17, 18, 0]])
    print("input01")
    print(input01.shape)
    print(input01[0])
    print("input02")
    print(input02.shape)
    print(input02[0])

    # output_01, output_02 = model.predict([input01, input02])
    output_02 = model.predict([input01, input02])
    # print("output_01")
    # print(output_01.shape)
    # print(output_01[0])
    print("output_02")
    print(output_02.shape)
    print(output_02[0])


def predict2():
    model = build_model()
    model.load_weights(save_path)
    input01 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    input02 = np.array([[3, 12, 18, 20, 0]])
    print("input01")
    print(input01.shape)
    print(input01[0])
    print("input02")
    print(input02.shape)
    print(input02[0])

    # output_01, output_02 = model.predict([input01, input02])
    output_02 = model.predict([input01, input02])
    # print("output_01")
    # print(output_01.shape)
    # print(output_01[0])
    print("output_02")
    print(output_02.shape)
    print(output_02[0])


while True:
    print("Crime predict")
    in01 = input("Please any key to continue..")
    if in01 == "exit" or in01 == "quit":
        break
    if in01 == "p" or in01 == "predict":
        predict()
        predict2()
    if in01 == "i" or in01 == "infinite":
        while True:
            input_data_01, input_data_02, output_data_01, output_data_02 = prep_data(batch_size)
            retrain(100000, input_data_01=input_data_01, input_data_02=input_data_02, output_data_01=output_data_01,
                    output_data_02=output_data_02)
    if in01 == "e" or in01 == "early":
        input_data_01, input_data_02, output_data_01, output_data_02 = prep_data(batch_size)
        retrain(100000, model, input_data_01=input_data_01, input_data_02=input_data_02, output_data_01=output_data_01,
                output_data_02=output_data_02)
    # only worthwhile if running on gpu because it takes cpu resources and io
    if in01 == "g":
        while True:
            retrain(-1, model, input_data_01=input_data_01, input_data_02=input_data_02, output_data_01=output_data_01,
                    output_data_02=output_data_02)
    if in01 == "r" or in01 == "rebuild":
        model = build_model()
        tf.keras.models.save_model(
            model,
            save_path,
            overwrite=True,
            include_optimizer=True
        )
    if in01 == "l" or in01 == "load":
        model = tf.keras.models.load_model(save_path)
        model.summary()

