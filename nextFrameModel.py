import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

image_width=720
image_height=720
number_of_frames = 10
last_frames = [360,487,818]
last_frames_X = [687]

number_of_batches = 100

conv01_factor=2
conv02_factor=5

horizontal_axis=2
vertical_axis=1

dropout_rate=0.5


def getBase(x):
    return x

def customLossCombine(denseR00,denseG00,denseB00,denseR01,denseG01,denseB01):
    def nextFrameLoss(y_true, y_pred):
        return (128) * K.sum(tf.keras.losses.mean_squared_error(denseR00, denseG00) +
                     tf.keras.losses.mean_squared_error(denseG00, denseB00) +
                     tf.keras.losses.mean_squared_error(denseB00, denseR00))  \
               + (1024) * K.sum(tf.keras.losses.mean_squared_error(denseR01, denseG01) +
                     tf.keras.losses.mean_squared_error(denseG01, denseB01) +
                     tf.keras.losses.mean_squared_error(denseB01, denseR01))  \
               + tf.keras.losses.mean_squared_error(y_true, y_pred)
    return nextFrameLoss

def customLossExternal(denseR, denseG, denseB):
    def nextFrameLoss(y_true, y_pred):
        return tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    return nextFrameLoss



def compile():
    #R model
    inputR01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,),dtype=tf.float32)
    baseR01= layers.Lambda(getBase)(inputR01)
    avgR01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputR01)
    avgR02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgR01)
    print ("avgR02 shape==========")
    print (avgR02.shape)
    reshapeR01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgR02)
    denseR00=layers.Dense(512,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(reshapeR01)
    dropoutR01=layers.Dropout(rate=dropout_rate) (denseR00)
    denseR01=layers.Dense(64,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(dropoutR01)

    #G model
    inputG01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,))
    baseG01= layers.Lambda(getBase)(inputG01)
    avgG01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputG01)
    avgG02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgG01)
    reshapeG01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgG02)
    denseG00=layers.Dense(512,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(reshapeG01)
    dropoutG01=layers.Dropout(rate=dropout_rate) (denseG00)
    denseG01=layers.Dense(64,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(dropoutG01)

    #B model
    inputB01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,))
    baseB01= layers.Lambda(getBase)(inputB01)
    avgB01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputB01)
    avgB02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgB01)
    reshapeB01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgB02)
    denseB00=layers.Dense(512,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(reshapeB01)
    dropoutB01=layers.Dropout(rate=dropout_rate) (denseB00)
    denseB01=layers.Dense(64,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform')(dropoutB01)

    '''
    input02 = tf.keras.Input(shape=(1,image_width,image_height,3,))
    print("input02")
    print(input02.shape)
    base02= layers.Lambda(getBase)(input02)
    avg01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(base02)
    avg02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avg01)
    flatten01 = tf.keras.layers.Flatten()(avg02)
    dense00=layers.Dense(256,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=255), kernel_initializer='random_uniform', bias_initializer = 'ones')(flatten01)
    dense01=layers.Dense(64,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=255), kernel_initializer='random_uniform', bias_initializer = 'ones')(dense00)

    merge01=tf.keras.layers.concatenate([denseR01,denseG01,denseB01,dense01])
    '''

    merge01=tf.keras.layers.concatenate([denseR01,denseG01,denseB01,denseR00,denseG00,denseB00])

    denseOut00=layers.Dense(192,activation="relu", kernel_initializer='random_uniform')(merge01)
    dropout00=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut00)
    denseOut01=layers.Dense(640,activation="relu", kernel_initializer='random_uniform')(dropout00)
    dropout01=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut01)
    denseOut02=layers.Dense(3200,activation="relu", kernel_initializer='random_uniform')(dropout01)
    dropout02=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut02)
    denseOut03=layers.Dense(15552,activation="relu", kernel_initializer='random_uniform')(dropout02)
    reshape03=tf.keras.layers.Reshape(target_shape=(1,int(image_width/(conv01_factor*conv02_factor)),int(image_height/(conv01_factor*conv02_factor)),3))(denseOut03)
    dense05=layers.UpSampling3D(size=(1,conv02_factor,conv02_factor))(reshape03)
    dense06=layers.UpSampling3D(size=(1,conv01_factor,conv01_factor))(dense05)

    print("reshape03")
    print(reshape03.shape)
    print("dense05")
    print(dense05.shape)
    print("dense06")
    print(dense06.shape)

    '''
    dense04 = tf.keras.layers.Dense(image_width*image_height*3,activation="relu",
                                    kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=255), kernel_initializer='random_uniform', bias_initializer = 'ones')(denseOut02)
    print("dense04")
    print(dense04.shape)

    reshape02 = tf.keras.layers.Reshape(target_shape=(1,image_width,image_height,3))(dense04)
    print("reshape02")
    print(reshape02.shape)
    '''

    model = tf.keras.Model(inputs=[inputR01,inputG01,inputB01], outputs=dense06)
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_error', optimizer='adam')
    #model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    model.compile(loss=customLossCombine(denseR00,denseG00,denseB00,denseR01,denseG01,denseB01), optimizer='adam')
    return model
