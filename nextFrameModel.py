import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

image_width=72
image_height=72
number_of_frames = 4

number_of_training_video=7
last_frames = [360,487,818,282,535,311,288]
last_frames_X = [687]

number_of_batches = 10000

#val_acc- 0.6777376532554626
#conv01_factor=18

#val_acc-0.7198
#conv01_factor=9

#val_acc-0.72 (1800 epoch)
#conv01_factor=4

#val_acc-
conv01_factor=2
conv02_factor=2
conv03_factor=3
conv04_factor=3

maxP01_factor=2
maxP02_factor=2
maxP03_factor=2
maxP04_factor=2

upsampling01_factor=2

scale01_factor=16
scale02_factor=8
scale03_factor=2

learning_rate=0.1
#expand01_factor=2.5

#conv02_factor=5
#conv03_factor=3
#conv04_factor=3

horizontal_axis=2
vertical_axis=1

dropout_rate=0.5

model_loss=tf.keras.losses.mean_squared_logarithmic_error
model_optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

def getBase(x):
    return x

def customLossCombine(reshape03):
    def nextFrameLoss(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    return nextFrameLoss


def customLossExternal():
    def nextFrameLoss(y_true, y_pred):
        return tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    return nextFrameLoss

def regLayer(x):
    return tf.multiply(x, 255)

def transform_model_20190204(model):
    # based on "best_model (conv_2_layers acc .88).h5"
    # remove upsampling output layer and replace it with a dense layer


    indexed_layer = model.layers[-3].output

    normal01 = tf.keras.layers.BatchNormalization()(indexed_layer)
    dense01 = layers.Dense( units=(image_width* image_height* 3) , activation="relu",
                            kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',
                            use_bias=False, trainable=True)(normal01)
    reshape00=tf.keras.layers.Reshape(target_shape=(1,int(image_width),int(image_height),3))(dense01)

    new_model = tf.keras.Model(model.input, reshape00)

    for i in range(len(new_model.layers)-3):
        new_model.layers[i].trainable = False

    new_model.compile(loss=model_loss, optimizer=model_optimizer, metrics=['acc'])
    new_model.summary()
    return new_model

def compile():
    #R model
    inputR01 = tf.keras.Input(shape=(image_width ,image_height, number_of_frames ))
    convR01=tf.keras.layers.Conv2D(kernel_size=conv01_factor,padding="same",filters=32,strides=conv01_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(inputR01)
    maxR01= tf.keras.layers.MaxPool2D([maxP01_factor,maxP01_factor])(convR01)
    convR02=tf.keras.layers.Conv2D(kernel_size=conv02_factor,padding="same",filters=64,strides=conv02_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(maxR01)
    maxR02= tf.keras.layers.MaxPool2D([maxP02_factor,maxP02_factor])(convR02)
    #convR03=tf.keras.layers.Conv2D(kernel_size=conv03_factor,padding="valid",filters=128,strides=conv03_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxR02)
    #maxR03= tf.keras.layers.MaxPool2D([maxP03_factor,maxP03_factor])(convR03)
    #convR04=tf.keras.layers.Conv2D(kernel_size=conv04_factor,padding="valid",filters=256,strides=1,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxR03)
    #maxR04= tf.keras.layers.MaxPool2D([maxP04_factor,maxP04_factor])(convR04)

    #G model
    inputG01 = tf.keras.Input(shape=(image_width ,image_height, number_of_frames))
    convG01=tf.keras.layers.Conv2D(kernel_size=conv01_factor,padding="same",filters=32,strides=conv01_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(inputG01)
    maxG01= tf.keras.layers.MaxPool2D([maxP01_factor,maxP01_factor])(convG01)
    convG02=tf.keras.layers.Conv2D(kernel_size=conv02_factor,padding="same",filters=64,strides=conv02_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(maxG01)
    maxG02= tf.keras.layers.MaxPool2D([maxP02_factor,maxP02_factor])(convG02)
    #convG03=tf.keras.layers.Conv2D(kernel_size=conv03_factor,padding="valid",filters=128,strides=conv03_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxG02)
    #maxG03= tf.keras.layers.MaxPool2D([maxP03_factor,maxP03_factor])(convG03)
    #convG04=tf.keras.layers.Conv2D(kernel_size=conv04_factor,padding="valid",filters=256,strides=1,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxG03)
    #maxG04= tf.keras.layers.MaxPool2D([maxP04_factor,maxP04_factor])(convG04)

    #B model
    inputB01 = tf.keras.Input(shape=(image_width ,image_height, number_of_frames))
    convB01=tf.keras.layers.Conv2D(kernel_size=conv01_factor,padding="same",filters=32,strides=conv01_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(inputB01)
    maxB01= tf.keras.layers.MaxPool2D([maxP01_factor,maxP01_factor])(convB01)
    convB02=tf.keras.layers.Conv2D(kernel_size=conv02_factor,padding="same",filters=64,strides=conv02_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=True)(maxB01)
    maxB02= tf.keras.layers.MaxPool2D([maxP02_factor,maxP02_factor])(convB02)
    #convB03=tf.keras.layers.Conv2D(kernel_size=conv03_factor,padding="valid",filters=128,strides=conv03_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxB02)
    #maxB03= tf.keras.layers.MaxPool2D([maxP03_factor,maxP03_factor])(convB03)
    #convB04=tf.keras.layers.Conv2D(kernel_size=conv04_factor,padding="valid",filters=256,strides=1,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(maxB03)
    #maxB04= tf.keras.layers.MaxPool2D([maxP04_factor,maxP04_factor])(convB04)

    merge01=tf.keras.layers.concatenate([maxR02,maxG02,maxB02],axis=-1)
    flatten01=tf.keras.layers.Flatten()(merge01)
    normal01=tf.keras.layers.BatchNormalization()(flatten01)

    denseOut00=layers.Dense(image_width*image_height*3/scale01_factor, activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False,trainable=False)(normal01)
    dropout00=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut00)
    #denseOut01=layers.Dense(image_width*image_height*3/scale01_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False)(dropout00)
    #dropout01=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut01)

    #denseOut02=layers.Dense(image_width*image_height*3/scale02_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False)(dropout01)
    #dropout02=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut02)

    #denseOut03=layers.Dense(bottleneck_size*expand01_factor*expand01_factor*expand01_factor,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(), kernel_initializer='random_uniform',use_bias=False)(dropout00)
    #dropout03=tf.keras.layers.Dropout(rate=dropout_rate) (denseOut03)

    output00=layers.Dense(units=(image_width*image_height*3)/upsampling01_factor**2,activation="relu", kernel_constraint=tf.keras.constraints.NonNeg(),kernel_initializer='random_uniform',use_bias=False,trainable=True)(dropout00)
    reshape04=tf.keras.layers.Reshape(target_shape=(1,int(image_width/upsampling01_factor),int(image_height/upsampling01_factor),3))(output00)
    upsampling01=tf.keras.layers.UpSampling3D([1,upsampling01_factor,upsampling01_factor])(reshape04)

    model = tf.keras.Model(inputs=[inputR01,inputG01,inputB01], outputs=upsampling01)

    model.compile(loss=model_loss, optimizer=model_optimizer, metrics=['acc'])

    model.summary()

    return model

#model=compile()

'''
tf.keras.models.save_model(
    model,
    "saves/savetest.h5",
    overwrite=True,
    include_optimizer=True
)
'''