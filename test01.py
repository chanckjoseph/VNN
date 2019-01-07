import tensorflow as tf

import numpy as np
from PIL import Image

import nextFrameModel


print(tf.VERSION)
print(tf.keras.__version__)
'''
###################################
# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Create a session with the above options specified.
tf.keras.backend.set_session(tf.Session(config=config))
###################################
'''

frame_x= np.random.randint(nextFrameModel.last_frames-nextFrameModel.number_of_frames-1, size=nextFrameModel.number_of_batches)
print(str(frame_x[0]).zfill(4))

# Load Images
big_array02 = []
big_array03 = []
#big_array04 = []
for j in range(nextFrameModel.number_of_batches):
    big_array = []
    for i in range(nextFrameModel.number_of_frames):
        big_array.append(np.array(Image.open("video/frames/frame-"+str(frame_x[j]+i).zfill(4)+".bmp")))
    big_array02.append(big_array)
    big_array03.append([np.array(Image.open("video/frames/frame-"+str(frame_x[j]+nextFrameModel.number_of_frames).zfill(4)+".bmp"))])
    #big_array04.append([np.array(Image.open("video/frames/frame-"+str(frame_x[j]+nextFrameModel.number_of_frames-1).zfill(4)+".bmp"))])

inFrames = np.array(big_array02)
outFrames = np.array(big_array03)
#lastFrame = np.array(big_array04)

inFramesR = inFrames[:, :, :, :, [0]]
inFramesG = inFrames[:, :, :, :, [1]]
inFramesB = inFrames[:, :, :, :, [2]]

print("in image shape-", inFrames.shape)
print("out image shape-", outFrames.shape)
#print("last image shape-", lastFrame.shape)

ad = np.arange(nextFrameModel.image_width*nextFrameModel.image_height)
ad = ad.reshape((nextFrameModel.image_width,nextFrameModel.image_height,1))
print("address shape- ", ad.shape)

# Addressed image
#big_array = []
#for i in range(number_of_frames):
#    big_array.append(np.append(ad, inFrames[i], axis=2))
#ima = np.array(big_array)
#print("addressed image shape- ", ima.shape)

#img = Image.fromarray(im[0],'RGB')
#img.show()
#img.save('out.png')

'''
def getBase(x):
    return x

def customLoss(denseR02, denseG02, denseB02):
    def nextFrameLoss(y_true, y_pred):
        return K.sum(tf.keras.losses.mean_squared_logarithmic_error(denseR02, denseG02) +
                     tf.keras.losses.mean_squared_logarithmic_error(denseG02, denseB02) +
                     tf.keras.losses.mean_squared_logarithmic_error(denseB02, denseR02))  \
               + tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)

        #return tf.keras.backend.std(y_pred -y_true)
    return nextFrameLoss

#R model
inputR01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,),dtype=tf.float32)
baseR01= layers.Lambda(getBase)(inputR01)
avgR01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputR01)
avgR02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgR01)
print ("avgR02 shape==========")
print (avgR02.shape)
reshapeR01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgR02)
denseR01=layers.Dense(128,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(reshapeR01)
denseR02=layers.Dense(8,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(denseR01)
#denseR03=layers.Dense(5,activation="relu")(denseR02)

#G model
inputG01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,))
baseG01= layers.Lambda(getBase)(inputG01)
avgG01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputG01)
avgG02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgG01)
reshapeG01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgG02)
denseG01=layers.Dense(128,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(reshapeG01)
denseG02=layers.Dense(8,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(denseG01)
#denseG03=layers.Dense(5,activation="relu")(denseG02)

#B model
inputB01 = tf.keras.Input(shape=(number_of_frames,720 ,720 , 1,))
baseB01= layers.Lambda(getBase)(inputB01)
avgB01 = layers.AvgPool3D(pool_size=(1,conv01_factor,conv01_factor),padding="same")(inputB01)
avgB02 = layers.AvgPool3D(pool_size=(1,conv02_factor,conv02_factor),padding="same")(avgB01)
reshapeB01 = tf.keras.layers.Reshape(target_shape=(int(number_of_frames*(image_width/(conv01_factor*conv02_factor))*(image_height/(conv01_factor*conv02_factor))),))(avgB02)
denseB01=layers.Dense(128,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(reshapeB01)
denseB02=layers.Dense(8,activation="relu", kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(denseB01)
#denseB03=layers.Dense(5,activation="relu")(denseB02)

merge01=tf.keras.layers.Add()([denseR02,denseG02,denseB02])
dense04 = tf.keras.layers.Dense(image_width*image_width*3,activation="relu",
                                kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0, max_value=1.0), kernel_initializer='random_uniform', bias_initializer = 'ones')(merge01)
print("dense04")
print(dense04.shape)

reshape02 = tf.keras.layers.Reshape(target_shape=(1,image_width,image_height,3))(dense04)
print("reshape02")
print(reshape02.shape)

input02 = tf.keras.Input(shape=(1,image_width,image_height,3,))
merge02 = tf.keras.layers.Multiply()([input02,reshape02])

model = tf.keras.Model(inputs=[inputR01,inputG01,inputB01,input02], outputs=merge02)
#model.compile(loss='mean_absolute_error', optimizer='adam')
#model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model.compile(loss=customLoss(denseR02,denseG02,denseB02), optimizer='adam')
'''

dataR = np.array(inFramesR)
dataG = np.array(inFramesG)
dataB = np.array(inFramesB)

print("dataR shape")
print(dataR.dtype)
print(dataR.shape)

#print("im shape")
#print(inFrames.shape)
#print(np.array([inFrames[9]]).shape)

#data02 = np.array(lastFrame)

#print("data02 shape")
#print(data02.dtype)
#print(data02.shape)

model = nextFrameModel.compile()

#model.fit([dataR,dataG,dataB,data02], outFrames, epochs=1, batch_size=1)
model.fit([dataR,dataG,dataB], outFrames, epochs=1, batch_size=1)

#labelRGB=model.predict([[dataR[0]],[dataG[0]],[dataB[0]],[data02[0]]])
labelRGB=model.predict([[dataR[0]],[dataG[0]],[dataB[0]]])
label=np.array(labelRGB, dtype=np.uint8)

print(label.dtype)
print(label.shape)

img = Image.fromarray(label[0][0])

img.show()
#img.save('out2.png')

tf.keras.models.save_model(
    model,
    "saves/savetest.h5",
    overwrite=True,
    include_optimizer=True
)


# nextFrameModel.model.to_json()

