import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
from PIL import Image

import nextFrameModel


frame_x= np.random.randint(nextFrameModel.last_frames-nextFrameModel.number_of_frames-1, size=nextFrameModel.number_of_batches+1)
print(str(frame_x[0]).zfill(4))



# Load Images
big_array02 = []
big_array03 = []
big_array04 = []
big_array05 = []
big_array06 = []
big_array07 = []

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

dataR = np.array(inFramesR)
dataG = np.array(inFramesG)
dataB = np.array(inFramesB)

#data02 = np.array(lastFrame)

#Prepare test set
big_array = []
for i in range(nextFrameModel.number_of_frames):
    big_array.append(np.array(Image.open("video/frames/frame-"+
                                         str(frame_x[nextFrameModel.number_of_batches]+i).zfill(4)+".bmp")))
big_array05.append(big_array)
#big_array06.append([np.array(Image.open("video/frames/frame-"+str(frame_x[nextFrameModel.number_of_batches] +
#                                                                  nextFrameModel.number_of_frames-1).zfill(4)+".bmp"))])
big_array07.append(
    [np.array(Image.open("video/frames/frame-" + str(frame_x[nextFrameModel.number_of_batches]
                                                     + nextFrameModel.number_of_frames).zfill(4) + ".bmp"))])
inFramesTest = np.array(big_array05)
outFramesTest = np.array(big_array07)
#lastFrameTest = np.array(big_array06)

inFramesTestR = inFramesTest[:, :, :, :, [0]]
inFramesTestG = inFramesTest[:, :, :, :, [1]]
inFramesTestB = inFramesTest[:, :, :, :, [2]]

dataTestR = np.array(inFramesTestR)
dataTestG = np.array(inFramesTestG)
dataTestB = np.array(inFramesTestB)
#data02Test = np.array(lastFrameTest)

#model = load_model("saves/savetest.h5")

#model.fit([dataR,dataG,dataB,data02], outFrames, epochs=1, batch_size=1)

#labelRGB=model.predict([[dataR[0]],[dataG[0]],[dataB[0]],[data02[0]]])
#label=np.array(labelRGB, dtype=np.uint8)

#print(label.dtype)
#print(label.shape)

#img = Image.fromarray(label[0][0])

#img.show()
#img.save('out2.png')


def reTrain():
    # model = tf.keras.models.load_model("saves/savetest.h5")
    # Custom loss function do not get save with model.
    # Call class function with dummy parameters for model reconstruction

    model = nextFrameModel.compile()
    model.load_weights("saves/savetest.h5")
    #model.fit([dataR, dataG, dataB, data02], outFrames, epochs=5, batch_size=25)
    model.fit([dataR, dataG, dataB], outFrames, epochs=5, batch_size=25)

    tf.keras.models.save_model(
        model,
        "saves/savetest.h5",
        overwrite=True,
        include_optimizer=True
    )

def predict():
    model = nextFrameModel.compile()
    model.load_weights("saves/savetest.h5")
    #labelRGB = model.predict([[dataTestR[0]], [dataTestG[0]], [dataTestB[0]], [data02Test[0]]])
    labelRGB = model.predict([[dataTestR[0]], [dataTestG[0]], [dataTestB[0]]])
    label = np.array(labelRGB, dtype=np.uint8)
    img = Image.fromarray(label[0][0])
    img.show()
    img.save('y_pred.png')

reTrain()

predict()

print(str(frame_x[0]).zfill(4))

print(str(frame_x[nextFrameModel.number_of_batches]).zfill(4))

img = Image.fromarray(outFramesTest[0][0])
img.show()
img.save('y_true.png')

