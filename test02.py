import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
from PIL import Image

import nextFrameModel

previous_video_x=None
frame_x= None
dataR= None
dataG= None
dataB= None
outFrames= None
dataTestR= None
dataTestG= None
dataTestB= None

outFramesTest=None


def prep_data(fix_video_x=-1):
    global frame_x
    global dataR
    global dataG
    global dataB
    global outFrames
    global previous_video_x

    if fix_video_x != -1:
        video_x= fix_video_x
    else:
        video_x=-1

    print("video_x")
    print(video_x)

    # Load Images
    big_array02 = []
    big_array03 = []
    big_array04 = []

    for j in range(nextFrameModel.number_of_batches):
        big_array = []
        if fix_video_x == -1:
            video_x = np.random.randint(nextFrameModel.number_of_training_video)
        frame_x = np.random.randint(nextFrameModel.last_frames[int(video_x)] - nextFrameModel.number_of_frames - 1)
        for i in range(nextFrameModel.number_of_frames):
            with open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+i).zfill(4)+".bmp", 'rb') as f:
                image = Image.open(f)
                # We need to consume the whole file inside the `with` statement
                image.load()
                big_array.append(np.array(image))
                image = None
            #big_array.append(np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+i).zfill(4)+".bmp")))
        big_array02.append(big_array)
        big_array03.append([np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+nextFrameModel.number_of_frames).zfill(4)+".bmp"))])

        with open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+nextFrameModel.number_of_frames-1).zfill(4)+".bmp",
                  'rb') as f:
            image = Image.open(f)
            # We need to consume the whole file inside the `with` statement
            image.load()
            big_array04.append(np.array(image))
            image = None
        #big_array04.append([np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x[j]+nextFrameModel.number_of_frames-1).zfill(4)+".bmp"))])

    inFrames = np.array(big_array02)
    outFrames = np.array(big_array03)

    inFramesR = np.squeeze(inFrames[:, :, :, :, [0]], axis=4)
    inFramesR_new = np.squeeze(np.stack(np.split(inFramesR, inFramesR.shape[1], axis=1), axis=-1), axis=1)
    inFramesG = np.squeeze(inFrames[:, :, :, :, [1]], axis=4)
    inFramesG_new = np.squeeze(np.stack(np.split(inFramesG, inFramesG.shape[1], axis=1), axis=-1), axis=1)
    inFramesB = np.squeeze(inFrames[:, :, :, :, [2]], axis=4)
    inFramesB_new = np.squeeze(np.stack(np.split(inFramesB, inFramesB.shape[1], axis=1), axis=-1), axis=1)

    dataR = np.array(inFramesR_new)
    dataG = np.array(inFramesG_new)
    dataB = np.array(inFramesB_new)

def prep_test_data(video_x=-1):
    global frame_x
    global outFramesTest
    global outFrames
    global dataTestR
    global dataTestG
    global dataTestB

    if video_x == -1:
        video_x= np.random.randint(nextFrameModel.number_of_training_video)
    print("video_x")
    print(video_x)

    frame_x= np.random.randint(nextFrameModel.last_frames[int(video_x)]-nextFrameModel.number_of_frames-1, size=nextFrameModel.number_of_batches+1)
    print(str(frame_x[0]).zfill(4))

    # Load Images
    big_array05 = []
    big_array07 = []

    #Prepare test set
    big_array = []
    for i in range(nextFrameModel.number_of_frames):
        big_array.append(np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-"+
                                             str(frame_x[nextFrameModel.number_of_batches]+i).zfill(4)+".bmp")))
    big_array05.append(big_array)
    big_array07.append(
        [np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-" + str(frame_x[nextFrameModel.number_of_batches]
                                                         + nextFrameModel.number_of_frames).zfill(4) + ".bmp"))])
    inFramesTest = np.array(big_array05)
    outFramesTest = np.array(big_array07)

    inFramesTestR = np.squeeze(inFramesTest[:, :, :, :, [0]], axis=4)
    inFramesTestR_new = np.squeeze(np.stack(np.split(inFramesTestR, inFramesTestR.shape[1], axis=1), axis=-1), axis=1)
    inFramesTestG = np.squeeze(inFramesTest[:, :, :, :, [1]], axis=4)
    inFramesTestG_new = np.squeeze(np.stack(np.split(inFramesTestG, inFramesTestG.shape[1], axis=1), axis=-1), axis=1)
    inFramesTestB = np.squeeze(inFramesTest[:, :, :, :, [2]], axis=4)
    inFramesTestB_new = np.squeeze(np.stack(np.split(inFramesTestB, inFramesTestB.shape[1], axis=1), axis=-1), axis=1)

    dataTestR = np.array(inFramesTestR_new)
    dataTestG = np.array(inFramesTestG_new)
    dataTestB = np.array(inFramesTestB_new)


def generator(number_of_batches):
    # Load Images
    big_array02 = []
    big_array03 = []
    big_array04 = []

    idx = 0
    while True:
        for j in range(number_of_batches):
            big_array = []
            video_x = np.random.randint(nextFrameModel.number_of_training_video)
            frame_x = np.random.randint(nextFrameModel.last_frames[int(video_x)] - nextFrameModel.number_of_frames - 1)
            for i in range(nextFrameModel.number_of_frames):
                with open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+i).zfill(4)+".bmp", 'rb') as f:
                    image = Image.open(f)
                    # We need to consume the whole file inside the `with` statement
                    image.load()
                    big_array.append(np.array(image))
                    image = None
            big_array02.append(big_array)
            big_array03.append([np.array(Image.open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+nextFrameModel.number_of_frames).zfill(4)+".bmp"))])
            with open("video/frames"+str(video_x).zfill(2)+"/frame-"+str(frame_x+nextFrameModel.number_of_frames-1).zfill(4)+".bmp",
                      'rb') as f:
                image = Image.open(f)
                # We need to consume the whole file inside the `with` statement
                image.load()
                big_array04.append(np.array(image))
                image = None

        inFrames = np.array(big_array02)
        outFrames = np.array(big_array03)

        inFramesR = np.squeeze(inFrames[:, :, :, :, [0]], axis=4)
        inFramesR_new = np.squeeze(np.stack(np.split(inFramesR, inFramesR.shape[1], axis=1), axis=-1), axis=1)
        inFramesG = np.squeeze(inFrames[:, :, :, :, [1]], axis=4)
        inFramesG_new = np.squeeze(np.stack(np.split(inFramesG, inFramesG.shape[1], axis=1), axis=-1), axis=1)
        inFramesB = np.squeeze(inFrames[:, :, :, :, [2]], axis=4)
        inFramesB_new = np.squeeze(np.stack(np.split(inFramesB, inFramesB.shape[1], axis=1), axis=-1), axis=1)

        dataR = np.array(inFramesR_new)
        dataG = np.array(inFramesG_new)
        dataB = np.array(inFramesB_new)

        yield [dataR,dataG,dataB], outFrames
        print('generator yielded a batch %d' % idx)
        idx += 1



def retrain(epochs=5):
    # model = tf.keras.models.load_model("saves/savetest.h5")
    # Custom loss function do not get save with model.
    # Call class function with dummy parameters for model reconstruction

    model = nextFrameModel.compile()
    model.load_weights("saves/savetest.h5")
    #model.fit([dataR, dataG, dataB, data02], outFrames, epochs=5, batch_size=25)
    if epochs == -1:
        model.fit_generator(generator=generator(25),max_queue_size=4,epochs=1, steps_per_epoch=4, workers=2)
    else:
        model.fit([dataR, dataG, dataB], outFrames, epochs=epochs, batch_size=25,use_multiprocessing=True)


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




while True:
    in01 = input("Please any key to continue..")
    if in01 == "exit" or in01 == "quit":
        break
    if in01.isdigit():
        in02 = input("Enter dataset number...")
        prep_data(int(in02))
        #prep_data()
        retrain(int(in01))
    if in01 == "p" or in01 == "predict":
        in02 = input("Enter dataset number...")
        prep_test_data(int(in02))
        predict()
        print(str(frame_x[0]).zfill(4))
        print(str(frame_x[nextFrameModel.number_of_batches]).zfill(4))
        img = Image.fromarray(outFramesTest[0][0])
        img.show()
        img.save('y_true.png')
    if in01 == "i" or in01 == "infinite":
        while True:
            prep_data()
            retrain(5)
    # only worthwhile if running on gpu because it takes cpu resources and io
    if in01 == "g":
        while True:
            retrain(-1)

'''
reTrain()
predict()

print(str(frame_x[0]).zfill(4))

print(str(frame_x[nextFrameModel.number_of_batches]).zfill(4))

img = Image.fromarray(outFramesTest[0][0])
#img.show()
img.save('y_true.png')
'''
