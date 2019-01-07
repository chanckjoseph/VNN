import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image

number_of_frames = 10
last_frames = 360

image_width=720
image_height=720

horizontal_axis=2
vertical_axis=1

print(tf.VERSION)
print(tf.keras.__version__)

frame_x= np.random.randint(last_frames-number_of_frames-1, size=1)
print(str(frame_x[0]).zfill(4))

# Load Images
big_array = []
for i in range(number_of_frames):
    big_array.append(np.array(Image.open("video/frames/frame-"+str(frame_x[0]+i).zfill(4)+".bmp")))
inFrames = np.array(big_array)

im = inFrames[0]

print(im.shape)
img = Image.fromarray(im,'RGB')
#img.show()

im = np.roll(im, 300, axis=1) # horizontal
im = np.roll(im, 300, axis=0) # vertical
#img = Image.fromarray(im,'RGB')
#img.show()
