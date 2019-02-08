from PIL import Image
import os, sys

path = "video/frames/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((72,72), Image.ANTIALIAS)
            im.close()
            print(imResize.size)
            imResize.save(f + '.bmp', 'BMP')

resize()