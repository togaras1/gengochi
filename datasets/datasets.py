import os
import numpy as np
from PIL import Image
import six
import json
import cv2
import glob
from io import BytesIO
import numpy as np
from .datasets_base import datasets_base

class gengochi_train(datasets_base):
    def __init__(self, flip=1, resize_to=128, crop_to=64):
        super(gengochi_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.traingochi = glob.glob("image/gochi/*.png")

    def __len__(self):
        return len(self.traingochi)

    def do_resize(self, img):
        #print(img.shape)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        #print(img.shape)
        return img

    def do_random_crop(self, img, crop_to=64):
        w, h, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = self.do_flip(img)

        if self.resize_to > 0:
            img = self.do_resize(img)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)
        return img

    def get_example(self, i):
        np.random.seed(None)
        idg = self.traingochi[np.random.randint(0,len(self.traingochi))]
        #print(idA)

        imgg = cv2.imread(idg, cv2.IMREAD_COLOR)

        imgg = self.do_augmentation(imgg)

        imgg = self.preprocess_image(imgg)

        return imgg
