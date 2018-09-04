import os
import sys

import numpy as np
import six
import glob
from PIL import Image

import chainer
from chainer import datasets

class gengochi_train(object):
    def __init__(self, size_to=128):
        # testdata / traindataを作るとしたらディレクトリを分ける。
        # ラベルはtsvかcsvを作り関連付けするかファイル名。
        self.size = size_to
        path = "image/gochi/*.png" #"image/gochi/*.png"
        imgs = glob.glob(path)
        self.rawdata = []
        for j in range(100): # generate anime image
            for i in imgs:
                # alphaチャンネルつきPNGだったときのために一応3チャンネルに変換
                self.rawdata.append(np.asarray(Image.open(i).convert("RGB").resize((size_to,size_to))))

    # refar to... chainer/datasets/cifar.py
    # cifarライクなデータセットなので参考にした
    def _get_gochiusa(self, withlabel=True, ndim=3, scale=1, dtype=None):
        images = np.asarray(self.rawdata)
        if ndim == 1: # i,x,y,rgb to i,r,g,b
            images = images.transpose(0,3,1,2).reshape(-1,3*self.size*self.size)
        elif ndim == 3:
            images = images.transpose(0,3,1,2)
        images = images.astype("f")
        images *= scale / 255.

        #if withlabel:
        #    labels = labels.astype(numpy.int32)
        #    return tuple_dataset.TupleDataset(images, labels)
        #else:
        print('{} images are loaded'.format(images.shape))
        return images
