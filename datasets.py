import os
import sys

import numpy as np
import six
import glob
from PIL import Image

import chainer
from chainer import datasets

class gengochi_train(object):
    def __init__(self):
        # testdata / traindataを作るとしたらディレクトリを分ける。
        # ラベルはtsvかcsvを作り関連付けするかファイル名。
        imgs = glob.glob("image/gochi/*.png")
        self.rawdata = []
        for i in imgs:
            self.rawdata.append(np.asarray(Image.open(i)))

    # refar to... chainer/datasets/cifar.py
    # cifarライクなデータセットなので参考にした
    def _get_gochiusa(self, withlabel=True, ndim=3, scale=1, dtype=None):
        images = np.asarray(self.rawdata)
        if ndim == 1:
            images = images.reshape(-1, 3*128*128)
        elif ndim == 3:
            images = images.reshape(-1, 3, 128, 128)
        images = images.astype("f")
        images *= scale / 255.

        #if withlabel:
        #    labels = labels.astype(numpy.int32)
        #    return tuple_dataset.TupleDataset(images, labels)
        #else:
        return images
