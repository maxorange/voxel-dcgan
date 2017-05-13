import numpy as np
import time
import util
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class ConstrainedOpt(QThread):

    def __init__(self, model):
        QThread.__init__(self)
        self.model = model
        self.latent_code = np.random.uniform(-1, 1, [model.batch_size, model.nz]).astype(np.float32)
        self.z0 = np.random.uniform(-1, 1, [self.model.nz]).astype(np.float32)
        self.z1 = np.random.uniform(-1, 1, [self.model.nz]).astype(np.float32)
        self.index = 0
        self.alpha = np.arange(0.0, 1.0, 0.01)

    def run(self):
        while True:
            self.update_voxel_model()
            self.msleep(10)

    def update_voxel_model(self):
        self.latent_code[0] = (1. - self.alpha[self.index])*self.z0 + self.alpha[self.index]*self.z1
        self.current_shape = self.model.generate(self.latent_code)
        self.emit(SIGNAL('update_voxels'))

        if self.index == 99:
            self.z0 = self.z1
            self.z1 = np.random.uniform(-1, 1, [self.model.nz]).astype(np.float32)
            self.index = 0
        else:
            self.index += 1
