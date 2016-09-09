import tensorflow as tf
import numpy as np
import binvox

def read_binvox(fname):
    with open(fname, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        model = (model.data.astype(np.float32) - 0.5) / 0.5
        return model

def save_binvox(data, fname):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(fname, 'wb') as f:
        model.write(f)
