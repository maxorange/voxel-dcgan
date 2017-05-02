import numpy as np
import binvox

def read_binvox(filename):
    with open(filename, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        data = model.data.astype(np.float32)
        return np.expand_dims(data, -1)

def save_binvox(filename, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(filename, 'wb') as f:
        model.write(f)
