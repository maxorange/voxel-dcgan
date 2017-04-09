# VoxelDCGAN

Implementation of a 3D shape generative model based on <a href="https://arxiv.org/abs/1511.06434">deep convolutional generative adversarial nets</a> (DCGAN) with techniques of <a href="https://github.com/openai/improved-gan">improved-gan</a>.

Experimental results on the <a href="http://shapenet.cs.stanford.edu/">ShapeNet</a> dataset are shown below.

### Random sampling

<img src="img/rs-1.png" width=700>

### Linear interpolation

<img src="img/li-1.gif" width=200>
<img src="img/li-2.gif" width=200>
<img src="img/li-3.gif" width=200>
<img src="img/li-4.gif" width=200>

## Dependencies

* tensorflow https://github.com/tensorflow/tensorflow
* numpy https://github.com/numpy/numpy
* binvox-rw-py https://github.com/dimatura/binvox-rw-py

