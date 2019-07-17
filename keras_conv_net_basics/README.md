# [Tensorflow Keras API](https://medium.com/@mageswaran1989/text-ublished information-extraction-2b4a976409ed)

Also the Story published @ https://blog.imaginea.com/text-information-extraction-2/

### The CALTECH-101 (subset) dataset

The CALTECH-101 dataset is a dataset of 101 object categories with 40 to 800 images per class.

Most images have approximately 50 images per class.

The goal of the dataset is to train a model capable of predicting the target class.

Prior to the resurgence of neural networks and deep learning, the state-of-the-art accuracy on was only ~65%.

However, by using Convolutional Neural Networks, it’s been possible to achieve 90%+ accuracy (as He et al. 
demonstrated in their 2014 paper, Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition).

Today we are going to implement a simple yet effective CNN that is capable of achieving 96%+ accuracy, on a 
4-class subset of the dataset:

    Faces: 436 images
    Leopards: 201 images
    Motorbikes: 799 images
    Airplanes: 801 images

The reason we are using a subset of the dataset is so you can easily follow along with this example and train the 
network from scratch, even if you do not have a GPU.

Again, the purpose of this tutorial is not meant to deliver state-of-the-art results on CALTECH-101 — it’s instead meant 
to teach you the fundamentals of how to use Keras’ Conv2D class to implement and train a custom Convolutional Neural Network.

### Dataset

```
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -zxvf 101_ObjectCategories.tar.gz
```

### How to run ?

```
python stridednet_train.py --dataset=101_ObjectCategories --plot=stridnet_results.png --epochs=50
# following command will download the Resnet weights from web!
python resnet_train.py --dataset=101_ObjectCategories --plot=resnet_results.png --epochs=50

```

**Tips**
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
Use above snippet to see whether your current Tensorflow installation has support for GPU

**References**

- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
- https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
- http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- https://github.com/jrosebr1/imutils
- [Resnet](https://arxiv.org/pdf/1512.03385.pdf) Materials
    - https://www.kaggle.com/cokastefan/keras-resnet-50
    - http://teleported.in/posts/decoding-resnet-architecture/
    - https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718
    - https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
    - http://teleported.in/posts/decoding-resnet-architecture/
    
