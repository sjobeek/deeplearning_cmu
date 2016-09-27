
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


IMG_DIM = 28

def tomat(vec):
    return vec.reshape((IMG_DIM,IMG_DIM))

def plot_vec(img_vec):
    """img_mat must be single vector-representation of image to plot"""
    img_mat = img_vec.reshape((IMG_DIM,IMG_DIM))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(img_mat, cmap=mpl.cm.binary)
    ax.axis('off')
    plt.show()

def plot_100(img_vec_array):
    first_100 = img_vec_array[:100]
    image_mat_10x10 = np.zeros((IMG_DIM*10, IMG_DIM*10))
    for x in range(10):
        for y in range(10):
            image_mat_10x10[IMG_DIM*y : IMG_DIM*y+IMG_DIM,
                            IMG_DIM*x : IMG_DIM*x+IMG_DIM] = tomat(first_100[10*y + x])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image_mat_10x10, cmap=mpl.cm.binary)
    ax.axis('off')
    plt.show()