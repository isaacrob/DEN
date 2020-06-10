import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import Counter
np.random.seed(37)

def plot_2d(reduced, y = None, s = 1, alpha = .2, show = True, no_legend = False, y_names = None, plot_device = None):
    if plot_device is None:
        plot_device = plt
        plot_device.figure()
    if y is not None:
        s = plt.scatter(reduced[:, 0], reduced[:, 1], s = s, alpha = alpha, c = y)
        if not no_legend:
            if y_names is None:
                y_names = list(np.unique(y))
            plot_device.legend(loc = 'upper right', handles = s.legend_elements()[0], labels = y_names)
    else:
        plot_device.scatter(reduced[:, 0], reduced[:, 1], s = s, alpha = alpha)

    if show:
        plt.show()

def plot_for_compare(embeddings, y = None, s = 1, alpha = .2, show = True):
    fig, axs = plt.subplots(1, len(embeddings))

    print(axs)
    print(axs.shape)

    for i, embedding in enumerate(embeddings):
        axs[i].scatter(embedding[:, 0], embedding[:, 1], c = y, alpha = alpha, s = s)

    fig.tight_layout()

    if show:
        plt.show()


def plot_3d(reduced, y = None, s = 1, alpha = .2, show = True):
    fig = plt.figure()
    if y is not None:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], s = s, alpha = alpha, c = y)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], s = s, alpha = alpha)

    if show:
        plt.show()

def plot_2d_with_images(reduced, images, show = True, n = 200, zoom = .5):
    fig, ax = plt.subplots()
    inds = np.random.choice(reduced.shape[0], n, replace = False)
    reduced = reduced[inds]
    images = images[inds]
    ax.scatter(reduced[:, 0], reduced[:, 1], s = 0)

    for point, img in zip(reduced, images):
        img = OffsetImage(np.array(img), cmap = 'gray_r', zoom = zoom)
        ab = AnnotationBbox(img, point, frameon = False)
        ax.add_artist(ab)

    if show:
        plt.show()
