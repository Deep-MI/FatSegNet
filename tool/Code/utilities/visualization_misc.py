# Copyright 2019 Population Health Sciences and Image Analysis, German Center for Neurodegenerative Diseases(DZNE)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import itertools

def get_colors(inp, colormap, vmin=None, vmax=None):
    """generate the normalize rgb values for matplolib
"""
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def multiview_plotting(data,labels,control_point, savepath,classes=5,alpha=0.5,nbviews=3,plot_labels=True,plot_control_point=True):
    """Plot data and label in different views
    Args:
        data: Original 3D volume
        labels: Original labels for the 3d Volume
        control_point: select the center point where the different views are going to be created
        savepath:path where the image is going to be safe
        classes: number of classes in the labeles
        alpha: transparency of the labels on the original data
        nbviews: 1 only axial view,2 axial and frontal, 3 the three views
        plot_labels: True plot labels, False only plot data


    Returns:
        An images with the diffent views and the corresponding label
"""
    # Create the colormap for the labels
    dz=np.arange(classes)
    colors = get_colors(dz, plt.cm.jet)
    #replace first color for black
    colors[0, 0:3] = [0, 0, 0]
    my_cm=LinearSegmentedColormap.from_list('mylist',colors,classes)
    plt.ioff()
    if plot_labels:
        grid_size = [2, nbviews]
    else:
        grid_size = [1, nbviews]
    #fig = plt.figure(dpi=600)
    fig = plt.gcf()


    outer_grid = gridspec.GridSpec(grid_size[0], grid_size[1], wspace=0.05, hspace=0.0005)
    index = 0
    for i in range(nbviews):
        if i == 0 :
            ax = plt.subplot(outer_grid[i])
            ax.imshow(data[control_point[0], :, :], cmap=cm.gray)
            if plot_control_point:
                ax.scatter(y=control_point[1], x=control_point[2], c='r', s=2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if plot_labels:
                ax = plt.subplot(outer_grid[i+nbviews])
                ax.imshow(data[control_point[0], :, :], cmap=cm.gray)
                ax.imshow(labels[control_point[0], :, :],vmin=0,vmax=classes, cmap=my_cm,alpha=alpha)
                if plot_control_point:
                    ax.scatter(y=control_point[1], x=control_point[2], c='r', s=2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        elif i == 1:
            ax = plt.subplot(outer_grid[i])
            ax.imshow(data[:, control_point[1], :], cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
            if plot_control_point:
                ax.scatter(y=control_point[0], x=control_point[2], c='r', s=2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if plot_labels:
                ax = plt.subplot(outer_grid[i+nbviews])
                ax.imshow(data[:, control_point[1], :], cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
                ax.imshow(labels[:, control_point[1], :],vmin=0,vmax=classes, cmap=my_cm,alpha=alpha,aspect=(data.shape[1]/data.shape[0]))
                if plot_control_point:
                    ax.scatter(y=control_point[0], x=control_point[2], c='r', s=2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        elif i == 2:
            img=np.zeros((data.shape[0],data.shape[2]))
            diff_spacing=int((data.shape[2]-data.shape[1])/2)
            img[:,diff_spacing:data.shape[2]-diff_spacing]=data[:, :, control_point[2]]

            ax = plt.subplot(outer_grid[i])
            ax.imshow(img, cmap=cm.gray,aspect=(data.shape[1]/data.shape[0]))
            if plot_control_point:
                ax.scatter(y=control_point[0], x=control_point[2], c='r', s=2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if  plot_labels:
                img_label = np.zeros((data.shape[0], data.shape[2]))
                img_label[:, diff_spacing:data.shape[2] - diff_spacing] = labels[:, :, control_point[2]]
                ax = plt.subplot(outer_grid[i + nbviews])
                ax.imshow(img, cmap=cm.gray, aspect=(data.shape[1] / data.shape[0]))
                ax.imshow(img_label,vmin=0,vmax=classes, cmap=my_cm,alpha=alpha, aspect=(data.shape[1] / data.shape[0]))
                if plot_control_point:
                    ax.scatter(y=control_point[0], x=control_point[2], c='r', s=2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    #fig.subplots_adjust(wspace=0.001, hspace=0.001)

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(savepath, transparent=True,bbbox_inches='tight',pad_inches=0,dpi=300)
    plt.close(fig)


