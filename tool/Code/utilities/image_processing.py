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

import numpy as np
from skimage.measure import label


def largets_connected_componets(labels):
    """Calculate the largest connected component, all the labels are unified to one
    Args:
        labels: ndarray (int or float) label image or volume
        neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-“connectivity”. In 3D, 4-“connectivity” means connected pixels have to share face, whereas with 8-“connectivity”,
        they have to share only edge or vertex. Deprecated, use ``connectivity`` instead.


    Returns:
        out :ndarray, the input array only with the largest connected component
"""
    mask = np.copy(labels)
    mask[labels > 0] = 1
    connected_labels, num = label(mask,connectivity=3,background=0, return_num=True)
    #0 is background data, so check with out zero
    #largestCC = np.argmax(np.bincount(connected_labels.flat)[1:])
    if num !=1 :
        unique, counts = np.unique(connected_labels, return_counts=True)
        largest=np.argmax(counts[1:]) + 1 #0 is background data, so check with out zero
        mask[connected_labels != largest] = 0

    mask = np.array(mask, dtype=np.int8)

    return labels * mask


def swap_axes(data,plane):
    if plane == 'axial':
        return  data
    elif plane == 'frontal':
        data = np.swapaxes(data, 1, 0)
        return data
    elif plane == 'sagital':
        data = np.swapaxes(data, 2, 0)
        return data

def check_size(data,patch_size):

    x_low=int(np.floor(-1*(data.shape[1]-patch_size[0])/2))
    x_high=int(np.ceil(-1*(data.shape[1]-patch_size[0])/2))



    y_low=int(np.floor(-1*(data.shape[2]-patch_size[1])/2))
    y_high=int(np.ceil(-1*(data.shape[2]-patch_size[1])/2))

    new_arr=np.zeros((data.shape[0],patch_size[0],patch_size[1]))

    new_arr[:,x_low:patch_size[0]-x_high,y_low:patch_size[1]-y_high]=data[:,:,:]

    return new_arr

def change_data_plane(arr, plane='axial',return_index=False):
    if plane == 'axial':
        if return_index:
            return arr,0, arr.shape[0]
        else:
            return arr

    elif plane == 'frontal' or plane == 'coronal':
            if len(arr.shape) == 4:
                new_arr = np.zeros((arr.shape[1], arr.shape[1], arr.shape[2],arr.shape[3]))
                for slice in range(arr.shape[3]):
                    aux_arr=arr[:,:,:,slice]
                    aux_arr = np.swapaxes(aux_arr, 1, 0)
                    idx_low = int((new_arr.shape[1] / 2) - (aux_arr.shape[1] / 2))
                    idx_high = int((new_arr.shape[1] / 2) + (aux_arr.shape[1] / 2))
                    new_arr[:, idx_low:idx_high, :,slice] = aux_arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
            else:
                new_arr = np.zeros((arr.shape[1], arr.shape[1], arr.shape[2]))
                arr = np.swapaxes(arr, 1, 0)
                idx_low = int((new_arr.shape[1] / 2) - (arr.shape[1] / 2))
                idx_high = int((new_arr.shape[1] / 2) + (arr.shape[1] / 2))
                new_arr[:, idx_low:idx_high, :] = arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
    elif plane == 'sagital' or plane == 'sagittal':
            if len(arr.shape)== 4:
                new_arr = np.zeros((arr.shape[2], arr.shape[1], arr.shape[2],arr.shape[3]))
                for slice in range(arr.shape[3]):
                    aux_arr=arr[:,:,:,slice]
                    aux_arr = np.swapaxes(aux_arr, 2, 0)

                    idx_low = int((new_arr.shape[2] / 2) - (aux_arr.shape[2] / 2))
                    idx_high = int((new_arr.shape[2] / 2) + (aux_arr.shape[2] / 2))

                    new_arr[:, :, idx_low:idx_high,slice] = aux_arr[:]
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr
            else:
                new_arr = np.zeros((arr.shape[2], arr.shape[1], arr.shape[2]))
                arr = np.swapaxes(arr, 2, 0)

                idx_low = int((new_arr.shape[2] / 2) - (arr.shape[2] / 2))
                idx_high = int((new_arr.shape[2] / 2) + (arr.shape[2] / 2))

                new_arr[:, :, idx_low:idx_high] = arr
                if return_index:
                    return new_arr,idx_low,idx_high
                else:
                    return new_arr

def find_labels(arr):
    idx=(np.where(arr > 0))
    min_idx=np.min(idx[0])
    max_idx=np.max(idx[0])
    return max_idx,min_idx
