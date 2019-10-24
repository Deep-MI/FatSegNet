
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
from keras import backend as K
from keras import metrics


# %%1.DICE LOSS
smooth = 1
w_dice = 0.5

K.set_epsilon(1e-7)
np.set_printoptions(threshold=np.inf)
K.set_image_data_format('channels_last')

def average_dice_coef(y_true,y_pred):
    avg_dice=0
    for i in range(y_pred.shape[-1]):
        avg_dice += dice_coef_axis(y_true,y_pred,i)
    return avg_dice/(i+1)

def dice_coef(y_true, y_pred):
        intersection = 0
        union = 0
        if len(y_pred.shape)==5:
            for i in range(y_pred.shape[-1]):
                intersection += (K.sum(y_true[:, :, :,:, i] * y_pred[:, :, :,:, i]))
                union += (K.sum(y_true[:, :, :,:, i] + y_pred[:, :, :,:, i]))
            return (2. * intersection + smooth) / (union + smooth)
        elif len(y_pred.shape)==4:
            for i in range(y_pred.shape[-1]):
                intersection +=  (K.sum(y_true[:, :, :, i] * y_pred[:, :, :, i]))
                union += (K.sum(y_true[:, :, :, i] + y_pred[:, :, :, i]))
            return (2. * intersection + smooth) / (union + smooth)


# %%CLASS-WISE-DICE
def dice_coef_axis(y_true, y_pred, i):

    intersection = 0
    #med_bal_factor = [1, 1, 1, 1]  # TODO_ remove it. After testing
    union = 0
    if len(y_pred.shape)==4:
        intersection += (K.sum(y_true[:, :, :, i] * y_pred[:, :, :, i]))
        union +=(K.sum(y_true[:, :, :, i] + y_pred[:, :, :, i]))
        return (2. * intersection + smooth) / (union + smooth)
    elif len(y_pred.shape)==5:
        intersection += (K.sum(y_true[:, :, :, :, i] * y_pred[:, :, :, :, i]))
        union += (K.sum(y_true[:, :, :, :, i] + y_pred[:, :, :, :, i]))
        return (2. * intersection + smooth) / (union + smooth)

def dice_coef_0(y_true, y_pred):
    return dice_coef_axis(y_true, y_pred, 0)


def dice_coef_1(y_true, y_pred):
    return dice_coef_axis(y_true, y_pred, 1)


def dice_coef_2(y_true, y_pred):
    return dice_coef_axis(y_true, y_pred, 2)


def dice_coef_3(y_true, y_pred):
    return dice_coef_axis(y_true, y_pred, 3)

def dice_coef_4(y_true, y_pred):
    return dice_coef_axis(y_true, y_pred, 4)

def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

def jaccard_coef(y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    intersection = K.tf.reduce_sum(y_pred * y_true) + smooth
    sum_=(K.tf.reduce_sum(y_true) + K.tf.reduce_sum(y_pred))
    union=sum_-intersection+smooth
    jac=intersection/union

    return jac


def custom_loss(MedBalFactor,sigma=3,loss_type='Dice'):
    n_classes=len(MedBalFactor)

    def get_gauss_kernel_3D(sigma):
        ker=np.zeros(shape=(3, 3, 3, n_classes, n_classes), dtype='float32')
        ind = np.linspace(-np.floor(ker.shape[1]), np.floor(ker.shape[1]), ker.shape[1])
        ind2 = np.linspace(-np.floor(ker.shape[2]), np.floor(ker.shape[2]), ker.shape[2])
        x, y = np.meshgrid(ind, ind2)
        G=np.zeros((3,3,3))
        for i in range(ker.shape[0]):
            G[i,:,:] = (np.exp((-1 / (2 * sigma ** 2)) * (x ** 2 + y ** 2)))
        G = G / np.sum(G)
        for i in range(n_classes):
            ker[:,:, :, i, i] = G
        ker = K.constant(ker)
        return ker


    def get_gauss_kernel(sigma):
        ker = np.zeros(shape=(3, 3, n_classes, n_classes), dtype='float32')

        ind = np.linspace(-np.floor(ker.shape[0]), np.floor(ker.shape[0]), ker.shape[0])
        ind2 = np.linspace(-np.floor(ker.shape[1]), np.floor(ker.shape[1]), ker.shape[1])
        x, y = np.meshgrid(ind, ind2)
        G = (np.exp((-1 / (2 * sigma ** 2)) * (x ** 2 + y ** 2)))
        G = G / np.sum(G)
        for i in range(n_classes):
            ker[:, :, i, i] = G
        ker = K.constant(ker)
        return ker

    def get_sobel_kernel_3D(axis):
        ker = np.zeros(shape=(3,3, 3, n_classes, n_classes), dtype='float32')

        if axis == 'z':
            S=np.array([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]])
        else:
            s = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]], dtype='float32')
            if axis == 'y':
                pass
            elif axis == 'x':
                s = np.transpose(s, )

            S = np.zeros((3, 3, 3))
            for i in range(ker.shape[0]):
                S[i,:,:] = s[:]

        for i in range(n_classes):
            ker[:,:, :, i, i] = S
        ker = K.constant(ker)
        return ker

    def get_sobel_kernel(axis):
        s = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype='float32')
        if axis == 'y':
            pass
        elif axis == 'x':
            s = np.transpose(s, )
        ker = np.zeros(shape=(3, 3, n_classes, n_classes), dtype='float32')
        for i in range(n_classes):
            ker[:, :, i, i] = s
        ker = K.constant(ker)
        return ker

    GAUSS_KERNEL_3D=get_gauss_kernel_3D(sigma)
    GAUSS_KERNEL = get_gauss_kernel(sigma)
    SOBEL_X = get_sobel_kernel('x')
    SOBEL_Y = get_sobel_kernel('y')
    SOBEL_X_3D=get_sobel_kernel_3D('x')
    SOBEL_Y_3D = get_sobel_kernel_3D('y')
    SOBEL_Z_3D = get_sobel_kernel_3D('z')

    def get_grad_tensor_3d(img_tensor,apply_gauss=True):

        grad_x = K.conv3d(img_tensor, SOBEL_X_3D, padding='same')
        grad_y = K.conv3d(img_tensor, SOBEL_Y_3D, padding='same')
        grad_z= K.conv3d(img_tensor, SOBEL_Z_3D, padding='same')
        grad_tensor = K.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z*grad_z)
        grad_tensor = K.greater(grad_tensor, 100.0 * K.epsilon())
        grad_tensor = K.cast(grad_tensor, K.floatx())
        grad_tensor = K.clip(grad_tensor, K.epsilon(), 1.0)
        grad_map = K.sum(grad_tensor, axis=-1, keepdims=True)
        for i in range(n_classes):
            if i ==0:
                grad_tensor=grad_map[:]
            else:
                grad_tensor = K.concatenate([grad_tensor,grad_map], axis=-1)
        # del grad_map
        # grad_tensor = K.concatenate([grad_tensor, grad_tensor], axis=CHANNEL_AXIS)
        grad_tensor = K.greater(grad_tensor, 100.0 * K.epsilon())
        grad_tensor = K.cast(grad_tensor, K.floatx())
        if apply_gauss:
            grad_tensor = K.conv3d(grad_tensor, GAUSS_KERNEL_3D, padding='same')
        return grad_tensor


    def get_grad_tensor(img_tensor, apply_gauss=True):
        grad_x = K.conv2d(img_tensor, SOBEL_X, padding='same')
        grad_y = K.conv2d(img_tensor, SOBEL_Y, padding='same')

        grad_tensor = K.sqrt(grad_x * grad_x + grad_y * grad_y)
        grad_tensor = K.greater(grad_tensor, 100.0 * K.epsilon())
        grad_tensor = K.cast(grad_tensor, K.floatx())
        grad_tensor = K.clip(grad_tensor, K.epsilon(), 1.0)
        grad_map = K.sum(grad_tensor, axis=-1, keepdims=True)
        for i in range(n_classes):
            if i ==0:
                grad_tensor=grad_map[:]
            else:
                grad_tensor = K.concatenate([grad_tensor,grad_map], axis=-1)
        # del grad_map
        # grad_tensor = K.concatenate([grad_tensor, grad_tensor], axis=CHANNEL_AXIS)
        grad_tensor = K.greater(grad_tensor, 100.0 * K.epsilon())
        grad_tensor = K.cast(grad_tensor, K.floatx())
        if apply_gauss:
            grad_tensor = K.conv2d(grad_tensor, GAUSS_KERNEL, padding='same')
        return grad_tensor

    def weighted_gradient_loss(y_true,y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        weights = []

        if len(y_pred.shape)==4:
            axis = [0, 1, 2]
            if np.max(MedBalFactor)> 5:
                edge_weights=10*get_grad_tensor(y_true,True)
            else:
                edge_weights = 2 * np.max(MedBalFactor) * get_grad_tensor(y_true, True)

            for i in range(len(MedBalFactor)):
                weights.append(MedBalFactor[i] * K.ones_like(y_true[:, :, :, i:i+1]))

        elif len(y_pred.shape) == 5:
            axis = [0, 1, 2, 3]

            if np.max(MedBalFactor) > 5:
                edge_weights = 10 * get_grad_tensor_3d(y_true, True)
            else:
                edge_weights = 2 * np.max(MedBalFactor) * get_grad_tensor_3d(y_true, True)

            for i in range(len(MedBalFactor)):
                weights.append(MedBalFactor[i] * K.ones_like(y_true[:, :, :, :, i:i + 1]))

        class_weights = K.concatenate(weights, axis=-1)
        class_weights=K.tf.add(class_weights,edge_weights)
        cross_entropy_part=-1.0 * K.tf.reduce_sum(K.tf.reduce_mean(K.tf.multiply(y_true * K.tf.log(y_pred),class_weights),axis=axis,keepdims=True))
        return cross_entropy_part

    def weighted_logistic_loss(y_true,y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

        weights = []
        if len(y_pred.shape)==4:
            axis=[0,1,2]
            for i in range(len(MedBalFactor)):
                weights.append(MedBalFactor[i] * K.ones_like(y_true[:,:,:, i:i + 1]))

        elif len(y_pred.shape)==5:
            axis=[0,1,2,3]
            for i in range(len(MedBalFactor)):
                weights.append(MedBalFactor[i] * K.ones_like(y_true[:,:,:,:, i:i + 1]))

        class_weights = K.concatenate(weights, axis=-1)
        cross_entropy_part=-1.0 * K.tf.reduce_sum(K.tf.reduce_mean(K.tf.multiply(y_true * K.tf.log(y_pred),class_weights),axis=axis,keepdims=True))
        return cross_entropy_part

    def logistic_loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        if len(y_pred.shape)==4:
            axis=[0,1,2]
        elif len(y_pred.shape)==5:
            axis=[0,1,2,3]
        cross_entropy_part=-1.0 * K.tf.reduce_sum(K.tf.reduce_mean((y_true * K.tf.log(y_pred)),axis=axis,keepdims=True))
        return cross_entropy_part

    def dice_loss(y_true,y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        intersection = K.tf.reduce_sum(y_pred * y_true) + smooth
        union = (K.tf.reduce_sum(y_true) + K.tf.reduce_sum(y_pred)) + smooth
        dice_part = -2.0 * (intersection / union)

        return dice_part

    def mixed_loss(y_true,y_pred):
        if loss_type == 'Dice':
            return dice_loss(y_true,y_pred)
        elif loss_type == 'Logistic':
            return logistic_loss(y_true,y_pred)
        elif loss_type == 'Weighted_Logistic':
            return weighted_logistic_loss(y_true,y_pred)
        elif loss_type == 'Weighted_Grad_Logistic':
            return weighted_gradient_loss(y_true,y_pred)
        elif loss_type == 'Mixed_Grad_Weighted':
            dice_part=dice_loss(y_true,y_pred)
            cross_entropy_part = weighted_gradient_loss(y_true, y_pred)
            return cross_entropy_part + dice_part
        elif loss_type== 'Mixed':
            dice_part=dice_loss(y_true,y_pred)
            cross_entropy_part=logistic_loss(y_true,y_pred)
            return cross_entropy_part + dice_part
        elif loss_type == 'Mixed_Weighted':
            dice_part = dice_loss(y_true, y_pred)
            cross_entropy_part=weighted_logistic_loss(y_true,y_pred)
            return cross_entropy_part + dice_part

    return mixed_loss










