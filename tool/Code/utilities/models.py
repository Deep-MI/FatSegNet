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



import sys
sys.path.append('../')
sys.path.append('./')
import os

import numpy as np
from keras.models import load_model
from keras import backend as K
import Code.utilities.loss as loss
from Code.utilities.image_processing import change_data_plane,swap_axes,largets_connected_componets,find_labels




def find_unique_index_slice(data):

    aux_index=[]

    for z in range(data.shape[0]):
        labels,counts=np.unique(data[z,:,:],return_counts=True)
        if 2 in labels:
            num_pixels=np.sum(counts[1:])
            position=np.where(labels==2)
            if counts[position[0][0]] >= (num_pixels*0.8):
                aux_index.append(z)


    higher_index=np.max(aux_index)
    lower_index= np.min(aux_index)

    return higher_index,lower_index


def run_adipose_localization(data,flags):

    print('-' * 30)
    print ('Run Abdominal Localization Block')
    planes=['coronal','sagittal']
    high_idx=0
    low_idx=0
    for plane in planes:
        plane_model = os.path.join(flags['localizationModels'], 'Loc_CDFNet_Baseline_' + str(plane))
        params_path = os.path.join(plane_model, 'train_parameters.npy')
        params = np.load(params_path).item()
        params['modelParams']['SavePath'] = plane_model
        tmp_high_idx,tmp_low_idx=test_localization_model(params,data)
        high_idx += tmp_high_idx
        low_idx +=tmp_low_idx

    high_idx=int(high_idx // 2)
    low_idx=int(low_idx // 2)
    return high_idx,low_idx



def run_adipose_segmentation(data,flags,args):

    print('-' * 30)
    print ('Run AAT Segmentation Block')

    # ============  Load Params ==================================
    # Multiviewmodel
    multiview_path = os.path.join(flags['multiviewModel'], 'Baseline_Mixed_Multi_Plane')
    multiview_params = np.load(os.path.join(multiview_path,'train_parameters.npy')).item()
    multiview_params['modelParams']['SavePath']= multiview_path
    nbclasses = multiview_params['modelParams']['nClasses']
    # uni axial Model Path
    if args.axial:
        print('-' * 30)
        print('Segmentation done only on the axial plane')
        print('-' * 30)
        base_line_dir_axial = os.path.join(flags['singleViewModels'], 'CDFNet_Baseline_axial')
        base_line_dirs=[]
        base_line_dirs.append(base_line_dir_axial)
    else:
        base_line_dir_axial  = os.path.join(flags['singleViewModels'],'CDFNet_Baseline_axial')
        base_line_dir_frontal= os.path.join(flags['singleViewModels'],'CDFNet_Baseline_coronal')
        base_line_dir_sagital= os.path.join(flags['singleViewModels'],'CDFNet_Baseline_sagittal')

        base_line_dirs=[]
        base_line_dirs.append(base_line_dir_axial)
        base_line_dirs.append(base_line_dir_frontal)
        base_line_dirs.append(base_line_dir_sagital)


        test_data = np.zeros((1, multiview_params['modelParams']['PatchSize'][0],
                              multiview_params['modelParams']['PatchSize'][1],
                              multiview_params['modelParams']['PatchSize'][2], len(base_line_dirs) * nbclasses))
    i = 0
    for plane_model in base_line_dirs:
        print(plane_model)
        params_path = os.path.join(plane_model, 'train_parameters.npy')
        params = np.load(params_path).item()
        params['modelParams']['SavePath']=plane_model
        if args.axial:
            y_predict = test_model(params, data)
        else:
            test_data[0, 0:data.shape[0], :, :, i * nbclasses:(i + 1) * nbclasses] = test_model(params, data)
        i += 1
    if args.axial:
        final_img = np.argmax(y_predict, axis=-1)
        final_img = np.asarray(final_img, dtype=np.int16)
    else:
        final_img = test_multiplane(multiview_params, test_data)

    return final_img


def test_multiplane(params,data):
    """Segmentation network for the probability maps of frontal,axial and sagittal
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing 15 probability maps

    Returns:
        out :ndarray, prediction array of 5 classes
"""
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    #model_path = os.path.join(params['modelParams']['SavePath'], model_name)
    model_path = params['modelParams']['SavePath']
    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor=params['modelParams']['MedFrequency']
    loss_type=params['modelParams']['Loss_Function']
    sigma=params['modelParams']['GradientSigma']
    print('-' * 30)
    print('model path')
    print(model_path + '/logs/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/logs/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})



    print('-' * 30)
    print('Evaluating Multiview model  ...')
    print('-' * 30)


    y_predict = model.predict(data, batch_size=batch_size, verbose=0)

    # Reorganize prediction data
    y_predict = np.argmax(y_predict, axis=-1)
    y_predict = y_predict.reshape(data.shape[1], data.shape[2], data.shape[3])
    y_predict = np.asarray(y_predict, dtype=np.int16)
    print(y_predict.shape)


    return y_predict

def test_model(params,data):
    """Segmentation network for each view (frontal,axial and sagittal)
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing the fat image

    Returns:
        out :ndarray, prediction array of 5 classes for each view
    """
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    model_path = params['modelParams']['SavePath']
    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor = params['modelParams']['MedFrequency']
    loss_type = params['modelParams']['Loss_Function']
    sigma = params['modelParams']['GradientSigma']
    plane = params['modelParams']['Plane']

    if plane == 'frontal':
        plane = 'coronal'
    if plane == 'sagital':
        plane = 'sagittal'

    print('-' * 30)
    print('Evaluating %s...' % plane)
    print('-' * 30)
    print('Testing %s'%model_name)
    print('model path')
    print(model_path + '/logs/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/logs/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})




    X_test=np.copy(data)

    print('input size')
    print(X_test.shape)
    X_test,idx_low,idx_high = change_data_plane(X_test, plane=params['modelParams']['Plane'],return_index=True)

    X_test=X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], n_ch))

    # ============  Evaluating ==================================
    print('-' * 30)
    print('Evaluating %s...'%plane)
    print('-' * 30)
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=0)
    print('Change Plane to %s'%plane)
    y_predict = change_data_plane(y_predict, plane=params['modelParams']['Plane'])
    y_predict=y_predict[idx_low:idx_high, :, :, :]
    print(y_predict.shape)


    return y_predict


def test_localization_model(params,data):
    """Segmentation network for localizing the region of intertest (frontal,axial and sagittal)
    Args:
        params: train parameters of the network
        data: ndarray (int or float) containing the fat image

    Returns:
        out : slices boundaries of the ROI
    """
    # ============  Path Configuration ==================================

    model_name = params['modelParams']['ModelName']
    #model_path = os.path.join(params['modelParams']['SavePath'], model_name)
    model_path = params['modelParams']['SavePath']

    # ============  Model Configuration ==================================
    n_ch = params['modelParams']['nChannels']
    nb_classes = params['modelParams']['nClasses']
    batch_size = params['modelParams']['BatchSize']
    MedBalFactor = params['modelParams']['MedFrequency']
    loss_type = params['modelParams']['Loss_Function']
    sigma = params['modelParams']['GradientSigma']
    plane = params['modelParams']['Plane']
    if plane == 'frontal':
        plane= 'coronal'
    if plane == 'sagital':
        plane = 'sagittal'

    print('-' * 30)
    print('Evaluating %s...'%plane)
    print('-' * 30)
    print('Testing %s'%model_name)
    print('model path')
    print(model_path + '/logs/' + model_name + '_best_weights.h5')

    model = load_model(model_path + '/logs/' + model_name + '_best_weights.h5',
                       custom_objects={'logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_logistic_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'weighted_gradient_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'mixed_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_loss': loss.custom_loss(MedBalFactor, sigma, loss_type),
                                       'dice_coef': loss.dice_coef,
                                       'dice_coef_0': loss.dice_coef_0,
                                       'dice_coef_1': loss.dice_coef_1,
                                       'dice_coef_2': loss.dice_coef_2,
                                       'dice_coef_3': loss.dice_coef_3,
                                       'dice_coef_4': loss.dice_coef_4,
                                       'average_dice_coef': loss.average_dice_coef})

    X_test = np.copy(data)
    print('input size')
    print(X_test.shape)
    X_test, idx_low, idx_high = change_data_plane(X_test, plane=params['modelParams']['Plane'], return_index=True)

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], n_ch))

    # ============  Evaluating ==================================
    print('-' * 30)
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_predict = np.argmax(y_predict, axis=-1)
    print('Change Plane to %s'%plane)
    print(y_predict.shape)
    y_predict = change_data_plane(y_predict, plane=params['modelParams']['Plane'])
    y_predict=y_predict[idx_low:idx_high, :, :]

    print(y_predict.shape)
    high_idx,low_idx=find_unique_index_slice(y_predict)


    return high_idx,low_idx
