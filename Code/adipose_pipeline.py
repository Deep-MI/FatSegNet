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


from __future__ import division
import sys
sys.path.append('./')
sys.path.append('../')

import os

import nibabel as nib
import pandas as pd
from Code.utilities.misc import locate_file
from Code.utilities.visualization_misc import multiview_plotting
from Code.utilities.metrics import calculate_statistics_v2
from Code.utilities.models import run_adipose_localization,run_adipose_segmentation
import numpy as np
from keras import backend as K
from Code.utilities.image_processing import largets_connected_componets,find_labels
from Code.utilities.conform import conform



def clean_segmentations(label_map):

    new_label_map=np.copy(label_map)

    new_label_map= largets_connected_componets(new_label_map)

    return new_label_map


def extreme_AAT_increase_flag(predict_array,threshold=0.3):

    extreme_increase_flag = False
    for slice in range(1,(predict_array.shape[0]-1)) :
        previous_sat =np.sum(predict_array[slice-1,:,:] == 1)
        previous_vat =np.sum(predict_array[slice-1,:,:] == 2)
        current_sat=np.sum(predict_array[slice,:,:] == 1)
        current_vat =np.sum(predict_array[slice,:,:] == 2)
        following_sat=np.sum(predict_array[slice+1,:,:] == 1)
        following_vat=np.sum(predict_array[slice+1,:,:] == 2)

        sat_threshold=current_sat*threshold
        vat_threshold = current_vat * threshold

        if np.abs(current_sat-previous_sat) > sat_threshold or np.abs(current_sat-following_sat) > sat_threshold:
            extreme_increase_flag= 'SAT increase over the threshold'
        elif np.abs(current_vat-previous_vat) > vat_threshold or np.abs(current_vat-following_vat) > vat_threshold:
            extreme_increase_flag = 'VAT increase over the threshold'

    return  extreme_increase_flag

def stats_variable_initialization(nb_comparments,weighted=True):

    # initialize Stats Variables
    variable_columns = []

    volume_variable_columns = ['VOL_cm3', 'SAT_VOL_cm3', 'VAT_VOL_cm3', 'AAT_VOL_cm3',
                               'VAT_VOL_TO_SAT_VOL', 'VAT_VOL_TO_AAT_VOL', 'SAT_VOL_TO_AAT_VOL']

    w_volume_variable_columns= ['W_VOL_cm3','WSAT_VOL_cm3', 'WVAT_VOL_cm3',
                               'WAAT_VOL_cm3', 'WVAT_VOL_TO_WSAT_VOL', 'WVAT_VOL_TO_WAAT_VOL', 'WSAT_VOL_TO_WAAT_VOL']

    area_variable_columns = ['HEIGHT_cm', 'AVG_AREA_cm2', 'AVG_PERIMETER_cm']

    base_variable_len={}
    base_variable_len['Area']=len(area_variable_columns)
    base_variable_len['Volume']=len(volume_variable_columns)
    base_variable_len['W_Volume']=len(w_volume_variable_columns)

    roi_areas = ['wb']
    if nb_comparments != 0:
        # From Feet to Head
        for i in range(int(nb_comparments), 0, -1):
            roi_areas.append('Q' + str(i))

    for roi in roi_areas:
        for area_id in area_variable_columns:
            variable_columns.append(roi + '_' + area_id)
        for vol_id in volume_variable_columns:
            variable_columns.append(roi + '_' + vol_id)

        if weighted:
            for w_vol_id in w_volume_variable_columns:
                variable_columns.append(roi + '_' + w_vol_id)


    variable_columns.insert(0, 'imageid')
    variable_columns.insert(1, '#_Slices')
    variable_columns.insert(2,'FLAGS')

    return variable_columns,base_variable_len

def check_image_contrast(water_array,fat_array):

    slice = fat_array.shape[0] // 2

    water_slice=water_array[slice,20:-20,20:-20]
    fat_slice=fat_array[slice,20:-20,20:-20]

    intensity_max=np.max([np.max(water_slice),np.max(fat_slice)])

    water_slice=water_slice/intensity_max
    fat_slice=fat_slice/intensity_max

    new_fat=np.zeros((fat_slice.shape[0],fat_slice.shape[1]))

    new_fat[fat_slice >= (0.10 * np.max(fat_slice))] = 2
    new_fat[fat_slice >= (0.30*np.max(fat_slice))] = 1

    border_idx=np.where(new_fat == 2)

    point_index=np.arange(0,len(border_idx[0]),10)
    point_y=border_idx[0][point_index]
    point_x=border_idx[1][point_index]

    fat_count=0
    no_fat_count=0
    for j in range(len(point_x)):

        value = fat_slice[point_y[j],point_x[j]] -water_slice[point_y[j],point_x[j]]

        if value < 0 :
            fat_count += 1
        else:
            no_fat_count += 1

    if no_fat_count > fat_count or ((no_fat_count/fat_count) > 0.75):
        FLAG='Check image contrast'
    else:
        FLAG = False

    return FLAG

def check_flags(predicted_array,water_array,fat_array,ratio_vat_sat,threshold=0.30,sat_to_vat_threshold=2.0):
    FLAG = check_image_contrast(water_array,fat_array)

    if FLAG == False:

        FLAG=extreme_AAT_increase_flag(predicted_array,threshold=threshold)

        if ratio_vat_sat > sat_to_vat_threshold:
            FLAG = 'High VAT to SAT ratio'

    return FLAG





def run_adipose_pipeline(args,flags,save_path='/',data_path='/',id='Test'):

    output_stats = 'AAT_stats.tsv'
    output_pred_fat = 'AAT_pred.nii.gz'
    output_pred = 'ALL_pred.nii.gz'
    qc_images = []



    print('-' * 30)
    print(' Loading Subject')
    print(id)
    sub = id


    fat_file = locate_file('*'+str(args.fat_image), data_path)
    water_file = locate_file('*'+str(args.water_image), data_path)



    # Check fat
    if fat_file:
        print('-' * 30)
        print('Loading Fat Image')
        print(fat_file[0])
        #Load Fat Images
        fat_img = nib.load(fat_file[0])
        ishape = fat_img.shape
        if len(ishape) > 3 and ishape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(fat_img.shape[3]) + ') not supported!')
        else:
            fat_img = conform(fat_img, flags=flags, order=args.order, save_path=save_path, mod='fat',
                              axial=args.axial)
            fat_array = fat_img.get_data()
            fat_array = np.swapaxes(fat_array, 0, 2)
            fat_zooms = fat_img.header.get_zooms()

        print('-' * 30)
        print('Loading Water Image')
        #Check water image
        if not water_file:
            weighted=False
            print('No water image found, weighted volumes would not be calculated')
            water_array=np.zeros(fat_array.shape)
        else:
            print(water_file[0])
            weighted=True
            water_img = nib.load(water_file[0])
            ishape = fat_img.shape
            if len(ishape) > 3 and ishape[3] != 1:
                sys.exit('ERROR: Multiple input frames (' + format(water_img.shape[3]) + ') not supported!')
            else:
                water_img = conform(water_img, flags=flags, order=args.order, save_path=save_path, mod='water',
                                   axial=args.axial)
                water_array = water_img.get_data()
                water_array = np.swapaxes(water_array, 0, 2)


        variable_columns, base_variable_len = stats_variable_initialization(args.compartments,weighted)
        ratio_position = variable_columns.index('wb_VAT_VOL_TO_SAT_VOL')

        pixel_matrix = np.zeros((1, len(variable_columns)), dtype=object)
        row_px = 0



        img_spacing=np.copy(fat_zooms)

        if not args.run_stats:

            if args.run_localization:
                high_idx,low_idx=run_adipose_localization(fat_array,flags)
                K.clear_session()
            else:
                high_idx=fat_array.shape[0]
                low_idx= 0

            print('the index values are %d, %d' % (low_idx, high_idx))

            # Image Segmentation
            pred_array=run_adipose_segmentation(fat_array,flags,args)
            K.clear_session()

        else:
            pred_file = locate_file('*AAT_pred.nii.gz', data_path)

            if pred_file :
                pred_img = nib.load(pred_file[0])
                pred_array = pred_img.get_data()
                pred_array = np.swapaxes(pred_array, 0, 2)
                pred_zooms = pred_img.header.get_zooms()
                img_spacing = np.copy(pred_zooms)
                # img_spacing[0] = pred_zooms[2]
                # img_spacing[2] = pred_zooms[0]


                high_idx, low_idx = find_labels(pred_array)
            else :
                print('Subject has no prediction map, a ATT_pred.nii.gz file is required to run the stats option')
                print('-' * 30)
                sys.exit('ERROR: Subject doesnt have a AAT_pred.nii.gz')

        print('-' * 30)
        print('Calculating Stats')

        pred_array[0:low_idx,:,:]=0
        pred_array[high_idx:,:,:]=0

        pred_array [low_idx:high_idx, :, :] = clean_segmentations(pred_array[low_idx:high_idx, :, :])

        pixel_matrix[row_px:row_px + 1, 0] = sub


        pixel_matrix[row_px:row_px + 1, 3:] = calculate_statistics_v2(pred_array[low_idx:high_idx, :, :],
                                                                      water_array[low_idx:high_idx, :, :],
                                                                      fat_array[low_idx:high_idx, :, :],
                                                                      low_idx, high_idx, variable_columns[3:],
                                                                      base_variable_len, img_spacing,
                                                                      args.compartments, weighted=weighted)


        pixel_matrix[row_px:row_px + 1, 1] = int(pixel_matrix[row_px:row_px + 1, 3] / (img_spacing[2] * 0.1))
        pixel_matrix[row_px:row_px + 1, 2] = check_flags(pred_array[low_idx:high_idx, :, :],water_array=water_array,fat_array=fat_array,
                                                         ratio_vat_sat=pixel_matrix[row_px, ratio_position],
                                                         threshold=args.increase_threshold,sat_to_vat_threshold=args.sat_to_vat_threshold)

        df = pd.DataFrame(pixel_matrix[row_px:row_px+1, :], columns=variable_columns)

        if not os.path.isdir(os.path.join(save_path, 'Segmentations')):
            os.mkdir(os.path.join(save_path, 'Segmentations'))

        seg_path=os.path.join(save_path, 'Segmentations')

        df.to_csv(seg_path+'/'+output_stats, sep='\t', index=False)
        df.to_json(seg_path+ '/AAT_variables_summary.json', orient='records')

        row_px += 1

        # Modified images for display
        disp_fat = np.flipud(fat_array[:])
        disp_fat = np.fliplr(disp_fat[:])
        disp_pred=np.copy(pred_array)
        disp_pred = np.flipud(disp_pred)
        disp_pred = np.fliplr(disp_pred)

        #only display SAT and VAT
        disp_pred[disp_pred>=3]=0

        idx = (np.where(disp_pred > 0))
        low_idx = np.min(idx[0])
        high_idx = np.max(idx[0])

        interval = (high_idx - low_idx) // 4

        # Control images of the segmentation
        if not args.control_images:
            if not os.path.isdir(os.path.join(save_path, 'QC')):
                os.mkdir(os.path.join(save_path, 'QC'))
            for i in range(4):
                control_point = [0, int(np.ceil(disp_fat.shape[1] / 2)), int(np.ceil(disp_fat.shape[2] / 2))]
                control_point[0] = int(np.ceil(np.random.uniform(high_idx - interval * i, high_idx - interval * ((i + 1)))))
                multiview_plotting(disp_fat, disp_pred, control_point, save_path+'/QC/QC_%s.png' % i,
                                   classes=5, alpha=0.5, nbviews=3)

        # Save prediction
        pred_array=np.swapaxes(pred_array,2,0)
        pred_img = nib.Nifti1Image(pred_array, fat_img.affine, fat_img.header)
        nib.save(pred_img, seg_path+'/'+output_pred)

        pred_array[pred_array>=3]=0
        pred_img = nib.Nifti1Image(pred_array, fat_img.affine, fat_img.header)
        nib.save(pred_img, seg_path+'/'+output_pred_fat)

    else:
        print('')
        print('-' * 30)

        sys.exit('ERROR: Subject doesnt have a Fat Image , Please verified input volume name')


    print('-' * 30)

    print('Finish Subject %s'%sub)

    print('-' * 30)
