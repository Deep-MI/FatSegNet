
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
from skimage.measure import perimeter

def dice(predictions, labels, num_classes):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.
    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for
    Returns:
        np.ndarray: dice coefficient per class
    """
    dice_scores = np.zeros((num_classes))
    for i in range(num_classes):
        tmp_den = (np.sum(predictions == i) + np.sum(labels == i))
        tmp_dice = 2. * np.sum((predictions == i) * (labels == i)) / \
            tmp_den if tmp_den > 0 else 1.
        dice_scores[i] = tmp_dice
    return dice_scores.astype(np.dtype(float).type)



def perimeter_calculation(label_mask):

    perimeter_val=[]

    for slice in range(label_mask.shape[0]):
        perimeter_val.append(perimeter(label_mask[slice,:,:]))

    average_perimeter = np.sum(perimeter_val)/label_mask.shape[0]

    return  average_perimeter

def calculate_areas(final_img, img_spacing,columns):

    if len(final_img.shape) == 2:
        final_img =np.reshape(final_img,(1,final_img.shape[0],final_img.shape[1]))


    pixel_area = (img_spacing[0] * img_spacing[1]) * 0.01

    statiscs_matrix = np.zeros(( 1, columns),dtype=np.float64)

    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    #  Metric Measurements
    statiscs_matrix[0,0]= final_img.shape[0] * img_spacing[2] * 0.1 # Height ROI
    statiscs_matrix[0,1]= (np.sum(abdominal_region_mask) * pixel_area) / final_img.shape[0] #Average_Area
    statiscs_matrix[0, 2] = perimeter_calculation(abdominal_region_mask) * img_spacing[0] * 0.1  # Average_perimeter

    return  statiscs_matrix.round(decimals=4)

def calculate_volumes(final_img, water_array, fat_array, img_spacing,columns,weighted=True):

    if len(final_img.shape) == 2:
        final_img =np.reshape(final_img,(1,final_img.shape[0],final_img.shape[1]))
        water_array = np.reshape(water_array, (1, water_array.shape[0], water_array.shape[1]))
        fat_array = np.reshape(water_array, (1, fat_array.shape[0], fat_array.shape[1]))

    voxel_volume = (img_spacing[0] * img_spacing[1] * img_spacing[2]) * 0.001



    abdominal_region_mask= np.zeros(final_img.shape,dtype=bool)
    abdominal_region_mask[final_img >= 1] = True

    vat_mask = np.zeros(final_img.shape, dtype=bool)
    vat_mask[final_img == 2] = True

    sat_mask = np.zeros(final_img.shape, dtype=bool)
    sat_mask[final_img == 1] = True

    combine_array = water_array + fat_array
    fat_fraction_array = np.clip(fat_array,0.00001,None) / np.clip(combine_array,0.00001,None)

    if weighted:
        vat_fraction = np.sum(fat_fraction_array[vat_mask])
        sat_fraction = np.sum(fat_fraction_array[sat_mask])
        abdominal_region_fraction= np.sum (fat_fraction_array[abdominal_region_mask])
    else:
        sat_fraction = np.sum(sat_mask)
        vat_fraction = np.sum(vat_mask)
        abdominal_region_fraction= np.sum (abdominal_region_mask)

    #print('the vat fraction values are %d, the sat fraction values are %d' % (vat_fraction, sat_fraction))

    statiscs_matrix = np.zeros(( 1, columns),dtype=np.float64)

    #  Metric Measurements
    statiscs_matrix[0,0] = abdominal_region_fraction * voxel_volume # Volume of Abdominal Region

    # Pixel not Weighted
    statiscs_matrix[0, 1] = sat_fraction * voxel_volume  # VOL_SAT
    statiscs_matrix[0, 2] = vat_fraction * voxel_volume  # VOL_VAT
    statiscs_matrix[0, 3] = statiscs_matrix[0, 1] + statiscs_matrix[0, 2]  # VOL_AAT

    statiscs_matrix[0, 4] = statiscs_matrix[0,2] / statiscs_matrix[0, 1]  # VAT/SAT
    statiscs_matrix[0, 5] = statiscs_matrix[0, 2] / statiscs_matrix[0, 3]  # VAT/AAT
    statiscs_matrix[0, 6] = statiscs_matrix[0, 1] / statiscs_matrix[0, 3]  # SAT/AAT

    #statiscs_matrix[0,17]=extreme_AAT_increase_flag(final_img,threshold=increase_thr)

    return statiscs_matrix.round(decimals=4)

def calculate_statistics_v2(final_img, water_array, fat_array, low_idx, high_idx, columns,base_variables_len,img_spacing,comparments=0,weighted=True):
    """
    Rhineland Stuty version
    :param final_img:
    :param water_array:
    :param fat_array:
    :param low_idx:
    :param high_idx:
    :param columns:
    :param base_variables_len:
    :param img_spacing:
    :param increase_thr:
    :param comparments:
    :param weighted:
    :return:
    """


    statiscs_matrix = np.zeros((1, len(columns)),dtype=object)
    size_base=base_variables_len['Area']+base_variables_len['Volume']+base_variables_len['W_Volume']


    #print('Whole Body')

    final_area=base_variables_len['Area']
    statiscs_matrix[0, 0:final_area]=calculate_areas(final_img,img_spacing,base_variables_len['Area'])
    final_volume=final_area +base_variables_len['Volume']
    statiscs_matrix[0,final_area:final_volume]=calculate_volumes(final_img,water_array,fat_array,
                                                                              img_spacing, base_variables_len['Volume'], weighted=False)


    if weighted:
        final_volume2= final_volume +base_variables_len['Volume']
        statiscs_matrix[0,final_volume:final_volume2] = calculate_volumes(final_img,water_array,fat_array,img_spacing,
                                                                                                          base_variables_len['Volume'],
                                                                                                          weighted=True)

    if comparments !=0:

        interval = (high_idx - low_idx)
        interval_step = np.around((interval / comparments), decimals=2)
        interval_steps = np.arange(0, interval, interval_step).round(decimals=2)
        if not interval_steps[-1] == interval:
            interval_steps = np.append(interval_steps, interval)

        slice=0

        for i in np.arange(0,comparments):
            lower_limit=np.ceil(interval_steps[i])
            higher_limit=np.floor(interval_steps[i+1])
            complete_slices=np.arange(lower_limit,higher_limit)
            #print (complete_slices)
            #Calculate Complete Slices
            if complete_slices.size != 0 :
                min_slice=int(np.min(complete_slices))
                max_slice=int(np.max(complete_slices))+1


                area_initial_len= size_base * (i+1)
                area_final_len =size_base * (i+1)+base_variables_len['Area']

                #TO-DO check that is empty
                if statiscs_matrix[0, area_initial_len + 1] != 0:

                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0, area_initial_len:area_final_len] + calculate_areas(final_img[min_slice:max_slice, :, :],img_spacing,base_variables_len['Area'])
                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + calculate_areas(final_img[min_slice:max_slice, :, :], img_spacing, base_variables_len['Area'])

                vol_final_len= area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] =  statiscs_matrix[0, area_final_len:vol_final_len] + calculate_volumes(final_img[min_slice:max_slice, :, :], water_array[min_slice:max_slice, :, :],
                                                                                                                                         fat_array[min_slice:max_slice, :, :],
                                                                                                                                         img_spacing, base_variables_len['Volume'],weighted=False)
                if weighted:
                    vol2_final_len=vol_final_len +base_variables_len['W_Volume']

                    statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + calculate_volumes(final_img[min_slice:max_slice, :, :], water_array[min_slice:max_slice, :, :],
                                                                                                                                           fat_array[min_slice:max_slice, :, :],
                                                                                                                                       img_spacing, base_variables_len['Volume'], weighted=True)
            residual=np.around(interval_steps[i+1]-int(interval_steps[i+1]),decimals=2)
            if residual !=0 :
                slice=int(np.floor(interval_steps[i+1]))

                area_stats= calculate_areas(final_img[slice, :, :], img_spacing, base_variables_len['Area'])
                volume_stats= calculate_volumes(final_img[slice, :, :], water_array[slice, :, :],fat_array[slice, :, :],img_spacing, base_variables_len['Volume'], weighted=False)
                weighted_volume_stats = calculate_volumes(final_img[slice, :, :], water_array[slice, :, :],fat_array[slice, :, :], img_spacing, base_variables_len['Volume'],weighted=True)

                area_initial_len = size_base * (i + 1)
                area_final_len = size_base * (i + 1) + base_variables_len['Area']
                areas_residual=np.ones(base_variables_len['Area'])
                areas_residual[0]=residual

                if statiscs_matrix[0,area_initial_len+1] != 0:

                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats


                vol_final_len = area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] = statiscs_matrix[0,area_final_len:vol_final_len] + residual * volume_stats

                if weighted:
                    vol2_final_len = vol_final_len + base_variables_len['W_Volume']

                    statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + residual * weighted_volume_stats


                residual_next_compartment=np.around(np.ceil(interval_steps[i+1])-interval_steps[i+1],decimals=2)


                area_initial_len = size_base * (i + 2)
                area_final_len = size_base * (i + 2) + base_variables_len['Area']
                areas_residual=np.ones(base_variables_len['Area'])
                areas_residual[0]=residual_next_compartment

                if statiscs_matrix[0, area_initial_len + 1] != 0:
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                    statiscs_matrix[0,area_initial_len+1] = statiscs_matrix[0,area_initial_len+1] / 2
                    statiscs_matrix[0, area_initial_len + 2] = statiscs_matrix[0, area_initial_len + 2] / 2

                else :
                    statiscs_matrix[0, area_initial_len:area_final_len] = statiscs_matrix[0,area_initial_len:area_final_len] + areas_residual * area_stats

                vol_final_len = area_final_len + base_variables_len['Volume']

                statiscs_matrix[0, area_final_len:vol_final_len] = statiscs_matrix[0,area_final_len:vol_final_len] + residual_next_compartment * volume_stats

                vol2_final_len = vol_final_len + base_variables_len['W_Volume']

                statiscs_matrix[0, vol_final_len:vol2_final_len] = statiscs_matrix[0,vol_final_len:vol2_final_len] + residual_next_compartment * weighted_volume_stats

    return statiscs_matrix



def calculate_statistics(final_img, water_array, fat_array, columns,base_variables_len,img_spacing,weighted=True):
    """
    Release version
    :param final_img:
    :param water_array:
    :param fat_array:
    :param low_idx:
    :param high_idx:
    :param columns:
    :param base_variables_len:
    :param img_spacing:
    :param increase_thr:
    :param comparments:
    :param weighted:
    :return:
    """
    print('-' * 30)
    print('Calculating Variables')
    print('-' * 30)

    statiscs_matrix = np.zeros((1, len(columns)),dtype=object)
    size_base=base_variables_len['Area']+base_variables_len['Volume']+base_variables_len['W_Volume']


    #print('Whole Body')

    final_area=base_variables_len['Area']
    statiscs_matrix[0, 0:final_area]=calculate_areas(final_img,img_spacing,base_variables_len['Area'])
    final_volume=final_area +base_variables_len['Volume']
    statiscs_matrix[0,final_area:final_volume]=calculate_volumes(final_img,water_array,fat_array,
                                                                              img_spacing, base_variables_len['Volume'], weighted=False)


    if weighted:
        final_volume2= final_volume +base_variables_len['Volume']
        statiscs_matrix[0,final_volume:final_volume2] = calculate_volumes(final_img,water_array,fat_array,img_spacing,
                                                                                                          base_variables_len['Volume'],
                                                                                                          weighted=True)
    return statiscs_matrix
