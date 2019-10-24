import numpy as np
import nibabel as nib
import sys
import scipy.ndimage
import os



def calculated_new_ornt(iornt,base_ornt):

    new_iornt=iornt[:]

    for axno, direction in np.asarray(base_ornt):
        idx=np.where(iornt[:,0] == axno)
        idirection=iornt[int(idx[0][0]),1]
        if direction == idirection:
            new_iornt[int(idx[0][0]), 1] = 1.0
        else:
            new_iornt[int(idx[0][0]), 1] = -1.0

    return new_iornt

def check_orientation(img,base_ornt=np.array([[0,-1],[1,1],[2,1]])):

    iornt=nib.io_orientation(img.affine)

    if not np.array_equal(iornt,base_ornt):
        img = img.as_reoriented(calculated_new_ornt(iornt,base_ornt))

    return img


def resample(image, spacing, new_spacing=[1, 1, 1],order=1,prefilter=True):
    # Determine current pixel spacing

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image,real_resize_factor,order=order,prefilter=prefilter)

    return image, new_spacing


def define_size(mov_dim,ref_dim):
    new_dim=np.zeros(len(mov_dim),dtype=np.int)
    borders=np.zeros((len(mov_dim),2),dtype=int)

    padd = [int(mov_dim[0] // 2), int(mov_dim[1] // 2), int(mov_dim[2] // 2)]

    for i in range(len(mov_dim)):
        new_dim[i]=int(max(2*mov_dim[i],2*ref_dim[i]))
        borders[i,0]= int(new_dim[i] // 2) -padd [i]
        borders[i,1]= borders[i,0] +mov_dim[i]

    return list(new_dim),borders

def map_size(arr,base_shape,axial):

    if axial:
        base_shape[2]=arr.shape[2]

    new_shape,borders=define_size(np.array(arr.shape),np.array(base_shape))
    new_arr=np.zeros(new_shape)
    final_arr=np.zeros(base_shape)

    new_arr[borders[0,0]:borders[0,1],borders[1,0]:borders[1,1],borders[2,0]:borders[2,1]]= arr[:]

    middle_point = [int(new_arr.shape[0] // 2), int(new_arr.shape[1] // 2), int(new_arr.shape[2] // 2)]
    padd = [int(base_shape[0]/2), int(base_shape[1]/2), int(base_shape[2]/2)]

    low_border=np.array((np.array(middle_point)-np.array(padd)),dtype=int)
    high_border=np.array(np.array(low_border)+np.array(base_shape),dtype=int)

    final_arr[:,:,:]= new_arr[low_border[0]:high_border[0],
                   low_border[1]:high_border[1],
                   low_border[2]:high_border[2]]


    # final_arr[:,:,:]= new_arr[middle_point[0]-padd[0]:middle_point[0]+padd[0],
    #                middle_point[1] - padd[1]:middle_point[1] + padd[1],
    #                middle_point[2] - padd[2]:middle_point[2] + padd[2]]

    return final_arr





def map_image(img_arr,base_zoom,izoom,order,axial):
    if not axial:

        resample_arr, izoom= resample(img_arr, spacing=np.array(izoom),
                                             new_spacing=np.array(base_zoom), order=order)

        resample_arr[resample_arr < 0] = 0


    else:
        print('Fat Segmentation will be done in the axial plane only')
        aux_zoom = base_zoom[:]
        aux_zoom[2] = izoom[2]

        resample_arr, izoom = resample(img_arr, spacing=np.array(izoom),
                                             new_spacing=np.array(aux_zoom), order=order)

        resample_arr[resample_arr < 0] = 0

    return resample_arr,izoom



def conform(img,flags,order,save_path,mod,axial=False):
    """
    Args:
        img: nibabel img: Loaded source image
        flags: dict : Dictionary containing the image size, spacing and orientation
        order: int : interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    Returns:
        new_img: nibabel img : conformed nibabel image
    """
    save=False
    # check orientation LAS
    img=check_orientation(img,base_ornt=flags['base_ornt'])

    img_arr=img.get_data()
    img_header = img.header

    # check voxel sizer
    izoom=img.header.get_zooms()
    if int(izoom[0]) != int(flags['spacing'][0]) or int(izoom[1]) != int(flags['spacing'][1]) or int(izoom[2]) != int(flags['spacing'][2]):
        img_arr, izoom= map_image(img_arr, flags['spacing'], izoom, order,axial)
        save=True

    ishape = img_arr.shape
    # check dimensions
    if int(ishape[0]) != int(flags['imgSize'][0]) or int(ishape[1]) != int(flags['imgSize'][1]) or int(ishape[2]) != int(flags['imgSize'][2]):
        img_arr=map_size(img_arr,flags['imgSize'],axial)
        save = True

    img_header.set_data_shape(img_arr.shape)
    img_header.set_zooms(izoom)

    new_img = nib.Nifti1Image(img_arr, img.affine, img_header)
    #save images if modified

    if save:

        if not os.path.isdir(os.path.join(save_path, 'MRI')):
            os.mkdir(os.path.join(save_path, 'MRI'))

        mri_path = os.path.join(save_path, 'MRI')

        if mod == 'fat':
            new_img_path = os.path.join(mri_path, 'FatImaging_F.nii.gz')
        else:
            new_img_path = os.path.join(mri_path, 'FatImaging_W.nii.gz')

        nib.save(new_img, new_img_path)

    return new_img

# FLAGS = {}
# FLAGS['multiviewModel'] = '/tool/Adipose_Seg_Models/Segmentation/'
# FLAGS['singleViewModels'] = '/tool/Adipose_Seg_Models/Segmentation/'
# FLAGS['localizationModels'] = '/tool/Adipose_Seg_Models/Localization/'
# FLAGS['input_path']='/tool/Data'
# FLAGS['output_path']='/tool/Output'
# FLAGS['imgSize'] = [256, 224, 72]
# FLAGS['spacing'] = [1.9531, 1.9531, 5.0]
# FLAGS['base_ornt'] = np.array([[0, -1], [1, 1], [2, 1]])
#
# path='/home/estradae/Downloads/100006/100006_3D_GRE_TRA_F/3D_GRE_TRA_F_3D_GRE_TRA_2.nii.gz'
# #path='/home/estradae/Downloads/Rotterdam/1/ab_fat.nii.gz'
# #path='/localmount/volume1/users/estradae/Fatimaging/scans/0e499732-aed3-4623-b976-97685cae0c59/FatImaging_F.nii.gz'
# img=nib.load(path)
# new_img=conform(img,FLAGS,order=3,save_path='/home/estradae/Downloads',mod='fat',axial=True)