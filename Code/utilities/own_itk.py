# -*- coding: UTF-8 -*-
"""Module containing functions enabling to read, make and
write ITK images.

"""


import os
import numpy as np
import SimpleITK as itk
import nibabel as nib

def make_nibabel_image(arr, proto_image=None):
    """Create an nibabel image given an image array.

    Parameters
    ----------
    arr : ndarray
        Array to create an itk image with.
    proto_image : nibabel image, optional
        Proto  image to provide header and affine transfomation.

    Returns
    -------
    image : Nibabel image
        The Nibabel image containing the input array `arr`.

    """
    if proto_image != None:
        clipped_img = nib.Nifti1Image(arr, proto_image.affine, proto_image.header)
    else:
        empty_header = nib.Nifti1Header()
        clipped_img = nib.Nifti1Image(arr, np.eye(4), empty_header)

    return clipped_img

def make_itk_image(arr, proto_image=None):
    """Create an itk image given an image array.

    Parameters
    ----------
    arr : ndarray
        Array to create an itk image with.
    proto_image : itk image, optional
        Proto itk image to provide Origin, Spacing and Direction.

    Returns
    -------
    image : itk image
        The itk image containing the input array `arr`.

    """
    image = itk.GetImageFromArray(arr)
    if proto_image != None:
        image.CopyInformation(proto_image)

    return image

def write_itk_image(image, path):
    """Write an itk image to a path.

    Parameters
    ----------
    image : itk image or np.ndarray
        Image to be written.
    path : str
        Path where the image should be written to.

    """

    if isinstance(image, np.ndarray):
        image = make_itk_image(image)

    writer = itk.ImageFileWriter()
    writer.SetFileName(path)

    if os.path.splitext(path)[1] == '.nii':
        Warning('You are converting nii, ' + \
                'be careful with type conversions')

    print(path)
    writer.Execute(image)

def get_itk_image(path):
    """Get an itk image given a path.

    Parameters
    ----------
    path : str
        Path pointing to an image file with extension among
        *TIFF, JPEG, PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze,
        SDT/SPR (Stimulate), Nrrd or VTK images*.

    Returns
    -------
    image : itk image
        The itk image.

    """
    if not os.path.exists(path):
        err = path + ' doesnt exist'
        raise AttributeError(err)

    reader = itk.ImageFileReader()
    reader.SetFileName(path)

    image = reader.Execute()

    return image

def get_itk_array(path_or_image):
    """ Get an image array given a path or itk image.

    Parameters
    ----------
    path_or_image : str or itk image
        Path pointing to an image file with extension among
        *TIFF, JPEG, PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze,
        SDT/SPR (Stimulate), Nrrd or VTK images* or an itk image.

    Returns
    -------
    arr : ndarray
        Image ndarray contained in the given path or the itk image.

    """

    if isinstance(path_or_image, np.ndarray):
        return path_or_image

    elif isinstance(path_or_image, str):
        image = get_itk_image(path_or_image)

    elif isinstance(path_or_image, itk.Image):
        image = path_or_image

    else:
        err = 'Image type not recognized: ' + path_or_image
        raise RuntimeError(err)

    arr = itk.GetArrayFromImage(image)

    return arr

def set_extension(path, extension):

    if extension[0] != '.':
        extension = '.' + extension

    wanted_filename = os.path.splitext(path)[0] + extension

    return wanted_filename

def get_itk_data(path_or_image, verbose=False):
    """Get the image array, image size and pixel dimensions given an itk
    image or a path.

    Parameters
    ----------
    path_or_image : str or itk image
        Path pointing to an image file with extension among
        *TIFF, JPEG, PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze,
        SDT/SPR (Stimulate), Nrrd or VTK images* or an itk image.
    verbose : boolean, optional
        If true, print image shape, spacing and data type of the image
        corresponding to `path_or_image.`

    Returns
    -------
    arr : ndarray
        Image array contained in the given path or the itk image.
    shape : tuple
        Shape of the image array contained in the given path or the itk
        image.
    spacing : tuple
        Pixel spacing (resolution) of the image array contained in the
        given path or the itk image.

    """

    if isinstance(path_or_image, str):
        image = get_itk_image(path_or_image)
    else:
        image = path_or_image

    arr = itk.GetArrayFromImage(image)
    shape = arr.shape
    spacing = image.GetSpacing()[::-1]
    data_type = arr.dtype

    if verbose:

        print('\t image shape: ' + str(shape))
        print('\t image spacing: ' + str(spacing))
        print('\t image data type: ' + str(data_type))

    return arr, shape, spacing

def get_spacing(path_or_image):

    if isinstance(path_or_image, str):
        image = get_itk_image(path_or_image)
    else:
        image = path_or_image

    spacing = image.GetSpacing()[::-1]

    return spacing

def set_spacing(path_or_image, spacing):

    if isinstance(path_or_image, str):
        image = get_itk_image(path_or_image)
    else:
        image = path_or_image

    image.SetSpacing(spacing[::-1])

    return image

"""
Convert images to other formats.
"""

def convert_to_nii(filenames):
    '''Convert image files to nifti image files.'''

    for filename in filenames:
        image = get_itk_image(filename)
        nii_filename = os.path.splitext(filename)[0] + '.nii'
        write_itk_image(image, nii_filename)

def convert_dicom(source_path, save_path):
    '''Converts dicom series to image format specified in path.

    Parameters
    ----------
    source_path : str
        path to dicom series.
    path : str
        path to save new image (extension determines image format)

    '''

    image = read_dicom(source_path, verbose=True)

    write_itk_image(image, save_path)

def read_dicom(source_path, verbose=True):
    '''Reads dicom series.

    Parameters
    ----------
    source_path : string
        path to dicom series.
    verbose : boolean
        print out all series file names.

    Returns
    -------
    image : itk image
        image volume.
    '''

    reader = itk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(source_path)
    if len(names) < 1:
        raise IOError('No Series can be found at the specified path!')
    elif verbose:
        print('image series found in :\n\t %s' % source_path)
#         print names
    reader.SetFileNames(names)
    image = reader.Execute()
    if verbose:
        get_itk_data(image, verbose=True)

    return image

