import argparse
import sys
import os
import time

sys.path.append('./')

from Code.adipose_pipeline import run_adipose_pipeline
from Code.utilities.misc import locate_file,locate_dir
import pandas as pd
import numpy as np


def check_paths(save_folder,subject_id,flags):

    save_path=os.path.join(flags['output_path'],save_folder)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not os.path.isdir(os.path.join(save_path,subject_id)):
        os.mkdir(os.path.join(save_path,subject_id))

    final_path = os.path.join(save_path,subject_id)
    return final_path


class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



def option_parse():


    parser = argparse.ArgumentParser(
        description='Adipose Pipeline to segment the  abdominal adipose tissue into VAT and SAT. '
                    'Each subject should have a independent folder with the water and fat images. '
                    'Input images have to be nifti files and should be named consistently in all subjects. '
                    'The Output path is define by the user ; all the outputs from the pipeline will be store under $output_path/$subject_id. '
                    'The predicted segmentation mask is save under ($AAT_pred) and all the statistics under $ATT_stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", help="csv file containing the subjects to process, the csv file should be order as follow : $subject_id,$subject_path", required=False,default='participants.csv')
    parser.add_argument("-outp", "--output_folder",
                        help="Main folder where the variables and control images are going to be store", required=False, default='')

    parser.add_argument("-fat", "--fat_image", type=str, help="Name of the fat image", required=False,
                        default='FatImaging_F.nii.gz')
    parser.add_argument("-water", "--water_image", type=str, help="Name of the water image", required=False,
                        default='FatImaging_W.nii.gz')

    parser.add_argument('-No_QC',"--control_images",action='store_true',help='Plot subjects prediction for visual quality control',required=False,default=False)

    parser.add_argument('-loc', "--run_localization", action='store_true',
                        help='run abdominal region localization model ', required=False, default=False)

    parser.add_argument('-axial', "--axial", action='store_true',
                        help='run only axial model ', required=False, default=False)

    parser.add_argument('-order', "--order", type=int,
                        help='interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic) ', required=False, default=1)

    parser.add_argument('-comp', "--compartments", type=int,
                        help='Number of equal compartments to run the analysis, by default the whole region(wb) is calculated', required=False, default=0)

    parser.add_argument('-AAT', "--increase_threshold", type=float,
                        help='Warning flag for an increase in AAT over threhold between consecutive scans', required=False, default=0.4)

    parser.add_argument('-ratio', "--sat_to_vat_threshold", type=float,
                        help='Warning flag for a high vat to sat ratio', required=False, default=2.0)

    parser.add_argument('-stats', "--run_stats", action='store_true',
                        help='run only stats , segmentation map required ', required=False, default=False)

    parser.add_argument('-gpu_id', "--gpu_id", type=int,
                        help='if using gpu, please give the gpu device name', required=False, default=0)




    args = parser.parse_args()

    FLAGS = {}
    FLAGS['multiviewModel'] = '/tool/Adipose_Seg_Models/Segmentation/'
    FLAGS['singleViewModels'] = '/tool/Adipose_Seg_Models/Segmentation/'
    FLAGS['localizationModels'] = '/tool/Adipose_Seg_Models/Localization/'
    FLAGS['input_path']='/tool/Data'
    FLAGS['output_path']='/tool/Output'
    FLAGS['imgSize'] = [256, 224, 72]
    FLAGS['spacing'] = [1.9531, 1.9531, 5.0]
    FLAGS['base_ornt'] = np.array([[0, -1], [1, 1], [2, 1]])
    #FLAGS['compartments']=0
    #control_images = True


    return args,FLAGS


def run_fatsegnet(args,FLAGS):

    # load file
    participant_file=locate_file('*'+args.file,FLAGS['input_path'])
    if participant_file:
        print(participant_file[0])
        df =pd.read_csv(participant_file[0],header=None)
        if df.empty:
            print('Participant file empty ')
        else:
            file_list=df.values
            for sub in file_list:
                id=sub[0]
                path = locate_dir('*'+str(id)+'*',FLAGS['input_path'])
                if path:
                    if os.path.isdir(path[0]):

                        start = time.time()

                        save_path = check_paths(save_folder=args.output_folder, subject_id=str(id),flags=FLAGS)

                        sys.stdout= Transcript(filename=save_path + '/temp.log')

                        run_adipose_pipeline(args=args, flags=FLAGS, save_path=save_path,data_path=path[0],id=str(id))

                        end = time.time() - start

                        print("Total time for computation of segmentation is %0.4f seconds."%end)

                        sys.stdout.logfile.close()
                        sys.stdout = sys.stdout.terminal
                    else:
                        print ('Directory %s not found'%path)
                else :
                    print('Directory name %s not found' % id)
    else:
        print('No partipant file found, please provide one the input data folder')



if __name__=='__main__':


    args,FLAGS= option_parse()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id);

    run_fatsegnet(args,FLAGS)

    sys.exit(0)

