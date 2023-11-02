import nibabel as nib
import os
from data_curation.helper_functions import move_smallest_axis_to_z, swap_to_original_axis
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist
from utils.read_write_data import save_nifti
import numpy as np

if __name__ == "__main__":
    subject_folder = '/home/bella/Phd/code/code_bella/log/201/output/HASTE_autoscale/test/Pat557_Se11_Res0.44921875_0.44921875_Spac3.0/'
    reference_subject = '/home/bella/Phd/code/code_bella/log/201/output/HASTE_autoscale/test/Pat454_Se10_Res1.25_1.25_Spac2.2000000476837003/'
    filename = 'data.nii.gz'

    input = nib.load(os.path.join(subject_folder, filename)).get_data().astype(int)
    #reference = nib.load(os.path.join(reference_subject, filename)).get_data()

    input, swap_axis = move_smallest_axis_to_z(input)
    input_adapthist = np.empty(input.shape)
    # reference, swap_axis = move_smallest_axis_to_z(reference)
    # reference = resize(reference, input.shape)
    for i in range(0,input.shape[2]):
        input_adapthist[:,:,i] = equalize_adapthist(input[:,:,i])

    input_adapthist = swap_to_original_axis(swap_axis, input_adapthist)
    save_nifti(input_adapthist, os.path.join(subject_folder, 'data_adapthist.nii.gz'))


