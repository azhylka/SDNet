'''
A collection of functions to be used for preprocessing, and manipulating resetting the HCP directories and testing 
that preprocessing has been run correctly. 

    fully_sampled_FOD - Calculate the fully sampled FODs
    
    undersampled_FOD - Undersampled the data (data_denoised.nii.gz, bvecs and bvals) and calculate the subsequent FODs. 
    
    fixels_and_mask - Populating the fixel directry for the fully sampled FODs and the masks which will be used to evaluate performance
    as well as ensure only white matter and grey matter voxels are used in training. Tractseg is used in this 
    function. 

    reset_HCP_dir - Removes the files and directories that are created in the HCP subject's folder due to preprocessing. 

    HCP_download_test - Tests that all of the files that are required to perform preprocessing can be found in the directory.

    preprocessing_test - Check that preprocessing has een performed correctly and that the data necessary to run 
    training and test are available. 
'''
import sys 
import os
sys.path.append(os.path.join(sys.path[0],'..'))
import preprocessing.dwi_undersample as dwiusamp
import preprocessing.fixel_threshold as fixel_threshold

import subprocess 
import os
import shutil
import concurrent.futures
import logging

# module-level logger (configured at runtime via setup_logger)
logger = None

def fully_sampled_FOD(path):
    # Fully sampled FOD
    # subprocess.run(['dwi2response', 'dhollander', os.path.join(path, 'data_denoised.nii.gz'), 
    #                 os.path.join(path, 'wm_response.txt'), os.path.join(path, 'gm_response.txt'), 
    #                 os.path.join(path, 'csf_response.txt'), '-fslgrad', 
    #                 os.path.join(path, 'bvecs'), os.path.join(path, 'bvals'),
    #                 '-nthreads', '9', '-force'])
    
    
    # subprocess.run(['dwi2fod', '-fslgrad', os.path.join(path, 'bvecs'), os.path.join(path, 'bvals'),
    #                 'msmt_csd', os.path.join(path, 'data_denoised.nii.gz'), os.path.join(path, 'wm_response.txt'),
    #                 os.path.join(path, 'wm.nii.gz'), os.path.join(path, 'gm_response.txt'), os.path.join(path, 'gm.nii.gz'), 
    #                 os.path.join(path, 'csf_response.txt'), os.path.join(path, 'csf.nii.gz'),
    #                 '-nthreads', '9', '-force'])

    subprocess.run(['mrcat', '-axis', '3', os.path.join(path, 'wm.nii.gz'), os.path.join(path, 'gm.nii.gz'), os.path.join(path, 'csf.nii.gz'), os.path.join(path, 'gt_fod.nii.gz')])

    return 0


def setup_logger(base_dir=None, log_name='preprocess_parallel.log', level=logging.INFO):
    """Configure and return a named logger.

    This is idempotent: calling it multiple times won't add duplicate handlers.
    If base_dir is provided, a FileHandler writing to base_dir/log_name is added.
    A StreamHandler is also added so messages appear on stdout.
    """
    global logger
    logger = logging.getLogger('sdnet.preprocess')
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s %(levelname)s [pid:%(process)d] %(name)s: %(message)s')

    if base_dir is not None:
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            # ignore directory creation errors; FileHandler will raise if necessary
            pass
        log_path = os.path.join(base_dir, log_name)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

def undersampled_FOD(path, usamp_folder_name = 'undersampled_fod', sampling_pattern = [3,9,9,9], bval_samples = [1000, 2000, 3000]):
    # If the undersampled_fod directory doesn't exist, make it
    if os.path.exists(os.path.join(path, usamp_folder_name)) == False:
        os.mkdir(os.path.join(path, usamp_folder_name))

    UspDset = dwiusamp.UndersampleDataset(path, os.path.join(path, usamp_folder_name), 
                                          sampling_pattern = sampling_pattern, bval_samples = bval_samples)
    UspDset.all_save()

    #Normalising the already undersampled data so the maximum value is 1.
    subprocess.run(['dwinormalise', 'individual', os.path.join(path, usamp_folder_name, 'data_denoised.nii.gz'), 
                    os.path.join(path, '../T1_stripped_bet.nii.gz'), os.path.join(path, usamp_folder_name, 'normalised_data_denoised.nii.gz'), 
                    '-fslgrad', os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals'),
                    '-intensity', '1', '-nthreads', '9', '-force'])
    
    # Calculating the undersampled FODs
    subprocess.run(['dwi2response', 'dhollander', os.path.join(path, usamp_folder_name, 'normalised_data_denoised.nii.gz'), os.path.join(path, usamp_folder_name, 'wm_response.txt'),
                    os.path.join(path, usamp_folder_name, 'gm_response.txt'), os.path.join(path, usamp_folder_name, 'csf_response.txt'), '-fslgrad',
                    os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals'),
                    '-nthreads', '9', '-force'])
    
    subprocess.run(['dwi2fod', '-fslgrad', os.path.join(path, usamp_folder_name, 'bvecs'), os.path.join(path, usamp_folder_name, 'bvals'),
                 'msmt_csd', os.path.join(path, usamp_folder_name, 'normalised_data_denoised.nii.gz'), os.path.join(path, usamp_folder_name, 'wm_response.txt'),
                 os.path.join(path, usamp_folder_name, 'wm.nii.gz'), os.path.join(path, usamp_folder_name, 'gm_response.txt'), os.path.join(path, usamp_folder_name, 'gm.nii.gz'),
                 os.path.join(path, usamp_folder_name, 'csf_response.txt'), os.path.join(path, usamp_folder_name, 'csf.nii.gz'),
                 '-nthreads', '9', '-force'])
    
    return 0

def T1w_processing(path:str):
    """Processing the T1w image to produce the 5ttgen mask and the white_matter_mask

    This function can be called regardless of whether the size of the T1w image matches 
    the size of the diffusion data. The images do however need to be co-registered. To run 
    SDNet only the 5ttgen and white_matter_mask images are required.

    Args:
        path (_type_): Path to the diffusion directory that is going to be processed.
    """    
    subprocess.run(['5ttgen', 'freesurfer', os.path.join(path, '..', 'aparc+aseg.nii.gz'), 
                    os.path.join(path, '..', '5ttgen_highres.nii.gz'), '-nocrop', '-force'])
    
    subprocess.run(['mrgrid', '-template', os.path.join(path, 'data_denoised.nii.gz'),
                os.path.join(path, '..', '5ttgen_highres.nii.gz'), 'regrid',
                os.path.join(path, '..', '5ttgen.nii.gz'), '-force'])

    subprocess.run(['mrconvert', os.path.join(path, '..', '5ttgen.nii.gz'),
                     '-coord', '3', '2', os.path.join(path,'..','white_matter_mask.nii.gz'), '-force'])

def fixels_and_masks(path):
    
    def mif_to_nifti(mif_path):
        '''
        Converts a mif file at location mif_path (MRtrix3 native file type) to a nifti file
        in the same location and deletes the mif file.
        '''
        assert os.path.exists(mif_path), f"The mif file {mif_path} doesn't exist." 
        nifti_path = ''.join(mif_path.split('.')[:-1])+'.nii.gz'
        subprocess.run(['mrconvert', mif_path, nifti_path])
        os.remove(mif_path)

        return 0

    # FOD segmentation
    subprocess.run(['fod2fixel', '-afd', 'afd.mif', '-peak_amp', 'peak_amp.mif', os.path.join(path, 'wm.nii.gz'), 
                    os.path.join(path, 'fixel_directory')])
    
    # Converting the mif fixel files to nifti files
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'afd.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'peak_amp.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'index.mif'))
    mif_to_nifti(os.path.join(path, 'fixel_directory', 'directions.mif'))

    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'peak_amp.nii.gz'), 
                    'none', os.path.join(path, 'fixel_directory', 'peak_amp_im.nii.gz')])
    
    subprocess.run(['fixel2voxel', '-number', '11', os.path.join(path, 'fixel_directory', 'afd.nii.gz'), 'none', 
                    os.path.join(path, 'fixel_directory', 'afd_im.nii.gz')])

    fixel_threshold.fixel_threshold(path)    
    
    # Tractseg
    # subprocess.run(['TractSeg', '-i', os.path.join(path, 'data_denoised.nii.gz'), '-o', os.path.join(path, 'tractseg'), 
    #                 '--bvals', os.path.join(path, 'bvals'), '--bvecs', os.path.join(path, 'bvecs'), '--raw_diffusion_input',
    #                 '--csd_type', 'csd_msmt'])

    # Extracting the number of fixels from the index image.
    subprocess.run(['mrconvert', os.path.join(path, 'fixel_directory', 'index.nii.gz'), '-coord', '3', '0', os.path.join(path, 'fixel_directory', 'index_1.nii.gz')])
    
    # # CC containing 1 fixel 
    # subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '1', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC.nii.gz'), 
    #                 '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_1fixel.nii.gz')])

    # # MCP CST intersection containing 2 fixels 
    # subprocess.run(['mrcalc', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_left.nii.gz'), os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_right.nii.gz'),
    #                 '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz')])
    # subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '2', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz'),
    #                 '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP.nii.gz'), '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP_CST_2fixel.nii.gz')])

    # # CC, CST and SLF intersection containing 3 fixels
    # subprocess.run(['mrcalc', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_III_left.nii.gz'), os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_III_right.nii.gz'),
    #                 '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_II_left.nii.gz'), '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_II_right.nii.gz'),
    #                 '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_I_left.nii.gz'), '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_I_right.nii.gz'),
    #                 '-or', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_whole.nii.gz')])
    
    # subprocess.run(['mrcalc', os.path.join(path, 'fixel_directory', 'index_1.nii.gz'), '3', '-eq', os.path.join(path, 'tractseg', 'bundle_segmentations', 'SLF_whole.nii.gz'),
    #                 '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC.nii.gz'), '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CST_whole.nii.gz'),
    #                 '-mult', os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_CST_SLF_3fixel.nii.gz')])
    
    # os.remove(os.path.join(path, 'fixel_directory', 'index_1.nii.gz'))

    return 0
    
def reset_HCP_dir(path, usamp_folder_name = 'undersampled_fod'):
    '''
    A utility function for returning a HCP directory to its original state i.e. removing all 
    processing. This script is useful for testing the above processing functions.
    '''
    print(f'Resetting {path}')

    def custom_rm(path_rm):
        if os.path.exists(path_rm):
            os.remove(path_rm)
        
    custom_rm(os.path.join(path, 'wm_response.txt'))
    custom_rm(os.path.join(path, 'gm_response.txt'))
    custom_rm(os.path.join(path, 'csf_response.txt'))

    custom_rm(os.path.join(path, 'wmfod.nii.gz'))
    custom_rm(os.path.join(path, 'gm.nii.gz'))
    custom_rm(os.path.join(path, 'csf.nii.gz'))

    custom_rm(os.path.join(path, '..', '5ttgen.nii.gz'))
    custom_rm(os.path.join(path, '..', 'white_matter_mask.nii.gz'))
    
    if os.path.exists(os.path.join(path, usamp_folder_name)):
        shutil.rmtree(os.path.join(path, usamp_folder_name))

    if os.path.exists(os.path.join(path, 'tractseg')):
        shutil.rmtree(os.path.join(path, 'tractseg'))
    
    if os.path.exists(os.path.join(path, 'fixel_directory')):
        shutil.rmtree(os.path.join(path, 'fixel_directory'))

    print(f'Finished resetting {path}')

    return 0
    
def HCP_download_test(path):
    folders_present = (os.path.exists(os.path.join(path, '..', 'T1_stripped.nii.gz'))
    and os.path.exists(os.path.join(path, 'bvecs'))
    and os.path.exists(os.path.join(path, 'bvals'))
    and os.path.exists(os.path.join(path, 'data_denoised.nii.gz'))
    and os.path.exists(os.path.join(path, '..', 'T1_stripped_bet.nii.gz')))

    print(folders_present)

    return folders_present

def preprocessing_test(path, usamp_folder_name = 'undersampled_fod'):
    '''
    Function to test that pre-processing has been performed and all of the correct files have been created. 
    For the given path each folder is checked that it contains the files that should have been calculated 
    during the pre-processing. The output is a tuple, the first is whether the files exist to use the 
    subject for training, and the second whether all of the necessary files exist. 
    '''
    assert os.path.exists(path), "The path being tested for does not exist"

    T1w_bool = (os.path.exists(os.path.join(path, '..', '5ttgen.nii.gz'))
                   )

    diffusion_bool = (os.path.exists(os.path.join(path, 'wm_response.txt'))
                         and os.path.exists(os.path.join(path, 'gm_response.txt'))
                         and os.path.exists(os.path.join(path, 'csf_response.txt'))
                         and os.path.exists(os.path.join(path, 'wmfod.nii.gz'))
                         and os.path.exists(os.path.join(path, 'gm.nii.gz'))
                         and os.path.exists(os.path.join(path, 'csf.nii.gz'))
                        and os.path.exists(os.path.join(path, 'gt_fod.nii.gz'))
                        )
    
    fixel_directory_train_bool = (os.path.exists(os.path.join(path, 'fixel_directory', 'index.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'fixnet_targets'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'fixnet_targets', 'gt_threshold_fixels.nii.gz'))
                       )

    fixel_directory_test_bool = (os.path.exists(os.path.join(path, 'fixel_directory', 'index.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'afd_im.nii.gz'))
                       and os.path.exists(os.path.join(path, 'fixel_directory', 'peak_amp_im.nii.gz'))
                       )
    
    undersampled_fod_train_bool = (os.path.exists(os.path.join(path, usamp_folder_name, 'bvals'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'bvecs'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'data_denoised.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'normalised_data_denoised.nii.gz'))
                        )
    
    undersampled_fod_test_bool = (os.path.exists(os.path.join(path, usamp_folder_name, 'wm_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'gm_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'csf_response.txt'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'wm.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'gm.nii.gz'))
                        and os.path.exists(os.path.join(path, usamp_folder_name, 'csf.nii.gz'))
                        )

    tractseg_bool = (os.path.exists(os.path.join(path, 'tractseg'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'peaks.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_1fixel.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'MCP_CST_2fixel.nii.gz'))
                     and os.path.exists(os.path.join(path, 'tractseg', 'bundle_segmentations', 'CC_CST_SLF_3fixel.nii.gz'))
                    )

    training_bool = diffusion_bool and undersampled_fod_train_bool and T1w_bool and fixel_directory_train_bool
    training_and_testing_bool = training_bool and tractseg_bool and fixel_directory_test_bool

    report =f'''
            Report for path: {path} \n
            *** TRAINING STATUS ***\n
            T1w folder status: {T1w_bool} \n
            Diffusion status: {diffusion_bool} \n
            Fixel directory status: {fixel_directory_train_bool}\n
            Undersampled FOD status: {undersampled_fod_train_bool}\n\n

            *** TESTING STATUS ***\n
            Fixel directory status: {fixel_directory_test_bool}\n
            Undersampled FOD status: {undersampled_fod_test_bool}\n
            Tractseg status: {tractseg_bool}\n\n

            TRAINING STATUS: {training_bool}
            TRAINING AND TESTING STATUS: {training_and_testing_bool}\n
            '''
    print(report) 

    return training_bool, training_and_testing_bool

def process_subject(dirname, base_dir):
    """Process a single subject directory. Returns (dirname, True, None) on success
    or (dirname, False, error_string) on failure.
    """
    if not str.isnumeric(dirname):
        return dirname, False, 'dirname is not numeric'

    # obtain the shared logger (handlers are inherited by forked workers)
    lg = logging.getLogger('sdnet.preprocess')

    diffusion_dir = os.path.join(base_dir, dirname, 'T1w', 'Diffusion')
    lg.info('STARTING %s', dirname)

    try:
        # Run processing without printing in child processes; parent will log status.
        preprocessing_test(diffusion_dir)
        # T1w_processing(diffusion_dir)
        fully_sampled_FOD(diffusion_dir)
        # fixels_and_masks(diffusion_dir)
        # undersampled_FOD(diffusion_dir, usamp_folder_name='undersampled_fod_b0_18_b1000_90',
        #                 sampling_pattern=[18, 90, 0, 0],
        #                 bval_samples=[1000, 2000, 3000])

        # undersampled_FOD(diffusion_dir, usamp_folder_name='undersampled_fod_b0_18_b1000_60',
        #                 sampling_pattern=[18, 60, 0, 0],
        #                 bval_samples=[1000, 2000, 3000])

        lg.info('COMPLETED %s', dirname)
        return dirname, True, None
    except Exception as e:
        lg.exception('ERROR processing %s', dirname)
        return dirname, False, repr(e)

  	                
if __name__ == '__main__':

    base_dir = '/homes/andrey/E_ResearchData/HCP_New_Download'
    # process_subject('100307', base_dir)

    # Gather numeric subject directories
    dirnames = [d for d in os.listdir(base_dir) if str.isnumeric(d)]

    # Configure number of workers (keep conservative default to avoid overloading the system)
    max_workers = min(20, (os.cpu_count() or 1))

    # Run processing in parallel using processes (external MRtrix/subprocess calls are CPU/IO-heavy)
    # Configure the shared logger (file + stdout).
    logger = setup_logger(base_dir)
    logger.info('Parallel preprocessing started')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject, d, base_dir): d for d in dirnames}

        # Log submissions
        for d in dirnames:
            logger.info('SUBMITTED %s', d)

        for fut in concurrent.futures.as_completed(futures):
            d = futures[fut]
            try:
                name, ok, err = fut.result()
                if ok:
                    logger.info('FINISHED %s', name)
                else:
                    logger.error('FAILED %s: %s', name, err)
            except Exception:
                logger.exception('EXCEPTION processing %s', d)

    logger.info('Parallel preprocessing finished')
