import os
import subprocess
import sys

def run_inference_tta(detection_dir, segmentation_dir, ids_dir, log_dir_path, input_path, out_dirname, return_preds):
    """
    Run inference with Test Time Augmentations
    :param detection_dir: directory of detection network
    :param segmentation_dir: directory of segmentation network
    :param cross_valid_ind: index of cross validation
    :return:
    """
    args = "--input_path {input_path} --output_folder {log_dir_path}/{segmentation_dir}/output/{out_dirname}/ " \
           "--config_dir {log_dir_path}/{detection_dir}/ --config2_dir {log_dir_path}/{segmentation_dir}/" \
           " --labeled true --preprocess window_1_99 --preprocess2 window_1_99  --augment2 all --num_augment2 32 --return_all_preds {return_preds}" \
           " --ids_list {log_dir_path}/{segmentation_dir}/{ids_dir}/test_ids.txt".format(detection_dir=detection_dir, segmentation_dir=segmentation_dir, input_path=input_path,
                                                                             ids_dir=ids_dir, log_dir_path=log_dir_path, out_dirname=out_dirname, return_preds=return_preds)
    print('running inference with arguments:')
    print(args)
    subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


def run_inference_tta_unlabeled(detection_dir, segmentation_dir, log_dir_path, input_path, out_dirname, return_preds,
                                z_autoscale=False, xy_autoscale=False, metadata_path=None, num_augment=16,
                                save_soft=True, ids_list = None):
    """
    Run inference with Test Time Augmentations
    :param detection_dir: directory of detection network
    :param segmentation_dir: directory of segmentation network
    :param cross_valid_ind: index of cross validation
    :return:
    """
    if detection_dir is not None:
        args = "--input_path {input_path} --output_folder {log_dir_path}/{segmentation_dir}/output/{out_dirname}/ " \
               "--config_dir {log_dir_path}/{detection_dir}/ --config2_dir {log_dir_path}/{segmentation_dir}/" \
               " --labeled false --preprocess window_1_99 --preprocess2 window_1_99  --augment2 all --num_augment2 {num_augment} " \
               "--return_all_preds {return_preds} --z_autoscale {z_autoscale} --xy_autoscale {xy_autoscale} " \
               "--metadata_path {metadata_path} --save_soft_labels {save_soft} --ids_list {ids_list}" \
               " ".format(detection_dir=detection_dir, segmentation_dir=segmentation_dir, input_path=input_path,
                    log_dir_path=log_dir_path, out_dirname=out_dirname, return_preds=return_preds,
                          z_autoscale=z_autoscale, xy_autoscale=xy_autoscale, metadata_path=metadata_path,
                          num_augment=num_augment, save_soft=save_soft, ids_list = ids_list)
    else:
        args = "--input_path {input_path} --output_folder {log_dir_path}/{segmentation_dir}/output/{out_dirname}/ " \
               "--config_dir {log_dir_path}/{segmentation_dir}/" \
               " --labeled false --preprocess window_1_99  --augment all --num_augment {num_augment} " \
               "--return_all_preds {return_preds} --z_autoscale {z_autoscale} --xy_autoscale {xy_autoscale} " \
               "--metadata_path {metadata_path} --save_soft_labels {save_soft} --ids_list {ids_list}" \
               " ".format(detection_dir=detection_dir, segmentation_dir=segmentation_dir, input_path=input_path,
                    log_dir_path=log_dir_path, out_dirname=out_dirname, return_preds=return_preds,
                          z_autoscale=z_autoscale, xy_autoscale=xy_autoscale, metadata_path=metadata_path,
                          num_augment=num_augment, save_soft=save_soft, ids_list=ids_list)
    print('running inference with arguments:')
    print(args)
    subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


def run_inference(detection_dir, segmentation_dir, ids_dir, log_dir_path, input_path, out_dirname, connected="True"):
    """
    Run inference
    :param detection_dir: directory of detection network. If it is None, inference with only segmentation network
    :param segmentation_dir: directory of segmentation network
    :param cross_valid_ind: index of cross validation
    :return:
    """
    if detection_dir is not None:
        args = "--input_path {input_path} --output_folder {log_dir_path}/{segmentation_dir}/output/{out_dirname}/ " \
               "--config_dir {log_dir_path}/{detection_dir}/ --config2_dir {log_dir_path}/{segmentation_dir}/" \
               " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 --connected_component {connected} " \
               "--ids_list {log_dir_path}/{segmentation_dir}/{ids_dir}/test_ids.txt".format(detection_dir=detection_dir, segmentation_dir=segmentation_dir, input_path=input_path,
                                                                                 ids_dir=ids_dir, log_dir_path=log_dir_path, out_dirname=out_dirname, connected=connected)
    else:
        args = "--input_path {input_path} --output_folder {log_dir_path}/{segmentation_dir}/output/{out_dirname}/ " \
               "--config_dir {log_dir_path}/{segmentation_dir} --labeled true --preprocess window_1_99 --connected_component {connected} " \
               "--ids_list {log_dir_path}/{segmentation_dir}/{ids_dir}/test_ids.txt".format(detection_dir=detection_dir, segmentation_dir=segmentation_dir, input_path=input_path,
                                                                                 ids_dir=ids_dir, log_dir_path=log_dir_path, out_dirname=out_dirname, connected=connected)
    print('running inference with arguments:')
    print(args)
    subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

if __name__ == "__main__":
    # # # # #

   # runs = [1027, 1029, 1028, 1037]
   #  runs = [1040, 1041, 1043, 1031]
   #  # # # # # #indices = [0,1,2,3,4]
   #  # # ##runs = [92]
   #  # # #
   #  # # # cros_valid_split = [2]
   #  # # # #
   #  for i in range(0,len(runs)):
   #
   #      args = "--input_path /home/bella/Phd/data/body/FIESTA/FIESTA_CHEO1/" \
   #             " --output_folder /media/bella/8A1D-C0A6/Phd/log/{run}/output/FIESTA_CHEO1_tta_test/" \
   #              " --config_dir /media/bella/8A1D-C0A6/Phd/log/{run}/ --labeled True --preprocess window_1_99"\
   #              " --metadata_path /home/bella/Phd/data/data_description/FIESTA/data_Elka/CHEO1/FIESTA_body.csv --xy_autoscale true --z_autoscale true" \
   #             " --augment all --num_augment 16 --return_all_preds False".format(run=runs[i], split=2)
   #      print('running with arguments:')
   #      print(args)
   #      subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

        # args = "--input_path /home/bella/Phd/data/body/TRUFI/TRUFI/  --augment all --num_augment 16 --return_all_preds False" \
        #        " --output_folder /media/bella/8A1D-C0A6/Phd/log/{run}/output/TRUFI_tta/" \
        #         " --config_dir /media/bella/8A1D-C0A6/Phd/log/{run}/ --labeled True --preprocess window_1_99"\
        #         " --metadata_path /home/bella/Phd/data/data_description/index_all_unified.csv " \
        #        " --ids_list /home/bella/Phd/code/code_bella/fetal_mr/config/config_body/TRUFI/debug_split_TRUFI/debug_split_40/test_ids.txt".format(run=runs[i], split=2)
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

        # args = "--input_path /home/bella/Phd/data/body/TRUFI/TRUFI/" \
        #        " --output_folder /media/bella/8A1D-C0A6/Phd/log/{run}/output/TRUFI/" \
        #         " --config_dir /media/bella/8A1D-C0A6/Phd/log/{run}/ --labeled True --preprocess window_1_99"\
        #         " --metadata_path /home/bella/Phd/data/data_description/index_all_unified.csv " \
        #        " --ids_list /home/bella/Phd/code/code_bella/fetal_mr/config/config_body/TRUFI/debug_split_TRUFI/debug_split_40/test_ids.txt".format(run=runs[i], split=2)
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)



        # args = "--input_path /home/bella/Phd/data/body/FIESTA/FIESTA_origin/" \
        #        " --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/FIESTA/" \
        #         " --config_dir /home/bella/Phd/code/code_bella/log/{run}/ --labeled True --preprocess window_1_99"\
        #         " --ids_list /home/bella/Phd/code/code_bella/log/92/2/test_ids.txt" \
        #        "".format(run=runs[i], split=2)
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


        # args = "--input_path /home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/" \
        #        " --output_folder /media/bella/8A1D-C0A6/Phd/log/{run}/output/FIESTA/" \
        #         " --config_dir /media/bella/8A1D-C0A6/Phd/log/{run}/ --labeled True --preprocess window_1_99"\
        #         "".format(run=runs[i])
        #
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


        # ###################################
        # #Error network
        # ###############################
        # args = "--input_path /home/bella/Phd/data/body/TRUFI/TRUFI/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/TRUFI_qe_0_TTA/" \
        #    " --config_dir /home/bella/Phd/code/code_bella/log/{run}/  --preprocess window_1_99 --labeled True --save_soft_labels true " \
        #        "--truth_filename diff.nii.gz --mask_filename prediction.nii.gz --augment all --num_augment 20 --return_all_preds false" \
        #        " --fill_holes true --ids_list /home/bella/Phd/code/code_bella/log/{run}/debug_split_0/test_ids.txt" \
        #       "".format(run=runs[i])
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
        #
        # args = "--input_path /home/bella/Phd/data/body/TRUFI/TRUFI/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/TRUFI/" \
        #    " --config_dir /home/bella/Phd/code/code_bella/log/{run}/  --preprocess window_1_99 --labeled True --fill_holes true" \
        #        " --ids_list /home/bella/Phd/code/code_bella/log/651/debug_split_TRUFI/test_ids.txt" \
        #       "".format(run=runs[i])
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

    # #     args = "--input_path /media/bella/8A1D-C0A6/Phd/daa/Body/FIESTA/FIESTA_origin_gt_errors/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/FIESTA_origin_gt_errors/" \
    # #        " --config_dir /home/bella/Phd/code/code_bella/log/{run}/  --preprocess window_1_99 --labeled True " \
    # #           "".format(run=runs[i])
    #     #############################
    # #   #  Body, with rescaling
    # # #     ##########################
    #     args = "--input_path /home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/FIESTA_large/ " \
    #            "--config_dir /home/bella/Phd/code/code_bella/log/{run}/ --preprocess window_1_99 --labeled True --xy_autoscale True " \
    #            "--z_autoscale True --metadata_path /home/bella/Phd/data/data_description/index_all.csv --fill_holes true --augment all --num_augment 32 --return_all_preds False" \
    #            "".format(run=runs[i])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
    #     args = "--input_path /media/bella/8A1D-C0A6/Phd/data/Body/HASTE/HASTE_body_normal/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/HASTE/ " \
    #            "--config_dir /home/bella/Phd/code/code_bella/log/{run}/ --preprocess window_1_99 --labeled False" \
    #            " --fill_holes true --all_in_one_dir True --augment all --num_augment 32 --return_all_preds False" \
    #            "".format(run=runs[i])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


    # runs = [90,91,92,93,94]
    # cros_valid_split = [0,1,2,3,4]
    #
    # for i in cros_valid_split:
    #     args = "--input_path /home/bella/Phd/data/body/FIESTA/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/FIESTA/" \
    #        " --config_dir /home/bella/Phd/code/code_bella/log/{run}/  --preprocess window_1_99 --labeled True --ids_list" \
    #            " /home/bella/Phd/code/code_bella/log/{run}/{split}/test_ids.txt".format(run=runs[i], split=i)
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
    # ###############
    # #Brain
    # # # # # ##############
  #  runs = [1098,1099,1100,1101]
    runs = [1102, 1103]
    # # # # cros_valid_split = [2]
    # #
    #  # for i in range(0,len(runs)):
    #  #     args = "--input_path /home/bella/Phd/data/brain/FR_FSE/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/FR-FSE_22/ " \
    #  #            "--config_dir /home/bella/Phd/code/code_bella/log/brain21_24/22/ --config2_dir /home/bella/Phd/code/code_bella/log/{run}/" \
    #  #            "  --labeled true --preprocess window_1_99 --fill_holes false " \
    #  #            "--ids_list /home/bella/Phd/code/code_bella/log/{run}/debug_split/test_ids.txt".format(run=runs[i])
    #  #     print('running with arguments:')
    #  #     print(args)
    #  #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

    for i in range(0,len(runs)):
         args = "--input_path \\\\10.101.119.14\\Dafna\\Bella\\data\\brain\\hemispheres\\ --output_folder \\\\10.101.119.14\\Dafna\\Bella\\tmp\\brain_networks\\{run}\\output\\brain_all\\ " \
                "--config_dir \\\\10.101.119.14\\Dafna\\Bella\\tmp\\brain_networks\\{run}\\ --truth_filename truth_all.nii.gz --labeled true " \
                "--ids_list \\\\10.101.119.14\\Dafna\\Bella\\tmp\\brain_networks\\1103\\debug_split_large_small\\test_ids.txt".format(run=runs[i])
         #    "--preprocess window_1_99 --fill_holes false --augment all --num_augment 16 " \
         print('running with arguments:')
         print(args)
         subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
    # # # # # #
    #FETA data
    # ####
    #runs = [507, 514]
    # # # cros_valid_split = [2]
    # #
    # for i in range(0,len(runs)):
    #     args = "--input_path /media/bella/8A1D-C0A6/Phd/data/FeTA/binary_data/feta_wm/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/gm/ " \
    #            "--config_dir /home/bella/Phd/code/code_bella/log/{run}/ --fill_holes True --connected_component True" \
    #            " --labeled true --preprocess window_1_99 --metadata_path /home/bella/Phd/data/data_description/feta.csv " \
    #            "--ids_list /home/bella/Phd/code/code_bella/log/{run}/debug_split_valid/test_ids.txt".format(run=runs[i])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

        #
        # args = "--input_path /media/bella/8A1D-C0A6/Phd/data/FeTA/binary_data/feta_cbm/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/cbm_2.1_470_det/ " \
        #        "--config_dir /home/bella/Phd/code/code_bella/log/470/ --config2_dir /home/bella/Phd/code/code_bella/log/{run}/" \
        #        " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 --metadata_path /home/bella/Phd/data/data_description/feta.csv " \
        #        "--ids_list /home/bella/Phd/code/code_bella/log/{run}/debug_split/test_ids.txt".format(run=runs[i])
        #    #    "--augment all --num_augment 10 --augment2 all --num_augment2 10  --return_all_preds False " \
        #
        # print('running with arguments:')
        # print(args)
        # subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
    #
 #    # # # ###Placenta
 #    det_nets = [1065]
 #    seg_nets = [1067]

 #    # #
 #    # # # #runs = [358, 361, 347, 416]
 #    # # # # for i in range(0,len(runs)):
 #    # # # #     args = "--input_path /home/bella/Phd/data/body/FIESTA/Fiesta_annotated_unique/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/Placenta_FIESTA_2_unsupervised/" \
 #    # # # #        " --config_dir /home/bella/Phd/code/code_bella/log/358/ --config2_dir /home/bella/Phd/code/code_bella/log/{run}/ --preprocess window_1_99 --labeled false --fill_holes true" \
 #    # # # #            " --augment2 all --num_augment2 32 --return_all_preds true".format(run=runs[i], split=2)
 #    # # # #     print('running with arguments:')
 #    # # # #     print(args)
 #    # # # #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
 #    # #
 #    for i in range(0, len(seg_nets)):
 #        args = "--input_path /home/bella/Phd/data/placenta/trufi/%/home/bella/Phd/data/placenta/ASL/placenta_test_set/ --output_folder /media/bella/8A1D-C0A6/Phd/log/{run}/output/Placenta_TRUFI/ " \
 #               "--config_dir /media/bella/8A1D-C0A6/Phd/log/{det_dir}/ --config2_dir /media/bella/8A1D-C0A6/Phd/log/{run}/" \
 #               " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 --ids_list" \
 #               " /media/bella/8A1D-C0A6/Phd/log/{run}/debug_split/test_ids.txt".format(run=seg_nets[i], det_dir=det_nets[i])
 #       #        " --augment2 all --num_augment2 32 --return_all_preds False".format(run=runs[i]).format(run=runs[i])
 #        print('running with arguments:')
 #        print(args)
 #        subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)
 #    #
 #    for i in range(0, len(seg_nets)):
 #        args = '--input_path /home/bella/Phd/data/placenta/ASL/placenta_test_set/%/home/bella/Phd/data/placenta/trufi/' \
 #               " --output_folder /media/bella/8A1D-C0A6/Phd/log/{seg}/output/placenta_trufi_tta/ " \
 #           "--config_dir /media/bella/8A1D-C0A6/Phd/log/{det}/ --config2_dir /media/bella/8A1D-C0A6/Phd/log/{seg}/" \
 #           " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 "\
 #           " --augment2 all --num_augment2 16 --return_all_preds False".format(det=det_nets[i], seg=seg_nets[i])
 #        print('running with arguments:')
 #        print(args)
 #        subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)


     # for i in range(0, len(seg_nets)):
     #    args = '--input_path /home/bella/Phd/data/placenta/ASL/placenta_test_set_cutted/%/home/bella/Phd/data/placenta/trufi_cutted/' \
     #           " --output_folder /media/bella/8A1D-C0A6/Phd/log/{seg}/output/placenta_trufi_gt_roi/ " \
     #       "--config_dir /media/bella/8A1D-C0A6/Phd/log/{seg}/" \
     #       " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 ".format(seg=seg_nets[i])
     #    print('running with arguments:')
     #    print(args)
     #    subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

    # for i in range(0, len(seg_nets)):
    #     args = '--input_path /home/bella/Phd/data/placenta/placenta_clean' \
    #            " --output_folder /media/bella/8A1D-C0A6/Phd/log/{seg}/output/Placenta_fiesta_tta/ " \
    #        "--config_dir /media/bella/8A1D-C0A6/Phd/log/{det}/ --config2_dir /media/bella/8A1D-C0A6/Phd/log/{seg}/" \
    #        " --labeled true --preprocess window_1_99 --preprocess2 window_1_99 --xy_autoscale true --z_autoscale true --ids_list" \
    #            " /home/bella/Phd/code/code_bella/fetal_mr/config/config_placenta/TRUFI/ASL_data/debug_split/test_ids_fiesta.txt" \
    #        " --metadata_path /home/bella/Phd/data/data_description/index_all_unified.csv"\
    #        " --augment2 all --num_augment2 16 --return_all_preds False".format(det=det_nets[i], seg=seg_nets[i])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

    # det_nets = [1097]
    # seg_nets = [1096]
    # for i in range(0,len(seg_nets)):
    #     args = "--input_path \\\\fmri-df3\\dafna\\Bella\\from_Dafi\\ASL_30.3_2022\\placenta\\ --output_folder \\\\fmri-df3\\dafna\\Bella\\from_Dafi\\ASL_30.3_2022\\placenta_res_tta\\" \
    #            " --config_dir \\\\10.101.119.14\\Dafna\\Bella\\best_networks\\best_networks_01.08.2022\\TRUFI_placenta\\{det_dir}\\ --config2_dir \\\\10.101.119.14\\Dafna\\Bella\\best_networks\\best_networks_01.08.2022\\TRUFI_placenta\\{run}\\" \
    #            " --labeled false --preprocess window_1_99 --preprocess2 window_1_99  --augment2 all --num_augment2 16 --fill_holes true " \
    #            "".format(run=seg_nets[i], det_dir=det_nets[0])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)

    # #no cascade
    # for i in range(0,len(runs)):
    #     args = "--input_path /home/bella/Phd/data/placenta/data_clean/ --output_folder /home/bella/Phd/code/code_bella/log/{run}/output/Placenta_FIESTA_all_test_corrected/ " \
    #            "--config_dir /home/bella/Phd/code/code_bella/log/{run}/ " \
    #            " --labeled true --preprocess window_1_99 --ids_list /home/bella/Phd/code/code_bella/log/{run}/debug_split/test_ids.txt".format(run=runs[i])
    #      #      " --augment2 all --num_augment2 32 --return_all_preds False".format(run=runs[i])
    #     print('running with arguments:')
    #     print(args)
    #     subprocess.call("python -m evaluation.predict_nifti_dir " + args, shell=True)