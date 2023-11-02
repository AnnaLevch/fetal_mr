import json

mean_resolution_body = [1.56,1.56, 3] #for body only!!
mean_resolution_brain = [0.757, 0.757]

if __name__ == "__main__":
    """
    A script to add resolution information to "norm params". In the future will be implemented as part of the saving data process
    """

 #   norm_params_path = '/home/bella/Phd/code/code_bella/log/183/norm_params.json'
    norm_params_path = '/home/bella/Phd/code/code_bella/log/370/norm_params.json'
    mean_resolution = mean_resolution_body


    with open(norm_params_path, 'r') as f:
        __norm_params = json.load(f)

    __norm_params['xy_resolution'] = mean_resolution[0:2]
    __norm_params['z_scale'] = mean_resolution[2]

    with open(norm_params_path, mode='w') as f:
        json.dump({'mean': __norm_params['mean'], 'std': __norm_params['std'], 'xy_resolution': __norm_params['xy_resolution'], 'z_scale': __norm_params['z_scale']}, f)

    print('finished updating resolution')