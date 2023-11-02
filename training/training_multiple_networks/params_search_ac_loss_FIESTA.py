import subprocess
import json
import os


if __name__ == "__main__":
    base_config = "../../config/config_body/cross_valid/config_ac_assym_contour_emphesize_loss_2.json"
    configs_folder = '../../../log/params_search_FIESTA'
    beta_vals = [0.5]
    gamma_vals = [0]
    labmda_vals = [2.5, 3]
    theta_vals = [2,4,8]
    initial_learning_rate_vals = [0.0001]
    num_epochs = 10
    my_path = os.path.abspath(os.path.dirname(__file__))

    ind = 1
    with open(os.path.join(my_path, base_config)) as json_file:
        data = json.load(json_file)
        for lr in initial_learning_rate_vals:
            for beta in beta_vals:
                for gamma in gamma_vals:
                    for labmda in labmda_vals:
                        for theta in theta_vals:
                            data['initial_learning_rate'] = lr
                            data['labmda_ac'] = labmda
                            data['beta_ac'] = beta
                            data['gamma_ac'] = gamma
                            data['theta_ac'] = theta
                            data['n_epochs'] = num_epochs
                            out_config_name = 'ac_loss_labmda' + str(labmda) + '_beta' + str(beta) + '_gamma' + str(gamma) + '_theta' + str(theta) + '_lr' +\
                                              str(lr) + '_' + str(ind) + '.json'
                            out_config_path = os.path.join(my_path, configs_folder,out_config_name)
                            with open(out_config_path,'w') as outfile :
                                json.dump(data,outfile, indent=2)

                            args = "with " + out_config_path
                            subprocess.call("python3 -m training.train_scripts.train_params_search_FIESTA " + args, shell=True)
                            ind = ind + 1