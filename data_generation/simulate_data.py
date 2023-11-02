import numpy as np
from utils.read_write_data import save_nifti
from skimage.filters import gaussian
import matplotlib.pyplot as plt


class DataSimulator:

    def simulate_ball(self, data_size, middle_point, ball_size):
        sphere = self.__create_ball(ball_size)
        sim_data_vol = np.zeros(data_size, dtype=np.float32)
        sim_data_gt = np.zeros(data_size, dtype=np.uint8)

        start_x = middle_point[0]-int(middle_point[0]/2)
        start_y = middle_point[1]-int(middle_point[1]/2)
        start_z = middle_point[2]-int(middle_point[2]/2)
        sim_data_gt[start_x:start_x+ball_size, start_y:start_y+ball_size, start_z:start_z+ball_size] = sphere
        sim_data_vol = np.float32(np.multiply(sim_data_gt,800))

        return sim_data_vol, sim_data_gt


    def __midpoints(self,x):
        sl = ()
        for i in range(x.ndim):
            x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
            sl += np.index_exp[:]
        return x


    def __create_ball(self, ball_size):
        x,y,z = np.indices((ball_size+1, ball_size+1, ball_size+1)) / (ball_size)

        sphere = (self.__midpoints(x) - 0.5)**2 + (self.__midpoints(y) - 0.5)**2 + (self.__midpoints(z) - 0.5)**2 < 0.5**2

        return sphere


    def blur_slices(self, volume, sigma):
        for z in range(0, volume.shape[2]):
            if np.nonzero(volume[:,:,z])[0].size != 0:
                blurred = gaussian(volume[:,:,z], sigma)
                if blurred is not None:
                    volume[:,:,z] = blurred
        return volume


if __name__ == "__main__":
    circle_sim = DataSimulator()
    sim_data_vol, sim_data_gt = circle_sim.simulate_ball([512,512,100],[150,150,39], 79)
    sim_data_vol = circle_sim.blur_slices(sim_data_vol, 20)
    save_nifti(sim_data_vol, '/home/bella/Phd/tmp/simulation/blurred/data.nii.gz')
    save_nifti(sim_data_gt, '/home/bella/Phd/tmp/simulation/blurred/truth.nii.gz')


