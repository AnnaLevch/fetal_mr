from keras.callbacks import Callback
import keras as K


class DynamicParam(Callback):
    def __init__(self, param):
        self.param = param

    def on_epoch_end(self, epoch, logs=None):
        print("Changing loss paramer")
        self.param += 0.01
        print(epoch, self.param)

    def get_param(self):
        return self.param
