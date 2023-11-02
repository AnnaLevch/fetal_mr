from training.train_functions.training import load_old_model, get_last_model_path


class ActiveLearning:

    def __init__(self, model_path):
        last_model_path = get_last_model_path(model_path)
        print('Model path:' + last_model_path)
        self._model = load_old_model(last_model_path, build_manually=False)