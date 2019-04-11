import abc
import os
import pickle
from tensorflow import keras
from tensorflow.python.client import device_lib
from ..model.classifier_model import DocumentClassifierModel, SKLearnClassifier

class ModelFactoryBase:

    def create_new_model(self, input_length, dataset_params, model_params = None):
        params = self.__merge_model_params(model_params)
        return self._create_model(input_length, dataset_params, params)

    def load_model(self, model_path):
        if self.__is_keras_model(model_path):
            return self.__load_keras_model(model_path)
        else:
            return self.__load_sklearn_model(model_path)

    def restore_model(self, model_path, dataset_params, model_params = None):
        params = self.__merge_model_params(model_params)
        model = self._create_model(dataset_params, params)
        model._model.load_weights(model_path)
        return model

    def __load_keras_model(self, model_path):
        model = keras.models.load_model(model_path, compile=False)
        return DocumentClassifierModel(model)

    def __load_sklearn_model(self, model_path):
        with open(model_path, "r+b") as model_file:
            model = pickle.load(model_file, encoding="utf-8")
            num_classes = len(model.classes_)
            input_length = model.shape_fit_[1]
            return SKLearnClassifier(model, num_classes, input_length)

    def __is_keras_model(self, model_path):
        return os.path.splitext(model_path)[1] == ".h5"

    def __merge_model_params(self, model_params):
        out_params = self._create_default_model_params()
        if model_params is not None:
            for name, value in model_params.items():
                setattr(out_params, name, value)
        return out_params

    @abc.abstractclassmethod
    def _create_model(self, input_length, dataset_params, model_params):
        raise Exception("Method _create_default_model_params not implemented")

    @abc.abstractclassmethod
    def _create_default_model_params(self):
        raise Exception("Method _create_default_model_params not implemented")

class DNNModelFactoryBase(ModelFactoryBase):

    def _create_model(self, input_length, dataset_params, model_params):
        return self._create_model_gpu(input_length, dataset_params, model_params) if self.__is_gpu_version() else self._create_model_cpu(input_length, dataset_params, model_params)

    def __is_gpu_version(self):
        devices = device_lib.list_local_devices()
        for device in devices:
            if device.device_type == "GPU":
                return True
        return False

    @abc.abstractclassmethod
    def _create_model_gpu(self, input_length, dataset_params, model_params):
        raise Exception("Method _create_model_gpu not implemented")

    @abc.abstractclassmethod
    def _create_model_cpu(self, input_length, dataset_params, model_params):
        raise Exception("Method _create_model_cpu not implemented")

