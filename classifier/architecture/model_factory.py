import abc
from tensorflow import keras
from tensorflow.python.client import device_lib
from ..model.classifier_model import DocumentClassifierModel

class ModelFactoryBase:

    def create_new_model(self, dataset_params, model_params = None):
        params = self.__merge_model_params(model_params)
        model = self.__create_model(dataset_params, params)
        return DocumentClassifierModel(model)

    def load_model(self, model_path):
        model = keras.models.load_model(model_path, compile=False)
        return DocumentClassifierModel(model)

    def restore_model(self, model_path, dataset_params, model_params = None):
        params = self.__merge_model_params(model_params)
        model = self.__create_model(dataset_params, params)
        model.load_weights(model_path)
        return DocumentClassifierModel(model)

    def __create_model(self, dataset_params, model_params):
        return self._create_model_gpu(dataset_params, model_params) if self.__is_gpu_version() else self._create_model_cpu(dataset_params, model_params)

    def __is_gpu_version(self):
        devices = device_lib.list_local_devices()
        for device in devices:
            if device.device_type == "GPU":
                return True
        return False

    def __merge_model_params(self, model_params):
        out_params = self._create_default_model_params()
        if model_params is not None:
            for name, value in model_params.items():
                setattr(out_params, name, value)
        return out_params

    @abc.abstractclassmethod
    def _create_model_gpu(self, dataset_params, model_params):
        raise Exception("Method not implemented")

    @abc.abstractclassmethod
    def _create_model_cpu(self, dataset_params, model_params):
        raise Exception("Method not implemented")


    @abc.abstractclassmethod
    def _create_default_model_params(self):
        raise Exception("Method not implemented")