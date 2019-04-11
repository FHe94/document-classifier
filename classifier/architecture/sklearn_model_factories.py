import sklearn
from ..architecture.model_factory import ModelFactoryBase
from ..model.classifier_model import SKLearnClassifier

class SVCModelFactory(ModelFactoryBase):

    def _create_model(self, input_length, dataset_params, model_params):
        model = sklearn.svm.SVC(C= model_params.C, kernel= model_params.kernel, degree= model_params.degree, probability=True)
        return SKLearnClassifier(model, dataset_params.num_classes, input_length)

    def _create_default_model_params(self):
        return SVCModelParams(1.0, "rbf", 3)

class RandomForestFactory(ModelFactoryBase):

    def _create_model(self, dataset_params, model_params):
        model = sklearn.ensemble.RandomForestClassifier()
        return SKLearnClassifier(model, dataset_params.num_classes, input_length)

    def _create_default_model_params(self):
        return RandomForestParams()

class SVCModelParams:

    def __init__(self, C, kernel, degree):
        self.C = C
        self.kernel = kernel
        self.degree = degree


class RandomForestParams:

    def __init__(self):
        pass