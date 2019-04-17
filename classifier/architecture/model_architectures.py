import utils.utils as utils
import classifier.architecture.sklearn_model_factories as sklearnmodels
from .model_factory import ModelFactoryBase
from .cnn_model_factory import CNNModelFactory
from .lstm_model_factory import LSTMModelFactory

lstm = LSTMModelFactory()
cnn = CNNModelFactory()
svm = sklearnmodels.SVCModelFactory()
linearsvc = sklearnmodels.LinearSVCModelFactory()
gaussian_nb = sklearnmodels.GaussianNBModelFactory()
multinomial_nb = sklearnmodels.MultinomialNBModelFactory()
bernoulli_nb = sklearnmodels.BernoulliNBModelFactory()
randomforest = sklearnmodels.RandomForestFactory()

def get(model_architecture):
    if isinstance(model_architecture, str):
        condition = lambda name, attribute: issubclass(attribute.__class__, ModelFactoryBase) and name == model_architecture
        return utils.get_resource_by_condition(globals(), condition)
    else:
        raise Exception("Parameter must be a string!")