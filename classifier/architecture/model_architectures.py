import utils.utils as utils
from .model_factory import ModelFactoryBase
from .cnn_model_factory import CNNModelFactory
from .lstm_model_factory import LSTMModelFactory
from .sklearn_model_factories import RandomForestFactory, SVCModelFactory

lstm = LSTMModelFactory()
cnn = CNNModelFactory()
svm = SVCModelFactory()
randomforest = RandomForestFactory()

def get(model_architecture):
    if isinstance(model_architecture, str):
        condition = lambda name, attribute: issubclass(attribute.__class__, ModelFactoryBase) and name == model_architecture
        return utils.get_resource_by_condition(globals(), condition)
    else:
        raise Exception("Parameter must be a string!")