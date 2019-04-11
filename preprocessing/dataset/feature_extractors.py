import utils.utils as utils
from .feature_extractor import WordCountFeatureExtractor, WordIndicesFeatureExtractor, FeatureExtractorBase

word_indices = WordIndicesFeatureExtractor()
word_count = WordCountFeatureExtractor()

def get(feature_extractor):
    if isinstance(feature_extractor, str):
        condition = lambda name, attribute: isinstance(attribute, FeatureExtractorBase) and name == feature_extractor
        return utils.get_resource_by_condition(globals(), condition)
    else:
        raise Exception("Parameter must be a string!")