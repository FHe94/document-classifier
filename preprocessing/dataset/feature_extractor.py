class WordIndicesFeatureExtractor():

    def prepare(self, dictionary):
        self.__dictionary = dictionary

    def extract_features(self, words):
        return [ self.__dictionary.get_word_index(word) for word in words ]