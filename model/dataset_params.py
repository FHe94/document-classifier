class DatasetParams:
    def __init__(self, dictionary_length, num_classes):
        self.dictionary_length = dictionary_length
        self.num_classes = num_classes

class CNNDatasetParams(DatasetParams):

    def __init__(self, dictionary_length, num_classes, max_sequence_length):
        super().__init__(dictionary_length, num_classes)
        self.max_sequence_length = max_sequence_length