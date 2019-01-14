class BatchCreator():

    def __init__(self, dictionary, document_processor):
        self.__dictionary = dictionary
        self.__document_processor = document_processor

    def create_batch(self, file_paths):
        batch = []
        for filepath in file_paths:
            batch.append(self.__prepare_file(filepath))
        self.__pad_samples(batch)
        return batch
  
    def __pad_samples(self, batch):
        max_length = self.__get_max_sample_length(batch)
        for i in range(len(batch)):
            for _ in range(max_length - len(batch[i])):
                batch[i].append(0)

    def __get_max_sample_length(self, batch):
        max_length = 0
        for sample in batch:
            max_length = max(len(sample), max_length)
        return max_length

    def __prepare_file(self, path):
        words = self.__document_processor.process_text_document(path)
        return [ self.__dictionary.get_word_index(word) for word in words ]
