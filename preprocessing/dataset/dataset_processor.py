import os
import os.path
import utils.utils as utils
from ..dictionary_operations.dictionary import Dictionary
from .dataset_params import DatasetParams


class DatasetProcessor:

    def __init__(self, document_processor, file_extensions = [ ".txt" ]):
        self.__file_extensions =  file_extensions
        self.__document_processor = document_processor

    def process_dataset_from_directory(self, dataset_dir):
        print("reading documents...")
        num_classes, arg_sets = self.__get_arg_sets_from_directory(dataset_dir)
        return self.__process_document_batches(arg_sets, num_classes)

    def process_dataset_from_data_map(self, data_map):
        print("reading documents...")
        num_classes, arg_sets = self.__get_arg_sets_from_data_map(data_map)
        return self.__process_document_batches(arg_sets, num_classes)

    def __process_document_batches(self, arg_sets, num_classes):
        processing_results = utils.run_operation_parallel(self._process_batch, arg_sets)
        print("merging dictionaries...")
        return self.__merge_processing_results(processing_results, num_classes)

    def _process_batch(self, filepaths):
        words = {}
        max_sequence_length = 0
        for filepath in filepaths:
            tokens = self.__document_processor.process_text_document(filepath)
            max_sequence_length = max(max_sequence_length, len(tokens))
            for token in tokens:
                words[token] = 1
        return max_sequence_length, words

    def __get_arg_sets_from_data_map(self, data_map):
        filepaths, labels = data_map.get_data_as_sequence()
        return data_map.get_num_classes(), [ (split,) for split in utils.split_list(filepaths, 12) ]

    def __get_arg_sets_from_directory(self, data_root_dir):
        filepaths = []
        classlabels = set()
        for rootdir, dirnames, filenames in os.walk(data_root_dir):
            if filenames:
                classlabels.add(rootdir)
                filepaths += [ os.path.join(rootdir, path) for path in self.__filter_valid_files(filenames) ]
        return len(classlabels), [ (split,) for split in utils.split_list(filepaths, 12) ]
    
    def __filter_valid_files(self, file_list):
        return [ filename for filename in file_list if os.path.splitext(filename)[1] in self.__file_extensions ] 

    def __merge_processing_results(self, processing_results, num_classes):
        result_dict = {}
        total_max_sequence_length = 0
        for max_sequence_length, dictionary in processing_results:
            result_dict = { **result_dict, **dictionary }
            max_sequence_length = max(total_max_sequence_length, max_sequence_length)
        result_dict["<unknown>"] = 1
        self.__index_words(result_dict)
        dictionary = Dictionary(result_dict)
        return DatasetParams(dictionary.get_length(), num_classes, total_max_sequence_length), dictionary

    def __index_words(self, dictionary):
        index = 0
        for key in dictionary.keys():
            dictionary[key] = index
            index += 1

    