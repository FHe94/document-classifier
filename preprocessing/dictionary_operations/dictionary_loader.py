import json
import os.path
import Levenshtein
from nltk.tokenize import word_tokenize
import preprocessing.document_processor
from .dictionary import Dictionary
import utils.utils as utils

class DictionaryLoader:

    def load_dictionary(self, path):
        try:
            return self.__try_load_dictionary(path)
        except Exception as e:
            raise Exception("Couldn't load dictionary: {}".format(e))

    def create_from_textdata(self, data_root_dir, document_processor):
        return DictionaryCreator(document_processor).create_dictionary(data_root_dir)

    def create_from_datamap(self, data_map, document_processor):
        return DictionaryCreator(document_processor).create_dictionary_from_data_map(data_map)

    def save_dictionary(self, dictionary, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.__is_json_dictionary(path):
            self.__save_json_dict(dictionary, path)
        else:
            self.__save_text_dict(dictionary, path)

    def __try_load_dictionary(self, path):
        self.__throw_error_if_file_not_found(path)
        file_content = self.__get_file_content(path)
        result_dict = None
        if(self.__is_json_dictionary(path)):
            result_dict = self.__load_json_dict(file_content)
        else:
            result_dict = self.__load_text_dict(file_content)
        return Dictionary(result_dict)

    def __load_json_dict(self, dict_string):
        dict_object = json.loads(dict_string, encoding="utf-8")
        if isinstance(dict_object, list):
            dict_object = self.__list_to_dict(dict_object)
        return dict_object

    def __list_to_dict(self, dict_list):
        result_dict = {}
        index = 0
        for value in dict_list:
            if value not in result_dict:
                result_dict[value] = index
                index += 1
        return result_dict

    def __save_json_dict(self, dictionary, path):
        with open(path, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(dictionary.get_values(), ensure_ascii=False, indent=4))

    def __save_text_dict(self, dictionary, path):
        with open(path, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(dictionary.get_token_list()))

    def __load_text_dict(self, dict_string):
        result_dict = {}
        index = 0
        for line in word_tokenize(dict_string, language="german"):
            result_dict[line] = index
            index += 1
        return result_dict

    def __get_file_content(self, path):
        with open(path, encoding="utf-8") as dict_file:
            return dict_file.read()

    def __throw_error_if_file_not_found(self, path):
        if not os.path.isfile(path):
            raise Exception("file \"{}\" not found".format(path))

    def __is_json_dictionary(self, path):
        return os.path.splitext(path)[1] == ".json"

class DictionaryCreator:

    def __init__(self, document_processor, file_extensions = [ ".txt" ]):
        self.__file_extensions =  file_extensions
        self.__document_processor = document_processor

    def create_dictionary(self, data_root_dir):
        print("reading documents...")
        arg_sets = self.__get_arg_sets_from_directory(data_root_dir)
        return self.__create_dict(arg_sets)
    
    def create_dictionary_from_data_map(self, data_map):
        print("reading documents...")
        arg_sets = self.__get_arg_sets_from_data_map(data_map)
        return self.__create_dict(arg_sets)

    def __create_dict(self, arg_sets):
        word_dicts = utils.run_operation_parallel(self.process_batch, arg_sets)
        print("merging dictionaries...")
        return self.__merge_word_dicts(word_dicts)

    def __get_arg_sets_from_data_map(self, data_map):
        filepaths, labels = data_map.get_data_as_sequence()
        return [ (split,) for split in utils.split_list(filepaths, 12) ]

    def __get_arg_sets_from_directory(self, data_root_dir):
        filepaths = []
        for rootdir, dirnames, filenames in os.walk(data_root_dir):
            if filenames:
                filepaths += [ os.path.join(rootdir, path) for path in self.__filter_valid_files(filenames) ]
        return [ (split,) for split in utils.split_list(filepaths, 12) ]
    
    def __filter_valid_files(self, file_list):
        return [ filename for filename in file_list if os.path.splitext(filename)[1] in self.__file_extensions ] 

    def __merge_word_dicts(self, dicts):
        result_dict = {}
        for current_dict in dicts:
            result_dict = { **result_dict, **current_dict }
        result_dict["<unknown>"] = 1
        self.__index_words(result_dict)
        return Dictionary(result_dict)

    def __index_words(self, dictionary):
        index = 0
        for key in dictionary.keys():
            dictionary[key] = index
            index += 1

    def process_batch(self, filepaths):
        words = {}
        for filepath in filepaths:
            tokens = self.__document_processor.process_text_document(filepath)
            for token in tokens:
                words[token] = 1
        return words