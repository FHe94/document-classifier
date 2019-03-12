import json
import os.path
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