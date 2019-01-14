import json
import os
import random


class DataMapFactory:

    @staticmethod
    def create_from_file(path):
        file_content_json = json.load(open(path, encoding="utf-8"))
        data_dict = {}
        for key, value in file_content_json.items():
            data_dict[key] = TestDataInfo(value["label"], value["index"], value["filenames"], value["path"])
        return TrainingDataMap(data_dict)

    @staticmethod
    def create_from_testdata(path, file_extensions = [".txt"], label_extraction_function = None):
        testdata_dict = {}
        label_index = 0
        for objectname in os.listdir(path):
            label_path = os.path.join(path, objectname)
            if os.path.isdir(label_path):
                label = objectname if label_extraction_function is None else label_extraction_function(objectname)
                test_files = TrainingDataMap.__filter_test_files(os.listdir(label_path), file_extensions)
                testdata_dict[label] = TestDataInfo(label, label_index, test_files, label_path)
                label_index += 1
        return TrainingDataMap(testdata_dict)

    @staticmethod
    def __filter_test_files(files, extensions):
        return [ filename for filename in files if os.path.splitext(filename)[1] in extensions ]



class TrainingDataMap:

    def __init__(self, train_data_dict):
        self.__data_dict = train_data_dict
        self.__num_classes = len(train_data_dict.keys())

    def get_num_classes(self):
        return self.__num_classes

    def get_data_as_sequence(self):
        samples = []
        labels = []
        for label, classinfo in self.__data_dict.items():
            for filename in classinfo.filenames:
                samples.append(os.path.join(classinfo.path, filename))
                labels.append(classinfo.index)
        return samples, labels

    def split_data(self, train_samples_per_class = 1000, min_test_samples = 500):
        total_samples = sum( [ value.num_samples for key, value in self.__data_dict.items() ] )
        train_samples = {}
        test_samples = {}
        rest = 0
        for classlabel, classinfo in self.__data_dict.items():
            num_test_samples_for_class = round(min_test_samples * classinfo.num_samples / total_samples + rest)
            rest = min_test_samples * classinfo.num_samples / total_samples - num_test_samples_for_class
            train_split, test_split = self.__get_train_test_split(classinfo.filenames, train_samples_per_class, num_test_samples_for_class)
            train_samples[classlabel] = TestDataInfo(classlabel, classinfo.index, train_split, classinfo.path)
            test_samples[classlabel] = TestDataInfo(classlabel, classinfo.index, test_split, classinfo.path)
        return TrainingDataMap(train_samples), TrainingDataMap(test_samples)
            

    def __get_train_test_split(self, trainfiles, num_train_samples_per_class, num_test_samples_for_class):
        num_samples = len(trainfiles)
        train_samples_shuffled = self.__shuffle_list(trainfiles)
        test_split_start_index = random.randrange(len(train_samples_shuffled) - num_test_samples_for_class)
        test_split = train_samples_shuffled[test_split_start_index:test_split_start_index + num_test_samples_for_class]
        train_split = train_samples_shuffled[0:test_split_start_index] + train_samples_shuffled[test_split_start_index + num_test_samples_for_class : len(train_samples_shuffled)]

        if(len(train_split) < num_train_samples_per_class):
            samples_to_duplicate = max(0, num_train_samples_per_class - len(train_split))
            train_split = self.__upsample_train_files(train_split, samples_to_duplicate)
        else:
            train_split = train_split[0:num_train_samples_per_class]
     
        return train_split, test_split

    def __upsample_train_files(self, train_split, samples_to_duplicate):
        return train_split + random.choices(train_split, k = samples_to_duplicate)

    def __shuffle_list(self, target_list):
        shuffled_list = target_list
        random.shuffle(target_list)
        return shuffled_list

    def save(self, path):
        serializable_map = { key : value.to_serializable() for key, value in self.__data_dict.items() }
        json.dump(serializable_map, open(path, mode="w", encoding="utf-8"), ensure_ascii=False, indent=4)

    def __is_file_path(self, path):
        return os.path.splitext(path)[1] != ''

    @staticmethod
    def create_from_file(path):
        file_content_json = json.load(open(path, encoding="utf-8"))
        data_dict = {}
        for key, value in file_content_json.items():
            data_dict[key] = TestDataInfo(value["label"], value["index"], value["filenames"], value["path"])
        return TrainingDataMap(data_dict)

    @staticmethod
    def create_from_testdata(path, file_extensions = [".txt"], label_extraction_function = None):
        testdata_dict = {}
        label_index = 0
        for objectname in os.listdir(path):
            label_path = os.path.join(path, objectname)
            if os.path.isdir(label_path):
                label = objectname if label_extraction_function is None else label_extraction_function(objectname)
                test_files = TrainingDataMap.__filter_test_files(os.listdir(label_path), file_extensions)
                testdata_dict[label] = TestDataInfo(label, label_index, test_files, label_path)
                label_index += 1
        return TrainingDataMap(testdata_dict)

    @staticmethod
    def __filter_test_files(files, extensions):
        return [ filename for filename in files if os.path.splitext(filename)[1] in extensions ]


class TestDataInfo:

    def __init__(self, label, label_index, filenames, path):
        self.label = label
        self.path = path
        self.filenames = filenames
        self.num_samples = len(filenames)
        self.index = label_index
    
    def to_serializable(self):
        return self.__dict__

    