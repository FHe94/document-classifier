import json
import os
import random
import utils.utils as utils

class TrainingDataMap:

    def __init__(self, train_data_dict):
        self.__data_dict = train_data_dict
        self.__num_classes = len(train_data_dict.keys())

    def get_num_classes(self):
        return self.__num_classes

    def get_labels(self):
        out_labels = [""] * self.get_num_classes()
        for classname, classinfo in self.__data_dict.items():
            out_labels[classinfo.index] = classinfo.label
        return out_labels

    def get_data_as_sequence(self):
        samples = []
        labels = []
        for label, classinfo in self.__data_dict.items():
            for filename in classinfo.filenames:
                samples.append(os.path.join(classinfo.path, filename))
                labels.append(classinfo.index)
        return samples, labels

    def split_data_with_upsampling(self, train_samples_per_class = 1000, min_test_samples = 500):
        total_samples = self.__get_total_num_samples()
        train_samples = {}
        test_samples = {}
        rest = 0
        for classlabel, classinfo in self.__data_dict.items():
            num_test_samples_for_class = round(min_test_samples * classinfo.num_samples / total_samples + rest)
            rest = min_test_samples * classinfo.num_samples / total_samples - num_test_samples_for_class
            self.__insert_train_and_test_data(classinfo, (train_samples_per_class, num_test_samples_for_class), train_samples, test_samples)
        return TrainingDataMap(train_samples), TrainingDataMap(test_samples)
            
    def split_data(self, test_data_fraction = 0.1):
        total_samples = self.__get_total_num_samples()
        train_samples = {}
        test_samples = {}
        for classlabel, classinfo in self.__data_dict.items():
            num_test_samples_for_class = round(test_data_fraction * classinfo.num_samples)
            num_samples = (classinfo.num_samples - num_test_samples_for_class, num_test_samples_for_class)
            self.__insert_train_and_test_data(classinfo, num_samples, train_samples, test_samples)
        return TrainingDataMap(train_samples), TrainingDataMap(test_samples)

    def split_data_n_parts(self, num_parts = 4):
        total_samples = self.__get_total_num_samples()
        splits = [{}] * num_parts
        for classlabel, classinfo in self.__data_dict.items():
            splits_per_class = utils.split_list(classinfo.filenames, num_parts)
            for i in range(len(splits_per_class)):
                splits[i][classlabel] = TestDataInfo(classinfo.label, classinfo.index, splits_per_class[i], classinfo.path)
        return [ TrainingDataMap(split) for split in splits ]

    def __insert_train_and_test_data(self, classinfo, num_samples, train_samples, test_samples):
        train_split, test_split = self.__get_train_test_split(classinfo.filenames, *num_samples)
        train_samples[classinfo.label] = TestDataInfo(classinfo.label, classinfo.index, train_split, classinfo.path)
        test_samples[classinfo.label] = TestDataInfo(classinfo.label, classinfo.index, test_split, classinfo.path)

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
        utils.save_json_file(path, serializable_map)

    def __get_total_num_samples(self):
        return sum( [ value.num_samples for key, value in self.__data_dict.items() ] )

    def __is_file_path(self, path):
        return os.path.splitext(path)[1] != ''

    @staticmethod
    def create_from_file(path):
        file_content_json = utils.read_json_file(path)
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
