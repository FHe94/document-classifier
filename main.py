import argparse
import os
import json
from preprocessing.dataset.dataset_generator import DatasetGenerator
from model.classifier_model import DocumentClassifierModel
from model.model_factory import ModelFactory
from preprocessing.dictionary_operations.dictionary_loader import DictionaryLoader
from preprocessing.document_processor import DocumentProcessor
from preprocessing.dataset.train_data_map import TrainingDataMap
from preprocessing.processing_steps import processing_steps
import preprocessing.document_processors as processors

DOCUMENT_PROCESSOR = processors.default_document_processor

def main():
    args = parse_args()
    model_dir = os.path.join(args.meta_data_dir, args.model_name)
    model, dictionary, train_data, test_data = initialize_model_dir(model_dir, args.data_dir, args.mode)
    if args.mode == "train":
        train_model(model, train_data, test_data, dictionary, model_dir)
    elif args.mode == "test":
        test_model(model, test_data, dictionary)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", metavar="mode", choices = ["train", "test"])
    parser.add_argument("model_name", metavar="model_name")
    parser.add_argument("-d", "--data_dir", metavar="data_dir", default=None)
    parser.add_argument("-m", "--meta_data_dir", metavar="meta_data_dir", default="./metadata")
    return parser.parse_args()

def initialize_model_dir(model_dir, train_data_dir, mode):
    if not os.path.exists(os.path.join(model_dir, "model.h5")) and mode == "test":
        raise Exception("Model does not exist yet. Use \"train\" first!")
    os.makedirs(model_dir, exist_ok=True)
    data_map = get_or_create_samples_map(os.path.join(model_dir, "data_map.json"), train_data_dir)
    train_data, test_data = get_or_create_train_test_split(model_dir, data_map)
    dictionary = get_or_create_dict(model_dir, train_data)
    model = load_or_create_model(model_dir, dictionary.get_length(), data_map.get_num_classes())
    return model, dictionary, train_data, test_data
    
def get_or_create_dict(dict_dir, data_map):
    dictionary = None
    dict_path = os.path.join(dict_dir, "dictionary.json")
    loader = DictionaryLoader()
    if os.path.isfile(dict_path):
        print("dictionary found")
        dictionary = loader.load_dictionary(dict_path)
    else:
        print("dictionary not found, creating from testdata")
        dictionary = loader.create_from_datamap(data_map, DOCUMENT_PROCESSOR)
        loader.save_dictionary(dictionary, dict_path)
    return dictionary

def get_or_create_samples_map(samples_map_path, data_dir):
    train_data_map = None
    if os.path.isfile(samples_map_path):
        print("training data map found")
        train_data_map = TrainingDataMap.create_from_file(samples_map_path)
    else:
        print("training data map not found, creating from testdata")
        train_data_map = TrainingDataMap.create_from_testdata(data_dir, label_extraction_function=get_label_from_dirname)
        train_data_map.save(samples_map_path)
    return train_data_map

def get_or_create_train_test_split(model_dir, data_map):
    train_data_path = os.path.join(model_dir, "train_data_map.json")
    test_data_path = os.path.join(model_dir, "test_data_map.json")
    if os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
        return TrainingDataMap.create_from_file(train_data_path), TrainingDataMap.create_from_file(test_data_path)
    else:
        train_data_map, test_data_map = data_map.split_data(train_samples_per_class = 1000, min_test_samples = 400)
        train_data_map.save(train_data_path)
        test_data_map.save(test_data_path)
        return train_data_map, test_data_map
    

def get_label_from_dirname(dirname):
    dir_parts = dirname.split()
    return " ".join(dir_parts[2:len(dir_parts)])


def load_or_create_model(model_dir, dictionary_length, num_classes):
    model_factory = ModelFactory()
    model_path = os.path.join(model_dir, "model.h5")
    model = None
    if os.path.exists(model_path):
        try:
            print("Loading model from \"{}\"".format(model_path))
            model = model_factory.load_model(model_path)
        except Exception as e:
            print("Error while trying to load model. Trying to restore")
            model = model_factory.restore_model(model_path, dictionary_length, num_classes)
    else:
        print("No model found. Creating new one!")
        model = model_factory.create_new_model(dictionary_length, num_classes)
    return model

def train_model(model, train_data_map, test_data_map, dictionary, model_dir):
    print("training model")
    train_sample_paths, train_labels = train_data_map.get_data_as_sequence()
    test_sample_paths, test_labels = test_data_map.get_data_as_sequence()
    train_data_generator = DatasetGenerator(train_sample_paths, train_labels, 128, dictionary, DOCUMENT_PROCESSOR)
    test_data_generator = DatasetGenerator(test_sample_paths, test_labels, 64, dictionary, DOCUMENT_PROCESSOR)
    model.train(train_data_generator, 30, os.path.join(model_dir, "model.h5"), test_data_generator)

def test_model(model, test_data_map, dictionary):
    print("testing model")
    test_sample_paths, test_labels = test_data_map.get_data_as_sequence()
    data_generator = DatasetGenerator(test_sample_paths, test_labels, 128, dictionary, DOCUMENT_PROCESSOR)
    model.test(data_generator)

if __name__ == "__main__":
    main()