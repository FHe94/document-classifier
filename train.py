import argparse
import os
import json
import preprocessing.document_processors as processors
from preprocessing.dataset.dataset_generator import DatasetGenerator, CNNDatasetGenerator
from model.cnn_model_factory import CNNModelFactory
from model.lstm_model_factory import LSTMModelFactory
from model.model_config import ModelConfig, LSTMModelConfig, CNNModelConfig
from preprocessing.dataset.dataset_params import DatasetParams
from experiment import Experiment

DOCUMENT_PROCESSOR = processors.default_document_processor

def main():
    args = parse_args()
    model_configs = create_configs()
    train_experiment = Experiment(args.experiment_dir, model_configs, args.data_dir)
    train_experiment.run()


def create_configs():
    lstm_config = LSTMModelConfig("./metadata/config_lstm_test", DOCUMENT_PROCESSOR, LSTMModelFactory())
    cnn_config = CNNModelConfig("./metadata/config_cnn_test", DOCUMENT_PROCESSOR, CNNModelFactory())
    return [lstm_config, cnn_config]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", metavar="experiment_dir")
    parser.add_argument("-d", "--data_dir", metavar="data_dir", default=None)
    parser.add_argument("-m", "--meta_data_dir",
                        metavar="meta_data_dir", default="./metadata")
    return parser.parse_args()

if __name__ == "__main__":
    main()
