import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import preprocessing.document_processors as processors
from model.model_config import ModelConfig, LSTMModelConfig, CNNModelConfig
from experiment import Experiment

def main():
    args = parse_args()
    model_configs = create_configs(args)
    train_experiment = Experiment(args.experiment_dir, model_configs, args.data_dir)
    train_experiment.run(1)

def create_configs(args):
    lstm_config = LSTMModelConfig(os.path.join(args.meta_data_dir, "config_lstm_test"), processors.default_document_processor)
    cnn_config = CNNModelConfig(os.path.join(args.meta_data_dir, "config_cnn_test"), processors.default_document_processor)
    return [cnn_config, lstm_config]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", metavar="experiment_dir")
    parser.add_argument("-d", "--data_dir", metavar="data_dir", default=None)
    parser.add_argument("-m", "--meta_data_dir",
                        metavar="meta_data_dir", default="./metadata")
    return parser.parse_args()

if __name__ == "__main__":
    main()
