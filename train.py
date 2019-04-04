import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from classifier.loading.training_config_parser import TrainingConfigParser
from experiment import Experiment
import argparse
import utils.interactive_config_reader as config_reader


def main():
    args = parse_args()
    training_config = get_training_config(args)
    train_experiment = Experiment(args.experiment_dir, training_config)
    train_experiment.run_train()

def get_training_config(args):
    if args.config == "":
        return config_reader.IneractiveConfigReader().read_training_config()
    else:
        return TrainingConfigParser().parse_config(args.training_config_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("-c", "--config", metavar = "training_config_path", default = "")
    return parser.parse_args()


if __name__ == "__main__":
    main()
