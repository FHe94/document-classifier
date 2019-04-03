import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from classifier.loading.training_config_parser import TrainingConfigParser
from experiment import Experiment
import argparse


def main():
    args = parse_args()
    training_config = TrainingConfigParser().parse_config(args.training_config_path)
    train_experiment = Experiment(args.experiment_dir, training_config)
    train_experiment.run_train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("training_config_path")
    return parser.parse_args()


if __name__ == "__main__":
    main()
