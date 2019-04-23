import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from classifier.loading.model_loader import ModelLoader
from preprocessing.dataset.train_data_map import TrainingDataMap

def main():
    args = parse_args()
    model = ModelLoader().load_model(args.model_config_path)
    print(model.test_model(TrainingDataMap.create_from_file(args.test_data_map_path).get_data_as_sequence()))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_path")
    parser.add_argument("test_data_map_path")
    return parser.parse_args()


if __name__ == "__main__":
    main()