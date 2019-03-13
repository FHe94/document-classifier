import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import preprocessing.document_processors as processors
from preprocessing.dataset.feature_extractor import WordIndicesFeatureExtractor
from model.lstm_model_factory import LSTMModelFactory
from model.model_config import ModelConfig

def main():
    args = parse_args()
    model_config = ModelConfig(args.model_dir, processors.default_document_processor,
     WordIndicesFeatureExtractor())
    model_config.load_model_from_data_map()
    print(model_config.predict(args.document_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("document_path")
    return parser.parse_args()

if __name__ == "__main__":
    main()