import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from classifier.loading.model_loader import ModelLoader

def main():
    args = parse_args()
    model = ModelLoader().load_model(args.model_config_path)
    print(model.predict(args.document_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config_path")
    parser.add_argument("document_path")
    return parser.parse_args()

if __name__ == "__main__":
    main()