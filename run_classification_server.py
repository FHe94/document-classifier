import argparse
from server.classification_server import ClassificationServer

def start_server():
    args = parse_args()
    with ClassificationServer(("localhost", args.port), args.classifier_config_path) as classification_server:
        classification_server.serve_forever()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("classifier_config_path")
    parser.add_argument("-p", "--port", metavar="port", default=8888, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    start_server()