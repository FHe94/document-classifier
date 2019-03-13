from server.classification_server import ClassificationServer

if __name__ == "__main__":
    with ClassificationServer(("localhost", 8888)) as classification_server:
        classification_server.serve_forever()