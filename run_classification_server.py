from server.classification_server import ClassificationServer

def start_server():
    with ClassificationServer(("localhost", 8888)) as classification_server:
        classification_server.serve_forever()

if __name__ == "__main__":
    start_server()