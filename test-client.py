import time
from server.classification_client import ClassificationClient

HOST, PORT = "localhost", 8888
DOCUMENT_PATH = "./arzta1.txt"

def main():
    with ClassificationClient() as client:
        print("trying to connect")
        try_connect(client)
        prediction = client.predict(DOCUMENT_PATH, "top-5")
        print(prediction)

def try_connect(client, max_attempts = 5):
    start_triggered = False
    for i in range(max_attempts):
        try:
            client.connect(HOST, PORT)
            return
        except ConnectionRefusedError:
            print("connection failed. Starting server")
            if not start_triggered:
                client.start_server()
                start_triggered = True
            time.sleep(5)
    raise Exception("Could not connect to server after {} attempts".format(max_attempts))



if __name__ == "__main__":
    main()