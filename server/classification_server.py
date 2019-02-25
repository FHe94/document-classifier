import socketserver
import os.path
import json
# import tensorflow
# from model.model_factory import ModelFactoryBase
# from model.lstm_model_factory import LSTMModelFactory
# from model.dataset_params import DatasetParams
from server.message_utils import MessageUtils
from server.arg_parser import ClassificationArgsParser

MODEL_DIRECTORY = "./metadata/baseline"

class ClassificationServer(socketserver.ThreadingTCPServer):

    def __init__(self, server_address, bind_and_activate=True):
        super().__init__(server_address, ClassificationRequestHandler, bind_and_activate)
        self.__load_classifier_model()
        print("Server running on {}:{}".format(server_address[0], server_address[1]))

    def __load_classifier_model(self):
        pass

class ClassificationRequestHandler(socketserver.BaseRequestHandler):

    def __init__(self, request, clientaddress, server):
        super().__init__(request, clientaddress, server)

    def handle(self):
        self.request.settimeout(5)
        decoded_message = MessageUtils.receive_message(self.request).decode()
        classification_args = self.__get_classification_args(decoded_message)
        print(json.dumps(classification_args.__dict__, ensure_ascii=False, indent=4))
        MessageUtils.send_message(self.request, "Done!")

    def __get_classification_args(self, message_string):
        return ClassificationArgsParser().parse_args(message_string)

    def __start_classification(self, classification_args):
        pass
        