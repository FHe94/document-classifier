import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import re
import json
import socketserver
import socket
from server.arg_parser import ClassificationArgsParser
from server.message_utils import MessageUtils
from preprocessing.dataset.feature_extractor import WordIndicesFeatureExtractor
from preprocessing.document_processors import default_document_processor
from classifier.loading.model_loader import ModelLoader

class ClassificationServer(socketserver.ThreadingTCPServer):

    def __init__(self, server_address, classifier_config_path, bind_and_activate=True):
        super().__init__(server_address, ClassificationRequestHandler, bind_and_activate)
        self.__load_classifier_model(classifier_config_path)
        print("Server running on {}:{}".format(
            server_address[0], server_address[1]))

    def __load_classifier_model(self, config_path):
        print("loading model")
        self.session = tf.Session()
        tf.keras.backend.set_session(self.session)
        self.classifier_model = ModelLoader().load_model(config_path)
        self.graph = tf.get_default_graph()
        self.graph.finalize()
        print("model {} loaded".format(self.classifier_model.name))


class ClassificationRequestHandler(socketserver.BaseRequestHandler):

    def __init__(self, request, clientaddress, server):
        self.__command_map = {
            "shutdown": self.__shutdown_server,
            "predict": self.__predict
        }
        super().__init__(request, clientaddress, server)

    def handle(self):
        try:
            self.__try_handle_request()
        except ConnectionAbortedError:
            print("connection closed")
        except socket.timeout:
            print("Connection timed out")
        except Exception as e:
            message = 'Command failed: {}'.format(e)
            print(message)
            MessageUtils.send_message(self.request, message)

    def __try_handle_request(self):
        while True:
            decoded_message = MessageUtils.receive_message(
                self.request).decode()
            classification_args = self.__get_classification_args(
                decoded_message)
            result = self.__execute_command(classification_args)
            MessageUtils.send_message(self.request, str(result))

    def __get_classification_args(self, message_string):
        return ClassificationArgsParser().parse_args(message_string)

    def __execute_command(self, classification_args):
        command_function = self.__command_map.get(classification_args.command)
        if command_function is None:
            raise Exception("Unknown command {}".format(
                classification_args.command))
        return command_function(classification_args.args)

    def __predict(self, args):
        with self.server.session.as_default():
            with self.server.graph.as_default():
                print("predicting")
                to_classify = args["to_classify"]
                top_n = self.__parse_expected_output(
                    args.get("expected_output"))
                raw_predictions = self.server.classifier_model.predict([
                                                                       to_classify])
                prediction_results = self.__process_prediction_result(
                    raw_predictions, top_n)
                return prediction_results

    def __parse_expected_output(self, expected_output):
        regex = re.match(r"top-(\d{1,2})", expected_output)
        return 1 if regex is None else int(regex[1])

    def __process_prediction_result(self, predictions, top_n):
        sorted_indices = np.argsort(predictions)
        indices_reversed = np.flip(sorted_indices, axis=-1)
        return indices_reversed[..., 0:top_n]

    def __shutdown_server(self, args):
        print("Server shutting down...")
        self.server.shutdown()
        return "Server shut down"
