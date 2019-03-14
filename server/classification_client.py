import socket
import multiprocessing
import run_classification_server
from .message_utils import MessageUtils
from .arg_parser import ClassificationServerCommand

class ClassificationClient:

    def connect(self, hostname, port, timeout=20):
        self.socket = socket.create_connection((hostname, port), timeout)

    def disconnect(self):
        if hasattr(self, "socket"):
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()

    def predict(self, document_path, expected_output = "top-1"):
        args = { "to_classify": document_path, "expected_output": expected_output }
        return self.send_command("predict", args)

    def shutdown_server(self):
        return self.send_command("shutdown", None)

    def start_server(self):
        process = multiprocessing.Process(target = run_classification_server.start_server)
        process.start()

    def send_command(self, command_name, command_args):
        command = ClassificationServerCommand(command_name, command_args)
        MessageUtils.send_message(self.socket, command.to_json_string())
        response = MessageUtils.receive_message(self.socket, timeout=20).decode()
        return response
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()