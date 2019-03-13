import socket
from .message_utils import MessageUtils
from .arg_parser import ClassificationServerCommand

class ClassificationClient:

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, hostname, port):
        self.socket.connect((hostname, port))

    def disconnect(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()

    def predict(self, document_path, expected_output = "top-1"):
        args = { "to_classify": document_path, "expected_output": expected_output }
        return self.send_command("predict", args)

    def shutdown_server(self):
        return self.send_command("shutdown", None)

    def send_command(self, command_name, command_args):
        command = ClassificationServerCommand(command_name, command_args)
        MessageUtils.send_message(self.socket, command.to_json_string())
        response = MessageUtils.receive_message(self.socket, timeout=20).decode()
        return response
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()