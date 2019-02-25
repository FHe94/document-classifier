import socket
import sys
from server.message_utils import MessageUtils

HOST, PORT = "localhost", 8888

message = """
{
    "to_classify_url_str": "abc",
    "expected_output": "top-1"
}"""

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))
    MessageUtils.send_message(sock, message)
    received = MessageUtils.receive_message(sock, 5)

print("Received: {}".format(received.decode()))