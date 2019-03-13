import struct

class MessageUtils:

    __message_length_format = ">I"
    __buffer_size = 4096

    def receive_message(client_socket, timeout = 20):
        message_parts = []
        client_socket.settimeout(timeout)
        message_length_raw = MessageUtils.__read_message(client_socket, 4)
        message_length = struct.unpack(MessageUtils.__message_length_format, message_length_raw)[0]
        return MessageUtils.__read_message(client_socket, message_length)

    def __read_message(client_socket, message_length):
        message_parts = []
        total_bytes_read = 0
        while total_bytes_read < message_length:
            message_part = MessageUtils.__read_from_socket(client_socket, message_length, total_bytes_read)
            if len(message_part) > 0:
                total_bytes_read += len(message_part)
                message_parts.append(message_part)
            else:
                raise ConnectionAbortedError
        return bytearray().join(message_parts)

    def __read_from_socket(client_socket, message_length, total_bytes_read):
        num_bytes_to_read = min(MessageUtils.__buffer_size,message_length - total_bytes_read)
        return client_socket.recv(num_bytes_to_read)

    def send_message(client_socket, message):
        message_length = len(message)
        message_encoded = struct.pack(MessageUtils.__message_length_format, message_length) + message.encode("utf-8")
        client_socket.sendall(message_encoded)