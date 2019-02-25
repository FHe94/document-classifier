import struct

class MessageUtils:

    __message_length_format = ">I"
    __buffer_size = 4096

    def receive_message(socket, timeout = 10):
        socket.settimeout(timeout)
        message_parts = []
        message_length = struct.unpack(MessageUtils.__message_length_format, socket.recv(4))[0]
        total_bytes_read = 0

        while total_bytes_read < message_length:
            num_bytes_to_read = min(MessageUtils.__buffer_size,message_length - total_bytes_read)
            message_part = socket.recv(num_bytes_to_read)
            total_bytes_read += len(message_part)
            message_parts.append(message_part)

        return bytearray().join(message_parts)

    def send_message(socket, message):
        message_length = len(message)
        message_encoded = struct.pack(MessageUtils.__message_length_format, message_length) + message.encode("utf-8")
        socket.sendall(message_encoded)