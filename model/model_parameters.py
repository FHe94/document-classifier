class LSTMModelParams:
    def __init__(self, lstm_layers, lstm_units_per_layer, dense_layers, dense_units_per_layer, embedding_size):
        self.num_lstm_layers = lstm_layers
        self.lstm_units_per_layer = lstm_units_per_layer
        self.num_dense_layers = dense_layers
        self.dense_units_per_layer = dense_units_per_layer
        self.embedding_size = embedding_size

class CNNModelParams:
    def __init__(self, filter_sizes, num_filters, dropout_rate, embedding_size):
        self.filter_sizes = filter_sizes
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate