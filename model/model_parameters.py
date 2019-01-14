class ModelParams:
    def __init__(self, lstm_layers=2, lstm_units_per_layer=128, dense_layers=3, dense_units_per_layer=128, embedding_size=32):
        self.num_lstm_layers = lstm_layers
        self.lstm_units_per_layer = lstm_units_per_layer
        self.num_dense_layers = dense_layers
        self.dense_units_per_layer = dense_units_per_layer
        self.embedding_size = embedding_size