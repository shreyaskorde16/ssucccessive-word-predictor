from src.logger import logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

class LSTM_Model:
    def __init__(self, vocab_size = int, max_len = int):
        self.vocab_size = vocab_size
        self.max_len = max_len
        logger.info(f"max_len_in_lstm_model is {self.max_len}")
        self.Model = None
        self.embedding_dim = 100
        self.units = 100    # hidden units
        
        
    def Create_LSTMmodel(self):
        self.Model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length= self.max_len),
            LSTM(units=self.units),
            Dense(units=self.vocab_size, activation='softmax')
        ])
        logger.info(f"Created LSTM Model ")
        logger.info(f"{self.Model.summary()}")
        return self.Model