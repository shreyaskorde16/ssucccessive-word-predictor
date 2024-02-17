from sklearn.model_selection import train_test_split
from src.components.create_model import LSTM_Model
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import matplotlib.pyplot as plt
from src.logger import logger
from src.logger import log_path
import os

class TrainingPipeline:
    def __init__ (self, X, y, vocab_size=int, max_len=int, epochs=int):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.epochs= epochs
        self.epochs_plt=None
        
    
        
    def training_model(self):
        
        model = LSTM_Model(vocab_size=self.vocab_size, max_len=self.max_len)
        lstm_model = model.Create_LSTMmodel()
        #model_path = Path("artifacts/best_model_checkpoint.h5") 
        checkpoint = ModelCheckpoint("best_model_checkpoint.h5", monitor="loss", verbose=1, save_best_only=True)
        logger.info("Checkpoint created")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42 )
        lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        history = lstm_model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32, validation_data=(self.X_test, self.y_test), callbacks=[checkpoint])
        
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        self.epochs_plt = range(1, len(train_accuracy) + 1)
        
        self.plot_graph(train_accuracy, val_accuracy, "Training Accuracy", "Validation Accuracy", "Training and Validation accuracy","Accuracy")
        self.plot_graph(train_loss, val_loss, "Training loss", "Validation loss", "Training adn validation loss", "loss" )
        last_model_path = Path("artifacts/model.h5")
        lstm_model.save(last_model_path)
    
    
    def plot_graph(self, y1, y2, y1_label, y2_label, title, y_title):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs_plt, y1, 'bo', label=y1_label)
        plt.plot(self.epochs_plt, y2, 'b', label=y2_label)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(y_title)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(log_path, f"{title}"+".png")
        plt.savefig(plot_path)
        