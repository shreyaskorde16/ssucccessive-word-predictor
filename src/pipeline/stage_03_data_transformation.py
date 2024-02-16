from src.config.configuration import ConfigurationManager
from src.logger import logger
from src.components.data_transformation import DataTransformation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path


class DataTransformationPipeline:
    def __init__(self):
        self.tokenizer = None
        self.max_input_len = None
        self.classes = None
        self.X = None
        self.y = None
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        text_data = data_transformation.get_text_data()
        logger.info("Text data imported")
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text_data])
        logger.info("Text data tokenize")
        self.classes = len(self.tokenizer.word_index)
        logger.info(f"The text data has {self.classes} word index.")
        
        text_data = text_data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('"', '')
        text_data = text_data.split()
        text_data = ' '.join(text_data)
        logger.info(f"Text data has been splited")
        token_file_name = data_transformation.dataset_name + '.pkl'
        token_path = os.path.join(data_transformation.config.root_dir, token_file_name )
        token_file_path = Path(token_path)
        logger.info(f"Path: {token_file_path}")
        pickle.dump(self.tokenizer, open(token_file_path, 'wb'))
        logger.info(f"Tokens are saved for {data_transformation.dataset_name}")
        
        
        input_sequence = []
        for sentence in text_data.split('.'):
            tokenized_sentence = self.tokenizer.texts_to_sequences([sentence])[0]
            
            for i in range(1, len(tokenized_sentence)):
                sent = tokenized_sentence[:i+1]
                input_sequence.append(sent)
        
        self.max_input_len =  max([len(x) for x in input_sequence])  
        logger.info(f"The maximum length of Input is {self.max_input_len}")  
        padded_input_sequence = pad_sequences(input_sequence, maxlen=self.max_input_len, padding='pre')    
        logger.info(f"The shape of padded input sequence is {padded_input_sequence.shape}")
        
        
        
        self.X = padded_input_sequence[:,:-1]
        self.y = padded_input_sequence[:,-1]
        self.y = to_categorical(self.y, num_classes=self.classes+1) 
        logger.info(f"The total Output shape is {self.y.shape}")
        
        