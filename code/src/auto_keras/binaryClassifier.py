import numpy as np
import pandas as pd
import tensorflow as tf

import autokeras as ak

class BinaryClassifier:
    def __init__(self,train_file:str, test_file: str):
        self.train_file_path = tf.keras.utils.get_file("train.csv", train_file)
        self.test_file_path = tf.keras.utils.get_file("eval.csv", test_file)
    
    def train(self,max_trials=3,label_column='response',epochs=20):
        self.label_column=label_column
        self.clf = ak.StructuredDataClassifier(
            overwrite=True, max_trials=3
        )  # It tries 3 different models.
        # Feed the structured data classifier with training data.
        self.clf.fit(
            # The path to the train.csv file.
            self.train_file_path,
            # The name of the label column.
            label_column,
            epochs=epochs,
        )
    
    def test(self):
        if self.clf:
            print(self.clf.evaluate(self.test_file_path, self.label_column))
        else:
            print("classifier not defined")
