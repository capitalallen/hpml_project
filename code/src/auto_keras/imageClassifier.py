import argparse
import os

import autokeras
from keras.datasets import cifar10
from sklearn.metrics import classification_report



class ImageClassifer:
    def __init__(self):
        self.output_path = './output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.CIFAR_10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.arguments = []
    def load_data(self):
        ((self.X_train, self.y_train), (self.X_test, self.y_test)) = cifar10.load_data()
        # normalize the data
        self.X_train = self.X_train.astype('float') / 255.0
        self.X_test = self.X_test.astype('float') / 255.0
    def train(self,epochs):
        classifier = autokeras.ImageClassifier() # verbose=True

        # Trains and tries to find the best architecture.
        classifier.fit(self.X_train, self.y_train,epochs=epochs) # time_limit=seconds

        # Trains the best found architecture.
        classifier.final_fit(self.X_train, self.y_train, self.X_test, self.y_test, retrain=True)

        print('[INFO] Evaluating model.')
        score = classifier.evaluate(self.X_test, self.y_test)
        predictions = classifier.predict(self.X_test)
        report = classification_report(self.y_test, predictions, target_names=self.CIFAR_10_LABELS)

        print('[INFO] Saving report to disk.')
        path = os.path.sep.join([self.output_path, f'{0}.txt'])
        with open(path, 'w') as f:
            f.write(report)
            f.write(f'\nScore: {score}')