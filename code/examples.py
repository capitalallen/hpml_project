
from src.client import Client 
"""
binary classifier - auto_keras 
"""
# library='auto-keras'
# type='binary_classifier'
# train_path='https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv'
# test_path='https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv'
# client = Client(library)
# client.execute(type=type,train_file=train_path,test_file=test_path)


"""
binary classifier - h2o 
"""
library='h2o'
label='response'
train_path='https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv'
test_path='https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv'
client = Client(library)
client.execute(label_column=label,train_file=train_path,test_file=test_path)