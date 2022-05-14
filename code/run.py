
from src.client import Client 
library='auto_keras'
train_path='https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv'
test_path='https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv'
client = Client(library)
client.execute(train_path,test_path)