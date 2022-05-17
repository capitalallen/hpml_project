import h2o
from h2o.automl import H2OAutoML
class BinaryClassifier: 
    def __init__(self,train_path:str,test_path:str):
        # Import a sample binary outcome train/test set into H2O
        h2o.init()
        self.train_data = h2o.import_file(train_path)
        self.test_data = h2o.import_file(test_path)
        
    def train(self,label_column='response',max_model=20):
        # Identify predictors and response
        x = self.train_data.columns
        y = label_column
        x.remove(y)
        self.aml = H2OAutoML(max_models=max_model, seed=1)
        self.aml.train(x=x, y=y, training_frame=self.train_data)
    
    def get_results(self):
        # View the AutoML Leaderboard
        lb = self.aml.leaderboard
        print(lb.head(rows=lb.nrows)) # Print all rows instead of default (10 rows)
