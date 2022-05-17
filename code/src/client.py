from src.auto_keras import binaryClassifier,imageClassifier
# from src.auto_goal import run 
from src.auto_h2o import binaryClassifier as h2o_binaryCliassifer

class Client: 
    def __init__(self,library):
        self.library=library
    
    def auto_goal_execute(self,data_path:str):
        pass 
        # ex=run.Run() 
        # ex.load_data(data_path)
        # ex.train()
    def auto_keras_execute(self,type:str,train_file=None, test_file=None):
        if type=='binary_classifier':
            bc = binaryClassifier.BinaryClassifier(train_file,test_file)
            bc.train() 
            bc.test() 
        elif type=='image_classifier':
            ic = imageClassifier.ImageClassifer()
            ic.load_data() 
            ic.train(10)
        else:
            print('classifier undefined')
    def h2o_execute(self,label_column='response',train_file=None, test_file=None):
        bc = h2o_binaryCliassifer.BinaryClassifier(train_file,test_file)
        bc.train(label_column=label_column,max_model=20) 
        bc.get_result()
    def execute(self,**kwargs):
        if self.library=='auto-goal':
            self.auto_goal_execute(kwargs['data_path'])
        elif self.library=='auto-keras':
            self.auto_keras_execute(kwargs['type'],kwargs['train_file'],kwargs['test_file']) 
        elif self.library=='h2o':
            self.h2o_execute(kwargs['label_column'],kwargs['train_file'],kwargs['test_file'])
        