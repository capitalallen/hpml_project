from src import auto_goal, auto_keras, auto_h2o

class Client: 
    def __init__(self,library):
        self.library=library
    
    def auto_goal_execute(self,data_path:str):
        ex=auto_goal.run.Run() 
        ex.load_data(data_path)
        ex.train()
    def auto_keras_execute(self,type:str,train_file=None, test_file=None):
        if type=='binary_classifier':
            bc = auto_keras.binaryClassifier.BinaryClassifier(train_file,test_file)
            bc.train() 
            bc.test() 
        elif type=='image_classifier':
            ic = auto_keras.imageClassifier.ImageClassifer()
            ic.load_data() 
            ic.train(10)
        else:
            print('classifier undefined')
    def h2o_execute(self,label_column='response',train_file=None, test_file=None):
        bc = auto_h2o.binaryClassifier.BinaryClassifier(train_file,test_file)
        bc.train(label_column=label_column,max_model=20) 
        bc.get_result()
    def execute(self,*argv):
        type=argv['type']
        if self.library=='auto-goal':
            self.auto_goal_execute(argv['data_path'])
        elif self.library=='auto-keras':
            self.auto_keras_execute(argv['type'],argv['train_file'],argv['test_file']) 
        elif self.library=='h2o':
            self.h2o_execute(argv['label_column'],argv['train_file'],argv['test_file'])
        