from torchvision.datasets import CIFAR10
from autogoal.grammar import generate_cfg
from src.auto_goal.pipeline import Pipeline
class Run: 
    def __init__(self):
        pass 
    def load_data(self,data_path):
        print('got sample')
        self.train_data = CIFAR10(
            root=data_path, train=True, download=True
        )

        self.val_data = CIFAR10(
            root=data_path, train=False, download=True
        )
    def train(self,*argv):
        data_path=argv['data_path'] # ./data
        self.load_data(data_path)
        print('start')
        grammar = generate_cfg(Pipeline)

        candidate = grammar.sample()
        print("training")
        candidate.fit(self.train_data, self.val_data)