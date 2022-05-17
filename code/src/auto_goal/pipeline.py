from torch.utils.data import DataLoader
from autogoal.grammar import *
from autogoal.utils import nice_repr

from src.auto_goal.preprocessing.processor import Preprocessor
from src.auto_goal.util.pdarts import PDarts
@nice_repr
class Pipeline:

    def __init__(
        self,
        p: Preprocessor,
        d: PDarts,
        b: DiscreteValue(60, 96),

    ):
        self.preprocessing = p
        self.pdarts = d
        self.batch_size = b

    
    def fit(self, train_dataset, test_dataset):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )

        valid_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )
        print("pipeline - data loading completed")


        train_transform, valid_transform = self.preprocessing.fit(train_dataset)

        train_dataset.transform = train_transform
        test_dataset.transform = valid_transform
        print("pipeline - transformation completed")

        return self.pdarts.fit(train_loader, valid_loader)