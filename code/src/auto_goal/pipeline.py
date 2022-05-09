from torch.utils.data import DataLoader
from autogoal.grammar import *
from autogoal.utils import nice_repr

from src.auto_goal.preprocessing.processor import Preprocessor
from src.auto_goal.util.pdarts import PDarts
"""
define the overall system, we create a class that 
when given to AutoGOAL will generate valid end-2-end pipelines
"""
@nice_repr
class Pipeline:

    def __init__(
        self,
        preprocessing: Preprocessor,
        pdarts: PDarts,
        batch_size: DiscreteValue(60, 96),

    ):
        self.preprocessing = preprocessing
        self.pdarts = pdarts
        self.batch_size = batch_size

    
    def fit(self, train_dataset, valid_dataset):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1
        )
        print("pipeline - data loading completed")


        train_transform, valid_transform = self.preprocessing.fit(train_dataset)

        train_dataset.transform = train_transform
        valid_dataset.transform = valid_transform
        print("pipeline - transformation completed")

        return self.pdarts.fit(train_loader, valid_loader)