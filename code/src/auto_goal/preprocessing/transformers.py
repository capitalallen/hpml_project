import torchvision.transforms as tt
from autogoal.grammar import *


class RandomAffine(tt.RandomAffine):

    def __init__(
        self,
        d: ContinuousValue(0, 360),
        t_w: ContinuousValue(0, 1),
        t_h: ContinuousValue(0, 1),
        s_m: ContinuousValue(0.1, 1),
        s_r: ContinuousValue(1, 10),
        s: ContinuousValue(0, 360),
        works: BooleanValue()
    ):
        super().__init__(
            d,
            translate=(t_w, t_h),
            scale=(s_m, s_m + s_r),
            shear=s
        )
        self.is_enabled = works

class GaussianBlur(tt.GaussianBlur):
    def __init__(
        self, 
        k_s: DiscreteValue(0, 20),
        s: ContinuousValue(0.1, 2.0),
        works: BooleanValue()
    ):
        super().__init__(kernel_size=k_s*2+1, sigma=s)
        self.is_enabled = works

class RandomHorizontalFlip(tt.RandomHorizontalFlip):
    def __init__(
        self,
        p: ContinuousValue(0, 1),
        works: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = works

class RandomVerticalFlip(tt.RandomVerticalFlip):

    def __init__(
        self,
        p: ContinuousValue(0, 1),
        works: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = works

class ColorJitter(tt.ColorJitter):

    def __init__(
        self,
        b: ContinuousValue(0, 20),
        c: ContinuousValue(0, 20),
        s: ContinuousValue(0, 20),
        h: ContinuousValue(0, 0.5),
        works: BooleanValue()
    ):
        super().__init__(
            brightness=b,
            contrast=c,
            saturation=s,
            hue=h
        )
        self.is_enabled = works

class RandomErasing(tt.RandomErasing):

    def __init__(
        self,
        p: ContinuousValue(0, 1.0),
        works: BooleanValue()
    ):
        super().__init__(p=p)
        self.is_enabled = works