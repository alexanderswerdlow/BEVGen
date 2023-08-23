from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, Subset, Dataset

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, 
                batch_size: int, 
                num_workers: int,
                pin_memory: bool,
                train: Optional[Dataset] = None,
                validation: Optional[Dataset] = None,
                test: Optional[Dataset] = None,
                smoke_test: bool = False,
                small_val: bool = False,
                val_batch_size: Optional[int] = None,
                num_gpus: Optional[int] = None,
                single_val: bool = False,
                shuffle_test: bool = True,
                large_mini_dataset: bool = False
                ):
        super().__init__()

        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.smoke_test = smoke_test
        self.small_val = small_val
        self.num_workers = 0 if self.smoke_test else num_workers
        self.val_batch_size = val_batch_size
        self.num_gpus = num_gpus
        self.single_val = single_val
        self.shuffle_test = shuffle_test    
        self.large_mini_dataset = large_mini_dataset

        self.data_train: Optional[Dataset] = train
        self.data_val: Optional[Dataset] = validation
        self.data_test: Optional[Dataset] = test

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            print(f'Train dataset has {len(self.data_train)} samples')
        
        if self.data_val is not None:
            print(f'Val dataset has {len(self.data_val)} samples')

        if self.data_test is not None:
            print(f'Test dataset has {len(self.data_test)} samples')

    def get_random_subset(self, dataset, samples, replacement=False):
        return Subset(dataset, list(RandomSampler(self.data_val, num_samples=samples, replacement=replacement)))

    def train_dataloader(self):
        return DataLoader(self.get_random_subset(self.data_train, 10000, True) if self.large_mini_dataset else self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        batch_size = self.val_batch_size if self.val_batch_size else self.batch_size
        if self.num_gpus:
            data_val = self.get_random_subset(self.data_val, self.num_gpus * batch_size)
        elif self.smoke_test or self.small_val:
            data_val = self.get_random_subset(self.data_val, 2 * batch_size)
        elif self.single_val:
            self.data_val.reset_selected()
            data_val = self.data_val
        else:
            data_val = self.data_val
        return DataLoader(dataset=data_val, batch_size=batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=self.shuffle_test)
