#dataloaders.py


#_InfiniteSampler -> Converts a sampler into an infinite sampler
#FastDataLoader ->  DataLoader wrapper with slightly improved speed by not respawning worker processes at every epoch
#InfiniteDataLoader -> Infinite DataLoader that continuously iterates using a sampler (optionally weighted), with no defined length or epoch end.



import torch


#Creates an infinite loop to generate data endlessly
#Takes a sampler, which can be any PyTorch sampler (e.g., RandomSampler, WeightedRandomSampler)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self): 
        while True: 
            for batch in self.sampler:
                yield batch


#Faster DataLoader, introduces the improvement: it does not respawn worker processes (num_workers) each epoch. Uses an infinite sampler.

class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker processes at every epoch
    """
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)
        self.dataset = dataset

    #Instead of being truly infinite, it runs for len(self) iterations, that is, one full epoch
    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


#Continuously iterates using a sampler, with no defined length or epoch end.
class InfiniteDataLoader:

    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler( 
                weights, replacement=True, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        #Gruops data in batches
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

       
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler) #use the InfiniteSampler
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self): #No length is defined because the DataLoader is infinite
        raise ValueError
