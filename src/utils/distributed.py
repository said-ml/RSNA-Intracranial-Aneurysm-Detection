import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
#from typing import Optional, List
from  src.training.trainer import CVTrainer  as  Trainer

class DDPTrainer(Trainer):
    """
    DDP-enabled Trainer subclass for multi-GPU training.
    Wraps CVTrainer to support DistributedDataParallel training.
    """
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        init_method: str = "env://",
        *args,
        **kwargs
    ):
        self.rank = rank
        self.world_size = world_size
        super().__init__(*args, **kwargs)

        # Initialize distributed process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )

        # Wrap model in DDP
        self.device = torch.device(f"cuda:{rank}")
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        # Update DataLoaders with DistributedSampler
        self.train_dataloader = self._wrap_dataloader(self.train_dataloader, shuffle=True)
        if self.val_dataloader is not None:
            self.val_dataloader = self._wrap_dataloader(self.val_dataloader, shuffle=False)

    def _wrap_dataloader(self, dataloader: DataLoader, shuffle: bool = True) -> DataLoader:
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last
        )

    def train_epoch(self, epoch: int):
        # Set epoch for sampler to shuffle differently each epoch
        self.train_dataloader.sampler.set_epoch(epoch)
        super().train_epoch(epoch)

    def save_checkpoint(self, epoch: int):
        # Only rank 0 should save checkpoints
        if self.rank == 0:
            super().save_checkpoint(epoch)

    def validate(self) -> float:
        # Only rank 0 computes/prints validation
        if self.rank == 0:
            return super().validate()
        return 0.0

    def cleanup(self):
        dist.destroy_process_group()
