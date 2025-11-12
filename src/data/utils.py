import h5py
import lightning as L
import numpy as np
import torch
import torchvision.transforms.v2 as T
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler


def load_mat_file(file_path, decompose_mode="real_imag"):
    """Load a .mat files and return the images as numpy array."""
    channel_data = loadmat(file_path)["H"]
    if decompose_mode == "magnitude_phase":   # Decompose into magnitude and phase parts
        mag_channel = np.sqrt(channel_data[:, 0, :, :]** 2 + channel_data[:, 1, :, :]**2)
        phase_channel = np.arctan2(channel_data[:, 1, :, :], channel_data[:, 0, :, :])
        channel_data = np.concatenate((mag_channel[:, :, np.newaxis, :], phase_channel[:, :, np.newaxis, :]), axis=2)
        return channel_data.astype(np.float32)
    elif decompose_mode == "real_imag":         # Decompose into real and imaginary parts
        return channel_data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported decompose_mode: {decompose_mode}")

def load_hdf5_file(file_path, decompose_mode="real_imag"):
    """Load a file in HDF5 format and return the images as numpy array."""
    file = h5py.File(file_path, 'r')
    channel_data = file['H'][:]
    if decompose_mode == "magnitude_phase":   # Decompose into magnitude and phase parts
        mag_channel = np.sqrt(channel_data[:, :, 0, :]** 2 + channel_data[:, :, 1, :]**2)
        phase_channel = np.arctan2(channel_data[:, :, 1, :], channel_data[:, :, 0, :])
        channel_data = np.concatenate((mag_channel[:, :, np.newaxis, :], phase_channel[:, :, np.newaxis, :]), axis=2)
        return channel_data.astype(np.float32)
    elif decompose_mode == "real_imag":         # Decompose into real and imaginary parts
        return channel_data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported decompose_mode: {decompose_mode}")

def make_loader(dataset, batch_size, shuffle=True, num_workers=4, world_size=1, rank=0):
    """Create a DataLoader with distributed sampler if needed."""
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle = False
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
        )

class HDF5Dataset(Dataset):
    """Custom Dataset for loading HDF5 files."""
    def __init__(self, file_path, decompose_mode="real_imag", transforms=None):
        self.data = load_hdf5_file(file_path, decompose_mode)
        self.transforms = transforms
    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, idx):
        x = np.transpose(self.data[:, :, :, idx], (2, 0, 1))
        if self.transforms:
            x = self.transforms(x)
        else:
            x = torch.from_numpy(x).float()
        return x
    
        
class MATDataset(Dataset):
    """Custom Dataset for loading MAT files."""
    def __init__(self, file_path, decompose_mode="real_imag", transforms=None):
        self.data = load_mat_file(file_path, decompose_mode)
        self.transforms = transforms
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = np.transpose(self.data[idx], (2, 0, 1))
        if self.transforms:
            x = self.transforms(x)
        else:
            x = torch.from_numpy(x).float()
        return x

class ChannelDataModule(L.LightningDataModule):
    def __init__(self,
                train_path: str,
                val_path: str | None = None,
                test_path: str | None = None,
                file_type: str = "hdf5",          # "mat" | "hdf5" | "lazy_hdf5"
                decompose_mode: str = "real_imag",
                batch_size: int = 8,
                num_workers: int = 8,
                pin_memory: bool = True,
                prefetch_factor: int = 2,
                persistent_workers: bool = True,
                split_val: float | None = None
                ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.file_type = file_type
        self.decompose_mode = decompose_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.split_val = split_val
        
        # Define transforms in setup() after calculating stats
        self.transforms = None
        self.data_mean = None
        self.data_std = None
        
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        # build a dataset according to file_type
    def _build_dataset(self, path):
        if self.file_type == "mat":
            return MATDataset(path, self.decompose_mode, transforms=self.transforms)
        elif self.file_type == "hdf5":
            return HDF5Dataset(path, self.decompose_mode, transforms=self.transforms)

    def setup(self, stage=None):
        # Called on every process in DDP
        if self.data_mean is None:
            print("Calculating dataset statistics...")
            # Build a temporary dataset *without* normalization
            # to calculate stats from the raw training data
            temp_transforms = T.Compose([T.ToDtype(torch.float32)])
        
        if self.data_mean is None:
            print("Calculating dataset statistics...")
            # Build a temporary dataset *without* normalization
            # to calculate stats from the raw training data
            temp_transforms = T.Compose([T.ToDtype(torch.float32)])
            if self.file_type == "mat":
                stat_dataset = MATDataset(self.train_path, self.decompose_mode, transforms=temp_transforms)
                raw_data = stat_dataset.data # Shape is (N, H, W, C)
                self.data_mean = np.mean(raw_data, axis=(0, 1, 2))
                self.data_std = np.std(raw_data, axis=(0, 1, 2))
            elif self.file_type == "hdf5":
                stat_dataset = HDF5Dataset(self.train_path, self.decompose_mode, transforms=temp_transforms)
                raw_data = stat_dataset.data # Shape is (H, W, C, N)
                print("Raw data shape for stats:", raw_data.shape)
                self.data_mean = np.mean(raw_data, axis=(0, 1, 3))
                self.data_std = np.std(raw_data, axis=(0, 1, 3))
            
            # Convert to list for the transform
            self.data_mean = list(self.data_mean)
            self.data_std = list(self.data_std)
            
            print(f"Calculated Mean: {self.data_mean}")
            print(f"Calculated Std: {self.data_std}")

            # Now, create the *real* transforms
            self.transforms = T.Compose([
                T.ToDtype(torch.float32),
                T.Normalize(mean=self.data_mean, std=self.data_std)
            ])
        
        if stage in (None, "fit"):
            full_train = self._build_dataset(self.train_path)
            if self.val_path is not None:
                self.train_set = full_train
                self.val_set = self._build_dataset(self.val_path)
            else:
                if self.split_val is None:
                    # default split 5% for val if not given
                    self.split_val = 0.05
                n_total = len(full_train)
                n_val = max(1, int(n_total * self.split_val))
                n_train = n_total - n_val
                self.train_set, self.val_set = random_split(full_train, [n_train, n_val])

        if stage in (None, "test"):
            if self.test_path is not None:
                self.test_set = self._build_dataset(self.test_path)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )