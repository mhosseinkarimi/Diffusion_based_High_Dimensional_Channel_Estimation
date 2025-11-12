import numpy as np

# Defer heavy Torch imports and turn low-level errors into a clearer message
try:
    import torch
    import torchvision.transforms.v2 as T
except Exception as e:
    raise ImportError(
        "Failed to import PyTorch/torchvision. This project requires a working PyTorch install.\n"
        "Tip: On Windows, prefer a fresh conda env with Python 3.12 and install via:\n"
        "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia\n"
        f"Original import error: {e}"
    )

from utils import HDF5Dataset, MATDataset, make_loader


def build_transforms(dataset_type):
    """Build data transformations for training and evaluation."""
    if dataset_type == 'train':
        transforms = T.Compose([
            # Keep shape-agnostic defaults; dataset tensors are expected to be float tensors
            T.Normalize(mean=[0.0], std=[1.0]),
            T.RandomHorizontalFlip()
        ])
    else:  # 'test' or 'val'
        transforms = T.Compose([
            T.ToDtype(torch.float32),
            T.Normalize(mean=[0.0], std=[1.0])
        ])
    return transforms
    
    
def preprocess_data(
    data_dir,
    dataset_type,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    world_size=1,
    rank=0,
    data_format='hdf5',
    decompose_mode='real_imag'
    ):
    """Apply preprocessing to the data and make a DataLoader."""
    transforms = build_transforms(dataset_type)
    if data_format == 'hdf5':
        dataset = HDF5Dataset(data_dir, decompose_mode=decompose_mode, transforms=transforms)
    elif data_format == 'mat':
        dataset = MATDataset(data_dir, decompose_mode=decompose_mode, transforms=transforms)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
    data_loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank
    )
    return data_loader


if __name__ == "__main__":
    data_dir = "E:/Diffusion-based Channel Estimation/data/CDL-A/val_dataset_beamspace.mat"
    loader = preprocess_data(
        data_dir=data_dir,
        dataset_type='val',
        batch_size=4,
        data_format='mat',
        decompose_mode='magnitude_phase'
    )
    for batch in loader:
        print(batch.shape)
        print(batch.dtype)
        print(batch.device)
        print(batch[0])
        break
        
            


