import os, datetime, yaml, pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DiffusionWrapper
from ldm.modules.diffusionmodules import openaimodel
from ldm.util import instantiate_from_config

from dataclasses import dataclass

from torch.optim import Adamax

rescale = lambda x: (x + 1.) / 2.

@dataclass
class TrainingConfig:
    image_size:                  int   = 64
    train_batch_size:            int   = 16
    eval_batch_size:             int   = 16
    num_epochs:                  int   = 50
    gradient_accumulation_steps: int   = 50
    learning_rate:               float = 1e-4
    lr_warmup_steps:             int   = 500
    save_video_epochs:           int   = 10
    save_model_epochs:           int   = 30
    mixed_precision:             str   = 'fp16'
    output_directory:            str   = '../tiktok_videos'


class VideoDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.videos = []
        self.dir = dir
        self.transform = transform
        for f in os.path.listdir(dir):
            filepath = os.path.join(dir, f)
            frames = []
            for f_prime in os.path.listdir(filepath):
                framepath = os.path.join(filepath, f_prime)
                frame = Image.open(framepath)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            self.videos.append(frames)
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return {'image': self.videos[idx]}


if __name__ == "__main__":
    config = TrainingConfig()
    # Create transforms to apply to data
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    # Load the data
    train_dataset = VideoDataset(f'{os.getcwd()}/data/tiktok_videos/train', transform=preprocess)
    val_dataset = VideoDataset(f'{os.getcwd()}/data/tiktok_videos/val')
    # Load data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    diffusion_model = instantiate_from_config(yaml.load(f'{os.getcwd()}/configs/latent-diffusion/tiktok-dataset.yaml'))
    # First train the encoder
    ae_trainer = pl.Trainer(max_epochs=config.num_epochs)
    ae_trainer.fit(diffusion_model.first_stage_model, train_dataloders=train_dataloader)
    # Now train the diffusion model
    dm_trainer = pl.Trainer(max_epochs=config.num_epochs)
    dm_trainer.fit(diffusion_model, train_dataloaders=train_dataloader)