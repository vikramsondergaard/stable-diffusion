import argparse, os, sys, glob, datetime, yaml
import torch
import time
import pickle
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from sample_diffusion import rescale, custom_to_pil, custom_to_np, logs2pil, convsample, save_logs, get_parser, load_model

@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    n = shape[1]
    shape = shape[2:]
    samples, intermediates = ddim.sample(steps, batch_size=bs*n, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


class VideoDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.videos = []
        self.dir = dir
        start = 0
        if os.path.exists(os.path.join(dir, 'save.pickle')):
            with open(os.path.join(dir, 'save.pickle'), 'rb') as loaded_videos:
                self.videos = pickle.load(loaded_videos)
                start = len(self.videos)
        self.transform = transform
        for f in os.listdir(dir)[start:]:
            if f == 'save.pickle': continue
            filepath = os.path.join(dir, f, 'images')
            frames = []
            for f_prime in os.listdir(filepath):
                framepath = os.path.join(filepath, f_prime)
                with Image.open(framepath) as frame:
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(np.asarray(frame))
            self.videos.append(np.asarray(frames))
            with open(os.path.join(dir, 'save.pickle'), 'wb') as video_file:
                pickle.dump(self.videos, video_file)
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return {'video': self.videos[idx]}

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if opt.resume is not None and os.path.exists(opt.resume) and os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    elif opt.resume is not None and os.path.exists(opt.resume):
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
    else:
        logdir = opt.logdir

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    if opt.resume is not None and os.path.exists(opt.resume):
        model, global_step = load_model(config, ckpt, gpu, eval_mode)
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )
        # Load the data
        print('Loading train data...')
        train_dataset = VideoDataset(f'{os.getcwd()}/data/tiktok_videos/train', transform=preprocess)
        print('Loading validation data...')
        val_dataset = VideoDataset(f'{os.getcwd()}/data/tiktok_videos/val')
        # Load data loaders
        print('Converting to dataloader...')
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=28)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print('Instantiating diffusion model...')
        with open(os.path.join(os.getcwd(), 'configs', 'latent-diffusion', 'tiktok-dataset.yaml'), 'r') as tiktok_configs:
            config_file = yaml.safe_load(tiktok_configs)
            print(config_file)
            diffusion_model = instantiate_from_config(config_file['model'])
            diffusion_model.learning_rate = config_file['model']['base_learning_rate']
            diffusion_model.cuda()
        # First train the encoder
        # ae_trainer = pl.Trainer(max_epochs=50)
        # print('Training VAE...')
        # ae_trainer.fit(diffusion_model.first_stage_model, train_dataloaders=train_dataloader, ckpt_path=os.path.join(os.getcwd(), 'models', 'first_stage_models', 'tiktok'))
        # Now train the diffusion model
        dm_trainer = pl.Trainer(max_epochs=50, default_root_dir=os.path.join(os.getcwd(), 'models', 'ldm', 'tiktok'), accelerator='gpu', devices=2)
        print('Training latent diffusion model...')
        dm_trainer.fit(diffusion_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # Set a default place to log the samples to
        logdir = os.path.join(os.getcwd(), 'logs')
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")
