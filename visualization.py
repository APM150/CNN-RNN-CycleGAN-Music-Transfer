import torch
import torch.nn as nn
import numpy as np
from glob import glob
from tqdm import trange
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

from utils.filesys_manager import ExperimentPath
from trainers.cycleGAN_trainer import CycleGANTrainer
from utils.utils import load_npy_data

dataset_A_dir = "JC_J"
dataset_B_dir = "JC_C"
train_size = int(1e8)
batch_size = 16

dataA = glob("datasets/{}/train/*.*".format(dataset_A_dir))
dataB = glob("datasets/{}/train/*.*".format(dataset_B_dir))
batch_idxs = min(min(len(dataA), len(dataB)), train_size) // batch_size

idx = 0
batch_files = list(zip(dataA[idx * batch_size:(idx + 1) * batch_size],
                       dataB[idx * batch_size:(idx + 1) * batch_size]))
batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
batch_images = np.array(batch_images).astype(np.float32)
batch_images = torch.tensor(batch_images)                                  # (b, 64, 84, 2)
dataA_batch = batch_images[:,:,:,0]                      # (b, 64, 84)
dataB_batch = batch_images[:,:,:,1]                      # (b, 64, 84)



# plt.figure(figsize=(1, 13))
# plt.imshow(dataA_batch.view(-1, 84), cmap='hot', interpolation='nearest')
# plt.show()
# plt.savefig("Jazz_real")

cycleGAN_dir = "exp_music/JC_J_JC_C_2023_04_13_21_25_58/checkpoint/trainer_70100.pth"
cycleGAN_trainer = CycleGANTrainer.load(cycleGAN_dir)
generatorAB = cycleGAN_trainer.generatorAB
generatorBA = cycleGAN_trainer.generatorBA
dataB_batch_hat = generatorAB(dataA_batch.unsqueeze(-1).to('cuda')).squeeze(-1).cpu().detach()            # (b, 64, 84)
dataB_batch_hat = torch.round(dataB_batch_hat)
dataB_batch_hat[dataB_batch_hat<0.0] = 0.0
dataA_batch_hat = generatorBA(dataB_batch.unsqueeze(-1).to('cuda')).squeeze(-1).cpu().detach()
dataA_batch_hat = torch.round(dataA_batch_hat)
dataA_batch_hat[dataA_batch_hat<0.0] = 0.0
plt.figure(figsize=(10, 1))
plt.imshow(dataA_batch_hat.reshape(-1, 84).T, cmap='hot', interpolation='nearest')
# plt.show()
plt.savefig("Jazz_fake_hori")
