import torch
import torch.nn as nn
import numpy as np
from glob import glob
from tqdm import trange
from torch.distributions.normal import Normal
# from torch.optim.lr_scheduler import StepLR

from utils.filesys_manager import ExperimentPath
from utils.utils import load_npy_data
from models import Classifer


class ClassiferTrainer:

    def __init__(self, args):
        self.args = args
        self.exp_path = ExperimentPath(args.directory)
        self.t = 0
        self.init_model()

    def init_model(self):
        self.classifier = Classifer().to(self.args.device)
        # loss
        self.classifier_criterion = nn.CrossEntropyLoss()
        # data
        self.dataA = glob("datasets/{}/train/*.*".format(self.args.dataset_A_dir))
        self.dataB = glob("datasets/{}/train/*.*".format(self.args.dataset_B_dir))
        self.batch_idxs = min(min(len(self.dataA), len(self.dataB)),
                              self.args.train_size) // self.args.batch_size
        self.dataA_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_A_dir))
        self.dataB_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_B_dir))
        self.batch_idxs_test = min(min(len(self.dataA_test), len(self.dataB_test)),
                                   self.args.train_size) // self.args.batch_size
        # optimizers
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.args.lr)

    def train(self):
        """ train one epoch """
        # shuffle training data
        np.random.shuffle(self.dataA)
        np.random.shuffle(self.dataB)

        for idx in trange(0, self.batch_idxs):
            self.t += 1
            # to feed real_data
            batch_files = list(zip(self.dataA[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                   self.dataB[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
            batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32)
            batch_images = torch.tensor(batch_images).to(self.args.device)             # (b, 64, 84, 2)
            dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
            dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
            dataA_label = torch.zeros((dataA_batch.shape[0], 2)).to(self.args.device)  # (b, 2)
            dataB_label = torch.zeros((dataB_batch.shape[0], 2)).to(self.args.device)  # (b, 2)
            dataA_label[:, 0] = 1.0
            dataB_label[:, 1] = 1.0
            gaussian_noise = torch.abs(Normal(torch.zeros((self.args.batch_size, 64, 84, 1)).to(self.args.device),
                                              torch.ones((self.args.batch_size, 64, 84, 1)).to(self.args.device) *
                                              self.args.sigma_c).sample())
            # classifier prediction
            pred_A = self.classifier(dataA_batch)                     # (b, 2)
            pred_B = self.classifier(dataB_batch)                     # (b, 2)
            pred = torch.concat([pred_A, pred_B], dim=0)
            # loss
            label = torch.concat([dataA_label, dataB_label], dim=0)
            loss = self.classifier_criterion(pred, label)
            self.classifier_optimizer.zero_grad()
            # backward
            loss.backward()
            self.classifier_optimizer.step()
            with torch.no_grad():
                self.exp_path['train_stats']['loss'].csv_writerow([self.t, loss.item()])

    def test(self):
        with torch.no_grad():
            # shuffle training data
            np.random.shuffle(self.dataA_test)
            np.random.shuffle(self.dataB_test)
            print("Evaluation")
            loss_total = 0
            for idx in trange(0, self.batch_idxs_test):
                # data
                batch_files = list(zip(self.dataA_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                       self.dataB_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
                batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                batch_images = torch.tensor(batch_images).to(self.args.device)             # (b, 64, 84, 2)
                dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                dataA_label = torch.zeros((dataA_batch.shape[0], 2)).to(self.args.device)  # (b, 2)
                dataB_label = torch.zeros((dataB_batch.shape[0], 2)).to(self.args.device)  # (b, 2)
                dataA_label[:, 0] = 1.0
                dataB_label[:, 1] = 1.0
                gaussian_noise = torch.abs(Normal(torch.zeros((self.args.batch_size, 64, 84, 1)).to(self.args.device),
                                                  torch.ones((self.args.batch_size, 64, 84, 1)).to(self.args.device) *
                                                  self.args.sigma_c).sample())
                # classifier prediction
                pred_A = self.classifier(dataA_batch)                     # (b, 2)
                pred_B = self.classifier(dataB_batch)                     # (b, 2)
                pred = torch.concat([pred_A, pred_B], dim=0)
                # loss
                label = torch.concat([dataA_label, dataB_label], dim=0)
                loss = self.classifier_criterion(pred, label)
                loss_total += loss.item()
            self.exp_path['test_stats']['loss'].csv_writerow([self.t, loss_total/self.batch_idxs_test])

    def save(self, name='trainer'):
        """ save this trainer """
        self.exp_path['checkpoint'][f'{name}_{self.t}.pth'].save_model(self)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return torch.load(f)
