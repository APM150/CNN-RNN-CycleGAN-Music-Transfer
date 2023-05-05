import torch
import torch.nn as nn
import numpy as np
from glob import glob
from tqdm import trange
from torch.distributions.normal import Normal

from utils.filesys_manager import ExperimentPath
from utils.utils import load_npy_data
from models import Discriminator, Generator, GeneratorV2


class CycleGANTrainer:

    def __init__(self, args):
        self.args = args
        self.exp_path = ExperimentPath(args.directory)
        self.t = 0
        self.init_model()

    def init_model(self):
        self.discriminatorA = Discriminator().to(self.args.device)
        self.discriminatorB = Discriminator().to(self.args.device)
        if self.args.generator == "CNNv2":
            self.generatorAB = GeneratorV2().to(self.args.device)
            self.generatorBA = GeneratorV2().to(self.args.device)
        else:
            self.generatorAB = Generator().to(self.args.device)
            self.generatorBA = Generator().to(self.args.device)
        if self.args.model == "partial":
            self.discriminatorAM = Discriminator().to(self.args.device)
            self.discriminatorBM = Discriminator().to(self.args.device)
        # loss
        self.abs_criterion = nn.L1Loss()
        self.criterionGAN = nn.MSELoss()
        # data
        self.dataA = glob("datasets/{}/train/*.*".format(self.args.dataset_A_dir))
        self.dataB = glob("datasets/{}/train/*.*".format(self.args.dataset_B_dir))
        self.batch_idxs = min(min(len(self.dataA), len(self.dataB)),
                              self.args.train_size) // self.args.batch_size
        self.dataA_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_A_dir))
        self.dataB_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_B_dir))
        self.batch_idxs_test = min(min(len(self.dataA_test), len(self.dataB_test)),
                                   self.args.train_size) // self.args.batch_size
        if self.args.model == "partial":
            self.dataMixed = self.dataA + self.dataB
            self.dataMixed_test = self.dataA_test + self.dataB_test
        # optimizers
        if self.args.generator == "CNNv2":
            self.discriminatorA_optimizer = torch.optim.Adam(self.discriminatorA.parameters(),
                                                             lr=self.args.lr, betas=self.args.betas)
            self.discriminatorB_optimizer = torch.optim.Adam(self.discriminatorB.parameters(),
                                                             lr=self.args.lr, betas=self.args.betas)
            self.generatorAB_optimizer = torch.optim.Adam(self.generatorAB.parameters(),
                                                          lr=self.args.lr, betas=self.args.betas)
            self.generatorBA_optimizer = torch.optim.Adam(self.generatorBA.parameters(),
                                                          lr=self.args.lr, betas=self.args.betas)
            if self.args.model == "partial":
                self.discriminatorAM_optimizer = torch.optim.Adam(self.discriminatorAM.parameters(),
                                                                  lr=self.args.lr, betas=self.args.betas)
                self.discriminatorBM_optimizer = torch.optim.Adam(self.discriminatorBM.parameters(),
                                                                  lr=self.args.lr, betas=self.args.betas)
        else:
            self.discriminatorA_optimizer = torch.optim.Adam(self.discriminatorA.parameters(), lr=self.args.lr)
            self.discriminatorB_optimizer = torch.optim.Adam(self.discriminatorB.parameters(), lr=self.args.lr)
            self.generatorAB_optimizer = torch.optim.Adam(self.generatorAB.parameters(), lr=self.args.lr)
            self.generatorBA_optimizer = torch.optim.Adam(self.generatorBA.parameters(), lr=self.args.lr)
            if self.args.model == "partial":
                self.discriminatorAM_optimizer = torch.optim.Adam(self.discriminatorAM.parameters(), lr=self.args.lr)
                self.discriminatorBM_optimizer = torch.optim.Adam(self.discriminatorBM.parameters(), lr=self.args.lr)

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
            if self.args.model == "partial":
                batch_files_mixed = self.dataMixed[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]
                batch_images_mixed = [np.load(batch_file) * 1. for batch_file in batch_files_mixed]
                batch_images_mixed = np.array(batch_images_mixed).astype(np.float32)
                batch_images_mixed = torch.tensor(batch_images_mixed).to(self.args.device)  # (b, 64, 84, 1)
            gaussian_noise = torch.abs(Normal(torch.zeros((self.args.batch_size, 64, 84, 1)).to(self.args.device),
                                              torch.ones((self.args.batch_size, 64, 84, 1)).to(self.args.device) *
                                              self.args.sigma_d).sample())

            # fake data
            dataA_batch_hat = self.generatorBA(dataB_batch)                            # (b, 64, 84, 1)
            dataB_batch_hat = self.generatorAB(dataA_batch)                            # (b, 64, 84, 1)
            dataA_batch_tail = self.generatorBA(dataB_batch_hat)                       # (b, 64, 84, 1)
            dataB_batch_tail = self.generatorAB(dataA_batch_hat)                       # (b, 64, 84, 1)
            dataA_batch_hat_sample = self.generatorBA(dataB_batch)
            dataB_batch_hat_sample = self.generatorAB(dataA_batch)
            if self.args.model == "partial":
                dataA_batch_hat_sample_mixed = self.generatorBA(dataB_batch)
                dataB_batch_hat_sample_mixed = self.generatorAB(dataA_batch)

            # discriminator prediction
            DA_batch_real = self.discriminatorA(dataA_batch + gaussian_noise)          # (b, 16, 21, 1)
            DB_batch_real = self.discriminatorB(dataB_batch + gaussian_noise)          # (b, 16, 21, 1)
            DA_batch_fake = self.discriminatorA(dataA_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
            DB_batch_fake = self.discriminatorB(dataB_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
            DA_batch_fake_sample = self.discriminatorA(dataA_batch_hat_sample + gaussian_noise)
            DB_batch_fake_sample = self.discriminatorB(dataB_batch_hat_sample + gaussian_noise)
            if self.args.model == "partial":
                DAM_batch_mixed = self.discriminatorAM(batch_images_mixed + gaussian_noise)
                DBM_batch_mixed = self.discriminatorBM(batch_images_mixed + gaussian_noise)
                DAM_batch_fake = self.discriminatorAM(dataA_batch_hat_sample_mixed + gaussian_noise)
                DBM_batch_fake = self.discriminatorBM(dataB_batch_hat_sample_mixed + gaussian_noise)

            # generator loss
            cycle_loss = self.args.L1_lambda * self.abs_criterion(dataA_batch, dataA_batch_tail) + \
                            self.args.L1_lambda * self.abs_criterion(dataB_batch, dataB_batch_tail)
            g_loss_a2b = self.criterionGAN(DB_batch_fake.detach(), torch.ones(DB_batch_fake.shape).to(self.args.device))
            g_loss_b2a = self.criterionGAN(DA_batch_fake.detach(), torch.ones(DA_batch_fake.shape).to(self.args.device))
            g_loss = (g_loss_a2b + g_loss_b2a).mean() + cycle_loss.mean()
            self.generatorAB_optimizer.zero_grad()
            self.generatorBA_optimizer.zero_grad()

            # discriminator loss
            da_loss_real = self.criterionGAN(DA_batch_real, torch.ones(DA_batch_real.shape).to(self.args.device))
            da_loss_fake = self.criterionGAN(DA_batch_fake_sample,
                                             torch.zeros(DA_batch_fake.shape).to(self.args.device))
            da_loss = (da_loss_real + da_loss_fake) / 2
            db_loss_real = self.criterionGAN(DB_batch_real, torch.ones(DB_batch_real.shape).to(self.args.device))
            db_loss_fake = self.criterionGAN(DB_batch_fake_sample,
                                             torch.zeros(DB_batch_fake.shape).to(self.args.device))
            db_loss = (db_loss_real + db_loss_fake) / 2
            d_loss = (da_loss + db_loss).mean()
            self.discriminatorA_optimizer.zero_grad()
            self.discriminatorB_optimizer.zero_grad()
            if self.args.model == "partial":
                da_all_loss_real = self.criterionGAN(DAM_batch_mixed,
                                                     torch.ones(DAM_batch_mixed.shape).to(self.args.device))
                da_all_loss_fake = self.criterionGAN(DAM_batch_fake,
                                                     torch.zeros(DAM_batch_fake.shape).to(self.args.device))
                da_all_loss = (da_all_loss_real + da_all_loss_fake) / 2
                db_all_loss_real = self.criterionGAN(DBM_batch_mixed,
                                                     torch.ones(DBM_batch_mixed.shape).to(self.args.device))
                db_all_loss_fake = self.criterionGAN(DBM_batch_fake,
                                                     torch.zeros(DBM_batch_fake.shape).to(self.args.device))
                db_all_loss = (db_all_loss_real + db_all_loss_fake) / 2
                d_all_loss = da_all_loss + db_all_loss
                D_loss = d_loss + self.args.gamma * d_all_loss
                self.discriminatorAM_optimizer.zero_grad()
                self.discriminatorBM_optimizer.zero_grad()

            # backward
            g_loss.backward()
            if self.args.model == "partial":
                D_loss.backward()
            else:
                d_loss.backward()
            self.generatorAB_optimizer.step()
            self.generatorBA_optimizer.step()
            self.discriminatorA_optimizer.step()
            self.discriminatorB_optimizer.step()
            if self.args.model == "partial":
                self.discriminatorAM_optimizer.step()
                self.discriminatorBM_optimizer.step()

            with torch.no_grad():
                self.exp_path['train_stats']['g_loss'].csv_writerow([self.t, g_loss.item()])
                self.exp_path['train_stats']['d_loss'].csv_writerow([self.t, d_loss.item()])

    def test(self):
        with torch.no_grad():
            # shuffle training data
            np.random.shuffle(self.dataA_test)
            np.random.shuffle(self.dataB_test)
            print("Evaluation")
            g_loss_total = 0
            d_loss_total = 0
            D_loss_total = 0
            for idx in trange(0, self.batch_idxs_test):
                # to feed real_data
                batch_files = list(zip(self.dataA_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                       self.dataB_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
                batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                batch_images = torch.tensor(batch_images).to(self.args.device)             # (b, 64, 84, 2)
                dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                if self.args.model == "partial":
                    batch_files_mixed = self.dataMixed_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]
                    batch_images_mixed = [np.load(batch_file) * 1. for batch_file in batch_files_mixed]
                    batch_images_mixed = np.array(batch_images_mixed).astype(np.float32)
                    batch_images_mixed = torch.tensor(batch_images_mixed).to(self.args.device)  # (b, 64, 84, 1)
                gaussian_noise = torch.abs(Normal(torch.zeros((self.args.batch_size, 64, 84, 1)).to(self.args.device),
                                                  torch.ones((self.args.batch_size, 64, 84, 1)).to(self.args.device) *
                                                  self.args.sigma_d).sample())

                # fake data
                dataA_batch_hat = self.generatorBA(dataB_batch)                            # (b, 64, 84, 1)
                dataB_batch_hat = self.generatorAB(dataA_batch)                            # (b, 64, 84, 1)
                dataA_batch_tail = self.generatorBA(dataB_batch_hat)                       # (b, 64, 84, 1)
                dataB_batch_tail = self.generatorAB(dataA_batch_hat)                       # (b, 64, 84, 1)
                dataA_batch_hat_sample = self.generatorBA(dataB_batch)
                dataB_batch_hat_sample = self.generatorAB(dataA_batch)
                if self.args.model == "partial":
                    dataA_batch_hat_sample_mixed = self.generatorBA(dataB_batch)
                    dataB_batch_hat_sample_mixed = self.generatorAB(dataA_batch)

                # discriminator prediction
                DA_batch_real = self.discriminatorA(dataA_batch + gaussian_noise)          # (b, 16, 21, 1)
                DB_batch_real = self.discriminatorB(dataB_batch + gaussian_noise)          # (b, 16, 21, 1)
                DA_batch_fake = self.discriminatorA(dataA_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
                DB_batch_fake = self.discriminatorB(dataB_batch_hat + gaussian_noise)      # (b, 16, 21, 1)
                DA_batch_fake_sample = self.discriminatorA(dataA_batch_hat_sample + gaussian_noise)
                DB_batch_fake_sample = self.discriminatorB(dataB_batch_hat_sample + gaussian_noise)
                if self.args.model == "partial":
                    DAM_batch_mixed = self.discriminatorAM(batch_images_mixed + gaussian_noise)
                    DBM_batch_mixed = self.discriminatorBM(batch_images_mixed + gaussian_noise)
                    DAM_batch_fake = self.discriminatorAM(dataA_batch_hat_sample_mixed + gaussian_noise)
                    DBM_batch_fake = self.discriminatorBM(dataB_batch_hat_sample_mixed + gaussian_noise)

                # generator loss
                cycle_loss = self.args.L1_lambda * self.abs_criterion(dataA_batch, dataA_batch_tail) + \
                                self.args.L1_lambda * self.abs_criterion(dataB_batch, dataB_batch_tail)
                g_loss_a2b = self.criterionGAN(DB_batch_fake.detach(),
                                               torch.ones(DB_batch_fake.shape).to(self.args.device))
                g_loss_b2a = self.criterionGAN(DA_batch_fake.detach(),
                                               torch.ones(DA_batch_fake.shape).to(self.args.device))
                g_loss = (g_loss_a2b + g_loss_b2a).mean() + cycle_loss.mean()
                g_loss_total += g_loss.item()

                # discriminator loss
                da_loss_real = self.criterionGAN(DA_batch_real, torch.ones(DA_batch_real.shape).to(self.args.device))
                da_loss_fake = self.criterionGAN(DA_batch_fake_sample,
                                                 torch.zeros(DA_batch_fake.shape).to(self.args.device))
                da_loss = (da_loss_real + da_loss_fake) / 2
                db_loss_real = self.criterionGAN(DB_batch_real, torch.ones(DB_batch_real.shape).to(self.args.device))
                db_loss_fake = self.criterionGAN(DB_batch_fake_sample,
                                                 torch.zeros(DB_batch_fake.shape).to(self.args.device))
                db_loss = (db_loss_real + db_loss_fake) / 2
                d_loss = (da_loss + db_loss).mean()
                d_loss_total += d_loss.item()
                if self.args.model == "partial":
                    da_all_loss_real = self.criterionGAN(DAM_batch_mixed,
                                                         torch.ones(DAM_batch_mixed.shape).to(self.args.device))
                    da_all_loss_fake = self.criterionGAN(DAM_batch_fake,
                                                         torch.zeros(DAM_batch_fake.shape).to(self.args.device))
                    da_all_loss = (da_all_loss_real + da_all_loss_fake) / 2
                    db_all_loss_real = self.criterionGAN(DBM_batch_mixed,
                                                         torch.ones(DBM_batch_mixed.shape).to(self.args.device))
                    db_all_loss_fake = self.criterionGAN(DBM_batch_fake,
                                                         torch.zeros(DBM_batch_fake.shape).to(self.args.device))
                    db_all_loss = (db_all_loss_real + db_all_loss_fake) / 2
                    d_all_loss = da_all_loss + db_all_loss
                    D_loss = d_loss + self.args.gamma * d_all_loss
                    D_loss_total += D_loss.item()

            self.exp_path['test_stats']['g_loss'].csv_writerow([self.t, g_loss_total/self.batch_idxs_test])
            self.exp_path['test_stats']['d_loss'].csv_writerow([self.t, d_loss_total/self.batch_idxs_test])
            print("g_loss:", g_loss_total / self.batch_idxs_test)
            print("d_loss:", d_loss_total / self.batch_idxs_test)
            if self.args.model == "partial":
                self.exp_path['test_stats']['D_loss'].csv_writerow([self.t, D_loss_total/self.batch_idxs_test])
                print("D_loss:", D_loss_total / self.batch_idxs_test)


    def save(self, name='trainer'):
        """ save this trainer """
        self.exp_path['checkpoint'][f'{name}_{self.t}.pth'].save_model(self)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return torch.load(f)
