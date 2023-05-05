import os
import argparse
import pprint
import traceback
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from tqdm import trange
import torch.nn.functional as F

from utils.filesys_manager import ExperimentPath
from utils.utils import load_npy_data, now
from trainers.cycleGAN_trainer import CycleGANTrainer
from trainers.classifer_trainer import ClassiferTrainer


class Evaluate:

    def __init__(self, args):
        self.args = args
        self.exp_path = ExperimentPath(args.directory)
        self.t = 0
        self.init_model()

    def init_model(self):
        # load model
        self.cycleGAN_trainer = CycleGANTrainer.load(self.args.cycleGAN_dir)
        self.classifier_trainer = ClassiferTrainer.load(self.args.classifier_dir)

        self.generatorAB = self.cycleGAN_trainer.generatorAB
        self.generatorBA = self.cycleGAN_trainer.generatorBA
        self.classifier = self.classifier_trainer.classifier
        # data
        self.dataA_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_A_dir))
        self.dataB_test = glob('datasets/{}/test/*.*'.format(self.args.dataset_B_dir))
        self.batch_idxs_test = min(min(len(self.dataA_test), len(self.dataB_test)),
                                   self.args.train_size) // self.args.batch_size

    def evaluate(self):
        with torch.no_grad():
            for idx in trange(0, self.batch_idxs_test):
                self.t += 1
                # to feed real_data
                batch_files = list(zip(self.dataA_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size],
                                       self.dataB_test[idx * self.args.batch_size:(idx + 1) * self.args.batch_size]))
                batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                batch_images = torch.tensor(batch_images).to(self.args.device)             # (b, 64, 84, 2)
                dataA_batch = batch_images[:,:,:,0].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                dataB_batch = batch_images[:,:,:,1].unsqueeze(dim=-1)                      # (b, 64, 84, 1)
                # fake data
                dataA_batch_hat = self.generatorBA(dataB_batch)                            # (b, 64, 84, 1)
                dataB_batch_hat = self.generatorAB(dataA_batch)                            # (b, 64, 84, 1)
                dataA_batch_tail = self.generatorBA(dataB_batch_hat)                       # (b, 64, 84, 1)
                dataB_batch_tail = self.generatorAB(dataA_batch_hat)                       # (b, 64, 84, 1)
                # eval
                dataA_batch_dist = F.softmax(self.classifier(dataA_batch), dim=1)
                dataA_batch_hat_dist = F.softmax(self.classifier(dataA_batch_hat), dim=1)
                dataA_batch_tail_dist = F.softmax(self.classifier(dataA_batch_tail), dim=1)
                dataB_batch_dist = F.softmax(self.classifier(dataB_batch), dim=1)
                dataB_batch_hat_dist = F.softmax(self.classifier(dataB_batch_hat), dim=1)
                dataB_batch_tail_dist = F.softmax(self.classifier(dataB_batch_tail), dim=1)
                # log
                self.exp_path['dataA_dist'].csv_writerow([self.t, dataA_batch_dist.mean(0)[0].item()])
                self.exp_path['dataA_hat_dist'].csv_writerow([self.t, dataA_batch_hat_dist.mean(0)[0].item()])
                self.exp_path['dataA_tail_dist'].csv_writerow([self.t, dataA_batch_tail_dist.mean(0)[0].item()])
                self.exp_path['dataB_dist'].csv_writerow([self.t, dataB_batch_dist.mean(0)[1].item()])
                self.exp_path['dataB_hat_dist'].csv_writerow([self.t, dataB_batch_hat_dist.mean(0)[1].item()])
                self.exp_path['dataB_tail_dist'].csv_writerow([self.t, dataB_batch_tail_dist.mean(0)[1].item()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # directories
    parser.add_argument('--directory', type=str, default='exp_music')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='datasets', help='path of the dataset')
    parser.add_argument('--dataset_A_dir', dest='dataset_A_dir', default='JC_J', help='path of the dataset of domain A')
    parser.add_argument('--dataset_B_dir', dest='dataset_B_dir', default='JC_C', help='path of the dataset of domain B')
    parser.add_argument('--cycleGAN_dir', dest='cycleGAN_dir', default='exp_music/JC_J_JC_C_2023_04_30_20_18_10/checkpoint/trainer_70100.pth')
    parser.add_argument('--classifier_dir', dest='classifier_dir', default='exp_music/JC_J_JC_C_Classifier_2023_04_20_20_35_27/checkpoint/trainer_70100.pth')
    # CNN: JC_J_JC_C_2023_04_13_16_05_02
    # CNN2: JC_J_JC_C_2023_04_30_04_39_38
    # CNN2_base: JC_J_JC_C_2023_04_30_15_36_51
    # rnn: JC_J_JC_C_2023_04_13_21_25_58
    # rnn_base: JC_J_JC_C_2023_04_30_20_18_10
    # lstm: JC_J_JC_C_2023_04_13_22_46_54
    # lstm2: JC_J_JC_C_2023_04_20_14_35_06

    # training
    parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=10, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=10, help='save a model every save_freq epoch')
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=1, help='test a model every eval_freq epoch')
    parser.add_argument('--device', type=str, default='cuda')
    # choose model
    parser.add_argument('--model', dest='model', default='partial', help='three different models, base, partial, full')
    parser.add_argument('--type', dest='type', default='cyclegan', help='cyclegan or classifier')

    args = parser.parse_args()
    # log
    pprint.pprint(vars(args))
    directory_name = args.dataset_A_dir + '_' + args.dataset_B_dir + '_Evaluation'
    if args.debug:
        args.directory = os.path.join(args.directory, directory_name)
    else:
        args.directory = os.path.join(args.directory, directory_name + '_' + now())

    # init trainer
    evaluation = Evaluate(args)

    # save config dict, and model
    logger = ExperimentPath(args.directory)
    logger['config'].json_write(vars(args))
    logger['model_info'].txt_write(repr(evaluation.generatorAB))
    logger['model_info'].txt_write(repr(evaluation.classifier))

    evaluation.evaluate()

    # # train loop
    # try: # handle keyboard interrupt
    #     for epoch in range(args.epoch):
    #         print(f"Epoch {epoch}:")
    #         trainer.train()
    #         if epoch % args.save_freq == 0:
    #             trainer.save()
    #         if epoch % args.test_freq == 0:
    #             trainer.test()
    #     trainer.save()
    # except Exception as e:
    #     tb = traceback.format_exc()
    #     print(tb)
    # finally:
    #     trainer.save()
