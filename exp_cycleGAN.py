import argparse
import os
import traceback
import json
import pprint

from trainers.cycleGAN_trainer import CycleGANTrainer
from trainers.rnn_trainer import RNNTrainer
from utils.utils import now
from utils.filesys_manager import ExperimentPath


parser = argparse.ArgumentParser(description='')
# directories
parser.add_argument('--directory', type=str, default='exp_music')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--dataset_dir', dest='dataset_dir', default='datasets', help='path of the dataset')
parser.add_argument('--dataset_A_dir', dest='dataset_A_dir', default='JC_J', help='path of the dataset of domain A')
parser.add_argument('--dataset_B_dir', dest='dataset_B_dir', default='JC_C', help='path of the dataset of domain B')
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
parser.add_argument('--generator', dest='generator', default='RNN')
parser.add_argument('--type', dest='type', default='cyclegan', help='cyclegan or classifier')
# model hyperparameters
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--betas', dest='betas', type=float, default=(0.5, 0.999), help='initial momentum decay rates for adam')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--sigma_c', dest='sigma_c', type=float, default=1.0, help='sigma of gaussian noise of classifiers')
parser.add_argument('--sigma_d', dest='sigma_d', type=float, default=1.0, help='sigma of gaussian noise of discriminators')
parser.add_argument('--gamma', dest='gamma', type=float, default=1.0, help='weight of extra discriminators')


if __name__ == '__main__':
    args = parser.parse_args()
    # log
    pprint.pprint(vars(args))
    directory_name = args.dataset_A_dir + '_' + args.dataset_B_dir
    if args.debug:
        args.directory = os.path.join(args.directory, directory_name)
    else:
        args.directory = os.path.join(args.directory, directory_name + '_' + now())

    # init trainer
    if args.generator != "CNN" and args.generator != "CNNv2":
        trainer = RNNTrainer(args)
    else:
        trainer = CycleGANTrainer(args)

    # save config dict, and model
    logger = ExperimentPath(args.directory)
    logger['config'].json_write(vars(args))
    logger['model_info'].txt_write(repr(trainer.discriminatorA))
    logger['model_info'].txt_write(repr(trainer.generatorAB))

    # train loop
    try: # handle keyboard interrupt
        for epoch in range(args.epoch):
            print(f"Epoch {epoch}:")
            trainer.train()
            if epoch % args.save_freq == 0:
                trainer.save()
            if epoch % args.test_freq == 0:
                trainer.test()
        trainer.save()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
    finally:
        trainer.save()
