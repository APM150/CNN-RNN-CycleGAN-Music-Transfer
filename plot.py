import os
import numpy as np
import matplotlib.pyplot as plt
from utils.filesys_manager import ExperimentPath

# CNN: JC_J_JC_C_2023_04_13_16_05_02
# CNN2: JC_J_JC_C_2023_04_30_04_39_38
# CNN2_base: JC_J_JC_C_2023_04_30_15_36_51
# rnn: JC_J_JC_C_2023_04_13_21_25_58
# rnn_base: JC_J_JC_C_2023_04_30_20_18_10
# lstm: JC_J_JC_C_2023_04_13_22_46_54
# lstm2: JC_J_JC_C_2023_04_20_14_35_06

dirpath = "JC_J_JC_C_2023_04_30_20_18_10"
generator = 'RNN_base_generator'

def plot_loss(filedir):
    try:
        data = np.loadtxt(filedir, delimiter=',')
        x, y = data[:, 0], data[:, 1]
        stats_name = filedir.split('/')[2][:-6] + '_' + \
                     filedir.split('/')[-1][:-4]
        plt.title(stats_name)
        plt.plot(x, y)
    except:
        pass
    plt.grid()
    exp_plot[generator][stats_name].savefig()
    plt.close()

if __name__ == '__main__':
    exp = ExperimentPath("exp_music")
    exp_plot = ExperimentPath("exp_music_plot")
    filepaths = list(exp[dirpath].iglob(f"*_stats/*_loss.csv"))

    for filedir in filepaths:
        try:
            plot_loss(filedir)
        except Exception as e:
            print(e, 'skip', filedir)

    # CNN: JC_J_JC_C_Evaluation_2023_04_20_21_32_59
    # CNN2: JC_J_JC_C_Evaluation_2023_04_30_14_45_44
    # CNN2_base: JC_J_JC_C_Evaluation_2023_04_30_20_37_07
    # RNN: JC_J_JC_C_Evaluation_2023_04_20_21_37_17
    # RNN_base: JC_J_JC_C_Evaluation_2023_04_30_21_27_34
    # LSTM: JC_J_JC_C_Evaluation_2023_04_20_21_37_31
    # LSTM2: JC_J_JC_C_Evaluation_2023_04_20_21_37_43
    print(generator)
    evalpaths = list(exp['JC_J_JC_C_Evaluation_2023_04_30_21_27_34'].iglob("*.csv"))
    for evalpath in evalpaths:
        data = np.loadtxt(evalpath, delimiter=',')
        x, y = data[:, 0], data[:, 1]
        print(evalpath)
        exp_plot[generator]["eval"].txt_write(evalpath+'\n')
        print(y.mean())
        exp_plot[generator]["eval"].txt_write(str(y.mean())+'\n')
