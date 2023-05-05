import numpy as np
import pretty_midi
from glob import glob
import torch

from utils.utils import load_npy_data


def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    # fill pitches above C8 and below C0 with zero
    piano_roll = np.pad(piano_roll, (22, 22), 'constant', constant_values=(0, 0))[22:-22]
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches

    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))
        # print('start_time:', start_time)
        # print(len(start_time))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))
        # print('end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('duration each note:', duration)
        # print(len(duration))

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            # print(start_time)
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                # print('i and start_time:', i, start_time[i])
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    # print(j, temp_start_time[j])
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        # print('popped start time:', temp_start_time[j])
                        t.append(j)
                        # print('popped temp_start_time:', t)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])
                # print('popped temp_start_time:', temp_start_time)

        start_time = temp_start_time
        # print('After checking, start_time:', start_time)
        # print(len(start_time))
        end_time = temp_end_time
        # print('After checking, end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('After checking, duration each note:', duration)
        # print(len(duration))

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
                # Add the note to the Instrument object
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                # Add the note to the Instrument object
                instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    # print(max([i.end for i in instrument.notes]))
    # print('tpp, threshold, phrases_end_time:', tpp, threshold, phrase_end_time)


def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100,
                             tempo=120.0, beat_resolution=16):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # Set the piano roll to the Instrument object
    set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def write_piano_rolls_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100,
                              tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(program=program_nums[idx], is_drum=is_drum[idx])
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


if __name__ == '__main__':
    from trainers.cycleGAN_trainer import CycleGANTrainer
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


    cycleGAN_dir = "exp_music/JC_J_JC_C_2023_04_13_21_25_58/checkpoint/trainer_70100.pth"
    cycleGAN_trainer = CycleGANTrainer.load(cycleGAN_dir)
    generatorAB = cycleGAN_trainer.generatorAB
    generatorBA = cycleGAN_trainer.generatorBA
    dataB_batch_hat = generatorAB(dataA_batch.unsqueeze(-1).to('cuda')).squeeze(-1).cpu().detach()            # (b, 64, 84)
    dataB_batch_hat = torch.round(dataB_batch_hat)
    dataB_batch_hat[dataB_batch_hat<0.0] = 0.0
    dataB_batch_hat = dataB_batch_hat.numpy()
    dataA_batch_hat = generatorBA(dataB_batch.unsqueeze(-1).to('cuda')).squeeze(-1).cpu().detach()
    dataA_batch_hat = torch.round(dataA_batch_hat)
    dataA_batch_hat[dataA_batch_hat<0.0] = 0.0
    dataA_batch_hat = dataA_batch_hat.numpy()
    print(dataB_batch_hat)

    write_piano_roll_to_midi(piano_roll=dataA_batch_hat[15].reshape(1, 64, 84), filename="Jazz_fake_short.midi")
