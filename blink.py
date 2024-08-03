# Import Libraries
import os
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, fcluster

# Algorithm Parameters
fs = 250.0
chan_id = 1

flag_soft = True # if True, consider soft blinks as ground truth

blink_len_max = 2.0 # in seconds
blink_len_min = 0.2 # in seconds

delta_init = 100 # in uvolts

corr_threshold_1 = 0.2
corr_threshold_2 = 0.7

std_threshold_window = int(5 * fs)  # in seconds - for each direction


def lowpass(sig, fc, fs, butter_filt_order):
    B, A = butter(butter_filt_order, np.array(fc) / (fs / 2), btype='low')
    return lfilter(B, A, sig, axis=0)

def plot_signals(original_signal, filtered_signal, time):


    # Plot original signal
    plt.figure(figsize=(14, 10))
    plt.plot(time, original_signal[:, chan_id], label='Original Signal')
    plt.title('EEG Signal Before Filtering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Plot filtered signal
    plt.figure(figsize=(14, 7))
    plt.plot(time, filtered_signal[:, chan_id], label='Filtered Signal', color='orange')
    plt.title('EEG Signal After Filtering')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_signal[:, chan_id], label='Original Signal', alpha=0.7)
    plt.plot(time, filtered_signal[:, chan_id], label='Filtered Signal', color='orange', alpha=0.7)
    plt.title('Comparison of Original and Filtered Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


def decode_stim(data_path, file_stim):
    interval_corrupt = []
    blinks = []
    n_corrupt = 0
    with open(os.path.join(data_path, file_stim)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "corrupt":
                n_corrupt = int(row[1])
            elif n_corrupt > 0:
                if float(row[1]) == -1:
                    t_end = data_sig[-1, 0]
                else:
                    t_end = float(row[1])
                interval_corrupt.append([float(row[0]), t_end])
                n_corrupt -= 1
            elif row[0] == "blinks":
                if n_corrupt != 0:
                    print("Error in parsing")
            else:
                blinks.append([float(row[0]), int(row[1])])
    blinks = np.array(blinks)

    return interval_corrupt, blinks

base_path = "data"

for mode, folder in enumerate(['EEG-IO','EEG-VR', 'EEG-VV']):
    data_path = os.path.join(base_path, folder)
    list_of_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and '_data' in f]

    for file_idx, file_sig in enumerate(list_of_files):
        file_stim = file_sig.replace('_data', '_labels')
        print("Processing File:", file_sig, file_stim)

        # Loading Data
        try:
            if mode == 0:  # EEG-IO
                data_sig = np.loadtxt(open(os.path.join(data_path, file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0, 1, 2))
            else:  # EEG-VV or EEG-VR
                data_sig = np.loadtxt(open(os.path.join(data_path, file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0, 1, 2))
                data_sig = data_sig[0:(int(200 * fs) + 1), :]
                data_sig = data_sig[:, 0:3]
                data_sig[:, 0] = np.array(range(0, len(data_sig))) / fs
        except ValueError as e:
            print(f"Error processing {file_sig}: {e}")
            continue
      
        time = data_sig[:, 0]
        original_signal = np.copy(data_sig)  # Make a copy of the original signal for comparison

# Step1: Low Pass Filter
        filtered_signal = np.copy(data_sig)
        filtered_signal[:, 1] = lowpass(filtered_signal[:, 1], 10, fs, 4)
        filtered_signal[:, 2] = lowpass(filtered_signal[:, 2], 10, fs, 4)

# Plot the signals
        plot_signals(original_signal, filtered_signal, time)

        data_len = len(data_sig)

        # Decoding stimulations
        interval_corrupt, gt_blinks = decode_stim(data_path, file_stim)

        def compute_running_std(data_sig, chan_id, fs):
            # Find running std
            std_length = int(0.5 * fs)  # in seconds
            running_std = np.zeros([data_len, 1])
            idx = 0
            while idx < len(data_sig) - std_length:
                running_std[idx] = np.std(data_sig[idx:(idx + std_length), chan_id])
                idx += 1
            running_std[idx:-1] = running_std[idx - 1]

            # fixing the corrupted signal's std
            for idx in range(data_len):
                if running_std[idx] < 1:
                    l_index_lhs = max(0, idx - std_length)
                    l_index_rhs = max(0, (idx - std_length - 2 * std_threshold_window - int(fs)))
                    r_index_lhs = min(data_len, idx + std_length)
                    r_index_rhs = max(0, idx - std_length - int(fs))
                    running_std[l_index_lhs:r_index_lhs] = min(running_std[l_index_rhs:r_index_rhs])
                    idx += std_length - 1

            return running_std

        running_std = compute_running_std(data_sig, chan_id, fs)

        # Function to find peaks
        def args_init(delta_uV):
            args = {}
            args['mintab'], args['maxtab'] = [], []
            args['mn'], args['mx'] = float("inf"), -1 * float("inf")
            args['mnpos'], args['mxpos'] = None, None
            args['min_left'], args['min_right'] = [], []
            args['lookformax'] = True
            args['delta'] = delta_uV
            return args

        def peakdet(time, value, args):
            foundMin = False
            if value > args['mx']:
                args['mx'] = value
                args['mxpos'] = time
            if value < args['mn']:
                args['mn'] = value
                args['mnpos'] = time
            if args['lookformax']:
                if value < args['mx'] - args['delta']:
                    args['maxtab'].append([args['mxpos'], args['mx']])
                    args['mn'] = value
                    args['mnpos'] = time
                    args['lookformax'] = False
            else:
                if value > args['mn'] + args['delta']:
                    args['mintab'].append([args['mnpos'], args['mn']])
                    args['min_left'].append([-1, -1])
                    args['min_right'].append([-1, -1])
                    args['mx'] = value
                    args['mxpos'] = time
                    args['lookformax'] = True
                    foundMin = True
            return foundMin

        ## Finding extreme points
        def find_expoints(stat_min2, data_sig, chan_id):
            # Parameters
            offset_t = 0.00 # in seconds
            win_size = 25
            win_offset = 10
            search_maxlen_t = 1.5 # in seconds

            offset_f = int(offset_t * fs)
            search_maxlen_f = int(search_maxlen_t * fs)
            iters = int(search_maxlen_f / win_offset)

            data_len = len(data_sig)
            p_blinks_t, p_blinks_val = [], []
            for idx in range(len(stat_min2)):
                # x_indR and x_indL are starting points for left and right window
                x_indR = int(fs * stat_min2[idx, 0]) + offset_f
                x_indL = int(fs * stat_min2[idx, 0]) - offset_f
                start_index = max(0, int(fs * stat_min2[idx, 0]) - std_threshold_window)
                end_index = min(int(fs * stat_min2[idx, 0]) + std_threshold_window, data_len)
                stable_threshold = 2 * min(running_std[start_index:end_index])
                #search left window
                for i in range(iters):
                    for j in range(win_size):
                        if (x_indL - i * win_offset + j > 0) and (x_indL - i * win_offset + j < data_len):
                            if data_sig[x_indL - i * win_offset + j, chan_id] > stable_threshold:
                                if (x_indL - i * win_offset + j) / fs < stat_min2[idx, 0]:
                                    stat_min2[idx, 1] = min(data_sig[int(fs * stat_min2[idx, 0]):int(fs * stat_min2[idx, 0]) + search_maxlen_f, chan_id])
                                    p_blinks_t.append(stat_min2[idx, 0])
                                    p_blinks_val.append(stat_min2[idx, 1])
                                    break
                    else:
                        continue
                    break
                #search right window
                for i in range(iters):
                    for j in range(win_size):
                        if (x_indR + i * win_offset + j > 0) and (x_indR + i * win_offset + j < data_len):
                            if data_sig[x_indR + i * win_offset + j, chan_id] > stable_threshold:
                                if (x_indR + i * win_offset + j) / fs > stat_min2[idx, 0]:
                                    stat_min2[idx, 1] = min(data_sig[int(fs * stat_min2[idx, 0]):int(fs * stat_min2[idx, 0]) + search_maxlen_f, chan_id])
                                    p_blinks_t.append(stat_min2[idx, 0])
                                    p_blinks_val.append(stat_min2[idx, 1])
                                    break
                    else:
                        continue
                    break

            p_blinks = np.array([p_blinks_t, p_blinks_val]).T
            return p_blinks

        args = args_init(delta_init)
        for idx in range(data_len):
            time, val = data_sig[idx, 0], data_sig[idx, chan_id]
            found_min = peakdet(time, val, args)
            if found_min:
                break

        p_blinks = find_expoints(np.array(args['mintab']), data_sig, chan_id)
        print("Detected Blinks:", p_blinks)
