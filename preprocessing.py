import os
import numpy as np
import mne
from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components
import matplotlib.pyplot as plt

#%%
# Define the file directory where you saved the data
working_dir = "C:/Users/ausra/Desktop/internship/data_analysis/raw_data/"
file = "ssc_07_03_2024.bdf" # the file we're going to use
fpath = os.path.join(working_dir, file)
raw = mne.io.read_raw_bdf(fpath, preload = True)

#%%
# Print basic information
print(raw)
print(raw.info)
print(raw.info['ch_names'])
print('there are {} channels'.format(len(raw.info['ch_names']))) # of channels
print(raw.info['sfreq']) # the sampling frequency
print(raw.info['bads'])

# Delete empty extra electrodes and bad channels
raw.drop_channels(['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EOGL', 'EOGR', 'EOGT', 'EOGB', 'Iz'])
print(raw.info['ch_names'])

# Find events
events = mne.find_events(raw, stim_channel="Status")
print(events)

# Re-reference data to average of all electrodes (default)
raw.set_eeg_reference(ref_channels = "average")

# Bandpass filter
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq = 1, h_freq = 100)

# Preparing for epoching (adding events)
event_id = {'low_freq_gabor': 2, 'high_freq_gabor': 3, 'annulus': 4,
            'button_high_1': 16386, 'button_high_2': 16387, 'button_high_3': 16388,
            'button_low_1': 4098, 'button_low_2': 4099, 'button_low_3': 4100}
# Visualize the time course of all events (not very necessary)
fig = mne.viz.plot_events(events,
                          sfreq=filt_raw.info['sfreq'],
                          event_id=event_id)

# Mark bad channels manually
filt_raw.plot(event_id=event_id, events=events)
print(filt_raw.info['bads'])

#%%
# Show the distribution of electrodes in a 3D view
montage = mne.channels.make_standard_montage('biosemi64')
montage.plot(kind = '3d')
# Apply the montage to the data
filt_raw.set_montage(montage)
# 'bad' channels will be red, click individual dots to see the label
filt_raw.plot_sensors()
# If bad channels are not near the edge, they need to be interpolated
# Otherwise, better remove them directly and apply another re-reference
#filt_raw.set_eeg_reference(ref_channels = "average")

#%%
# Epoching
# Only create epochs for the onset of gabors
event_id_epoch = {'low_freq_gabor': 2, 'high_freq_gabor': 3}
epochs = mne.Epochs(filt_raw,
                    events,
                    event_id = event_id_epoch,
                    tmin = -0.2, tmax = 2, # The maximum length of a trial is 2s
                    proj = False, baseline = None, # No baseline correction at this step
                    preload = True, reject = None, detrend = 1)
print(epochs.events)

# How many epochs of each type were created?
print(np.count_nonzero(epochs.events[:,2] == 2)) # high-freq gabor
print(np.count_nonzero(epochs.events[:,2] == 3)) # low-freq gabor
print(np.count_nonzero(epochs.events[:,2] == 4)) # annulus

# n_epochs = 3 controls the number of epochs displayed in the window at the same time
epochs.plot(n_epochs=3, events=True, event_id=event_id)

#%%
# ICA
nchan = len(epochs.pick_types(eeg=True).ch_names)
nbad = len(epochs.info['bads'])
ncomp = (nchan - nbad) - 1 # Minus 1 because we use average re-reference
print(ncomp)

ica = mne.preprocessing.ICA(n_components=ncomp, random_state=99, method='infomax', fit_params=dict(extended=True))
ica.fit(epochs)
gui = label_ica_components(epochs, ica)

ic_labels = label_components(epochs, ica, method="iclabel")
print(ic_labels["labels"]) # The number of labels is equal to the number of ICA channels (i.e., ncomp)
# "labels" contains labels for each component, such as "brain", "eye blink", "channel noise" etc.
# label_components also return "y_pred_proba",
# which is the estimated predicted probability of the output class for each independent component.

# Exclude labels that are not "brain" or "other"
# Labels like "eye blink", "channel noise", "muscle artifact" will be excluded
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")

# Plot the properties of a single component to inspect it
# picks=[1,2,3,4,5] refers to the indices of components that you want to plot
#ica.plot_properties(epochs, picks=[1,2,3,4,5])
#ica.plot_sources(epochs)

ica.exclude = exclude_idx
ica.plot_components()

#%%
# Apply ICA to preprocessed epoched data
epochs_ica = epochs.copy()
ica.apply(epochs_ica)

# Baseline correction
baseline = (None, 0)
trad_low = epochs_ica["low_freq_gabor"].average().apply_baseline(baseline)
trad_high = epochs_ica["high_freq_gabor"].average().apply_baseline(baseline)
epochs_ica.plot(events=True, event_id=event_id)

# Save the preprocessed file
file_name = "C:/Users/ausra/Desktop/internship/data_analysis/preprocessing/preprocessed_data_epo.fif"
epochs_ica.save(file_name, overwrite = True)

#%%
# Get the EEG data from the epochs
# The resulting 3D array have 3 dimensions representing epochs, channels, and time points in each epoch
data = epochs_ica.get_data()
print("Shape of data:", data.shape)

#%%
# Calculate SVD using trial-averaged EEG data
averaged_data = np.mean(data, axis=0)
# The above line can be replaced by: evoked = epochs_ica.average() to average data on all trials
averaged_data = np.transpose(averaged_data) # T * C matrix of trial-averaged EEG
U, s, V = np.linalg.svd(averaged_data, full_matrices=False)

def perexp(s):
    pexp = np.square(s) / float(np.sum(np.square(s)))
    return pexp
# Print the percentage of variance explained by the first component
print(perexp(s)[0])

first_component = V[0, :] # C * 1 matrix of the first right singular vector

#%%
# Apply SVD weights to EEG data in every trial
for trial in range(110, 111): # data.shape[0]
    trial_data = np.transpose(data[trial, :, :]) # T * C matrix of EEG data in every trial
    weighted_trial_data = np.tensordot(trial_data, first_component, axes=(1, 0)) # T * 1 matrix

indices = np.arange(weighted_trial_data.shape[0])
window_size = 50
smoothed_values = np.convolve(weighted_trial_data*1000000, np.ones(window_size)/window_size, mode='valid')
adjusted_indices = indices[(window_size//2):-(window_size//2) + 1]

plt.figure(figsize=(8, 6))
plt.plot(adjusted_indices, smoothed_values)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('SVD weighted single-trial potential (μV)', fontsize=16)
new_xticks = np.arange(-200, 2001, 200)
old_xticks = np.linspace(0, 4507, num=len(new_xticks))
plt.xticks(old_xticks, new_xticks, fontsize=14)
plt.yticks(fontsize=14)
for i, xtick in enumerate(new_xticks):
    if xtick == 0:
        plt.axvline(x=old_xticks[i], color='red') 
        plt.text(old_xticks[i] + 50, np.max(smoothed_values), 'Onset of stimuli', color='red', fontsize=12, rotation=0, verticalalignment='bottom')
plt.show()


