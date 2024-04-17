import numpy as np
import mne
from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components

#%%
# Define the file directory where you saved the data
file_path = "C:/Users/ausra/Desktop/internship/data_analysis/raw_data/ssc_07_03_2024.bdf"
raw = mne.io.read_raw_bdf(file_path, preload = True)
raw.resample(1000)

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

#%%
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
# ICA cannot be applied without montage!
# Show the distribution of electrodes in a 3D view
montage = mne.channels.make_standard_montage('biosemi64')
montage.plot(kind = '3d')
# Apply the montage to the data
filt_raw.set_montage(montage)
# 'bad' channels will be red, click individual dots to see the label
filt_raw.plot_sensors()

#%%
# ICA
nchan = len(filt_raw.pick_types(eeg=True).ch_names)
nbad = len(filt_raw.info['bads'])
ncomp = (nchan - nbad) - 1 # Minus 1 because we use average re-reference
print(ncomp)

ica = mne.preprocessing.ICA(n_components=ncomp, random_state=99, method='infomax', fit_params=dict(extended=True))
ica.fit(filt_raw)
gui = label_ica_components(filt_raw, ica)

ic_labels = label_components(filt_raw, ica, method="iclabel")
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
filt_raw_ica = filt_raw.copy()
ica.apply(filt_raw_ica)

#%%
# Re-filter for N200
re_filt_raw_ica = filt_raw_ica.copy()
re_filt_raw_ica.load_data().filter(l_freq = 1, h_freq = 10)

#%%
# Epoching
# Only create epochs for the onset of gabors
event_id_epoch = {'low_freq_gabor': 2, 'high_freq_gabor': 3}
epochs = mne.Epochs(re_filt_raw_ica,
                    events,
                    event_id=event_id_epoch,
                    tmin=-0.2, tmax=2, # The maximum length of a trial is 2s
                    proj=False, baseline=(None, 0), # Baseline correction from min to 0
                    preload=True, reject=None, detrend=1) # Linear detrending

# How many epochs of each type were created?
high_freq = np.count_nonzero(epochs.events[:,2] == 2) # high-freq gabor
low_freq = np.count_nonzero(epochs.events[:,2] == 3) # low-freq gabor

# n_epochs = 3 controls the number of epochs displayed in the window at the same time
epochs.plot(n_epochs=3, events=True, event_id=event_id)

#%%
# Save the preprocessed file
file_name = "C:/Users/ausra/Desktop/internship/data_analysis/preprocessing/ICA_filter(1-10)_epo.fif"
epochs.save(file_name, overwrite=True)

