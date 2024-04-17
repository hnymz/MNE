import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

#%%
time_start = 150
time_end = 320

# Get the EEG data from the epochs
working_dir = 'C:/Users/ausra/Desktop/internship/data_analysis'
behavioral_data = pd.read_csv(f'{working_dir}/behavioral_data.csv')
trial_index = behavioral_data['trial_index']
accuracy = behavioral_data['accuracy']
rt = behavioral_data['rt']
# The resulting 3D array have 3 dimensions representing epochs, channels, and time points in each epoch
epochs_ica = mne.read_epochs(f'{working_dir}/preprocessing/ICA_filter(1-10)_epo.fif')
data = epochs_ica.get_data()
print("Shape of data:", data.shape)
windowed_data = data[trial_index,:,time_start+200:time_end+200]

# Calculate SVD using trial-averaged EEG data
averaged_data = np.mean(windowed_data, axis=0)
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
# Plot the averaged N200 using trial-averaged data
averaged_N200 = np.tensordot(averaged_data, first_component, axes=(1, 0))
latency_averaged_N200 = np.argmin(averaged_N200) + time_start
plt.figure(figsize=(10, 6))
plt.plot(np.arange(time_start, time_end), averaged_N200)
plt.xlabel('Post-stimulus time (ms)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('N200 waveform', fontsize=14)
plt.show()

#%%
bad_counter = 0
latency_list = []
trial_list = []

# Apply SVD weights to EEG data in every trial
for trial in range(0, windowed_data.shape[0]):
    trial_data = np.transpose(windowed_data[trial, :, :]) # T * C matrix of EEG data in every trial
    weighted_trial_data = np.tensordot(trial_data, first_component, axes=(1, 0)) # T * 1 matrix (shape: (T,))

    latency = np.argmin(weighted_trial_data)
    latency_list.append(latency)
    # Detect if the latencies are found at boundaries
    if latency == 0 or latency == len(weighted_trial_data)-1:
        bad_counter += 1
    else:
        trial_list.append(trial)
        # Plot the waveform for every trial
        #plt.figure(figsize=(10, 6))
        #plt.plot(np.arange(time_start, time_end), weighted_trial_data)
        #plt.xlabel('Post-stimulus time (ms)')
        #plt.ylabel('Amplitude')
        #plt.title('N200 waveform')
        #plt.savefig(f'{working_dir}/template_matching/trial_{trial}.png')
        #plt.close()

drop_percentage = bad_counter / (windowed_data.shape[0]) * 100
print(drop_percentage)

trial_for_modelling = [trial_index[i] for i in trial_list]
accuracy_for_modelling = [accuracy[i] for i in trial_list]
rt_for_modelling = [rt[i] for i in trial_list]
latency_modelling = [latency_list[i] + time_start for i in trial_list]
behavioral_data_modelling = pd.DataFrame({'trial_index': trial_for_modelling, 'accuracy': accuracy_for_modelling,
                                          'rt': rt_for_modelling, 'latency': latency_modelling})
behavioral_data_modelling.to_csv("C:/Users/ausra/Desktop/internship/data_analysis/behavioral_data_modelling.csv", index=False)

#%%
# Plot the distribution of N200 latencies in a histogram
latency_list = [i + 151 for i in latency_list]
num_bins = 30
bin_width = (max(latency_list) - min(latency_list) + 1) / num_bins

plt.figure(figsize=(10, 6))
plt.hist(latency_list,
         bins=np.arange(min(latency_list), max(latency_list) + bin_width + 1e-9, bin_width),
         align='left', rwidth=1, density=True)
sns.kdeplot(latency_list, color='red', label='Density Curve')
plt.xlabel('Post-stimulus time (ms)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Distribution of N200 latencies', fontszie=14)
plt.show()

#%%
#############################################################################
# average all of the trials for a grand-average (collapsed across conditions)
evoked = epochs_ica.average()
evoked.plot()
evoked.plot_joint()

# select a specific electrode to plot
evoked.plot(picks = ['Fz'])

#%%
# Import template waveform data
waveform_file = "C:/Users/ausra/Desktop/template_data_waveform.xlsx"
reference_waveform_data = pd.read_excel(waveform_file).values.flatten()
# Reminder: reference_waveform_data contains both pre-stimulus 100 ms and post-stimulus 1000 ms
# N200 time window: 126-275 post-stimulus
windowed_reference = reference_waveform_data[time_start+100:time_end+100]
# Standardize
windowed_reference_waveform_data = (windowed_reference - np.mean(windowed_reference)) / np.std(windowed_reference)

# Plot the reference waveform
plt.figure(figsize=(10, 6))
plt.plot(np.arange(-100, 300), reference_waveform_data[0:400])
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('N200 template waveform')
plt.axvline(x=time_start, color='black', linestyle='--')  # N200 time window
plt.axvline(x=time_end, color='black', linestyle='--')
plt.show()
