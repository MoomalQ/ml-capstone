import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_timestep_col(dataframe, array_only = False):

	if array_only:
		timesteps = dataframe.index.to_series().diff().dt.total_seconds()
		return timesteps

	else:
		dataframe['timestep'] = dataframe.index.to_series().diff().dt.total_seconds()
		return dataframe

def get_scaled_array(dataframe, scaler = StandardScaler()):

	scaler.fit(dataframe)

	scaled_array = scaler.transform(dataframe)

	return scaled_array

def create_array_dict(label_list, array):

	num_labels = len(label_list)

	labelled_arrays = {label_list[i]: array[:,i] for i in range(num_labels)}

	return labelled_arrays

def create_skip_mask(timestep_array, skip_threshold):

	skip_mask = timestep_array > skip_threshold

	return skip_mask

def split_by_skips(array, skip_mask):

	'''
	Takes a timeseries array and splits an array into subarrays separated by timestep skips
	'''

    # Find indices where there there is a change from normal timesteps to skipped timestep or vice versa
    indices = np.nonzero(skip_mask[1:] != skip_mask[:-1])[0] + 1
    
    # Split array into sub arrays at these indices (subarrays will be normal, skipped, normal, skipped etc.)
    split_array = np.split(array, indices)
    
    # Only keep arrays where the data does not contain any time series skips
    split_array = split_array[0::2] if not skips[0] else split_array[1::2]
    
    return split_array_list

def drop_arrays_smaller_than_window(array_list, window_size):

	data_points_dropped = sum([len(subarray) for subarray in array if len(subarray) < window_size])
	print('{} data points will be dropped.'.format(data_points_dropped))

	clean_array_list = [subarray for subarray in split_mains if len(subarray) > window_size]

	return clean_array_list

def create_sliding_windows(array, window_size):

    timesteps = len(array)

    indices = np.arange(window_size)[None,:] + np.arange(timesteps - window_size + 1)[:, None]
    
    return array[indices]

def create_input_array(array_list, window_size):

	array_windows_list = []

	for subarray in array_list:
		subarray_windows = create_sliding_windows(subarray, window_size)
		array_windows_list.append(subarray_windows)

	input_array = np.concatenate(array_windows_list, axis = 0)

	return input_array