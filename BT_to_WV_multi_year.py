import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import sys
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

#Defines the inputs, activation functions, and layers of the neural network.
class Model(nn.Module):
	def __init__(self, inp=22, hidden=16, out=1):

		super().__init__()

		#Hidden layer 1 
		self.hidden_one = nn.Linear(inp, hidden)

		#Hidden layer one activation
		self.hidden_one_activation = nn.ELU()

		#Output layer
		self.output = nn.Linear(hidden,out)

		#Initialize weights
		self._init_weights()

	#The hidden layer has 16 neurons, each of which accepts 22 inputs (corresponding to the 22 brightness temperature frequencies). 
	#The kaiming initialization sets the 22 weights for each neuron in the hidden layer. The 22 weights for each neuron are then scaled 
	#by the correlation of their corresponding frequency's brightness temperature signal and the vapor signal.
	def _init_weights(self):

		#Kaiming Initialization
		for layer in [self.hidden_one]:
			nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
			nn.init.zeros_(layer.bias)

		nn.init.kaiming_normal_(self.output.weight, nonlinearity='linear')
		nn.init.zeros_(self.output.bias)

	#Runs the neural network.
	def forward(self, inp):
		hidden_lin_one_out = self.hidden_one(inp)
		hidden_act_one_out = self.hidden_one_activation(hidden_lin_one_out)
		output_lin_out = self.output(hidden_act_one_out)
		return output_lin_out

#Retrieving the brightness temperature data.
def load_brightness_data(site, year, level, key, subkey):
    prefix = f"{key}_{subkey}" if level != "lv2" else key
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)
    
    return pd.to_datetime(dates), values

#Retrieving the vapor data.
def load_vapor_data(site, year, level, key):
    prefix = f"{key}"
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)

    return pd.to_datetime(dates), values

def smooth_array(array, window):
    return np.convolve(array, np.ones(window)/window, mode='same')

#Shortens the larger of either the bright or vapor arrays while maintaining time alignment
def time_align(bright_dates, bright_vals, vapor_dates, vapor_vals):

	bigger_dates = vapor_dates
	bigger_vals = vapor_vals
	smaller_dates = bright_dates
	smaller_vals = bright_vals

	# Find insertion positions
	positions = np.searchsorted(bigger_dates, smaller_dates)

	# Get closest timestamps
	aligned_dates = []
	aligned_vals = []
	for i, pos in enumerate(positions):
	    if pos == 0:
	        aligned_dates.append(bigger_dates[0])
	        aligned_vals.append(bigger_vals[0])
	    elif pos == len(bigger_dates):
	        aligned_dates.append(bigger_dates[-1])
	        aligned_vals.append(bigger_vals[-1])
	    else:
	        # Compare neighbors to find the closest
	        prev_val = bigger_dates[pos - 1]
	        next_val = bigger_dates[pos]
	        if abs(smaller_dates[i] - prev_val) <= abs(smaller_dates[i] - next_val):
	            aligned_dates.append(bigger_dates[pos-1])
	            aligned_vals.append(bigger_vals[pos-1])
	        else:
	            aligned_dates.append(bigger_dates[pos])
	            aligned_vals.append(bigger_vals[pos])

	aligned_dates = pd.to_datetime(aligned_dates)

	vapor_dates = aligned_dates
	vapor_vals = aligned_vals

	return aligned_dates, aligned_vals



def remove_days(dates, vals, days_to_remove): 

	mask = ~dates.normalize().isin(days_to_remove)
	dates = dates[mask]
	vals = vals[mask]

	return dates, vals

#Creates a training data set and test data set from the brightness temperature and vapor, 
def prep_train_test_data(site, year, bright_level, vapor_level, freq_list):

	#Loads the brightness temperature data from 22 different frequencies into lists.
	bright_matrix_vals_1 = []
	bright_matrix_dates_1 = []
	bright_matrix_vals_2 = []
	bright_matrix_dates_2 = []
	bright_matrix_vals_3 = []
	bright_matrix_dates_3 = []

	for i in range(len(freq_list)):
			bright_dates_1, bright_vals_1 = load_brightness_data(site, '2022', bright_level, freq_list[i], 'brightness')
			bright_dates_2, bright_vals_2 = load_brightness_data(site, '2023', bright_level, freq_list[i], 'brightness')
			bright_dates_3, bright_vals_3 = load_brightness_data(site, '2024', bright_level, freq_list[i], 'brightness')

			days_to_remove = pd.to_datetime(['2022-12-22', '2023-04-03', '2023-05-01', '2023-06-14', '2023-06-20', '2023-07-08', '2023-11-14', '2023-11-24', '2023-12-03', '2024-01-16', '2024-02-08', '2024-02-14', '2024-02-15', '2024-02-20', '2024-02-26', '2024-03-12', '2024-03-23', '2024-03-26', '2024-04-09' , '2024-04-15', '2024-05-07', '2024-08-24', '2024-10-22'])
			'''
			bright_dates_1, bright_vals_1 = remove_days(bright_dates_1, bright_vals_1, days_to_remove)
			'''
			bright_dates_2, bright_vals_2 = remove_days(bright_dates_2, bright_vals_2, days_to_remove)
			'''
			bright_dates_3, bright_vals_3 = remove_days(bright_dates_3, bright_vals_3, days_to_remove)
			'''
			window = 25
			bright_vals_1 = smooth_array(bright_vals_1, window)
			bright_vals_2 = smooth_array(bright_vals_2, window)
			bright_vals_3 = smooth_array(bright_vals_3, window)

			#remove the first few and last
			bright_dates_1 = bright_dates_1[window:-window]
			bright_vals_1 = bright_vals_1[window:-window]

			#remove the first few and last
			bright_dates_2 = bright_dates_2[window:-window]
			bright_vals_2 = bright_vals_2[window:-window]

			#remove the first few and last
			bright_dates_3 = bright_dates_3[window:-window]
			bright_vals_3 = bright_vals_3[window:-window]

			bright_matrix_vals_1.append(bright_vals_1)
			bright_matrix_dates_1.append(bright_dates_1)

			bright_matrix_vals_2.append(bright_vals_2)
			bright_matrix_dates_2.append(bright_dates_2)

			bright_matrix_vals_3.append(bright_vals_3)
			bright_matrix_dates_3.append(bright_dates_3)

	bright_array_1 = np.array(bright_matrix_vals_1)
	bright_tensor_1 = torch.tensor(bright_array_1, dtype=torch.float32).T

	bright_array_2 = np.array(bright_matrix_vals_2)
	bright_tensor_2 = torch.tensor(bright_array_2, dtype=torch.float32).T

	bright_array_3 = np.array(bright_matrix_vals_3)
	bright_tensor_3 = torch.tensor(bright_array_3, dtype=torch.float32).T

	#Loads the vapor data into lists.
	vapor_dates_1, vapor_vals_1 = load_vapor_data(site, '2022', vapor_level, 'Int. Vapor(cm)')
	vapor_dates_2, vapor_vals_2 = load_vapor_data(site, '2023', vapor_level, 'Int. Vapor(cm)')
	vapor_dates_3, vapor_vals_3 = load_vapor_data(site, '2024', vapor_level, 'Int. Vapor(cm)')

	'''
	vapor_dates_1, vapor_vals_1 = remove_days(vapor_dates_1, vapor_vals_1, days_to_remove)
	'''
	vapor_dates_2, vapor_vals_2 = remove_days(vapor_dates_2, vapor_vals_2, days_to_remove)
	'''
	vapor_dates_3, vapor_vals_3 = remove_days(vapor_dates_3, vapor_vals_3, days_to_remove)
	'''
	#Shorten the length of the vapor data to match the length of the brightness temperature data
	vapor_dates_1, vapor_vals_1 = time_align(bright_matrix_dates_1[0], bright_matrix_vals_1[0], vapor_dates_1, vapor_vals_1)
	vapor_tensor_1 = torch.tensor(vapor_vals_1, dtype=torch.float32)

	vapor_dates_2, vapor_vals_2 = time_align(bright_matrix_dates_2[0], bright_matrix_vals_2[0], vapor_dates_2, vapor_vals_2)
	vapor_tensor_2 = torch.tensor(vapor_vals_2, dtype=torch.float32)

	vapor_dates_3, vapor_vals_3 = time_align(bright_matrix_dates_3[0], bright_matrix_vals_3[0], vapor_dates_3, vapor_vals_3)
	vapor_tensor_3 = torch.tensor(vapor_vals_3, dtype=torch.float32)

	#Replace the NaN's in the data with mean values
	bright_mean_1 = torch.nanmean(bright_tensor_1, dim=0)
	bright_tensor_1 = torch.where(torch.isnan(bright_tensor_1), bright_mean_1, bright_tensor_1)

	bright_mean_2 = torch.nanmean(bright_tensor_2, dim=0)
	bright_tensor_2 = torch.where(torch.isnan(bright_tensor_2), bright_mean_2, bright_tensor_2)

	bright_mean_3 = torch.nanmean(bright_tensor_3, dim=0)
	bright_tensor_3 = torch.where(torch.isnan(bright_tensor_3), bright_mean_3, bright_tensor_3)

	vapor_mean_1 = torch.nanmean(vapor_tensor_1)
	vapor_tensor_1 = torch.where(torch.isnan(vapor_tensor_1), vapor_mean_1, vapor_tensor_1)

	vapor_mean_2 = torch.nanmean(vapor_tensor_2)
	vapor_tensor_2 = torch.where(torch.isnan(vapor_tensor_2), vapor_mean_2, vapor_tensor_2)

	vapor_mean_3 = torch.nanmean(vapor_tensor_3)
	vapor_tensor_3 = torch.where(torch.isnan(vapor_tensor_3), vapor_mean_3, vapor_tensor_3)

	return bright_dates_1, bright_tensor_1, bright_tensor_2, bright_tensor_3, vapor_tensor_1, vapor_tensor_2, vapor_tensor_3

#Train the model
def train_model(model, bright_train, vapor_train):
	
	#Normalize the brightness temperature input data
	mean = bright_train.mean(dim=0, keepdim=True)
	std = bright_train.std(dim=0, keepdim=True)
	bright_train_norm = (bright_train - mean) / (std + 1e-6)

	#Set up the training settings 
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.MSELoss()
	epochs = 100

	#Repeat the trianing process 100 times to allow the weights to converge 
	for i in range(epochs):
		vapor_pred = model.forward(bright_train_norm)

		vapor_pred = vapor_pred.float()
		vapor_pred = vapor_pred.view(-1)

		#Measure the error
		error = criterion(vapor_pred, vapor_train)

		#Change the weights based on the loss value 
		optimizer.zero_grad()
		error.backward()
		optimizer.step()

	return model

#Scales and shifts the neural network's generated vapor signal to match that of the known vapor signal
def scale(vapor_eval, vapor_test):
    # Ensure 1D tensor
    vapor_eval = vapor_eval.flatten()
    vapor_test = vapor_test.flatten()

    num_sections = 1
    section_size = len(vapor_eval) // num_sections
    test_range = 50

    eval_ranged_min_means = []
    test_ranged_min_means = []
    for i in range(num_sections):
        # Define section bounds
        start = i * section_size
        end = (i + 1) * section_size if i < num_sections - 1 else len(vapor_eval)
        eval_section = vapor_eval[start:end]
        test_section = vapor_test[start:end]

        # Find min in this section
        eval_min_val, eval_min_idx_local = torch.min(eval_section, dim=0)
        eval_min_idx = start + eval_min_idx_local.item()  # Convert to global index

        # Define a safe window around the min
        eval_r_start = max(0, eval_min_idx - test_range)
        eval_r_end = min(len(vapor_eval), eval_min_idx + test_range)

        test_min_val, test_min_idx_local = torch.min(test_section, dim=0)
        test_min_idx = start + test_min_idx_local.item()  # Convert to global index

        # Define a safe window around the min
        test_r_start = max(0, test_min_idx - test_range)
        test_r_end = min(len(vapor_test), test_min_idx + test_range)
        
        eval_ranged_min = vapor_eval[eval_r_start:eval_r_end]
       	test_ranged_min = vapor_test[test_r_start:test_r_end]

        # Compute local mean
        eval_ranged_min_mean = eval_ranged_min.mean()
        eval_ranged_min_means.append(eval_ranged_min_mean)

        test_ranged_min_mean = test_ranged_min.mean()
        test_ranged_min_means.append(test_ranged_min_mean)

    # Compute average of all local means (baseline)
    mean_of_eval_ranged_min_means = sum(eval_ranged_min_means) / len(eval_ranged_min_means)
    mean_of_test_ranged_min_means = sum(test_ranged_min_means) / len(test_ranged_min_means)

    #sometimes the nysm vapor data is not situated at 0. so the offset takes into account where the baseline actually is. this requires actually viewing it
    baseline_sub = mean_of_eval_ranged_min_means
    offset = mean_of_test_ranged_min_means
    
    #Baseline sub brigns the vapor_eval minimum to 0. In otherwords, sets the baseline to 0.
    #However, the real vapor data sometimes is not situated at 0. So offset finds the offset from 0
    #and adds it to vapor_eval
    vapor_eval = vapor_eval - baseline_sub + offset

    # Scale to match known vapor data
    vapor_eval_mean = vapor_eval.mean()
    vapor_test_mean = vapor_test.mean()
    scale = vapor_test_mean / vapor_eval_mean

    vapor_eval = vapor_eval * scale


    return vapor_eval

def calculate_error(vapor_eval, vapor_test):
	return np.mean(np.abs(vapor_eval - vapor_test))

#Main

#Settings and Constant Values
SITE = 'queens'
YEAR = '2024'
BRIGHT_LEVEL = 'lv1'
VAPOR_LEVEL = 'lv2'
BASE_DIR = "./NYS_Mesonet_Data"
OUTPUT_DIR = "RFI_detection_plots_and_csv"
FREQ_LIST = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]

#Create the neural network model
model = Model()

#Seperate the data into training and test datasets
bright_dates_1, bright_tensor_1, bright_tensor_2, bright_tensor_3, vapor_tensor_1, vapor_tensor_2, vapor_tensor_3 = prep_train_test_data(SITE, YEAR, BRIGHT_LEVEL, VAPOR_LEVEL, FREQ_LIST)

#Train the neural network

bright_train = torch.cat([bright_tensor_1, bright_tensor_3], dim=0)
vapor_train = torch.cat([vapor_tensor_1, vapor_tensor_3], dim=0)

model = train_model(model, bright_train, vapor_train)

#Test the neural network
with torch.no_grad():

	bright_dates_sections_1 = np.array_split(bright_dates_1, 5)
	bright_dates_1 = bright_dates_sections_1[2]

	print(bright_dates_1[0])
	print(bright_dates_1[-1])

	bright_sections = np.array_split(bright_tensor_2, 5)
	bright_tensor_2 = bright_sections[2]

	vapor_sections = np.array_split(vapor_tensor_2, 5)
	vapor_tensor_2 = vapor_sections[2]

	vapor_eval = model.forward(bright_tensor_2)
	vapor_eval = scale(vapor_eval, vapor_tensor_2)
	vapor_eval = vapor_eval.detach().numpy()
	
	vapor_test = vapor_tensor_2.detach().numpy()

	print('this is error')
	print(calculate_error(vapor_eval, vapor_test))

	#Plot the nerual network
	plt.figure()
	plt.xlabel("Date")
	plt.ylabel("Vapor (cm)")
	plt.title("Taylor Neural Network Vapor Output")
	plt.plot(vapor_eval)
	plt.plot()

	plt.figure()
	plt.xlabel("Date")
	plt.ylabel("Vapor (cm)")
	plt.title("NYSM Neural Network Output")
	plt.plot(vapor_tensor_2)
	plt.plot()

	plt.show()














