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

#Creates a training data set and test data set from the brightness temperature and vapor, 
def prep_train_test_data(site, year, bright_level, vapor_level, freq_list):

	bright_matrix_vals = []
	bright_matrix_dates = []
	for i in range(len(freq_list)):
			bright_dates, bright_vals = load_brightness_data(site, year , bright_level, freq_list[i], 'brightness')
			
			#smoothen the array using a moving sum with a window length of 3
			
			window = 25
			bright_vals = smooth_array(bright_vals, window)

			#remove the first few and last
			bright_vals = bright_vals[window:-window]
			bright_dates = bright_dates[window:-window]

			bright_matrix_vals.append(bright_vals)
			bright_matrix_dates.append(bright_dates)

	bright_array = np.array(bright_matrix_vals)
	bright_tensor = torch.tensor(bright_array, dtype=torch.float32).T

	#Loads the vapor data into lists.
	vapor_dates, vapor_vals = load_vapor_data(site, year, vapor_level, 'Int. Vapor(cm)')	

	#Shorten the length of the vapor data to match the length of the brightness temperature data
	vapor_dates, vapor_vals = time_align(bright_matrix_dates[0], bright_matrix_vals[0], vapor_dates, vapor_vals)
	vapor_tensor = torch.tensor(vapor_vals, dtype=torch.float32)

	#Replace the NaN's in the data with mean values
	bright_mean = torch.nanmean(bright_tensor, dim=0)
	bright_tensor = torch.where(torch.isnan(bright_tensor), bright_mean, bright_tensor)

	vapor_mean = torch.nanmean(vapor_tensor)
	vapor_tensor = torch.where(torch.isnan(vapor_tensor), vapor_mean, vapor_tensor)

	# Split the data into train/test sets. The first 20% is test data. The last 80% is training data.
	bright_tensor_flipped = bright_tensor.flip(dims=[0])
	vapor_tensor_flipped = vapor_tensor.flip(dims=[0])

	bright_train, bright_test, vapor_train, vapor_test = train_test_split(bright_tensor, vapor_tensor, test_size=0.2, shuffle=False)

	bright_train = bright_train.flip(dims=[0]).float()
	bright_test = bright_test.flip(dims=[0]).float()
	vapor_train = vapor_train.flip(dims=[0]).float()
	vapor_test = vapor_test.flip(dims=[0]).float()

	bright_train = bright_train.float()
	bright_test = bright_test.float()
	vapor_train = vapor_train.float()
	vapor_test = vapor_test.float()

	#Grab the last 20% of date values to use in the time 
	bright_test_size = bright_test.shape[0]
	dates = bright_matrix_dates[0]
	dates = dates[-bright_test_size:]

	return bright_train, bright_test, vapor_train, vapor_test, dates

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
		error.backward() # computes the gradient of the performance/loss function
		optimizer.step() # updates weights based on the gradients

	return model

#Scales and shifts the neural network's generated vapor signal to match that of the known vapor signal
def scale(vapor_eval, vapor_test):

	print('this is vapor eval length')
	print(len(vapor_eval))

	print('this is vapor_test length')
	print(len(vapor_test))

	#Find the mean of the 20 most minimum values in the neural network generated vapor data. Set that mean value to the zero point.
	num_test_min_points = 20
	test_range = 700
	min_values, min_indices = torch.topk(vapor_eval, dim=0, k=num_test_min_points, largest=False)

	ranged_min_means = []
	for i in range (min_values.shape[0]):
		index = min_indices[i]
		ranged_min = vapor_eval[index-test_range:index+test_range]
		ranged_min_mean = ranged_min.mean()
		ranged_min_means.append(ranged_min_mean) 

	mean_of_ranged_min_means = sum(ranged_min_means) / len(ranged_min_means)
	vapor_eval = vapor_eval - mean_of_ranged_min_means
	
	#Scale the neural network generated vapor data to match that of the known vapor data. 
	vapor_eval_mean = vapor_eval.mean()
	vapor_test_mean = vapor_test.mean()

	scale = vapor_test_mean / vapor_eval_mean
	vapor_eval = vapor_eval * scale

	return vapor_eval

#Main

#Settings and Constant Values
SITE = 'queens'
YEAR = '2022'
BRIGHT_LEVEL = 'lv1'
VAPOR_LEVEL = 'lv2'
BASE_DIR = "./NYS_Mesonet_Data"
OUTPUT_DIR = "RFI_detection_plots_and_csv"
FREQ_LIST = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]

#Create the neural network model
model = Model()

#Seperate the data into training and test datasets
bright_train, bright_test, vapor_train, vapor_test, dates = prep_train_test_data(SITE, YEAR, BRIGHT_LEVEL, VAPOR_LEVEL, FREQ_LIST)

#Train the neural network
model = train_model(model, bright_train, vapor_train)

#Test the neural network
with torch.no_grad():

	vapor_eval = model.forward(bright_test)
	vapor_eval = scale(vapor_eval, vapor_test)
	vapor_eval = vapor_eval.detach().numpy()

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
	plt.plot(vapor_test)
	plt.plot()

	plt.show()














