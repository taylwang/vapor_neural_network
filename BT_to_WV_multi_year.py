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

#The neural network is composed of only two layers: the input layer and the output layer. The input layer has 16 neurons,
#each receving an input from each of the frequencies. The input layer uses the ELU activation function since since it is mostly
#non negative. The mostly positive ELU function reflects the fact that the vapor values are all positive. 

#--------------- Neural Network Model --------------- 
class Model(nn.Module):
	def __init__(self, num_inputs, num_input_neurons, num_output_neurons, activation_function):

		super().__init__()

		#Hidden layer 1 
		self.hidden_one = nn.Linear(num_inputs, num_input_neurons)

		#Hidden layer one activation
		self.hidden_one_activation = activation_function

		#Output layer
		self.output = nn.Linear(num_input_neurons,num_output_neurons)

		#Initialize weights
		self._init_weights()

	def _init_weights(self):

		#The neural network learns by determining the error between its predicted output and the actual output
		#It then corrects weights depending on the output. However, the weights must first be initialized to begin
		#this process. Kaiming initialization is used to initialize weights in a layer that used the relu activation function
		for layer in [self.hidden_one]:
			nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
			nn.init.zeros_(layer.bias)

		nn.init.kaiming_normal_(self.output.weight, nonlinearity='linear')
		nn.init.zeros_(self.output.bias)

	#Run the neural network
	def forward(self, inp):
		hidden_lin_one_out = self.hidden_one(inp)
		hidden_act_one_out = self.hidden_one_activation(hidden_lin_one_out)
		output_lin_out = self.output(hidden_act_one_out)
		return output_lin_out

#--------------- Data Loading Functions --------------- 

#Retrieve brightness temperature data
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

#Retrieve vapor data
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

#--------------- Auxilary Functions --------------- 

#Matches the length of the brightness temperature and vapor data while maintaining time alignment
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

#Remove specific dates and their corresponding values from the dataset
def remove_days(dates, vals, days_to_remove): 

	mask = ~dates.normalize().isin(days_to_remove)
	dates = dates[mask]
	vals = vals[mask]

	return dates, vals

#Creates a training data set and test data set from the brightness temperature and vapor, 
def prep_train_test_data(site, year, bright_level, vapor_level, freq_list):

	#List used to hold the brightness temperature data from different freuqencies in one place
	bright_matrix_vals_1 = []
	bright_matrix_dates_1 = []
	bright_matrix_vals_2 = []
	bright_matrix_dates_2 = []
	bright_matrix_vals_3 = []
	bright_matrix_dates_3 = []

	#For loop used to load brightness temperature data from different frequencies of interest
	for i in range(len(freq_list)):

			#Loads the brigthness temperature data from a specific location, year, and frequency
			bright_dates_1, bright_vals_1 = load_brightness_data(site, '2022', bright_level, freq_list[i], 'brightness')
			bright_dates_2, bright_vals_2 = load_brightness_data(site, '2023', bright_level, freq_list[i], 'brightness')
			bright_dates_3, bright_vals_3 = load_brightness_data(site, '2024', bright_level, freq_list[i], 'brightness')

			#Optional: Remove the dates and their corresponding values from the datasets. This was incorperated into the code to remove RFI
			'''
			days_to_remove = pd.to_datetime(['2022-12-22', '2023-04-03', '2023-05-01', '2023-06-14', '2023-06-20', '2023-07-08', '2023-11-14', '2023-11-24', '2023-12-03', '2024-01-16', '2024-02-08', '2024-02-14', '2024-02-15', '2024-02-20', '2024-02-26', '2024-03-12', '2024-03-23', '2024-03-26', '2024-04-09' , '2024-04-15', '2024-05-07', '2024-08-24', '2024-10-22'])
			bright_dates_1, bright_vals_1 = remove_days(bright_dates_1, bright_vals_1, days_to_remove)
			bright_dates_2, bright_vals_2 = remove_days(bright_dates_2, bright_vals_2, days_to_remove)
			bright_dates_3, bright_vals_3 = remove_days(bright_dates_3, bright_vals_3, days_to_remove)
			'''

			#The brigthness temperature data is very noisy. Training the neural network with noisy brigthness temperature data
			#causes a noisy neural network output. The noise is reduced by performing a moving average over the data.
			#The smooth_array function performs this denoising/data smoothening
			window = 25
			bright_vals_1 = np.convolve(bright_vals_1, np.ones(window)/window, mode='same')
			bright_vals_2 = np.convolve(bright_vals_2, np.ones(window)/window, mode='same')
			bright_vals_3 = np.convolve(bright_vals_3, np.ones(window)/window, mode='same')

			#The moving average is inaccurate at indices that are less than a window length away from the data/array edges.
			#Near the edges, the moving average attempts to probe outside the array. Points outside the array are automatically considered 0,
			#so the computed average value at the index of interest is much less than it should be
			#Therefore, points near the edges are removed from the data to avoid confusing the neural network during training. `
			bright_dates_1 = bright_dates_1[window:-window]
			bright_vals_1 = bright_vals_1[window:-window]

			bright_dates_2 = bright_dates_2[window:-window]
			bright_vals_2 = bright_vals_2[window:-window]

			bright_dates_3 = bright_dates_3[window:-window]
			bright_vals_3 = bright_vals_3[window:-window]

			#Add the processed input data into the list 
			bright_matrix_vals_1.append(bright_vals_1)
			bright_matrix_dates_1.append(bright_dates_1)

			bright_matrix_vals_2.append(bright_vals_2)
			bright_matrix_dates_2.append(bright_dates_2)

			bright_matrix_vals_3.append(bright_vals_3)
			bright_matrix_dates_3.append(bright_dates_3)

	#Turn the list of data sets at different freqauncies into a matrix
	#Then turn that matrix into a tensor, which is the format the neural network works in
	bright_array_1 = np.array(bright_matrix_vals_1)
	bright_tensor_1 = torch.tensor(bright_array_1, dtype=torch.float32).T

	bright_array_2 = np.array(bright_matrix_vals_2)
	bright_tensor_2 = torch.tensor(bright_array_2, dtype=torch.float32).T

	bright_array_3 = np.array(bright_matrix_vals_3)
	bright_tensor_3 = torch.tensor(bright_array_3, dtype=torch.float32).T

	#Loads the vapor data from a specific location and year
	vapor_dates_1, vapor_vals_1 = load_vapor_data(site, '2022', vapor_level, 'Int. Vapor(cm)')
	vapor_dates_2, vapor_vals_2 = load_vapor_data(site, '2023', vapor_level, 'Int. Vapor(cm)')
	vapor_dates_3, vapor_vals_3 = load_vapor_data(site, '2024', vapor_level, 'Int. Vapor(cm)')

	#Interference can effect the neural network's training and output.
	#The remove_days function removes instances of interference from the data
	'''
	vapor_dates_1, vapor_vals_1 = remove_days(vapor_dates_1, vapor_vals_1, days_to_remove)
	vapor_dates_2, vapor_vals_2 = remove_days(vapor_dates_2, vapor_vals_2, days_to_remove)
	vapor_dates_3, vapor_vals_3 = remove_days(vapor_dates_3, vapor_vals_3, days_to_remove)
	'''

	#The neural network requires the brigthness temperature data and the vapor data to be the same length
	#because the brigthness temperature at one time point determines the vapor data at the same time point.
	#However, the brigthness temperature data and vapor data supplied by NYSM are rarely the same length.
	#It is necessary to shorten the longer of the two data sets so that they are the same lengths, while 
	#ensuring the dates at corresponding indices remain nearly identical.
	#The time align function matches the brightness temperature and vapor lengths and dates.
	vapor_dates_1, vapor_vals_1 = time_align(bright_matrix_dates_1[0], bright_matrix_vals_1[0], vapor_dates_1, vapor_vals_1)
	vapor_tensor_1 = torch.tensor(vapor_vals_1, dtype=torch.float32)

	vapor_dates_2, vapor_vals_2 = time_align(bright_matrix_dates_2[0], bright_matrix_vals_2[0], vapor_dates_2, vapor_vals_2)
	vapor_tensor_2 = torch.tensor(vapor_vals_2, dtype=torch.float32)

	vapor_dates_3, vapor_vals_3 = time_align(bright_matrix_dates_3[0], bright_matrix_vals_3[0], vapor_dates_3, vapor_vals_3)
	vapor_tensor_3 = torch.tensor(vapor_vals_3, dtype=torch.float32)


	#Sometimes the brigthness temperature and vapor data have NaN's, which disrupt further processing in the data.
	#These NaN's are repalced by the mean value of the data
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

#This is still under works
#The neural network can accurately predict the shape of the output. However, the scale and the offset 
#rarely match the actual scale and offset of the output. Post processing of the neural network output
#must be done to properly scale and offset the output. 
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

#--------------- Train Neural Network --------------- 

def train_model(bright_train, vapor_train, optimizer, criterion, epochs):
	
	#Normalize the brightness temperature for each frequency. This puts all frequency data sets in the same scale
	#allowing the model to learn evenly across frequencies. It also reduces the risk of diverging gradients.

	mean = bright_train.mean(dim=0, keepdim=True)
	std = bright_train.std(dim=0, keepdim=True)
	bright_train_norm = (bright_train - mean) / (std + 1e-6)

	for i in range(epochs):
		vapor_pred = model.forward(bright_train_norm)

		vapor_pred = vapor_pred.float()
		vapor_pred = vapor_pred.view(-1)

		#Measure the error
		error = criterion(vapor_pred, vapor_train)
		
		#Clears old gradients to recompute the gradients
		optimizer.zero_grad()

		#Comptutes the gradient of the error with respect to each parameter
		error.backward()

		#Updates the parameters based on the gradient. A larger gradient for a specific parameter
		#generally indicates the weight associating with that parameter must be changed more. 
		optimizer.step()

	return model

#--------------- Data Parameters --------------- 
SITE = 'queens'
YEAR = '2024'
BRIGHT_LEVEL = 'lv1'
VAPOR_LEVEL = 'lv2'
BASE_DIR = "./NYS_Mesonet_Data"
OUTPUT_DIR = "RFI_detection_plots_and_csv"
FREQ_LIST = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]

#--------------- Neural Network Structure Parameters --------------- 

#The number of inputs is the number of brightness temperature frequencies used to predict the vapor
NUM_INPUTS = 22

#There is no exact equation to determine the number of input neurons. Too little input neurons,
#and the neural network is incapable of predicting complex behavior. Too many inpput neurons,
#and the nueral network runs too slowly. A good rule of thumb is to 
#pick a value between the number of inputs and the number of outputs, and experiment from there
NUM_INPUT_NEURONS = 16

#The number of output neurons is the number of output types. Vapor is just one value, so it consist
#of just one output type
NUM_OUTPUT_NEURONS = 1

#Activation functions should model the relationship between the input and the output. The brightness temperature vapor
#are always positive so the activation function should only exhibit change in the first quadrant (positive regions).
#ELU shows a linear 
#so the activation function should mostly be positive. And as the brig
ACTIVATION_FUNCTION = nn.ELU()

#--------------- Create the Neural Network --------------- 

#Create the neural network model
model = Model(NUM_INPUTS, NUM_INPUT_NEURONS, NUM_OUTPUT_NEURONS, ACTIVATION_FUNCTION)

#--------------- Neural Network Weight Correction Parameters --------------- 

#The learning rate determines the scale at which the weights are changed. The learning rate should also be small enough
#such that the weights converge rather than bounce between extreme values. The learning rate should als0 be big enough
#such that a reasonable number of epochs lead to convergence. 

LEARNING_RATE = 0.001

#The optimizer is the 'rule' for how the weights are changed based on the error.
#Adam is a well-known optimizer in pytorch that keeps track of the average of past gradients 
#and adaptive learning rates for each parameter.
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#The neural network uses mean squared error as a metric for how far off the predicted output is from the actual output,
#and how much weight correction is needed to improve the neural network.
CRITERION = nn.MSELoss()

#Each time the neural network changes its weights, it may undo corrections it previously did. Despite this,
#the neural network's weights tends to converge to a certain value. By repeating the training multiple times,#the weights convergence becomes more accurate and more consistent
EPOCHS = 100

#--------------- Main --------------- 

#Seperate the data into training and test datasets
bright_dates_1, bright_tensor_1, bright_tensor_2, bright_tensor_3, vapor_tensor_1, vapor_tensor_2, vapor_tensor_3 = prep_train_test_data(SITE, YEAR, BRIGHT_LEVEL, VAPOR_LEVEL, FREQ_LIST)

#Combine the training data from different years into one dataset
bright_train = torch.cat([bright_tensor_1, bright_tensor_3], dim=0)
vapor_train = torch.cat([vapor_tensor_1, vapor_tensor_3], dim=0)

#Train the neural network
model = train_model(model, bright_train, vapor_train, OPTIMIZER, CRITERION, EPOCHS)

with torch.no_grad():

	#Select brightness temperature data to input into the neural network
	bright_sections = np.array_split(bright_tensor_2, 5)
	bright_tensor_2 = bright_sections[2]

	#Select the corresponding actual vapor data to compare the predicted vapor output with the actual vapor output
	vapor_sections = np.array_split(vapor_tensor_2, 5)
	vapor_tensor_2 = vapor_sections[2]

	#Pass the brightness temperature data into the the neural network to predict the vapor
	vapor_predicted = model.forward(bright_tensor_2)
	
	#Properly scale the predicted vapor to match the actual vapor
	vapor_predicted = scale(vapor_predicted, vapor_tensor_2)
	
	#Turn the predicted 
	vapor_predicted = vapor_eval.detach().numpy()
	vapor_actual = vapor_tensor_2.detach().numpy()

	#Plot the predicted vapor
	plt.figure()
	plt.xlabel("Date")
	plt.ylabel("Vapor (cm)")
	plt.title("Taylor Neural Network Vapor Output")
	plt.plot(vapor_predicted)
	plt.plot()

	#Plot the actual vapor
	plt.figure()
	plt.xlabel("Date")
	plt.ylabel("Vapor (cm)")
	plt.title("NYSM Neural Network Output")
	plt.plot(vapor__actual)
	plt.plot()

	plt.show()














