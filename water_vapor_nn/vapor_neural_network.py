import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Definitions and Classes
# --------------------------------------------------

#Load the Input Data
def load_data(site, year, level, key, subkey):
    prefix = f"{key}_{subkey}" if level != "lv2" else key
    dir_path = os.path.join(BASE_DIR, str(year), site, level)
    dates_path = os.path.join(dir_path, f"{prefix}_dates.npy")
    values_path = os.path.join(dir_path, f"{prefix}_values.npy")
    if not os.path.exists(dates_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"Missing {dates_path} or {values_path}")
    dates = np.load(dates_path, allow_pickle=True)
    values = np.load(values_path, allow_pickle=True)
    
    return pd.to_datetime(dates), values

#Create a Matrix of the Weights from a Text File
def get_weights(path): 
    with open(path, "r") as file:
        lines = file.readlines()

    all_text = ''.join(lines).replace('\\\n', '').replace('\n', '')
    chunks = re.split(r'(?=\b\d{1,3},)', all_text)
    chunks = [c.strip() for c in chunks if c.strip()]
   
    weights_array = []
    current_neuron = []

    for item in chunks:
        if re.match(r'^\d+,$', item.strip()):
            if current_neuron:
                weights_array.append(current_neuron)
            current_neuron = []
        else:
            try:
                weight = float(item.split(',')[1])
                current_neuron.append(weight)
            except (IndexError, ValueError):
                continue

    if current_neuron:
        weights_array.append(current_neuron)

    wH = weights_array[:49]
    wO = weights_array[-58:] 

    return wH, wO

#Describe the Neural Network
class RadiometerNN(nn.Module):
    
    input_size = 26
    hidden_size = 49
    output_size = 58

    def __init__(self, wH, bH, WO, bO):

        super(RadiometerNN, self).__init__()
        
        #Input layer 
        self.input = nn.Linear(input_size, input_size)

        #Hidden layer
        self.hidden = nn.Linear(input_size, hidden_size)
        self.hidden.weight = nn.Parameter(WH)
        self.hidden.bias = nn.Parameter(bH)
        
        #Hidden layer activation function (assume tanh unless specified)
        self.hidden_activation = nn.Tanh()
        
        #Output layer
        self.output = nn.Linear(input_size + hidden_size, output_size)
        self.output.weight = nn.Parameter(WO)
        self.output.bias = nn.Parameter(bO)

        #Output layer activation function
        self.output_activation = nn.Sigmoid()
        
        # Output scaling (*50+0 from config)
        self.scale = 50.0
        self.shift = 0.0
        
    def forward(self, input_layer_input):

        #Pass the features through the input layer
        input_layer_output = self.input(input_layer_input)
        
        #Pass the input layer outputs through the hidden layer
        hidden_layer_input = input_layer_output
        hidden_layer_output = self.hidden(hidden_layer_input)
        hidden_layer_output = self.hidden_activation(hidden_layer_output)
        
        #Pass the hidden layer outputs through the output layer
        output_layer_input = hidden_layer_output

        output_layer_input = torch.cat((input_layer_output, output_layer_input),dim=0) 
        output = self.output(output_layer_input)
        output = self.output_activation(output)
        output = output * self.scale + self.shift
        
        return output

# --------------------------------------------------
# 2. Neural Network Data and Structure Information
# --------------------------------------------------

#Data Loadup Detials
SITE = 'bronx'
YEAR = '2022'
LEVEL = 'lv1'
DATA_TYPE = 'brightness'
BASE_DIR = "./NYS_Mesonet_Data"
OUTPUT_DIR = "RFI_detection_plots_and_csv"
FREQ_LIST = [22.234,22.500,23.034,23.834,25.000,26.234,28.000,30.000,51.248,51.760,52.280,52.804,53.336,53.848,54.400,54.940,55.500,56.020,56.660,57.288,57.964,58.800]

NN_INPUT_SIZE = 26
brightness_matrix = []
integ_vapor = []

#Neural Network Structure Details
input_size = 26
hidden_size = 49
output_size = 58

FREQ = FREQ_LIST[0]
bt_time, bt_values = load_data(SITE, YEAR, LEVEL, FREQ, DATA_TYPE)

bH = [-17.1458,-6.27777,4.85305,1.85955,9.36397,-1.36327,-3.80258,-0.77420,-8.87321,-1.32277,0.46771
,-6.02680,-1.25969,-6.67079,13.21855,-12.6569,0.38531,2.99938,2.58431,-0.79303,-6.03854,-0.89705,18.82808
,1.85779,-8.79119,0.42465,9.50065,1.42123,3.49830,-4.14872,-6.27683,7.38380,3.90473,-2.27920,0.12496,24.89363
,-0.29222,-8.50061,-1.54072,-2.43413,0.30390,8.06277,2.69888,-0.01792,-7.82661,-0.42325,-0.56842,-6.62297,-2.29360]
bH = torch.tensor(bH)

bO = [-2.10158,-5.40590,-5.11700,-4.54615,-2.87012,-1.36690,-0.52724,-1.31673,-0.40364,-0.63195
,-0.40136,-0.93328,-2.32036,-2.51165,-1.94269,-1.28612,-3.08508,-3.40224,-2.14134,-2.63310,-1.36442
,1.33210,-1.59348,-2.42501,-2.29514,-1.60872,-0.29671,-1.04231,0.53489,0.23275,-0.47396,0.74654
,0.07773,-0.74891,-0.67878,-2.04663,-2.20283,-1.52127,-1.68730,-3.45714,-2.42358,-2.20107,-3.15334,-2.46129,-1.75923
,-0.71552,-1.57562,-0.88129,-0.80951,-1.63012,-0.07654,-1.75423,-1.00007,-0.99999,-0.21565,-0.28861,-0.24999,-0.85492]
bO = torch.tensor(bO)

WH, WO = get_weights("/Users/taylorwang/desktop/zussman_research/water_vapor_nn/weight_details.txt")
WH = torch.tensor(WH)
WO = torch.tensor(WO)

# --------------------------------------------------
# 3. Main
# --------------------------------------------------

#Obtaining the Input Data for the Neural Network

print('mark 1')
for i in range(len(FREQ_LIST)): 
    print('mark 2')
    FREQ = FREQ_LIST[i]
    bt_time, bt_values = load_data(SITE, YEAR, LEVEL, FREQ, DATA_TYPE)
    brightness_matrix.append(bt_values)

brightness_matrix = np.array(brightness_matrix)   # now it’s a single ndarray
brightness_matrix = torch.tensor(brightness_matrix, dtype=torch.float32)

time_points = len(bt_time)
for j in range(time_points):
    print('mark 3')
    brightness_vector = brightness_matrix[:,j]
    surf_met_vector = torch.zeros(4)
    input_vector = torch.cat((surf_met_vector, brightness_vector))

    model = RadiometerNN(WH,bH,WO,bO)
    x = input_vector
    y = model(x)

    vert_profile_sum = y.sum()
    integ_vapor.append(vert_profile_sum)

print(integ_vapor)
integ_vapor = np.array([t.detach().numpy() for t in integ_vapor])
plt.plot(integ_vapor)
plt.show()
