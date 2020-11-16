###########################################################################################################
'''
1 Oct, 2020 
PLSM V6:    (THIS IS JUST CORRECTED PLSM_V5)
class PLSM:
    init():
        -All weights are weighed by W_scale
        
        Input:
        - each input is represented by num_of_input_duplicates number of dummy inputs (default = 1, so same as original input)
        - only (primary_to_auxiliary ratio * num_of_liquid_neurons) neurons receive inputs
        - Each primary neuron receives (input_feature_selection_desnity * intermediate_input_size) random inputs
        
        Liquid:
        - excitatory : inhibitory = exc_to_inh_ratio
        - No self recurrence for any neuron
        - Probabilities of connections are calculated using lambda rule E [0,1] : C{} scales prob., lambda scales proximity spread
        - Weights are alloted as {'EE': 3,'EI': 2,'II': -1,'IE': -4} * W_scale(default=1)
        - Delay buffer (3D Queue) is used to implement synaptic delay.

        Read-Out:
        - Default output size = 2 (binary)
        - Modes: readout_from = 'excitatory_only' or 'all'
        - readout_hidden_layer_list = [10] #Introduces a hidden layer of 10 neurons between LSM and final output layer (Default: None)
        - output_activation = 'softmax' || 'relu' || 'sigmoid' (Default: 'softmax')
        - readout_loss = 'BCE'(default) || 'MSE' || 'CC'(categorical_cross_entropy) 

    predict():
        - Expects an input of (num_inputs x spike_encoding_time_window)
        - output = 'ST_activation'(spatio-temporal activation)(default) || 'readout_values' || 'average_firing_rate' || 'average_firing_rate_and_readout_values'
        - saves the final LSM state N_t after prediction (which is used as the initial LSM state for next prediction)

    predict_batch():
        - Expects input of shape (batch_size, num_neurons, timesteps)
        - dynamically allocates a memory to store the final batch state vector (batch_N_t)
        - output = 'ST_activation'(spatio-temporal activation)(default) || 'readout_values' || 'average_firing_rate' || 'average_firing_rate_and_readout_values'
        - saves the final LSM state batch_N_t after prediction (which is used as the initial LSM state for next prediction on same sequential batched data)
        - WARNING! IF RANDOM MINI-BATCHES OF DATA ARE USED, THEN CALL reset() AFTER EVERY predict_on_batch()

    reset_state():
        - resets the LIF liquid layer neurons
        - resets the state vector N_t of the liquid layer (accumulated by using predict())
        - resets the batch state vector batch_N_t of the liquid layer (accumulated by using predict_on_batch()) 

    train_readout_network():
        - Expects a batch of inputs (LSM states) and outputs (targets)
        - performs one batch update on the readout network

class Readout_network:
    init():
        - initializes a FC network
        - input_size
        - hidden_layers : list of hidden layer units (Ex: [5,3] for 2 hidden layers with 5 and 3 neurons)
        - output_size
        - output_activation = 'softmax' || 'relu' || 'sigmoid' 

'''
###########################################################################################################
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from collections import Deque

import random
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from Spiking_Neurons.LIF import Parallelized_Dense_LIF_Layer_torch as dense_LIF_layer_torch
from Spiking_Neurons.LIF import Parallelized_Dense_LIF_Layer_numpy as dense_LIF_layer_numpy

class PLSM(object):

    def __init__(self, input_size, height, width, depth, 
        output_size=2, readout_from='excitatory_only', readout_hidden_layer_list = None, 
        output_activation = 'softmax', readout_loss='BCE', 
        num_of_input_duplicates=1, neuron_type_ratio = 0.8, input_feature_selection_density=0.6, primary_to_auxiliary_ratio=0.6, 
        stp_alpha = 0.01, stp_beta = 0.3, W_scale=1, lamda=4, synaptic_strength=None, 
        LIF_Vth=0.1, LIF_tau_m=5, 
        version='numpy', device=None, delay_device=None, debug=False):

        self.input_size = input_size
        self.width = width
        self.height = height
        self.depth = depth
        self.output_size = output_size

        self.stp_alpha = stp_alpha
        self.stp_beta = stp_beta

        self.num_of_liquid_layer_neurons = width * height * depth

        self.version = version
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if delay_device is not None:
            self.delay_device = torch.device(delay_device)
        else:
            self.delay_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.debug = debug
        
        self.C = {'EE': 0.6,'EI': 1,'II': 0.2,'IE': 0.8}
        self.lamda = {'EE': lamda,'EI': lamda,'II': lamda,'IE': lamda}
        if synaptic_strength is None:
            self.synaptic_strength = {'EE': 3,'EI': 2,'II': -1,'IE': -4}    #Inhibitory synapses are given negetive strengths
        else:
            self.synaptic_strength = synaptic_strength

        #Input Layer------------------------------------------------------------------------------------------

        intermediate_input_size = self.input_size * num_of_input_duplicates
        self.intermediate_input_weight_matrix = np.zeros((intermediate_input_size, self.input_size)) #Creates 10 copies of each input dimension

        for i in range(self.input_size):
            self.intermediate_input_weight_matrix[i*num_of_input_duplicates:(i+1)*num_of_input_duplicates,i] = np.ones(num_of_input_duplicates)

        self.input_weight_matrix = np.zeros((self.num_of_liquid_layer_neurons, intermediate_input_size))
        primary_neuron_ids = random.sample(list(range(self.num_of_liquid_layer_neurons)), int(primary_to_auxiliary_ratio*self.num_of_liquid_layer_neurons))
        for neuron_id in primary_neuron_ids:
            input_vector_indices = np.random.randint(0, intermediate_input_size, int(input_feature_selection_density*intermediate_input_size))
            for i in input_vector_indices:
                self.input_weight_matrix[neuron_id][i] = np.random.uniform(0, 10, 1) * W_scale #Weight scaling factor

        #Liquid Layer-----------------------------------------------------------------------------------------

        self.liquid_layer_neurons = dense_LIF_layer_numpy(num_of_neurons = self.num_of_liquid_layer_neurons, Vth=LIF_Vth, V_rest=0, V_spike=1.0, tau_m=LIF_tau_m, Rm = 10, tau_ref=1)
        self.N_t = np.zeros(self.num_of_liquid_layer_neurons) #State vector
        self.batch_N_t = None
        #self.S_t = np.ones(self.num_of_liquid_layer_neurons) #STP vector            
        self.liquid_weight_matrix = np.zeros((self.num_of_liquid_layer_neurons, self.num_of_liquid_layer_neurons)) #(TO X FROM)

        #DICTIONARIES : neuron_id <=> coordinates (Row, Col, Depth)
        self.neuron_id_to_coordinate = {}
        neuron_id = 0
        for k in range(self.depth):            
            for i in range(self.height):
                for j in range(self.width):
                    self.neuron_id_to_coordinate[neuron_id] = (i,j,k)   #(row, col, depth)
                    neuron_id += 1
        self.coordinate_to_neuron_id = {}
        neuron_id = 0
        for k in range(self.depth):            
            for i in range(self.height):
                for j in range(self.width):
                    self.coordinate_to_neuron_id[(i,j,k)] = neuron_id
                    neuron_id += 1

        #create dictionary : neuron_id --> excitory ('E') / Inhibitory ('I')
        self.get_neuron_type = {}   
        self.neuron_type_ratio = neuron_type_ratio #80% Excitory & 20% Inhibitory if == 0.8
        excitory_neuron_indices = np.random.randint(0, self.num_of_liquid_layer_neurons, int(self.neuron_type_ratio*self.num_of_liquid_layer_neurons))
        for neuron_id in range(self.num_of_liquid_layer_neurons):
            if neuron_id in excitory_neuron_indices:
                self.get_neuron_type[neuron_id] = 'E'
            else:
                self.get_neuron_type[neuron_id] = 'I'
        self.excitory_neuron_indices = excitory_neuron_indices      
        
        #PROBABILITY OF CONNECTIONS
        probability_of_connection = np.zeros_like(self.liquid_weight_matrix)
        for n_1 in range(self.num_of_liquid_layer_neurons):         #FROM
            temp = []
            for n_2 in range(self.num_of_liquid_layer_neurons):     #TO
                if n_1 != n_2:                                      #No self recurrence
                    connection_type = self.get_neuron_type[n_1]+self.get_neuron_type[n_2]
                    d = np.sqrt(((self.neuron_id_to_coordinate[n_1][0]-self.neuron_id_to_coordinate[n_2][0])**2)+((self.neuron_id_to_coordinate[n_1][1]-self.neuron_id_to_coordinate[n_2][1])**2)+((self.neuron_id_to_coordinate[n_1][2]-self.neuron_id_to_coordinate[n_2][2])**2))
                    prob = self.C[connection_type]*np.exp((-d/self.lamda[connection_type])**2)
                    probability_of_connection[n_2][n_1] = prob
        probability_of_connection /= np.max(probability_of_connection)  #Normalize to obtain probabilities E [0,1]

        #Allocate weights according to calculated probabilies
        for neuron_id_1 in range(self.num_of_liquid_layer_neurons):         #FROM
            for neuron_id_2 in range(self.num_of_liquid_layer_neurons):     #TO
                if np.random.uniform(0.000,1.000,1) < probability_of_connection[neuron_id_2][neuron_id_1]:
                    connection_type = self.get_neuron_type[neuron_id_1] + self.get_neuron_type[neuron_id_2]
                    self.liquid_weight_matrix[neuron_id_2][neuron_id_1] = self.synaptic_strength[connection_type] * W_scale #Weight scaling factor
        
        #(LATER) Normalize (separately) sum of excitory and sum of inhibitory pre-synapses for each neuron in liquid layer

        #Delay buffer
        delay_distribution = np.zeros((len(self.N_t), len(self.N_t)))
        #calculate delay for each neuron-neuron pair
        for i in range(len(self.N_t)):
            for j in range(len(self.N_t)):
                coordinate_1, coordinate_2 = self.neuron_id_to_coordinate[i], self.neuron_id_to_coordinate[j]   #returns (x,y,z) of each neuron
                distance = self.calc_distance(coordinate_1, coordinate_2)
                delay_distribution[i][j] = int(distance)
        max_delay = int(np.max(delay_distribution)) + 1     

        self.delay_buffer = Binary_3D_Queue(dim=(len(self.N_t), len(self.N_t), max_delay), rear_array=delay_distribution.astype(np.int), device=self.delay_device)

        if version=='torch':
            #Input Layer------------------------------------------------------------------------------------------
            self.intermediate_input_weight_matrix = torch.from_numpy(self.intermediate_input_weight_matrix).float().to(self.device)
            self.input_weight_matrix = torch.from_numpy(self.input_weight_matrix).float().to(self.device)
            #Liquid Layer-----------------------------------------------------------------------------------------
            self.liquid_layer_neurons = dense_LIF_layer_torch(num_of_neurons = self.num_of_liquid_layer_neurons, Vth=LIF_Vth, V_rest=0, V_spike=1.0, tau_m=LIF_tau_m, Rm = 10, tau_ref=1, device=self.device)
            self.N_t = torch.from_numpy(self.N_t).float().to(self.device)
            self.liquid_weight_matrix = torch.from_numpy(self.liquid_weight_matrix).float().to(self.device)

        #Read-Out Layer-----------------------------------------------
        self.readout_input_neuron_ids = self.excitory_neuron_indices if readout_from == 'excitatory_only' else list(range(self.num_of_liquid_layer_neurons))
        readout_input_size = len(self.readout_input_neuron_ids)
        self.readout_network = Readout_network(input_size = readout_input_size, hidden_layers = readout_hidden_layer_list, output_size = output_size, output_activation = output_activation)
        self.readout_optimizer = optim.RMSprop(self.readout_network.parameters(), lr=0.0002)
        if readout_loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif readout_loss == 'CC':
            self.criterion = nn.CrossEntropyLoss()
        elif readout_loss == 'MSE':
            self.criterion = nn.MSELoss()

        self.readout_network = self.readout_network.to(self.device)
            
    def predict(self, input_state, output='ST_activation'): #Input shape : (num_of_neurons, timesteps)
        if self.version == 'numpy':
            activation = [] #activation of LSM over entire input spike train duration
            input_state = input_state.astype('float')            
            for t in range(input_state.shape[-1]):    
                
                N_s_prev = self.liquid_weight_matrix * self.delay_buffer.pop()  #N X N matrix
                past_current = np.sum(N_s_prev, axis=1)    #N dim vector

                pseudo_input_current = np.matmul(self.intermediate_input_weight_matrix, input_state[:,t])
                input_current = np.matmul(self.input_weight_matrix, pseudo_input_current)

                total_current = (input_current + past_current)

                self.N_t, _, _ = self.liquid_layer_neurons.update(total_current) 
                
                N_t_matrix = np.expand_dims(self.N_t, axis=0).repeat(self.N_t.shape[0], axis=0)
                #N_s_new = self.liquid_weight_matrix * N_t_matrix
                self.delay_buffer.push(N_t_matrix)

                activation.append(np.expand_dims(self.N_t, axis=-1))
            activation = np.concatenate(activation, axis=-1)   #Shape : N x T
            
            average_firing_rate = np.sum(activation[self.readout_input_neuron_ids], axis=-1) / input_state.shape[-1]

            if output=='average_firing_rate':
                return average_firing_rate
            elif output=='readout_values':
                return self.readout_network(average_firing_rate).detach().cpu().numpy()
            elif output=='average_firing_rate_and_readout_values':
                return [average_firing_rate, self.readout_network(average_firing_rate).detach().cpu().numpy()]
            elif output=='ST_activation':
                return activation

        elif self.version == 'torch':
            activation = [] #activation of LSM over entire input spike train duration
            if not torch.is_tensor(input_state):
                input_state = torch.from_numpy(input_state).to(self.device)
            input_state = input_state.float()

            for t in range(input_state.size()[-1]):  
                
                N_s_prev = self.liquid_weight_matrix * self.delay_buffer.pop().cpu().to(self.device)  #N X N matrix
                past_current = torch.sum(N_s_prev, dim=1)    #N dim vector

                pseudo_input_current = torch.matmul(self.intermediate_input_weight_matrix, input_state[:,t])
                input_current = torch.matmul(self.input_weight_matrix, pseudo_input_current)
                
                total_current = (input_current + past_current)

                self.N_t, V, _ = self.liquid_layer_neurons.update(total_current) 

                N_t_matrix = self.N_t.unsqueeze(0).repeat(self.N_t.size(0),1)
                self.delay_buffer.push(N_t_matrix.cpu().to(self.delay_device))

                activation.append(self.N_t.unsqueeze(-1))

            activation = torch.cat(activation, dim=-1)   #Shape : N x T

            average_firing_rate = torch.sum(activation[self.readout_input_neuron_ids], dim=-1) / input_state.shape[-1]

            if output=='average_firing_rate':
                return average_firing_rate
            elif output=='readout_values':
                return self.readout_network(average_firing_rate).detach()
            elif output=='average_firing_rate_and_readout_values':
                return [average_firing_rate, self.readout_network(average_firing_rate).detach()]
            elif output=='ST_activation':
                return activation

    def predict_on_batch(self, input_state, output='ST_activation'): #Input shape: (batch_size, num_of_neurons, timesteps) 

        #call reset() if non-sequential mini-batches are used
        if self.version == 'numpy':
            input_state = input_state.astype(float)
            
            batch_size = input_state.shape[0]
            if self.batch_N_t is None:   #If first batch of data
                N_t = np.zeros((batch_size, self.num_of_liquid_layer_neurons))  #B X N
            else:
                N_t = self.batch_N_t

            activation = [] #activation of LSM over entire input spike train duration

            for t in range(input_state.shape[-1]):          
                pseudo_input_current = np.matmul(self.intermediate_input_weight_matrix, input_state[:,:,t].T) #input_state[:,:,t].T gives shape: (N, B, :)
                input_current = np.matmul(self.input_weight_matrix, pseudo_input_current)
                past_current = np.matmul(self.liquid_weight_matrix, N_t.T)                  #Expects N_t to be N X B 
                total_current = (input_current + past_current)             
                N_t, _, _ = self.liquid_layer_neurons.update_on_batch(total_current.T)        #Expects and Returns dim: B X N          
                activation.append(np.expand_dims(N_t, axis=-1))  #Shape: (B, N, 1)
            activation = np.concatenate(activation, axis=-1)   #Shape : (B, N, T)

            self.batch_N_t = N_t  #Save last N_t for this batch of data
            
            #Calculate average firing rate of each neuron during the entire input duration
            average_firing_rate = np.sum(activation[:,self.readout_input_neuron_ids,:], axis=-1) / input_state.shape[-1]

            #Output
            if output == 'readout_values':
                return self.readout_network(average_firing_rate).detach().cpu().numpy()
            elif output == 'average_firing_rate':
                return average_firing_rate
            elif output == 'average_firing_rate_and_readout_values':
                return [average_firing_rate, self.readout_network(average_firing_rate).detach().cpu().numpy()]
            elif output=='ST_activation':
                return activation

        elif self.version == 'torch':
            if not torch.is_tensor(input_state):
                input_state = torch.from_numpy(input_state).float().to(self.device)

            batch_size = input_state.size()[0]
            if self.batch_N_t is None:
                N_t = torch.zeros(batch_size, self.num_of_liquid_layer_neurons).to(self.device) #B X N
            else:
                N_t = self.batch_N_t

            activation = [] #activation of LSM over entire input spike train duration

            for t in range(input_state.size()[-1]):          
                pseudo_input_current = torch.matmul(self.intermediate_input_weight_matrix, input_state[:,:,t].T) #input_state[:,:,t].T gives shape: (N, B, :)
                input_current = torch.matmul(self.input_weight_matrix, pseudo_input_current)
                past_current = torch.matmul(self.liquid_weight_matrix, N_t.T)                  #Expects N_t to be N X B 
                total_current = (input_current + past_current)             
                N_t, _, _ = self.liquid_layer_neurons.update_on_batch(total_current.T)        #Expects and Returns dim: B X N          
                activation.append(N_t.unsqueeze(-1).cpu().numpy())  #Shape: (B, N, 1)

            self.batch_N_t = N_t  #Save last N_t for this batch of data

            del input_state, pseudo_input_current, input_current, past_current, total_current, N_t
            torch.cuda.empty_cache()

            #activation = torch.cat(activation, dim=-1)   #Shape : (B, N, T)

            #Output
            if output == 'readout_values':
                activation = torch.from_numpy(np.concatenate(activation, axis=-1))
                average_firing_rate = torch.sum(activation[:,self.readout_input_neuron_ids,:], dim=-1) / activation.size()[-1]
                return self.readout_network(average_firing_rate.to(self.device)).detach()
            elif output == 'average_firing_rate':
                activation = torch.from_numpy(np.concatenate(activation, axis=-1))
                average_firing_rate = torch.sum(activation[:,self.readout_input_neuron_ids,:], dim=-1) / activation.size()[-1]
                return average_firing_rate
            elif output == 'average_firing_rate_and_readout_values':
                activation = torch.from_numpy(np.concatenate(activation, axis=-1))
                average_firing_rate = torch.sum(activation[:,self.readout_input_neuron_ids,:], dim=-1) / activation.size()[-1]
                return [average_firing_rate, self.readout_network(average_firing_rate.to(self.device)).detach()]
            elif output=='ST_activation':
                return np.concatenate(activation, axis=-1)

    def reset(self):
        if self.version=='numpy':
            self.N_t = np.zeros(self.num_of_liquid_layer_neurons)
        elif self.version=='torch':
            self.N_t = torch.zeros(self.num_of_liquid_layer_neurons).to(self.device)
        
        self.batch_N_t = None
        self.liquid_layer_neurons.reset()
        self.delay_buffer.reset()

    def calc_distance(self, c1, c2):
        sum = 0
        for i in range(len(c1)):
            sum += (c1[i] - c2[i])**2
        return np.sqrt(sum)

    def train_readout_network(self, inputs, targets):        
        inputs, targets = torch.tensor(inputs).float().to(self.device), torch.tensor(targets).float().to(self.device)
        self.readout_optimizer.zero_grad()
        predictions = self.readout_network(inputs)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.readout_optimizer.step()
        return loss.cpu()
        
class Binary_3D_Queue(object):
    def __init__(self, dim, rear_array, device):    #Queue shape : (W X H X T)
        
        self.queue = torch.zeros(dim[0], dim[1], dim[2]).bool().to(device)
        self.dim = dim
        self.device = device

        self.pop_kernel = torch.zeros(dim[0], dim[1], dim[2])
        for r in range(dim[0]):
            for c in range(dim[1]):
                self.pop_kernel[r][c][rear_array[r][c]] = 1
        self.pop_kernel = self.pop_kernel.bool().to(device)

    def shift_queue(self):

        for i in range(self.queue.size(2)-2,-1,-1):    
            self.queue[:,:,i+1] = self.queue[:,:,i]

    def push(self, new_vector):     #Expects a column vector of dimension (dim[0])
        
        self.shift_queue()

        if not torch.is_tensor(new_vector):
            new_vector = torch.from_numpy(new_vector)
        self.queue[:,:,0] = new_vector.to(self.device).bool()

    def pop(self):
        
        return torch.sum(self.queue * self.pop_kernel, dim=-1)

    def reset(self):
        
        self.queue = torch.zeros(self.dim[0], self.dim[1], self.dim[2]).bool().to(self.device)

    def show(self):
        return self.queue.float()

class Readout_network(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, output_activation):
        super(Readout_network, self).__init__()
        
        if hidden_layers == None:
            self.fc = nn.Linear(input_size, output_size)
        else:
            assert 0 not in hidden_layers   #No layer with zero neurons
            
            last_layer_shape = input_size   #Start connecting from input layer
            for idx in range(len(hidden_layers)):
                self.add_module('fc_'+str(idx), nn.Linear(last_layer_shape, hidden_layers[idx]))
                self.add_module('act_'+str(idx), nn.ReLU())
                last_layer_shape = hidden_layers[idx]
            self.add_module('fc_output', nn.Linear(last_layer_shape, output_size))  #Output FC layer

        if output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=-1)
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'relu':
            self.output_activation == nn.ReLU()

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x).float().to(self.device)
        
        for m in self.children():
            x = m(x)

        return x


if __name__ == '__main__':

    device = torch.device('cuda:1')

    LSM = PLSM(2, 3,4,5, version='torch', device=device, LIF_Vth=0.05, W_scale=1)

    s = (torch.randn(2,100)*3).to(device)

    a = LSM.predict(s, output='ST_activation')
    
    print(a)

    

