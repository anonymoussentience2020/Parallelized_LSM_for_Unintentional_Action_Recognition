import torch
import numpy as np

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LIF_neuron(object):
    def __init__(self, Vth=0.5, dt=0.001, V_rest=0, tau_ref=4, tau_m = -1, Rm=1, Cm=10):

        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF parameters
        self.V_rest = V_rest                     #resting potential (mV)
        self.Vm = self.V_rest                  #initial potential (mV)
        self.refraction_counter = -1         #refraction counter
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)
        self.refraction = False                     #Refraction flag

        self.Rm = Rm                         
        self.Cm = Cm                          
        if tau_m!=-1:
            self.tau_m = tau_m               #(ms)
        else:
            self.tau_m = self.Rm * self.Cm   #(ms)

        self.V_spike = Vth+0.5         #spike delta (mV)
            
    def update_(self, I, time_stamp):
        if time_stamp > self.t_rest:
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt)
            
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.t_rest = time_stamp + self.tau_ref
        else:
            self.Vm = self.V_rest
        return self.Vm

    def update(self, I): 
        if not self.refraction:
            self.Vm = self.Vm + (((I*self.Rm + self.V_rest - self.Vm) / self.tau_m) * self.dt)
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.refraction = True
                self.refraction_counter = self.tau_ref
        else:
            self.Vm = self.V_rest
            self.refraction_counter -= 1
            if self.refraction_counter <= 0:
                self.refraction = False

        return self.Vm
            
    def reset(self):
        self.Vm = self.V_rest
        self.refraction = False
        self.refraction_counter = -1         


class Parallelized_Dense_LIF_Layer_torch(object):
    def __init__(self, num_of_neurons, Vth=None, dt=0.001, V_rest=0, V_spike=None, tau_ref=4, tau_m = -1, Rm=1, Cm=10, device=None):
        
        assert device is not None
        self.device = device
        
        #Layer variables - state vectors
        self.num_of_neurons = num_of_neurons
        self.N_t = torch.zeros(num_of_neurons).to(self.device)
        self.V_m = torch.zeros(num_of_neurons).to(self.device)
        self.R_c = torch.zeros(num_of_neurons).to(self.device)  

        self.batch_V_m = None
        self.batch_R_c = None

        #Misc. vectors
        self.zeros = torch.zeros(num_of_neurons).to(self.device)
        
        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF neuron parameters
        self.V_rest = V_rest                 #resting potential (mV)
        self.tau_ref = tau_ref               #(ms) : refractory period
        #self.Vth = Vth                       #(mV)
        if Vth is None:
            self.Vth = torch.zeros(num_of_neurons).to(self.device) + 0.5
        elif type(Vth) == type(0.1):
            self.Vth = torch.zeros(num_of_neurons).to(self.device) + Vth
        elif type(Vth) == type(torch.randn(10,)):
           self.Vth = torch.zeros(num_of_neurons).to(self.device) + Vth.to(device) 

        self.Rm = Rm                         
        self.Cm = Cm                          
        self.V_spike = self.Vth+0.5 if V_spike is None else V_spike                                #spike delta (mV)
        self.tau_m = self.Rm * self.Cm if tau_m==-1 else tau_m  #(ms)
    
    def update(self, I):  #This funtion updates the original states of this layer
        #shape(I) : (num_of_neurons,)
        if not torch.is_tensor(I):
            I = torch.from_numpy(I).to(self.device)       
        assert len(I.size()) == 1

        V_m = self.V_m + ((I*self.Rm + self.V_rest - self.V_m)/self.tau_m)*self.dt
        R_f = (self.R_c.bool()).int()        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest        
        S = (torch.max(self.zeros, V_m_prime - self.Vth)).bool().int()        
        self.N_t = S * self.V_spike
        self.V_m = ((1-S) * V_m_prime) + self.N_t
        R_c_prime = self.R_c - R_f
        self.R_c = (S * self.tau_ref) + R_c_prime

        return self.N_t, self.V_m, self.R_c        

    def update_on_batch(self, I):    #This function updates a set of dummy state vectors and returns them again        
        #shape(I) : (batch_size, num_of_neurons) type: float/double
        if not torch.is_tensor(I):
            I = torch.from_numpy(I).to(self.device)        
        assert len(I.size()) == 2

        if self.batch_V_m is None:  #if first batch of data
            V_m = torch.zeros(I.shape[0], self.num_of_neurons).to(self.device)
            R_c = torch.zeros(I.shape[0], self.num_of_neurons).to(self.device)
        else:                       #else remember from last timestep
            V_m = self.batch_V_m
            R_c = self.batch_R_c
        
        zeros = torch.zeros_like(I).float().to(self.device)

        V_m = V_m + ((I*self.Rm + self.V_rest - V_m)/self.tau_m)*self.dt        
        R_f = (R_c.bool()).int()        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest           
        S = (torch.max(zeros, V_m_prime - self.Vth)).bool().int()        
        N_t = S * self.V_spike
        V_m = ((1-S) * V_m_prime) + N_t
        R_c_prime = R_c - R_f
        R_c = (S * self.tau_ref) + R_c_prime

        self.batch_V_m = V_m
        self.batch_R_c = R_c 

        return N_t, V_m, R_c 

    def reset(self):
        self.N_t = torch.zeros(self.num_of_neurons).to(self.device)
        self.V_m = torch.zeros(self.num_of_neurons).to(self.device)
        self.R_c = torch.zeros(self.num_of_neurons).to(self.device)  

        self.batch_V_m = None
        self.batch_R_c = None

class Parallelized_Dense_LIF_Layer_numpy(object):
    def __init__(self, num_of_neurons, Vth=0.5, dt=0.001, V_rest=0, V_spike=None, tau_ref=4, tau_m = -1, Rm=1, Cm=10):
        #Layer variables - state vectors
        self.num_of_neurons = num_of_neurons
        self.N_t = np.zeros(num_of_neurons)
        self.V_m = np.zeros(num_of_neurons)
        self.R_c = np.zeros(num_of_neurons)

        self.batch_V_m = None
        self.batch_R_c = None

        #Misc. vectors
        self.zeros = np.zeros(num_of_neurons)
        
        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF neuron parameters
        self.V_rest = V_rest                 #resting potential (mV)
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)
        self.Rm = Rm                         
        self.Cm = Cm                          
        self.V_spike = Vth+0.5 if V_spike is None else V_spike                                #spike delta (mV)
        self.tau_m = self.Rm * self.Cm if tau_m==-1 else tau_m  #(ms)
    
    def update(self, I):  #This funtion updates the original states of this layer
        #shape(I) : (num_of_neurons,)    
        assert len(I.shape) == 1

        V_m = self.V_m + ((I*self.Rm + self.V_rest - self.V_m)/self.tau_m)*self.dt
        R_f = (self.R_c.astype(bool)).astype(int)        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest        
        S = ((np.maximum(self.zeros, V_m_prime - self.Vth)).astype(bool)).astype(int)
        self.N_t = S * self.V_spike
        self.V_m = ((1-S) * V_m_prime) + self.N_t
        R_c_prime = self.R_c - R_f
        self.R_c = (S * self.tau_ref) + R_c_prime

        return self.N_t, self.V_m, self.R_c        

    def update_on_batch(self, I):    #This function updates a set of dummy state vectors and returns them again        
        #shape(I) : (batch_size, num_of_neurons)       
        assert len(I.shape) == 2
        
        if self.batch_V_m is None:  #if first batch of data
            V_m = np.zeros((I.shape[0], self.num_of_neurons))
            R_c = np.zeros((I.shape[0], self.num_of_neurons))
        else:                       #else remember from last timestep
            V_m = self.batch_V_m
            R_c = self.batch_R_c

        zeros = np.zeros_like(I)

        V_m = V_m + ((I*self.Rm + self.V_rest - V_m)/self.tau_m)*self.dt        
        R_f = (R_c.astype(bool)).astype(int)   
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest        
        S = ((np.maximum(zeros, V_m_prime - self.Vth)).astype(bool)).astype(int)
        N_t = S * self.V_spike
        V_m = ((1-S) * V_m_prime) + N_t
        R_c_prime = R_c - R_f
        R_c = (S * self.tau_ref) + R_c_prime

        self.batch_V_m = V_m
        self.batch_R_c = R_c 

        return N_t, V_m, R_c 

    def reset(self):
        self.N_t = np.zeros(self.num_of_neurons)
        self.V_m = np.zeros(self.num_of_neurons)
        self.R_c = np.zeros(self.num_of_neurons)

        self.batch_V_m = None
        self.batch_R_c = None

if __name__ == '__main__':
    '''
    layer = Parallelized_Dense_LIF_Layer_numpy(10, Vth=0.02)

    signal = np.random.randint(0,2,(5,10,100))*50.

    act = []
    for t in range(signal.shape[-1]):
        N, V, _ = layer.update_on_batch(signal[:,:,t])
        act.append(np.expand_dims(V, axis=-1))
    act = np.concatenate(act, axis=-1)

    import matplotlib.pyplot as plt

    plt.plot(act[3][0])
    plt.show()
    print(act.shape)
    '''
    layer = Parallelized_Dense_LIF_Layer_torch(10, Vth=0.02)

    signal = torch.from_numpy(np.random.randint(0,2,(10,100))*50.).float().to(device)

    act = []
    for t in range(signal.shape[-1]):
        N, V, _ = layer.update(signal[:,t])
        act.append(N.unsqueeze(-1))
    act = torch.cat(act, dim=-1)

    import matplotlib.pyplot as plt

    plt.imshow(act.cpu().numpy())
    plt.show()
    print(act.size())