import numpy as np

class spike_encoding:

    def __init__(self, scheme='rate_coding', time_window=100, input_range=(-10,10), output_freq_range=(0, 100)):
        
        self.scheme = scheme
        
        if scheme == 'rate_coding' or scheme == 'poisson_rate_coding':
            self.min_input, self.max_input = input_range[0], input_range[1]
            self.min_output, self.max_output = output_freq_range[0], output_freq_range[1]
            self.time_window = time_window
            
        elif scheme == 'rank_order_coding':
            self.time_window = time_window
        else:
            raise Exception('No valid spike encoding scheme selected!')
            
        
    def encode(self, signal):
    
        if self.scheme == 'rate_coding':
            assert self.min_input <= np.max(signal) <= self.max_input
            if len(signal.shape) == 1:  #signal is vector : (N,)
                signal = np.expand_dims(signal, axis=-1)   #(N,1), meaning single time-step feature vector

            repeat = signal.shape[0] 

            total_spikes = []
            for r in range(repeat):     #Iterate over signal dimension
                spike_train = []        
                for s in signal[r]:     #Iterate over timesteps of this signal
                    freq = ((s-self.min_input)/(self.max_input-self.min_input)) * (self.max_output-self.min_output) + self.min_output 
                    t = (1 / freq) * 1000 #ms
                    
                    spikes = np.zeros(self.time_window)
                    k=0
                    while k<self.time_window:
                        spikes[k] = 1
                        k += int(t)
                    spike_train.append(spikes)
                spike_train = np.hstack(([x for x in spike_train]))  
                total_spikes.append(spike_train) 

            return np.asarray(total_spikes)

        elif self.scheme == 'poisson_rate_coding':
            assert (self.min_input <= np.min(signal)) and (np.max(signal) <= self.max_input)
            if len(signal.shape) == 1:  #signal is vector : (N,)
                signal = np.expand_dims(signal, axis=-1)   #(N,1), meaning single time-step feature vector
           
            repeat = signal.shape[0] 

            total_spikes = []
            for r in range(repeat):
                spike_train = []        
                for s in signal[r]:
                    freq = np.interp(s, [self.min_input, self.max_input], [self.min_output, self.max_output]) 
                    
                    spikes = np.random.uniform(0,1,self.time_window)        
                    dt = 0.01 #second
                    spikes[np.where(spikes<freq*dt)] = 1
                    spikes[np.where(spikes!=1)] = 0

                    spike_train.append(spikes)
                spike_train = np.hstack(([x for x in spike_train]))  
                total_spikes.append(spike_train) 

            return np.asarray(total_spikes)

        
        elif self.scheme == 'rank_order_coding':
        
            if len(signal.shape)!=2:
                raise Exception('Input signal should have more than one input dimension!')

            spike_train = np.zeros((signal.shape[1], signal.shape[0], self.time_window+1))

            for t in range(signal.shape[1]):
                s = signal[:,t]
                s = np.max(s) - s
                latency = self.time_window * ((s - np.min(s))/(np.max(s) - np.min(s)))

                for i in range(latency.shape[0]): #iterate over each dimension of data
                    spike_train[t][i][int(latency[i])] = 1 

            #Total encoded data
            seq = spike_train[0]
            for w in spike_train[1:]:
                seq = np.hstack((seq, w))
            return seq


 
if __name__=='__main__':

    signal = np.random.randint(0,255, size=(10))

    encoder = spike_encoding(scheme='poisson_rate_coding')

    spike_train = []

    for i in range(-10,11,1):
        s = encoder.encode(np.asarray([i]))
        print(i)
        spike_train.append(s)

    spike_train = np.concatenate(spike_train)

    import matplotlib.pyplot as plt
    plt.imshow(spike_train)
    plt.show()
