# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg

class ESN:
    def __init__(self, ninput, ninternal, noutput, W, W_in, W_fb, W_out, 
                 activation, out_activation, invout_activation, encode,
                 spectral_radius, dynamics, regression,
                 noise_level, delta, C, leakage):
        
        self.ninput = ninput
        self.ninternal = ninternal
        self.noutput = noutput
        self.ntotal = ninput + ninternal + noutput
        self.spectral_radius = spectral_radius
        
        self.W = W
        self.W_in = W_in
        self.W_fb = W_fb
        self.W_out = W_out
        self.activation = activation
        self.out_activation = out_activation
        self.invout_activation = invout_activation
        
        self._update = self.leaky if dynamics == 'leaky' else self.plain
        self.noise_level = noise_level
        self.regression = regression
        self.trained = False
        self._last_input = np.zeros((self.ninput, 1))  
        self._last_state = np.zeros((self.ninternal, 1))
        self._last_output = np.zeros((self.noutput, 1))  
        self.delta = delta
        self.C = C
        self.leakage = leakage

    def fit(self, inputs, outputs, nforget):
        ntime = inputs.shape[1]
        states = np.zeros((self.ninternal, ntime))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
        
        S = np.vstack((states, inputs)).T[nforget:]
        D = self.invout_activation(outputs.T[nforget:])
        self.W_out = self.regression(S, D)
        
        self._last_input = inputs[:, -1]
        self._last_state = states[:, -1]
        self._last_output = outputs[:, -1]
        self.trained = True

        return states

    def trained_outputs(self, inputs, outputs):
        ntime = inputs.shape[1]
        trained_outputs = np.zeros((self.noutput, ntime))
        states = np.zeros((self.ninternal, ntime))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
            trained_outputs[:, t] = self.out_activation(self.W_out @ np.hstack((states[:, t], inputs[:, t])))
            
        return trained_outputs

    def predict(self, inputs, turnoff_noise=True, continuing=True):
        if turnoff_noise:
            self.noise_level = 0
        if not continuing:
            self._last_input = np.zeros((self.ninput, 1))
            self._last_state = np.zeros((self.ninternal, 1))
            self._last_output = np.zeros((self.noutput, 1))

        ntime = inputs.shape[1]
        outputs = np.zeros((self.noutput, ntime))
        states = np.zeros((self.ninternal, ntime))
        states[:, 0] = self._update(self._last_state, inputs[:, 0], self._last_output)
        outputs[:, 0] = self.out_activation(self.W_out @ np.hstack((states[:, 0], inputs[:, 0])))
        for t in range(1, ntime):
            states[:, t] = self._update(states[:, t - 1], inputs[:, t], outputs[:, t - 1])
            outputs[:, t] = self.out_activation(self.W_out @ np.hstack((states[:, t], inputs[:, t])))

        return outputs
    
    def leaky(self, previous_internal, new_input, previous_output):
        new_internal = (1 - self.delta * self.C * self.leakage) * previous_internal \
                       + self.delta * self.C * self.activation(self.W_in @ new_input
                                                               + self.W @ previous_internal
                                                               + self.W_fb @ previous_output
                                                               + self.noise_level)
        return new_internal

    def plain(self, previous_internal, new_input, previous_output):
        new_internal = self.activation(self.W_in @ new_input
                                       + self.W @ previous_internal
                                       + self.W_fb @ previous_output) \
                       + self.noise_level * (np.random.rand(self.ninternal) - 0.5)
        return new_internal
