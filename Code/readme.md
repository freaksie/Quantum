
========================================================
Data summary:

Sampling Rate: 2 samples / ns.
Total time : 4µs = 8192 samples.
Total shots = 6000 for state 0 and 6000 for state 1.
Readout Frequency = 6558500000.003296.
Qubit Name: Q1
Chip Name: X4Y2
Readout Pulse width = 2µs.
Readout start : end time = 350ns : 2350ns;
Envelope : cos-edge-square (5% rising edge; 90% flat edge; 5% falling edge)

============================================================

state0.npy : shape (6000,8192). Readout pulse begin from [700:4700] ie 350ns to 2350ns. 
state1.npy : shape (6000,8192). Readout pulse begin from [700:4700] ie 350ns to 2350ns. 

============================================================

The npy file consists of raw signals from ADC. 
If you want IQ component, then multiply the raw signal with DLO of same frequency as readout. 
Code snippet given below generated DLO for given frequency.

```
    def getComplexSignal(frequency,st,end,step):
        time=np.arange(st*1e-9, end*1e-9, step*1e-9)
        rot=np.exp(1j*(-2*np.pi*frequency*time))
        return rot

    readout_freq=6558500000.003296
    dlo = getComplexSignal(readout_freq,0,4096,0.5) #(8192)


    #Muliply og signal and convert into complex signals
    state0_complex=((state0+0j)*dlo) #(1000,8192)
    state1_complex=((state1+0j)*dlo)
```

Then state0.real and state0.imag can give you I and Q component of the signal which can be used to plot trajactory
