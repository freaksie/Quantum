import numpy as np
import pandas as pd
from scipy.fft import  rfft
from scipy.fft import  rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class Graphs:
    def plotSignals(shotsMean0_I,shotsMean0_Q,shotsMean1_I,shotsMean1_Q,lo=0,hi=4096,window_size=100,steps=1):
        '''
        Takes 4 signals as input and plots interactive graphs

        Parameter
        ----------
            shotsMean0_I
        '''
        x=np.arange(lo*1e-9,hi*1e-9,steps*1e-9)
        y0_I=shotsMean0_I[lo*int(1/steps):hi*int(1/steps)]
        y0_Q=shotsMean0_Q[lo*int(1/steps):hi*int(1/steps)]
        y1_I=shotsMean1_I[lo*int(1/steps):hi*int(1/steps)]
        y1_Q=shotsMean1_Q[lo*int(1/steps):hi*int(1/steps)]

        # Calculate the moving average with a window size of your choice
        # Adjust this to set the size of the moving average window
        mov0_I = np.convolve(y0_I, np.ones(window_size) / window_size, mode='valid')
        mov0_Q = np.convolve(y0_Q, np.ones(window_size) / window_size, mode='valid')
        mov1_I = np.convolve(y1_I, np.ones(window_size) / window_size, mode='valid')
        mov1_Q = np.convolve(y1_Q, np.ones(window_size) / window_size, mode='valid')

        #Plotings
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

        # Add the original signals as traces
        trace1 = go.Scatter(x=x, y=y0_I, mode='lines', name='State 0 I')
        fig.add_trace(trace1, row=1, col=1)
        trace5 = go.Scatter(x=x, y=mov0_I, mode='lines', name='Moving Average 0 I')
        # fig.add_trace(trace5, row=1, col=1)

        trace2 = go.Scatter(x=x, y=y0_Q, mode='lines', name='State 0 Q')
        fig.add_trace(trace2, row=2, col=1)
        trace6 = go.Scatter(x=x, y=mov0_Q, mode='lines', name='Moving Average 0 Q')
        # fig.add_trace(trace6, row=2, col=1)

        trace3 = go.Scatter(x=x, y=y1_I, mode='lines', name='State 1 I')
        fig.add_trace(trace3, row=3, col=1)
        trace7 = go.Scatter(x=x, y=mov1_I, mode='lines', name='Moving Average 1 I')
        # fig.add_trace(trace7, row=3, col=1)

        trace4 = go.Scatter(x=x, y=y1_Q, mode='lines', name='State 1 Q')
        fig.add_trace(trace4, row=4, col=1)
        trace8 = go.Scatter(x=x, y=mov1_Q, mode='lines', name='Moving Average 1 Q')
        # fig.add_trace(trace8, row=4, col=1)

        # Set plot layout
        fig.update_layout(
            title='Original Signals and Moving Averages (Window of '+str(window_size)+'ns)',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Amplitude'),
            height=800,
            width=1000,
        )
        return fig
    
    def getFFT(signal,sr):

        time=1.0/sr
        frequencies=rfftfreq(len(signal),d=time)
        fourier = rfft(signal)
        amplitude= 2*np.abs(fourier)/len(signal)
        phase = np.angle(fourier,deg=False)
        return frequencies, amplitude, phase

    def plotFFT(frequencies,amplitude):
        freq_trace = go.Line(x=frequencies, y=amplitude)
        frequency_domain = [freq_trace]
        layout = go.Layout(title=dict(text='Spectrum (Frequency Domain)',
                                        x=0.5,
                                        xanchor='center',
                                        yanchor='top',
                                        font=dict(size=25, family='Arial, bold')),
                                xaxis=dict(title='Frequency[Hz]'),
                                yaxis=dict(title='amplitude'),
                                width=1000,
                                height=400)
        fig = go.Figure(data=frequency_domain, layout=layout)
        return fig
   
    def getComplexSignal(frequency,amplitude,st,end,step,phase=0,t0=2e-7):
        time=np.arange(st*1e-9, end*1e-9, step*1e-9)
        rot= amplitude * np.exp(1j * (2 * np.pi * frequency * (time+t0) + phase))
        return rot

    def plotSignal(signal, lo,hi,step):
        x=np.arange(lo*1e-9, hi*1e-9, step*1e-9)
        y=signal
        mov = np.convolve(y, np.ones(100) / 100, mode='valid')
        fig = make_subplots(rows=1, cols=1)
        trace1 = go.Scatter(x=x, y=y, mode='lines', name='Signal')
        fig.add_trace(trace1, row=1, col=1)
        trace2 = go.Scatter(x=x, y=mov, mode='lines', name='Moving Average')
        fig.add_trace(trace2, row=1, col=1)
        fig.update_layout(
                title='Signal',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Amplitude'),
                height=400,
                width=1000,
            )
        
        return fig
    

    def plotComplexTrajectory(s0,s1,st,ed,window):
        if np.iscomplex(s0).all() and np.iscomplex(s1).all():
            trc0=np.add.reduceat(s0[st:ed], range(0,ed-st,window),axis=0)/float(window)
            trc1=np.add.reduceat(s1[st:ed], range(0,ed-st,window),axis=0)/float(window)
            fig = make_subplots(rows=1, cols=1)
            # Add traces for |0> and |1>
            trace0 = go.Scatter(x=trc0.real, y=trc0.imag, mode='markers+lines', name='|0>', line=dict(color='blue'), marker=dict(size=6))
            trace1 = go.Scatter(x=trc1.real, y=trc1.imag, mode='markers+lines', name='|1>', line=dict(color='red'), marker=dict(size=6))
            trace3 = go.Scatter(x=[trc0[0].real], y=[trc0[0].imag], mode='markers+lines', name='Start of |0>', line=dict(color='blue'), marker=dict(size=16, symbol='star-dot'))
            trace4 = go.Scatter(x=[trc1[0].real], y=[trc1[0].imag], mode='markers+lines', name='Start of |1>', line=dict(color='red'), marker=dict(size=16, symbol='star-dot'))
            # Add the traces to the figure
            fig.add_trace(trace0)
            fig.add_trace(trace1)
            fig.add_trace(trace3)
            fig.add_trace(trace4)
            # Set the layout
            fig.update_layout(
                title="Trajectory",
                xaxis_title="Real",
                yaxis_title="Imaginary",
                legend=dict(x=0, y=1, traceorder="normal"),
                showlegend=True,
                width=800,  # Set the width of the plot
                height=400,  # Set the height of the plot
            )
            return fig  
        else:
            print('Expected complex array, try giving I and Q separatly if not complex number')
            return None
        

    def plotTrajectory(s0I,s0Q,s1I,s1Q,st,ed,window):
        if (np.iscomplex(s0I).all() or np.iscomplex(s0Q).all() or np.iscomplex(s1I).all() or np.iscomplex(s1Q).all())==False:
            trc0I=np.add.reduceat(s0I[st:ed], range(0,ed-st,window),axis=0)/float(window)
            trc0Q=np.add.reduceat(s0Q[st:ed], range(0,ed-st,window),axis=0)/float(window)
            trc1I=np.add.reduceat(s1I[st:ed], range(0,ed-st,window),axis=0)/float(window)
            trc1Q=np.add.reduceat(s1Q[st:ed], range(0,ed-st,window),axis=0)/float(window)
            fig = make_subplots(rows=1, cols=1)
            # Add traces for |0> and |1>
            trace0 = go.Scatter(x=trc0I, y=trc0Q, mode='markers+lines', name='|0>', line=dict(color='blue'), marker=dict(size=6))
            trace1 = go.Scatter(x=trc1I, y=trc1Q, mode='markers+lines', name='|1>', line=dict(color='red'), marker=dict(size=6))
            trace3 = go.Scatter(x=[trc0I[0]], y=[trc0Q[0]], mode='markers+lines', name='Start of |0>', line=dict(color='blue'), marker=dict(size=16, symbol='star-dot'))
            trace4 = go.Scatter(x=[trc1I[0]], y=[trc1Q[0]], mode='markers+lines', name='Start of |1>', line=dict(color='red'), marker=dict(size=16, symbol='star-dot'))
            # Add the traces to the figure
            fig.add_trace(trace0)
            fig.add_trace(trace1)
            fig.add_trace(trace3)
            fig.add_trace(trace4)
            # Set the layout
            fig.update_layout(
                title="Trajectory",
                xaxis_title="Real",
                yaxis_title="Imag",
                legend=dict(x=0, y=1, traceorder="normal"),
                showlegend=True,
                width=800,  # Set the width of the plot
                height=400,  # Set the height of the plot
            )
            return fig  
        else:
            print('Expected real array')
            return None
        

    def plotTrajectoryError(s0I, s0Q, s1I, s1Q,E0I,E0Q,E1I,E1Q, st, ed, window):
        trc0I = np.add.reduceat(s0I[st:ed], range(0, ed - st, window), axis=0) / float(window)
        trc0Q = np.add.reduceat(s0Q[st:ed], range(0, ed - st, window), axis=0) / float(window)
        trc1I = np.add.reduceat(s1I[st:ed], range(0, ed - st, window), axis=0) / float(window)
        trc1Q = np.add.reduceat(s1Q[st:ed], range(0, ed - st, window), axis=0) / float(window)
        
        # Calculate the standard deviation for error bars
        std0I = np.add.reduceat(E0I[st:ed], range(0, ed - st, window), axis=0) / float(window)
        std0Q = np.add.reduceat(E0Q[st:ed], range(0, ed - st, window), axis=0) / float(window)
        std1I = np.add.reduceat(E1I[st:ed], range(0, ed - st, window), axis=0) / float(window)
        std1Q = np.add.reduceat(E1Q[st:ed], range(0, ed - st, window), axis=0) / float(window)
        fig = make_subplots(rows=1, cols=1)

        # Add traces for |0> and |1>
        trace0 = go.Scatter(x=trc0I, y=trc0Q, mode='markers+lines', name='|0>', line=dict(color='blue'), marker=dict(size=6))
        trace1 = go.Scatter(x=trc1I, y=trc1Q, mode='markers+lines', name='|1>', line=dict(color='red'), marker=dict(size=6))
        trace3 = go.Scatter(x=[trc0I[0]], y=[trc0Q[0]], mode='markers+lines', name='Start of |0>', line=dict(color='blue'), marker=dict(size=16, symbol='star-dot'))
        trace4 = go.Scatter(x=[trc1I[0]], y=[trc1Q[0]], mode='markers+lines', name='Start of |1>', line=dict(color='red'), marker=dict(size=16, symbol='star-dot'))

        # Add error bars as standard deviation
        std0I_list = std0I.tolist()
        std0Q_list = std0Q.tolist()
        std1I_list = std1I.tolist()
        std1Q_list = std1Q.tolist()
        # Add error bars as standard deviation
        trace0.error_x = dict(array=std0I_list, visible=True)
        trace0.error_y = dict(array=std0Q_list, visible=True)
        trace1.error_x = dict(array=std1I_list, visible=True)
        trace1.error_y = dict(array=std1Q_list, visible=True)

        # Add the traces to the figure
        fig.add_trace(trace0)
        fig.add_trace(trace1)
        fig.add_trace(trace3)
        fig.add_trace(trace4)

        # Set the layout
        fig.update_layout(
            title="Trajectory with Error Bars (Standard Deviation)",
            xaxis_title="I",
            yaxis_title="Q",
            legend=dict(x=0, y=1, traceorder="normal"),
            showlegend=True,
            width=800,  # Set the width of the plot
            height=400,  # Set the height of the plot
        )
        
        return fig



    def ewmaComplexSignals(s0,s1,alpha=100):
        if 1:
            s0I=pd.DataFrame({'A': np.real(s0)}).ewm(span=alpha).mean().to_numpy().reshape((s0.shape[0]))
            s0Q=pd.DataFrame({'A': np.imag(s0)}).ewm(span=alpha).mean().to_numpy().reshape((s0.shape[0]))
            s1I=pd.DataFrame({'A': np.real(s1)}).ewm(span=alpha).mean().to_numpy().reshape((s1.shape[0]))
            s1Q=pd.DataFrame({'A': np.imag(s1)}).ewm(span=alpha).mean().to_numpy().reshape((s1.shape[0]))
            return s0I,s0Q,s1I,s1Q 
        else:
            print("Expected complex array")
            return None
    
    def ewmaSignals(s0I,s0Q,s1I,s1Q,alpha=100):
        if (np.iscomplex(s0I).all() or np.iscomplex(s0Q).all() or np.iscomplex(s1I).all() or np.iscomplex(s1Q).all())==False:
            s0I=pd.DataFrame({'A': s0I}).ewm(span=alpha).mean().to_numpy().reshape((s0I.shape[0]))
            s0Q=pd.DataFrame({'A': s0Q}).ewm(span=alpha).mean().to_numpy().reshape((s0Q.shape[0]))
            s1I=pd.DataFrame({'A': s1I}).ewm(span=alpha).mean().to_numpy().reshape((s1I.shape[0]))
            s1Q=pd.DataFrame({'A': s1Q}).ewm(span=alpha).mean().to_numpy().reshape((s1Q.shape[0]))
            return s0I,s0Q,s1I,s1Q 
        else:
            print("Expected real array with IQ data")
            return None
    
    def plotCluster(s0I,s0Q,s1I,s1Q):
        '''
        Plots the cluster given 4 parameters.

        Parameter
        ------------
            * s0I (State0 I) - np array of shape (n) ; n = counts of instances.
            * s0Q (State0 Q) - np array of shape (n) ; n = counts of instances.
            * s1I (State1 I) - np array of shape (n) ; n = counts of instances.
            * s1Q (State1 Q) - np array of shape (n) ; n = counts of instances.
        
        Returns
        ----------
            fig object
        '''
        fig = make_subplots(rows=1, cols=1)
        trace0 = go.Scatter(x=s0I, y=s0Q, name='|0>',mode='markers',line=dict(color='blue'), marker=dict(size=6))
        trace1 = go.Scatter(x=s1I, y=s1Q, name='|1>',mode='markers',line=dict(color='red'), marker=dict(size=6))
        fig.add_trace(trace0)
        fig.add_trace(trace1)
        # Set the layout
        fig.update_layout(
            title="Signals",
            xaxis_title="I",
            yaxis_title="Q",
            showlegend=True,
            width=600,  # Set the width of the plot
            height=600,  # Set the height of the plot
            # aspectratio=dict(x=1, y=1),
        )
        return fig  
    