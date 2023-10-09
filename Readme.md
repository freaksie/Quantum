This zip file consists of Qubit read out project in collaboration with Lawrence Berkeley National Lab.

3 main folders
    ---Code: 
            dataProcess.ipynb : This will fetch raw data, process it and split it into train and test. 
                                Three section 
                                            1. For Tradional ML, NN
                                            2. For RNN !!! Need to reconfigure parameters based on microseconds !!! 
                                            3. For CRNN
            
            svm.ipynb : This will perform PCA, then uses logistic regression and svm for different experiments.

            NN.ipynb :  This will perform PCA, then uses Neural Network for different experiments.

            rnn.ipynb : This will perform PCA, then uses Recurrent Neural Network for different experiments.

            crnn.ipynb : This file will use CRNN architecture for experiments.

            trans.ipynb : This file is for using transformer model.
    
    ---Data: 
            cnn_split: It contains 1 microsecond spectrogram data ( for 4 microsecond make changes in dataProcess.ipynb file).

            npy : Raw qubit data.

            rnn_split : It containes time series sequence (Change dataProcess.ipynb for different time)

            splitData : Data for Tradional ML.

    ---Model:
           NN : Saved model of neural networks.
                    1. 0.9150.h5 = 91% accuracy of NN model
           
           RNN : Saved model of RNNs. 
                    1. 0.9481.h5 = 94% accuracy 4μs
                    2. 0.7750.h5 = 77% accuracy 2μs
                    3. 0.6637.h5 = 66% accuracy 1μs