# Multipath_Wavelet_Neural_Networks
The Neural network architecture consists of a parallel multi-path wavelets followed by fully connected layers.

Multi-path wavelet neural network architecture for image classification is implemented with Tensorflow on MNIST data. The model architecture consists of a multi-path layout with several levels of wavelet decompositions performed in parallel followed by fully connected layers. These decomposition operations comprise wavelet neurons with learnable parameters, which are updated during the training phase using the back-propagation algorithm. This work is published in ICMV 2019 under title of Multi-path learnable wavelet neural network for image classification.

**Network architecture**

The architecture has two main parts: Wavelet processing layers with wavelet neurons and conventional fully connected neural network. 
