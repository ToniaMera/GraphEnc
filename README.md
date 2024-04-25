# GraphEnc
Prediction of lipophilicity with graph convolutional neural networks.

The model utilizes a modular neural network wich consututes of a graph convolutional neural network that processes the node features and a feed forward network responsible for processing molecular descriptors. A graph level mean pooling operation is applied in the output of the GCN and is aggragated with the output of the FFN by addition. The aggragated vector is processed by another feed forward neural network wich predicts the logP value of a compound. 
