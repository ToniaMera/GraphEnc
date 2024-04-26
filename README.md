# GraphEnc
Prediction of lipophilicity with graph convolutional neural networks.

The model utilizes a modular neural network which consitutes of a graph convolutional neural network that processes the node features and a feed forward network, refered as encoder here, responsible for processing molecular descriptors. A graph level mean pooling operation is applied in the output of the GCN and is aggragated with the output of the encoder by addition. The aggragated vector is processed by another feed forward neural network wich predicts the logP value of a compound. An oversimplified version is depicted in the image below.

![ge_torchviz](https://github.com/ToniaMera/GraphEnc/assets/77622398/e347f711-df97-4a50-a25a-cd955076adfc)

The addition of an encoder together with the graph nn decreases both the training and the validation loss and most importantly it stabilizes the validation loss.

<img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/a6fa83d4-5654-44c3-b9f9-40a61308adef" width="500" height="350">
<br>

# How to use the model to make predictions
## 1. Prepare your data:

If you would like to use both the graph and the encoder model then 2 processing steps are required. First create the matrix for molecular descriptors as:

MD = mol_descriptors(smiles).oned()

Next create the graph data:

GD = ToGraph(smiles = smiles, MD = MD).process()  
