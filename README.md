# GraphEnc

Prediction of lipophilicity with graph convolutional neural networks.

The model utilizes a modular neural network which consitutes of a graph convolutional neural network that processes the node features and a feed forward network, refered as encoder here, responsible for processing molecular descriptors. A graph level mean pooling operation is applied in the output of the GCN and is aggragated with the output of the encoder by addition. The aggragated vector is processed by another feed forward neural network wich predicts the logP value of a compound. An abstract depiction is shown in the left subfigure below and an oversimplified version of the pytorch model is shown in the right subfigure. The last layer is linear without an activation function since the model is designed for regression.

<p align="center">
  <img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/4a7bdaea-5fa0-4459-9690-89353054d39c" width="900" height="350">
</p>

The addition of an encoder together with the graph neural network decreases both the training and the validation loss and most importantly it highly stabilizes the validation loss as can be seen in the left figure below. 


<p align="center">
  <img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/ec9a1692-791e-40d0-b1f7-de2ad56dfd3e" width="500" height="350">
</p>

To estimate the performance of the model the following metrics were utilized: Root Mean Squared Error (RMSE), R2 and Mean Absolute Error (MAE). The results are shown in the figure below.

<p align="center">
  <img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/01160241-fefa-4c00-a1eb-8f5c0a9388ec", width = "550", heght="400">
</p>

# Installation instructions

To install the package use the following command:


```python
pip install git+https://github.com/ToniaMera/GraphEnc.git
```
when the installation has been succesfully completed import the package.

```python
from graph_encoder_module import graphenc as gr
```
# Dependencies

Install dependencies:

```python
pip install torch rdkit numpy torch_geometric
```

# How to use the pretrained model to make predictions.
## 1. Prepare your data:

The preparation of data only requires two steps. First create the matrix for molecular descriptors as:

```python
MD = gr.mol_descriptors(smiles).oned()
```

... and next create the graph data:

```python
Graph_Data = gr.ToGraph(smiles = smiles, MD = MD).process()  
```

## 2. Make predictions 

Making predictions with a new dataset can be done with two lines of code:

```python
GE = gr.GraphEnc(X = Graph_Data)
predictions = GE.predict()
```

# How to train/validate your own model with the GraphEnc architecture.

## 1. Prepare your data:

Process data as previously shown but also include the true values y:

```python
MD = gr.mol_descriptors(smiles).oned()
Graph_Data = gr.ToGraph(smiles = smiles, y = y, MD = MD).process()  
```

There is also the possibility of not including the encoder submodule in which case the pipeline changes as it follows (exclude MD data):

```python
Graph_Data = gr.ToGraph(smiles = smiles, y = y).process()  
```

## 2. Train the model:

Initialize the process:

```python
tm = gr.TrainModel(train_data = Graph_Data, batch_sz = batch_size, epochs = num_epochs,
  Xy_eval = val_data, model_name='model path')
```
...and train the GraphEnc model. The class stores and returns all losses over the epochs for further data analysis.

```python
training_losses, validation_losses = tm.train()
```
... or without using validation data:

```python
tm = gr.TrainModel(train_data = Graph_Data, batch_sz = batch_size, epochs = num_epochs,
  model_name='model name')
training_losses = tm.train()
```
To print the model architecture

```python
tm.model
```

There is no need to explicitly define wether to train the graphenc model or only the graph convolutional model. The class detects if data of molecular descriptors are available and trains the corresponding model.
## 3. Make predictions:

Use the test data and the path of your trained model.

```python
test_data = gr.ToGraph(smiles = smiles_test).process() 
GE = gr.GraphEnc(X = test_data, model_name = 'madel path')
predictions = GE.predict()
```

Please read the documentation.md file for more information about the functions and the arguments.

