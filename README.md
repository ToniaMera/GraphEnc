# GraphEnc
Prediction of lipophilicity with graph convolutional neural networks.

The model utilizes a modular neural network which consitutes of a graph convolutional neural network that processes the node features and a feed forward network, refered as encoder here, responsible for processing molecular descriptors. A graph level mean pooling operation is applied in the output of the GCN and is aggragated with the output of the encoder by addition. The aggragated vector is processed by another feed forward neural network wich predicts the logP value of a compound. An oversimplified version is depicted in the image below.

<p align="center">
  <img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/e347f711-df97-4a50-a25a-cd955076adfc" width="800" height="800">
</p>

The addition of an encoder together with the graph nn decreases both the training and the validation loss and most importantly it stabilizes the validation loss.

<p align="center">
    <img src="https://github.com/ToniaMera/GraphEnc/assets/77622398/a6fa83d4-5654-44c3-b9f9-40a61308adef" width="500" height="300">
</p>

# How to use the pretrained model to make predictions.
## 1. Prepare your data:

The preparation of data only requires to steps. First create the matrix for molecular descriptors as:

```python
MD = mol_descriptors(smiles).oned()
```

... and next create the graph data:

```python
GraphData = ToGraph(smiles = smiles, MD = MD).process()  
```

## 2. Make predictions 

To make predictions with your data can be done with two lines of code:

```python
GE = GraphEnc(Graph_Data)
predictions = GE.predict()
```

# How to train/validate your own model with the GraphEnc architecture.

## 1. Prepare your data:

Process data as previously shown but also include the true values y:

```python
MD = gr.mol_descriptors(smiles).oned()
GraphData = gr.ToGraph(smiles = smiles, y = y, MD = MD).process()  
```

There is also the possibility of not including the encoder submodule in which case the pipeline changes as it follows (exclude MD data):

```python
GraphData = gr.ToGraph(smiles = smiles, y = y).process()  
```

## 2. Train the model:

Initialize the process:

```python
tm = gr.TrainModel(train_data = train_data, batch_sz = batch_size, epochs = num_epochs,
  Xy_eval = val_data, model_name='model path')
```
...and train the GraphEnc model. It also returns all losses over the epochs for further data analysis.

```python
training_losses, validation_losses = tm.train()
```
... or without using validation data:

```python
tm = gr.TrainModel(train_data = train_data, batch_sz = batch_size, epochs = num_epochs,
  model_name='model name')
training_losses = tm.train()
```

##3. Make predictions:

Use the test data and the path of the model.

```python
GE = gr.GraphEnc(test_Xy, model_name = 'madel path')
pr = GE.predict()
```





