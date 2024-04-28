```python
mol_descriptors(smiles)

'''Produces the following descriptors: Descriptors._descList from rdkit. Data are normalized and Nan values, due to 0 variance, are set to 0.

smiles: a list/numpy array/pandas series of smiles
'''

ToGraph(smiles, y = None, MD = None)

''' Transform smiles to graph representation by using the Dataset function from pytorch geometric library.

smiles: a list/numpy array/pandas series of smiles
y: true values for training/validation. Must be list or numpy array. Default None
MD: a matrix containing molecular descriptors e.g. the output of mol_descriptors(). Default None

Node features that are used: atom type, atom charge, hybridization, degree, number of hydrogen atoms,
stereochemistry, atom mass, is aromatic, is in ring. Each feature is transformed into one hot encoder
 and they are concatenated into a binary vector.
'''

TrainModel(train_data, batch_sz, epochs, model_name, Xy_eval = None).train(print_loss = True)

'''
This class automates the training and validation process of the GraphEnc model that
is defined by the GNN() class (see below).

train_data: training data of Data type or the output of ToGraph.
batch_sz: defines batch size
epochs: number of epochs
model_name: the path of the model to be saved to
Xy_eval: validation dataset of Data type. Default None for no validation

print loss: False to block printing the epochs and losses. Default True
'''

GraphEnc(X, model_name = 'GraphEnc.pt')

'''
X: test dataset of Data type
model_name: the path where the trained model was stored. Default the pretrained model 'GraphEnc.pt'
'''

GNN(input_fea, input_fea2 = None)

'''The model that is used in the TrainModel() pipeline. No need to define it.

input_fea: number of node features
input_fea2: number of molecular descriptors
'''

```
