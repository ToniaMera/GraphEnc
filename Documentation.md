```python
mol_descriptors(smiles)

'''Produces the following descriptors: Descriptors._descList from rdkit.

smiles: a list/numpy array/pandas series of smiles
'''

ToGraph(smiles, y = None, MD = None)

''' Transform smiles to graph representation by using the Dataset function from pytorch geometric library.

smiles: a list/numpy array/pandas series of smiles
y: true values for training/validation. Default None
MD: a matrix containing molecular descriptors e.g. the output of mol_descriptors(). Default None

Node features that are used: atom type, atom charge, hybridization, degree, number of hydrogen atoms, stereochemistry, atom mass, is aromatic, is in ring. Each feature is transformed into one hot encoder and they are concatenated into a binary vector.

TrainModel(train_data, batch_sz, epochs, Xy_eval = None, model_name = 'GraphEnc.pt')

train_data: training data of Data type or the output of ToGraph.
batch_sz: defines batch size
epochs: number of epochs
model_name: the path of the model to be saved to
Xy_eval: validation dataset of Data type. Default None for no validation


'''
```
