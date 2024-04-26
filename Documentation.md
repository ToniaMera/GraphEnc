```python
mol_descriptors(smiles)

'''Produces the following descriptors: Descriptors._descList from rdkit.

smiles: a list/numpy array/pandas series of smiles
'''

ToGraph(smiles, y = None, MD = None)

''' Transform smiles to graph representation.

smiles: a list/numpy array/pandas series of smiles
y: true values for training/validation
MD: a matrix containing molecular descriptors e.g. the output of mol_descriptors()

Node features that are used: atom type, atom charge, hybridization, degree, number of hydrogen atoms, stereochemistry, atom mass, is aromatic, is in ring. Each feature is transformed into one hot encoder and they are concatenated into a binary vector.


'''
```
