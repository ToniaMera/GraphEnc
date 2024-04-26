mol_descriptors(smiles)

'''Produces the following descriptors: Descriptors._descList from rdkit.
smiles: a list/numpy array/pandas series of smiles
'''

ToGraph(smiles, y = None, MD = None)

''' Transform smiles to graph representation.
smiles: a list/numpy array/pandas series of smiles
y: true values for training/validation
MD: a matrix containing molecular descriptors e.g. the output of mol_descriptors()

None features that are used: 

