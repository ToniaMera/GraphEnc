from rdkit import Chem
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data, Batch
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors


class ToGraph(Dataset):
    def __init__(self, smiles, y = None, MD = None):
        self.smiles = smiles
        self.y = y
        self.MD = MD
        self.all_data = []

    def process(self):
        c = 0
        for smile, Y in zip(self.smiles, self.y):
            mol = Chem.MolFromSmiles(smile)
            nf = self.node_features(mol)
            ei = self.edge_features(mol)
            if self.MD is not None:
                md = self.MD[c]
                data = Data(x = nf, edge_index = ei, y = Y, md = md)
            else:
                data = Data(x = nf, edge_index = ei, y = Y)
            self.all_data.append(data)
            c+=1
            
        return self.all_data

    def get(self, idx):
        return self.process()[idx]
    
    def __len__(self):
        return len(self.all_data)

    def one_hot_encoding(self, element, unique_elements):
        one_hot_vector = np.zeros(len(unique_elements))
        if element in unique_elements:
            one_hot_vector += np.array(unique_elements) == element
        return list(one_hot_vector)

    def node_features(self, molecule):
        
        node_feats = []
        chiral_centers = Chem.FindMolChiralCenters(molecule)
        chiral_inds = [cc[0] for cc in chiral_centers]
        
        c = 0
        for atom in molecule.GetAtoms():
            if atom.GetIdx() in chiral_inds:
                stereo = chiral_centers[c][1]
                c+=1
                
            else:
                stereo = 'None'
                
            node_feats.append((
                *self.one_hot_encoding(str(atom.GetSymbol()), 
                                       ['C', 'N', 'O', 'F', 'P', 'S', 'I', 'K', 'Cl', 'Br', 'Mg', 'Na']), 
                *self.one_hot_encoding(atom.GetFormalCharge(), [i for i in range(-3, 4)]), 
                *self.one_hot_encoding(str(atom.GetHybridization()), ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']),  
                *self.one_hot_encoding(atom.GetDegree(), [i for i in range(5)]),
                *self.one_hot_encoding(atom.GetTotalNumHs(),  [i for i in range(5)]),
                *self.one_hot_encoding(stereo,  ['S', 'R']),
                (atom.GetMass() - 10.812)/116.092,
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                )) 

        node_feats = np.asarray(node_feats)
        node_feats = torch.tensor(node_feats, dtype = torch.float)
        
        return node_feats
        
    def edge_features(self, molecule):

        edges = []
        
        for bond in molecule.GetBonds():

            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([(i, j), (j, i)])

        edges = torch.tensor(list(zip(*edges)), dtype = torch.long)

        return (edges)
    
class mol_descriptors():
    def __init__(self,smiles):
        self.smiles = smiles
        self.mols = np.array([Chem.MolFromSmiles(sm) for sm in self.smiles])
        
    def oned(self):
        descriptors = [x[0] for x in Descriptors._descList]
        descr = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
        md = []
        for i in range(len(self.mols)):
            dsc = np.array(descr.CalcDescriptors(self.mols[i]))
            morgdesc = np.array(dsc)
            md.append(morgdesc)
        
        MD = torch.tensor(md,dtype = torch.float)
        return MD
        
class GNN(nn.Module):
    def __init__(self, input_fea, input_fea2 = None):
        super(GNN, self).__init__()
        self.input_fea = input_fea
        self.input_fea2 = input_fea2

        if self.input_fea is not None:
            
            self.conv0 = GCNConv(self.input_fea, 2*self.input_fea)
            self.conv1 = GCNConv(2*self.input_fea, 4*self.input_fea)
            self.conv2 = GCNConv(4*self.input_fea, 6*self.input_fea)
            self.conv3 = GCNConv(6*self.input_fea, 8*self.input_fea)

        if self.input_fea2 is not None:
            
            self.encoder = nn.Sequential(
                                    nn.Linear(self.input_fea2, 220), 
                                    nn.BatchNorm1d(220), nn.ReLU(True), nn.Dropout(0.2),
                                    nn.Linear(220, 282), 
                                    nn.BatchNorm1d(282), nn.ReLU(True), nn.Dropout(0.2),
                                    nn.Linear(282, 8*self.input_fea), 
                                    nn.BatchNorm1d(8*self.input_fea), nn.ReLU(True), nn.Dropout(0.2))
        
        self.decoder = nn.Sequential(
                                nn.Linear(8*self.input_fea, 4*self.input_fea), 
                                nn.BatchNorm1d(4*self.input_fea), nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(4*self.input_fea, 2*self.input_fea), 
                                nn.BatchNorm1d(2*self.input_fea), nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(2*self.input_fea, self.input_fea), 
                                nn.BatchNorm1d(self.input_fea), nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(self.input_fea, 1))
        
    def forward(self, x1, edge_index, batch_ind = None, x2 = None):
        x1 = self.conv0(x1, edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv1(x1, edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv2(x1, edge_index)
        x1 = torch.relu(x1)
        x1 = self.conv3(x1, edge_index)
        x1 = torch.relu(x1)

        if batch_ind is not None:
            x1 = global_max_pool(x1, batch_ind)
        else:
            dummy_btc = torch.zeros(x1.shape[0], dtype=torch.int64)
            x1 = global_max_pool(x1, dummy_btc)

        if x2 is not None:
            x2 = self.encoder(x2)
            x = x1 + x2
        else:
            x = x1
        
        y = self.decoder(x)
        
        return y
            
            
class TrainModel():
    def __init__(self, train_data, batch_sz, epochs, Xy_eval = None, model_name = 'GraphEnc.pt'):
        self.train_data = train_data
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.Xy_train = DataLoader(dataset = self.train_data, batch_size = self.batch_sz, 
                                   collate_fn = self.custom_collate)
        self.Xy_eval = Xy_eval

        self.model_name = model_name
        try:
            self.model = GNN(self.train_data[0].num_node_features, self.train_data[0].md.shape[0])
        except AttributeError:
            self.model = GNN(self.train_data[0].num_node_features)
            
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def custom_collate(self, batch):
        return Batch.from_data_list(batch)
    
    def train(self, print_loss = True):
        
        all_losses = []
        all_losses_eval = []
        
        for epoch in range(self.epochs):
            self.model.train()
            ls = 0
            for (k, batch) in enumerate(self.Xy_train):
                x1 = batch.x
                ed = batch.edge_index
                y_true = batch.y
                
                try:
                    x2 = batch.md
                    x2 = x2.reshape(torch.unique(batch.batch).shape[0], -1)
                except AttributeError:
                    x2 = None

                self.optimizer.zero_grad()

                y_pred = self.model.forward(x1, ed, batch_ind = batch.batch, x2 = x2)
                y_pred = y_pred.reshape(-1)

                loss = self.criterion(y_pred, y_true)

                loss.backward()
                self.optimizer.step()
                
                ls += loss
            all_losses.append(ls/k)
            
            if print_loss:
                print('Epoch [{}/{}], Loss_class: {:.4f}'.format(epoch+1, self.epochs, ls/k))
                
            if self.Xy_eval is not None:

                self.model.eval()
                
                with torch.no_grad():
                    
                    true_values = []
                    predicted_values = []
                    ls_eval = 0
                    
                    for data_point in self.Xy_eval:
                        x1 = data_point.x
                        ed = data_point.edge_index
                        y_true = data_point.y
                        
                        try:
                            x2 = data_point.md
                            x2 = x2.reshape(1,-1)
                        except AttributeError:
                            x2 = None

                        y_pred = self.model(x1, ed, data_point.batch, x2)
                        predicted_values.append(y_pred)
                        true_values.append(y_true)
                        
                        eval_loss = self.criterion(torch.tensor(predicted_values), 
                                                   torch.tensor(true_values))
                        ls_eval += eval_loss
                    all_losses_eval.append(ls_eval/len(self.Xy_eval))
                    
        torch.save(self.model.state_dict(), self.model_name)
        if self.Xy_eval is not None:
            return (all_losses, all_losses_eval)
        else:
            return (all_losses)
        
class GraphEnc():
    def __init__(self, X, model_name = 'GraphEnc.pt'):
        self.X = X
        self.model_name = model_name
        self.model = GNN(self.X[0].num_node_features, self.X[0].md.shape[0])
        self.model.load_state_dict(torch.load(self.model_name))
        
    def predict(self):
        self.model.eval()
        predicted_values = []
        with torch.no_grad():
            for data_point in self.X:
                x1 = data_point.x
                ed = data_point.edge_index
                try:
                    x2 = data_point.md
                    x2 = x2.reshape(1,-1)
                except AttributeError:
                    x2 = None

                y_pred = self.model(x1, ed, data_point.batch, x2)
                predicted_values.append(y_pred.item())
        return predicted_values
