import pandas as pd
import matplotlib.pyplot as plt
from graph_encoder_module import graphenc as gr
from sklearn.model_selection import train_test_split
import torch

plt.rcParams['figure.dpi'] = 300
plt.style.use('ggplot')  

data = pd.read_csv('train.csv')

md = gr.mol_descriptors(data['Drug']).oned()


tg = gr.ToGraph(data['Drug'], data['Y']).process()   

train_Xy, test_XY = train_test_split(tg, test_size=0.4, random_state=42)
val_Xy, test_Xy = train_test_split(test_XY, test_size=0.5, random_state=42)
tm1 = gr.TrainModel(train_data = train_Xy, batch_sz = 24, epochs = 200, Xy_eval = val_Xy)
g1, g2 = tm1.train()

tg = gr.ToGraph(data['Drug'], data['Y'], md).process()   
train_Xy, test_XY = train_test_split(tg, test_size=0.4, random_state=42)
val_Xy, test_Xy = train_test_split(test_XY, test_size=0.5, random_state=42)
tm2 = gr.TrainModel(train_data = train_Xy, batch_sz = 24, epochs = 200, Xy_eval = val_Xy)
e1, e2 = tm2.train()

with torch.no_grad():
    plt.plot(g1, label = 'Graph Training Loss')
    plt.plot(g2, label = 'Graph Validation Loss')
    plt.plot(e1, label = 'Graph + Encoder Training Loss')
    plt.plot(e2, label = 'Graph + Encoder Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()
        
