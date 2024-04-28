
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graph_encoder_module import graphenc as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('train.csv')

md = gr.mol_descriptors(data['Drug']).oned()


tg = gr.ToGraph(data['Drug'], data['Y'], md).process()  
train_Xy, test_Xy = train_test_split(tg, test_size=0.2, random_state=42)
tm = gr.TrainModel(train_data = train_Xy, batch_sz = 24, epochs = 200, 
                    model_name='graph_enc_model.pt')
tm.train()

GE = gr.GraphEnc(test_Xy, 'graph_enc_model.pt')
pr = GE.predict()

true_values = []
for data_point in test_Xy:
    y_true = data_point.y
    true_values.append(y_true)

rmse = np.sqrt(mean_squared_error(true_values, pr))
r2 = r2_score(true_values, pr)

print(rmse, r2)

plt.scatter(pr, true_values, s = 20, alpha = 0.5)
plt.plot([-1, 4], [-1, 4], label = 'x = y', c = 'blue', alpha = 0.5)
plt.xlabel('Predicted logP')
plt.ylabel('True LogP')
