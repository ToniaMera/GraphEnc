import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graph_encoder_module import graphenc as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


plt.rcParams['figure.dpi'] = 300
plt.style.use('ggplot')  

data = pd.read_csv('train.csv')

md = gr.mol_descriptors(data['Drug']).oned()

rmses = []
r2s = []
maes = []

tg = gr.ToGraph(data['Drug'], data['Y'], md).process()  

for i in range(14):
    train_Xy, test_Xy = train_test_split(tg, test_size=0.2, random_state=i)
    tm = gr.TrainModel(train_data = train_Xy, batch_sz = 24, epochs = 100, 
                       model_name='g_enc.pt')
    tm.train()
        
    GE = gr.GraphEnc(test_Xy, 'g_enc.pt')
    pr = GE.predict()
    
    true_values = []
    for data_point in test_Xy:
        y_true = data_point.y
        true_values.append(y_true)
    
    rmse = np.sqrt(mean_squared_error(true_values, pr))
    r2 = r2_score(true_values, pr)
    
    print(rmse, r2)
    
    rmses.append(rmse)
    r2s.append(r2)
    maes.append(np.mean(abs(np.array(true_values) - np.array(pr))))


my_dict = {'RMSE': rmses, 'R2': r2s, 'MAE': maes}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())

plt.scatter(pr, true_values, s = 20, alpha = 0.5)
plt.plot([-1, 4], [-1, 4], label = 'x = y', c = 'blue', alpha = 0.5)
plt.xlabel('Predicted logP')
plt.ylabel('True LogP')
