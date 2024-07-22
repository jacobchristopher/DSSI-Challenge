# Citation: Code adapted from https://www.kaggle.com/code/plavak10/ecg-heartbeat-classification
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch

from kan import *
from kan.utils import ex_round



df1 = pd.read_csv('ecg_dataset/mitbih_train.csv')
df2 = pd.read_csv('ecg_dataset/mitbih_test.csv')

dfs = [df1,df2]
for df in dfs:
    df.columns = list(range(len(df.columns)))

data = pd.concat(dfs,axis=0).sample(frac=1.0,random_state=1).reset_index(drop=True)
data.rename(columns={data.columns[-1]:'Target'},inplace=True)


X = data.drop('Target',axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print(type(X_train))
dataset = {
    'train_input': torch.tensor(X_train.to_numpy()),
    'train_label': torch.tensor(y_train.to_numpy()).unsqueeze(1),
    'test_input': torch.tensor(X_test.to_numpy()),
    'test_label': torch.tensor(y_test.to_numpy()).unsqueeze(1),
}


torch.set_default_dtype(torch.float64)
model = KAN(width=[187, 100, 1], grid=3, k=3, seed=42)
dataset['train_input'].shape, dataset['train_label'].shape

# plot KAN at initialization
model(dataset['train_input']);
# model.plot()

# train the model
model.fit(dataset, opt="LBFGS", steps=300, lamb=0.001)
# model.plot()

torch.save(model.state_dict(), f'kan_task2_model_state.pt')


model = model.prune()
# model.plot()

model.fit(dataset, opt="LBFGS", steps=300)
torch.save(model.state_dict(), f'kan_task2_model_state.pt')


model = model.refine(10)
model.fit(dataset, opt="LBFGS", steps=300)
torch.save(model.state_dict(), f'kan_task2_model_state.pt')


mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

model.fit(dataset, opt="LBFGS", steps=50);


ex_round(model.symbolic_formula()[0][0],4)

torch.save(model.state_dict(), f'kan_task2_model_state.pt')