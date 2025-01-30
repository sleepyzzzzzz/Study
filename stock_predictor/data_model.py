import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap
from qiskit_algorithms.optimizers import ADAM
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler

class StockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=1, output_size=1):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1])
        return predictions

def ans(n, depth):    
	qc = QuantumCircuit(n)    
	for j in range(depth):        
		for i in range(n):            
			param_name = f'theta_{j}_{i}'            
			theta_param = Parameter(param_name)            
			qc.rx(theta_param, i)            
			qc.ry(theta_param, i)            
			qc.rz(theta_param, i)    
	for i in range(n):        
		if i == n-1:            
			qc.cnot(i, 0)        
		else:            
			qc.cnot(i, i+1)   
	return qc

def build_and_train_lstm_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    model = StockPredictor(input_size=1, hidden_layer_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test.view(-1, 1))
    print(f'Test Loss: {test_loss.item():.4f}')
    
    return model