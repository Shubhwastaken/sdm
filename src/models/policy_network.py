class PolicyNetwork:
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], activation='relu'):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.model = nn.Sequential()
        last_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            self.model.add_module(f'fc{i}', nn.Linear(last_dim, hidden_dim))
            self.model.add_module(f'activation{i}', getattr(nn, activation)())
            last_dim = hidden_dim
        
        self.model.add_module('output', nn.Linear(last_dim, output_dim))
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        import torch
        return self.model(x)

    def train(self, states, actions, targets):
        self.optimizer.zero_grad()
        outputs = self.forward(states)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()