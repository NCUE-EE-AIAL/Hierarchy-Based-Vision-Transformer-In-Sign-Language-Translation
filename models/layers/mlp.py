from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.device = device  # Store device

        # Define layers and move them to the specified device
        self.fc1 = nn.Linear(in_features, hidden_features).to(device)
        self.act = act_layer().to(device)
        self.fc2 = nn.Linear(hidden_features, out_features).to(device)
        self.drop = nn.Dropout(drop).to(device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
