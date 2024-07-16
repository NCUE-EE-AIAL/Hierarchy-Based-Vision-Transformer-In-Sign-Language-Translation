import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# dataset class
class MSASLDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.labels = self.load_labels(label_file)
        self.data = self.load_data()

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = f.readlines()
            labels = [int(label.strip()) for label in labels]
        return labels

    def load_data(self):
        data_files = sorted(os.listdir(self.data_dir))
        data = []
        for file in data_files:
            file_path = os.path.join(self.data_dir, file)
            landmarks = np.load(file_path)
            data.append(landmarks)
        return np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# CNN+LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, num_classes, time_steps, point_dim):
        super(CNNLSTM, self).__init__()
        self.time_steps = time_steps
        self.point_dim = point_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()
        self.conv_output_size = self._get_conv_output_size((1, self.time_steps, self.point_dim))
        self.lstm = nn.LSTM(input_size=self.conv_output_size // self.time_steps, hidden_size=128,
                           num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def _get_conv_output_size(self, shape):
        x = torch.zeros(1, *shape)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.numel()

    def forward(self, x):
        batch_size = x.size(0)
        # Reshape x to have a single channel
        x = x.view(batch_size, 1, self.time_steps, self.point_dim)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the tensor except the batch dimension
        x = x.view(batch_size, self.time_steps, -1)
        h0 = torch.zeros(2, batch_size, 128).to(x.device)
        c0 = torch.zeros(2, batch_size, 128).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x

# main function to train the model

def main():
    #parameters
    num_classes = 10
    time_steps = 30
    point_dim = 52
    data_dir = 'data/landmarks'
    label_file = 'data/labels.txt'

    #create dataset and data loader
    train_dataset = MSASLDataset(data_dir, label_file)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #initialize model, optimizer, and loss function
    model = CNNLSTM(num_classes=num_classes, time_steps=time_steps, point_dim=point_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    model_path = 'cnn_lstm_model.pth'
    torch.save(model.state_dict(), model_path)
    print('Model saved to', model_path)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_data_dir = 'data/test_landmarks'
    test_label_file = 'data/test_labels.txt'
    test_dataset = MSASLDataset(test_data_dir, test_label_file)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test dataset: {}%'.format(100 * correct / total))

if __name__ == '__main__':
    main()