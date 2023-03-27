import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd


# Load data into a pandas DataFrame
data = pd.read_csv('data/voice.csv', sep=',')

# Split data into features (X) and labels (y)
X = data.iloc[:, :-1].values.astype('float32')  # convert input data to float32
y = data.iloc[:, -1].values.astype('int64')    # convert output labels to int64

# Split data into train, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.2, random_state=42)


# Define the PyTorch model architecture
class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

# Create PyTorch DataLoader objects for train, validation, and test sets
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train the model
model = GenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Evaluate the model on the validation set
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total
    
    print('Epoch: {}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, val_loss, val_accuracy))

# Evaluate the model on the test set
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

test_loss /= len(test_loader.dataset)
test_accuracy = correct / total

print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))

torch.save(model.state_dict(), 'best_model.pt')
