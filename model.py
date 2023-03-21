import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data into a pandas DataFrame
data = pd.read_csv('data/voice.csv')

# Split data into features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define PyTorch DataLoader
train_data = list(zip(torch.FloatTensor(X_train), torch.LongTensor(y_train)))
test_data = list(zip(torch.FloatTensor(X_test), torch.LongTensor(y_test)))
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(21, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 25

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == target)
        
    train_loss /= len(train_loader)
    train_acc = train_acc.float() / len(train_loader.dataset)
    
    print('Epoch: {} \tTrain Loss: {:.6f} \tTrain Accuracy: {:.2f}'.format(
            epoch+1, train_loss, (train_acc*100)))
  
# Test the model
test_loss = 0.0
test_acc = 0.0

for batch_idx, (data, target) in enumerate(test_loader):
    outputs = model(data)
    loss = criterion(outputs, target)
    
    test_loss += loss.item()
    _, preds = torch.max(outputs, 1)
    test_acc += torch.sum(preds == target)

test_loss /= len(test_loader)
test_acc = test_acc.float() / len(test_loader.dataset)

print('Test Loss: {:.6f} \tTest Accuracy: {:.2f}'.format(test_loss, (test_acc*100)))


# predicted labels for test data
model.eval()
with torch.no_grad():
    y_pred = []
    for data, _ in test_loader:
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.tolist())

# Print predicted labels and voices for a number of test samples
num_samples = 10
for i in range(num_samples):
    print('Predicted label: {} \tActual label: {} \tVoice: {}'.format(
        y_pred[i], y_test[i], X_test[i]))

