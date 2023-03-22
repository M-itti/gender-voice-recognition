import torch
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

# Define PyTorch Datasets and DataLoaders
train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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

def train():
    # Train the model
    num_epochs = 25
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0

        # Train loop
        for data, target in train_loader:
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

        # Validation loop
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == target)

            val_loss /= len(val_loader)
            val_acc = val_acc.float() / len(val_loader.dataset)

        print(f"Epoch: {epoch+1}\t"
              f"Train Loss: {train_loss:.6f}\t"
              f"Train Accuracy: {train_acc*100:.2f}\t"
              f"Validation Loss: {val_loss:.6f}\t"
              f"Validation Accuracy: {val_acc*100:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    return best_model_state

def test(best_model_state):
    # Test the best model
    model.load_state_dict(best_model_state)
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == target)

    test_loss /= len(test_loader)
    test_acc = test_acc.float() / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.6f}\t"
          f"Test Accuracy: {test_acc*100:.2f}") 

if __name__ == "__main__":
    best_model_state = train()   
    test(best_model_state)
    torch.save(model.state_dict(), 'model.pth')

