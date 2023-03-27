import torch
import torch.nn as nn

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

# Instantiate the model and load saved weights
model = GenderClassifier()
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()

# Transform your voice features into a PyTorch tensor
# Replace the placeholders with your actual voice feature values!
input_features = torch.tensor([1.5, 0.7, 0.9, 1.2, 0.5, 0.8, 0.6, 0.3, 0.2, 0.7, 0.1, 
                               0.4, 0.6, 0.9, 1.1, 1.2, 0.8, 0.5, 0.3, 0.6, 0.7], 
                               dtype=torch.float32).unsqueeze(0)

# Make a prediction on your voice features
output = model(input_features)
prediction = torch.argmax(output, dim=1).item()

# Print the predicted gender
if prediction == 0:
    print("Your voice is predicted to be male.")
else:
    print("Your voice is predicted to be female.")
