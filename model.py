import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from models import VGGish 

df = pd.read_csv('data/voice.csv')

# Separate the targets (or labels) from the features
y = df['label']
X = df.drop('label', axis=1)

# Split the original data into train, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=42, test_size=0.25)

# Load the pre-trained VGGish model
vggish_model = VGGish()
vggish_model.load_state_dict(torch.load('vggish_model.pth'))

# Define a custom classifier network on top of VGGish
classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
    nn.Sigmoid())

# Combine VGGish and the custom classifier
model = nn.Sequential(vggish_model, classifier)

# Set the model to evaluation mode
model.eval()

# Input your audio waveform data as a PyTorch tensor
audio_tensor = torch.FloatTensor(audio_data)

# Predict the binary classification label for the input audio
binary_classification = model(audio_tensor)

