import torch
import torchaudio
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model import Net

# Load the trained model
model_path = 'voice_model.pth'
model = Net()
model.load_state_dict(torch.load(model_path))

# Define the audio preprocessing pipeline
transform = Compose([
    torchaudio.transforms.Resample(44100, 8000),  # resample audio to same rate as training data
    lambda x: np.array(x[0]),  # convert audio tensor to numpy array
    Resize(8000),              # resize audio to same length as training data
    lambda x: x.astype('float32'),  # convert audio to float32
    ToTensor(),                # convert audio to tensor
    Normalize(mean=[0.5], std=[0.5])  # normalize audio between -1 and 1
])

# Load the audio sample
sample_path = 'sample.wav'
sample, sample_rate = torchaudio.load(sample_path)

# Preprocess the audio sample
sample = transform(sample)

# Make a prediction
output = model(sample.unsqueeze(0))  # unsqueeze to add a batch dimension
prediction = torch.argmax(output)

# Print the predicted label
if prediction == 0:
    print("The model predicts that the voice is male.")
else:
    print("The model predicts that the voice is female.")

