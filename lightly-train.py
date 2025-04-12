
# Install LightTrain
!pip install lightly-train

import lightly_train
lightly_train.__version__

# Download the dataset
!git clone https://github.com/lightly-ai/dataset_clothing_images.git my_data_dir
!rm -r my_data_dir/.git

# Train the model with LightlyTrain
lightly_train.train(
    out="out/my_experiment",
    data="my_data_dir",
    model="torchvision/resnet18",
    epochs = 100,
    batch_size = 128
)

# Fine-Tune the pretrain model

# Prepare the dataset
import torch
import torchvision.transforms.v2 as v2
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Load the dataset
from torch.utils.data import DataLoader
from torchvision import datasets
dataset = datasets.ImageFolder(root="my_data_dir", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

# Load the pretrain model
from torchvision import models
model = models.resnet18()
model.load_state_dict(torch.load("out/my_experiment/exported_models/exported_last.pt", weights_only=True))

# Set the architecture head
from torch import nn
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

# Set the GPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Set the loss function and optimizer
from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
import tqdm
print("Starting fine-tuning...")
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for inputs, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

