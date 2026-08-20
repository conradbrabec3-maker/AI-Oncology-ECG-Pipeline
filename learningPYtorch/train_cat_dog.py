import torch
import torch.nn as nn
import torch.optim as optim
# ==========================================
# 1. THE DATA
# ==========================================
# I am generating 100 fake images (3 color channels, 64x64 pixels)
# and 100 fake labels (0 for Cat, 1 for Dog) so you can test your code immediately.
dummy_images = torch.randn(100, 3, 64, 64) 
dummy_labels = torch.randint(0, 2, (100,))  

# ==========================================
# 2. THE BRAIN (NEURAL NETWORK)
# ==========================================

class CatDogBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: The Detective's Magnifying Glass (Convolution)
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # Adds non-linearity (helps the brain learn complex patterns)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks the image to save memory
        
        # Block 2: The Decision Maker (Linear Layer)
        # After shrinking a 64x64 image, we flatten it into 16384 numbers.
        # We want to output 2 numbers (Confidence it's a Cat, Confidence it's a Dog)
        self.decision_layer = nn.Linear(in_features=16384, out_features=2)

    def forward(self, x):
        # TODO: Pass 'x' through the conv_layer
        x = self.conv_layer(x)
        
        # TODO: Pass 'x' through the relu activation
        x = self.relu(x)
        
        # TODO: Pass 'x' through the pool layer
        x = self.pool(x)
        
        # Flatten the image into a single line of numbers so the Linear layer can read it
        x = torch.flatten(x, start_dim=1)
        
        # TODO: Pass 'x' through the decision_layer
        x = self.decision_layer(x)
        
        return x

# Create an instance of our brain
model = CatDogBrain()

# ==========================================
# 3. THE TEACHER & OPTIMIZER
# ==========================================
# TODO: Create a CrossEntropyLoss function (This is the standard for classification)
criterion = nn.CrossEntropyLoss()

# TODO: Create an Adam optimizer. Pass it 'model.parameters()' and set learning rate (lr) to 0.001
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# ==========================================
# 4. THE TRAINING LOOP (The 5 Sacred Steps)
# ==========================================
epochs = 5 # We will look through our 100 images 5 times

print("Starting Training...")

for epoch in range(epochs):
    
    # --- THE 5 SACRED STEPS ---
    
    # 1. Clear the mind (zero out the gradients/mistakes from last time)
    # TODO: Call zero_grad() on your optimizer
    optimizer.zero_grad()
    
    # 2. Make a guess (pass the dummy_images into the model)
    # TODO: Pass dummy_images into 'model'
    predictions = model(dummy_images)

    
    # 3. Grade the guess (calculate the loss)
    # TODO: Pass your predictions and the true 'dummy_labels' into your criterion
    loss = criterion(predictions, dummy_labels)
    
    # 4. Learn from the mistake (calculate the math derivatives backwards)
    # TODO: Call backward() on your loss variable
    loss.backward()
    
    # 5. Update the brain (take a step in the right direction)
    # TODO: Call step() on your optimizer
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs} | Error (Loss): {loss.item():.4f}")

print("Training Complete! You just built your first AI model.")
