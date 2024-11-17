import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Custom dataset for handwriting samples
class HandwritingDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

# Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

# VAE model
class HandwritingVAE(nn.Module):
    def __init__(self, latent_dim):
        super(HandwritingVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# Training function
def train_model(model, train_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            
            # Reconstruction loss
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + kl_loss
            loss.backward()
            total_loss += loss.item()
            
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

# Font generation class
class FontGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HandwritingVAE(latent_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def generate_character(self, z):
        with torch.no_grad():
            sample = self.model.decoder(z)
            return sample.cpu().numpy()

    def process_user_input(self, input_image):
        # Convert user input to tensor
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        input_tensor = transform(input_image).unsqueeze(0).to(self.device)
        
        # Generate character
        with torch.no_grad():
            mu, log_var = self.model.encoder(input_tensor)
            z = self.model.reparameterize(mu, log_var)
            generated = self.model.decoder(z)
            
        return generated.cpu().numpy()

# Usage example
def main():
    # Initialize the model
    latent_dim = 128
    model = HandwritingVAE(latent_dim)
    
    # Create synthetic dataset (replace with real handwriting data)
    num_samples = 1000
    image_size = 64
    synthetic_images = [Image.new('L', (image_size, image_size)) for _ in range(num_samples)]
    synthetic_labels = list(range(num_samples))
    
    # Create dataset and dataloader
    dataset = HandwritingDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_model(model, train_loader, num_epochs=50, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'handwriting_font_model.pth')
    
    # Initialize font generator
    font_generator = FontGenerator('handwriting_font_model.pth')
    
    # Generate new characters (example)
    random_z = torch.randn(1, latent_dim).to(device)
    generated_char = font_generator.generate_character(random_z)

if __name__ == "__main__":
    main()
