import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Combined loss function for training the autoencoder.
        Args:
            alpha (float): Weight for latent code matching loss
            beta (float): Weight for regularization loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_latent, target_latent, predicted_sdf, target_sdf):
        # Latent code matching loss
        latent_loss = self.mse_loss(predicted_latent, target_latent)
        
        # SDF reconstruction loss
        sdf_loss = self.mse_loss(predicted_sdf, target_sdf)
        
        # L2 regularization on latent codes
        reg_loss = torch.mean(torch.norm(predicted_latent, dim=1))
        
        # Combine losses
        total_loss = sdf_loss + self.alpha * latent_loss + self.beta * reg_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'sdf_loss': sdf_loss.item(),
            'latent_loss': latent_loss.item(),
            'reg_loss': reg_loss.item()
        }

class ShapeNetTrainer:
    def __init__(
        self,
        encoder,
        decoder,
        device,
        learning_rate=1e-4,
        alpha=1.0,
        beta=0.1
    ):
        """
        Trainer for ShapeNet encoder-decoder architecture.
        Args:
            encoder: VoxelCNNEncoder instance
            decoder: SDFDecoder instance
            device: torch device
            learning_rate: Learning rate for optimizer
            alpha: Weight for latent code matching loss
            beta: Weight for regularization loss
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        
        # Initialize loss function
        self.criterion = CombinedLoss(alpha=alpha, beta=beta)
        
        # Initialize optimizers
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        # Initialize learning rate schedulers
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', patience=5, factor=0.5
        )
        self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        epoch_losses = []
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move data to device
            voxels = batch['voxels'].to(self.device)
            points = batch['points'].to(self.device)
            target_sdfs = batch['sdfs'].to(self.device)
            target_latent = batch['target_latent'].to(self.device)
            
            # Zero gradients
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            
            # Forward pass
            predicted_latent = self.encoder(voxels)
            predicted_sdfs = self.decoder(predicted_latent, points)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                predicted_latent, target_latent,
                predicted_sdfs, target_sdfs
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses over epoch
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        return avg_losses
    
    def validate(self, dataloader):
        """Validate the model."""
        self.encoder.eval()
        self.decoder.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                voxels = batch['voxels'].to(self.device)
                points = batch['points'].to(self.device)
                target_sdfs = batch['sdfs'].to(self.device)
                target_latent = batch['target_latent'].to(self.device)
                
                predicted_latent = self.encoder(voxels)
                predicted_sdfs = self.decoder(predicted_latent, points)
                
                loss, loss_dict = self.criterion(
                    predicted_latent, target_latent,
                    predicted_sdfs, target_sdfs
                )
                
                val_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])
        
        return avg_losses
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir=None):
        """Full training routine."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            print("Training Losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
            
            # Validate
            val_losses = self.validate(val_loader)
            print("Validation Losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
            
            # Update learning rate schedulers
            self.encoder_scheduler.step(val_losses['total_loss'])
            self.decoder_scheduler.step(val_losses['total_loss'])
            
            # Save checkpoint if best model
            if val_losses['total_loss'] < best_val_loss and checkpoint_dir:
                best_val_loss = val_losses['total_loss']
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, f"{checkpoint_dir}/best_model.pth")

# Example usage
def train_shapenet_model(encoder, decoder, train_dataset, val_dataset, config):
    """
    Main training function.
    Args:
        encoder: VoxelCNNEncoder instance
        decoder: SDFDecoder instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration dictionary
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize trainer
    trainer = ShapeNetTrainer(
        encoder=encoder,
        decoder=decoder,
        device=config['device'],
        learning_rate=config['learning_rate'],
        alpha=config['alpha'],
        beta=config['beta']
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )