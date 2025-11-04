import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import pickle
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, precision_score, 
                           f1_score, confusion_matrix, matthews_corrcoef)
from sklearn.model_selection import train_test_split
import random
import logging
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
import json
from datetime import datetime

# Configuration class for easy ablation studies
class Config:
    def __init__(self):
        # Model architecture flags for ablation studies
        self.use_vgg_features = True
        self.use_signal_patching = False
        self.use_rope_attention = True
        self.use_dim_reduction = True
        self.use_data_augmentation = True
        self.use_validation = True
        self.use_lr_scheduling = True
        
        # Model parameters
        self.time_shape = (33, 72)  # Will be updated based on data
        self.freq_shape = (33, 72)  # Will be updated based on data
        self.vgg_flat = 256
        self.num_classes = 2
        self.num_heads = 4
        self.embedding_dim = 64
        self.patch_size = 16
        self.max_patches = 20
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 0.001
        self.dropout_rate = 0.1
        
        # Device configuration
        self.device = 'auto'  # auto, cpu, cuda:0, cuda:1, etc.
        
        # Training data paths
        self.train_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_data.pickle'
        self.train_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_train_data_final/'
        self.train_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_md.pickle'

        # Testing data paths
        self.test_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_data.pickle'
        self.test_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_test_data_final/'
        self.test_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_md.pickle'
        
        # Output paths
        self.model_save_path = 'enhanced_model.pt'
        self.best_model_path = 'best_model.pt'
        self.checkpoint_path = 'checkpoint.pt'
        self.results_path = 'training_results.json'

# Set up logging
def setup_logging(config):
    log_level = logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    return logger

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check and configure device
def setup_device(config):
    device = config.device
    
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device)
        logger.info(f"Using device: {device}")
    
    return device

def process_signals(data):
    """Process signals to extract time and frequency domain features"""
    sample_rate = 7e9
    start_time = 1.00e-9
    stop_time = 5.85e-9

    start_index = int(start_time * sample_rate)
    stop_index = int(stop_time * sample_rate)

    time_domain_signals = []
    freq_domain_signals = []
    
    num_freq_points = stop_index - start_index
    
    for sample in range(data.shape[0]):
        spectrogram_complex = data[sample, :, :]
        
        # Time domain processing
        spectrogram_time_domain = np.fft.ifft(spectrogram_complex, axis=0)
        spectrogram_magnitude_time = np.abs(spectrogram_time_domain)
        cropped_signal_time = spectrogram_magnitude_time[start_index:stop_index, :]
        cropped_signal_normalized_time = cropped_signal_time / np.max(cropped_signal_time)
        
        # Frequency domain processing
        spectrogram_magnitude_freq = np.abs(spectrogram_complex)
        cropped_signal_freq = spectrogram_magnitude_freq[start_index:stop_index, :]
        cropped_signal_normalized_freq = cropped_signal_freq / np.max(cropped_signal_freq)
        
        time_domain_signals.append(cropped_signal_normalized_time)
        freq_domain_signals.append(cropped_signal_normalized_freq)
    
    return np.array(time_domain_signals), np.array(freq_domain_signals)

class RotaryPositionalEmbedding(nn.Module):
    """Proper implementation of Rotary Position Embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, seq_len=None):
        # RoPE implementation - simplified version
        if seq_len is None:
            seq_len = x.size(-2)
        
        # Create position indices
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        # Create frequency bands
        freqs = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device) 
                         * (-np.log(10000.0) / self.dim))
        
        # Compute rotary embeddings
        emb = position * freqs
        cos_emb = emb.cos()[None, :, None]
        sin_emb = emb.sin()[None, :, None]
        
        # Apply to x
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)

class RoPEAttention(nn.Module):
    """RoPE Attention with proper rotary position embeddings"""
    def __init__(self, dim, num_heads=4):
        super(RoPEAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
        # RoPE implementation
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to queries and keys
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.wo(out)
        
        return out

class SimpleAttention(nn.Module):
    """Regular attention without RoPE for ablation"""
    def __init__(self, dim, num_heads=4):
        super(SimpleAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5)
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.wo(out)
        
        return out

class SignalPatching(nn.Module):
    """Signal patching for local pattern capture"""
    def __init__(self, patch_size=16, max_patches=20, embedding_dim=64):
        super(SignalPatching, self).__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embedding_dim = embedding_dim
        
        # Linear layer to embed flattened patches
        self.patch_embedding = nn.Linear(patch_size * patch_size, self.embedding_dim)
        
    def create_patches(self, signal):
        """Create patches from signal"""
        B, H, W = signal.shape
        all_patches = []
        patch_count = 0
        
        # Calculate stride for overlapping patches
        stride = max(1, self.patch_size // 2)
        
        # Create patches with overlap
        for i in range(0, H - self.patch_size + 1, stride):
            for j in range(0, W - self.patch_size + 1, stride):
                if patch_count >= self.max_patches:
                    break
                    
                # Extract patch
                patch = signal[:, i:i+self.patch_size, j:j+self.patch_size]
                
                # Pad patch if needed
                if patch.shape[1] < self.patch_size or patch.shape[2] < self.patch_size:
                    pad_h = self.patch_size - patch.shape[1]
                    pad_w = self.patch_size - patch.shape[2]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h))
                
                # Flatten and embed
                patch_flat = patch.reshape(B, -1)
                embedded = self.patch_embedding(patch_flat)
                all_patches.append(embedded)
                patch_count += 1
        
        # If no patches created, create a single global patch
        if not all_patches:
            # Use adaptive pooling to create a fixed-size feature
            global_patch = F.adaptive_avg_pool2d(signal, (self.patch_size, self.patch_size))
            patch_flat = global_patch.reshape(B, -1)
            embedded = self.patch_embedding(patch_flat)
            all_patches.append(embedded)
        
        # Concatenate all patches
        if len(all_patches) == 1:
            patches = all_patches[0].unsqueeze(1)  # (B, 1, embedding_dim)
        else:
            patches = torch.stack(all_patches, dim=1)  # (B, num_patches, embedding_dim)
        
        return patches
        
    def forward(self, x):
        """x: (B, H, W)"""
        return self.create_patches(x)

class DimensionalityReducer(nn.Module):
    """Dimensionality reduction to prevent feature duplication"""
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.linear(x)

class FeatureExtractor(nn.Module):
    """CNN feature extractor for signal processing"""
    def __init__(self, input_channels=1, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Linear(64 * 4 * 4, feature_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EnhancedSeparateDomainModel(nn.Module):
    """Enhanced model with configurable components for ablation studies"""
    def __init__(self, config):
        super(EnhancedSeparateDomainModel, self).__init__()
        
        self.config = config
        self.use_vgg_features = config.use_vgg_features
        self.use_signal_patching = config.use_signal_patching
        self.use_rope_attention = config.use_rope_attention
        self.use_dim_reduction = config.use_dim_reduction
        
        # Get flattened dimensions
        time_flat = config.time_shape[0] * config.time_shape[1]
        freq_flat = config.freq_shape[0] * config.freq_shape[1]
        vgg_flat = config.vgg_flat
        
        # Signal feature extractors
        self.time_extractor = FeatureExtractor(input_channels=1, feature_dim=128)
        self.freq_extractor = FeatureExtractor(input_channels=1, feature_dim=128)
        
        # Signal patching (optional)
        if self.use_signal_patching:
            self.time_patcher = SignalPatching(config.patch_size, config.max_patches, config.embedding_dim)
            self.freq_patcher = SignalPatching(config.patch_size, config.max_patches, config.embedding_dim)
        
        # Dimensionality reduction (optional)
        if self.use_dim_reduction:
            self.time_reducer = DimensionalityReducer(128, 64)
            self.freq_reducer = DimensionalityReducer(128, 64)
        else:
            self.time_reducer = nn.Identity()
            self.freq_reducer = nn.Identity()
        
        # VGG feature reducer (optional)
        if self.use_vgg_features:
            if self.use_dim_reduction:
                self.vgg_reducer = DimensionalityReducer(vgg_flat, 64)
            else:
                self.vgg_reducer = nn.Linear(vgg_flat, 128)
        
        # Attention mechanism
        attention_class = RoPEAttention if self.use_rope_attention else SimpleAttention
        self.time_attention = attention_class(64, num_heads=4)
        self.freq_attention = attention_class(64, num_heads=4)
        
        # Calculate final dimension
        if self.use_vgg_features:
            if self.use_signal_patching:
                final_dim = 64 * 2 + config.embedding_dim * 2  # time + freq + patches
            else:
                final_dim = 64 * 3  # time + freq + vgg
        else:
            if self.use_signal_patching:
                final_dim = 64 * 2 + config.embedding_dim * 2  # time + freq + patches
            else:
                final_dim = 64 * 2  # time + freq
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_classes)
        )

    def forward(self, time_input, freq_input, vgg_input=None):
        # Process time domain signals
        time_features = self.time_extractor(time_input)  # (B, 128)
        
        if self.use_signal_patching:
            time_patches = self.time_patcher(time_input)  # (B, num_patches, embedding_dim)
            time_patch_features = time_patches.mean(dim=1)  # (B, embedding_dim)
        
        time_features = self.time_reducer(time_features)
        time_features = time_features.unsqueeze(1)  # Add seq dim
        time_attended = self.time_attention(time_features).squeeze(1)  # (B, 64)
        
        # Process frequency domain signals
        freq_features = self.freq_extractor(freq_input)  # (B, 128)
        
        if self.use_signal_patching:
            freq_patches = self.freq_patcher(freq_input)  # (B, num_patches, embedding_dim)
            freq_patch_features = freq_patches.mean(dim=1)  # (B, embedding_dim)
        
        freq_features = self.freq_reducer(freq_features)
        freq_features = freq_features.unsqueeze(1)  # Add seq dim
        freq_attended = self.freq_attention(freq_features).squeeze(1)  # (B, 64)
        
        # Combine features
        features = [time_attended, freq_attended]
        
        if self.use_signal_patching:
            features.extend([time_patch_features, freq_patch_features])
        
        if self.use_vgg_features and vgg_input is not None:
            vgg_flat = vgg_input.view(vgg_input.size(0), -1)
            vgg_features = self.vgg_reducer(vgg_flat)
            features.append(vgg_features)
        
        combined = torch.cat(features, dim=1)
        
        # Final classification
        output = self.classifier(combined)
        return output

class FusionDataset(Dataset):
    def __init__(self, time_signals, freq_signals, folder_path, metadata_path, transform=None, use_vgg=True):
        self.time_signals = time_signals
        self.freq_signals = freq_signals
        self.folder_path = folder_path
        self.use_vgg = use_vgg
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.labels = self.extract_labels()
        self.transform = transform
        self.image_names = self.get_image_names()

        logger.info(f"Dataset initialized with {len(self.time_signals)} samples")
        
        if len(self.time_signals) != len(self.freq_signals):
            raise ValueError("Mismatch between time and frequency signals")
        
        if self.use_vgg and len(self.time_signals) != len(self.image_names):
            logger.warning(f"Mismatch: signals={len(self.time_signals)}, images={len(self.image_names)}")
            # Adjust image names to match signals
            if len(self.image_names) < len(self.time_signals):
                # Repeat last image if needed
                while len(self.image_names) < len(self.time_signals):
                    self.image_names.append(self.image_names[-1])
            else:
                # Truncate images
                self.image_names = self.image_names[:len(self.time_signals)]

    def extract_labels(self):
        return [1 if isinstance(entry, dict) and 'tum_rad' in entry and pd.notna(entry['tum_rad']) else 0 
                for entry in self.metadata]

    def get_image_names(self):
        image_names = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_names.append(os.path.join(root, file))
        return sorted(image_names)

    def __len__(self):
        return len(self.time_signals)

    def __getitem__(self, idx):
        time_signal = self.time_signals[idx]
        freq_signal = self.freq_signals[idx]
        
        if self.use_vgg:
            img_name = self.image_names[idx]
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            with torch.no_grad():
                image = image.unsqueeze(0).to(device)
                vgg_features = vgg16_feature_extractor(image).cpu()
        else:
            vgg_features = None

        time_signal = torch.tensor(time_signal, dtype=torch.float32)
        freq_signal = torch.tensor(freq_signal, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return time_signal, freq_signal, vgg_features.squeeze(0) if vgg_features is not None else torch.zeros(256), label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch=0):
    """Training function with validation and checkpointing"""
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    
    for epoch in range(start_epoch, config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for time_inputs, freq_inputs, vgg_inputs, labels in train_loader:
            time_inputs = time_inputs.to(device)
            freq_inputs = freq_inputs.to(device)
            if config.use_vgg_features:
                vgg_inputs = vgg_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            if config.use_vgg_features:
                outputs = model(time_inputs, freq_inputs, vgg_inputs)
            else:
                outputs = model(time_inputs, freq_inputs)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * time_inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation phase
        val_accuracy = 0.0
        if config.use_validation and val_loader is not None:
            val_accuracy = evaluate_model(model, val_loader, config)
            val_accuracies.append(val_accuracy)
            
            if config.use_lr_scheduling and scheduler is not None:
                scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), config.best_model_path)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': epoch_loss,
            'train_accuracy': epoch_accuracy,
            'val_accuracy': val_accuracy,
            'config': config.__dict__
        }
        torch.save(checkpoint, config.checkpoint_path)

        logger.info(f"Epoch {epoch+1}/{config.num_epochs}, "
                   f"Train Loss: {epoch_loss:.4f}, "
                   f"Train Acc: {epoch_accuracy:.4f}, "
                   f"Val Acc: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, config):
    """Evaluate model on test/validation set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for time_inputs, freq_inputs, vgg_inputs, labels in test_loader:
            time_inputs = time_inputs.to(device)
            freq_inputs = freq_inputs.to(device)
            if config.use_vgg_features:
                vgg_inputs = vgg_inputs.to(device)
            labels = labels.to(device)
            
            if config.use_vgg_features:
                outputs = model(time_inputs, freq_inputs, vgg_inputs)
            else:
                outputs = model(time_inputs, freq_inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def evaluate_detailed(model, test_loader, config):
    """Detailed evaluation with all metrics"""
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for time_inputs, freq_inputs, vgg_inputs, labels in test_loader:
            time_inputs = time_inputs.to(device)
            freq_inputs = freq_inputs.to(device)
            if config.use_vgg_features:
                vgg_inputs = vgg_inputs.to(device)
            labels = labels.to(device)
            
            if config.use_vgg_features:
                outputs = model(time_inputs, freq_inputs, vgg_inputs)
            else:
                outputs = model(time_inputs, freq_inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

    return np.array(true_labels), np.array(predictions), np.array(probabilities)

def plot_training_metrics(train_losses, train_accuracies, val_accuracies, config):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train')
    if val_accuracies:
        plt.plot(val_accuracies, label='Validation')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if val_accuracies:
        plt.plot(val_accuracies, label='Validation')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(config, train_losses, train_accuracies, val_accuracies, test_metrics):
    """Save training results to JSON"""
    results = {
        'config': config.__dict__,
        'training': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        },
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(config.results_path, 'w') as f:
        json.dump(results, f, indent=2)

def run_ablation_study():
    """Run ablation study with different configurations"""
    ablation_configs = [
        # Config 1: Baseline - only time and frequency features
        {'use_vgg_features': False, 'use_signal_patching': False, 'use_rope_attention': False},
        # Config 2: Add VGG features
        {'use_vgg_features': True, 'use_signal_patching': False, 'use_rope_attention': False},
        # Config 3: Add signal patching
        {'use_vgg_features': False, 'use_signal_patching': True, 'use_rope_attention': False},
        # Config 4: Add RoPE attention
        {'use_vgg_features': False, 'use_signal_patching': False, 'use_rope_attention': True},
        # Config 5: VGG + RoPE
        {'use_vgg_features': True, 'use_signal_patching': False, 'use_rope_attention': True},
        # Config 6: All components
        {'use_vgg_features': True, 'use_signal_patching': True, 'use_rope_attention': True},
    ]
    
    results = {}
    
    for i, config_params in enumerate(ablation_configs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Running ablation experiment {i+1}/6")
        logger.info(f"Config: {config_params}")
        logger.info(f"{'='*50}")
        
        # Create config with current parameters
        config = Config()
        for param, value in config_params.items():
            setattr(config, param, value)
        
        # Update data paths
        config.train_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_data.pickle'
        config.train_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_train_data_final/'
        config.train_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_md.pickle'
        config.test_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_data.pickle'
        config.test_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_test_data_final/'
        config.test_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_md.pickle'
        
        # Run experiment
        test_metrics = run_experiment(config)
        results[f'experiment_{i+1}'] = {
            'config': config_params,
            'metrics': test_metrics
        }
    
    # Save ablation results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nAblation study completed. Results saved to ablation_results.json")
    return results

def run_experiment(config):
    """Run a single experiment with given configuration"""
    global device, vgg16_feature_extractor, logger
    
    # Setup
    set_seed(42)
    device = setup_device(config)
    
    # Load data
    logger.info("Loading training data...")
    with open(config.train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    train_time_signals, train_freq_signals = process_signals(train_data)
    
    # Create validation split if using validation
    if config.use_validation:
        train_idx, val_idx = train_test_split(
            range(len(train_time_signals)), 
            test_size=0.2, 
            random_state=42,
            stratify=[1 if i in [j for j in range(len(train_time_signals)) if train_time_signals[j].sum() > 0] else 0 
                     for i in range(len(train_time_signals))]
        )
        
        train_time_signals = train_time_signals[train_idx]
        train_freq_signals = train_freq_signals[val_idx]
        val_time_signals = train_time_signals[val_idx]
        val_freq_signals = train_freq_signals[val_idx]
    
    logger.info("Loading test data...")
    with open(config.test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    test_time_signals, test_freq_signals = process_signals(test_data)
    
    # Update config with actual data shapes
    config.time_shape = train_time_signals.shape[1:]
    config.freq_shape = train_freq_signals.shape[1:]
    
    # VGG16 feature extractor setup
    if config.use_vgg_features:
        logger.info("Setting up VGG16 feature extractor...")
        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        global vgg16_feature_extractor
        vgg16_feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:-1]).to(device)
        vgg16_feature_extractor.eval()
    
    # Image transformation
    if config.use_data_augmentation:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    # Create datasets
    if config.use_validation:
        train_dataset = FusionDataset(train_time_signals, train_freq_signals, 
                                    config.train_folder_path, config.train_metadata_path, 
                                    transform, use_vgg=config.use_vgg_features)
        val_dataset = FusionDataset(val_time_signals, val_freq_signals, 
                                  config.train_folder_path, config.train_metadata_path, 
                                  transform, use_vgg=config.use_vgg_features)
    else:
        train_dataset = FusionDataset(train_time_signals, train_freq_signals, 
                                    config.train_folder_path, config.train_metadata_path, 
                                    transform, use_vgg=config.use_vgg_features)
        val_dataset = None
    
    test_dataset = FusionDataset(test_time_signals, test_freq_signals, 
                               config.test_folder_path, config.test_metadata_path, 
                               transform, use_vgg=config.use_vgg_features)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    logger.info("Creating model...")
    model = EnhancedSeparateDomainModel(config).to(device)
    
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = None
    if config.use_lr_scheduling:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path)
        if checkpoint['config']['use_vgg_features'] == config.use_vgg_features:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resuming from epoch {start_epoch}")
    
    # Train model
    logger.info("Starting training...")
    train_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch
    )
    
    # Load best model for evaluation
    if os.path.exists(config.best_model_path):
        model.load_state_dict(torch.load(config.best_model_path))
        logger.info("Loaded best model for evaluation")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    true_labels, predictions, probabilities = evaluate_detailed(model, test_loader, config)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, probabilities)
    specificity = recall_score(true_labels, predictions, pos_label=0)
    mcc = matthews_corrcoef(true_labels, predictions)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Test Set Metrics for current configuration:")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"{'='*50}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, train_accuracies, val_accuracies, config)
    
    # Save results
    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    save_results(config, train_losses, train_accuracies, val_accuracies, test_metrics)
    
    # Save final model
    torch.save(model.state_dict(), config.model_save_path)
    logger.info(f"Training completed. Model saved as {config.model_save_path}")
    
    return test_metrics

if __name__ == "__main__":
    global logger, device, vgg16_feature_extractor
    
    # Setup logging
    logger = setup_logging(None)
    
    # Example 1: Run single experiment with default settings
    config = Config()
    
    # Data paths are already set in Config class, but can be overridden here if needed
    # config.train_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_data.pickle'
    # config.train_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_train_data_final/'
    # config.train_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/train_md.pickle'
    # config.test_data_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_data.pickle'
    # config.test_folder_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/figs_gen1_clean_test_data_final/'
    # config.test_metadata_path = 'E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-one/clean/test_md.pickle'
    
    # Customize config for your experiment
    config.use_vgg_features = True
    config.use_signal_patching = False
    config.use_rope_attention = True
    config.use_validation = True
    config.use_lr_scheduling = True
    
    # Run single experiment
    test_metrics = run_experiment(config)
    
    # Example 2: Run ablation study (uncomment to run)
    # ablation_results = run_ablation_study()
    
    print("\nExperiment completed successfully!")
