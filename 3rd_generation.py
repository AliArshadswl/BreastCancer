# Import necessary libraries
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import pickle
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, precision_score, 
                             f1_score, confusion_matrix, matthews_corrcoef)
import matplotlib.pyplot as pltplt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Ensure workers in DataLoader are seeded
def worker_init_fn(worker_id):
    np.random.seed(worker_id + 42)

# Set device (GPU if available, else CPU)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")




# Function to process signals into separate time and frequency domains
def process_signals(data):
    sample_rate = 8e9  # 7 GHz
    start_time = 1.25e-9  # 1 ns
    stop_time = 6.49e-9  # 6.56 ns

    start_index = int(start_time * sample_rate)
    stop_index = int(stop_time * sample_rate)

    time_domain_signals = []
    freq_domain_signals = []
    
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




# Separate Domain Model Architecture
class SeparateDomainModel(nn.Module):
    def __init__(self, time_shape, freq_shape, num_classes):
        super(SeparateDomainModel, self).__init__()
        
        # Time domain branch
        self.time_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Frequency domain branch
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # VGG feature branch
        self.vgg_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate the flattened size
        self.time_flat = self._get_conv_output(time_shape, self.time_conv)
        self.freq_flat = self._get_conv_output(freq_shape, self.freq_conv)
        self.vgg_flat = 256 * 7 * 7  # VGG16 output is 512x14x14, after our conv and pool it's 256x7x7
        
        # Joint processing layers, Concatentation, multi-branch, multi-stream
        self.classifier = nn.Sequential(
            nn.Linear(self.time_flat + self.freq_flat + self.vgg_flat, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output(self, shape, conv):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, 1, *shape))
        output_feat = conv(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, time_input, freq_input, vgg_input):
        time_features = self.time_conv(time_input)
        freq_features = self.freq_conv(freq_input)
        vgg_features = self.vgg_conv(vgg_input)
        
        time_flat = time_features.view(time_features.size(0), -1)
        freq_flat = freq_features.view(freq_features.size(0), -1)
        vgg_flat = vgg_features.view(vgg_features.size(0), -1)
        
        combined = torch.cat((time_flat, freq_flat, vgg_flat), dim=1)
        output = self.classifier(combined)
        return output



# Custom Dataset class for fusion data
class FusionDataset(Dataset):
    def __init__(self, time_signals, freq_signals, folder_path, metadata_path, transform=None):
        self.time_signals = time_signals
        self.freq_signals = freq_signals
        self.folder_path = folder_path
        try:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error loading metadata: {e}")
            raise

        self.labels = self.extract_labels()
        self.transform = transform
        self.image_names = self.get_image_names()

        logger.info(f"Number of signals: {len(self.time_signals)}")
        logger.info(f"Number of images: {len(self.image_names)}")
        logger.info(f"Number of metadata entries: {len(self.metadata)}")
        
        if not (len(self.time_signals) == len(self.freq_signals) == len(self.image_names) == len(self.metadata)):
            raise ValueError("Mismatch between number of signals, images, and metadata entries")

    def extract_labels(self):
        labels = []
        for entry in self.metadata:
            if isinstance(entry, dict) and 'tum_diam' in entry:
                labels.append(1 if pd.notna(entry['tum_diam']) else 0)
            else:
                raise ValueError(f"Unexpected metadata format: {entry}")
        return labels

    def get_image_names(self):
        image_names = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_names.append(os.path.join(root, file))
        return sorted(image_names)

    def __len__(self):
        return len(self.time_signals)

    def __getitem__(self, idx):
        if idx >= len(self.time_signals) or idx >= len(self.image_names):
            raise IndexError("Index out of range")

        time_signal = self.time_signals[idx]
        freq_signal = self.freq_signals[idx]
        img_name = self.image_names[idx]
        try:
            image = Image.open(img_name).convert('RGB')
        except IOError as e:
            logger.error(f"Error opening image {img_name}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            vgg_features = vgg16_feature_extractor(image).cpu()

        time_signal = torch.tensor(time_signal, dtype=torch.float32).unsqueeze(0)
        freq_signal = torch.tensor(freq_signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return time_signal, freq_signal, vgg_features.squeeze(0), label




# Early Stopping class to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=30, delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.checkpoint_path)
            self.val_loss_min = val_loss

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=20, checkpoint_path='checkpoint.pt'):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for time_inputs, freq_inputs, vgg_inputs, labels in train_loader:
            time_inputs, freq_inputs, vgg_inputs, labels = time_inputs.to(device), freq_inputs.to(device), vgg_inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(time_inputs, freq_inputs, vgg_inputs)
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

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for time_inputs, freq_inputs, vgg_inputs, labels in val_loader:
                time_inputs, freq_inputs, vgg_inputs, labels = time_inputs.to(device), freq_inputs.to(device), vgg_inputs.to(device), labels.to(device)
                outputs = model(time_inputs, freq_inputs, vgg_inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * time_inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies, epoch+1

# Function to evaluate the model
def evaluate_model(model, test_loader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for time_inputs, freq_inputs, vgg_inputs, labels in test_loader:
            time_inputs, freq_inputs, vgg_inputs = time_inputs.to(device), freq_inputs.to(device), vgg_inputs.to(device)
            outputs = model(time_inputs, freq_inputs, vgg_inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

    return np.array(true_labels), np.array(predictions), np.array(probabilities)

# Function to plot training metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



# Function to plot ROC curve
def plot_roc_curve(true_labels, probabilities):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to visualize signal representations
def visualize_signal_representations(time_signal, freq_signal, sinogram, index):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Time-domain representation
    ax1.imshow(time_signal, aspect='auto', cmap='viridis')
    ax1.set_title('Time-domain Representation')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Time')
    
    # Frequency-domain representation
    ax2.imshow(freq_signal, aspect='auto', cmap='viridis')
    ax2.set_title('Frequency-domain Representation')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Frequency')
    
    # Sinogram
    ax3.imshow(sinogram, aspect='auto', cmap='viridis')
    ax3.set_title('Sinogram')
    ax3.set_xlabel('Angle')
    ax3.set_ylabel('Detector')
    
    plt.tight_layout()
    plt.savefig(f'signal_representation_{index}.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load and process data
    data_path = '/data2/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-three/clean/fd_data_s11_emp.pickle'
    folder_path = '/data2/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-three/clean/figs_gen3_final'
    metadata_path = '/data2/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-three/clean/md_list_s11_emp.pickle'
    checkpoint_path = 'model_checkpoint.pt'

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Process signals
    time_signals, freq_signals = process_signals(data)    #process_signals, return, time_domain_signals, frequency_domain_singal
    print(f"Time domain signals shape: {time_signals.shape}")
    print(f"Frequency domain signals shape: {freq_signals.shape}")

    # Define the feature extractor (VGG16) with the most recent weights
    vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)   
    vgg16_feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:-1]).to(device)   

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataset
    try:
        full_dataset = FusionDataset(time_signals, freq_signals, folder_path, metadata_path, transform)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

    # Print shapes of a sample
    time_sample, freq_sample, vgg_sample, _ = full_dataset[0]
    print(f"Time signal shape: {time_sample.shape}")
    print(f"Frequency signal shape: {freq_sample.shape}")
    print(f"VGG features shape: {vgg_sample.shape}")

    # Visualize a few samples
    for i in range(min(5, len(full_dataset))):  # Visualize the first 5 samples
        time_signal, freq_signal, vgg_features, label = full_dataset[i]
        sinogram = Image.open(full_dataset.image_names[i])
        visualize_signal_representations(time_signal.squeeze().numpy(), freq_signal.squeeze().numpy(), sinogram, i)

    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create model
    model = SeparateDomainModel(time_signals.shape[1:], freq_signals.shape[1:], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies, epochs_run = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=500, patience=30, checkpoint_path=checkpoint_path)

    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluate the model on the test set
    true_labels, predictions, probabilities = evaluate_model(model, test_loader, checkpoint_path)

    # Calculate and print metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, probabilities)
    sensitivity = recall_score(true_labels, predictions)
    specificity = recall_score(true_labels, predictions, pos_label=0)
    mcc = matthews_corrcoef(true_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Plot ROC curve and confusion matrix
    plot_roc_curve(true_labels, probabilities)
    plot_confusion_matrix(true_labels, predictions)

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pt')

    print("Training and evaluation completed. Final model saved as 'final_model.pt'")    
