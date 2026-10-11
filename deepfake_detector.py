import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
import time
import json
import platform
import sys
import subprocess

def check_gpu_compatibility():
    """
    Check GPU compatibility and print detailed information
    """
    print("\n" + "="*50)
    print("GPU COMPATIBILITY CHECK")
    print("="*50)
    
    # System information
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating System: {platform.system()} {platform.version()}")
    
    # Check for NVIDIA GPU
    try:
        if platform.system() == 'Windows':
            # On Windows, try to use the nvidia-smi command
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\nNVIDIA GPU detected via nvidia-smi:")
                lines = result.stdout.split('\n')
                # Print the GPU model name line and the first few usage stats
                for line in lines[:10]:
                    if any(keyword in line for keyword in ['NVIDIA', 'GPU', 'Name', 'Driver Version']):
                        print(line)
            else:
                print("\nNVIDIA GPU not detected or nvidia-smi not available")
    except Exception as e:
        print(f"\nError checking for NVIDIA GPU: {e}")
    
    # PyTorch GPU check
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print("\nPyTorch GPU Information:")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("\nNo GPU available for PyTorch. Training will be on CPU (slow).")
        print("Possible reasons:")
        print("  1. NVIDIA drivers not installed or incompatible")
        print("  2. CUDA version mismatch with PyTorch requirements")
        print("  3. Environment configuration issues")
        print("\nTroubleshooting tips:")
        print("  - Ensure NVIDIA drivers are installed and up to date")
        print("  - Install PyTorch with the correct CUDA version")
        print("  - Check PyTorch's GPU requirements: https://pytorch.org/get-started/locally/")
    
    print("="*50 + "\n")
    return gpu_available

# Check for GPU availability
print("PyTorch version:", torch.__version__)
gpu_available = check_gpu_compatibility()
device = torch.device("cuda" if gpu_available else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
if gpu_available:
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Paths to the video directories
ORIGINAL_DIR = 'C:/Users/Santoshi chotimath/Downloads/projectfiles/deepfake'
MANIPULATED_DIR = 'C:/Users/Santoshi chotimath/Downloads/projectfiles/original'

# Configuration variables
IMG_SIZE = 224  # ResNet50 input size
BATCH_SIZE = 16  # Can increase if using GPU
EPOCHS = 10
LEARNING_RATE = 0.0001
FRAME_SAMPLE_RATE = 10  # Extract 1 frame every 10 frames

# Determine the system platform and handle multiprocessing appropriately
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    print("Running on Windows. Disabling multiprocessing for better compatibility.")
    NUM_WORKERS = 0
else:
    NUM_WORKERS = 4

class DeepfakeDataset(Dataset):
    """Custom Dataset for loading deepfake detection data"""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        print(f"Dataset initialized with {len(dataframe)} samples")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        try:
            img_path = self.dataframe.iloc[idx]['frame_path']
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                # Return a black image as fallback
                image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Failed to load image at {img_path}")
                    image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            label = 1 if self.dataframe.iloc[idx]['label'] == "deepfake" else 0
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a fallback sample
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            return image, torch.tensor(0, dtype=torch.float32)

class DeepfakeDetector(nn.Module):
    """ResNet50-based model for deepfake detection"""
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze base layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Modify the classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Remove Sigmoid activation
        )
    
    def forward(self, x):
        return self.resnet(x)

def unfreeze_layers(model, percentage=0.3):
    """Unfreeze a percentage of the base model layers for fine-tuning"""
    # Get all layers except the final classifier
    base_layers = list(model.resnet.children())[:-1]
    num_layers = len(base_layers)
    num_unfreeze = int(num_layers * percentage)
    
    # Freeze all layers first
    for param in model.resnet.parameters():
        param.requires_grad = False
    
    # Unfreeze the top percentage of layers
    for layer in base_layers[-num_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Always unfreeze the classifier
    for param in model.resnet.fc.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir):
    """Train the model with mixed precision training if GPU is available"""
    scaler = GradScaler() if gpu_available else None
    best_val_loss = float('inf')
    train_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if gpu_available:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()  # Apply sigmoid here
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()  # Apply sigmoid here
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        
        # Save history
        train_history['loss'].append(epoch_loss)
        train_history['accuracy'].append(epoch_acc)
        train_history['val_loss'].append(val_epoch_loss)
        train_history['val_accuracy'].append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f'{results_dir}/best_model.pth')
    
    return train_history

def extract_frames_from_videos(video_paths, label, output_dir='extracted_frames', max_frames_per_video=20):
    """
    Extract frames from videos and save metadata to DataFrame
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_data = []
    
    for video_path in video_paths:
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue
        
        frame_count = 0
        save_count = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        while cap.isOpened() and save_count < max_frames_per_video:
            success, frame = cap.read()
            
            if not success:
                break
            
            if frame_count % FRAME_SAMPLE_RATE == 0:
                # Resize frame to match ResNet50 input size
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                
                # Save frame
                frame_filename = f"{output_dir}/{video_name}_frame_{save_count}.jpg"
                cv2.imwrite(frame_filename, frame)
                
                # Add to DataFrame
                frame_data.append({
                    'frame_path': frame_filename,
                    'label': "deepfake" if label == 1 else "original",  # Use string labels
                    'source_video': video_path
                })
                
                save_count += 1
            
            frame_count += 1
        
        cap.release()
    
    return frame_data

def collect_video_paths():
    """
    Collect paths to all videos in the dataset
    """
    original_videos = []
    for root, _, files in os.walk(ORIGINAL_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                original_videos.append(os.path.join(root, file))
    
    manipulated_videos = []
    for root, _, files in os.walk(MANIPULATED_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                manipulated_videos.append(os.path.join(root, file))
    
    return original_videos, manipulated_videos

def main():
    # Create frames directory if it doesn't exist
    if not os.path.exists('extracted_frames'):
        os.makedirs('extracted_frames')
    
    # Create results directory
    results_dir = 'performance_metrics'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load or create frame metadata
    if not os.path.exists('frame_metadata.csv'):
        # Extract frames (using existing extract_frames_from_videos function)
        original_videos, manipulated_videos = collect_video_paths()
        original_frames = extract_frames_from_videos(original_videos, label=0, 
                                                   output_dir='extracted_frames/original')
        manipulated_frames = extract_frames_from_videos(manipulated_videos, label=1, 
                                                      output_dir='extracted_frames/manipulated')
        
        all_frames = pd.DataFrame(original_frames + manipulated_frames)
        all_frames.to_csv('frame_metadata.csv', index=False)
    else:
        all_frames = pd.read_csv('frame_metadata.csv')
        if not isinstance(all_frames['label'].iloc[0], str):
            all_frames['label'] = all_frames['label'].apply(lambda x: "deepfake" if x == 1 else "original")
            all_frames.to_csv('frame_metadata.csv', index=False)
    
    # Split data
    video_names = all_frames['source_video'].unique()
    train_videos, val_videos = train_test_split(video_names, test_size=0.2, random_state=42)
    
    train_df = all_frames[all_frames['source_video'].isin(train_videos)]
    val_df = all_frames[all_frames['source_video'].isin(val_videos)]
    
    # Create datasets and dataloaders
    train_dataset = DeepfakeDataset(train_df, transform=train_transform)
    val_dataset = DeepfakeDataset(val_df, transform=val_transform)
    
    # Reduce initial batch size for debugging
    effective_batch_size = 8  # Reduced from 16
    if gpu_available:
        effective_batch_size *= 2
    
    print(f"Creating data loaders with batch size {effective_batch_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with additional options for better performance
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=gpu_available,  # Enable pin_memory if GPU is available
        prefetch_factor=2 if NUM_WORKERS > 0 else None,  # Prefetch 2 batches per worker
        persistent_workers=True if NUM_WORKERS > 0 else False,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=gpu_available,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True
    )
    
    print("Data loaders created successfully")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Initialize model
    model = DeepfakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # First phase: train only the classifier
    print("Phase 1: Training only the classifier...")
    print("Starting training loop...")
    
    try:
        start_time = time.time()
        history1 = train_model(model, train_loader, val_loader, criterion, optimizer, 
                             EPOCHS, device, results_dir)
        phase1_time = time.time() - start_time
        print("Phase 1 completed successfully")
    except Exception as e:
        print(f"Error during Phase 1 training: {str(e)}")
        raise
    
    # Second phase: fine-tuning
    print("Phase 2: Fine-tuning with unfrozen layers...")
    model = unfreeze_layers(model, percentage=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE/10)
    
    start_time = time.time()
    history2 = train_model(model, train_loader, val_loader, criterion, optimizer,
                          EPOCHS//2, device, results_dir)
    phase2_time = time.time() - start_time
    
    # Save final model
    torch.save(model.state_dict(), f'{results_dir}/final_model.pth')
    
    # Generate and save performance metrics
    # (Keep existing visualization and metrics code, just update the prediction part)
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate and save metrics using existing functions
    # (Keep the rest of the metrics and visualization code)
    
    # Save training stats
    training_stats = {
        "hardware": {
            "gpu_available": gpu_available,
            "platform": platform.system(),
            "pytorch_version": torch.__version__
        },
        "training_time": {
            "phase1_seconds": phase1_time,
            "phase2_seconds": phase2_time,
            "total_seconds": phase1_time + phase2_time
        },
        "hyperparameters": {
            "batch_size": effective_batch_size,
            "learning_rate": LEARNING_RATE,
            "epochs_phase1": EPOCHS,
            "epochs_phase2": EPOCHS // 2
        },
        "dataset_stats": {
            "total_frames": len(all_frames),
            "training_frames": len(train_df),
            "validation_frames": len(val_df),
            "total_videos": len(video_names),
            "training_videos": len(train_videos),
            "validation_videos": len(val_videos)
        }
    }
    
    with open(f"{results_dir}/training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=4)

if __name__ == "__main__":
    main() 