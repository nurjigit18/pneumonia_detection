import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from tqdm import tqdm
from skimage.transform import resize
from calflops import calculate_flops
import copy
import os
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Define attention mechanism
class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.BatchNorm2d(in_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # Added dropout to attention mechanism
            nn.Conv2d(in_features // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

# Define model with attention
class DenseNetWithAttention(nn.Module):
    def __init__(self, num_classes=2, unfreeze_layers=30):
        super(DenseNetWithAttention, self).__init__()
        # Load pre-trained DenseNet
        densenet = torchvision.models.densenet121(weights="IMAGENET1K_V1")

        # Extract features from DenseNet (everything except the classifier)
        self.features = densenet.features

        # Freeze the pre-trained weights except the last specified layers
        ct = 0
        for child in self.features.children():
            ct += 1
            if ct < len(list(self.features.children())) - 1:  # Freeze all except last dense block
                for param in child.parameters():
                    param.requires_grad = False

        # Unfreeze the last few layers for fine-tuning
        parameters = list(self.features.parameters())
        for param in parameters[-unfreeze_layers:]:
            param.requires_grad = True

        # Add attention mechanism
        self.attention = AttentionLayer(1024)  # DenseNet121 outputs 1024 channels

        # Classifier with improved dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Increased dropout rate
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Added an extra layer with BN
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        attention_features = self.attention(features)
        out = self.avgpool(attention_features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out, attention_features

# Enhanced data transformations - Fixed to handle PIL images correctly
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.RandomErasing(p=0.2),  # Apply after ToTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# Calculate class weights to handle imbalance
def get_class_weights(dataset):
    class_counts = [0, 0]  # Assuming binary classification
    for _, label in dataset:
        class_counts[label] += 1
    
    total = sum(class_counts)
    weights = [total / (len(class_counts) * count) for count in class_counts]
    return torch.FloatTensor(weights)

# Create weighted sampler for imbalanced dataset
def create_weighted_sampler(dataset):
    targets = [label for _, label in dataset]
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

# Load data with class balancing
def load_data(data_dir, batch_size=32):
    train_transform, val_transform = get_transforms()

    # Print available folders to debug
    print(f"Looking for data in: {data_dir}")
    if os.path.exists(data_dir):
        print(f"Data directory exists. Contents: {os.listdir(data_dir)}")
    else:
        print(f"Warning: Data directory {data_dir} does not exist!")
    
    # Path checking for train and val folders
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation directory not found: {val_path}")
    
    print(f"Train directory contents: {os.listdir(train_path)}")
    print(f"Validation directory contents: {os.listdir(val_path)}")

    # Assuming data_dir has 'train' and 'val' subfolders
    # Each with 'NORMAL' and 'PNEUMONIA' subfolders
    train_dataset = ImageFolder(
        root=train_path,
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root=val_path,
        transform=val_transform
    )

    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    # Create weighted sampler for training data
    train_sampler = create_weighted_sampler(train_dataset)

    # Calculate class weights for loss function
    class_weights = get_class_weights(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use weighted sampler instead of shuffle
        num_workers=2  # Reduced for better compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2  # Reduced for better compatibility
    )

    return train_loader, val_loader, class_weights

# Training function with early stopping and learning rate scheduling
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs=15, patience=5):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    counter = 0  # For early stopping
    # Initialize history dictionary with all required keys
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [],
        'learning_rate': []  # Add this key to the dictionary initialization
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        # Update scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Manually print if learning rate changed
        if old_lr != current_lr:
            print(f"Learning rate reduced from {old_lr} to {current_lr}")

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), 'best_pneumonia_model.pth')
            print("Saved best model!")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Visualize attention maps
def visualize_attention(model, dataloader, num_images=15):
    model.eval()
    
    # Try to get a batch of images
    try:
        images, labels = next(iter(dataloader))
        images = images[:min(num_images, len(images))].to(device)
        labels = labels[:min(num_images, len(labels))].to(device)
    except Exception as e:
        print(f"Error getting images for visualization: {e}")
        return None

    # Get model outputs including attention features
    try:
        with torch.no_grad():
            outputs, attention_features = model(images)
            
        # Get attention weights
        # We'll use the attention module to get the actual attention maps
        attention_maps = model.attention.attention(attention_features)

        # Get predictions
        _, predictions = torch.max(outputs, 1)
        
        # Class names
        class_names = ['NORMAL', 'PNEUMONIA']

        # Visualize
        fig, axes = plt.subplots(min(num_images, len(images)), 3, figsize=(12, 4*min(num_images, len(images))))
        
        # Handle case with only one image
        if min(num_images, len(images)) == 1:
            axes = [axes]

        for i in range(min(num_images, len(images))):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[i][0].imshow(img)
            axes[i][0].set_title(f"True: {class_names[labels[i]]}, Pred: {class_names[predictions[i]]}")
            axes[i][0].axis('off')

            # Feature activation (mean across channels)
            feature = attention_features[i].mean(dim=0).cpu().numpy()
            axes[i][1].imshow(feature, cmap='viridis')
            axes[i][1].set_title("Feature Map")
            axes[i][1].axis('off')

            # Attention map
            attention = attention_maps[i, 0].cpu().numpy()
            axes[i][2].imshow(attention, cmap='hot')
            axes[i][2].set_title("Attention Map")
            axes[i][2].axis('off')

        plt.tight_layout()
        plt.savefig('attention_visualization.png')
        plt.close()  # Close to prevent display in some environments
        
        return fig
    except Exception as e:
        print(f"Error during attention visualization: {e}")
        return None

# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = [] 
    
    try:
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
        
        if total == 0:
            print("Warning: No test samples processed")
            return 0, None, None, 0, 0
            
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
                # Convert to numpy arrays for sklearn functions
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate mAP for each class
        average_precisions = []
        for i in range(2):  # Assuming 2 classes: NORMAL (0) and PNEUMONIA (1)
            # For binary case, we may want to focus on the positive class (PNEUMONIA)
            if i == 1 or len(np.unique(all_labels)) <= 2:
                class_labels = (all_labels == i).astype(int)
                ap = average_precision_score(class_labels, all_probabilities[:, i])
                average_precisions.append(ap)
                print(f"AP for class {i} ({'PNEUMONIA' if i == 1 else 'NORMAL'}): {ap:.4f}")
        
        # Calculate mean AP
        mAP = np.mean(average_precisions)
        print(f"mAP: {mAP:.4f}")
        
        # Calculate precision-recall curve for PNEUMONIA class (usually the positive class)
        precision, recall, _ = precision_recall_curve(
            (all_labels == 1).astype(int), all_probabilities[:, 1]
        )
        pr_auc = auc(recall, precision)
        print(f"PR AUC for PNEUMONIA class: {pr_auc:.4f}")
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PNEUMONIA (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('precision_recall_curve.png')
        plt.close()
        
        try:
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(all_labels, all_predictions)
            print("Confusion Matrix:")
            print(cm)
            
            # Classification report
            report = classification_report(all_labels, all_predictions, target_names=['NORMAL', 'PNEUMONIA'])
            print("Classification Report:")
            print(report)
            
            return accuracy, cm, report, mAP, pr_auc
        
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            return accuracy, None, None
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0, None, None

def visualize_training_history(history):
    """
    Visualize training metrics history with enhanced styling
    """
    plt.figure(figsize=(16, 10))
    
    # Loss subplot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], 'g-', linewidth=2, label='Training')
    plt.plot(history['val_loss'], 'b-', linewidth=2, label='Validation')
    plt.title('Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Accuracy subplot
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], 'g-', linewidth=2, label='Training')
    plt.plot(history['val_acc'], 'b-', linewidth=2, label='Validation')
    plt.title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Learning rate subplot (if available in history)
    if 'learning_rate' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rate'], 'r-', linewidth=2)
        plt.title('Learning Rate Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
    
    # Training-validation gap (to visualize overfitting)
    plt.subplot(2, 2, 4)
    acc_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    plt.plot(acc_gap, 'r-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.title('Train-Validation Accuracy Gap', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (Train-Val)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_history_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot ROC and Precision-Recall curves
def plot_classification_curves(model, test_loader):
    """
    Plot ROC and Precision-Recall curves for model evaluation
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for positive class
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    
    axes[1].plot(recall, precision, 'r-', linewidth=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('classification_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc, avg_precision


# Enhanced attention visualization with overlays
def visualize_attention_enhanced(model, dataloader, num_images=15, save_dir='attention_maps'):
    """
    Visualize attention maps with heatmap overlays on original images
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:min(num_images, len(images))].to(device)
    labels = labels[:min(num_images, len(labels))].to(device)
    
    # Class names
    class_names = ['NORMAL', 'PNEUMONIA']
    
    with torch.no_grad():
        outputs, attention_features = model(images)
        # Get attention weights
        attention_maps = model.attention.attention(attention_features)
        # Get predictions
        _, predictions = torch.max(outputs, 1)
    
    # Create a custom colormap for attention overlay
    colors = [(1, 1, 1, 0), (1, 0, 0, 0.7)]  # Transparent white to semi-transparent red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    for i in range(min(num_images, len(images))):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Attention map
        attention = attention_maps[i, 0].cpu().numpy()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original\nTrue: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}", 
                         fontsize=12)
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(attention, cmap='hot')
        axes[1].set_title("Attention Map", fontsize=12)
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay attention on original image
        axes[2].imshow(img)
        # Resize attention map to match image dimensions
        resized_attention = resize(attention, (img.shape[0], img.shape[1]), 
                                  anti_aliasing=True)
        axes[2].imshow(resized_attention, cmap=cmap)
        axes[2].set_title("Attention Overlay", fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_visualization_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Function to visualize feature maps from different layers
def visualize_feature_maps(model, image, layer_names, save_dir='feature_maps'):
    """
    Visualize feature maps from different layers of the model
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    # Register hooks to get intermediate activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for specified layers
    hooks = []
    for name in layer_names:
        # This is a simplified example - you'll need to adapt to your model structure
        if name == 'initial':
            hook = model.features.conv0.register_forward_hook(get_activation(name))
        elif name == 'middle':
            hook = model.features.denseblock2.register_forward_hook(get_activation(name))
        elif name == 'final':
            hook = model.features.denseblock4.register_forward_hook(get_activation(name))
        hooks.append(hook)
    
    # Forward pass with the image
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        outputs, _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize activations for each layer
    for name, activation in activations.items():
        # Get the first 16 channels (or fewer if there are less)
        num_channels = min(16, activation.size(1))
        
        # Create a grid to plot feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_channels):
            feature_map = activation[0, i].cpu().numpy()
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(num_channels, 16):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_maps_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Function to visualize gradient-based class activation maps (Grad-CAM)
def visualize_gradcam(model, image, target_class=None, layer_name=None):
    """
    Visualize Grad-CAM for model interpretability
    """
    model.eval()
    
    # If layer_name is not specified, use the final convolutional layer
    if layer_name is None:
        # This needs to be adapted to your specific model architecture
        target_layer = model.features.norm5
    else:
        # This is a simplified example - adapt to your model structure
        if layer_name == 'initial':
            target_layer = model.features.conv0
        elif layer_name == 'middle':
            target_layer = model.features.denseblock2
        elif layer_name == 'final':
            target_layer = model.features.norm5
    
    # Register hooks for gradients and activations
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)
    
    # Register hook
    handle = target_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    model.zero_grad()
    output, _ = model(image)
    
    # If target_class is not specified, use the predicted class
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
    
    # Backward pass for the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)
    
    # Remove hook
    handle.remove()
    
    # Get gradients and activations
    gradients = gradients[0].cpu().data.numpy()[0]  # [C, H, W]
    activations = activations[0].cpu().data.numpy()[0]  # [C, H, W]
    
    # Calculate weights based on global average pooling of gradients
    weights = np.mean(gradients, axis=(1, 2))  # [C]
    
    # Create weighted combination of activation maps
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    # Apply ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize CAM to match image size
    from skimage.transform import resize
    image_np = image[0].cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)
    
    cam = resize(cam, (image_np.shape[0], image_np.shape[1]), anti_aliasing=True)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cam

# Function to visualize model's decision boundaries on dimensionality-reduced data
def visualize_decision_boundary(model, dataloader, n_samples=500):
    """
    Visualize decision boundaries using t-SNE for dimensionality reduction
    """
    from sklearn.manifold import TSNE
    
    model.eval()
    
    # Collect features and labels
    features = []
    labels = []
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            if len(features) >= n_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, attention_features = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Get features from the layer before classification
            batch_features = model.avgpool(attention_features).squeeze().cpu().numpy()
            
            features.extend(batch_features)
            labels.extend(targets.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            
            if len(features) >= n_samples:
                features = features[:n_samples]
                labels = labels[:n_samples]
                predictions = predictions[:n_samples]
                break
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot ground truth
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                         cmap='coolwarm', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Features (Ground Truth)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot predictions
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=predictions, 
                         cmap='coolwarm', alpha=0.8, s=50)
    plt.colorbar(scatter, label='Predicted Class')
    plt.title('t-SNE Visualization of Features (Predictions)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight misclassifications
    misclassified = labels != predictions
    if np.any(misclassified):
        plt.scatter(features_2d[misclassified, 0], features_2d[misclassified, 1], 
                   s=120, facecolors='none', edgecolors='red', linewidths=2,
                   label='Misclassified')
        plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_model_metrics(model, input_shape=(1, 3, 224, 224)):
    """
    Calculate FLOPs, MACs and parameters for the given model using calflops.
    
    Args:
        model: The PyTorch model to analyze
        input_shape: Input tensor shape (batch_size, channels, height, width)
        
    Returns:
        flops: Total number of FLOPs
        macs: Total number of MACs
        params: Total number of parameters
    """
    # Create a wrapper function to handle the dual output of your model
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Only return the first output (logits) for calculation
            output, _ = self.model(x)
            return output
    
    wrapped_model = ModelWrapper(model)
    
    # Calculate metrics
    flops, macs, params = calculate_flops(
        wrapped_model, 
        input_shape=input_shape,
        print_results=True,
        print_detailed=True,
        output_precision=4
    )
    
    return flops, macs, params


# Main execution
def main():
    # Parameters
    data_dir = "chest_xray"  # Update to your path
    batch_size = 16  # Reduced batch size
    num_epochs = 20
    learning_rate = 0.0005  # Reduced learning rate
    weight_decay = 1e-5  # L2 regularization
    
    try:
        # Load data
        train_loader, val_loader, class_weights = load_data(data_dir, batch_size)
        
        # Also load test data
        _, test_transform = get_transforms()
        test_path = os.path.join(data_dir, "test")
        
        if not os.path.exists(test_path):
            print(f"Warning: Test directory not found: {test_path}")
            test_loader = None
        else:
            print(f"Test directory contents: {os.listdir(test_path)}")
            test_dataset = ImageFolder(
                root=test_path,
                transform=test_transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            print(f"Found {len(test_dataset)} test images")
        
        # Create model
        model = DenseNetWithAttention(num_classes=2, unfreeze_layers=30).to(device)
        
        # Get all trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        # Loss function with class weights
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler - Removed verbose parameter
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Train the model with early stopping
        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            num_epochs=num_epochs, patience=7
        )
        
        # Visualize detailed training history with enhanced plots
        visualize_training_history(history)
        
        # If test data is available, evaluate and visualize results
        if test_loader:
            # Evaluate model performance
            accuracy, cm, report, mAP, pr_auc = evaluate_model(model, test_loader)
            
            # Plot ROC and Precision-Recall curves
            plot_classification_curves(model, test_loader)
            
            # Visualize attention maps for model interpretability
            visualize_attention_enhanced(model, test_loader, num_images=15, save_dir='attention_maps')
            
            # Visualize feature maps from different layers
            # Get a sample image from test set
            sample_images, _ = next(iter(test_loader))
            sample_image = sample_images[0].to(device)
            
            # Define layers to visualize
            layer_names = ['initial', 'middle', 'final']
            visualize_feature_maps(model, sample_image, layer_names, save_dir='feature_maps')
            
            # Visualize Grad-CAM for model interpretability
            visualize_gradcam(model, sample_image, target_class=1)  # 1 for PNEUMONIA
            
            # Visualize decision boundaries (t-SNE)
            visualize_decision_boundary(model, test_loader, n_samples=500)
            
        print("All visualizations completed and saved!")
        
        # Evaluate on test set if available
        if test_loader:
            accuracy, cm, report, mAP, pr_auc = evaluate_model(model, test_loader)
            
            if cm is not None:
                # Visualize confusion matrix
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                classes = ['NORMAL', 'PNEUMONIA']
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig('confusion_matrix.png')
                plt.close()  # Close to prevent display in some environments
            
            # Visualize attention maps
            visualize_attention(model, test_loader, num_images=15)
            
            # Calculate metrics
            flops, macs, params = calculate_model_metrics(model)
            
            print(f"\nModel Efficiency Metrics:")
            print(f"FLOPs: {flops}")
            print(f"MACs: {macs}")
            print(f"Parameters: {params}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()