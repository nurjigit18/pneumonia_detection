import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torchvision
import argparse

# Define attention mechanism (same as in training code)
class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.BatchNorm2d(in_features // 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_features // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

# Define model with attention (same as in training code)
class DenseNetWithAttention(nn.Module):
    def __init__(self, num_classes=2, unfreeze_layers=30):
        super(DenseNetWithAttention, self).__init__()
        # Load pre-trained DenseNet
        densenet = torchvision.models.densenet121(weights=None)  # No need to download weights for inference

        # Extract features from DenseNet
        self.features = densenet.features
        
        # Add attention mechanism
        self.attention = AttentionLayer(1024)  # DenseNet121 outputs 1024 channels

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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

# Function to predict single image
def predict_xray(image_path, model_path, show_visualization=True):
    # Check if the image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = DenseNetWithAttention(num_classes=2).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess the image
    try:
        # Define preprocessing transformations (same as validation in training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Open image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs, attention_features = model(input_tensor)
            
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Class names
            class_names = ['NORMAL', 'PNEUMONIA']
            prediction = class_names[predicted.item()]
            
            # Print results
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence.item():.4f}")
            
            # Calculate probabilities for each class
            class_probs = probabilities[0].cpu().numpy()
            for i, class_name in enumerate(class_names):
                print(f"{class_name} probability: {class_probs[i]:.4f}")
            
            # Create visualization
            if show_visualization:
                # Get attention map for visualization
                attention_map = model.attention.attention(attention_features)
                
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                img_np = np.array(image)
                axes[0].imshow(img_np)
                axes[0].set_title(f"Original X-ray")
                axes[0].axis('off')
                
                # Feature activation
                feature = attention_features[0].mean(dim=0).cpu().numpy()
                axes[1].imshow(feature, cmap='viridis')
                axes[1].set_title("Feature Map")
                axes[1].axis('off')
                
                # Attention map
                att_map = attention_map[0, 0].cpu().numpy()
                axes[2].imshow(att_map, cmap='hot')
                axes[2].set_title("Attention Map")
                axes[2].axis('off')
                
                plt.suptitle(f"Prediction: {prediction} (Confidence: {confidence.item():.4f})")
                plt.tight_layout()
                
                # Save visualization
                output_path = os.path.join(os.path.dirname(image_path), 
                                          f"result_{os.path.basename(image_path).split('.')[0]}.png")
                plt.savefig(output_path)
                plt.show()
                print(f"Visualization saved to {output_path}")
            
            return {
                "prediction": prediction,
                "confidence": confidence.item(),
                "probabilities": {class_names[i]: class_probs[i] for i in range(len(class_names))}
            }
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to predict multiple images in a directory
def batch_predict(dir_path, model_path, extension="*.jpg"):
    import glob
    
    # Check if directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
        
    # Get all image files with the specified extension
    image_paths = glob.glob(os.path.join(dir_path, extension))
    
    if not image_paths:
        print(f"No images with extension {extension} found in {dir_path}")
        return
        
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    results = []
    for img_path in image_paths:
        print(f"\nProcessing {img_path}...")
        result = predict_xray(img_path, model_path, show_visualization=False)
        if result:
            result["file"] = os.path.basename(img_path)
            results.append(result)
    
    # Print summary
    print("\nSummary of predictions:")
    print("-----------------------")
    for res in results:
        print(f"File: {res['file']}")
        print(f"Prediction: {res['prediction']}")
        print(f"Confidence: {res['confidence']:.4f}")
        print("-----------------------")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')
    parser.add_argument('--image', type=str, help='Path to a single X-ray image')
    parser.add_argument('--dir', type=str, help='Path to directory containing X-ray images')
    parser.add_argument('--model', type=str, default='best_pneumonia_model.pth', 
                        help='Path to trained model weights')
    parser.add_argument('--extension', type=str, default='*.jpg', 
                        help='File extension for batch processing')
    
    args = parser.parse_args()
    
    if args.image:
        predict_xray(args.image, args.model)
    elif args.dir:
        batch_predict(args.dir, args.model, args.extension)
    else:
        print("Please provide either an image path with --image or a directory path with --dir")