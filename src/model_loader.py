import os
import torch
from monai.networks.nets import SwinUNETR

# Path to the pretrained model weights
PRETRAINED_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models")
BRAIN_MODEL_WEIGHTS = os.path.join(PRETRAINED_MODEL_DIR, "fold1_f48_ep300_4gpu_dice0_9059", "model.pt")

def load_model():
    model = SwinUNETR()

def get_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_type} Using device: {device}")
    
    if model_type == 'brain':
        model = SwinUNETR(
            # img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True
        )
        weights_path = BRAIN_MODEL_WEIGHTS
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        # Handle checkpoint files that contain 'state_dict' key
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("   ✓ Detected checkpoint format, extracting state_dict...")
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights file not found at {weights_path}, using random initialization")

    model.to(device)
    return model, device

if __name__ == "__main__":
    try:
        model, device = get_model('brain')
        print(f"Model loaded successfully on device: {device}")
    
    except Exception as e:
        print(f"Error loading model: {e}")
    