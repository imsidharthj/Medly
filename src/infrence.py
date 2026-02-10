from model_loader import get_model
import torch
import nibabel as nib
import os
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImage,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    Orientation,
    Spacing,
    NormalizeIntensity,
    Activations,
    AsDiscrete,
)
from monai.data import decollate_batch


def get_preprocessing_transforms():
    preprocess = Compose([
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
    return preprocess


def get_post_transforms():
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])
    return post_trans


def run_inference(model_type, input_image_path, output_image_path):
    model, device = get_model(model_type)
    model.eval()
    
    loader = LoadImage(image_only=False)
    image, meta = loader(input_image_path)
    
    preprocess = get_preprocessing_transforms()
    image = preprocess(image)
    
    image = image.unsqueeze(0).to(device)
    
    roi_size = (96, 96, 96)
    sw_batch_size = 2
    
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=0.5,
        )
    
    prediction = torch.sigmoid(prediction)
    output_mask = (prediction > 0.5).float()
    
    output_mask = output_mask.squeeze(0).cpu().numpy()
    
    output_mask_combined = np.argmax(output_mask, axis=0).astype(np.uint8)
    
    unique_labels = np.unique(output_mask_combined)
    for label in unique_labels:
        count = np.sum(output_mask_combined == label)
        percentage = (count / output_mask_combined.size) * 100
        print(f"     - Label {label}: {count:,} voxels ({percentage:.2f}%)")
    
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    original_nii = nib.load(input_image_path)
    output_nifti = nib.Nifti1Image(output_mask_combined, original_nii.affine)
    nib.save(output_nifti, output_image_path)
    
    return output_mask_combined


if __name__ == "__main__":
    input_path = '/home/sid/Workspace/python/Medly/data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz'
    output_path = '/home/sid/Workspace/python/Medly/data/output/BRATS_001.nii.gz'
    
    try:
        result = run_inference('brain', input_path, output_path)
        print(f"\n✅ Test completed successfully!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
