from model_loader import get_model
import torch
import nibabel as nib
import os
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    Activationsd,
    AsDiscreted,
    Invertd,
)
from monai.data import decollate_batch


def get_load_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])


def get_spatial_transforms():
    return Compose([
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
    ])


def get_intensity_transforms():
    return Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


def get_post_transforms(spatial_transforms):
    post_trans = Compose([
        Activationsd(keys=["pred"], sigmoid=True),
        AsDiscreted(keys=["pred"], threshold=0.5),
        Invertd(
            keys=["pred"],
            transform=spatial_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
    ])
    return post_trans


def run_inference(model_type, input_image_path, output_image_path):
    print(f"Starting inference for: {input_image_path}")
    model, device = get_model(model_type)
    model.eval()
    input_data = {"image": input_image_path}
    load_trans = get_load_transforms()
    spatial_trans = get_spatial_transforms()
    intensity_trans = get_intensity_transforms()
    data = load_trans(input_data)
    original_affine = data["image"].meta["original_affine"]
    data = spatial_trans(data)
    data = intensity_trans(data)
    image = data["image"].unsqueeze(0).to(device)
    roi_size = (96, 96, 96)
    sw_batch_size = 2

    print("Running model prediction...")
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=0.5,
        )

    prediction = prediction.squeeze(0)
    from monai.data import MetaTensor
    if isinstance(data["image"], MetaTensor):
        prediction = MetaTensor(prediction, meta=data["image"].meta.copy())
        prediction.applied_operations = []

    data["pred"] = prediction
    post_transforms = get_post_transforms(spatial_trans)
    data = post_transforms(data)
    final_mask = data["pred"]
    output_mask = final_mask.cpu().numpy()
    any_positive = output_mask.max(axis=0) > 0
    argmax_labels = np.argmax(output_mask, axis=0) + 1
    output_mask_combined = (any_positive * argmax_labels).astype(np.uint8)
    unique_labels = np.unique(output_mask_combined)
    for label in unique_labels:
        count = np.sum(output_mask_combined == label)
        percentage = (count / output_mask_combined.size) * 100
        print(f"     - Label {label}: {count:,} voxels ({percentage:.2f}%)")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    output_nifti = nib.Nifti1Image(output_mask_combined, original_affine)
    nib.save(output_nifti, output_image_path)

    print(f"Saved segmentation to: {output_image_path}")
    print(f"Output shape: {output_mask_combined.shape}")
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