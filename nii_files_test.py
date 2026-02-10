import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

# Paths
BRAIN_SCAN_PATH = "data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
SEGMENTATION_MASK_PATH = "data/output/BRATS_001.nii.gz"
OUTPUT_IMAGE_DIR = "data/output"


def visualize_nii_fixed(input_path):
    """
    Fixed visualization function that handles both 3D and 4D NIfTI files
    Works for both abdominal CT and BRATS data
    """
    nii_img = nib.load(input_path)
    image_data = nii_img.get_fdata()

    print(f"Image Shape: {image_data.shape}")
    print(f"Data range: [{np.min(image_data):.2f}, {np.max(image_data):.2f}]")

    if len(image_data.shape) == 4:
        print(f"4D image detected with {image_data.shape[-1]} modalities")

        n_modalities = image_data.shape[-1]
        middle_slice_idx = image_data.shape[2] // 2
        fig, axes = plt.subplots(1, n_modalities, figsize=(5 * n_modalities, 5))
        if n_modalities == 1:
            axes = [axes]
        modality_names = ['FLAIR', 'T1w', 'T1Gd', 'T2w']

        for i in range(n_modalities):
            single_modality = image_data[..., i]
            slice_data = single_modality[:, :, middle_slice_idx]

            if np.max(slice_data) > np.min(slice_data):
                slice_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
            else:
                slice_normalized = slice_data

            axes[i].imshow(slice_normalized, cmap='gray')
            axes[i].set_title(modality_names[i] if i < len(modality_names) else f'Modality {i}')
            axes[i].axis('off')

        plt.suptitle(f'Slice {middle_slice_idx} - All Modalities')
        plt.tight_layout()
        plt.show()

    else:
        middle_slice_idx = image_data.shape[2] // 2

        if np.max(image_data) > np.min(image_data):
            normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        else:
            normalized_data = image_data

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Original Intensity - Slice {middle_slice_idx}")
        plt.imshow(image_data[:, :, middle_slice_idx], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Normalized - Slice {middle_slice_idx}")
        plt.imshow(normalized_data[:, :, middle_slice_idx], cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def visualize_overlay(brain_scan_path, mask_path, output_dir=None, modality_idx=0):
    """
    Overlay the segmentation mask on top of the brain scan.

    Args:
        brain_scan_path: Path to original 4D brain scan (.nii.gz)
        mask_path: Path to segmentation mask (.nii.gz)
        output_dir: Directory to save the output image (optional)
        modality_idx: Which MRI modality to use as background
                      0=FLAIR, 1=T1w, 2=T1Gd, 3=T2w
    """
    modality_names = ['FLAIR', 'T1w', 'T1Gd', 'T2w']

    # Step 1: Load original brain scan and segmentation mask
    print("📁 Loading brain scan and segmentation mask...")
    brain_data = nib.load(brain_scan_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    print(f"   Brain scan shape: {brain_data.shape}")
    print(f"   Mask shape:       {mask_data.shape}")
    print(f"   Mask labels:      {np.unique(mask_data)}")

    # Step 2: Find the best slice (slice with the most tumor voxels)
    tumor_counts_per_slice = np.sum(mask_data > 0, axis=(0, 1))  # sum over H, W for each depth slice
    best_slice = np.argmax(tumor_counts_per_slice)
    print(f"\n🎯 Best slice (most tumor): {best_slice} ({int(tumor_counts_per_slice[best_slice])} tumor voxels)")

    # Step 3: Extract background (one modality) and normalize to [0, 1]
    if len(brain_data.shape) == 4:
        background = brain_data[:, :, best_slice, modality_idx]
    else:
        background = brain_data[:, :, best_slice]

    # Normalize background for display
    bg_min, bg_max = np.min(background), np.max(background)
    if bg_max > bg_min:
        background = (background - bg_min) / (bg_max - bg_min)

    # Step 4: Extract the corresponding mask slice
    output_slice = mask_data[:, :, best_slice]

    # Step 5: Mask zeros (background) to make them transparent
    masked_tumor = np.ma.masked_where(output_slice < 0.5, output_slice)

    # Step 6: Plot brain in gray + mask on top in color
    print(f"\n🖼️  Plotting overlay using {modality_names[modality_idx]} as background...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Brain scan only
    axes[0].imshow(background, cmap='gray')
    axes[0].set_title(f"Brain Scan ({modality_names[modality_idx]}) - Slice {best_slice}", fontsize=12)
    axes[0].axis('off')

    # Panel 2: Segmentation mask only
    axes[1].imshow(output_slice, cmap='autumn')
    axes[1].set_title(f"Segmentation Mask - Slice {best_slice}", fontsize=12)
    axes[1].axis('off')

    # Panel 3: Overlay (the main goal)
    axes[2].imshow(background, cmap='gray')
    axes[2].imshow(masked_tumor, cmap='autumn', alpha=0.6)
    axes[2].set_title(f"Tumor Overlay - Slice {best_slice}", fontsize=12)
    axes[2].axis('off')

    plt.suptitle("Brain Tumor Segmentation - SwinUNETR", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Step 7: Save the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "segmentation_overlay.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved to: {save_path}")

    plt.show()
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    # visualize_overlay(
    #     brain_scan_path=BRAIN_SCAN_PATH,
    #     mask_path=SEGMENTATION_MASK_PATH,
    #     output_dir=OUTPUT_IMAGE_DIR,
    #     modality_idx=0,  # FLAIR (change to 2 for T1Gd)
    # )
    visualize_nii_fixed("data/output/BRATS_001.nii.gz")