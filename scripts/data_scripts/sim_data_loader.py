import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import tifffile as tif
import numpy as np
from skimage.transform import resize

def normalize(image, min_percentile=0.1, max_percentile=99.9):
    """Normalize the image to the range [0, 1] based on percentiles."""
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    image = np.clip(image, 0, 1)
    return image

def load_sim_data(raw_path, gt_path, num_z_slices=7, num_angles=3, num_phases=5, target_scale=2):
    """
    Loads and preprocesses the raw SIM data and ground truth with memory optimization
    and flexible resolution scaling.

    Args:
        raw_path (str): Path to the raw SIM data TIFF file.
        gt_path (str): Path to the ground truth TIFF file.
        num_z_slices (int): Number of Z slices.
        num_angles (int): Number of illumination angles.
        num_phases (int): Number of phase shifts.
        target_scale (int): Scale factor between input and ground truth resolution (default: 2).
                          If input is 512x512, and target_scale=2, ground truth will be 1024x1024.

    Returns:
        tuple: A tuple containing (widefield_volume, ground_truth_volume).
               - widefield_volume: A (num_z_slices, H, W) numpy array representing
                 the conventional widefield image.
               - ground_truth_volume: A (num_z_slices, H*scale, W*scale) numpy array.
    """
    print(f"\nLoading SIM data from:")
    print(f"Raw data: {raw_path}")
    print(f"Ground truth: {gt_path}")
    try:
        # Load and process raw data in chunks to save memory
        print("\nLoading raw data...")
        with tif.TiffFile(raw_path) as tif_file:
            total_frames = num_z_slices * num_angles * num_phases
            if len(tif_file.pages) != total_frames:
                raise ValueError(f"Raw data has {len(tif_file.pages)} frames, but expected {total_frames}")
            
            # Get dimensions from first page
            first_page = tif_file.pages[0]
            height, width = first_page.shape
            print(f"Raw data dimensions: {height}x{width}, {total_frames} frames")
            
            # Process data in chunks
            chunk_size = num_angles * num_phases  # Process one z-slice worth of data at a time
            widefield_volume = np.zeros((num_z_slices, height, width), dtype=np.float32)
            
            for z in range(num_z_slices):
                print(f"\rProcessing z-slice {z+1}/{num_z_slices}", end="")
                start_idx = z * chunk_size
                end_idx = start_idx + chunk_size
                
                # Load and average one z-slice worth of data
                chunk = np.stack([tif_file.pages[i].asarray() for i in range(start_idx, end_idx)])
                widefield_volume[z] = np.mean(chunk, axis=0)
            print("\nRaw data processing complete")

        # Create a conventional widefield image by averaging over angles and phases for each z-slice
        # Normalize the widefield volume
        print("\nNormalizing widefield data...")
        widefield_volume = normalize(widefield_volume)
        
        # --- Preprocess Ground Truth ---
        print("\nLoading ground truth data...")
        with tif.TiffFile(gt_path) as tif_file:
            if len(tif_file.pages) != num_z_slices:
                raise ValueError(f"Ground truth has {len(tif_file.pages)} z-slices, but expected {num_z_slices}")
            
            # Process ground truth in chunks
            first_page = tif_file.pages[0]
            gt_height, gt_width = first_page.shape
            print(f"Ground truth dimensions: {gt_height}x{gt_width}")
            
            target_size = (height * target_scale, width * target_scale)
            ground_truth_volume = np.zeros((num_z_slices, target_size[0], target_size[1]), dtype=np.float32)
            
            for z in range(num_z_slices):
                print(f"\rProcessing ground truth z-slice {z+1}/{num_z_slices}", end="")
                img = tif_file.pages[z].asarray().astype(np.float32)
                
                # Resize if necessary
                if img.shape != target_size:
                    img = resize(img, target_size, anti_aliasing=True)
                ground_truth_volume[z] = img
            print("\nGround truth processing complete")

        # Normalize the ground truth volume
        print("\nNormalizing ground truth data...")
        ground_truth_volume = normalize(ground_truth_volume)

        # Pad the volumes to have a depth of 32 to be compatible with the U-Net
        target_depth = 32
        depth_padding = target_depth - widefield_volume.shape[0]  # shape is (z, h, w)
        if depth_padding > 0:
            print(f"\nPadding depth dimension from {widefield_volume.shape[0]} to {target_depth}")
            # Pad with reflection to minimize edge artifacts
            pad_width = ((0, depth_padding), (0, 0), (0, 0))
            widefield_volume = np.pad(widefield_volume, pad_width, mode='reflect')
            ground_truth_volume = np.pad(ground_truth_volume, pad_width, mode='reflect')
        
        # Print memory usage
        wf_mem = widefield_volume.nbytes / (1024 * 1024)
        gt_mem = ground_truth_volume.nbytes / (1024 * 1024)
        print(f"\nMemory usage:")
        print(f"Widefield volume: {wf_mem:.2f} MB")
        print(f"Ground truth volume: {gt_mem:.2f} MB")
        print(f"Total: {(wf_mem + gt_mem):.2f} MB")
        
        print(f"\nFinal shapes:")
        print(f"Widefield: {widefield_volume.shape}")
        print(f"Ground truth: {ground_truth_volume.shape}")
        
        return widefield_volume, ground_truth_volume

    except Exception as e:
        print(f"\nAn error occurred while loading data: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None, None

if __name__ == '__main__':
    # Example usage:
    raw_data_path = "C:/Users/jules/Downloads/pi-ddpm-main/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nm.tif"
    ground_truth_path = "C:/Users/jules/Downloads/pi-ddpm-main/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nmRT.tif"

    wf_vol, gt_vol = load_sim_data(raw_data_path, ground_truth_path)

    if wf_vol is not None and gt_vol is not None:
        print("Data loading and preprocessing successful!")
        print(f"Widefield volume shape: {wf_vol.shape}, dtype: {wf_vol.dtype}, range: [{np.min(wf_vol)}, {np.max(wf_vol)}]")
        print(f"Ground truth volume shape: {gt_vol.shape}, dtype: {gt_vol.dtype}, range: [{np.min(gt_vol)}, {np.max(gt_vol)}]")

        # To verify, save the middle slice
        tif.imwrite("widefield_middle_slice.tif", wf_vol[wf_vol.shape[0] // 2])
        tif.imwrite("gt_middle_slice.tif", gt_vol[gt_vol.shape[0] // 2])
        print("Saved middle slices for verification.")

