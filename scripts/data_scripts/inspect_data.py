import tifffile as tif
import numpy as np

def inspect_image(file_path, name):
    try:
        image = tif.imread(file_path)
        print(f"--- {name} ---")
        print(f"File: {file_path}")
        print(f"Shape: {image.shape}")
        print(f"Data Type: {image.dtype}")
        print(f"Min Value: {np.min(image)}")
        print(f"Max Value: {np.max(image)}")
        print("\\n")
    except Exception as e:
        print(f"Error reading {name} image: {e}")

if __name__ == "__main__":
    ground_truth_path = "C:/Users/jules/Downloads/pi-ddpm-main/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nmRT.tif"
    raw_data_path = "C:/Users/jules/Downloads/pi-ddpm-main/pi-ddpm-main/Recon/OMX_LSEC_Actin_525nm.tif"

    inspect_image(ground_truth_path, "Ground Truth")
    inspect_image(raw_data_path, "Raw SIM Data")
