import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for image saving
import matplotlib.pyplot as plt

# Input and output directories
input_dir = "SuperInEar"
output_img_dir = "dataset5/images"
output_csv_dir = "dataset5/csvs"

# Sound signature classes (U-shape omitted as per instructions)
classes = ["Neutral", "Bright", "V-shape", "Warm"]

# Create output subdirectories for each class
for cls in classes:
    os.makedirs(os.path.join(output_img_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(output_csv_dir, cls), exist_ok=True)

# Define frequency bands (Hz) for analysis
bands = {
    "sub_bass":   (20, 60),
    "mid_bass":   (60, 250),
    "low_mids":   (250, 500),
    "mids":       (500, 2000),
    "upper_mids": (2000, 4000),
    "presence":   (4000, 6000),
    "brilliance": (6000, 12000)
}

# Threshold in dB for significant emphasis
threshold = 3.0

# Process each CSV file in the input directory
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".csv"):
        continue  # skip non-CSV files
    filepath = os.path.join(input_dir, filename)
    # Load frequency and SPL data
    df = pd.read_csv(filepath)
    if df.shape[1] < 2:
        continue  # skip if file doesn't have at least two columns
    freqs = df[df.columns[0]].to_numpy()
    spl = df[df.columns[1]].to_numpy()
    # Normalize the frequency response (subtract the mean SPL)
    spl_norm = spl - np.mean(spl)
    # Compute average SPL in each band
    band_means = {}
    for band, (low, high) in bands.items():
        # mask for frequencies in [low, high)
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_means[band] = np.mean(spl_norm[mask])
        else:
            band_means[band] = None

    # Define combined region averages for bass, mid, treble
    # Bass = average of sub_bass and mid_bass
    if band_means["sub_bass"] is not None and band_means["mid_bass"] is not None:
        bass_avg = (band_means["sub_bass"] + band_means["mid_bass"]) / 2.0
    else:
        # if one of sub_bass or mid_bass is missing, use the one present (or None)
        bass_avg = band_means["sub_bass"] if band_means["sub_bass"] is not None else band_means["mid_bass"]

    # Mid = average of low_mids, mids, and upper_mids
    mid_values = [band_means["low_mids"], band_means["mids"], band_means["upper_mids"]]
    mid_values = [m for m in mid_values if m is not None]
    mid_avg = np.mean(mid_values) if mid_values else None

    # Treble = average of presence and brilliance
    if band_means["presence"] is not None and band_means["brilliance"] is not None:
        treble_avg = (band_means["presence"] + band_means["brilliance"]) / 2.0
    else:
        treble_avg = band_means["presence"] if band_means["presence"] is not None else band_means["brilliance"]

    # Default to Neutral if any region is undefined (in case of incomplete data)
    label = "Neutral"
    if bass_avg is not None and mid_avg is not None and treble_avg is not None:
        # Calculate differences between bass/treble and midrange
        bass_diff = bass_avg - mid_avg
        treble_diff = treble_avg - mid_avg
        # Apply classification rules based on threshold
        if bass_diff >= threshold and treble_diff >= threshold:
            label = "V-shape"
        elif treble_diff >= threshold and bass_diff < threshold:
            label = "Bright"
        elif bass_diff >= threshold and treble_diff < threshold:
            label = "Warm"
        else:
            label = "Neutral"

    # Plot the frequency response curve
    plt.figure(figsize=(8, 5))  # consistent image size
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlim(20, 20000)    # frequency range 20 Hz – 20 kHz
    ax.set_ylim(-30, 30)      # fixed dB range for consistency
    # Plot the normalized FR curve
    ax.plot(freqs, spl_norm, color='blue', linewidth=2)
    # Clean up the plot aesthetics (no ticks, labels, or grid for a clean image)
    ax.tick_params(bottom=False, left=False, top=False, right=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')   # keep a thin black border for frame
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.grid(False)
    # Save plot to the class-specific folder
    base_name = os.path.splitext(filename)[0]
    img_path = os.path.join(output_img_dir, label, base_name + ".png")
    plt.savefig(img_path, dpi=100)  # save image (PNG by default), 800x600 pixels
    plt.close()

    # Copy the CSV file to the class-specific CSV folder
    csv_dest = os.path.join(output_csv_dir, label, filename)
    shutil.copy2(filepath, csv_dest)
