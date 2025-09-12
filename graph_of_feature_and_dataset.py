


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis, skew
import random
import warnings
warnings.filterwarnings('ignore')

# Set global font sizes
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12
})

# -----------------------------
# Parameters
# -----------------------------
movements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
movement_names = {
    1: "Extension",
    2: "Flexion",
    3: "Ulnar Deviation",
    4: "Radial Deviation",
    5: "Hook Grip",
    6: "Power Grip",
    7: "Spherical Grip",
    8: "Precision Grip",
    9: "Lateral Grip",
    10: "Pinch Grip"
}

data_folder = "/Users/arif/Desktop/Project 312/WyoFlex_Dataset/DIGITAL DATA"
num_participants = 28
num_cycles = 3
num_sensors = 4
num_forearms = 2
sampling_rate = 13000
window_size = 256
overlap = 128

# -----------------------------
# Data Augmentation Functions
# -----------------------------
def augment_stft(stft_matrix, method='noise', scale=0.05):
    """Apply data augmentation to STFT matrix"""
    if method == 'noise':
        noise = np.random.normal(0, scale, stft_matrix.shape)
        return stft_matrix + noise
    elif method == 'masking':
        mask = np.random.random(stft_matrix.shape) > 0.1
        return stft_matrix * mask
    elif method == 'time_warp':
        warped = np.roll(stft_matrix, random.randint(-5, 5), axis=1)
        return warped
    return stft_matrix

# -----------------------------
# Advanced Feature Extraction
# -----------------------------
def extract_advanced_features(signal):
    """Extract comprehensive feature set"""
    features = []
    
    # Time domain features
    features.extend([np.mean(signal), np.std(signal), np.median(signal),
                    np.sqrt(np.mean(signal**2)), np.var(signal),
                    kurtosis(signal), skew(signal)])
    
    # Zero crossings and waveform length
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    wl = np.sum(np.abs(np.diff(signal)))
    features.extend([zc, wl])
    
    # Frequency domain features
    try:
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)//4))
        if len(Pxx) > 0 and np.sum(Pxx) > 0:
            mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
            peak_freq = f[np.argmax(Pxx)]
            features.extend([mean_freq, peak_freq])
        else:
            features.extend([0, 0])
    except:
        features.extend([0, 0])
    
    return np.array(features)

# -----------------------------
# Ask for signal type
# -----------------------------
while True:
    b = input("Enter 1 for offset signal or 2 for non-offset signal: ")
    if b in ['1', '2']:
        b = int(b)
        break
    else:
        print("Invalid input, try again.")

# -----------------------------
# Load Sample Data for Visualization
# -----------------------------
print("Loading sample data for visualization...")

# Find available files
available_files = set(os.listdir(data_folder))
print(f"Found {len(available_files)} files in directory")

# Load sample data for each movement from Sensor 1
sensor1_data = {}
sample_signals = {}
all_movement_data = {}

for movement in movements:
    found = False
    for participant in range(1, min(3, num_participants + 1)):
        for cycle in range(1, num_cycles + 1):
            for forearm in range(1, num_forearms + 1):
                file_name = f"P{participant}C{cycle}S1M{movement}F{forearm}O{b}"
                file_path = os.path.join(data_folder, file_name)
                
                if file_name in available_files and os.path.exists(file_path):
                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        sensor1_data[movement] = signal
                        all_movement_data[movement] = signal
                        
                        # Also load all 4 sensors for one sample
                        all_sensors = []
                        for sensor in range(1, num_sensors + 1):
                            sensor_file = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                            sensor_path = os.path.join(data_folder, sensor_file)
                            if sensor_file in available_files and os.path.exists(sensor_path):
                                sensor_signal = np.loadtxt(sensor_path, delimiter=',')[:sampling_rate]
                                all_sensors.append(sensor_signal)
                        
                        if len(all_sensors) == 4:
                            sample_signals[movement] = all_sensors
                        
                        found = True
                        break
                    except:
                        continue
            if found:
                break
        if found:
            break

print(f"Loaded data for {len(sensor1_data)} movements")

# -----------------------------
# Visualization 1: Sensor 1 Data for All Movements (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 1: Sensor 1 Data for All Movements")
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Sensor 1 Data for All Movements', fontsize=14, fontweight='bold', y=0.98)

axes = axes.flatten()
for i, movement in enumerate(movements):
    if movement in sensor1_data:
        ax = axes[i]
        signal = sensor1_data[movement][:1000]
        ax.plot(signal, linewidth=0.8)
        
        # Remove individual titles from subplots and use movement labels
        ax.text(0.5, 0.95, f'M{movement}: {movement_names[movement]}', 
                transform=ax.transAxes, fontsize=9, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        ax.set_xlabel('Samples', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

# Increase space between subplots
plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.95)
plt.show()

# -----------------------------
# Visualization 2: 10 Movement Random Sensor Spectrograms (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 2: 10 Movement Random Sensor Spectrograms")
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Spectrograms for All Movements', fontsize=14, fontweight='bold', y=0.98)

axes = axes.flatten()
for i, movement in enumerate(movements):
    if movement in sensor1_data:
        ax = axes[i]
        signal = sensor1_data[movement][:2000]
        
        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
        
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('Intensity (dB)', fontsize=8)
        
        # Remove individual titles from subplots
        ax.text(0.5, 0.95, f'M{movement}: {movement_names[movement]}', 
                transform=ax.transAxes, fontsize=9, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        ax.set_ylabel('Freq [Hz]', fontsize=8)
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.set_ylim(0, 800)
        ax.tick_params(labelsize=7)

plt.subplots_adjust(hspace=0.6, wspace=0.5, top=0.95)
plt.show()

# -----------------------------
# Visualization 3: 10 Movement Augmented Spectrograms (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 3: 10 Movement Augmented Spectrograms")
augmentation_methods = ['noise', 'masking']

for method in augmentation_methods:
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle(f'Augmented Spectrograms - {method.capitalize()} Method', fontsize=14, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    for i, movement in enumerate(movements):
        if movement in sensor1_data:
            ax = axes[i]
            signal = sensor1_data[movement][:2000]
            
            f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
            stft_original = np.log1p(Sxx)
            augmented_stft = augment_stft(stft_original, method=method)
            
            im = ax.pcolormesh(t, f, augmented_stft, shading='gouraud', cmap='viridis')
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label('Log Intensity', fontsize=8)
            
            # Remove individual titles from subplots
            ax.text(0.5, 0.95, f'M{movement}: {movement_names[movement]}', 
                    transform=ax.transAxes, fontsize=9, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            ax.set_ylabel('Freq [Hz]', fontsize=8)
            ax.set_xlabel('Time [s]', fontsize=8)
            ax.set_ylim(0, 800)
            ax.tick_params(labelsize=7)
    
    plt.subplots_adjust(hspace=0.6, wspace=0.5, top=0.95)
    plt.show()

# -----------------------------
# Visualization 4: 10 Movement PSD (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 4: 10 Movement PSD")
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Power Spectral Density for All Movements', fontsize=14, fontweight='bold', y=0.98)

axes = axes.flatten()
for i, movement in enumerate(movements):
    if movement in sensor1_data:
        ax = axes[i]
        signal = sensor1_data[movement][:2000]
        
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)//4))
        
        ax.semilogy(f, Pxx, linewidth=0.8)
        
        # Remove individual titles from subplots
        ax.text(0.5, 0.95, f'M{movement}: {movement_names[movement]}', 
                transform=ax.transAxes, fontsize=9, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        ax.set_xlabel('Frequency [Hz]', fontsize=8)
        ax.set_ylabel('PSD', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 400)
        ax.tick_params(labelsize=7)

plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.95)
plt.show()

# -----------------------------
# Extract features for all movements
# -----------------------------
movement_features = {}
feature_names = ['Mean', 'Std', 'RMS', 'Kurtosis', 'Skewness', 'ZC', 'WL', 'MeanFreq', 'PeakFreq']

for movement in movements:
    if movement in sensor1_data:
        signal = sensor1_data[movement][:2000]
        features = extract_advanced_features(signal)
        movement_features[movement] = features

# -----------------------------
# Visualization 5: Feature Comparison in Single Graph (FIXED)
# -----------------------------
print("\nCreating Visualization 5: Feature Comparison in Single Graph")

# Select features to display
selected_features = ['RMS', 'ZC', 'WL', 'MeanFreq', 'PeakFreq']
feature_indices = [3, 5, 6, 7, 8]  # Indices of selected features

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Comparison Across All Movements', fontsize=16, fontweight='bold', y=0.98)

axes = axes.flatten()

for i, (feature_name, feature_idx) in enumerate(zip(selected_features, feature_indices)):
    ax = axes[i]
    feature_values = []
    movement_labels = []
    
    for movement in movements:
        if movement in movement_features:
            feature_values.append(movement_features[movement][feature_idx])
            movement_labels.append(f'M{movement}')
    
    bars = ax.bar(movement_labels, feature_values, color=plt.cm.Set3(i/len(selected_features)), alpha=0.8)
    ax.set_title(feature_name, fontsize=12, pad=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# Remove the last empty subplot if needed
if len(selected_features) < len(axes):
    fig.delaxes(axes[len(selected_features)])

plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
plt.show()

# -----------------------------
# Visualization 6: Statistical Features Heatmap (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 6: Statistical Features Heatmap")

stat_features = ['Mean', 'Std', 'RMS', 'Kurtosis', 'Skewness', 'ZC']
heatmap_data = []
movement_labels_short = []

for movement in movements:
    if movement in movement_features:
        heatmap_data.append(movement_features[movement][:6])
        movement_labels_short.append(f'M{movement}')

heatmap_data = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')

cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Feature Value', fontsize=9)

ax.set_title('Statistical Features Heatmap', fontsize=12, pad=15)
ax.set_xlabel('Features', fontsize=10)
ax.set_ylabel('Movements', fontsize=10)

ax.set_xticks(range(len(stat_features)))
ax.set_xticklabels(stat_features, rotation=45, fontsize=8)
ax.set_yticks(range(len(movement_labels_short)))
ax.set_yticklabels(movement_labels_short, fontsize=8)

# Add values with smaller font
for i in range(len(movement_labels_short)):
    for j in range(len(stat_features)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}', ha='center', va='center', 
                      color='white' if heatmap_data[i, j] > np.median(heatmap_data) else 'black',
                      fontsize=6)

plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 7: Frequency Features Comparison (FIXED TITLES)
# -----------------------------
print("\nCreating Visualization 7: Frequency Features Comparison")

freq_data = []
movement_labels_short = []

for movement in movements:
    if movement in movement_features:
        freq_data.append(movement_features[movement][7:9])
        movement_labels_short.append(f'M{movement}')

freq_data = np.array(freq_data)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Mean Frequency
bars1 = ax1.bar(movement_labels_short, freq_data[:, 0], color='lightcoral', alpha=0.7)
ax1.set_title('Mean Frequency Across Movements', fontsize=11, pad=10)
ax1.set_ylabel('Frequency (Hz)', fontsize=9)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Peak Frequency
bars2 = ax2.bar(movement_labels_short, freq_data[:, 1], color='lightgreen', alpha=0.7)
ax2.set_title('Peak Frequency Across Movements', fontsize=11, pad=10)
ax2.set_ylabel('Frequency (Hz)', fontsize=9)
ax2.set_xlabel('Movements', fontsize=10)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 8: Movement Summary Table
# -----------------------------
print("\nCreating Visualization 8: Movement Summary Table")

# Create a summary table of key statistics
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

table_data = []
for movement in movements:
    if movement in movement_features:
        features = movement_features[movement]
        table_data.append([
            f'M{movement}',
            movement_names[movement],
            f'{features[0]:.2f}',  # Mean
            f'{features[1]:.2f}',  # Std
            f'{features[3]:.2f}',  # RMS
            f'{features[5]:.0f}',  # Zero Crossings
            f'{features[7]:.1f}',  # Mean Freq
            f'{features[8]:.1f}'   # Peak Freq
        ])

table = ax.table(cellText=table_data,
                 colLabels=['ID', 'Movement', 'Mean', 'Std', 'RMS', 'ZC', 'Mean Freq', 'Peak Freq'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

ax.set_title('Movement Summary Statistics', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("\nAll visualizations completed successfully with proper title placement!")

freq_features = ['Mean Frequency', 'Peak Frequency']
freq_data = []

for movement in movements:
    if movement in movement_features:
        freq_data.append(movement_features[movement][9:11])  # Last 2 are frequency features

freq_data = np.array(freq_data)

plt.figure(figsize=(12, 8))
x = np.arange(len(movements))
width = 0.35

plt.bar(x - width/2, freq_data[:, 0], width, label='Mean Frequency', alpha=0.8)
plt.bar(x + width/2, freq_data[:, 1], width, label='Peak Frequency', alpha=0.8)

plt.xlabel('Movements', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title('Frequency Domain Features Comparison', fontsize=16, pad=20)
plt.xticks(x, [f'{m}: {movement_names[m][:10]}...' for m in movements if m in movement_features], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 8: Time Domain Features Radar Chart
# -----------------------------
print("\nCreating Visualization 8: Time Domain Features Radar Chart")

# Select 3 representative movements for radar chart
selected_movements = random.sample([m for m in movements if m in movement_features], 3)
time_features = ['Mean', 'Std', 'RMS', 'Kurtosis', 'Skewness', 'Zero Crossings']

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, polar=True)

# Normalize features for radar chart
max_vals = np.max([movement_features[m][:6] for m in selected_movements], axis=0)
min_vals = np.min([movement_features[m][:6] for m in selected_movements], axis=0)

angles = np.linspace(0, 2 * np.pi, len(time_features), endpoint=False).tolist()
angles += angles[:1]  # Close the circle

for i, movement in enumerate(selected_movements):
    values = (movement_features[movement][:6] - min_vals) / (max_vals - min_vals)
    values = np.concatenate((values, [values[0]]))  # Close the circle
    ax.plot(angles, values, linewidth=2, label=f'Movement {movement}: {movement_names[movement]}')
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), time_features)
ax.set_title('Time Domain Features Radar Chart (Normalized)', fontsize=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.show()

print("\nAll visualizations completed successfully!")
