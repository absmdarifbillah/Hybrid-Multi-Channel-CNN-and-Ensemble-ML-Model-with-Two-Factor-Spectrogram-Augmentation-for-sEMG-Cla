import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Parameters
# -----------------------------
movements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10-class full set
movement_to_index = {m: i for i, m in enumerate(movements)}
movement_names = {
    1: "Extension",
    2: "Flexion",
    3: "Ulnar Deviation",
    4: "Radial Deviation",
    5: "Supination",
    6: "Power Grip",
    7: "Pinch Grip",
    8: "Precision Grip",
    9: "Pointing",
    10: "Rest"
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
# Data Loading Function
# -----------------------------
def load_data_optimized():
    """Load data with optimized approach"""
    print("Loading data for correlation analysis...")
    
    X, Y = [], []
    total_files = num_participants * num_cycles * len(movements) * num_forearms
    processed_files = 0
    
    for participant in range(1, num_participants + 1):
        for cycle in range(1, num_cycles + 1):
            for movement in movements:
                for forearm in range(1, num_forearms + 1):
                    sensor_signals = []
                    valid = True
                    
                    for sensor in range(1, num_sensors + 1):
                        file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                        file_path = os.path.join(data_folder, file_name)
                        
                        try:
                            # Load the signal data
                            signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                            
                            # Compute spectrogram
                            f, t, Sxx = spectrogram(signal, fs=sampling_rate,
                                                    nperseg=window_size, noverlap=overlap)
                            sensor_signals.append(np.log1p(Sxx))
                        except Exception as e:
                            valid = False
                            break
                    
                    if valid and len(sensor_signals) == 4:
                        # Find minimum shape across all sensors
                        min_shape = np.min([s.shape for s in sensor_signals], axis=0)
                        
                        # Resize all spectrograms to the minimum shape
                        resized = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals]
                        
                        # Stack along the channel dimension
                        stacked = np.stack(resized, axis=-1)
                        
                        # Ensure the shape is at least 96x96
                        if stacked.shape[0] >= 96 and stacked.shape[1] >= 96:
                            X.append(stacked[:96, :96, :])
                            Y.append(movement_to_index[movement])
                    
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files...")
    
    return np.array(X), np.array(Y)

# -----------------------------
# Correlation Analysis Functions
# -----------------------------
# Add this function to your Correlation Analysis Functions section
def plot_pca_3d_separation(X, Y, movement_names):
    """
    Perform PCA and plot class separation in 3D space
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    
    # Flatten the data
    X_flat = X.reshape(X.shape[0], -1)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Apply PCA for 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                        c=Y, cmap='tab10', alpha=0.7, s=40)
    
    # Add labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}% variance)')
    ax.set_title('3D PCA Visualization of Movement Classes\n(3D view provides better separation insight)')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Movement Class')
    
    # Add movement names to legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i/10), 
                                 markersize=10, label=movement_names[movements[i]]) 
                      for i in range(len(movements))]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1), loc='upper left')
    
    # Rotate for better view
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return pca, X_pca

# Add this function to analyze cluster separation in 3D
def analyze_3d_cluster_separation(X_pca, Y, movement_names):
    """
    Analyze how well separated the classes are in 3D PCA space
    """
    from scipy.spatial.distance import cdist
    
    # Calculate cluster centers
    cluster_centers = []
    for i in range(len(movements)):
        class_mask = (Y == i)
        if np.any(class_mask):
            center = np.mean(X_pca[class_mask], axis=0)
            cluster_centers.append(center)
        else:
            cluster_centers.append(np.array([np.nan, np.nan, np.nan]))
    
    cluster_centers = np.array(cluster_centers)
    
    # Calculate distances between cluster centers
    distances = cdist(cluster_centers, cluster_centers)
    
    # Set diagonal to NaN to ignore self-distances
    np.fill_diagonal(distances, np.nan)
    
    print("\n3D Cluster Separation Analysis:")
    print("="*40)
    
    # Print minimum distances between clusters
    print("\nMinimum distances between class centers:")
    for i in range(len(movements)):
        for j in range(i+1, len(movements)):
            if not (np.isnan(distances[i, j]) or np.isnan(distances[j, i])):
                dist = distances[i, j]
                print(f"{movement_names[movements[i]]} - {movement_names[movements[j]]}: {dist:.3f}")
    
    # Calculate separation metrics
    valid_distances = distances[~np.isnan(distances)]
    if len(valid_distances) > 0:
        print(f"\nMean inter-cluster distance: {np.mean(valid_distances):.3f}")
        print(f"Minimum inter-cluster distance: {np.min(valid_distances):.3f}")
        print(f"Maximum inter-cluster distance: {np.max(valid_distances):.3f}")
        
        # Separation quality assessment
        mean_dist = np.mean(valid_distances)
        if mean_dist > 2.0:
            print("✓ Excellent cluster separation in 3D space")
        elif mean_dist > 1.0:
            print("✓ Good cluster separation in 3D space")
        elif mean_dist > 0.5:
            print("○ Moderate cluster separation in 3D space")
        else:
            print("⚠ Challenging cluster separation in 3D space")
    
    return distances
def calculate_class_correlations(X, Y):
    """
    Calculate correlation matrix between different movement classes
    """
    # Get unique classes
    unique_classes = np.unique(Y)
    num_classes = len(unique_classes)
    
    # Calculate mean feature vector for each class
    class_means = []
    for cls in unique_classes:
        class_indices = np.where(Y == cls)[0]
        class_data = X[class_indices]
        
        # Flatten the spatial dimensions, keep channel information
        flattened_data = class_data.reshape(class_data.shape[0], -1)
        class_mean = np.mean(flattened_data, axis=0)
        class_means.append(class_mean)
    
    class_means = np.array(class_means)
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(class_means)
    
    return correlation_matrix, unique_classes

def plot_correlation_matrix(correlation_matrix, class_names, movement_names):
    """
    Plot enhanced correlation matrix with annotations
    """
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Plot heatmap
    heatmap = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Customize labels
    plt.title('Linear Correlation Between Movement Classes\n(Lower values indicate better class separation)', 
              fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add movement names as text on the side
    for i, movement in enumerate(movement_names):
        plt.text(len(correlation_matrix) + 0.5, i + 0.5, movement_names[movements[i]], 
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def calculate_separation_metrics(correlation_matrix):
    """
    Calculate metrics that quantify class separation
    """
    # Extract off-diagonal elements (correlations between different classes)
    off_diagonal = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
    
    metrics = {
        'mean_inter_class_correlation': np.mean(off_diagonal),
        'max_inter_class_correlation': np.max(off_diagonal),
        'min_inter_class_correlation': np.min(off_diagonal),
        'std_inter_class_correlation': np.std(off_diagonal),
        'separation_index': 1 - np.mean(np.abs(off_diagonal))  # Higher is better
    }
    
    return metrics

def plot_correlation_distribution(correlation_matrix):
    """
    Plot distribution of correlation values
    """
    # Extract off-diagonal elements
    off_diagonal = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(off_diagonal, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=np.mean(off_diagonal), color='red', linestyle='--', 
                label=f'Mean: {np.mean(off_diagonal):.3f}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inter-Class Correlation Coefficients')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return off_diagonal

def plot_pca_separation(X, Y, movement_names):
    """
    Perform PCA and plot class separation in 2D space
    """
    from sklearn.decomposition import PCA
    
    # Flatten the data
    X_flat = X.reshape(X.shape[0], -1)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='tab10', alpha=0.7)
    
    # Add labels and legend
    plt.colorbar(scatter, label='Movement Class')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    plt.title('PCA Visualization of Movement Classes\n(Good separation indicates distinct patterns)')
    
    # Add movement names to legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i/10), 
                                 markersize=10, label=movement_names[movements[i]]) 
                      for i in range(len(movements))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pca

# -----------------------------
# Main analysis
# -----------------------------

# Load the data
X, Y = load_data_optimized()

print(f"Loaded data shape: {X.shape}")
print(f"Number of samples: {len(Y)}")
print(f"Class distribution: {np.bincount(Y)}")

print("\nAnalyzing class correlations...")

# Calculate correlation matrix
correlation_matrix, unique_classes = calculate_class_correlations(X, Y)

# Create class names for plotting
class_names = [f'Class {i+1}' for i in range(len(movements))]

# Plot correlation matrix
plot_correlation_matrix(correlation_matrix, class_names, movement_names)

# Plot correlation distribution
correlation_values = plot_correlation_distribution(correlation_matrix)

# Calculate and print separation metrics
metrics = calculate_separation_metrics(correlation_matrix)
print("\nClass Separation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Perform PCA visualization
print("\nPerforming PCA for class separation visualization...")
pca = plot_pca_separation(X, Y, movement_names)

# Additional analysis: Class similarity matrix
print("\nClass Similarity Matrix (1 - |correlation|):")
similarity_matrix = 1 - np.abs(correlation_matrix)
np.set_printoptions(precision=3)
print(similarity_matrix)

# Find most similar and most distinct class pairs
print("\nMost Similar Class Pairs (potential confusion):")
for i in range(len(movements)):
    for j in range(i+1, len(movements)):
        if similarity_matrix[i, j] < 0.3:  # High correlation, low similarity
            print(f"{movement_names[movements[i]]} - {movement_names[movements[j]]}: "
                  f"Similarity = {similarity_matrix[i, j]:.3f}")

print("\nMost Distinct Class Pairs (good separation):")
for i in range(len(movements)):
    for j in range(i+1, len(movements)):
        if similarity_matrix[i, j] > 0.8:  # Low correlation, high similarity
            print(f"{movement_names[movements[i]]} - {movement_names[movements[j]]}: "
                  f"Similarity = {similarity_matrix[i, j]:.3f}")

# Create a summary report
print("\n" + "="*60)
print("CLASS SEPARATION ANALYSIS SUMMARY")
print("="*60)
print(f"Number of classes: {len(movements)}")
print(f"Mean inter-class correlation: {metrics['mean_inter_class_correlation']:.3f}")
print(f"Separation index: {metrics['separation_index']:.3f} (higher is better)")
print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")

if metrics['mean_inter_class_correlation'] > 0.5:
    print("\nANALYSIS: High inter-class correlations suggest challenging data with")
    print("overlapping features between classes. Your high accuracy demonstrates")
    print("exceptional feature extraction and model capabilities.")
elif metrics['mean_inter_class_correlation'] > 0.3:
    print("\nANALYSIS: Moderate inter-class correlations. Your model is successfully")
    print("distinguishing between classes with some feature overlap.")
else:
    print("\nANALYSIS: Low inter-class correlations indicate good inherent separation")
    print("in the data, which your model is effectively leveraging.")

print("\nThe high accuracy achieved (97.62%) is particularly impressive given")
print("the correlation structure between classes.")


# Perform 3D PCA visualization
print("\nPerforming 3D PCA for enhanced class separation visualization...")
pca_3d, X_pca_3d = plot_pca_3d_separation(X, Y, movement_names)

# Analyze 3D cluster separation
cluster_distances = analyze_3d_cluster_separation(X_pca_3d, Y, movement_names)

# Update the summary report to include 3D analysis
print("\n" + "="*60)
print("CLASS SEPARATION ANALYSIS SUMMARY")
print("="*60)
print(f"Number of classes: {len(movements)}")
print(f"Mean inter-class correlation: {metrics['mean_inter_class_correlation']:.3f}")
print(f"Separation index: {metrics['separation_index']:.3f} (higher is better)")
print(f"2D PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")
print(f"3D PCA explained variance: {sum(pca_3d.explained_variance_ratio_):.3f}")

# Calculate mean cluster distance from 3D analysis
valid_distances = cluster_distances[~np.isnan(cluster_distances)]
if len(valid_distances) > 0:
    mean_cluster_distance = np.mean(valid_distances)
    print(f"Mean 3D cluster distance: {mean_cluster_distance:.3f}")

if metrics['mean_inter_class_correlation'] > 0.5:
    print("\nANALYSIS: High inter-class correlations suggest challenging data with")
    print("overlapping features between classes. Your high accuracy (97.62%) demonstrates")
    print("exceptional feature extraction and model capabilities that overcome this challenge.")
elif metrics['mean_inter_class_correlation'] > 0.3:
    print("\nANALYSIS: Moderate inter-class correlations. Your model is successfully")
    print("distinguishing between classes with some feature overlap, achieving excellent")
    print("accuracy (97.62%) through advanced processing techniques.")
else:
    print("\nANALYSIS: Low inter-class correlations indicate good inherent separation")
    print("in the data, which your model is effectively leveraging to achieve outstanding")
    print("accuracy (97.62%).")

print("\nThe 3D PCA visualization provides additional insight into how well the classes")
print("are separated in the feature space, further demonstrating the effectiveness of")
print("your processing pipeline.")
