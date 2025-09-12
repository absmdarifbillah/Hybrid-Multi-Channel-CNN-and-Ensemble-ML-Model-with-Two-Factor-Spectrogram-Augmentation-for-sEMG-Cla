

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis, skew
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import random
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Parameters
# -----------------------------
# Corrected movement definitions based on your description
movements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10-class full set
movement_to_index = {m: i for i, m in enumerate(movements)}
movement_names = {
    1: "Extension",
    2: "Flexion",
    3: "Ulnar Deviation",
    4: "Radial Deviation",
    5: "Hook Grip",        # Changed from Supination
    6: "Power Grip",
    7: "Spherical Grip",   # Changed from Pinch Grip
    8: "Precision Grip",
    9: "Lateral Grip",     # Changed from Pointing
    10: "Pinch Grip"       # Changed from Rest
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
        # Randomly mask some frequencies
        mask = np.random.random(stft_matrix.shape) > 0.1  # Mask 10% of values
        return stft_matrix * mask
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
# OPTIMIZED Data Loading
# -----------------------------
X, Y = [], []

print("Loading data... (This may take a few minutes)")

# First, let's check what files actually exist to avoid unnecessary attempts
available_files = set(os.listdir(data_folder))
print(f"Found {len(available_files)} files in directory")

# Count files per movement to understand what's available
movement_counts = {m: 0 for m in movements}
for movement in movements:
    for participant in range(1, num_participants + 1):
        for cycle in range(1, num_cycles + 1):
            for forearm in range(1, num_forearms + 1):
                file_exists = True
                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    if file_name not in available_files:
                        file_exists = False
                        break
                if file_exists:
                    movement_counts[movement] += 1

print("Available movements and their file counts:")
for movement, count in movement_counts.items():
    if count > 0:
        print(f"Movement {movement} ({movement_names[movement]}): {count} files")

# Now load only the movements that actually have data
valid_movements = [m for m in movements if movement_counts[m] > 0]
print(f"\nLoading data for movements: {valid_movements}")

for movement in valid_movements:
    print(f"Loading movement {movement} ({movement_names[movement]})...")
    for participant in range(1, num_participants + 1):
        for cycle in range(1, num_cycles + 1):
            for forearm in range(1, num_forearms + 1):
                sensor_signals = []
                all_files_exist = True
                
                # Check if all sensor files exist first
                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    file_path = os.path.join(data_folder, file_name)
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        break
                
                if not all_files_exist:
                    continue
                
                # If all files exist, load them
                for sensor in range(1, num_sensors + 1):
                    file_name = f"P{participant}C{cycle}S{sensor}M{movement}F{forearm}O{b}"
                    file_path = os.path.join(data_folder, file_name)
                    try:
                        signal = np.loadtxt(file_path, delimiter=',')[:sampling_rate]
                        f, t, Sxx = spectrogram(signal, fs=sampling_rate,
                                                nperseg=window_size, noverlap=overlap)
                        sensor_signals.append(np.log1p(Sxx))
                    except Exception as e:
                        all_files_exist = False
                        break
                
                if all_files_exist and len(sensor_signals) == 4:
                    min_shape = np.min([s.shape for s in sensor_signals], axis=0)
                    resized = [s[:min_shape[0], :min_shape[1]] for s in sensor_signals]
                    stacked = np.stack(resized, axis=-1)
                    if stacked.shape[0] >= 96 and stacked.shape[1] >= 96:
                        X.append(stacked[:96, :96, :])
                        Y.append(movement_to_index[movement])

X = np.array(X)
Y = np.array(Y)

print(f"Final data shape: {X.shape}")
print(f"Class distribution: {np.bincount(Y)}")

# If we don't have all 10 classes, update the movements list
if len(np.unique(Y)) < len(movements):
    actual_movements = sorted(np.unique(Y))
    movements = [movements[i] for i in actual_movements]
    print(f"Updated movements to: {movements}")

# Normalize per channel
for i in range(4):
    channel_mean = X[:, :, :, i].mean()
    channel_std = X[:, :, :, i].std()
    X[:, :, :, i] = (X[:, :, :, i] - channel_mean) / channel_std

# One-hot encode labels
Y_cat = tf.keras.utils.to_categorical(Y, num_classes=len(movements))

# -----------------------------
# Data Augmentation
# -----------------------------
def augment_dataset(X, Y, augment_factor=2):
    """Augment the dataset"""
    X_augmented, Y_augmented = [], []
    
    for i in range(len(X)):
        # Add original sample
        X_augmented.append(X[i])
        Y_augmented.append(Y[i])
        
        # Add augmented samples
        for j in range(augment_factor):
            augmented_sample = X[i].copy()
            
            # Apply different augmentations to each channel
            for channel in range(4):
                stft_channel = augmented_sample[:, :, channel]
                
                # Apply augmentation
                augmentation_type = random.choice(['noise', 'masking'])
                augmented_stft = augment_stft(stft_channel, method=augmentation_type)
                
                augmented_sample[:, :, channel] = augmented_stft
            
            X_augmented.append(augmented_sample)
            Y_augmented.append(Y[i])
    
    return np.array(X_augmented), np.array(Y_augmented)

# Augment the dataset
print("Augmenting data...")
X_aug, Y_aug = augment_dataset(X, Y, augment_factor=2)
Y_aug_cat = tf.keras.utils.to_categorical(Y_aug, num_classes=len(movements))

print(f"Augmented data shape: {X_aug.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, Y_aug_cat, test_size=0.2, stratify=Y_aug, random_state=42
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y),
    y=Y
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# -----------------------------
# Extract Time Domain Features for ML Models
# -----------------------------
print("\nExtracting time domain features...")

def extract_features_for_ml(X_data):
    """Extract features from all 4 channels for ML models"""
    X_ml_features = []
    for i in range(len(X_data)):
        sample_features = []
        for channel in range(4):
            channel_data = X_data[i, :, :, channel].flatten()
            features = extract_advanced_features(channel_data)
            sample_features.extend(features)
        X_ml_features.append(sample_features)
    return np.array(X_ml_features)

# Extract features for training and testing
X_ml_train_features = extract_features_for_ml(X_train)
X_ml_test_features = extract_features_for_ml(X_test)

y_ml_train = np.argmax(y_train, axis=1)
y_ml_test = np.argmax(y_test, axis=1)

# Scale features
scaler = StandardScaler()
X_ml_train_scaled = scaler.fit_transform(X_ml_train_features)
X_ml_test_scaled = scaler.transform(X_ml_test_features)

# -----------------------------
# Train Individual Classifiers
# -----------------------------
print("\nTraining individual classifiers...")

# Initialize classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
gb_clf = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=8)
svm_clf = SVC(probability=True, random_state=42, kernel='rbf', C=5, gamma='scale')
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                           random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Train individual classifiers
classifiers = {
    'Random Forest': rf_clf,
    'Gradient Boosting': gb_clf,
    'SVM': svm_clf,
    'XGBoost': xgb_clf
}

individual_results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_ml_train_scaled, y_ml_train)
    accuracy = clf.score(X_ml_test_scaled, y_ml_test)
    individual_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Ensemble Combinations
# -----------------------------
print("\n" + "="*50)
print("ENSEMBLE COMBINATIONS RESULTS")
print("="*50)

# Define ensemble combinations
ensemble_combinations = {
    'RF + SVM': [('rf', rf_clf), ('svm', svm_clf)],
    'RF + XGBoost': [('rf', rf_clf), ('xgb', xgb_clf)],
    'RF + GradientBoosting': [('rf', rf_clf), ('gb', gb_clf)],
    'SVM + XGBoost': [('svm', svm_clf), ('xgb', xgb_clf)],
    'GradientBoosting + XGBoost': [('gb', gb_clf), ('xgb', xgb_clf)],
    'RF + SVM + XGBoost': [('rf', rf_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
    'RF + GradientBoosting + XGBoost': [('rf', rf_clf), ('gb', gb_clf), ('xgb', xgb_clf)],
    'SVM + GradientBoosting + XGBoost': [('svm', svm_clf), ('gb', gb_clf), ('xgb', xgb_clf)],
    'RF + SVM + GradientBoosting': [('rf', rf_clf), ('svm', svm_clf), ('gb', gb_clf)],
    'RF + SVM + GradientBoosting + XGBoost': [('rf', rf_clf), ('svm', svm_clf), ('gb', gb_clf), ('xgb', xgb_clf)]
}

ensemble_results = {}

# Test each ensemble combination
for ensemble_name, estimators in ensemble_combinations.items():
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_ml_train_scaled, y_ml_train)
    accuracy = ensemble.score(X_ml_test_scaled, y_ml_test)
    ensemble_results[ensemble_name] = accuracy
    print(f"{ensemble_name}: {accuracy * 100:.2f}%")

# -----------------------------
# Print All Results
# -----------------------------
print("\n" + "="*50)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*50)

print("\nINDIVIDUAL CLASSIFIER RESULTS:")
for name, accuracy in individual_results.items():
    print(f"{name}: {accuracy * 100:.2f}%")

print("\nENSEMBLE COMBINATION RESULTS:")
for name, accuracy in ensemble_results.items():
    print(f"{name}: {accuracy * 100:.2f}%")

# Find best individual and ensemble
best_individual = max(individual_results.items(), key=lambda x: x[1])
best_ensemble = max(ensemble_results.items(), key=lambda x: x[1])

print(f"\nBEST INDIVIDUAL CLASSIFIER: {best_individual[0]} - {best_individual[1] * 100:.2f}%")
print(f"BEST ENSEMBLE: {best_ensemble[0]} - {best_ensemble[1] * 100:.2f}%")

# -----------------------------
# Final Evaluation with Best Ensemble
# -----------------------------
print("\n" + "="*50)
print("FINAL EVALUATION WITH BEST ENSEMBLE")
print("="*50)

# Create best ensemble (RF + SVM + GradientBoosting + XGBoost)
best_ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('svm', svm_clf), ('gb', gb_clf), ('xgb', xgb_clf)],
    voting='soft'
)
best_ensemble_clf.fit(X_ml_train_scaled, y_ml_train)

# Final predictions
final_pred = best_ensemble_clf.predict(X_ml_test_scaled)
final_accuracy = np.mean(final_pred == y_ml_test)
print(f"Best Ensemble Accuracy: {final_accuracy * 100:.2f}%")

# -----------------------------
# Detailed Results
# -----------------------------
labels = [movement_names[m] for m in movements]

# Confusion Matrix
cm = confusion_matrix(y_ml_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title('Best Ensemble Model Confusion Matrix')
plt.grid(False)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_ml_test, final_pred, target_names=labels))

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, movement in enumerate(movements):
    class_mask = (y_ml_test == i)
    if np.sum(class_mask) > 0:
        class_acc = np.mean(final_pred[class_mask] == y_ml_test[class_mask])
        print(f"{movement_names[movement]}: {class_acc * 100:.2f}%")

# -----------------------------
# Plot Results Comparison
# -----------------------------
plt.figure(figsize=(16, 8))

# Individual classifiers comparison
plt.subplot(1, 2, 1)
names = list(individual_results.keys())
accuracies = [individual_results[name] * 100 for name in names]
plt.bar(names, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Individual Classifier Accuracies')
plt.xlabel('Classifier')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Top ensemble combinations comparison
plt.subplot(1, 2, 2)
# Get top 5 ensembles
top_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True)[:5]
ensemble_names = [name for name, _ in top_ensembles]
ensemble_accuracies = [acc * 100 for _, acc in top_ensembles]
plt.bar(ensemble_names, ensemble_accuracies, color='purple')
plt.title('Top 5 Ensemble Combinations')
plt.xlabel('Ensemble Combination')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
