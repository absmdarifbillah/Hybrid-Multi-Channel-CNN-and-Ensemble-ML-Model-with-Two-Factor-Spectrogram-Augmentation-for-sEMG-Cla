

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
# Add these lines in the Parameters section
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
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
# Improved CNN Model
# -----------------------------
def create_advanced_cnn_model(input_shape, num_classes):
    """Create improved CNN model"""
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

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
# -----------------------------
# Split dataset into Train, Validation, and Test sets
# -----------------------------
print("\nSplitting dataset into Train, Validation, and Test sets...")

# First split: separate out test set
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_aug, Y_aug, test_size=test_ratio, stratify=Y_aug, random_state=42
)

# Second split: split remaining data into train and validation
val_ratio_adjusted = validation_ratio / (train_ratio + validation_ratio)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=val_ratio_adjusted, stratify=Y_temp, random_state=42
)

# Convert to categorical
y_train = tf.keras.utils.to_categorical(Y_train, num_classes=len(movements))
y_val = tf.keras.utils.to_categorical(Y_val, num_classes=len(movements))
y_test = tf.keras.utils.to_categorical(Y_test, num_classes=len(movements))

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y),
    y=Y
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# -----------------------------
# Create and Train Advanced Model
# -----------------------------
model = create_advanced_cnn_model(
    input_shape=(96, 96, 4), 
    num_classes=len(movements)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
model.summary()

# -----------------------------
# Training with Advanced Callbacks
# -----------------------------
checkpoint = callbacks.ModelCheckpoint(
    'best_advanced_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Train the model
print("Training model...")
# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),  # Changed from X_test to X_val
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

# Load the best model
model.load_weights('best_advanced_model.h5')

# -----------------------------
# Ensemble Learning with ML Models
# -----------------------------
print("\nTraining Ensemble Classifiers...")

# Extract features for traditional ML models from the AUGMENTED test set
X_ml_features = []
for i in range(len(X_test)):
    sample_features = []
    for channel in range(4):
        channel_data = X_test[i, :, :, channel].flatten()
        features = extract_advanced_features(channel_data)
        sample_features.extend(features)
    X_ml_features.append(sample_features)

X_ml_features = np.array(X_ml_features)
y_ml_test = np.argmax(y_test, axis=1)

# Scale features
scaler = StandardScaler()
X_ml_features_scaled = scaler.fit_transform(X_ml_features)

# Train ensemble of classifiers on the TRAINING data
X_ml_train_features = []
for i in range(len(X_train)):
    sample_features = []
    for channel in range(4):
        channel_data = X_train[i, :, :, channel].flatten()
        features = extract_advanced_features(channel_data)
        sample_features.extend(features)
    X_ml_train_features.append(sample_features)

X_ml_train_features = np.array(X_ml_train_features)
y_ml_train = np.argmax(y_train, axis=1)

# Scale training features
X_ml_train_scaled = scaler.transform(X_ml_train_features)

# Train ensemble classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
gb_clf = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=8)
svm_clf = SVC(probability=True, random_state=42, kernel='rbf', C=5, gamma='scale')

# Create ensemble
ensemble_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('gb', gb_clf), ('svm', svm_clf)],
    voting='soft'
)

ensemble_clf.fit(X_ml_train_scaled, y_ml_train)

# -----------------------------
# Final Evaluation
# -----------------------------
# CNN model evaluation
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAdvanced CNN Model Results:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")

# Combined prediction (CNN + Ensemble)
cnn_pred_proba = model.predict(X_test, verbose=0)
ensemble_pred_proba = ensemble_clf.predict_proba(X_ml_features_scaled)

# Weighted average of predictions
final_pred_proba = 0.7 * cnn_pred_proba + 0.3 * ensemble_pred_proba
final_pred = np.argmax(final_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

final_accuracy = np.mean(final_pred == y_true)
print(f"Combined CNN + Ensemble Accuracy: {final_accuracy * 100:.2f}%")

# -----------------------------
# Detailed Results
# -----------------------------
print("\n" + "="*50)
print("DETAILED RESULTS")
print("="*50)

# Confusion Matrix
labels = [movement_names[m] for m in movements]
cm = confusion_matrix(y_true, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title('Final Combined Model Confusion Matrix')
plt.grid(False)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, final_pred, target_names=labels))

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, movement in enumerate(movements):
    class_mask = (y_true == i)
    if np.sum(class_mask) > 0:
        class_acc = np.mean(final_pred[class_mask] == y_true[class_mask])
        print(f"{movement_names[movement]}: {class_acc * 100:.2f}%")

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_random_sample(model, ensemble_clf, scaler):
    """Predict a random sample"""
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test[idx:idx+1]
    true_label = y_true[idx]
    
    # Extract features for ensemble
    sample_features = []
    for channel in range(4):
        channel_data = sample[0, :, :, channel].flatten()
        features = extract_advanced_features(channel_data)
        sample_features.extend(features)
    
    sample_features = np.array(sample_features).reshape(1, -1)
    sample_features_scaled = scaler.transform(sample_features)
    
    # CNN prediction
    cnn_pred = model.predict(sample, verbose=0)
    cnn_label = np.argmax(cnn_pred)
    
    # Ensemble prediction
    ensemble_pred = ensemble_clf.predict_proba(sample_features_scaled)
    ensemble_label = np.argmax(ensemble_pred)
    
    # Combined prediction
    combined_pred = 0.7 * cnn_pred + 0.3 * ensemble_pred
    final_label = np.argmax(combined_pred)
    
    print(f"\nTrue Movement: {movement_names[movements[true_label]]}")
    print(f"CNN Prediction: {movement_names[movements[cnn_label]]} "
          f"(Confidence: {np.max(cnn_pred):.3f})")
    print(f"Ensemble Prediction: {movement_names[movements[ensemble_label]]} "
          f"(Confidence: {np.max(ensemble_pred):.3f})")
    print(f"Final Prediction: {movement_names[movements[final_label]]} "
          f"(Confidence: {np.max(combined_pred):.3f})")

# Run prediction
print("\nRandom Sample Prediction:")
predict_random_sample(model, ensemble_clf, scaler)

# -----------------------------
# ML Models Individual Accuracy
# -----------------------------
# -----------------------------
# ML Models Individual Accuracy
# -----------------------------
print("\n" + "="*50)
print("INDIVIDUAL ML MODEL RESULTS")
print("="*50)

# Train and evaluate individual ML models separately
rf_clf_individual = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
gb_clf_individual = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=8)
svm_clf_individual = SVC(probability=True, random_state=42, kernel='rbf', C=5, gamma='scale')

# Train individual models
rf_clf_individual.fit(X_ml_train_scaled, y_ml_train)
gb_clf_individual.fit(X_ml_train_scaled, y_ml_train)
svm_clf_individual.fit(X_ml_train_scaled, y_ml_train)

# Evaluate individual ML models
rf_acc = rf_clf_individual.score(X_ml_features_scaled, y_ml_test)
gb_acc = gb_clf_individual.score(X_ml_features_scaled, y_ml_test)
svm_acc = svm_clf_individual.score(X_ml_features_scaled, y_ml_test)
ensemble_acc = ensemble_clf.score(X_ml_features_scaled, y_ml_test)

print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
print(f"Gradient Boosting Test Accuracy: {gb_acc * 100:.2f}%")
print(f"SVM Test Accuracy: {svm_acc * 100:.2f}%")
print(f"Ensemble (Voting) Test Accuracy: {ensemble_acc * 100:.2f}%")

# -----------------------------
# Final Summary
# -----------------------------
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"CNN Model Accuracy: {test_acc * 100:.2f}%")
print(f"Ensemble ML Accuracy: {ensemble_acc * 100:.2f}%")
print(f"Combined CNN+Ensemble Accuracy: {final_accuracy * 100:.2f}%")

# Run prediction
print("\nRandom Sample Prediction:")
predict_random_sample(model, ensemble_clf, scaler)
# -----------------------------
# Add XGBoost and Create Comprehensive Ensemble Combinations
# -----------------------------
print("\n" + "="*50)
print("COMPREHENSIVE CNN + ML ENSEMBLE COMBINATIONS")
print("="*50)

# Train XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=6, 
    learning_rate=0.1, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)
xgb_clf.fit(X_ml_train_scaled, y_ml_train)
xgb_acc = xgb_clf.score(X_ml_features_scaled, y_ml_test)
print(f"XGBoost Test Accuracy: {xgb_acc * 100:.2f}%")

# Get CNN predictions
cnn_pred_proba = model.predict(X_test, verbose=0)

# Get individual ML model predictions
rf_pred_proba = rf_clf_individual.predict_proba(X_ml_features_scaled)
gb_pred_proba = gb_clf_individual.predict_proba(X_ml_features_scaled)
svm_pred_proba = svm_clf_individual.predict_proba(X_ml_features_scaled)
xgb_pred_proba = xgb_clf.predict_proba(X_ml_features_scaled)
ensemble_pred_proba = ensemble_clf.predict_proba(X_ml_features_scaled)

# Define ensemble combinations with different weights
ensemble_combinations = {
    'CNN Only': (cnn_pred_proba, 1.0),
    'CNN + RF': (0.7 * cnn_pred_proba + 0.3 * rf_pred_proba, 1.0),
    'CNN + XGBoost': (0.7 * cnn_pred_proba + 0.3 * xgb_pred_proba, 1.0),
    'CNN + GradientBoosting': (0.7 * cnn_pred_proba + 0.3 * gb_pred_proba, 1.0),
    'CNN + SVM': (0.7 * cnn_pred_proba + 0.3 * svm_pred_proba, 1.0),
    'CNN + RF + XGBoost': (0.6 * cnn_pred_proba + 0.2 * rf_pred_proba + 0.2 * xgb_pred_proba, 1.0),
    'CNN + All ML (Equal)': (0.5 * cnn_pred_proba + 0.125 * rf_pred_proba + 0.125 * gb_pred_proba + 
                            0.125 * svm_pred_proba + 0.125 * xgb_pred_proba, 1.0),
    'CNN + All ML (Weighted)': (0.6 * cnn_pred_proba + 0.2 * rf_pred_proba + 0.1 * gb_pred_proba + 
                               0.05 * svm_pred_proba + 0.05 * xgb_pred_proba, 1.0),
    'CNN + Ensemble ML': (0.7 * cnn_pred_proba + 0.3 * ensemble_pred_proba, 1.0),
}

# Evaluate all ensemble combinations
ensemble_results = {}
print("\nEnsemble Combination Results:")
print("-" * 50)

for name, (pred_proba, _) in ensemble_combinations.items():
    pred = np.argmax(pred_proba, axis=1)
    accuracy = np.mean(pred == y_true)
    ensemble_results[name] = accuracy
    print(f"{name}: {accuracy * 100:.2f}%")

# Find best ensemble
best_ensemble = max(ensemble_results.items(), key=lambda x: x[1])
print(f"\nBest Ensemble: {best_ensemble[0]} - {best_ensemble[1] * 100:.2f}%")

# -----------------------------
# Visualize Ensemble Comparison
# -----------------------------
plt.figure(figsize=(14, 8))

# Sort results for better visualization
sorted_results = dict(sorted(ensemble_results.items(), key=lambda item: item[1], reverse=True))
names = list(sorted_results.keys())
accuracies = [sorted_results[name] * 100 for name in names]

colors = ['green' if 'CNN Only' in name else 
          'blue' if 'Best' in name else 
          'orange' for name in names]

bars = plt.bar(range(len(names)), accuracies, color=colors)
plt.xlabel('Ensemble Combination')
plt.ylabel('Accuracy (%)')
plt.title('CNN + ML Ensemble Combination Performance')
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{accuracies[i]:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# -----------------------------
# Detailed Analysis of Best Ensemble
# -----------------------------
print("\n" + "="*50)
print(f"DETAILED ANALYSIS OF BEST ENSEMBLE: {best_ensemble[0]}")
print("="*50)

best_pred_proba, _ = ensemble_combinations[best_ensemble[0]]
best_pred = np.argmax(best_pred_proba, axis=1)

# Confusion Matrix for best ensemble
cm = confusion_matrix(y_true, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_ensemble[0]}')
plt.grid(False)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, best_pred, target_names=labels))

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, movement in enumerate(movements):
    class_mask = (y_true == i)
    if np.sum(class_mask) > 0:
        class_acc = np.mean(best_pred[class_mask] == y_true[class_mask])
        print(f"{movement_names[movement]}: {class_acc * 100:.2f}%")

# -----------------------------
# Feature Importance Analysis (for tree-based models)
# -----------------------------
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importances from Random Forest
feature_importances = rf_clf_individual.feature_importances_
top_features_idx = np.argsort(feature_importances)[-10:]  # Top 10 features
top_features_importance = feature_importances[top_features_idx]

# Create feature names (simplified)
feature_names = [f'Feature_{i}' for i in range(len(feature_importances))]
top_feature_names = [feature_names[i] for i in top_features_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_importance)), top_features_importance, align='center')
plt.yticks(range(len(top_features_importance)), top_feature_names)
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.tight_layout()
plt.show()

# -----------------------------
# Final Prediction with Best Ensemble
# -----------------------------
def predict_with_best_ensemble(model, ml_models, scaler, ensemble_type='CNN + RF'):
    """Predict using the best ensemble combination"""
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test[idx:idx+1]
    true_label = y_true[idx]
    
    # Extract features for ML models
    sample_features = []
    for channel in range(4):
        channel_data = sample[0, :, :, channel].flatten()
        features = extract_advanced_features(channel_data)
        sample_features.extend(features)
    
    sample_features = np.array(sample_features).reshape(1, -1)
    sample_features_scaled = scaler.transform(sample_features)
    
    # Get predictions from all models
    cnn_pred = model.predict(sample, verbose=0)
    rf_pred = ml_models['rf'].predict_proba(sample_features_scaled)
    gb_pred = ml_models['gb'].predict_proba(sample_features_scaled)
    svm_pred = ml_models['svm'].predict_proba(sample_features_scaled)
    xgb_pred = ml_models['xgb'].predict_proba(sample_features_scaled)
    
    # Apply ensemble combination
    if ensemble_type == 'CNN Only':
        final_pred = cnn_pred
    elif ensemble_type == 'CNN + RF':
        final_pred = 0.7 * cnn_pred + 0.3 * rf_pred
    elif ensemble_type == 'CNN + XGBoost':
        final_pred = 0.7 * cnn_pred + 0.3 * xgb_pred
    elif ensemble_type == 'CNN + GradientBoosting':
        final_pred = 0.7 * cnn_pred + 0.3 * gb_pred
    elif ensemble_type == 'CNN + SVM':
        final_pred = 0.7 * cnn_pred + 0.3 * svm_pred
    elif ensemble_type == 'CNN + RF + XGBoost':
        final_pred = 0.6 * cnn_pred + 0.2 * rf_pred + 0.2 * xgb_pred
    elif ensemble_type == 'CNN + All ML (Equal)':
        final_pred = 0.5 * cnn_pred + 0.125 * rf_pred + 0.125 * gb_pred + 0.125 * svm_pred + 0.125 * xgb_pred
    elif ensemble_type == 'CNN + All ML (Weighted)':
        final_pred = 0.6 * cnn_pred + 0.2 * rf_pred + 0.1 * gb_pred + 0.05 * svm_pred + 0.05 * xgb_pred
    else:  # CNN + Ensemble ML
        ensemble_pred = (rf_pred + gb_pred + svm_pred + xgb_pred) / 4
        final_pred = 0.7 * cnn_pred + 0.3 * ensemble_pred
    
    final_label = np.argmax(final_pred)
    confidence = np.max(final_pred)
    
    print(f"\nTrue Movement: {movement_names[movements[true_label]]}")
    print(f"Ensemble Type: {ensemble_type}")
    print(f"Final Prediction: {movement_names[movements[final_label]]} (Confidence: {confidence:.3f})")
    
    return true_label, final_label, confidence

# ML models dictionary
ml_models = {
    'rf': rf_clf_individual,
    'gb': gb_clf_individual,
    'svm': svm_clf_individual,
    'xgb': xgb_clf
}

# Run prediction with best ensemble
print(f"\nRandom Sample Prediction with Best Ensemble ({best_ensemble[0]}):")
predict_with_best_ensemble(model, ml_models, scaler, best_ensemble[0])

# -----------------------------
# Final Summary
# -----------------------------
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"CNN Model Accuracy: {test_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gb_acc * 100:.2f}%")
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
print(f"Ensemble ML Accuracy: {ensemble_acc * 100:.2f}%")
print(f"Best Ensemble Combination: {best_ensemble[0]} - {best_ensemble[1] * 100:.2f}%")
