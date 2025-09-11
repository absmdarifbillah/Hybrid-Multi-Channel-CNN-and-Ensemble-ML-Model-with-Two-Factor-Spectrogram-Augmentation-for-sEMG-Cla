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
import random
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Parameters
# -----------------------------
# Focus on movements 1, 2, 3, 4, 6, and 8 only
movements = [1, 2, 3, 4, 6, 8]
movement_to_index = {m: i for i, m in enumerate(movements)}
movement_names = {
    1: "Extension",
    2: "Flexion",
    3: "Ulnar Deviation",
    4: "Radial Deviation",
    6: "Power Grip",
    8: "Precision Grip"
}

data_folder = "/Users/arif/Desktop/Project 312/WyoFlex_Dataset/DIGITAL DATA"
num_participants = 28
num_cycles = 3
num_sensors = 4
num_forearms = 2
sampling_rate = 13000
window_size = 256
overlap = 128
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

# If we don't have all classes, update the movements list
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
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_aug, Y_aug, test_size=test_ratio, stratify=Y_aug, random_state=42
)

# Second split: split remaining data into train and validation
val_ratio_adjusted = validation_ratio / (train_ratio + validation_ratio)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=val_ratio_adjusted, stratify=Y_temp, random_state=42
)

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
history = model.fit(
    X_train, Y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, Y_val),
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

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
y_ml_test = np.argmax(Y_test, axis=1)

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
y_ml_train = np.argmax(Y_train, axis=1)

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
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, Y_test, verbose=0)
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
y_true = np.argmax(Y_test, axis=1)

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
# Additional Analysis
# -----------------------------
print("\n" + "="*50)
print("ADDITIONAL ANALYSIS")
print("="*50)

# Calculate and display feature importance for Random Forest
feature_importances = rf_clf_individual.feature_importances_
print(f"Top 10 most important features (out of {len(feature_importances)}):")
top_indices = np.argsort(feature_importances)[-10:][::-1]
for i, idx in enumerate(top_indices):
    print(f"{i+1}. Feature {idx+1}: {feature_importances[idx]:.6f}")

# Calculate precision, recall, and F1-score for each class
precision = []
recall = []
f1_scores = []

for i in range(len(movements)):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    
    if tp + fp > 0:
        p = tp / (tp + fp)
    else:
        p = 0
        
    if tp + fn > 0:
        r = tp / (tp + fn)
    else:
        r = 0
        
    if p + r > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0
        
    precision.append(p)
    recall.append(r)
    f1_scores.append(f1)
    
    print(f"\n{movement_names[movements[i]]}:")
    print(f"  Precision: {p*100:.2f}%")
    print(f"  Recall: {r*100:.2f}%")
    print(f"  F1-score: {f1*100:.2f}%")

# Calculate macro and weighted averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1_scores)

# Weighted averages
support = np.sum(cm, axis=1)
weighted_precision = np.sum(np.array(precision) * support) / np.sum(support)
weighted_recall = np.sum(np.array(recall) * support) / np.sum(support)
weighted_f1 = np.sum(np.array(f1_scores) * support) / np.sum(support)

print(f"\nMacro Averages:")
print(f"  Precision: {macro_precision*100:.2f}%")
print(f"  Recall: {macro_recall*100:.2f}%")
print(f"  F1-score: {macro_f1*100:.2f}%")

print(f"\nWeighted Averages:")
print(f"  Precision: {weighted_precision*100:.2f}%")
print(f"  Recall: {weighted_recall*100:.2f}%")
print(f"  F1-score: {weighted_f1*100:.2f}%")

# -----------------------------
# Model Architecture Visualization
# -----------------------------
print("\n" + "="*50)
print("MODEL ARCHITECTURE")
print("="*50)

# Display model layers and parameters
total_params = 0
trainable_params = 0
non_trainable_params = 0

for layer in model.layers:
    params = layer.count_params()
    total_params += params
    if layer.trainable:
        trainable_params += params
    else:
        non_trainable_params += params
        
    print(f"{layer.name}: {params} parameters")

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")

# -----------------------------
# Training History Analysis
# -----------------------------
print("\n" + "="*50)
print("TRAINING HISTORY ANALYSIS")
print("="*50)

# Find best epoch
best_epoch = np.argmax(history.history['val_accuracy'])
print(f"Best epoch: {best_epoch+1}")
print(f"Best validation accuracy: {history.history['val_accuracy'][best_epoch]*100:.2f}%")
print(f"Best training accuracy: {history.history['accuracy'][best_epoch]*100:.2f}%")

# Calculate overfitting gap
overfitting_gap = history.history['accuracy'][best_epoch] - history.history['val_accuracy'][best_epoch]
print(f"Overfitting gap: {overfitting_gap*100:.2f}%")

# Learning rate analysis
if 'lr' in history.history:
    final_lr = history.history['lr'][-1]
    initial_lr = history.history['lr'][0]
    print(f"Initial learning rate: {initial_lr:.6f}")
    print(f"Final learning rate: {final_lr:.6f}")
    print(f"Learning rate reduction: {((initial_lr - final_lr)/initial_lr)*100:.2f}%")

# -----------------------------
# Data Statistics
# -----------------------------
print("\n" + "="*50)
print("DATA STATISTICS")
print("="*50)

print(f"Original samples: {len(X)}")
print(f"Augmented samples: {len(X_aug)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Class distribution in augmented dataset
aug_class_dist = np.bincount(Y_aug)
print("\nClass distribution in augmented dataset:")
for i, movement in enumerate(movements):
    print(f"{movement_names[movement]}: {aug_class_dist[i]} samples")

# Calculate balance ratio
min_samples = np.min(aug_class_dist)
max_samples = np.max(aug_class_dist)
balance_ratio = min_samples / max_samples
print(f"Dataset balance ratio: {balance_ratio:.3f}")

# -----------------------------
# Confidence Analysis
# -----------------------------
print("\n" + "="*50)
print("CONFIDENCE ANALYSIS")
print("="*50)

# Calculate confidence statistics
confidences = np.max(final_pred_proba, axis=1)
mean_confidence = np.mean(confidences)
median_confidence = np.median(confidences)
std_confidence = np.std(confidences)

print(f"Mean prediction confidence: {mean_confidence*100:.2f}%")
print(f"Median prediction confidence: {median_confidence*100:.2f}%")
print(f"Standard deviation of confidence: {std_confidence*100:.2f}%")

# Calculate accuracy by confidence level
confidence_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(confidence_bins)-1):
    low = confidence_bins[i]
    high = confidence_bins[i+1]
    mask = (confidences >= low) & (confidences < high)
    if np.sum(mask) > 0:
        bin_accuracy = np.mean(final_pred[mask] == y_true[mask])
        print(f"Confidence {low:.1f}-{high:.1f}: {bin_accuracy*100:.2f}% accuracy "
              f"({np.sum(mask)} samples)")
    else:
        print(f"Confidence {low:.1f}-{high:.1f}: No samples")

# -----------------------------
# Error Analysis
# -----------------------------
print("\n" + "="*50)
print("ERROR ANALYSIS")
print("="*50)

# Find most common misclassifications
error_indices = np.where(final_pred != y_true)[0]
if len(error_indices) > 0:
    error_pairs = []
    for idx in error_indices:
        true_class = y_true[idx]
        pred_class = final_pred[idx]
        error_pairs.append((true_class, pred_class))
    
    # Count frequency of each error type
    from collections import Counter
    error_counter = Counter(error_pairs)
    
    print("Most common misclassifications:")
    for (true, pred), count in error_counter.most_common(5):
        print(f"{movement_names[movements[true]]} → {movement_names[movements[pred]]}: {count} times")
else:
    print("No errors found!")

# -----------------------------
# Final Recommendations
# -----------------------------
print("\n" + "="*50)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("="*50)

# Based on analysis, provide recommendations
if balance_ratio < 0.8:
    print("• Consider additional data augmentation for minority classes")
    
if overfitting_gap > 0.15:
    print("• Increase regularization (dropout, weight decay)")
    print("• Consider simplifying the model architecture")

if mean_confidence < 0.7:
    print("• Model is uncertain, consider collecting more diverse training data")
    
if len(error_indices) > 0:
    most_common_error = error_counter.most_common(1)[0][0]
    true_class, pred_class = most_common_error
    print(f"• Focus on distinguishing {movement_names[movements[true_class]]} from {movement_names[movements[pred_class]]}")

print("• Consider hyperparameter tuning for optimal performance")
print("• Experiment with different architectures (ResNet, Inception)")
print("• Try different fusion strategies for multi-sensor data")

# -----------------------------
# End of Analysis
# -----------------------------
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
