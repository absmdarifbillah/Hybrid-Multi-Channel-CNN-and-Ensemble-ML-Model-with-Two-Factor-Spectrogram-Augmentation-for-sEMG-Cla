import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
from scipy.signal import spectrogram, welch, stft
from scipy.stats import kurtosis, skew
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import random
import warnings
import pywt
warnings.filterwarnings('ignore')

# -----------------------------
# Parameters
# -----------------------------
movements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
movement_to_index = {m: i for i, m in enumerate(movements)}
movement_names = {
    1: "Extension", 2: "Flexion", 3: "Ulnar Deviation", 4: "Radial Deviation",
    5: "Hook Grip", 6: "Power Grip", 7: "Spherical Grip", 8: "Precision Grip",
    9: "Lateral Grip", 10: "Pinch Grip"
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
# Signal Processing Functions
# -----------------------------
def apply_fft(signal):
    """Apply FFT to signal"""
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude[:len(fft_magnitude)//2]  # Return only positive frequencies

def apply_dwt(signal, wavelet='db4', level=4):
    """Apply Discrete Wavelet Transform to signal"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

def extract_time_domain_features(signal):
    """Extract time domain features"""
    features = [
        np.mean(signal), np.std(signal), np.median(signal),
        np.sqrt(np.mean(signal**2)), np.var(signal),
        kurtosis(signal), skew(signal),
        np.sum(np.diff(np.sign(signal)) != 0),  # Zero crossings
        np.sum(np.abs(np.diff(signal)))  # Waveform length
    ]
    return np.array(features)

def extract_frequency_domain_features(signal):
    """Extract frequency domain features"""
    try:
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)//4))
        if len(Pxx) > 0 and np.sum(Pxx) > 0:
            mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
            peak_freq = f[np.argmax(Pxx)]
            return np.array([mean_freq, peak_freq])
        else:
            return np.array([0, 0])
    except:
        return np.array([0, 0])

# -----------------------------
# Data Processing Functions
# -----------------------------
def process_signal(signal, method='stft'):
    """Process signal with different methods"""
    if method == 'stft':
        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=window_size, noverlap=overlap)
        return np.log1p(Sxx)
    elif method == 'fft':
        fft_result = apply_fft(signal)
        # Reshape to 2D for compatibility with CNN
        fft_2d = np.tile(fft_result, (64, 1))[:96, :96]
        return fft_2d
    elif method == 'dwt':
        dwt_result = apply_dwt(signal)
        # Reshape to 2D for compatibility with CNN
        dwt_2d = np.tile(dwt_result, (64, 1))[:96, :96]
        return dwt_2d
    return None

def load_data(b, method='stft'):
    """Load and process data"""
    X, Y = [], []
    
    for movement in movements:
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
                            processed_signal = process_signal(signal, method)
                            sensor_signals.append(processed_signal)
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
    
    if len(X) == 0:
        return np.array([]), np.array([])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Normalize per channel
    for i in range(4):
        channel_mean = X[:, :, :, i].mean()
        channel_std = X[:, :, :, i].std()
        X[:, :, :, i] = (X[:, :, :, i] - channel_mean) / channel_std
    
    return X, Y

def extract_features_from_data(X, use_time_features=True, use_freq_features=True):
    """Extract features from CNN data"""
    if not use_time_features and not use_freq_features:
        return None
    
    X_features = []
    for i in range(len(X)):
        sample_features = []
        for channel in range(4):
            channel_data = X[i, :, :, channel].flatten()
            if use_time_features:
                time_features = extract_time_domain_features(channel_data)
                sample_features.extend(time_features)
            if use_freq_features:
                freq_features = extract_frequency_domain_features(channel_data)
                sample_features.extend(freq_features)
        X_features.append(sample_features)
    
    return np.array(X_features)

# -----------------------------
# Model Creation
# -----------------------------
def create_cnn_model(input_shape, num_classes):
    """Create CNN model"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
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
# Evaluation Function
# -----------------------------
def evaluate_configuration(b, method='stft', use_time_features=True, use_freq_features=True):
    """Evaluate a specific configuration"""
    print(f"\nEvaluating configuration: method={method}, "
          f"time_features={use_time_features}, freq_features={use_freq_features}")
    
    # Load data
    X, Y = load_data(b, method)
    
    if len(X) == 0:
        print("No data found for this configuration")
        return 0, 0, 0
    
    # Extract features
    X_features = extract_features_from_data(X, use_time_features, use_freq_features)
    
    # One-hot encode labels
    Y_cat = tf.keras.utils.to_categorical(Y, num_classes=len(movements))
    
    # Split data
    if X_features is not None:
        # Split both CNN data and feature data
        X_train, X_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
            X, X_features, Y_cat, test_size=0.2, stratify=Y, random_state=42
        )
    else:
        # Use only CNN features
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y_cat, test_size=0.2, stratify=Y, random_state=42
        )
        X_features_train, X_features_test = None, None
    
    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(Y),
        y=Y
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Train CNN model
    cnn_model = create_cnn_model(input_shape=(96, 96, 4), num_classes=len(movements))
    cnn_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]
    
    # Train model
    history = cnn_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=0
    )
    
    # Evaluate CNN model
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    
    # If we have handcrafted features, train ensemble model
    ensemble_acc = 0
    combined_acc = cnn_acc
    
    if X_features is not None:
        # Scale features
        scaler = StandardScaler()
        X_features_train_scaled = scaler.fit_transform(X_features_train)
        X_features_test_scaled = scaler.transform(X_features_test)
        
        # Train ensemble classifiers
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        gb_clf = GradientBoostingClassifier(n_estimators=30, random_state=42, max_depth=6)
        svm_clf = SVC(probability=True, random_state=42, kernel='rbf', C=3, gamma='scale')
        
        # Create ensemble
        ensemble_clf = VotingClassifier(
            estimators=[('rf', rf_clf), ('gb', gb_clf), ('svm', svm_clf)],
            voting='soft'
        )
        
        y_train_labels = np.argmax(y_train, axis=1)
        ensemble_clf.fit(X_features_train_scaled, y_train_labels)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble_clf.predict(X_features_test_scaled)
        y_true = np.argmax(y_test, axis=1)
        ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
        
        # Combined prediction (CNN + Ensemble)
        cnn_pred_proba = cnn_model.predict(X_test, verbose=0)
        ensemble_pred_proba = ensemble_clf.predict_proba(X_features_test_scaled)
        
        # Weighted average of predictions
        final_pred_proba = 0.7 * cnn_pred_proba + 0.3 * ensemble_pred_proba
        final_pred = np.argmax(final_pred_proba, axis=1)
        combined_acc = accuracy_score(y_true, final_pred)
    
    print(f"CNN Accuracy: {cnn_acc * 100:.2f}%")
    if X_features is not None:
        print(f"Ensemble Accuracy: {ensemble_acc * 100:.2f}%")
    print(f"Combined Accuracy: {combined_acc * 100:.2f}%")
    
    return cnn_acc, ensemble_acc, combined_acc

# -----------------------------
# Main Execution
# -----------------------------
# Ask for signal type
while True:
    b = input("Enter 1 for offset signal or 2 for non-offset signal: ")
    if b in ['1', '2']:
        b = int(b)
        break
    else:
        print("Invalid input, try again.")

# Define configurations to test
configurations = [
    # Baseline with all features
    {'method': 'stft', 'time_features': True, 'freq_features': True, 'name': 'All Features (STFT)'},
    
    # Remove one feature at a time
    {'method': 'stft', 'time_features': False, 'freq_features': True, 'name': 'No Time Features'},
    {'method': 'stft', 'time_features': True, 'freq_features': False, 'name': 'No Frequency Features'},
    
    # Replace STFT with alternatives
    {'method': 'fft', 'time_features': True, 'freq_features': True, 'name': 'FFT Instead of STFT'},
    {'method': 'dwt', 'time_features': True, 'freq_features': True, 'name': 'DWT Instead of STFT'},
    
    # Remove both time and frequency features
    {'method': 'stft', 'time_features': False, 'freq_features': False, 'name': 'Only STFT (No Features)'},
]

# Evaluate all configurations
results = []
for config in configurations:
    cnn_acc, ensemble_acc, combined_acc = evaluate_configuration(
        b, config['method'], config['time_features'], config['freq_features']
    )
    results.append({
        'name': config['name'],
        'cnn_acc': cnn_acc,
        'ensemble_acc': ensemble_acc,
        'combined_acc': combined_acc
    })

# Display results
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS RESULTS")
print("="*60)

for result in results:
    print(f"\n{result['name']}:")
    print(f"  CNN Accuracy: {result['cnn_acc'] * 100:.2f}%")
    if result['ensemble_acc'] > 0:
        print(f"  Ensemble Accuracy: {result['ensemble_acc'] * 100:.2f}%")
    print(f"  Combined Accuracy: {result['combined_acc'] * 100:.2f}%")

# Plot results
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(results))
width = 0.25

cnn_accs = [r['cnn_acc'] * 100 for r in results]
ensemble_accs = [r['ensemble_acc'] * 100 for r in results]
combined_accs = [r['combined_acc'] * 100 for r in results]

rects1 = ax.bar(x - width, cnn_accs, width, label='CNN Only')
rects2 = ax.bar(x, ensemble_accs, width, label='Ensemble Only')
rects3 = ax.bar(x + width, combined_accs, width, label='Combined')

ax.set_xlabel('Configuration')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Feature Importance Analysis')
ax.set_xticks(x)
ax.set_xticklabels([r['name'] for r in results], rotation=45, ha='right')
ax.legend()

# Add value labels on top of bars
for i, v in enumerate(combined_accs):
    ax.text(i + width, v + 1, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Find best configuration
best_config = max(results, key=lambda x: x['combined_acc'])
print(f"\nBest configuration: {best_config['name']} with {best_config['combined_acc'] * 100:.2f}% accuracy")

# Calculate importance of each feature
baseline_acc = results[0]['combined_acc']  # All features baseline

print("\n" + "="*60)
print("FEATURE IMPORTANCE CALCULATION")
print("="*60)

# Time features importance
no_time_acc = results[1]['combined_acc']
time_importance = baseline_acc - no_time_acc
print(f"Time Features Importance: {time_importance * 100:.2f}%")

# Frequency features importance
no_freq_acc = results[2]['combined_acc']
freq_importance = baseline_acc - no_freq_acc
print(f"Frequency Features Importance: {freq_importance * 100:.2f}%")

# STFT vs FFT comparison
fft_acc = results[3]['combined_acc']
stft_vs_fft = baseline_acc - fft_acc
print(f"STFT vs FFT Difference: {stft_vs_fft * 100:.2f}%")

# STFT vs DWT comparison
dwt_acc = results[4]['combined_acc']
stft_vs_dwt = baseline_acc - dwt_acc
print(f"STFT vs DWT Difference: {stft_vs_dwt * 100:.2f}%")
