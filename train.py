import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import pickle

# Set random seeds for reproducibility
def set_seeds(seed_value=42):
    import os
    import random
    
    # Set environment and built-in seeds
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Initialize seeds
set_seeds()

# Hyperparameters
HYPER_DIMENSION = 64
HYPER_BATCH_SIZE = 128
HYPER_EPOCHS = 200
HYPER_CHANNELS = 1
HYPER_MODE = 'grayscale'

# File paths (update these to your specific paths)
TRAIN_PATH = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
VAL_PATH = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/"
TEST_PATH = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/"

# Data Generators with Augmentation
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0, 
        shear_range=0.2,
        zoom_range=0.2, 
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_PATH, 
        target_size=(HYPER_DIMENSION, HYPER_DIMENSION),
        batch_size=HYPER_BATCH_SIZE, 
        color_mode=HYPER_MODE,
        class_mode='binary', 
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        directory=VAL_PATH, 
        target_size=(HYPER_DIMENSION, HYPER_DIMENSION),
        batch_size=HYPER_BATCH_SIZE, 
        color_mode=HYPER_MODE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        directory=TEST_PATH, 
        target_size=(HYPER_DIMENSION, HYPER_DIMENSION),
        batch_size=HYPER_BATCH_SIZE, 
        color_mode=HYPER_MODE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator, test_generator

# Create CNN Model
def create_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(HYPER_DIMENSION, HYPER_DIMENSION, HYPER_CHANNELS)),
        
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Flatten(),
        
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy', 
        metrics=[AUC()]
    )
    
    return model

# Create visualization charts

def create_charts(model, history, test_generator):
    # Prediction
    y_true = test_generator.classes
    Y_pred = model.predict(test_generator)
    y_pred = (Y_pred > 0.5).flatten()
    y_pred_prob = Y_pred.flatten()
    
    # Print available metrics to debug
    print("Available metrics:", list(history.history.keys()))
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Training vs Validation Loss
    plt.subplot(2,2,1)
    plt.title("Training vs. Validation Loss")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Number of Epochs")
    plt.legend()

    # Plot 2: Training vs Validation AUC
    plt.subplot(2,2,2)
    plt.title("Training vs. Validation AUC Score")
    
    # Check for exact metric names
    auc_metric = 'auc' if 'auc' in history.history else 'AUC'
    val_auc_metric = 'val_auc' if 'val_auc' in history.history else 'val_AUC'
    
    plt.plot(history.history.get(auc_metric, []), label='Training AUC')
    plt.plot(history.history.get(val_auc_metric, []), label='Validation AUC')
    plt.xlabel("Number of Epochs")
    plt.legend()
    
    # Plot 3: Confusion Matrix
    plt.subplot(2,2,3)
    cm = confusion_matrix(y_true, y_pred)
    names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2,2)
    ticklabels = ['Normal', 'Pneumonia']

    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', 
                xticklabels=ticklabels, yticklabels=ticklabels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Plot 4: ROC Curve
    plt.subplot(2,2,4)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 50%)")
    plt.plot(fpr, tpr, label=f'CNN (AUC = {auc*100:.2f}%)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print summary statistics
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * recall / (precision + recall)
    
    print('[Summary Statistics]')
    print(f'Accuracy = {accuracy:.2%}')
    print(f'Precision = {precision:.2%}')
    print(f'Recall = {recall:.2%}')
    print(f'Specificity = {specificity:.2%}')
    print(f'F1 Score = {f1:.2%}')
def save_model(model, history):
    """
    Save the trained model and its training history
    
    Args:
        model (keras.Model): Trained Keras model
        history (keras.callbacks.History): Training history
    """
    # Create a directory to store models if it doesn't exist
    import os
    os.makedirs('/kaggle/working/saved_models', exist_ok=True)
    
    # Save model in multiple formats
    
    # 1. Save entire model (recommended for TensorFlow/Keras)
    model.save('/kaggle/working/saved_models/pneumonia_cnn_model.h5')
    
    # 2. Save model weights with correct extension
    model.save_weights('/kaggle/working/saved_models/pneumonia_cnn_model.weights.h5')
    
    # 3. Save model architecture as JSON
    model_json = model.to_json()
    with open('/kaggle/working/saved_models/pneumonia_cnn_architecture.json', 'w') as json_file:
        json_file.write(model_json)
    
    # 4. Pickle the entire model (less recommended, but can work)
    with open('/kaggle/working/saved_models/pneumonia_cnn_model.pkl', 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    
    # Save training history
    with open('/kaggle/working/saved_models/training_history.pkl', 'wb') as hist_file:
        pickle.dump(history.history, hist_file)
    
    print("Models and training history saved successfully!")
def main():
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()
    
    # Create and train the model
    model = create_cnn_model()
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Fit the model
    history = model.fit(
        train_generator, 
        epochs=HYPER_EPOCHS, 
        validation_data=val_generator,
        callbacks=[early_stopping],
        verbose=2
    )
    
    # Create charts and print summary
    create_charts(model, history, test_generator)
    
    # Save the model
    save_model(model, history)

# Run the main function
if __name__ == '__main__':
    main()


