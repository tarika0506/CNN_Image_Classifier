import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def show_classwise_images(dataset_path, classes, images_per_class=5):
    num_classes = len(classes)
    fig, axs = plt.subplots(images_per_class, num_classes, figsize=(3*num_classes, 3*images_per_class))

    for col, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_path, class_name)
        all_images = os.listdir(class_folder)
        chosen_images = random.sample(all_images, min(images_per_class, len(all_images)))

        for row, img_name in enumerate(chosen_images):
            img_path = os.path.join(class_folder, img_name)
            img = mpimg.imread(img_path)
            axs[row, col].imshow(img)
            if row == 0:  # Label class only at the top
                axs[row, col].set_title(class_name, fontsize=10)
            axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()

# Check and install required packages
def install_package(package):
    """Install package if not available"""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"{sys.executable} -m pip install {package}")

# Install required packages
required_packages = ['tensorflow', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'pillow']
for pkg in required_packages:
    install_package(pkg)

# Now import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class SimpleIntelImageClassifier:    
    def __init__(self, data_dir, img_size=(150, 150), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.num_classes = len(self.classes)
        self.model = None
        self.results = {}
        
    def verify_dataset_structure(self):
        print("Verifying dataset structure...")
        train_path = os.path.join(self.data_dir, 'seg_train')
        test_path = os.path.join(self.data_dir, 'seg_test')
        
        if not os.path.exists(train_path):
            print(f"[ERROR] Training path not found: {train_path}")
            return False
            
        if not os.path.exists(test_path):
            print(f"[ERROR] Test path not found: {test_path}")
            return False
    
        # Check for class folders
        missing_classes = []
        for class_name in self.classes:
            train_class_path = os.path.join(train_path, class_name)
            test_class_path = os.path.join(test_path, class_name)
            
            if not os.path.exists(train_class_path):
                missing_classes.append(f"train/{class_name}")
            if not os.path.exists(test_class_path):
                missing_classes.append(f"test/{class_name}")
        
        if missing_classes:
            print(f"[ERROR] Missing class folders: {missing_classes}")
            return False
        
        print("[OK] Dataset structure verified")
        return True

    def create_datasets(self):
        '''Create datasets using tf.data (modern approach)'''
        print("Creating datasets...")
        
        train_path = os.path.join(self.data_dir, 'seg_train')
        test_path = os.path.join(self.data_dir, 'seg_test')
        
        # Create training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="training",
            seed=123
        )
        
        # Create validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="validation", 
            seed=123
        )
        
        # Create test dataset
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Normalize and add augmentation
        normalization = tf.keras.layers.Rescaling(1./255)
        
        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        
        # Apply preprocessing
        train_ds = train_ds.map(lambda x, y: (normalization(data_augmentation(x)), y))
        val_ds = val_ds.map(lambda x, y: (normalization(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization(x), y))
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        
        print("✅ Datasets created successfully")
        
    def build_simple_cnn(self):
        """Build a simple CNN model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_enhanced_cnn(self):
        '''enhanced CNN with batch normalization'''
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(), 
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_model(self, base_model_name='VGG16'):
        """Build transfer learning model"""
        if base_model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, model, model_name, epochs=3):
        """Train a model and return results"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Train
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1   # 👈 this controls whether you see training progress
        )

        
        # Evaluate
        test_loss, test_acc = model.evaluate(self.test_ds, verbose=0)
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
        
        print(f"✅ {model_name} - Test Accuracy: {test_acc:.4f}")
        
        return history, test_acc
    
    def run_all_experiments(self):
        """Run all experiments"""
        if not self.verify_dataset_structure():
            return
        
        self.create_datasets()
        
        experiments = [
            ("Simple CNN", self.build_simple_cnn()),
            ("Enhanced CNN", self.build_enhanced_cnn()),
            ("VGG16 Transfer", self.build_transfer_model('VGG16')),
            ("ResNet50 Transfer", self.build_transfer_model('ResNet50'))
        ]
        
        for name, model in experiments:
            try:
                self.train_model(model, name)
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        self.print_comparison()
    
    def print_comparison(self):
        """Print comparison of all models"""
        print(f"\n{'='*60}")
        print("FINAL RESULTS COMPARISON")
        print(f"{'='*60}")
        
        for name, result in self.results.items():
            print(f"{name:20}: {result['test_accuracy']:.4f}")
    
    def plot_results(self):
        """Plot training history for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (name, result) in enumerate(self.results.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            history = result['history']
            
            axes[row, col].plot(history.history['accuracy'], label='Train Accuracy')
            axes[row, col].plot(history.history['val_accuracy'], label='Val Accuracy') 
            axes[row, col].set_title(f'{name} - Accuracy')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Accuracy')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()

def run_simple_experiments(dataset_path):
    """Simple function to run experiments"""
    classifier = SimpleIntelImageClassifier(dataset_path)
    classifier.run_all_experiments()
    classifier.plot_results()
    return classifier

if __name__ == "__main__":
    dataset_path = r"C:\Users\dummy\Downloads\archive\seg_train"  # root folder
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # Show sample images
    show_classwise_images(os.path.join(dataset_path, "seg_train"), classes, images_per_class=5)

    # Train models and display accuracy
    classifier = run_simple_experiments(dataset_path)
