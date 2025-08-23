import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import multiprocessing as mp
from functools import partial
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Enhanced Configuration
CONFIG = {
    'DATA_DIR': "D:\Python\Images\Input",
    'IMAGE_SIZE': (64, 64),  # Reduced for efficiency
    'RANDOM_STATE': 42,
    'MODEL_PATH': "microplastic_xgb_model.joblib",
    'SCALER_PATH': "microplastic_scaler.joblib",
    'CLASSES_PATH': "microplastic_classes.joblib",
    'TEST_SIZE': 0.2,
    'N_JOBS': -1,  # Use all available cores
    'MAX_WORKERS': mp.cpu_count(),
    'BATCH_SIZE': 100,
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicroplasticClassifier:
    def __init__(self, config=CONFIG):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def extract_features(self, img_array):
        """Extract multiple features from image for better classification"""
        features = []
        
        # Flatten pixel values (normalized)
        features.extend(img_array.flatten())
        
        # Statistical features
        features.extend([
            np.mean(img_array),
            np.std(img_array),
            np.min(img_array),
            np.max(img_array),
            np.median(img_array)
        ])
        
        # Histogram features (reduced bins for efficiency)
        hist, _ = np.histogram(img_array, bins=16, range=(0, 1))
        features.extend(hist / np.sum(hist))  # Normalize histogram
        
        return np.array(features, dtype=np.float32)
    
    def process_single_image(self, args):
        """Process a single image - designed for parallel processing"""
        img_path, label, image_size = args
        try:
            # Use OpenCV for faster loading (if available)
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("OpenCV failed to load image")
                img = cv2.resize(img, image_size)
                img_array = img.astype(np.float32) / 255.0
            except:
                # Fallback to PIL
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
            
            features = self.extract_features(img_array)
            return features, label
            
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            return None, None
    
    def load_and_preprocess_images(self, data_dir):
        """Load and preprocess images with parallel processing"""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Get class directories
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")
        
        self.class_names = sorted([d.name for d in class_dirs])
        logger.info(f"Found classes: {self.class_names}")
        
        # Prepare image paths and labels
        image_tasks = []
        for label, class_name in enumerate(self.class_names):
            class_path = data_path / class_name
            image_files = list(class_path.glob('*.png')) + \
                         list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.jpeg')) + \
                         list(class_path.glob('*.PNG')) + \
                         list(class_path.glob('*.JPG')) + \
                         list(class_path.glob('*.JPEG'))
            
            for img_path in image_files:
                image_tasks.append((img_path, label, self.config['IMAGE_SIZE']))
        
        logger.info(f"Processing {len(image_tasks)} images...")
        
        # Process images in parallel
        features_list, labels_list = [], []
        
        with ProcessPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
            results = list(tqdm(
                executor.map(self.process_single_image, image_tasks),
                total=len(image_tasks),
                desc="Loading images"
            ))
        
        # Filter out failed images and collect results
        for features, label in results:
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
        
        if not features_list:
            raise ValueError("No images were successfully processed")
        
        logger.info(f"Successfully processed {len(features_list)} images")
        return np.array(features_list), np.array(labels_list)
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate the model with optimized parameters"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['TEST_SIZE'], 
            random_state=self.config['RANDOM_STATE'],
            stratify=y
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimized XGBoost parameters for speed and performance
        self.model = XGBClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['RANDOM_STATE'],
            n_jobs=self.config['N_JOBS'],
            eval_metric='mlogloss',
            tree_method='hist',  # Faster tree construction
            enable_categorical=False
        )
        
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3, n_jobs=self.config['N_JOBS'])
        logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model
    
    def save_model(self):
        """Save model, scaler, and class names"""
        try:
            joblib.dump(self.model, self.config['MODEL_PATH'])
            joblib.dump(self.scaler, self.config['SCALER_PATH'])
            joblib.dump(self.class_names, self.config['CLASSES_PATH'])
            logger.info(f"Model artifacts saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self):
        """Load model, scaler, and class names"""
        try:
            self.model = joblib.load(self.config['MODEL_PATH'])
            self.scaler = joblib.load(self.config['SCALER_PATH'])
            self.class_names = joblib.load(self.config['CLASSES_PATH'])
            logger.info("Model artifacts loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Process image
            features, _ = self.process_single_image(
                (Path(image_path), 0, self.config['IMAGE_SIZE'])
            )
            
            if features is None:
                return None, None
            
            # Scale and predict
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            pred_label = self.model.predict(features_scaled)[0]
            pred_proba = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(pred_proba)
            
            return self.class_names[pred_label], confidence
            
        except Exception as e:
            logger.error(f"Failed to predict {image_path}: {e}")
            return None, None
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images efficiently"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare tasks
        tasks = [(Path(path), 0, self.config['IMAGE_SIZE']) for path in image_paths]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
            results = list(tqdm(
                executor.map(self.process_single_image, tasks),
                total=len(tasks),
                desc="Processing images"
            ))
        
        # Collect valid features
        valid_features = []
        valid_indices = []
        
        for i, (features, _) in enumerate(results):
            if features is not None:
                valid_features.append(features)
                valid_indices.append(i)
        
        if not valid_features:
            return [None] * len(image_paths)
        
        # Batch prediction
        features_array = np.array(valid_features)
        features_scaled = self.scaler.transform(features_array)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Map results back to original indices
        results_list = [None] * len(image_paths)
        for i, (pred_idx, proba) in enumerate(zip(predictions, probabilities)):
            original_idx = valid_indices[i]
            confidence = np.max(proba)
            results_list[original_idx] = (self.class_names[pred_idx], confidence)
        
        return results_list

def main():
    """Main execution function"""
    classifier = MicroplasticClassifier()
    
    try:
        # Load and preprocess data
        logger.info("Starting data loading...")
        X, y = classifier.load_and_preprocess_images(CONFIG['DATA_DIR'])
        
        # Train model
        logger.info("Starting model training...")
        classifier.train_and_evaluate(X, y)
        
        # Save model
        classifier.save_model()
        
        # Example prediction
        test_img_path = "path/to/test_image.png"  # Update this path
        if Path(test_img_path).exists():
            prediction, confidence = classifier.predict_image(test_img_path)
            if prediction:
                logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

def predict_single_image(image_path, model_path=None):
    """Utility function to predict a single image with pre-trained model"""
    config = CONFIG.copy()
    if model_path:
        config['MODEL_PATH'] = model_path
    
    classifier = MicroplasticClassifier(config)
    if classifier.load_model():
        return classifier.predict_image(image_path)
    return None, None

if __name__ == "__main__":
    main()