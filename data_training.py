import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import multiprocessing as mp
from functools import partial
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration for Different Assistant Devices
DEVICE_CONFIGS = {
    'object_recognition': {
        'DATA_DIR': "D:/Python/Images/ObjectRecognition",
        'IMAGE_SIZE': (224, 224),  # Higher resolution for object details
        'CLASSES': ['person', 'vehicle', 'animal', 'furniture', 'electronics', 'food'],
        'MODEL_PREFIX': 'object_recognition',
        'CONFIDENCE_THRESHOLD': 0.7,
        'REAL_TIME_OPTIMIZE': True,
        'FEATURE_FOCUS': 'shape_texture',
    },
    'wearable_navigation': {
        'DATA_DIR': "D:/Python/Images/Navigation", 
        'IMAGE_SIZE': (128, 128),  # Balanced for mobile processing
        'CLASSES': ['obstacle', 'path_clear', 'stairs_up', 'stairs_down', 'door', 'wall'],
        'MODEL_PREFIX': 'wearable_navigation',
        'CONFIDENCE_THRESHOLD': 0.85,  # Higher confidence for safety
        'REAL_TIME_OPTIMIZE': True,
        'FEATURE_FOCUS': 'depth_edges',
    },
    'smart_doorbell': {
        'DATA_DIR': "D:/Python/Images/Doorbell",
        'IMAGE_SIZE': (160, 160),  # Good for face/person detection
        'CLASSES': ['known_person', 'unknown_person', 'delivery', 'no_person', 'suspicious_activity'],
        'MODEL_PREFIX': 'smart_doorbell',
        'CONFIDENCE_THRESHOLD': 0.8,
        'REAL_TIME_OPTIMIZE': False,  # Can afford more processing time
        'FEATURE_FOCUS': 'face_person',
    }
}

# Base Configuration
BASE_CONFIG = {
    'RANDOM_STATE': 42,
    'TEST_SIZE': 0.2,
    'VALIDATION_SIZE': 0.15,
    'N_JOBS': -1,
    'MAX_WORKERS': min(mp.cpu_count(), 6),
    'BATCH_SIZE': 32,
    'MIN_IMAGES_PER_CLASS': 20,
    'CROSS_VALIDATION_FOLDS': 5,
    'MODEL_TYPE': 'xgboost',  # Options: 'xgboost', 'random_forest', 'ensemble'
    'AUGMENTATION': True,
}

# Setup logging
def setup_logging(device_type):
    log_filename = f'{device_type}_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class AssistantDeviceClassifier:
    def __init__(self, device_type='object_recognition'):
        if device_type not in DEVICE_CONFIGS:
            raise ValueError(f"Device type must be one of: {list(DEVICE_CONFIGS.keys())}")
        
        self.device_type = device_type
        self.config = {**BASE_CONFIG, **DEVICE_CONFIGS[device_type]}
        self.logger = setup_logging(device_type)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = self.config['CLASSES']
        self.num_classes = len(self.class_names)
        
        # Device-specific paths
        self.model_path = f"{self.config['MODEL_PREFIX']}_model.joblib"
        self.scaler_path = f"{self.config['MODEL_PREFIX']}_scaler.joblib"
        self.classes_path = f"{self.config['MODEL_PREFIX']}_classes.joblib"
        
        self.logger.info(f"Initialized {device_type} classifier with {self.num_classes} classes")
        
    def extract_features(self, img_array):
        """Extract features optimized for different device types"""
        features = []
        
        # Base features for all devices
        features.extend(img_array.flatten())
        
        # Statistical features
        features.extend([
            np.mean(img_array), np.std(img_array),
            np.min(img_array), np.max(img_array),
            np.median(img_array), np.percentile(img_array, 25),
            np.percentile(img_array, 75)
        ])
        
        # Device-specific feature extraction
        if self.config['FEATURE_FOCUS'] == 'shape_texture':
            features.extend(self._extract_shape_texture_features(img_array))
        elif self.config['FEATURE_FOCUS'] == 'depth_edges':
            features.extend(self._extract_depth_edge_features(img_array))
        elif self.config['FEATURE_FOCUS'] == 'face_person':
            features.extend(self._extract_face_person_features(img_array))
        
        # Histogram features
        hist, _ = np.histogram(img_array, bins=32, range=(0, 1))
        features.extend(hist / np.sum(hist))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_shape_texture_features(self, img_array):
        """Features for object recognition - focus on shapes and textures"""
        features = []
        
        # Sobel edge detection
        if len(img_array.shape) == 2:
            img_uint8 = (img_array * 255).astype(np.uint8)
            sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            features.extend([
                np.mean(edge_magnitude),
                np.std(edge_magnitude),
                np.sum(edge_magnitude > 50) / edge_magnitude.size  # Edge density
            ])
        
        # Texture features (variance in local windows)
        h, w = img_array.shape[:2]
        window_size = min(8, h//4, w//4)
        texture_vars = []
        for i in range(0, h-window_size, window_size):
            for j in range(0, w-window_size, window_size):
                window = img_array[i:i+window_size, j:j+window_size]
                texture_vars.append(np.var(window))
        
        if texture_vars:
            features.extend([np.mean(texture_vars), np.std(texture_vars)])
        else:
            features.extend([0, 0])
        
        return features
    
    def _extract_depth_edge_features(self, img_array):
        """Features for navigation - focus on obstacles and paths"""
        features = []
        
        if len(img_array.shape) == 2:
            img_uint8 = (img_array * 255).astype(np.uint8)
            
            # Canny edge detection for navigation
            edges = cv2.Canny(img_uint8, 50, 150)
            
            # Horizontal and vertical line detection
            h_lines = np.sum(edges, axis=1)  # Horizontal projection
            v_lines = np.sum(edges, axis=0)  # Vertical projection
            
            features.extend([
                np.max(h_lines) / img_array.shape[1],  # Strongest horizontal edge
                np.max(v_lines) / img_array.shape[0],  # Strongest vertical edge
                np.mean(edges) / 255.0,  # Average edge intensity
                np.sum(edges > 0) / edges.size,  # Edge density
            ])
            
            # Quadrant analysis for obstacle detection
            h_mid, w_mid = img_array.shape[0]//2, img_array.shape[1]//2
            quadrants = [
                img_array[:h_mid, :w_mid],      # Top-left
                img_array[:h_mid, w_mid:],      # Top-right
                img_array[h_mid:, :w_mid],      # Bottom-left
                img_array[h_mid:, w_mid:]       # Bottom-right
            ]
            
            for quad in quadrants:
                features.append(np.mean(quad))
        
        return features
    
    def _extract_face_person_features(self, img_array):
        """Features for doorbell - focus on people and faces"""
        features = []
        
        if len(img_array.shape) == 2:
            img_uint8 = (img_array * 255).astype(np.uint8)
            
            # Gradient features for person detection
            grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y)),
                np.std(grad_x),
                np.std(grad_y)
            ])
            
            # Central region analysis (where faces/people typically appear)
            h, w = img_array.shape
            center_region = img_array[h//4:3*h//4, w//4:3*w//4]
            
            features.extend([
                np.mean(center_region),
                np.std(center_region),
                np.max(center_region),
                np.min(center_region)
            ])
            
            # Symmetry features (faces are roughly symmetric)
            left_half = img_array[:, :w//2]
            right_half = np.fliplr(img_array[:, w//2:])
            
            if left_half.shape == right_half.shape:
                symmetry_diff = np.mean(np.abs(left_half - right_half))
                features.append(symmetry_diff)
            else:
                features.append(0.5)  # Default value
        
        return features
    
    def augment_image(self, img_array):
        """Simple data augmentation for better generalization"""
        if not self.config.get('AUGMENTATION', False):
            return [img_array]
        
        augmented = [img_array]  # Original
        
        # Rotation (small angles)
        for angle in [-5, 5]:
            if len(img_array.shape) == 2:
                h, w = img_array.shape
                center = (w//2, h//2)
                img_uint8 = (img_array * 255).astype(np.uint8)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img_uint8, M, (w, h))
                augmented.append(rotated.astype(np.float32) / 255.0)
        
        # Brightness adjustment
        brighter = np.clip(img_array * 1.2, 0, 1)
        darker = np.clip(img_array * 0.8, 0, 1)
        augmented.extend([brighter, darker])
        
        return augmented
    
    def process_single_image(self, args):
        """Process a single image with device-specific optimizations"""
        img_path, label, image_size, use_augmentation = args
        try:
            # Load image
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
            
            # Apply augmentation if enabled
            if use_augmentation:
                augmented_images = self.augment_image(img_array)
                results = []
                for aug_img in augmented_images:
                    features = self.extract_features(aug_img)
                    results.append((features, label))
                return results
            else:
                features = self.extract_features(img_array)
                return [(features, label)]
            
        except Exception as e:
            self.logger.warning(f"Failed to process {img_path}: {e}")
            return [(None, None)]
    
    def load_and_preprocess_images(self, data_dir=None):
        """Load and preprocess images with device-specific handling"""
        if data_dir is None:
            data_dir = self.config['DATA_DIR']
            
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        
        # Get class directories
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")
        
        found_classes = sorted([d.name for d in class_dirs])
        
        # Validate classes match expected classes
        if set(found_classes) != set(self.class_names):
            self.logger.warning(f"Found classes {found_classes} don't match expected {self.class_names}")
            self.logger.info("Using found classes instead")
            self.class_names = found_classes
            self.num_classes = len(self.class_names)
        
        self.logger.info(f"Found classes: {self.class_names}")
        
        # Prepare image tasks
        image_tasks = []
        class_counts = {}
        
        for label, class_name in enumerate(self.class_names):
            class_path = data_path / class_name
            if not class_path.exists():
                self.logger.warning(f"Class directory {class_name} not found, skipping")
                continue
                
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(list(class_path.glob(ext)))
            
            class_counts[class_name] = len(image_files)
            
            if len(image_files) < self.config['MIN_IMAGES_PER_CLASS']:
                self.logger.warning(f"Class {class_name} has only {len(image_files)} images (minimum: {self.config['MIN_IMAGES_PER_CLASS']})")
            
            for img_path in image_files:
                image_tasks.append((
                    img_path, 
                    label, 
                    self.config['IMAGE_SIZE'],
                    self.config.get('AUGMENTATION', False)
                ))
        
        self.logger.info(f"Class distribution: {class_counts}")
        self.logger.info(f"Processing {len(image_tasks)} images...")
        
        # Process images in parallel
        features_list, labels_list = [], []
        
        with ProcessPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
            results = list(tqdm(
                executor.map(self.process_single_image, image_tasks),
                total=len(image_tasks),
                desc=f"Loading {self.device_type} images"
            ))
        
        # Flatten results (accounting for augmentation)
        for result_batch in results:
            for features, label in result_batch:
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
        
        if not features_list:
            raise ValueError("No images were successfully processed")
        
        self.logger.info(f"Successfully processed {len(features_list)} image instances")
        return np.array(features_list), np.array(labels_list)
    
    def create_model(self):
        """Create model optimized for device type"""
        if self.config['MODEL_TYPE'] == 'xgboost':
            if self.config['REAL_TIME_OPTIMIZE']:
                # Faster model for real-time applications
                model = XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config['RANDOM_STATE'],
                    n_jobs=self.config['N_JOBS'],
                    eval_metric='mlogloss',
                    tree_method='hist'
                )
            else:
                # More accurate model for non-real-time applications
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.config['RANDOM_STATE'],
                    n_jobs=self.config['N_JOBS'],
                    eval_metric='mlogloss',
                    tree_method='hist'
                )
        
        elif self.config['MODEL_TYPE'] == 'random_forest':
            n_estimators = 50 if self.config['REAL_TIME_OPTIMIZE'] else 100
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None if not self.config['REAL_TIME_OPTIMIZE'] else 10,
                random_state=self.config['RANDOM_STATE'],
                n_jobs=self.config['N_JOBS']
            )
        
        return model
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate model with device-specific metrics"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['TEST_SIZE'], 
            random_state=self.config['RANDOM_STATE'],
            stratify=y
        )
        
        # Scale features
        self.logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self.create_model()
        
        self.logger.info(f"Training {self.config['MODEL_TYPE']} model for {self.device_type}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        self.logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Accuracy: {accuracy:.3f}")
        
        print(f"\n{self.device_type.upper()} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Device-specific evaluation
        self._evaluate_device_specific(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=self.config['CROSS_VALIDATION_FOLDS'], 
            n_jobs=self.config['N_JOBS']
        )
        self.logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model
    
    def _evaluate_device_specific(self, y_test, y_pred, y_pred_proba):
        """Device-specific evaluation metrics"""
        confidence_threshold = self.config['CONFIDENCE_THRESHOLD']
        
        # High-confidence predictions
        max_probas = np.max(y_pred_proba, axis=1)
        high_conf_mask = max_probas >= confidence_threshold
        
        if np.any(high_conf_mask):
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            self.logger.info(f"High-confidence accuracy (>{confidence_threshold:.2f}): {high_conf_accuracy:.3f}")
            self.logger.info(f"Percentage of high-confidence predictions: {np.mean(high_conf_mask):.3f}")
        
        # Device-specific metrics
        if self.device_type == 'wearable_navigation':
            # Safety-critical: obstacle detection recall
            obstacle_idx = self.class_names.index('obstacle') if 'obstacle' in self.class_names else None
            if obstacle_idx is not None:
                obstacle_recall = np.sum((y_test == obstacle_idx) & (y_pred == obstacle_idx)) / np.sum(y_test == obstacle_idx)
                self.logger.info(f"Obstacle detection recall (safety critical): {obstacle_recall:.3f}")
        
        elif self.device_type == 'smart_doorbell':
            # Security: known vs unknown person distinction
            if 'known_person' in self.class_names and 'unknown_person' in self.class_names:
                known_idx = self.class_names.index('known_person')
                unknown_idx = self.class_names.index('unknown_person')
                
                # False acceptance rate (unknown classified as known)
                far = np.sum((y_test == unknown_idx) & (y_pred == known_idx)) / np.sum(y_test == unknown_idx)
                # False rejection rate (known classified as unknown)  
                frr = np.sum((y_test == known_idx) & (y_pred == unknown_idx)) / np.sum(y_test == known_idx)
                
                self.logger.info(f"False Acceptance Rate: {far:.3f}")
                self.logger.info(f"False Rejection Rate: {frr:.3f}")
    
    def save_model(self):
        """Save model artifacts"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.class_names, self.classes_path)
            
            # Save configuration
            config_path = f"{self.config['MODEL_PREFIX']}_config.joblib"
            joblib.dump(self.config, config_path)
            
            self.logger.info(f"Model artifacts saved for {self.device_type}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self):
        """Load model artifacts"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.class_names = joblib.load(self.classes_path)
            self.logger.info(f"Model artifacts loaded for {self.device_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            features, _ = self.process_single_image(
                (Path(image_path), 0, self.config['IMAGE_SIZE'], False)
            )[0]
            
            if features is None:
                return None, None, None
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            pred_label = self.model.predict(features_scaled)[0]
            pred_proba = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(pred_proba)
            
            # Check if confidence meets threshold
            meets_threshold = confidence >= self.config['CONFIDENCE_THRESHOLD']
            
            return self.class_names[pred_label], confidence, meets_threshold
            
        except Exception as e:
            self.logger.error(f"Failed to predict {image_path}: {e}")
            return None, None, None

def main():
    """Main execution function with device selection"""
    print("Available devices:")
    for i, device in enumerate(DEVICE_CONFIGS.keys(), 1):
        print(f"{i}. {device.replace('_', ' ').title()}")
    
    try:
        choice = int(input("\nSelect device type (1-3): ")) - 1
        device_types = list(DEVICE_CONFIGS.keys())
        
        if 0 <= choice < len(device_types):
            device_type = device_types[choice]
        else:
            print("Invalid choice, using object_recognition")
            device_type = 'object_recognition'
    except:
        print("Invalid input, using object_recognition")
        device_type = 'object_recognition'
    
    classifier = AssistantDeviceClassifier(device_type)
    
    try:
        # Load and preprocess data
        classifier.logger.info(f"Starting {device_type} data loading...")
        X, y = classifier.load_and_preprocess_images()
        
        # Train model
        classifier.logger.info(f"Starting {device_type} model training...")
        classifier.train_and_evaluate(X, y)
        
        # Save model
        classifier.save_model()
        
        classifier.logger.info(f"{device_type} training completed successfully!")
        
    except Exception as e:
        classifier.logger.error(f"Error in {device_type} training: {e}")
        raise

def predict_with_device(image_path, device_type='object_recognition', model_dir='.'):
    """Utility function to predict with specific device model"""
    classifier = AssistantDeviceClassifier(device_type)
    
    # Update model paths if in different directory
    if model_dir != '.':
        classifier.model_path = os.path.join(model_dir, classifier.model_path)
        classifier.scaler_path = os.path.join(model_dir, classifier.scaler_path)
        classifier.classes_path = os.path.join(model_dir, classifier.classes_path)
    
    if classifier.load_model():
        return classifier.predict_image(image_path)
    return None, None, None

if __name__ == "__main__":
    main()