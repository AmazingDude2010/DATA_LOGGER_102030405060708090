import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
import logging

# AI Classification imports
try:
    from data_training import MicroplasticClassifier
    AI_AVAILABLE = True
except ImportError:
    print("[WARN] AI classifier not available. Install requirements and train model first.")
    AI_AVAILABLE = False

# Hardware imports (graceful fallback if not available)
try:
    import cv2
except ImportError:
    cv2 = None
    print("[WARN] OpenCV not available - camera functions disabled")

try:
    import numpy as np
except ImportError:
    np = None
    print("[WARN] NumPy not available - image processing disabled")

try:
    import Adafruit_DHT
except ImportError:
    Adafruit_DHT = None
    print("[WARN] DHT sensor library not available")

try:
    from RPLCD.i2c import CharLCD
except ImportError:
    CharLCD = None
    print("[WARN] LCD library not available")

# ---------- ENHANCED SETTINGS ----------
CONFIG = {
    # File paths
    'CSV_FILE': "microplastic_data.csv",
    'JSON_FILE': "microplastic_data.ndjson",
    'IMAGES_DIR': "captured_images",
    'CLASSIFIED_IMAGES_DIR': "classified_images",
    
    # Hardware settings
    'DHT_SENSOR_TYPE': Adafruit_DHT.DHT22 if Adafruit_DHT else None,
    'DHT_PIN': 4,
    'CAMERA_INDEX': 0,
    'LCD_ADDRESS': 0x27,
    'REFRESH_INTERVAL': 30,  # seconds
    
    # AI Classification settings
    'ENABLE_AI_CLASSIFICATION': True,
    'CLASSIFICATION_CONFIDENCE_THRESHOLD': 0.6,
    'SAVE_CLASSIFIED_IMAGES': True,
    'IMAGE_PROCESSING_SIZE': (128, 128),
    
    # Detection settings
    'MIN_PARTICLE_AREA': 50,  # pixels
    'MAX_PARTICLE_AREA': 5000,  # pixels
    'DETECTION_THRESHOLD': 127,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('microplastic_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMicroplasticSystem:
    def __init__(self, config=CONFIG):
        self.config = config
        self.lcd = None
        self.ai_classifier = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.init_directories()
        self.init_lcd()
        self.init_ai_classifier()
        
    def init_directories(self):
        """Create necessary directories"""
        Path(self.config['IMAGES_DIR']).mkdir(exist_ok=True)
        Path(self.config['CLASSIFIED_IMAGES_DIR']).mkdir(exist_ok=True)
        
        # Create subdirectories for each class
        if AI_AVAILABLE:
            for class_name in ['fiber', 'fragment', 'film', 'pellet', 'unknown']:
                Path(self.config['CLASSIFIED_IMAGES_DIR'], class_name).mkdir(exist_ok=True)
    
    def init_lcd(self):
        """Initialize LCD display"""
        if CharLCD:
            try:
                self.lcd = CharLCD('PCF8574', self.config['LCD_ADDRESS'])
                logger.info("LCD initialized successfully")
                return self.lcd
            except Exception as e:
                logger.error(f"LCD initialization failed: {e}")
        return None
    
    def init_ai_classifier(self):
        """Initialize AI classifier"""
        if not AI_AVAILABLE or not self.config['ENABLE_AI_CLASSIFICATION']:
            logger.info("AI classification disabled")
            return None
        
        try:
            self.ai_classifier = MicroplasticClassifier()
            if self.ai_classifier.load_model():
                logger.info("AI classifier loaded successfully")
                return self.ai_classifier
            else:
                logger.error("Failed to load AI model - train the model first!")
                return None
        except Exception as e:
            logger.error(f"AI classifier initialization failed: {e}")
            return None
    
    def capture_and_save_image(self):
        """Capture image from microscope camera and save it"""
        if not cv2 or not np:
            logger.error("OpenCV/NumPy not available for image capture")
            return None, None
        
        cap = cv2.VideoCapture(self.config['CAMERA_INDEX'])
        if not cap.isOpened():
            logger.error("Cannot access microscope camera")
            return None, None
        
        try:
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Failed to capture image")
                return None, None
            
            # Save original image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_filename = f"capture_{timestamp}.jpg"
            image_path = Path(self.config['IMAGES_DIR']) / image_filename
            
            cv2.imwrite(str(image_path), frame)
            logger.info(f"Image saved: {image_filename}")
            
            return frame, str(image_path)
            
        except Exception as e:
            logger.error(f"Image capture error: {e}")
            cap.release()
            return None, None
    
    def detect_particles_advanced(self, frame):
        """Advanced particle detection with size filtering"""
        if not cv2 or not np:
            return None, [], []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive thresholding for better edge detection
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter particles by size and extract features
            valid_particles = []
            particle_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if (self.config['MIN_PARTICLE_AREA'] <= area <= 
                    self.config['MAX_PARTICLE_AREA']):
                    
                    # Extract bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract particle region for classification
                    particle_roi = frame[y:y+h, x:x+w]
                    
                    # Calculate additional features
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    particle_info = {
                        'contour': contour,
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    }
                    
                    valid_particles.append(particle_info)
                    particle_regions.append(particle_roi)
            
            return len(valid_particles), valid_particles, particle_regions
            
        except Exception as e:
            logger.error(f"Particle detection error: {e}")
            return None, [], []
    
    def classify_particles(self, particle_regions, image_path):
        """Classify detected particles using AI"""
        if not self.ai_classifier or not particle_regions:
            return [], {}
        
        classifications = []
        class_counts = {'fiber': 0, 'fragment': 0, 'film': 0, 'pellet': 0, 'unknown': 0}
        
        try:
            for i, particle_roi in enumerate(particle_regions):
                if particle_roi.size == 0:
                    continue
                
                # Resize particle to standard size
                resized_particle = cv2.resize(
                    particle_roi, 
                    self.config['IMAGE_PROCESSING_SIZE']
                )
                
                # Save particle image temporarily
                temp_particle_path = f"temp_particle_{i}.jpg"
                cv2.imwrite(temp_particle_path, resized_particle)
                
                # Classify particle
                prediction, confidence = self.ai_classifier.predict_image(temp_particle_path)
                
                # Clean up temp file
                if os.path.exists(temp_particle_path):
                    os.remove(temp_particle_path)
                
                if prediction and confidence >= self.config['CLASSIFICATION_CONFIDENCE_THRESHOLD']:
                    classifications.append({
                        'particle_id': i,
                        'class': prediction,
                        'confidence': confidence
                    })
                    class_counts[prediction] += 1
                else:
                    classifications.append({
                        'particle_id': i,
                        'class': 'unknown',
                        'confidence': confidence or 0.0
                    })
                    class_counts['unknown'] += 1
            
            # Save classified image if enabled
            if self.config['SAVE_CLASSIFIED_IMAGES'] and image_path:
                self.save_classified_image(image_path, classifications)
            
            return classifications, class_counts
            
        except Exception as e:
            logger.error(f"Particle classification error: {e}")
            return [], class_counts
    
    def save_classified_image(self, original_image_path, classifications):
        """Save image with classification annotations"""
        try:
            # Read original image
            image = cv2.imread(original_image_path)
            if image is None:
                return
            
            # Add annotations for each classified particle
            for classification in classifications:
                particle_id = classification['particle_id']
                class_name = classification['class']
                confidence = classification['confidence']
                
                # You would need particle coordinates from detection
                # This is a simplified version
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    image, text, (10, 30 + particle_id * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            
            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classified_{timestamp}.jpg"
            output_path = Path(self.config['CLASSIFIED_IMAGES_DIR']) / filename
            
            cv2.imwrite(str(output_path), image)
            logger.info(f"Classified image saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving classified image: {e}")
    
    def get_environmental_data(self):
        """Get temperature and humidity data"""
        if not Adafruit_DHT or not self.config['DHT_SENSOR_TYPE']:
            logger.warning("DHT sensor not available")
            return None, None
        
        try:
            humidity, temperature = Adafruit_DHT.read_retry(
                self.config['DHT_SENSOR_TYPE'], 
                self.config['DHT_PIN']
            )
            
            if humidity is None or temperature is None:
                logger.warning("Failed to read from DHT22 sensor")
                return None, None
            
            return round(temperature, 2), round(humidity, 2)
            
        except Exception as e:
            logger.error(f"Environmental sensor error: {e}")
            return None, None
    
    def log_enhanced_data(self, particle_count, classifications, class_counts, 
                         temperature, humidity, image_path):
        """Log comprehensive data including classifications"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Enhanced data entry
        data_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "total_particles": particle_count or 0,
            "classified_particles": len(classifications) if classifications else 0,
            "fiber_count": class_counts.get('fiber', 0),
            "fragment_count": class_counts.get('fragment', 0),
            "film_count": class_counts.get('film', 0),
            "pellet_count": class_counts.get('pellet', 0),
            "unknown_count": class_counts.get('unknown', 0),
            "temperature": temperature,
            "humidity": humidity,
            "image_file": os.path.basename(image_path) if image_path else None,
            "ai_enabled": self.ai_classifier is not None
        }
        
        # Save to CSV
        self.save_to_csv(data_entry)
        
        # Save to NDJSON
        self.save_to_ndjson(data_entry)
        
        # Save detailed classification data
        if classifications:
            self.save_classification_details(timestamp, classifications)
        
        logger.info(f"Enhanced data logged at {timestamp}")
    
    def save_to_csv(self, data_entry):
        """Save data to CSV file"""
        file_exists = os.path.isfile(self.config['CSV_FILE'])
        try:
            with open(self.config['CSV_FILE'], mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_entry)
        except Exception as e:
            logger.error(f"CSV writing error: {e}")
    
    def save_to_ndjson(self, data_entry):
        """Save data to NDJSON file"""
        try:
            with open(self.config['JSON_FILE'], "a") as jf:
                jf.write(json.dumps(data_entry) + "\n")
        except Exception as e:
            logger.error(f"NDJSON writing error: {e}")
    
    def save_classification_details(self, timestamp, classifications):
        """Save detailed classification results"""
        details_file = "particle_classifications.json"
        
        classification_record = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "classifications": classifications
        }
        
        try:
            with open(details_file, "a") as f:
                f.write(json.dumps(classification_record) + "\n")
        except Exception as e:
            logger.error(f"Classification details saving error: {e}")
    
    def display_enhanced_results(self, particle_count, class_counts, temperature, humidity):
        """Display comprehensive results on LCD and console"""
        # Prepare display strings
        total = particle_count or 0
        plastics = sum(class_counts.values()) - class_counts.get('unknown', 0)
        temp_str = f"{temperature}C" if temperature is not None else "N/A"
        hum_str = f"{humidity}%" if humidity is not None else "N/A"
        
        # Console output
        print(f"\n{'='*50}")
        print(f"DETECTION RESULTS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        print(f"Total Particles: {total}")
        print(f"Classified Plastics: {plastics}")
        if class_counts:
            print(f"  - Fibers: {class_counts.get('fiber', 0)}")
            print(f"  - Fragments: {class_counts.get('fragment', 0)}")
            print(f"  - Films: {class_counts.get('film', 0)}")
            print(f"  - Pellets: {class_counts.get('pellet', 0)}")
            print(f"  - Unknown: {class_counts.get('unknown', 0)}")
        print(f"Environment: {temp_str}, {hum_str}")
        print(f"{'='*50}")
        
        # LCD display (simplified due to space constraints)
        if self.lcd:
            try:
                self.lcd.clear()
                self.lcd.write_string(f"P:{total} Pl:{plastics}")
                self.lcd.crlf()
                self.lcd.write_string(f"T:{temp_str} H:{hum_str}")
            except Exception as e:
                logger.error(f"LCD display error: {e}")
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        try:
            # Capture image
            frame, image_path = self.capture_and_save_image()
            
            if frame is None:
                logger.warning("Skipping cycle - image capture failed")
                return False
            
            # Detect particles
            particle_count, particle_info, particle_regions = self.detect_particles_advanced(frame)
            
            if particle_count is None:
                logger.warning("Skipping cycle - particle detection failed")
                return False
            
            # Classify particles (if AI is available)
            classifications = []
            class_counts = {'fiber': 0, 'fragment': 0, 'film': 0, 'pellet': 0, 'unknown': 0}
            
            if self.ai_classifier and particle_regions:
                classifications, class_counts = self.classify_particles(particle_regions, image_path)
            
            # Get environmental data
            temperature, humidity = self.get_environmental_data()
            
            # Log all data
            self.log_enhanced_data(
                particle_count, classifications, class_counts, 
                temperature, humidity, image_path
            )
            
            # Display results
            self.display_enhanced_results(particle_count, class_counts, temperature, humidity)
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring cycle error: {e}")
            return False
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        logger.info("Starting enhanced microplastic monitoring system")
        logger.info(f"AI Classification: {'Enabled' if self.ai_classifier else 'Disabled'}")
        logger.info(f"Session ID: {self.session_id}")
        
        try:
            while True:
                cycle_success = self.run_monitoring_cycle()
                
                if not cycle_success:
                    logger.warning("Monitoring cycle failed, continuing...")
                
                # Wait for next cycle
                time.sleep(self.config['REFRESH_INTERVAL'])
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            if self.lcd:
                self.lcd.clear()
                self.lcd.write_string("System Stopped")
        except Exception as e:
            logger.error(f"Monitoring system error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.lcd:
            try:
                self.lcd.clear()
            except:
                pass
        logger.info("System cleanup completed")

def main():
    """Main function to run the integrated system"""
    print("🔬 Enhanced Microplastic Detection & Classification System")
    print("="*60)
    
    # Initialize system
    system = EnhancedMicroplasticSystem()
    
    # Run monitoring
    system.run_continuous_monitoring()

if __name__ == "__main__":
    main()