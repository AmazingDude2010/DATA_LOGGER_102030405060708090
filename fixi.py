#!/usr/bin/env python3
"""
Enhanced Assistive Technology Projects for Raspberry Pi
High School Hackathon - Professional Implementation
Author: Your Team Name
"""

import RPi.GPIO as GPIO
import time
import threading
import json
import os
import sys
from datetime import datetime
import logging

# Enhanced imports for different projects
try:
    import pyaudio
    import wave
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    print("Voice libraries not installed. Voice features disabled.")
    VOICE_AVAILABLE = False

try:
    import cv2
    import numpy as np
    VISION_AVAILABLE = True
except ImportError:
    print("Computer vision libraries not available. Vision features disabled.")
    VISION_AVAILABLE = False

try:
    import requests
    import smtplib
    from email.mime.text import MIMEText
    COMMUNICATION_AVAILABLE = True
except ImportError:
    print("Communication libraries not available. Network features disabled.")
    COMMUNICATION_AVAILABLE = False

# ========== ENHANCED SYSTEM CONFIGURATION ==========
class Config:
    """System configuration with file persistence"""
    
    def __init__(self):
        self.config_file = 'assistive_tech_config.json'
        self.default_config = {
            'current_mode': 'menu',
            'sensitivity': 50,
            'volume': 70,
            'emergency_contacts': [
                {'name': 'Emergency Contact 1', 'phone': '+1234567890', 'email': 'emergency1@email.com'},
                {'name': 'Emergency Contact 2', 'phone': '+1234567891', 'email': 'emergency2@email.com'}
            ],
            'voice_notes_path': '/home/pi/voice_notes/',
            'detection_threshold': 0.7,
            'auto_save': True,
            'debug_mode': True
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in self.default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def save_config(self, config=None):
        """Save current configuration to file"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        self.save_config()

# ========== GPIO PIN DEFINITIONS ==========
class PinConfig:
    """Centralized GPIO pin configuration"""
    
    # User interface pins
    BUTTON_1 = 18       # Main action button
    BUTTON_2 = 19       # Select/menu button
    STATUS_LED = 20     # Main status LED
    BUZZER = 21         # Status buzzer/speaker
    
    # Project-specific pins
    LED_RED = 16        # Alert/emergency LED
    LED_GREEN = 12      # GPS/system ready LED  
    LED_BLUE = 13       # Activity/recording LED
    
    # Sensor pins
    VIBRATION_1 = 22    # Vibrating motor 1
    VIBRATION_2 = 23    # Vibrating motor 2
    
    # Additional I/O
    RELAY_1 = 24        # For external devices
    RELAY_2 = 25        # For external devices

# ========== ENHANCED SYSTEM MANAGER ==========
class AssistiveTechSystem:
    """Main system coordinator with enhanced features"""
    
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        self.setup_gpio()
        self.current_mode = 'menu'
        self.system_ready = False
        self.emergency_active = False
        self.running = True
        
        # Initialize subsystems
        self.voice_system = None
        self.fire_alert = None  
        self.emergency_beacon = None
        self.menu_system = MenuSystem(self)
        
        # System monitoring
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.uptime_start = time.time()
        
        self.initialize_projects()
    
    def setup_logging(self):
        """Configure system logging"""
        log_level = logging.DEBUG if self.config.get('debug_mode') else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('assistive_tech.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Assistive Technology System Starting...")
    
    def setup_gpio(self):
        """Initialize GPIO with proper cleanup"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup input pins with pull-up resistors
        GPIO.setup(PinConfig.BUTTON_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PinConfig.BUTTON_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup output pins
        output_pins = [PinConfig.STATUS_LED, PinConfig.BUZZER, 
                      PinConfig.LED_RED, PinConfig.LED_GREEN, PinConfig.LED_BLUE,
                      PinConfig.VIBRATION_1, PinConfig.VIBRATION_2]
        
        for pin in output_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup button interrupts with debouncing
        GPIO.add_event_detect(PinConfig.BUTTON_1, GPIO.FALLING, 
                             callback=self.button_1_pressed, bouncetime=200)
        GPIO.add_event_detect(PinConfig.BUTTON_2, GPIO.FALLING,
                             callback=self.button_2_pressed, bouncetime=200)
        
        self.logger.info("GPIO initialized successfully")
    
    def initialize_projects(self):
        """Initialize available project modules"""
        try:
            if VOICE_AVAILABLE:
                self.voice_system = VoiceNoteSystem(self)
                self.logger.info("Voice Note System initialized")
            
            self.fire_alert = FireAlertSystem(self)
            self.logger.info("Fire Alert System initialized")
            
            if COMMUNICATION_AVAILABLE:
                self.emergency_beacon = EmergencyBeacon(self)
                self.logger.info("Emergency Beacon initialized")
            
            self.system_ready = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize projects: {e}")
            self.system_ready = False
    
    def button_1_pressed(self, channel):
        """Handle button 1 press - context sensitive"""
        self.play_button_feedback()
        
        if self.current_mode == 'menu':
            self.menu_system.navigate_next()
        elif self.current_mode == 'voice_notes':
            if self.voice_system:
                self.voice_system.toggle_recording()
        elif self.current_mode == 'emergency':
            self.trigger_emergency()
        else:
            self.return_to_menu()
    
    def button_2_pressed(self, channel):
        """Handle button 2 press - select/confirm"""
        self.play_button_feedback()
        
        if self.current_mode == 'menu':
            self.menu_system.select_current()
        elif self.current_mode == 'voice_notes':
            if self.voice_system:
                self.voice_system.play_latest_note()
        elif self.current_mode == 'fire_alert':
            self.fire_alert.test_alert()
        elif self.current_mode == 'emergency':
            self.emergency_beacon.send_status_update()
    
    def play_button_feedback(self):
        """Provide audio feedback for button presses"""
        GPIO.output(PinConfig.BUZZER, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(PinConfig.BUZZER, GPIO.LOW)
    
    def switch_mode(self, new_mode):
        """Switch between different operational modes"""
        self.logger.info(f"Switching from {self.current_mode} to {new_mode}")
        
        # Cleanup current mode
        self.cleanup_current_mode()
        
        # Switch to new mode
        self.current_mode = new_mode
        self.config.set('current_mode', new_mode)
        
        # Initialize new mode
        if new_mode == 'voice_notes' and self.voice_system:
            self.voice_system.activate()
        elif new_mode == 'fire_alert':
            self.fire_alert.activate()
        elif new_mode == 'emergency' and self.emergency_beacon:
            self.emergency_beacon.activate()
        
        self.update_status_leds()
    
    def cleanup_current_mode(self):
        """Clean up resources from current mode"""
        if self.current_mode == 'voice_notes' and self.voice_system:
            self.voice_system.deactivate()
        elif self.current_mode == 'fire_alert':
            self.fire_alert.deactivate()
        elif self.current_mode == 'emergency' and self.emergency_beacon:
            self.emergency_beacon.deactivate()
    
    def update_status_leds(self):
        """Update LED indicators based on current state"""
        # Turn off all LEDs first
        GPIO.output(PinConfig.LED_RED, GPIO.LOW)
        GPIO.output(PinConfig.LED_GREEN, GPIO.LOW)
        GPIO.output(PinConfig.LED_BLUE, GPIO.LOW)
        
        # Set LEDs based on mode and status
        if self.current_mode == 'menu':
            GPIO.output(PinConfig.STATUS_LED, GPIO.HIGH)
        elif self.current_mode == 'voice_notes':
            GPIO.output(PinConfig.LED_BLUE, GPIO.HIGH)
        elif self.current_mode == 'fire_alert':
            GPIO.output(PinConfig.LED_RED, GPIO.HIGH)
        elif self.current_mode == 'emergency':
            GPIO.output(PinConfig.LED_GREEN, GPIO.HIGH)
        
        if self.emergency_active:
            # Flash red LED for emergency
            threading.Thread(target=self.flash_emergency_led, daemon=True).start()
    
    def flash_emergency_led(self):
        """Flash emergency LED when emergency is active"""
        while self.emergency_active:
            GPIO.output(PinConfig.LED_RED, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(PinConfig.LED_RED, GPIO.LOW)
            time.sleep(0.5)
    
    def trigger_emergency(self):
        """Trigger emergency alert system"""
        self.emergency_active = True
        self.logger.critical("EMERGENCY TRIGGERED!")
        
        if self.emergency_beacon:
            self.emergency_beacon.send_emergency_alert()
        
        # Activate all alert mechanisms
        self.activate_vibration_alert()
        self.update_status_leds()
    
    def activate_vibration_alert(self):
        """Activate vibration motors for alerts"""
        def vibrate():
            for _ in range(10):
                GPIO.output(PinConfig.VIBRATION_1, GPIO.HIGH)
                GPIO.output(PinConfig.VIBRATION_2, GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output(PinConfig.VIBRATION_1, GPIO.LOW)
                GPIO.output(PinConfig.VIBRATION_2, GPIO.LOW)
                time.sleep(0.2)
        
        threading.Thread(target=vibrate, daemon=True).start()
    
    def return_to_menu(self):
        """Return to main menu from any mode"""
        self.switch_mode('menu')
        self.menu_system.show_menu()
    
    def run(self):
        """Main system loop"""
        self.logger.info("System ready - starting main loop")
        self.menu_system.show_welcome()
        
        try:
            while self.running:
                # System heartbeat
                self.last_heartbeat = time.time()
                
                # Run mode-specific updates
                if self.current_mode == 'fire_alert':
                    self.fire_alert.monitor_sound()
                elif self.current_mode == 'emergency' and self.emergency_beacon:
                    self.emergency_beacon.update_location()
                
                # System monitoring
                if time.time() % 60 < 1:  # Every minute
                    self.log_system_status()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.shutdown()
    
    def log_system_status(self):
        """Log system status periodically"""
        uptime = time.time() - self.uptime_start
        self.logger.info(f"Status: Mode={self.current_mode}, Uptime={uptime:.1f}s, Errors={self.error_count}")
    
    def shutdown(self):
        """Clean system shutdown"""
        self.logger.info("System shutting down...")
        self.running = False
        
        # Cleanup all modes
        self.cleanup_current_mode()
        
        # Turn off all outputs
        for pin in [PinConfig.STATUS_LED, PinConfig.LED_RED, PinConfig.LED_GREEN, 
                   PinConfig.LED_BLUE, PinConfig.VIBRATION_1, PinConfig.VIBRATION_2]:
            GPIO.output(pin, GPIO.LOW)
        
        GPIO.cleanup()
        self.logger.info("Shutdown complete")

# ========== ENHANCED MENU SYSTEM ==========
class MenuSystem:
    """Interactive menu system with voice and visual feedback"""
    
    def __init__(self, parent_system):
        self.system = parent_system
        self.current_selection = 0
        self.menu_items = [
            {'name': 'Voice Notes', 'mode': 'voice_notes', 'available': VOICE_AVAILABLE},
            {'name': 'Fire Alert', 'mode': 'fire_alert', 'available': True},
            {'name': 'Emergency Beacon', 'mode': 'emergency', 'available': COMMUNICATION_AVAILABLE},
            {'name': 'System Settings', 'mode': 'settings', 'available': True}
        ]
    
    def show_welcome(self):
        """Display welcome message and instructions"""
        print("\n" + "="*50)
        print("🚀 ASSISTIVE TECHNOLOGY MULTI-TOOL v3.0")
        print("   Raspberry Pi Enhanced Edition")
        print("="*50)
        print("🎯 Developed for High School Hackathon")
        print("💡 Press Button 1: Navigate | Button 2: Select")
        print("⚡ Type 'help' for voice commands")
        print("="*50)
        self.show_menu()
    
    def show_menu(self):
        """Display current menu options"""
        print("\n📋 MAIN MENU:")
        print("-" * 30)
        
        for i, item in enumerate(self.menu_items):
            marker = ">>> " if i == self.current_selection else "    "
            status = "✅" if item['available'] else "❌"
            print(f"{marker}{i+1}. {item['name']} {status}")
        
        print("-" * 30)
        print(f"Selected: {self.menu_items[self.current_selection]['name']}")
        print("Button 1: Next | Button 2: Select | 'menu' to return here")
    
    def navigate_next(self):
        """Move to next menu item"""
        self.current_selection = (self.current_selection + 1) % len(self.menu_items)
        self.show_menu()
    
    def select_current(self):
        """Select current menu item"""
        selected_item = self.menu_items[self.current_selection]
        
        if not selected_item['available']:
            print(f"❌ {selected_item['name']} is not available on this system")
            return
        
        if selected_item['mode'] == 'settings':
            self.show_settings()
        else:
            print(f"🚀 Launching {selected_item['name']}...")
            self.system.switch_mode(selected_item['mode'])
    
    def show_settings(self):
        """Show system configuration options"""
        print("\n⚙️  SYSTEM SETTINGS:")
        print("-" * 30)
        print(f"Sensitivity: {self.system.config.get('sensitivity')}%")
        print(f"Volume: {self.system.config.get('volume')}%") 
        print(f"Debug Mode: {self.system.config.get('debug_mode')}")
        print(f"Auto Save: {self.system.config.get('auto_save')}")
        print("-" * 30)
        print("Use voice commands to change settings")

# ========== PROJECT 1: ENHANCED VOICE NOTE SYSTEM ==========
class VoiceNoteSystem:
    """Professional voice recording and recognition system"""
    
    def __init__(self, parent_system):
        self.system = parent_system
        self.is_recording = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.notes_directory = self.system.config.get('voice_notes_path', '/home/pi/voice_notes/')
        
        # Create notes directory if it doesn't exist
        os.makedirs(self.notes_directory, exist_ok=True)
        
        # Configure TTS
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', self.system.config.get('volume', 70) / 100)
    
    def activate(self):
        """Activate voice notes mode"""
        print("\n🎙️  VOICE NOTES ACTIVATED")
        print("Button 1: Start/Stop Recording")
        print("Button 2: Play Latest Note")
        print("Say 'list notes' to hear all notes")
        
        # Calibrate microphone for ambient noise
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("✅ Microphone calibrated")
    
    def toggle_recording(self):
        """Start or stop voice recording"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording voice note"""
        if self.is_recording:
            return
        
        self.is_recording = True
        GPIO.output(PinConfig.LED_BLUE, GPIO.HIGH)
        print("🔴 Recording started... Press Button 1 to stop")
        
        # Start recording in separate thread
        threading.Thread(target=self._record_audio, daemon=True).start()
    
    def stop_recording(self):
        """Stop recording voice note"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        GPIO.output(PinConfig.LED_BLUE, GPIO.LOW)
        print("⏹️  Recording stopped")
    
    def _record_audio(self):
        """Internal method to handle audio recording"""
        try:
            with self.microphone as source:
                print("Listening...")
                # Record with timeout to prevent infinite recording
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=30)
            
            if self.is_recording:  # Check if we're still supposed to be recording
                # Save audio file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"note_{timestamp}.wav"
                filepath = os.path.join(self.notes_directory, filename)
                
                with open(filepath, "wb") as f:
                    f.write(audio.get_wav_data())
                
                print(f"📁 Note saved: {filename}")
                
                # Try to transcribe
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"📝 Transcription: {text}")
                    
                    # Save transcription
                    txt_file = filepath.replace('.wav', '.txt')
                    with open(txt_file, 'w') as f:
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Transcription: {text}\n")
                    
                except sr.UnknownValueError:
                    print("Could not transcribe audio")
                except sr.RequestError as e:
                    print(f"Transcription service error: {e}")
        
        except Exception as e:
            print(f"Recording error: {e}")
            self.system.logger.error(f"Voice recording error: {e}")
    
    def play_latest_note(self):
        """Play the most recent voice note"""
        try:
            # Find most recent .wav file
            wav_files = [f for f in os.listdir(self.notes_directory) if f.endswith('.wav')]
            if not wav_files:
                print("📭 No voice notes found")
                self.tts_engine.say("No voice notes found")
                self.tts_engine.runAndWait()
                return
            
            # Get most recent file
            latest_file = max(wav_files, key=lambda x: os.path.getctime(
                os.path.join(self.notes_directory, x)))
            
            print(f"🔊 Playing: {latest_file}")
            
            # Try to read transcription and speak it
            txt_file = latest_file.replace('.wav', '.txt')
            txt_path = os.path.join(self.notes_directory, txt_file)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    content = f.read()
                    if 'Transcription:' in content:
                        transcription = content.split('Transcription:')[1].strip()
                        self.tts_engine.say(f"Your note says: {transcription}")
                        self.tts_engine.runAndWait()
                        return
            
            # Fallback: just announce that we're playing the note
            self.tts_engine.say("Playing your latest voice note")
            self.tts_engine.runAndWait()
            
            # Here you could add actual audio playback with pygame or similar
            
        except Exception as e:
            print(f"Playback error: {e}")
            self.system.logger.error(f"Voice playback error: {e}")
    
    def list_all_notes(self):
        """List all saved voice notes"""
        try:
            txt_files = [f for f in os.listdir(self.notes_directory) if f.endswith('.txt')]
            
            if not txt_files:
                self.tts_engine.say("You have no saved voice notes")
                self.tts_engine.runAndWait()
                return
            
            print(f"📋 Found {len(txt_files)} voice notes:")
            self.tts_engine.say(f"You have {len(txt_files)} voice notes")
            
            for i, filename in enumerate(sorted(txt_files)[-5:]):  # Last 5 notes
                filepath = os.path.join(self.notes_directory, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    if 'Transcription:' in content:
                        transcription = content.split('Transcription:')[1].strip()
                        print(f"{i+1}. {transcription[:50]}...")
            
            self.tts_engine.runAndWait()
            
        except Exception as e:
            print(f"Error listing notes: {e}")
            self.system.logger.error(f"Voice notes listing error: {e}")
    
    def deactivate(self):
        """Deactivate voice notes mode"""
        if self.is_recording:
            self.stop_recording()
        GPIO.output(PinConfig.LED_BLUE, GPIO.LOW)

# ========== PROJECT 2: ENHANCED FIRE ALERT SYSTEM ==========
class FireAlertSystem:
    """Advanced fire/smoke detection with pattern recognition"""
    
    def __init__(self, parent_system):
        self.system = parent_system
        self.monitoring = False
        self.alert_active = False
        self.sound_history = []
        self.detection_threshold = self.system.config.get('detection_threshold', 0.7)
        
        # Smoke alarm pattern characteristics
        self.typical_patterns = [
            {'frequency_range': (3000, 4000), 'beep_duration': 0.5, 'gap_duration': 0.5, 'repeats': 3},
            {'frequency_range': (2800, 3200), 'beep_duration': 0.3, 'gap_duration': 0.7, 'repeats': 4}
        ]
    
    def activate(self):
        """Activate fire alert monitoring"""
        print("\n🔥 FIRE ALERT SYSTEM ACTIVATED")
        print("Monitoring for smoke detector patterns...")
        print("Button 1: Return to menu")
        print("Button 2: Test alert system")
        print("Say 'test fire alert' to trigger test")
        
        self.monitoring = True
        GPIO.output(PinConfig.LED_RED, GPIO.HIGH)
        
        # Start monitoring thread
        threading.Thread(target=self._monitor_continuously, daemon=True).start()
    
    def monitor_sound(self):
        """Monitor sound levels and patterns"""
        if not self.monitoring:
            return
        
        try:
            # Simulate sound level detection
            # In a real implementation, you'd use pyaudio to capture and analyze audio
            import random
            sound_level = random.randint(0, 100)  # Simulated sound level
            
            # Store sound history for pattern analysis
            timestamp = time.time()
            self.sound_history.append({'timestamp': timestamp, 'level': sound_level})
            
            # Keep only recent history (last 30 seconds)
            cutoff_time = timestamp - 30
            self.sound_history = [s for s in self.sound_history if s['timestamp'] > cutoff_time]
            
            # Analyze for alarm patterns
            if self.detect_alarm_pattern():
                if not self.alert_active:
                    self.trigger_alert()
        
        except Exception as e:
            self.system.logger.error(f"Sound monitoring error: {e}")
    
    def _monitor_continuously(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            self.monitor_sound()
            time.sleep(0.1)
    
    def detect_alarm_pattern(self):
        """Analyze sound history for smoke alarm patterns"""
        if len(self.sound_history) < 10:
            return False
        
        # Simple pattern detection: look for repeated high-level sounds
        recent_sounds = self.sound_history[-10:]
        high_sounds = [s for s in recent_sounds if s['level'] > 80]
        
        if len(high_sounds) >= 5:
            # Check if sounds are evenly spaced (indicating alarm pattern)
            if len(high_sounds) >= 2:
                intervals = []
                for i in range(1, len(high_sounds)):
                    interval = high_sounds[i]['timestamp'] - high_sounds[i-1]['timestamp']
                    intervals.append(interval)
                
                # Smoke alarms typically beep every 0.5-1.5 seconds
                avg_interval = sum(intervals) / len(intervals)
                if 0.3 <= avg_interval <= 2.0:
                    return True
        
        return False
    
    def trigger_alert(self):
        """Trigger fire alert response"""
        self.alert_active = True
        self.system.logger.critical("FIRE/SMOKE ALARM DETECTED!")
        
        print("\n🚨 FIRE/SMOKE ALARM DETECTED!")
        print("Activating all alert mechanisms...")
        
        # Activate vibration alert
        self.system.activate_vibration_alert()
        
        # Flash all LEDs
        self.flash_all_leds()
        
        # If TTS is available, announce emergency
        if hasattr(self.system.voice_system, 't
