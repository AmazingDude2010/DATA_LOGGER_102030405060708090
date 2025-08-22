import csv
import json
import os
import time
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    import Adafruit_DHT
except ImportError:
    Adafruit_DHT = None
try:
    from RPLCD.i2c import CharLCD
except ImportError:
    CharLCD = None

# ---------- SETTINGS ----------
CSV_FILE = "microplastic_data.csv"
JSON_FILE = "microplastic_data.ndjson"  # Efficient: each entry per line
DHT_SENSOR_TYPE = Adafruit_DHT.DHT22 if Adafruit_DHT else None
DHT_PIN = 4
CAMERA_INDEX = 0
LCD_ADDRESS = 0x27
REFRESH_INTERVAL = 30  # seconds

# ---------- INIT LCD (only if available) ----------
def init_lcd():
    if CharLCD:
        try:
            return CharLCD('PCF8574', LCD_ADDRESS)
        except Exception as e:
            print(f"[ERROR] LCD init failed: {e}")
    return None

# ---------- PARTICLE DETECTION ----------
def get_particle_data():
    if not cv2 or not np:
        print("[ERROR] OpenCV/Numpy not available!")
        return None, None
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot access microscope camera.")
        return None, None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Failed to capture image.")
        return None, None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        particle_count = len(contours)
        # TODO: Replace with real AI classification for plastics
        plastic_count = particle_count
        return particle_count, plastic_count
    except Exception as e:
        print(f"[ERROR] Image processing: {e}")
        return None, None

# ---------- GET ENVIRONMENTAL DATA ----------
def get_environmental_data():
    if not Adafruit_DHT or not DHT_SENSOR_TYPE:
        print("[ERROR] DHT Library/Sensor not available!")
        return None, None
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR_TYPE, DHT_PIN)
    if humidity is None or temperature is None:
        print("[WARN] Failed to read from DHT22 sensor")
        return None, None
    return round(temperature, 2), round(humidity, 2)

# ---------- LOG DATA (Atomic, NDJSON, Efficient) ----------
def log_data(particle_count, plastic_count, temperature, humidity):
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    data_entry = {
        "timestamp": timestamp,
        "particle_count": particle_count,
        "plastic_count": plastic_count,
        "temperature": temperature,
        "humidity": humidity
    }

    # CSV: write header only once, use buffering
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_entry)
    except Exception as e:
        print(f"[ERROR] Writing CSV: {e}")

    # NDJSON: append one JSON object per line for efficiency
    try:
        with open(JSON_FILE, "a") as jf:
            jf.write(json.dumps(data_entry) + "\n")
    except Exception as e:
        print(f"[ERROR] Writing NDJSON: {e}")

    print(f"[LOG] Data saved at {timestamp}")

# ---------- DISPLAY DATA ON LCD ----------
def display_results(lcd, particle_count, plastic_count, temperature, humidity):
    # Show values or N/A if missing
    p = str(particle_count) if particle_count is not None else "N/A"
    pl = str(plastic_count) if plastic_count is not None else "N/A"
    t = f"{temperature}C" if temperature is not None else "N/A"
    h = f"{humidity}%" if humidity is not None else "N/A"

    if lcd:
        try:
            lcd.clear()
            lcd.write_string(f"P:{p} Pl:{pl}")
            lcd.crlf()
            lcd.write_string(f"T:{t} H:{h}")
        except Exception as e:
            print(f"[ERROR] LCD display: {e}")
    print(f"[DISPLAY] P:{p} Pl:{pl} | T:{t} H:{h}")

# ---------- MAIN LOOP ----------
def main():
    lcd = init_lcd()
    print("[INIT] Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            particle_count, plastic_count = get_particle_data()
            temperature, humidity = get_environmental_data()

            # Only log/display if all data was captured (skip partial)
            if None not in (particle_count, plastic_count, temperature, humidity):
                log_data(particle_count, plastic_count, temperature, humidity)
                display_results(lcd, particle_count, plastic_count, temperature, humidity)
            else:
                print("[WARN] Incomplete data, skipping log/display.")

            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        if lcd:
            lcd.clear()
        print("\n[EXIT] Stopped live monitoring.")
    except Exception as ex:
        print(f"[EXCEPTION] {ex}")
        if lcd:
            lcd.clear()

if __name__ == "__main__":
    main()
