# Mock classes for development environment
class MockDHT:
    DHT22 = "DHT22"
    
    @staticmethod
    def read_retry(sensor_type, pin):
        # Return mock temperature and humidity values
        import random
        humidity = round(random.uniform(40, 70), 1)
        temperature = round(random.uniform(20, 30), 1)
        return humidity, temperature

class MockCharLCD:
    def __init__(self, i2c_expander, address, port=1, cols=20, rows=4):
        self.cols = cols
        self.rows = rows
        print(f"Mock LCD initialized: {cols}x{rows} at address {address}")
    
    def write_string(self, text):
        print(f"LCD Display: {text}")
    
    def clear(self):
        print("LCD cleared")
    
    def cursor_pos(self, row, col):
        print(f"Cursor moved to row {row}, col {col}")
    
    def close(self, clear=False):
        print("LCD connection closed")

# Your original import logic with mocking fallback
try:
    import Adafruit_DHT
except ImportError:
    print("[WARN] DHT sensor library not available - using mock")
    Adafruit_DHT = MockDHT()

try:
    from RPLCD.i2c import CharLCD
except ImportError:
    print("[WARN] LCD library not available - using mock")
    CharLCD = MockCharLCD

# Now your code can continue normally
# Example usage:
if __name__ == "__main__":
    # DHT sensor example
    humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, 4)
    if humidity is not None and temperature is not None:
        print(f'Temperature: {temperature:.1f}°C  Humidity: {humidity:.1f}%')
    
    # LCD example
    try:
        lcd = CharLCD('PCF8574', 0x27)
        lcd.write_string("Hello World!")
        lcd.close(clear=True)
    except Exception as e:
        print(f"LCD error: {e}")