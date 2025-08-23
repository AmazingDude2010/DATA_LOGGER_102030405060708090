#!/usr/bin/env python3
"""
Test script to run a single detection cycle
"""

from integrated_microplastic_system import EnhancedMicroplasticSystem
import logging

# Set up logging to see everything
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_single_cycle():
    """Test a single monitoring cycle"""
    print("🧪 Testing Single Detection Cycle")
    print("=" * 50)
    
    # Initialize system
    print("1. Initializing system...")
    system = EnhancedMicroplasticSystem()
    
    # Check system status
    print(f"   - AI Classifier: {'✅ Loaded' if system.ai_classifier else '❌ Not Available'}")
    print(f"   - LCD Display: {'✅ Ready' if system.lcd else '⚠️  Not Available'}")
    
    # Run single cycle
    print("\n2. Running detection cycle...")
    print("   - Position sample under microscope")
    print("   - Press Enter to capture and analyze...")
    input()
    
    success = system.run_monitoring_cycle()
    
    if success:
        print("\n✅ Detection cycle completed successfully!")
        print("Check the following files:")
        print("   - microplastic_data.csv (data log)")
        print("   - captured_images/ (raw images)")
        print("   - classified_images/ (AI results)")
    else:
        print("\n❌ Detection cycle failed")
        print("Check camera connection and try again")
    
    # Cleanup
    system.cleanup()

if __name__ == "__main__":
    test_single_cycle()