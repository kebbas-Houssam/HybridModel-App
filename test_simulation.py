# test_simulation.py
# Safe ransomware I/O pattern simulation

import os
import time
import random
from pathlib import Path

class SafeSimulator:
    def __init__(self):
        self.test_dir = Path("./test_simulation")
        self.test_dir.mkdir(exist_ok=True)
        self.running = False
    
    def create_test_files(self, count=50):
        """Create test files"""
        print(f"Creating {count} test files...")
        for i in range(count):
            file_path = self.test_dir / f"test_{i:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(f"Test file {i} - " + "x" * random.randint(100, 1000))
        print("✓ Test files created")
    
    def simulate_ransomware(self, duration=60):
        """Simulate ransomware I/O patterns"""
        print(f"Starting simulation for {duration} seconds...")
        print("This will create suspicious I/O patterns (read-then-write)")
        
        self.running = True
        start_time = time.time()
        
        files = list(self.test_dir.glob("*.txt"))
        
        while self.running and (time.time() - start_time) < duration:
            # Select random files
            selected = random.sample(files, min(5, len(files)))
            
            for file_path in selected:
                try:
                    # Read file (suspicious: reading before writing)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Write back modified content (simulate encryption)
                    with open(file_path, 'w') as f:
                        f.write("ENCRYPTED: " + content[:100])
                    
                    time.sleep(0.01)  # Brief pause
                except:
                    continue
            
            time.sleep(random.uniform(0.1, 0.3))  # Burst pattern
        
        print("Simulation completed")
    
    def cleanup(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        print("✓ Cleanup completed")

if __name__ == "__main__":
    simulator = SafeSimulator()
    try:
        simulator.create_test_files(50)
        simulator.simulate_ransomware(60)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        simulator.running = False
        response = input("Clean up test files? (y/n): ")
        if response.lower() == 'y':
            simulator.cleanup()
