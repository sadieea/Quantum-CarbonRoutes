#!/usr/bin/env python3
import requests
import json
import subprocess
import time
import sys

# Start the server in background
print("Starting FastAPI server...")
server_process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait a bit for server to start
time.sleep(5)

# Test endpoints
try:
    url_base = "http://127.0.0.1:8000"

    # Test quantum-hybrid
    print("Testing /solve/quantum-hybrid...")
    data = {
        "file_path": "c101.csv",
        "vehicle_capacity": 200,
        "p": 5,
        "T": 10,
        "lam1": 0.4,
        "quadratic_weight": 1.0,
        "subproblem_size_limit": 22,
        "cluster_radius": 15.0
    }
    response = requests.post(f"{url_base}/solve/quantum-hybrid", json=data)
    if response.status_code == 200:
        result = response.json()
        print("Quantum-hybrid success:", result["total_vehicles"], "vehicles,", result["total_cost"], "cost")
    else:
        print("Quantum-hybrid failed:", response.status_code, response.text)

    # Test classical-baseline
    print("Testing /solve/classical-baseline...")
    response = requests.post(f"{url_base}/solve/classical-baseline", json=data)
    if response.status_code == 200:
        result = response.json()
        print("Classical-baseline success:", result["total_vehicles"], "vehicles,", result["total_cost"], "cost")
    else:
        print("Classical-baseline failed:", response.status_code, response.text)

except Exception as e:
    print("Error:", e)

finally:
    # Kill the server process
    server_process.terminate()
    server_process.wait()
    print("Server stopped.")
