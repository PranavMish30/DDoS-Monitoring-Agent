#!/bin/bash
while true
do
    echo "Starting monitoring cycle..."
    
    # 1. Packet Capture (e.g., for 60 seconds)
    python3 ./Agent/pcap.py -i wlan0 -t 60 -o /Packets/output.pcap

    # 2. Preprocessing & Cleaning
    python3 ./Agent/pre-processing.py

    # 3. Feature Generation
    python3 ./Agent/feature-aggregation.py

    # 4. Prediction & Alerting
    python3 ./Agent/prediction.py
    
    # Pause before the next cycle (e.g., 5 seconds)
    echo "Cycle complete. Waiting for 15 seconds..."
    sleep 15
done