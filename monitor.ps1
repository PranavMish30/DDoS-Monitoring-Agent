# Continuous monitoring loop for a Windows environment
# Save this file as 'monitor_cycle.ps1'

# The path to your Python executable might need to be 'python' instead of 'python3' 
# depending on your Windows setup.
$PythonExec = "python" 

while ($true) {
    Write-Host "Starting monitoring cycle..."
    
    # 1. Packet Capture (e.g., for 60 seconds)
    # The '-i Wi-Fi' parameter assumes your pcap.py can use this interface name on Windows.
    & $PythonExec ".\Agent\pcap.py" -i "Wi-Fi" -t 60 -o ".\Packets\output.pcap"

    # Check the exit code of the last command ($LASTEXITCODE). 
    # If it's not 0, the Python script failed.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error in pcap.py. Skipping remaining steps and continuing loop." -ForegroundColor Red
        # Pause briefly to prevent rapid looping on continuous failure
        Start-Sleep -Seconds 5
        continue
    }

    # 2. Preprocessing & Cleaning
    & $PythonExec ".\Agent\pre-processing.py"

    # 3. Feature Generation
    & $PythonExec ".\Agent\feature-aggregation.py"

    # 4. Prediction & Alerting
    & $PythonExec ".\Agent\prediction.py"
    
    # Pause before the next cycle (e.g., 15 seconds)
    Write-Host "Cycle complete. Waiting for 15 seconds..."
    Start-Sleep -Seconds 15
}