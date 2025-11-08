# Start with a base image that includes Python
FROM python:3.13.7

# Set the working directory
WORKDIR /app

# Copy all your scripts, the RF model file, and data
# The assumption is your model and scripts are in the same folder as the Dockerfile
COPY . /app

# Install necessary Python packages (e.g., pandas, scikit-learn, scapy)
RUN pip install --no-cache-dir -r requirements.txt

# (Optional: If using Method 2) Install a supervisor
# RUN apt-get update && apt-get install -y supervisor

# Set the entry point to the monitoring loop script
# You might need to change permissions for the packet capture tool (e.g., if using tshark/tcpdump)
RUN chmod +x monitor.sh 

# Command to run the monitoring loop script
# CMD ["/usr/bin/supervisord", "-c", "supervisord.conf"] # For Method 2
CMD ["/bin/bash", "monitor.sh"] # For Method 1