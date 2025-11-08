#!/usr/bin/env python3
from scapy.all import sniff, wrpcap
import argparse
import time
import os

def capture_packets(interface: str, duration: int, output: str):
    """
    Capture packets for a specified duration and save to a pcap file.
    """
    print(f"[+] Starting packet capture on interface '{interface}' for {duration} seconds...")
    start_time = time.time()

    # Capture packets for given time
    packets = sniff(iface=interface, timeout=duration)

    # Write packets to file
    wrpcap(output, packets)

    print(f"[+] Capture complete. {len(packets)} packets saved to '{output}'")
    print(f"[+] Duration: {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scapy timed packet capture")
    parser.add_argument("-i", "--interface", required=True, help="Network interface (e.g., eth0, wlan0)")
    parser.add_argument("-t", "--time", type=int, required=True, help="Capture duration in seconds")
    parser.add_argument("-o", "--output", default="capture.pcap", help="Output pcap filename")

    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    capture_packets(args.interface, args.time, args.output)
