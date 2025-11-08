from scapy.all import rdpcap, TCP, IP
import pandas as pd
from collections import defaultdict
import numpy as np

def extract_flow_features(pcap_file):
    packets = rdpcap(pcap_file)
    flows = defaultdict(lambda: {
        "timestamps": [],
        "lengths_fwd": [],
        "lengths_bwd": [],
        "tcp_flags": {
            "FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0, "ECE": 0
        },
        "fwd_packets": 0,
        "bwd_packets": 0
    })

    for pkt in packets:
        if IP not in pkt or TCP not in pkt:
            continue
        
        ip = pkt[IP]
        tcp = pkt[TCP]
        proto = ip.proto
        src, dst = ip.src, ip.dst
        sport, dport = tcp.sport, tcp.dport

        # Define a flow key (bidirectional)
        flow_fwd = (src, sport, dst, dport, proto)
        flow_bwd = (dst, dport, src, sport, proto)

        direction = "fwd"
        if flow_bwd in flows:
            flow_key = flow_bwd
            direction = "bwd"
        else:
            flow_key = flow_fwd

        # Update flow data
        f = flows[flow_key]
        f["timestamps"].append(pkt.time)
        f["tcp_flags"]["FIN"] += int(tcp.flags & 0x01 != 0)
        f["tcp_flags"]["SYN"] += int(tcp.flags & 0x02 != 0)
        f["tcp_flags"]["RST"] += int(tcp.flags & 0x04 != 0)
        f["tcp_flags"]["PSH"] += int(tcp.flags & 0x08 != 0)
        f["tcp_flags"]["ACK"] += int(tcp.flags & 0x10 != 0)
        f["tcp_flags"]["URG"] += int(tcp.flags & 0x20 != 0)
        f["tcp_flags"]["ECE"] += int(tcp.flags & 0x40 != 0)
        
        pkt_len = len(pkt)
        if direction == "fwd":
            f["fwd_packets"] += 1
            f["lengths_fwd"].append(pkt_len)
        else:
            f["bwd_packets"] += 1
            f["lengths_bwd"].append(pkt_len)

    # Convert flows to DataFrame
    data = []
    for (src, sport, dst, dport, proto), f in flows.items():
        timestamps = f["timestamps"]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        data.append({
            "Flow ID": f"{src}-{sport}-{dst}-{dport}-{proto}",
            "Source IP": src,
            "Destination IP": dst,
            "Source Port": sport,
            "Destination Port": dport,
            "Protocol": proto,
            "Timestamp": min(timestamps),
            "Flow Duration": duration,
            "FIN Flag Count": f["tcp_flags"]["FIN"],
            "SYN Flag Count": f["tcp_flags"]["SYN"],
            "RST Flag Count": f["tcp_flags"]["RST"],
            "PSH Flag Count": f["tcp_flags"]["PSH"],
            "ACK Flag Count": f["tcp_flags"]["ACK"],
            "URG Flag Count": f["tcp_flags"]["URG"],
            "ECE Flag Count": f["tcp_flags"]["ECE"],
            "Total Fwd Packets": f["fwd_packets"],
            "Total Backward Packets": f["bwd_packets"],
            "Fwd Packet Length Max": np.max(f["lengths_fwd"]) if f["lengths_fwd"] else 0,
            "Fwd Packet Length Mean": np.mean(f["lengths_fwd"]) if f["lengths_fwd"] else 0,
            "Fwd Packet Length Std": np.std(f["lengths_fwd"]) if f["lengths_fwd"] else 0,
        })

    return pd.DataFrame(data)

df = extract_flow_features(r".\Packets\output.pcap")
df.columns = df.columns.astype(str).str.strip().str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df.to_csv(r".\Packets\packet-features.csv",index=False)

