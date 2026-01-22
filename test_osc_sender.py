#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["python-osc"]
# ///
"""
Test OSC sender for RALF Gesture Studio.

Sends fake skeleton data to port 6448 at ~60fps.

Usage: uv run test_osc_sender.py
"""

import time
import math
import argparse

from pythonosc import udp_client

def main():
    parser = argparse.ArgumentParser(description='Send test OSC messages to RALF Gesture Studio')
    parser.add_argument('--host', default='127.0.0.1', help='Target host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=6448, help='Target port (default: 6448)')
    parser.add_argument('--address', default='/wek/inputs', help='OSC address (default: /wek/inputs)')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second (default: 60)')
    parser.add_argument('--dimensions', type=int, default=4, help='Number of floats per frame (default: 4)')
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.host, args.port)

    print(f"Sending OSC to {args.host}:{args.port}{args.address}")
    print(f"  {args.dimensions} floats per frame at {args.fps} fps")
    print("Press Ctrl+C to stop")
    print()

    frame = 0
    interval = 1.0 / args.fps

    try:
        while True:
            # Generate some oscillating test data
            t = frame * interval
            data = [
                math.sin(t * 2) * 0.5 + 0.5,  # Oscillate 0-1
                math.cos(t * 2) * 0.5 + 0.5,
                math.sin(t * 3) * 0.5 + 0.5,
                math.cos(t * 3) * 0.5 + 0.5,
            ]

            # Extend or trim to requested dimensions
            while len(data) < args.dimensions:
                data.append(math.sin(t * (len(data) + 1)) * 0.5 + 0.5)
            data = data[:args.dimensions]

            client.send_message(args.address, data)

            if frame % args.fps == 0:
                print(f"Sent {frame} frames...")

            frame += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped after {frame} frames")

if __name__ == '__main__':
    main()
