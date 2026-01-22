#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["python-osc"]
# ///
"""
Test OSC receiver for RALF Gesture Studio.

Listens for hit messages on port 12000.

Usage: uv run test_osc_receiver.py
"""

import argparse
from datetime import datetime

from pythonosc import dispatcher, osc_server


def handle_message(address, *args):
    """Handle incoming OSC messages."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    args_str = ", ".join(str(a) for a in args) if args else "(no args)"
    print(f"[{timestamp}] {address} → {args_str}")


def main():
    parser = argparse.ArgumentParser(description='Receive test OSC messages from RALF Gesture Studio')
    parser.add_argument('--host', default='127.0.0.1', help='Listen host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=12000, help='Listen port (default: 12000)')
    args = parser.parse_args()

    disp = dispatcher.Dispatcher()
    disp.set_default_handler(handle_message)

    server = osc_server.ThreadingOSCUDPServer((args.host, args.port), disp)

    print(f"Listening for OSC on {args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == '__main__':
    main()
