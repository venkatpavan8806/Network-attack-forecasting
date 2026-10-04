Network Attack Forecasting -- capture agent
============================================

This agent watches the TCP traffic of the computer it runs on, computes the
forecasting model's 30-second window features locally, and sends them to
your account on the website. Packet payloads are never uploaded -- only the
numeric window features and a small sample of packet headers (ports, TCP
flags, TTL, sizes) for the per-host packet view.

1. Install Python 3.9 or newer.
2. In this folder:      pip install -r requirements.txt
3. Packet-capture permission:
     Windows: install Npcap from https://npcap.com (keep "WinPcap API-compatible
              mode" ticked), then open the terminal "as Administrator".
     Linux / macOS: run the agent with sudo.
4. Start it with the token shown on the website when you added the sensor:

     python nadf_agent.py --token nadf_xxxxxxxx

   (Linux/macOS: sudo python3 nadf_agent.py --token nadf_xxxxxxxx)

   Optional:  --list-interfaces       show capture interfaces
              --iface "Wi-Fi"         capture on a specific interface
              --server URL            use a different backend

Leave it running. Within ~30 seconds the sensor shows as "online" on the
website; every remote host that opens connections TO this machine appears
as a live host with a prediction after every window.

To test it, run a port scan against this machine from another computer on
the same network, e.g.:   nmap -sS <this machine's IP>

Stop with Ctrl+C. Revoke a sensor on the website to disable its token.
