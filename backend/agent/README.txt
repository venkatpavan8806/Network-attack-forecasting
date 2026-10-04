Network Attack Forecasting -- capture agent
============================================

This lets the website's "Live Capture" capture the real traffic of THIS
computer. Only numeric window features and packet headers (ports, TCP
flags, sizes) are sent to your account -- never packet contents.

1. Install Python 3.9 or newer.
2. In this folder run:     pip install -r requirements.txt
3. Capture permission:
     Windows: install Npcap from https://npcap.com, then open the terminal
              "as Administrator".
     Linux / macOS: use sudo in the next step.
4. Start the agent with the token shown on the website:
     python nadf_agent.py --token nadf_xxxxxxxx
   (Linux/macOS: sudo python3 nadf_agent.py --token nadf_xxxxxxxx)
5. Leave it running. On the website's Live Capture panel pick an interface
   and press "Start live capture". "Stop capture" pauses it.

Try it by scanning this computer from another device on the same network,
e.g.  nmap -sS <this computer's IP>
