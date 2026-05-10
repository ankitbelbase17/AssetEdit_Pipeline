import subprocess
import re
import os
import time
import shutil
import urllib.request

def ensure_cloudflared_windows():
    """Auto-downloads cloudflared.exe if it doesn't exist locally."""
    if shutil.which("cloudflared") is not None:
        return "cloudflared"
        
    local_binary = "cloudflared.exe"
    if os.path.exists(local_binary):
        return local_binary
        
    print("\n[Setup] 'cloudflared' not found globally on Windows.")
    print("[Setup] Auto-downloading portable Windows executable...")
    try:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        urllib.request.urlretrieve(url, local_binary)
        print("[Setup] Download complete and ready to tunnel!\n")
        return local_binary
    except Exception as e:
        print(f"[Setup Error] Failed to download cloudflared.exe: {e}")
        return "cloudflared"  # Fallback

def main():
    print("====== FRONTEND TUNNEL AUTO-UPDATER ======")
    print("Starting Cloudflared Tunnel for local port 8000...")
    
    binary_path = ensure_cloudflared_windows()
    
    # Start cloudflared
    try:
        proc = subprocess.Popen(
            [binary_path, "tunnel", "--url", "http://127.0.0.1:8000"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
    except FileNotFoundError:
        print("ERROR: 'cloudflared' is not installed or not in PATH.")
        print("Please install cloudflared on windows to use this script.")
        return

    url_found = False
    
    print("Waiting for Cloudflare to assign a TryCloudflare URL...")
    
    for line in proc.stdout:
        # Cloudflare outputs progress, we hunt for the URL
        if not url_found:
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                tunnel_url = match.group(0)
                print("\n" + "★" * 60)
                print(f"  SUCCESS! FRONTEND PUBLIC URL ACQUIRED: ")
                print(f"  -->  {tunnel_url}  <--")
                print("★" * 60 + "\n")
                
                # Auto-update config.py
                config_path = os.path.join("project_showcase", "config.py")
                
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Regex to replace the FRONTEND_PUBLIC_URL assignment safely
                    new_content = re.sub(
                        r'FRONTEND_PUBLIC_URL\s*=\s*".*"',
                        f'FRONTEND_PUBLIC_URL = "{tunnel_url}"',
                        content
                    )
                    
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                        
                    print(f"✅ Successfully injected {tunnel_url} into project_showcase/config.py!")
                    print("✅ Django will automatically detect this change and hot-reload.")
                else:
                    print(f"⚠️ Could not find {config_path}. Are you running this from the Project-Display root?")
                    
                url_found = True
                print("\n[Tunnel is active. Do not close this terminal.]")

if __name__ == '__main__':
    main()
