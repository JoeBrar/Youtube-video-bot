
import urllib.request
import re

url = "https://download.pytorch.org/whl/nightly/cu124/torch/"
print(f"Checking {url}...")
try:
    with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')
        matches = re.findall(r'torch-[\d\w\.]+\+cu124-cp313-cp313-win_amd64.whl', html)
        if matches:
            print("Found wheels:")
            for m in matches:
                print(m)
        else:
            print("No cp313 wheels found.")
except Exception as e:
    print(f"Error: {e}")
