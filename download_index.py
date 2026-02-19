
import urllib.request

url = "https://download.pytorch.org/whl/nightly/cu124/torch/"
try:
    with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')
        with open("torch_index.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Downloaded index.html")
except Exception as e:
    print(f"Error: {e}")
