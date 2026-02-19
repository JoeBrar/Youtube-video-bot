
try:
    with open('debug_output.txt', 'r', encoding='utf-16-le') as f:
        content = f.read()
except:
    try:
        with open('debug_output.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

# Extract HTML part
start_marker = "[DEBUG HTML START]"
end_marker = "[DEBUG HTML END]"

if start_marker in content:
    start = content.find(start_marker)
    end = content.find(end_marker)
    print(content[start:end+len(end_marker)])
else:
    print("HTML markers not found. Dumping raw content (first 1000 chars):")
    print(content[:1000])
