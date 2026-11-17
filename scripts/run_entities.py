import subprocess
import sys

result = subprocess.run([sys.executable, "scripts/lab_entities.py", "--chunks", "data/lab/chunks_512_50.jsonl"], 
                       capture_output=True, text=True, cwd="c:\\Users\\Compu Art Store\\AI-project-group\\AI-project-group")
print(result.stdout)
