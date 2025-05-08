import os

project_root = "neurooracle"
output_txt = "neurooracle_core_py_only.txt"
collected_code = []

# Only collect known core files we built together
core_py_files = [
    "main.py",
    "ingest.py",
    "embed.py",
    "graph.py",
    "cluster.py",
    "write.py",
    "forecast.py",
    "utils.py"
]

# Read only the specified files
for filename in core_py_files:
    file_path = os.path.join(project_root, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        collected_code.append(f"# ==== {filename} ====\n{code}\n")

# Save to a smaller text file
output_path = os.path.join(project_root, output_txt)
with open(output_path, 'w', encoding='utf-8') as out_f:
    out_f.write("\n".join(collected_code))

output_path
