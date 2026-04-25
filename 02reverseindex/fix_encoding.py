import nbformat

notebook_path = "c:/Users/DELL/OneDrive - Escuela Politécnica Nacional/DANIEL/EPN/SÉPTIMO SEMESTRE/RI/CODIGO/ir26a/02reverseindex/02reverseindex.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace the problematic encoding to include error ignoring
        cell.source = cell.source.replace("encoding ='utf-8'", "encoding='utf-8', errors='ignore'")
        cell.source = cell.source.replace("encoding='utf-8'", "encoding='utf-8', errors='ignore'")

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully. Added errors='ignore' to open() calls.")
