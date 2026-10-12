"""Converteix el codi amb cel·les en un notebook, l'executa i exporta HTML."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

carpeta = Path(__file__).resolve().parent
os.chdir(carpeta)
font = (carpeta / 'practica1.py').read_text(encoding='utf-8')
cel_les = []
for fragment in font.split('# %%'):
    if not fragment.strip():
        continue
    if fragment.startswith(' [markdown]'):
        linies = fragment.splitlines()[1:]
        text = []
        for linia in linies:
            if linia.startswith('# '):
                text.append(linia[2:])
            elif linia == '#':
                text.append('')
            else:
                text.append(linia)
        cel_les.append(nbformat.v4.new_markdown_cell('\n'.join(text).strip()))
    else:
        cel_les.append(nbformat.v4.new_code_cell(fragment.strip()))
notebook = nbformat.v4.new_notebook(cells=cel_les)
notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata.language_info = {'name': 'python', 'version': '3.13'}
ruta = carpeta / 'Practica_1_Titanic.ipynb'
nbformat.write(notebook, ruta)
client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(carpeta)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, ruta)
exportador = HTMLExporter()
exportador.exclude_input_prompt = True
exportador.exclude_output_prompt = True
html, recursos = exportador.from_notebook_node(notebook)
(carpeta / 'Practica_1_Titanic.html').write_text(html, encoding='utf-8')
print('Notebook executat i HTML generat:', ruta)
