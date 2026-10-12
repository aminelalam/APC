"""Comprovacions de coherència de l'entrega ja executada, sense reentrenar."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from PIL import Image

carpeta = Path(__file__).resolve().parent
resultats = carpeta / 'resultats'
notebook = nbformat.read(carpeta / 'Practica_1_Titanic.ipynb', as_version=4)
nbformat.validate(notebook)
comptador = 0
for cell in notebook.cells:
    if cell.cell_type == 'code':
        comptador += 1
        assert cell.execution_count is not None, 'Hi ha una cel·la sense executar.'
        for output in cell.outputs:
            assert output.output_type != 'error', 'Hi ha una cel·la amb error.'
            if output.output_type == 'stream':
                assert 'Traceback' not in output.text
                assert 'ConvergenceWarning' not in output.text

# El codi entregat i el que s'ha executat han de coincidir.
font = (carpeta / 'practica1.py').read_text(encoding='utf-8')
blocs = []
for fragment in font.split('# %%'):
    if fragment.strip() and not fragment.startswith(' [markdown]'):
        blocs.append(fragment.strip())
executats = []
for cell in notebook.cells:
    if cell.cell_type == 'code':
        executats.append(cell.source.strip())
assert blocs == executats, 'El codi .py difereix del notebook.'

dades = pd.read_csv(carpeta / 'titanic_train.csv')
particions = pd.read_csv(resultats / 'particions.csv')
prediccions = pd.read_csv(resultats / 'prediccions_test.csv')
resum = json.loads((resultats / 'resum.json').read_text(encoding='utf-8'))
assert hashlib.sha256((carpeta / 'titanic_train.csv').read_bytes()).hexdigest() == resum['sha256_dades']
assert len(particions) == len(dades) == 891
assert particions['PassengerId'].is_unique
ids_test = set(particions.loc[particions['particio'] == 'test', 'PassengerId'])
ids_train = set(particions.loc[particions['particio'] == 'desenvolupament', 'PassengerId'])
assert ids_test.isdisjoint(ids_train)
assert len(ids_test) == len(prediccions) == 179
assert len(ids_train) == 712
assert ids_test == set(prediccions['PassengerId'])
etiquetes = dades.set_index('PassengerId')['Survived']
assert np.array_equal(prediccions['Survived'], etiquetes.loc[prediccions['PassengerId']])

y = prediccions['Survived']
pred = prediccions['predit']
scores = prediccions['score']
assert np.isclose(accuracy_score(y, pred), resum['test']['accuracy'])
assert np.isclose(f1_score(y, pred), resum['test']['f1'])
assert np.isclose(roc_auc_score(y, scores), resum['test']['roc_auc'])
assert np.isclose(average_precision_score(y, scores), resum['test']['average_precision'])
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
assert resum['confusio'] == {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

graella = pd.read_csv(resultats / 'totes_les_configuracions.csv')
assert len(graella) == resum['configuracions_graella'] == 184
assert graella['familia'].nunique() == 5
assert np.isclose(graella['mean_test_f1'].max(), resum['cv_f1_seleccio'])
assert graella['mean_test_f1'].notna().all()
subgrups = pd.read_csv(resultats / 'errors_per_subgrup.csv')
for variable in subgrups['variable'].unique():
    grup = subgrups[subgrups['variable'] == variable]
    assert grup['n'].sum() == 179
    assert grup['TP'].sum() == tp
    assert grup['FN'].sum() == fn
    assert grup['FP'].sum() == fp
    assert grup['TN'].sum() == tn

grafics = sorted((resultats / 'grafics').glob('*.png'))
assert len(grafics) == 26
for grafic in grafics:
    with Image.open(grafic) as imatge:
        assert imatge.width > 500 and imatge.height > 300
        imatge.verify()

presentacio = (carpeta / 'Presentacio_4_minuts.html').read_text(encoding='utf-8')
assert presentacio.count('<section class="slide') == 6
assert presentacio.count('data:image/png;base64,') == 2
assert 'ArrowRight' in presentacio and 'ArrowLeft' in presentacio
assert (carpeta / 'Practica_1_Titanic.html').stat().st_size > 500000
assert (carpeta / 'RESULTATS_COMENTATS.md').stat().st_size > 5000
print('Comprovacions correctes:', comptador, 'cel·les de codi, 26 figures, 184 candidats i test coherent.')
