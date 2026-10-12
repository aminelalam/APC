"""Genera una presentació HTML i el guió a partir dels resultats executats."""
from pathlib import Path
import base64
import json
import html
import pandas as pd
import nbformat
from nbconvert import HTMLExporter

carpeta = Path(__file__).resolve().parent
resultats = carpeta / 'resultats'
resum = json.loads((resultats / 'resum.json').read_text(encoding='utf-8'))
pre = pd.read_csv(resultats / 'preprocessament.csv').set_index('experiment')
graella = pd.read_csv(resultats / 'totes_les_configuracions.csv')
millors = graella.sort_values('mean_test_f1', ascending=False).groupby('familia', sort=False).head(1)
final = resum['test']
cm = resum['confusio']

def imatge(nom, descripcio):
    contingut = (resultats / 'grafics' / nom).read_bytes()
    codificada = base64.b64encode(contingut).decode('ascii')
    return '<img alt="' + html.escape(descripcio) + '" src="data:image/png;base64,' + codificada + '">'

files_taula = ''
for _, fila in millors.iterrows():
    files_taula += ('<tr><td>' + html.escape(fila['familia']) + '</td><td>'
                   + html.escape(fila['representacio']) + '</td><td>'
                   + f"{fila['mean_test_accuracy']:.3f}" + '</td><td>'
                   + f"{fila['mean_test_f1']:.3f}" + '</td></tr>')

diapositives = [
    ('Supervivència al Titanic',
     '<p class="gran">Pràctica 1 · Aprenentatge Computacional</p>'
     '<p>891 passatgers. Una pregunta: què podem predir i amb quins límits?</p>'
     '<p>Memòria amb codi, experiments i resultats reproduïbles.</p>'
     '<p class="nota">Autors i NIU: completar abans del lliurament.<br>'
     'Treball preparat amb assistència d’IA, declarada a la memòria.</p>'),
    ('Les dades tenen informació incompleta',
     '<div class="dues"><div><p>177 edats desconegudes.<br>687 cabines sense registrar.</p>'
     '<p>No tenir cabina apuntada no vol dir no tenir-ne.</p>'
     '<p>Sexe i classe mostren associacions fortes. Una associació no demostra una causa.</p></div>'
     + imatge('03_sexe_classe.png', 'Taxa de supervivència i suport per sexe i classe') + '</div>'),
    ('Preparació i validació',
     '<p>712 passatgers per investigar. 179 reservats per al test final.</p>'
     '<p>Cinc folds estratificats. Imputació i escalat ajustats dins de cada fold.</p>'
     '<p>KNN sense escalar: <strong>' + f"{pre.loc['KNN: escala cap', 'cv_accuracy']:.1%}" + '</strong> d’accuracy CV.<br>'
     'KNN amb StandardScaler: <strong>' + f"{pre.loc['KNN: escala estandard', 'cv_accuracy']:.1%}" + '</strong>.</p>'
     '<p>També hem provat eliminar files, imputació KNN, títols, família, PCA i polinomis.</p>'),
    ('Cinc famílies i una cerca comparable',
     '<p>' + str(resum['configuracions_graella']) + ' configuracions, '
     + str(resum['ajustos_graella']) + ' ajustos de CV. Mateixos folds per a tothom.</p>'
     '<table><thead><tr><th>Model</th><th>Variables</th><th>Accuracy CV</th><th>F1 CV</th></tr></thead><tbody>'
     + files_taula + '</tbody></table>'
     '<p class="nota">Millor configuració de cada família segons F1. Són resultats de selecció, no del test.</p>'),
    ('Resultat del test reservat',
     '<div class="dues"><div><p class="gran">' + html.escape(resum['model']) + '</p>'
     '<p>Accuracy: <strong>' + f"{final['accuracy']:.1%}" + '</strong><br>'
     'F1: <strong>' + f"{final['f1']:.3f}" + '</strong><br>'
     'Recall: <strong>' + f"{final['recall']:.1%}" + '</strong></p>'
     '<p>' + str(cm['FP']) + ' falsos positius.<br>' + str(cm['FN']) + ' falsos negatius.</p></div>'
     + imatge('23_final_confusio.png', 'Matriu de confusió del test final') + '</div>'),
    ('Què hem après i què queda obert',
     '<p>Escalar i representar bé les dades pot aportar tant com canviar de model.</p>'
     '<p>F1 ens obliga a mirar els supervivents que perdem. Accuracy sola no explica els errors.</p>'
     '<p>Els subgrups i els bitllets compartits limiten la lectura de la mitjana global.</p>'
     '<p>Pròxim pas: més dades i validació per grups de tot el procés de selecció.</p>'
     '<p class="nota">Ús educatiu sobre aquest registre històric. Sense extrapolar a decisions de rescat.</p>')]

seccions = ''
for index, (titol, cos) in enumerate(diapositives):
    seccions += '<section class="slide' + (' actual' if index == 0 else '') + '">'
    seccions += '<h1>' + titol + '</h1>' + cos
    seccions += '<footer>' + str(index + 1) + ' / 6</footer></section>'

document = '''<!doctype html><html lang="ca"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pràctica 1 · Titanic · Presentació</title><style>
*{box-sizing:border-box}body{margin:0;background:#eff2f5;color:#163247;font-family:Arial,sans-serif}
main{max-width:1280px;margin:20px auto}.slide{display:none;background:white;min-height:720px;padding:58px 70px 70px;position:relative}
.slide.actual{display:block}h1{font-size:43px;line-height:1.12;margin:0 0 38px;max-width:1100px}
p{font-size:27px;line-height:1.38;margin:24px 0}.gran{font-size:34px}.nota{font-size:19px;color:#526777}
.dues{display:grid;grid-template-columns:0.9fr 1.2fr;gap:30px;align-items:center}
img{width:100%;max-height:465px;object-fit:contain}table{width:100%;border-collapse:collapse;font-size:25px}
th,td{text-align:left;padding:14px;border-bottom:1px solid #d1dce4}th{background:#e9f1f6}
footer{position:absolute;bottom:25px;right:60px;color:#526777;font-size:18px}
nav{display:flex;justify-content:center;align-items:center;gap:20px;padding:16px}button{font:inherit;padding:10px 22px;cursor:pointer}
@media(max-width:800px){main{margin:0}.slide{padding:30px 25px 65px;min-height:90vh}h1{font-size:32px}p{font-size:22px}.dues{grid-template-columns:1fr}table{font-size:17px}th,td{padding:10px}}
@media print{body{background:white}main{margin:0;max-width:none}.slide{display:block;break-after:page;min-height:0;height:180mm;padding:10mm}nav{display:none}h1{font-size:30px}p{font-size:21px}img{max-height:110mm}table{font-size:20px}@page{size:A4 landscape;margin:10mm}}
</style></head><body><main>''' + seccions + '''</main>
<nav aria-label="Canvi de diapositiva"><button id="anterior">Anterior</button><span id="pagina" aria-live="polite">1 / 6</span><button id="seguent">Següent</button></nav>
<script>const slides=document.querySelectorAll('.slide');let pagina=0;
function mostrar(n){pagina=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('actual',i===pagina));document.getElementById('pagina').textContent=(pagina+1)+' / '+slides.length;}
document.getElementById('anterior').onclick=()=>mostrar(pagina-1);document.getElementById('seguent').onclick=()=>mostrar(pagina+1);
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();mostrar(pagina+1);}if(e.key==='ArrowLeft'){e.preventDefault();mostrar(pagina-1);}});
</script></body></html>'''
(carpeta / 'Presentacio_4_minuts.html').write_text(document, encoding='utf-8')

guio = f'''# Guió oral de la pràctica 1

Sis diapositives. Temps orientatiu: 3 minuts i 40 segons, deixant 20 segons de marge. Cal assajar-lo amb la veu pròpia i completar els autors. Les dades són les de l'execució, no números inventats.

## 1. Pregunta i objectiu · 20 segons

En aquesta pràctica intentem predir la supervivència dels passatgers del Titanic. Però no volíem quedar-nos amb el model que dona el número més alt. Hem investigat què representen les dades, com preparar-les i quins errors fa el resultat final.

## 2. Dades i absències · 35 segons

Tenim 891 passatgers i dotze columnes, inclòs el target. Hi ha 177 edats desconegudes i 687 cabines sense registrar. Ens va semblar important no interpretar un buit com un zero. La cabina registrada està relacionada amb la classe. Per això hem mirat taxes i recomptes dins dels grups. Sexe i classe mostren associacions fortes, però les dades no demostren causalitat.

## 3. Preparació i protocol · 40 segons

Hem reservat 179 passatgers per al test i hem investigat amb els 712 restants. Fem cinc folds estratificats i ajustem imputació i escalat només amb les files d'entrenament de cada fold. Això evita filtracions d'informació. Una comprovació clara dels apunts és KNN: passa del {pre.loc['KNN: escala cap', 'cv_accuracy']:.1%} sense escalar al {pre.loc['KNN: escala estandard', 'cv_accuracy']:.1%} amb StandardScaler. També hem provat idees que no milloren, com eliminar la meitat de variables.

## 4. Models i selecció · 45 segons

Comparem KNN, logística, SVM, arbre i Random Forest, a més d'una baseline que sempre prediu la classe majoritària. La cerca principal té {resum['configuracions_graella']} configuracions i {resum['ajustos_graella']} ajustos. Primer comparem accuracy i després tornem a ordenar tots els candidats per F1. Escollim F1 perquè volem equilibrar la precisió de les supervivències predites amb els supervivents reals que detectem. Els resultats de la taula són de selecció i poden ser optimistes.

## 5. Resultat final · 45 segons

Amb les decisions tancades, el model {resum['model']} dona una accuracy del {final['accuracy']:.1%} i F1 de {final['f1']:.3f} al test. Detecta {cm['TP']} de {cm['TP']+cm['FN']} supervivents. Hi ha {cm['FP']} falsos positius i {cm['FN']} falsos negatius. Hem mirat també ROC i precision-recall, intervals orientatius i errors per sexe, classe i edat. Això explica més que dir només el percentatge d'encert.

## 6. Aprenentatge i límits · 35 segons

La representació de les dades té molt pes. Més complexitat no garanteix millor resultat. També hem comprovat que validar passatgers aleatoris no és exactament el mateix que validar bitllets nous. Amb més temps, faríem validació niada de tot el procés i aconseguiríem més dades. El model serveix com a exercici sobre un registre històric, no per prendre decisions actuals de rescat.

## Preguntes que convé poder explicar

- Per què no es calcula la mediana abans de fer cross-validation?
- Quina diferència hi ha entre el `test_score` de GridSearchCV i el test reservat?
- Què vol dir un fals negatiu si Survived=1 és la classe positiva?
- Per què un score de SVM no és una probabilitat?
- Què aporta F1 i quina informació deixa fora?
- Per què menys MAE d'imputació no implica més accuracy de classificació?
- Per què una correlació o una importància no demostra causalitat?
'''
(carpeta / 'GUIO_PRESENTACIO.md').write_text(guio, encoding='utf-8')
print('Presentació HTML i guió generats amb els resultats reals.')

# Aquest comentari es construeix amb les taules guardades, sense reentrenar.
imputacio = pd.read_csv(resultats / 'imputacio_edat.csv').groupby('metode')['MAE_anys'].mean()
eliminacio = pd.read_csv(resultats / 'eliminacio_files.csv').groupby('politica').mean(numeric_only=True)
base = pd.read_csv(resultats / 'models_per_defecte.csv').set_index('experiment')
grups = pd.read_csv(resultats / 'validacio_per_bitllet.csv').set_index('experiment')
subgrups = pd.read_csv(resultats / 'errors_per_subgrup.csv')
intervals = pd.read_csv(resultats / 'intervals_test.csv', index_col=0)
imports = pd.read_csv(resultats / 'importancies.csv').set_index('variable')
estabilitat = pd.read_csv(resultats / 'estabilitat.csv')
temps = pd.read_csv(resultats / 'metodes_cerca.csv').set_index('metode')
f1_svm = graella.loc[graella['familia'] == 'SVM', 'mean_test_f1'].max()
f1_log = graella.loc[graella['familia'] == 'Logística', 'mean_test_f1'].max()
homes = subgrups[(subgrups['variable'] == 'Sex') & (subgrups['grup'] == 'male')].iloc[0]
dones = subgrups[(subgrups['variable'] == 'Sex') & (subgrups['grup'] == 'female')].iloc[0]
tercera = subgrups[(subgrups['variable'] == 'Pclass') & (subgrups['grup'] == '3')].iloc[0]

discussio = f'''## 6. Discussió dels resultats: què defensem i què no?

Aquesta lectura complementa les explicacions de cada experiment. Tots els números provenen de les taules de l'execució. Les decisions de selecció es prenen amb desenvolupament. Les observacions del test són diagnòstics posteriors i no han provocat canvis del model.

### 6.1. Les dades suggereixen interaccions, no una única regla

El creuament sexe/classe és més informatiu que mirar només una mitjana. A les figures veiem que les dones de primera i segona classe tenen taxes altes, mentre que les de tercera queden força per sota. Entre els homes també canvia la taxa segons la classe. Això dona sentit a provar models amb interaccions, però no permet reconstruir les circumstàncies de cada persona.

La matriu de correlació mostra una relació forta entre classe i cabina registrada. Per tant, dues columnes predictives poden estar recollint informació semblant. En el cas d'Age i EdatDesconeguda apareix una casella buida: quan Age té valor, l'indicador sempre és zero. No hi ha variació per calcular aquesta correlació per parelles. **El blanc és una correlació indefinida, no una correlació igual a zero.**

MidaFamilia tampoc compta tota la xarxa social: només suma els familiars recollits a SibSp i Parch, més el passatger. Ticket pot compartir-se entre persones, però no identifica totes les famílies. Fare és la tarifa registrada i no n'hem inventat un preu individual dividint-lo pel nombre de files del mateix bitllet, que pot estar incomplet.

### 6.2. Perdre files no és una solució neutra

Entrenar només amb files completes de les variables base deixa de mitjana {eliminacio.loc['Entrenar només completes', 'n_train']:.1f} files per fold, davant {eliminacio.loc['Imputar totes', 'n_train']:.1f} amb imputació. En la mateixa validació completa, les accuracies són {eliminacio.loc['Entrenar només completes', 'accuracy_tota_val']:.3f} i {eliminacio.loc['Imputar totes', 'accuracy_tota_val']:.3f}, respectivament. El filtre de casos complets només cobreix el {eliminacio.loc['Imputar totes', 'cobertura_val_completa']:.1%} de les files de validació. El seu resultat parcial respondria a una població diferent.

Preferim conservar passatgers i imputar. És una decisió que combina resultat, cobertura i mida de mostra. Si exigíssim també Cabin, perdríem moltes més files. No hem convertit una cabina desconeguda en una falsa cabina coneguda: n'hem conservat l'indicador.

### 6.3. Reconstruir millor una edat no garanteix classificar millor

En deu simulacions, el MAE mitjà és {imputacio['knn']:.2f} anys amb KNNImputer, {imputacio['median']:.2f} amb mediana i {imputacio['mean']:.2f} amb mitjana. KNN reconstrueix millor les edats ocultades en aquesta prova. Però la logística amb imputació KNN dona accuracy {pre.loc['Logística: imputació KNN', 'cv_accuracy']:.3f}, davant {pre.loc['Logística: median', 'cv_accuracy']:.3f} amb mediana.

No és una contradicció. El MAE penalitza la distància en anys, mentre que la classificació depèn de com aquests valors afecten una frontera conjunta amb altres atributs. A més, hem amagat edats conegudes de manera aleatòria; els nuls reals podrien seguir un altre patró. Per això no podem afirmar que hem descobert les edats reals ni que KNN sigui sempre millor.

### 6.4. La prova de geometria sí que té un efecte gran

Amb k=5 fixat, KNN sense escalar obté {pre.loc['KNN: escala cap', 'cv_accuracy']:.3f} d'accuracy; amb StandardScaler, {pre.loc['KNN: escala estandard', 'cv_accuracy']:.3f}. Són {100*(pre.loc['KNN: escala estandard', 'cv_accuracy']-pre.loc['KNN: escala cap', 'cv_accuracy']):.2f} punts percentuals de diferència. Les unitats de Fare estaven condicionant la noció de proximitat.

MinMax dona {pre.loc['KNN: escala minmax', 'cv_accuracy']:.3f} i RobustScaler {pre.loc['KNN: escala robust', 'cv_accuracy']:.3f}. Aquí la diferència important és escalar respecte a no escalar; no hi ha evidència suficient per proclamar un guanyador universal entre els escaladors. RobustScaler redueix la influència dels extrems en l'escala, però no elimina passatgers.

### 6.5. Més transformacions no sempre ajuden

La logística base amb mediana dona {pre.loc['Logística: median', 'cv_accuracy']:.3f}. Afegir només el títol puja a {pre.loc['Logística: títol', 'cv_accuracy']:.3f}; la representació ampliada arriba a {pre.loc['Logística: ampliada', 'cv_accuracy']:.3f}. El títol pot resumir edat, sexe i convencions socials. No l'interpretem com una causa de supervivència.

Afegir només família dona {pre.loc['Logística: família', 'cv_accuracy']:.3f}, i substituir tarifa per log de tarifa dona {pre.loc['Logística: log tarifa', 'cv_accuracy']:.3f}. Són exemples d'idees raonables que no milloren aquesta logística per si soles. El resultat de combinar-les pot ser diferent, i per això comparem representacions completes dins de la cerca.

PCA al 95% dona {pre.loc['Logística: PCA 95%', 'cv_accuracy']:.3f}, molt a prop de la base. No necessitem comprimir una representació tan petita si no hi ha un benefici clar i perdrem interpretació. Seleccionar el 50% dona {pre.loc['Logística: selecció 50%', 'cv_accuracy']:.3f}: una regla univariant pot perdre informació complementària. Grau 2 puja l'accuracy a {pre.loc['Logística: grau 2', 'cv_accuracy']:.3f}, però l'F1 és {pre.loc['Logística: grau 2', 'cv_f1']:.3f} i la dispersió entre folds augmenta. No l'escollim només perquè train millori.

L'encoding ordinal de totes les categories dona {pre.loc['Logística: ordinal', 'cv_accuracy']:.3f}, davant {pre.loc['Logística: median', 'cv_accuracy']:.3f} amb one-hot. La diferència és petita; el motiu principal per mantenir one-hot és que Embarked no té un ordre numèric justificat.

### 6.6. Sobreajust i cerca: no confondre un màxim amb una victòria clara

L'arbre per defecte arriba a accuracy de train {base.loc['Arbre', 'train_accuracy']:.3f}, però CV baixa a {base.loc['Arbre', 'cv_accuracy']:.3f}. Això fa visible el sobreajust. Limitar la profunditat i el nombre mínim de casos per fulla és una manera comprensible de controlar-lo. El Random Forest per defecte també té una separació gran entre train i CV: combinar arbres no elimina tots els problemes.

Amb la cerca, Random Forest i una SVM RBF empaten en accuracy CV arrodonida a sis decimals: 0.831498. Per F1, Random Forest dona {resum['cv_f1_seleccio']:.6f}, la millor SVM {f1_svm:.6f} i logística {f1_log:.6f}. La diferència entre RF i SVM és de només {resum['cv_f1_seleccio']-f1_svm:.6f}. **Escollim el màxim segons la regla fixada, però no afirmem una superioritat estadística clara del Random Forest.** Una logística més senzilla continuaria sent una alternativa defensable si la interpretació fos prioritària. No l'hem avaluat després al test per triar la que hi quedés millor.

Canviar de mètrica sí que canvia la configuració preferida de SVM: la millor per F1 utilitza pesos de classe equilibrats, mentre que la millor per accuracy no. Aquesta és la resposta pràctica a «la comparació amb accuracy aguanta?»: el primer model pot continuar destacant, però les configuracions i el compromís d'errors no són els mateixos.

La logística fixada dona mitjanes d'accuracy entre {estabilitat['cv_accuracy'].min():.3f} i {estabilitat['cv_accuracy'].max():.3f} quan variem folds i llavor. Aquesta prova no és un test estadístic de RF contra SVM, però recorda que les particions aporten variabilitat i que una quarta xifra decimal no és una garantia.

Grid SVM prova 48 candidats i dona {temps.loc['Grid SVM', 'accuracy_cv']:.3f}, amb {temps.loc['Grid SVM', 'segons']:.1f} segons. Random prova 12 candidats i dona {temps.loc['Random SVM', 'accuracy_cv']:.3f}, amb {temps.loc['Random SVM', 'segons']:.1f} segons. Aquesta comparació il·lustra un pressupost de candidats menor, no una prova amb temps igual ni una conclusió universal sobre els mètodes.

### 6.7. El test és menys bo que la selecció i això és part del resultat

L'F1 baixa de {resum['cv_f1_seleccio']:.3f} en CV de selecció a {final['f1']:.3f} en el test. No hem intentat arreglar aquesta diferència buscant una altra llavor o altres paràmetres després de veure-la. Pot intervenir l'optimisme de selecció i la variabilitat d'una mostra petita.

La matriu és TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']} i TP={cm['TP']}. El model detecta {cm['TP']} dels {cm['TP']+cm['FN']} supervivents. F1 resumeix aquest compromís, però no explica els {cm['TN']} no supervivents ben classificats, que sí que contribueixen a accuracy. Per això presentem les dues mètriques.

El bootstrap situa accuracy entre {intervals.loc['accuracy', 'percentil_2_5']:.3f} i {intervals.loc['accuracy', 'percentil_97_5']:.3f}, i F1 entre {intervals.loc['f1', 'percentil_2_5']:.3f} i {intervals.loc['f1', 'percentil_97_5']:.3f}. Són intervals orientatius condicionats al model fix i a la mostra, amb la limitació d'independència entre passatgers. No demostren un rendiment mínim garantit.

### 6.8. La mitjana global amaga un problema amb els homes supervivents

El recall entre dones és {int(dones['TP'])}/{int(dones['positius'])} = {dones['recall']:.1%}; entre homes, {int(homes['TP'])}/{int(homes['positius'])} = {homes['recall']:.1%}. Dels {cm['FN']} falsos negatius globals, {int(homes['FN'])} són homes. L'accuracy d'homes encara és {homes['accuracy']:.1%}, perquè hi ha molts no supervivents ben classificats. Aquest és un exemple concret de per què accuracy sola no ens basta.

A tercera classe, l'accuracy és {tercera['accuracy']:.1%}, però només detectem {int(tercera['TP'])} de {int(tercera['positius'])} supervivents. Una accuracy alta dins d'un subgrup tampoc garanteix bon recall. Amb infants només tenim 17 casos i 10 positius al test: les conclusions d'aquest grup són especialment inestables.

Amb validació per bitllet, F1 passa de {grups.loc['Passatgers aleatoris', 'cv_f1']:.3f} a {grups.loc['Bitllets separats', 'cv_f1']:.3f}. No podem atribuir tota la diferència als vincles perquè també canvien els folds, però la prova assenyala un límit del protocol per passatgers. Si volguéssim predir famílies noves, caldria seleccionar i avaluar tot el procés amb grups.

### 6.9. La cabina no aporta el mateix en totes les proves

En la permutació del model final, la caiguda mitjana de F1 en barrejar Cabin és {imports.loc['Cabin', 'caiguda_f1']:.3f}: és negativa, és a dir, aquesta alteració millora lleugerament l'F1 en aquesta mostra. No ho amaguem. Pot intervenir soroll, redundància amb classe i la mida petita del test. No justifica eliminar Cabin i tornar a provar el mateix test, perquè estaríem adaptant el model a l'avaluació final.

Pclass i Sex tenen caigudes de {imports.loc['Pclass', 'caiguda_f1']:.3f} i {imports.loc['Sex', 'caiguda_f1']:.3f}, respectivament. Name influeix a través del títol, no perquè s'aprenguin noms complets. Les importàncies descriuen dependència predictiva amb aquesta prova i aquest model; les correlacions i les combinacions poc realistes impedeixen llegir-les com efectes causals.

### 6.10. Decisió que podem defensar

Hem seleccionat un model mitjançant una regla explícita i hem conservat les proves que no milloraven. El resultat és útil per aprendre a classificar aquest registre, amb errors concrets i límits clars. La millora més rellevant per a un projecte posterior seria definir bé la població futura i la unitat de validació, abans d'ampliar la graella o provar un model més complex.
'''
(carpeta / 'RESULTATS_COMENTATS.md').write_text(discussio, encoding='utf-8')

# Inserim el mateix comentari a la memòria sense tocar les cel·les executades.
ruta_notebook = carpeta / 'Practica_1_Titanic.ipynb'
notebook = nbformat.read(ruta_notebook, as_version=4)
cel_les = []
for cell in notebook.cells:
    if cell.metadata.get('comentari_resultats', False) is not True:
        cel_les.append(cell)
comentari = nbformat.v4.new_markdown_cell(discussio)
comentari.metadata['comentari_resultats'] = True
cel_les.insert(len(cel_les) - 1, comentari)
notebook.cells = cel_les
nbformat.write(notebook, ruta_notebook)
exportador = HTMLExporter()
exportador.exclude_input_prompt = True
exportador.exclude_output_prompt = True
contingut_html, recursos = exportador.from_notebook_node(notebook)
(carpeta / 'Practica_1_Titanic.html').write_text(contingut_html, encoding='utf-8')
print('Discussió detallada afegida a la memòria i exportació HTML actualitzada.')
