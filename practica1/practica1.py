# %% [markdown]
# # Pràctica 1. Què podem aprendre dels passatgers del Titanic?
#
# **Memòria i codi.** Aprenentatge Computacional, curs 2026-2027.
#
# **Autors i NIU:** cal completar aquesta dada abans de lliurar la pràctica.
#
# Volem predir `Survived`: 1 si el passatger va sobreviure i 0 si no.
# La pregunta no és només quin model encerta més. També volem entendre què
# representen les dades, què passa quan falta informació i en quins passatgers
# ens equivoquem. Una associació històrica no demostra una causa.
#
# Seguim els cinc apartats de l'enunciat. A cada bloc expliquem la pregunta,
# fem la prova i interpretem el resultat. No totes les proves han de millorar:
# una prova que surt pitjor també serveix per descartar una idea.
#
# **Protocol fixat abans de comparar:** 20% de test reservat, 5 folds
# estratificats sobre el 80% restant i llavor 42. Primer comparem accuracy,
# tal com demana l'enunciat. Per a la decisió final prioritzarem F1 de la classe
# supervivent, perquè ens interessa equilibrar precision i recall d'aquesta
# classe. És una elecció didàctica explícita, no una regla universal del Titanic.
# Mantindrem també accuracy, balanced accuracy, ROC-AUC i average precision.
# El llindar final serà el que porta el model per defecte; la prova de llindars
# de logística serà explicativa i no modificarà el model final.
#
# ## 0. Materials que hem utilitzat
#
# | Material de la carpeta | Idea que comprovem aquí |
# |---|---|
# | Pràctica 1 i notebook de primers passos | Cabina desconeguda, baseline, títol del nom |
# | P1 Encoding i NaNs | No confondre absència amb zero, imputar dins de train |
# | S01 i S02 | Representació de les dades, distàncies i escalat de KNN |
# | P2 i S03-S04 | Error d'entrenament, generalització i regularització |
# | P3 Classificació | Models de classificació i lectura dels errors |
# | S05 Validació | Test separat i preprocessament dins de cada fold |
# | S06 Llindars | Separar score, probabilitat i decisió |
# | S07 Mètriques | Baseline, ROC/PR i errors per subgrups |
# | S08 SVM | Marge, C, gamma i kernels |
#
# Els CSV de cases, sensors i diabetis són d'altres problemes. No es barregen
# amb el Titanic. Els ZIP contenen els mateixos materials descomprimits.
# No tenim el llibre complet a la carpeta: les connexions es basen en els
# apunts i notebooks disponibles, no en una lectura inventada del llibre.

# %%
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

LLAVOR = 42
TREBALLADORS = 2
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 35)
pd.set_option('display.precision', 4)

# El notebook funciona des de la seva carpeta, des de l'arrel o des de Colab.
candidats = [Path('dades/titanic_train.csv'),
             Path('solucio_practica1/dades/titanic_train.csv'),
             Path('../APC_Practica_1_2026/titanic_train.csv'),
             Path('APC_Practica_1_2026/titanic_train.csv'),
             Path('titanic_train.csv')]
fitxer = None
for candidat in candidats:
    if candidat.exists():
        fitxer = candidat
        break
if fitxer is None:
    raise FileNotFoundError('Cal posar titanic_train.csv al costat del notebook o a dades/.')

sortida = Path('resultats')
sortida.mkdir(exist_ok=True)
(sortida / 'grafics').mkdir(exist_ok=True)

def guardar_grafic(nom):
    plt.tight_layout()
    plt.savefig(sortida / 'grafics' / (nom + '.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def comentar(text):
    display(Markdown(text))

versions = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}
print('Versions:', versions)
dades = pd.read_csv(fitxer)
empremta = hashlib.sha256(fitxer.read_bytes()).hexdigest()
print('SHA256 del CSV:', empremta)
print('Files i columnes:', dades.shape)
display(dades.head())

# %% [markdown]
# ## 1. EDA: entendre les files abans d'entrenar
#
# ### 1.1. Què vol dir cada columna?
#
# Cada fila és un passatger del conjunt d'entrenament proporcionat. No són tots
# els passatgers del vaixell i no hi ha una columna de data. Tenim 12 columnes:
# el target i 11 atributs originals; això no vol dir que els 11 siguin útils.
#
# | Columna | Significat | Tipus conceptual i decisió |
# |---|---|---|
# | PassengerId | Identificador del registre | Identificador; exclòs de X |
# | Survived | Supervivència observada | Target binari, 0/1 |
# | Pclass | Classe del bitllet, 1/2/3 | Categòrica ordinal; comparem codificacions |
# | Name | Nom i tractament | Text; n'extraiem el títol |
# | Sex | Sexe registrat | Categoria binària del fitxer |
# | Age | Edat en anys | Numèrica; els decimals poden representar infants |
# | SibSp | Germans/germanes i cònjuge a bord | Recompte, no total de parents |
# | Parch | Pares/mares i fills/filles a bord | Recompte |
# | Ticket | Codi de bitllet | Identificador compartit per alguns passatgers |
# | Fare | Tarifa registrada | Numèrica; pot correspondre a bitllets compartits |
# | Cabin | Cabina registrada | Text amb molts desconeguts |
# | Embarked | Port C, Q o S | Categoria nominal, sense ordre numèric |
#
# Un `int64` no implica una magnitud: restar identificadors no mesura similitud.
# Tampoc assumim que la distància social entre primera i segona classe sigui
# igual que entre segona i tercera. Per això la base fa one-hot de Pclass.
#
# Fem una auditoria general d'esquema i nuls, i separem el test abans de les
# associacions amb el target. Els gràfics exploratoris següents són de desenvolupament.

# %%
auditoria = pd.DataFrame({'tipus_python': dades.dtypes.astype(str),
                          'nuls': dades.isna().sum(),
                          'percentatge_nuls': dades.isna().mean() * 100,
                          'valors_diferents': dades.nunique()})
display(auditoria)
print('Files exactament duplicades:', dades.duplicated().sum())
print('Identificadors repetits:', dades['PassengerId'].duplicated().sum())
assert dades['PassengerId'].is_unique
assert dades['Survived'].isin([0, 1]).all()
assert dades['Survived'].notna().all()
assert dades['Age'].dropna().between(0, 110).all()
assert (dades['Fare'].dropna() >= 0).all()

desenvolupament, test_reservat = train_test_split(
    dades, test_size=0.20, stratify=dades['Survived'], random_state=LLAVOR)
desenvolupament = desenvolupament.copy()
test_reservat = test_reservat.copy()
assert set(desenvolupament.index).isdisjoint(test_reservat.index)
print('Desenvolupament:', len(desenvolupament), 'Test reservat:', len(test_reservat))
particions = pd.DataFrame({'PassengerId': dades['PassengerId'], 'particio': 'desenvolupament'})
particions.loc[test_reservat.index, 'particio'] = 'test'
particions.to_csv(sortida / 'particions.csv', index=False)
eda = desenvolupament.copy()

# %% [markdown]
# ### 1.2. Target i desequilibri
#
# Comprovem si una accuracy aparentment bona es pot obtenir sense mirar cap
# atribut. El desequilibri és moderat; no el tractarem com un cas d'1% de positius.
# Afegim recompte i proporció perquè un percentatge sense denominador enganya.

# %%
target = eda['Survived'].value_counts().sort_index()
display(pd.DataFrame({'passatgers': target, 'proporcio': target / len(eda)}))
ax = target.rename(index={0: 'No sobreviu', 1: 'Sobreviu'}).plot.bar(rot=0)
ax.set(ylabel='Nombre de passatgers', title='Target al conjunt de desenvolupament')
for i, valor in enumerate(target):
    ax.text(i, valor + 3, str(valor), ha='center')
guardar_grafic('01_target')
comentar(f"La classe majoritària ocupa el **{target.max()/len(eda):.1%}**. "
         'Aquesta és la referència mínima: predir sempre 0 pot encertar bastant, '
         'però el seu recall de supervivents serà zero.')

# %% [markdown]
# ### 1.3. Qui sobreviu més en aquesta mostra?
#
# Mirem sexe, classe, port i mida familiar. Els gràfics mostren associacions,
# no efectes causals. Per exemple, el port pot estar relacionat amb la classe
# i la tarifa. Una taxa alta amb tres persones és molt menys estable que amb 200.

# %%
eda['MidaFamilia'] = eda['SibSp'] + eda['Parch'] + 1
eda['CabinaConeguda'] = eda['Cabin'].notna().astype(int)
eda['EdatDesconeguda'] = eda['Age'].isna().astype(int)
fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
for columna, ax in zip(['Sex', 'Pclass', 'Embarked', 'MidaFamilia'], eixos.flat):
    taula = eda.groupby(columna)['Survived'].agg(['count', 'mean'])
    display(taula)
    taula['mean'].plot.bar(ax=ax, rot=0)
    ax.set(title='Supervivència segons ' + columna, ylabel='Proporció', ylim=(0, 1.15))
    for posicio, nombre in enumerate(taula['count']):
        ax.text(posicio, taula['mean'].iloc[posicio] + 0.03, 'n=' + str(nombre), ha='center', fontsize=8)
guardar_grafic('02_taxa_per_grups')

fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
taxes = eda.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
comptes = pd.crosstab(eda['Sex'], eda['Pclass'])
sns.heatmap(taxes, annot=True, fmt='.2f', vmin=0, vmax=1, ax=eixos[0], cmap='Blues')
eixos[0].set_title('Taxa de supervivència: sexe i classe')
sns.heatmap(comptes, annot=True, fmt='d', ax=eixos[1], cmap='Greys')
eixos[1].set_title('Suport de cada combinació')
guardar_grafic('03_sexe_classe')
taxes_sexe = eda.groupby('Sex')['Survived'].mean()
taxes_classe = eda.groupby('Pclass')['Survived'].mean()
comentar(f"En desenvolupament sobreviuen el **{taxes_sexe['female']:.1%} de les dones** "
         f"i el **{taxes_sexe['male']:.1%} dels homes**. A primera classe la taxa és "
         f"{taxes_classe.loc[1]:.1%}, i a tercera {taxes_classe.loc[3]:.1%}. "
         'El creuament evita quedar-nos només amb dues mitjanes globals. '
         'Són patrons del registre històric; no una justificació per valorar persones de manera diferent.')

# %% [markdown]
# ### 1.4. Edat, tarifa i valors extrems
#
# La tarifa té una cua llarga. Fem també log(1 + tarifa) per veure els valors
# baixos sense perdre els alts. No eliminem automàticament un passatger ric ni
# un infant: un valor extrem pot ser real i informatiu. Zero euros tampoc és
# necessàriament un valor perdut. No tenim informació per corregir-lo a mà.

# %%
display(eda[['Age', 'Fare', 'SibSp', 'Parch']].describe())
fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(data=eda, x='Age', hue='Survived', bins=25, element='step', ax=eixos[0, 0])
eixos[0, 0].set_title('Edat observada: els nuls no apareixen')
sns.histplot(data=eda, x='Fare', hue='Survived', bins=35, element='step', ax=eixos[0, 1])
eixos[0, 1].set_title('Tarifa original')
sns.boxplot(data=eda, x='Pclass', y='Fare', ax=eixos[1, 0])
eixos[1, 0].set_title('Valors extrems segons classe')
sns.histplot(x=np.log1p(eda['Fare']), hue=eda['Survived'], bins=25, element='step', ax=eixos[1, 1])
eixos[1, 1].set(xlabel='log(1 + Fare)', title='Tarifa amb escala logarítmica')
guardar_grafic('04_edat_tarifa')
print('Tarifes zero:', int((eda['Fare'] == 0).sum()))

# %% [markdown]
# ### 1.5. Què significa no tenir cabina apuntada?
#
# Un buit de Cabin vol dir «no coneixem la cabina», no «no tenia cabina».
# Comprovem si l'absència es relaciona amb la classe i amb la supervivència.
# Aquesta prova ve directament del notebook inicial de la pràctica.

# %%
fig, eixos = plt.subplots(1, 3, figsize=(15, 4))
eda.isna().mean().mul(100).sort_values(ascending=False).head(5).plot.bar(ax=eixos[0], rot=25)
eixos[0].set(title='Informació perduda', ylabel='% de files')
taula_cabina = eda.groupby('CabinaConeguda')['Survived'].agg(['count', 'mean'])
display(taula_cabina)
taula_cabina['mean'].plot.bar(ax=eixos[1], rot=0)
eixos[1].set(title='Cabina registrada i supervivència', ylim=(0, 1))
cabina_classe = eda.groupby(['Pclass', 'CabinaConeguda'])['Survived'].agg(['count', 'mean'])
display(cabina_classe)
cabina_classe['mean'].unstack().plot.bar(ax=eixos[2], rot=0)
eixos[2].set(title='Comparació dins de cada classe', ylim=(0, 1))
guardar_grafic('05_nuls_i_cabina')
display(pd.crosstab(eda['Pclass'], eda['CabinaConeguda'], normalize='index'))
display(eda.groupby('EdatDesconeguda')['Survived'].agg(['count', 'mean']))
comentar(f"Amb cabina registrada, la taxa és {taula_cabina.loc[1, 'mean']:.1%}; "
         f"sense cabina registrada, {taula_cabina.loc[0, 'mean']:.1%}. "
         'Aquesta diferència justifica provar un indicador de presència. '
         'La taula per classe mostra que cal controlar la composició dels grups. '
         'No podem demostrar que els nuls siguin aleatoris ni que conèixer la cabina causi supervivència.')

# %% [markdown]
# ### 1.6. Correlacions i límits
#
# Pearson resumeix relacions lineals. Spearman compara rangs i ajuda amb la
# tarifa asimètrica. Cap de les dues detecta totes les interaccions. Afegim
# `Dona` només per llegir aquesta matriu; al model farem one-hot.
# No posem codis arbitraris de port ni identificadors a la correlació.

# %%
correlacions = eda[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch',
                    'MidaFamilia', 'CabinaConeguda', 'EdatDesconeguda']].copy()
correlacions['Dona'] = (eda['Sex'] == 'female').astype(int)
fig, eixos = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(correlacions.corr(), annot=True, fmt='.2f', center=0, cmap='vlag', ax=eixos[0])
eixos[0].set_title('Pearson; càlcul per parelles disponibles')
sns.heatmap(correlacions.corr(method='spearman'), annot=True, fmt='.2f', center=0, cmap='vlag', ax=eixos[1])
eixos[1].set_title('Spearman; associació entre rangs')
guardar_grafic('06_correlacions')
comentar('**Lectura de l’EDA.** Sexe i classe separen grups amb taxes diferents. '
         'Tarifa i cabina també poden recollir part de la informació de classe. '
         'Una correlació baixa d’edat no prova que l’edat sigui inútil: pot importar '
         'de manera no lineal o en combinació amb el sexe. Per això compararem '
         'models lineals amb models que poden representar interaccions.')

# %% [markdown]
# ## 2. Preprocessament: decisions i experiments controlats
#
# Base: Age, Fare, SibSp i Parch numèriques; Sex, Pclass i Embarked categòriques.
# Age s'omple amb mediana i Embarked amb moda apreses **dins de cada fold**.
# StandardScaler resta la mitjana i divideix per la desviació; MinMax usa el
# rang; RobustScaler usa mediana i rang interquartílic. Els arbres no necessiten
# distàncies comparables. Per KNN, logística regularitzada i SVM sí que provem escala.
#
# One-hot evita inventar un ordre per a Sex i Embarked. Desconeixements en
# categories noves es toleren amb `handle_unknown='ignore'`. No fem target encoding.
# `Pipeline` aquí no és per complicar el codi: impedeix calcular la mediana o
# l'escalat amb les files que després utilitzarem per validar.
#
# Creem també variables amb regles fixes per fila: mida familiar, viatjar sol,
# cabina coneguda, edat desconeguda, log de tarifa i títol del nom. No comptem
# supervivents de la família ni fem servir etiquetes d'altres passatgers.

# %%
def crear_variables(taula):
    nova = taula.copy()
    nova['MidaFamilia'] = nova['SibSp'] + nova['Parch'] + 1
    nova['Sol'] = (nova['MidaFamilia'] == 1).astype(int)
    nova['CabinaConeguda'] = nova['Cabin'].notna().astype(int)
    nova['EdatDesconeguda'] = nova['Age'].isna().astype(int)
    nova['LogTarifa'] = np.log1p(nova['Fare'])
    # El tractament està entre la coma i el punt: "Cognom, Mr. Nom".
    nova['Titol'] = nova['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    nova['Titol'] = nova['Titol'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    habituals = ['Mr', 'Mrs', 'Miss', 'Master']
    nova.loc[~nova['Titol'].isin(habituals), 'Titol'] = 'Altres'
    return nova

X = crear_variables(desenvolupament.drop(columns='Survived'))
y = desenvolupament['Survived'].copy()
numeriques = ['Age', 'Fare', 'SibSp', 'Parch']
categories = ['Sex', 'Pclass', 'Embarked']
num_ampliades = ['Age', 'LogTarifa', 'SibSp', 'Parch', 'MidaFamilia', 'Sol',
                 'CabinaConeguda', 'EdatDesconeguda']
cat_ampliades = categories + ['Titol']
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=LLAVOR).split(X, y))
metriques_cv = {'accuracy': 'accuracy', 'f1': 'f1', 'auc': 'roc_auc',
                'ap': 'average_precision', 'balanced': 'balanced_accuracy'}

def preparar(num, cat, escala='estandard', imputacio='median', encoding='onehot'):
    passos_num = [('imputar', SimpleImputer(strategy=imputacio))]
    if escala == 'estandard':
        passos_num.append(('escalar', StandardScaler()))
    elif escala == 'minmax':
        passos_num.append(('escalar', MinMaxScaler()))
    elif escala == 'robust':
        passos_num.append(('escalar', RobustScaler()))
    if encoding == 'onehot':
        codificador = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        codificador = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    passos_cat = [('imputar', SimpleImputer(strategy='most_frequent')),
                  ('codificar', codificador)]
    return ColumnTransformer([('num', Pipeline(passos_num), num),
                              ('cat', Pipeline(passos_cat), cat)])

def fer_model(classificador, num=numeriques, cat=categories, escala='estandard',
              imputacio='median', encoding='onehot'):
    return Pipeline([('preparar', preparar(num, cat, escala, imputacio, encoding)),
                     ('model', classificador)])

def avaluar_cv(nom, model, particions=folds):
    inici = perf_counter()
    resultat = cross_validate(model, X, y, cv=particions, scoring=metriques_cv,
                              return_train_score=True, n_jobs=TREBALLADORS,
                              error_score='raise')
    fila = {'experiment': nom, 'segons': perf_counter() - inici}
    for metrica in metriques_cv:
        fila['train_' + metrica] = resultat['train_' + metrica].mean()
        fila['cv_' + metrica] = resultat['test_' + metrica].mean()
        fila['std_' + metrica] = resultat['test_' + metrica].std(ddof=1)
    return fila

def comparar_grafic(taula, nom, titol):
    taula = taula.sort_values('cv_accuracy')
    plt.figure(figsize=(10, max(4, len(taula) * 0.42)))
    plt.barh(taula['experiment'], taula['cv_accuracy'], xerr=taula['std_accuracy'], alpha=0.8)
    plt.xlabel('Accuracy CV (barres: desviació entre folds, no interval de confiança)')
    plt.title(titol)
    plt.xlim(0.4, 1)
    guardar_grafic(nom)

# %% [markdown]
# ### 2.1. Eliminar files o omplir-les?
#
# Fem una comparació justa sobre les mateixes files de validació. En cada fold,
# una logística s'entrena amb totes les files i l'altra només amb les files
# completes de les variables base. Totes dues poden predir tota la validació
# gràcies a l'imputador. També mostrem el resultat en la part completa, però
# **no es pot comparar directament** amb una mètrica sobre tota la població.
# Exigir Cabin completa seria encara més restrictiu i canviaria molt la mostra.

# %%
print('Files completes, totes les columnes originals:', desenvolupament.dropna().shape[0])
print('Files completes, només variables base:', X[numeriques + categories].dropna().shape[0])
files_eliminacio = []
for numero, (index_train, index_val) in enumerate(folds, start=1):
    X_train = X.iloc[index_train]
    X_val = X.iloc[index_val]
    y_train = y.iloc[index_train]
    y_val = y.iloc[index_val]
    completa_train = X_train[numeriques + categories].notna().all(axis=1)
    completa_val = X_val[numeriques + categories].notna().all(axis=1)
    for politica in ['Imputar totes', 'Entrenar només completes']:
        model = fer_model(LogisticRegression(max_iter=2000))
        if politica == 'Imputar totes':
            model.fit(X_train, y_train)
            n_train = len(X_train)
        else:
            model.fit(X_train.loc[completa_train], y_train.loc[completa_train])
            n_train = int(completa_train.sum())
        pred = model.predict(X_val)
        files_eliminacio.append({'fold': numero, 'politica': politica, 'n_train': n_train,
                                 'accuracy_tota_val': accuracy_score(y_val, pred),
                                 'f1_tota_val': f1_score(y_val, pred),
                                 'accuracy_val_completa': accuracy_score(y_val.loc[completa_val], pred[completa_val]),
                                 'cobertura_val_completa': completa_val.mean()})
eliminacio = pd.DataFrame(files_eliminacio)
display(eliminacio.groupby('politica').mean(numeric_only=True).drop(columns='fold'))
eliminacio.to_csv(sortida / 'eliminacio_files.csv', index=False)
eliminacio.boxplot(column='accuracy_tota_val', by='politica')
plt.suptitle('')
plt.title('Imputar o perdre files: mateixa validació')
plt.ylabel('Accuracy')
guardar_grafic('07_eliminar_o_imputar')
comentar('Eliminar files té dos costos: menys exemples per aprendre i una mostra '
         'que pot deixar fora sistemàticament passatgers amb informació incompleta. '
         'Per això mantindrem la imputació com a política general, encara que una '
         'mètrica calculada només amb files completes pogués semblar millor.')

# %% [markdown]
# ### 2.2. Comprovació directa de la imputació d'edat
#
# Seguim P1: amaguem el 20% d'edats conegudes i intentem reconstruir-les.
# Repetim deu vegades amb les mateixes màscares per a cada mètode. Els valors
# amagats no intervenen en la mitjana, mediana ni escalat. Per KNNImputer
# escalem primer, perquè Fare no domini les distàncies.
# És una simulació de nuls aleatoris en edats conegudes, no una demostració
# que s'imputin bé les edats realment desconegudes.

# %%
resultats_imputacio = []
columnes_imputar = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
for llavor in range(10):
    rng = np.random.default_rng(llavor)
    conegudes = X.index[X['Age'].notna()]
    amagades = rng.choice(conegudes, size=int(len(conegudes) * 0.2), replace=False)
    visibles = X[columnes_imputar].copy()
    visibles.loc[amagades, 'Age'] = np.nan
    for metode in ['mean', 'median', 'knn']:
        if metode == 'knn':
            escalador = StandardScaler()
            escalades = escalador.fit_transform(visibles)
            reomplertes = KNNImputer(n_neighbors=5).fit_transform(escalades)
            reconstruides = escalador.inverse_transform(reomplertes)
            edats = pd.Series(reconstruides[:, 0], index=X.index)
        else:
            imputador = SimpleImputer(strategy=metode)
            reconstruides = imputador.fit_transform(visibles)
            edats = pd.Series(reconstruides[:, 0], index=X.index)
        error = mean_absolute_error(X.loc[amagades, 'Age'], edats.loc[amagades])
        resultats_imputacio.append({'llavor': llavor, 'metode': metode, 'MAE_anys': error})
imputacions = pd.DataFrame(resultats_imputacio)
display(imputacions.groupby('metode')['MAE_anys'].agg(['mean', 'std']))
sns.boxplot(data=imputacions, x='metode', y='MAE_anys')
plt.title('Reconstrucció d’edats ocultades artificialment')
guardar_grafic('08_imputacio_edat')
imputacions.to_csv(sortida / 'imputacio_edat.csv', index=False)
millor_imputacio = imputacions.groupby('metode')['MAE_anys'].mean().idxmin()
comentar(f"El MAE mitjà més baix correspon a **{millor_imputacio}**. "
         'Això avalua reconstrucció d’edat, no supervivència. La prova següent '
         'avalua directament classificació; els dos criteris poden donar ordres diferents.')

# %% [markdown]
# ### 2.3. Escalat, codificació i noves variables: una cosa cada vegada
#
# Per comprovar S02, mantenim KNN amb k=5 i canviem només l'escala.
# Amb logística comparem mitjana/mediana, imputació KNN, ordinal/one-hot,
# log de tarifa i informació derivada. L'ordinal de Sex/Embarked es prova
# com a contrast: els números inventen distàncies sense significat.
# Afegim PCA (95% de variància), selecció del 50% de columnes i polinomis de grau 2.
# PCA conserva variància de X, no necessàriament informació útil sobre y.
# PolynomialFeatures permet interaccions a una logística, però augmenta dimensió
# i risc de sobreajust. Totes aquestes transformacions s'ajusten dins dels folds.
# Les proves exploratòries no utilitzen el test ni defineixen per si soles el guanyador final.

# %%
experiments = []
for escala in ['cap', 'estandard', 'minmax', 'robust']:
    model = fer_model(KNeighborsClassifier(), escala=escala)
    experiments.append(avaluar_cv('KNN: escala ' + escala, model))
for metode in ['median', 'mean']:
    model = fer_model(LogisticRegression(max_iter=2000), imputacio=metode)
    experiments.append(avaluar_cv('Logística: ' + metode, model))

pre_knn = preparar(numeriques, categories)
pre_knn.set_params(num=Pipeline([('escalar', StandardScaler()), ('imputar', KNNImputer(n_neighbors=5))]))
model = Pipeline([('preparar', pre_knn), ('model', LogisticRegression(max_iter=2000))])
experiments.append(avaluar_cv('Logística: imputació KNN', model))
model = fer_model(LogisticRegression(max_iter=2000), encoding='ordinal')
experiments.append(avaluar_cv('Logística: ordinal', model))
model = fer_model(LogisticRegression(max_iter=2000),
                  num=['Age', 'LogTarifa', 'SibSp', 'Parch'])
experiments.append(avaluar_cv('Logística: log tarifa', model))
for nom, num, cat in [
    ('cabina coneguda', numeriques + ['CabinaConeguda'], categories),
    ('edat desconeguda', numeriques + ['EdatDesconeguda'], categories),
    ('títol', numeriques, cat_ampliades),
    ('família', numeriques + ['MidaFamilia', 'Sol'], categories),
    ('ampliada', num_ampliades, cat_ampliades)]:
    model = fer_model(LogisticRegression(max_iter=2000), num, cat)
    experiments.append(avaluar_cv('Logística: ' + nom, model))

for nom, transformacio in [('PCA 95%', PCA(n_components=0.95)),
                           ('selecció 50%', SelectPercentile(f_classif, percentile=50)),
                           ('grau 2', PolynomialFeatures(degree=2, include_bias=False))]:
    model = Pipeline([('preparar', preparar(numeriques, categories)),
                      ('transformar', transformacio),
                      ('model', LogisticRegression(C=1, max_iter=3000))])
    experiments.append(avaluar_cv('Logística: ' + nom, model))
preprocessament = pd.DataFrame(experiments)
display(preprocessament[['experiment', 'train_accuracy', 'cv_accuracy', 'std_accuracy', 'cv_f1']])
preprocessament.to_csv(sortida / 'preprocessament.csv', index=False)
comparar_grafic(preprocessament, '09_preprocessament', 'Hipòtesis de preprocessament, mateixos folds')
acc_sense = preprocessament.set_index('experiment').loc['KNN: escala cap', 'cv_accuracy']
acc_amb = preprocessament.set_index('experiment').loc['KNN: escala estandard', 'cv_accuracy']
comentar(f"KNN passa de **{acc_sense:.3f} sense escalar a {acc_amb:.3f} amb StandardScaler** "
         f"(canvi de {100*(acc_amb-acc_sense):+.2f} punts percentuals). "
         'Això quantifica l’efecte de canviar la geometria. No deduïm que el mateix '
         'escalador sigui sempre el millor per a tots els models.')
ordre_pre = preprocessament.sort_values('cv_accuracy', ascending=False)
comentar(f"En aquesta prova exploratòria destaca **{ordre_pre.iloc[0]['experiment']}** "
         f"amb accuracy {ordre_pre.iloc[0]['cv_accuracy']:.3f}. Les diferències s’han "
         'de llegir amb la variabilitat entre folds. Mantindrem dues representacions '
         'senzilles, base i ampliada, en la cerca principal, per comprovar també '
         'si l’efecte de les noves variables depèn del classificador.')

# %% [markdown]
# ## 3. Models i validació creuada
#
# Comparem cinc famílies, a més del DummyClassifier:
#
# - **KNN:** voten els k veïns. Pocs veïns poden seguir soroll; molts poden suavitzar massa.
# - **Logística:** suma ponderada i sigmoide; una frontera lineal en les variables transformades.
# - **SVM:** busca un marge de separació. C alt penalitza més les violacions;
#   RBF amb gamma alt fa una similitud molt local. Lineal i RBF són la mateixa família.
# - **Arbre:** divideix les dades amb preguntes successives. És fàcil d'entendre,
#   però un arbre sense límits pot memoritzar. Ens serveix per veure interaccions.
# - **Random Forest:** combina molts arbres entrenats amb mostres bootstrap i
#   subconjunts de variables. Redueix la variància d'un sol arbre, amb més cost
#   i menys interpretació directa. El provem perquè les relacions no semblen totes lineals.
#
# Primer fem servir els hiperparàmetres per defecte. Només fixem random_state,
# max_iter de logística per assegurar convergència i n_jobs per controlar recursos.
# Tots comparteixen folds i representació base. Els arbres van sense escalat.

# %%
classificadors = {
    'Dummy': DummyClassifier(strategy='most_frequent'),
    'KNN': KNeighborsClassifier(),
    'Logística': LogisticRegression(max_iter=2000),
    'SVM': SVC(),
    'Arbre': DecisionTreeClassifier(random_state=LLAVOR),
    'Random Forest': RandomForestClassifier(random_state=LLAVOR, n_jobs=1)}
models_base = {}
files_base = []
for nom, classificador in classificadors.items():
    escala = 'estandard'
    if nom in ['Arbre', 'Random Forest', 'Dummy']:
        escala = 'cap'
    model = fer_model(classificador, escala=escala)
    models_base[nom] = model
    files_base.append(avaluar_cv(nom, model))
base = pd.DataFrame(files_base).sort_values('cv_accuracy', ascending=False)
display(base[['experiment', 'train_accuracy', 'cv_accuracy', 'std_accuracy', 'cv_f1', 'segons']])
base.to_csv(sortida / 'models_per_defecte.csv', index=False)
comparar_grafic(base, '10_models_base', 'Cinc famílies i una referència sense informació')
base.set_index('experiment')[['train_accuracy', 'cv_accuracy']].plot.bar(rot=20, figsize=(10, 5))
plt.ylabel('Accuracy')
plt.title('Encert d’entrenament i encert fora de mostra')
guardar_grafic('11_train_validacio')
comentar(f"Amb valors per defecte guanya **{base.iloc[0]['experiment']}** per accuracy "
         f"({base.iloc[0]['cv_accuracy']:.3f}). Un train molt superior a CV és un "
         'símptoma de sobreajust, especialment en arbres sense límit. No descartem '
         'una família només per aquest primer resultat: encara podem regularitzar-la. '
         'Dummy sí que queda com a referència, no com a candidat útil per detectar supervivents.')

# %% [markdown]
# ### 3.1. Implementem el bucle K-fold a mà
#
# En cada volta, quatre parts entrenen i la cinquena valida. Clonem el pipeline
# perquè cap estat après passi d'un fold a l'altre. Cada passatger es valida
# exactament una vegada. Estratifiquem per mantenir una proporció similar de classes.
# La columna `test_score` de scikit-learn vol dir **validació del fold**, no el
# test final reservat. Ho distingim a totes les taules.

# %%
files_manuals = []
visites = np.zeros(len(X), dtype=int)
for numero, (index_train, index_val) in enumerate(folds, start=1):
    model = clone(models_base['Logística'])
    model.fit(X.iloc[index_train], y.iloc[index_train])
    pred_train = model.predict(X.iloc[index_train])
    pred_val = model.predict(X.iloc[index_val])
    visites[index_val] += 1
    files_manuals.append({'fold': numero, 'n_train': len(index_train), 'n_val': len(index_val),
                         'proporcio_positius_val': y.iloc[index_val].mean(),
                         'train_accuracy': accuracy_score(y.iloc[index_train], pred_train),
                         'val_accuracy': accuracy_score(y.iloc[index_val], pred_val)})
manual = pd.DataFrame(files_manuals)
display(manual)
assert (visites == 1).all()
acc_sklearn = base.set_index('experiment').loc['Logística', 'cv_accuracy']
assert np.isclose(manual['val_accuracy'].mean(), acc_sklearn)
print('La mitjana manual coincideix amb cross_validate:', manual['val_accuracy'].mean())

# %% [markdown]
# ### 3.2. Què passa si canviem folds o llavor?
#
# Provem 3, 5 i 10 folds, amb quatre llavors. No escollim la partició que dona
# millor nota: és una prova d'estabilitat amb logística fixada. Més folds vol
# dir més ajustos i un entrenament més gran en cada ajust; no garanteix una
# estimació millor en una mostra concreta. Les puntuacions dels folds no són
# independents, perquè els conjunts d'entrenament se solapen.

# %%
files_estabilitat = []
for nombre_folds in [3, 5, 10]:
    for llavor in [0, 7, 42, 123]:
        cv = StratifiedKFold(n_splits=nombre_folds, shuffle=True, random_state=llavor)
        fila = avaluar_cv('Logística', models_base['Logística'], cv)
        fila['folds'] = nombre_folds
        fila['llavor'] = llavor
        files_estabilitat.append(fila)
estabilitat = pd.DataFrame(files_estabilitat)
display(estabilitat[['folds', 'llavor', 'cv_accuracy', 'std_accuracy', 'cv_f1', 'segons']])
sns.lineplot(data=estabilitat, x='folds', y='cv_accuracy', hue='llavor', marker='o', palette='tab10')
plt.xticks([3, 5, 10])
plt.title('El resultat també depèn de la partició')
guardar_grafic('12_folds_llavors')
estabilitat.to_csv(sortida / 'estabilitat.csv', index=False)
comentar(f"En aquesta logística, les mitjanes d’accuracy van de "
         f"{estabilitat['cv_accuracy'].min():.3f} a {estabilitat['cv_accuracy'].max():.3f}. "
         'Aquesta variació ajuda a posar en perspectiva millores molt petites. '
         'Mantenim els cinc folds inicials per no escollir la llavor amb el millor resultat.')

# %% [markdown]
# ### 3.3. Cerca d'hiperparàmetres
#
# Fem GridSearchCV: provem totes les combinacions de la graella. Incloem la
# representació base i ampliada. L'imputació és mediana i l'encoding one-hot.
# Provem C de 0.1 a 10, k entre 3 i 21, pesos de classe, profunditat i mida
# mínima de fulla. Són graelles petites i comprensibles, no una garantia d'òptim global.
#
# Es calculen totes les mètriques en els mateixos ajustos. En aquest apartat
# ordenem per accuracy; a l'apartat 4 reconsiderarem el resultat segons F1.
# Guardem **totes** les configuracions, no només la guanyadora. Els temps són
# els d'aquesta màquina i no una comparació de maquinari universal.

# %%
graelles = {
    'KNN': {'model__n_neighbors': [3, 5, 9, 15, 21],
            'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]},
    'Logística': {'model__C': [0.1, 1, 10], 'model__class_weight': [None, 'balanced']},
    'SVM': [
        {'model__kernel': ['linear'], 'model__C': [0.1, 1, 10],
         'model__class_weight': [None, 'balanced']},
        {'model__kernel': ['rbf'], 'model__C': [0.1, 1, 10],
         'model__gamma': ['scale', 0.01, 0.1], 'model__class_weight': [None, 'balanced']}],
    'Arbre': {'model__max_depth': [3, 5, 8, None],
              'model__min_samples_leaf': [1, 5, 10], 'model__class_weight': [None, 'balanced']},
    'Random Forest': {'model__n_estimators': [100], 'model__max_depth': [5, 10, None],
                     'model__min_samples_leaf': [1, 3, 6], 'model__max_features': ['sqrt', 0.8]}}
cerques = {}
resultats_graella = []
for nom in graelles:
    inici = perf_counter()
    escala = 'estandard'
    if nom in ['Arbre', 'Random Forest']:
        escala = 'cap'
    graella = graelles[nom]
    if isinstance(graella, dict):
        graella = [graella]
    graella_completa = []
    for part in graella:
        nova_part = part.copy()
        nova_part['preparar'] = [preparar(numeriques, categories, escala),
                                 preparar(num_ampliades, cat_ampliades, escala)]
        graella_completa.append(nova_part)
    cerca = GridSearchCV(models_base[nom], graella_completa, cv=folds,
                         scoring=metriques_cv, refit='accuracy', n_jobs=TREBALLADORS,
                         return_train_score=True, error_score='raise')
    cerca.fit(X, y)
    cerques[nom] = cerca
    taula = pd.DataFrame(cerca.cv_results_)
    taula['familia'] = nom
    representacions = []
    parametres_llegibles = []
    for parametres in taula['params']:
        pre = parametres['preparar']
        if len(pre.transformers[0][2]) == len(numeriques):
            representacions.append('base')
        else:
            representacions.append('ampliada')
        text_params = {}
        for clau, valor in parametres.items():
            if clau != 'preparar':
                text_params[clau.replace('model__', '')] = valor
        parametres_llegibles.append(str(text_params))
    taula['representacio'] = representacions
    taula['parametres'] = parametres_llegibles
    taula['segons_cerca'] = perf_counter() - inici
    resultats_graella.append(taula)
    print(nom, ':', len(taula), 'configuracions;', round(perf_counter() - inici, 1), 'segons')

graella_total = pd.concat(resultats_graella, ignore_index=True)
columnes_guardar = ['familia', 'representacio', 'parametres', 'mean_fit_time', 'segons_cerca']
for metrica in metriques_cv:
    columnes_guardar += ['mean_train_' + metrica, 'mean_test_' + metrica, 'std_test_' + metrica]
graella_total[columnes_guardar].to_csv(sortida / 'totes_les_configuracions.csv', index=False)
millors_accuracy = graella_total.sort_values('mean_test_accuracy', ascending=False).groupby('familia', sort=False).head(1)
display(millors_accuracy[['familia', 'representacio', 'parametres', 'mean_train_accuracy',
                          'mean_test_accuracy', 'std_test_accuracy', 'mean_test_f1']])
print('Configuracions totals:', len(graella_total))
print('Ajustos de CV de la graella:', len(graella_total) * len(folds))

# %% [markdown]
# ### 3.4. Veure per què canvia el resultat
#
# Els gràfics següents fixen els altres paràmetres: no barregen arbitràriament
# C, pesos i representacions. En KNN busquem la relació entre k i generalització;
# en l'arbre, la distància entre train i validació; en SVM, la interacció C/gamma.
# El punt amb més accuracy de train no té per què ser el que convé.

# %%
knn = pd.DataFrame(cerques['KNN'].cv_results_)
knn = knn[(knn['param_model__weights'] == 'uniform') & (knn['param_model__p'] == 2)].copy()
# La primera representació de cada configuració és la base; la identifiquem per columnes.
es_base = []
for pre in knn['param_preparar']:
    es_base.append(len(pre.transformers[0][2]) == 4)
knn = knn.loc[es_base].sort_values('param_model__n_neighbors')
plt.plot(knn['param_model__n_neighbors'].astype(int), knn['mean_train_accuracy'], 'o-', label='Train')
plt.plot(knn['param_model__n_neighbors'].astype(int), knn['mean_test_accuracy'], 'o-', label='Validació')
plt.xlabel('k veïns')
plt.ylabel('Accuracy')
plt.title('KNN base, distància euclidiana i vot uniforme')
plt.legend()
guardar_grafic('13_knn_k')

svm = pd.DataFrame(cerques['SVM'].cv_results_)
svm = svm[(svm['param_model__kernel'] == 'rbf') & svm['param_model__class_weight'].isna()].copy()
es_base = []
for pre in svm['param_preparar']:
    es_base.append(len(pre.transformers[0][2]) == 4)
svm = svm.loc[es_base]
sns.heatmap(svm.pivot(index='param_model__C', columns='param_model__gamma', values='mean_test_accuracy'),
            annot=True, fmt='.3f', cmap='Blues')
plt.title('SVM RBF base: accuracy de validació segons C i gamma')
guardar_grafic('14_svm_c_gamma')

arbre = pd.DataFrame(cerques['Arbre'].cv_results_)
arbre = arbre[(arbre['param_model__min_samples_leaf'] == 1) & arbre['param_model__class_weight'].isna()].copy()
es_base = []
for pre in arbre['param_preparar']:
    es_base.append(len(pre.transformers[0][2]) == 4)
arbre = arbre.loc[es_base]
etiquetes = arbre['param_model__max_depth'].fillna('Sense límit').astype(str)
plt.plot(etiquetes, arbre['mean_train_accuracy'], 'o-', label='Train')
plt.plot(etiquetes, arbre['mean_test_accuracy'], 'o-', label='Validació')
plt.xlabel('Profunditat màxima')
plt.ylabel('Accuracy')
plt.title('Un arbre més profund pot memoritzar')
plt.legend()
guardar_grafic('15_profunditat_arbre')

# %% [markdown]
# ### 3.5. Grid Search o Random Search?
#
# Fem una prova petita i transparent: RandomizedSearchCV avalua 12 de les
# configuracions SVM de la mateixa graella. Mesurem temps i resultat.
# **No és una comparació amb igual pressupost temporal** ni hem executat la
# prova opcional d'una hora. Grid és exhaustiu dins d'aquesta graella; random
# pot explorar espais més grans amb menys combinacions. Amb una hora i moltes
# dimensions, començaria amb random i després afinaria prop de bones zones.
# La cerca bayesiana usa resultats anteriors per proposar candidats; successive
# halving reparteix recursos i elimina candidats. No els hem implementat aquí.

# %%
inici = perf_counter()
cerca_aleatoria = RandomizedSearchCV(
    models_base['SVM'], cerques['SVM'].param_grid, n_iter=12,
    scoring='accuracy', cv=folds, random_state=LLAVOR, n_jobs=TREBALLADORS,
    return_train_score=True, error_score='raise')
cerca_aleatoria.fit(X, y)
temps_random = perf_counter() - inici
comparacio_cerca = pd.DataFrame([
    {'metode': 'Grid SVM', 'candidats': len(cerques['SVM'].cv_results_['params']),
     'accuracy_cv': cerques['SVM'].best_score_,
     'segons': graella_total.loc[graella_total['familia'] == 'SVM', 'segons_cerca'].iloc[0]},
    {'metode': 'Random SVM', 'candidats': 12,
     'accuracy_cv': cerca_aleatoria.best_score_, 'segons': temps_random}])
display(comparacio_cerca)
comparacio_cerca.to_csv(sortida / 'metodes_cerca.csv', index=False)
comentar('La cerca aleatòria és una prova de pressupost reduït; no l’afegim '
         'com una sisena família. Els seus candidats ja formen part de la graella '
         'exhaustiva, així que no amplia l’espai de solucions de la selecció final.')

# %% [markdown]
# ## 4. Quina mètrica ens interessa i què amaga l'accuracy?
#
# Considerem positiu `Survived=1`. Un FP és predir supervivència per a algú que
# no va sobreviure; un FN és no detectar un supervivent real. Aquí reconstruïm
# dades històriques: no hi ha un cost operatiu definit ni una política de rescat.
#
# - Accuracy: (TP + TN) / N. Tots els encerts pesen igual.
# - Precision: TP / (TP + FP). De les supervivències predites, quantes són certes?
# - Recall: TP / (TP + FN). Quants supervivents reals detectem?
# - F1: 2PR / (P + R). Equilibra les dues anteriors, però ignora TN.
# - ROC: recall contra FP / (FP + TN), per a tots els llindars.
# - PR: precision contra recall. Fa més visible el rendiment de la classe positiva.
# - Average precision (AP): resumeix precision ponderant increments de recall.
#   No és exactament l'àrea trapezoidal sota la corba PR.
#
# Priorizem F1 pel criteri indicat al principi; accuracy continua sent útil
# perquè el desequilibri no és extrem. ROC-AUC avalua ordenació, no encert amb
# un llindar ni calibratge. La referència PR és la prevalença; en ROC és la
# diagonal. Donem més protagonisme a PR per analitzar supervivents, sense amagar ROC.
# `classification_report` mostra precision, recall, F1 i suport de cada classe,
# a més de mitjanes macro i ponderades; el suport és el nombre de casos reals.

# %%
def resum_metriques(real, predit, puntuacions):
    return {'accuracy': accuracy_score(real, predit),
            'balanced_accuracy': balanced_accuracy_score(real, predit),
            'precision': precision_score(real, predit, zero_division=0),
            'recall': recall_score(real, predit, zero_division=0),
            'f1': f1_score(real, predit, zero_division=0),
            'roc_auc': roc_auc_score(real, puntuacions),
            'average_precision': average_precision_score(real, puntuacions)}

def mostrar_corbes(real, puntuacions, titol, nom_fitxer):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    RocCurveDisplay.from_predictions(real, puntuacions, ax=eixos[0], name=titol)
    eixos[0].plot([0, 1], [0, 1], '--', color='grey', label='Aleatori')
    eixos[0].set_title('ROC: detecció i falsos positius')
    eixos[0].legend(fontsize=8)
    PrecisionRecallDisplay.from_predictions(real, puntuacions, ax=eixos[1], name=titol)
    eixos[1].axhline(np.mean(real), linestyle='--', color='grey', label='Prevalença')
    eixos[1].set_title('PR: detecció i fiabilitat dels positius')
    eixos[1].legend(fontsize=8)
    guardar_grafic(nom_fitxer)

def mostrar_confusio(real, predit, titol, nom_fitxer):
    ConfusionMatrixDisplay.from_predictions(real, predit,
        display_labels=['No sobreviu', 'Sobreviu'], cmap='Blues', colorbar=False)
    plt.title(titol)
    guardar_grafic(nom_fitxer)

# La logística d'aquesta demostració té paràmetres fixos, no triats amb y.
prob_logistica = cross_val_predict(models_base['Logística'], X, y, cv=folds,
                                   method='predict_proba', n_jobs=TREBALLADORS)[:, 1]
pred_logistica = (prob_logistica >= 0.5).astype(int)
display(pd.DataFrame([resum_metriques(y, pred_logistica, prob_logistica)]))
print(classification_report(y, pred_logistica, target_names=['No sobreviu', 'Sobreviu'], zero_division=0))
mostrar_corbes(y, prob_logistica, 'Logística OOF', '16_logistica_roc_pr')
mostrar_confusio(y, pred_logistica, 'Logística: prediccions fora del fold d’entrenament', '17_logistica_confusio')

# %% [markdown]
# ### 4.1. El llindar canvia els errors, no l'ordenació
#
# Les probabilitats anteriors són out-of-fold (OOF): cada fila es prediu amb un
# model que no l'ha entrenada. Mantenim aquests scores i variem el llindar.
# Baixar-lo tendeix a augmentar recall i falsos positius; precision no ha de
# canviar monòtonament en una mostra finita. L'F1 màxim d'aquesta taula seria
# optimista si l'uséssim alhora per seleccionar i avaluar el llindar. Per això
# aquesta prova és explicativa i mantenim la política final per defecte.

# %%
files_llindar = []
for llindar in np.arange(0.10, 0.91, 0.05):
    pred = (prob_logistica >= llindar).astype(int)
    fila = resum_metriques(y, pred, prob_logistica)
    fila['llindar'] = llindar
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    fila.update({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
    files_llindar.append(fila)
llindars = pd.DataFrame(files_llindar)
display(llindars[['llindar', 'precision', 'recall', 'f1', 'accuracy', 'FP', 'FN']])
llindars.plot(x='llindar', y=['precision', 'recall', 'f1', 'accuracy'], marker='o')
plt.title('Mateixos scores OOF, decisions diferents')
plt.ylabel('Mètrica')
guardar_grafic('18_llindars')
llindars.to_csv(sortida / 'llindars_logistica.csv', index=False)

# %% [markdown]
# ### 4.2. Es manté el millor model quan prioritzem F1?
#
# Tornem a ordenar **tots els candidats** per F1, no només els que havien
# guanyat amb accuracy. Si només canviéssim l'etiqueta de la mètrica del millor
# model anterior, no estaríem fent una selecció coherent.
#
# Les puntuacions d'aquesta cerca són de selecció i poden ser optimistes per
# haver provat moltes opcions. No les presentem com una avaluació independent.
# El test reservat donarà l'avaluació final, després de tancar les decisions.

# %%
millors_f1 = graella_total.sort_values('mean_test_f1', ascending=False).groupby('familia', sort=False).head(1)
display(millors_f1[['familia', 'representacio', 'parametres', 'mean_train_f1',
                    'mean_test_f1', 'std_test_f1', 'mean_test_accuracy', 'mean_test_auc', 'mean_test_ap']])
comparacio_metrica = millors_accuracy[['familia', 'mean_test_accuracy', 'mean_test_f1']].merge(
    millors_f1[['familia', 'mean_test_accuracy', 'mean_test_f1']], on='familia',
    suffixes=('_triat_accuracy', '_triat_f1'))
display(comparacio_metrica)
comparacio_metrica.to_csv(sortida / 'accuracy_contra_f1.csv', index=False)
millors_f1.set_index('familia')[['mean_test_accuracy', 'mean_test_f1', 'mean_test_ap', 'mean_test_auc']].plot.bar(
    rot=20, figsize=(11, 5))
plt.ylim(0.4, 1)
plt.ylabel('Mitjana CV')
plt.title('Configuració de cada família seleccionada per F1')
plt.legend(['Accuracy', 'F1', 'AP', 'ROC-AUC'], ncol=4)
guardar_grafic('19_comparacio_metriques')

guanyador_accuracy = millors_accuracy.iloc[0]
guanyador_f1 = millors_f1.iloc[0]
nom_final = guanyador_f1['familia']
cerca_final = cerques[nom_final]
index_final = int(np.argmax(cerca_final.cv_results_['mean_test_f1']))
parametres_final = cerca_final.cv_results_['params'][index_final]
model_final = clone(cerca_final.estimator).set_params(**parametres_final)
comentar(f"Per accuracy guanya **{guanyador_accuracy['familia']}** amb "
         f"{guanyador_accuracy['mean_test_accuracy']:.3f}. Per F1 escollim **{nom_final}**, "
         f"representació **{guanyador_f1['representacio']}**, amb F1 CV "
         f"**{guanyador_f1['mean_test_f1']:.3f}** i accuracy CV "
         f"{guanyador_f1['mean_test_accuracy']:.3f}. La configuració queda fixada ara: "
         f"`{guanyador_f1['parametres']}`. El test no ha intervingut en aquesta decisió.")

# %% [markdown]
# ### 4.3. Aprèn més si li donem més dades?
#
# Una corba d'aprenentatge compara train i validació amb diferents mides.
# La fem sobre desenvolupament amb la configuració ja escollida. És un
# diagnòstic de selecció, no una nova avaluació independent. Si train es manté
# alt i validació queda més baixa, encara hi ha variància o informació que falta.

# %%
mides, puntuacions_train, puntuacions_val = learning_curve(
    model_final, X, y, cv=folds, scoring='f1', train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    shuffle=True, random_state=LLAVOR, n_jobs=TREBALLADORS)
plt.plot(mides, puntuacions_train.mean(axis=1), 'o-', label='Train')
plt.plot(mides, puntuacions_val.mean(axis=1), 'o-', label='Validació')
plt.fill_between(mides, puntuacions_val.mean(axis=1) - puntuacions_val.std(axis=1),
                 puntuacions_val.mean(axis=1) + puntuacions_val.std(axis=1), alpha=0.2)
plt.xlabel('Passatgers d’entrenament en cada fold')
plt.ylabel('F1')
plt.title('Corba d’aprenentatge: ' + nom_final)
plt.legend()
guardar_grafic('20_corba_aprenentatge')
display(pd.DataFrame({'n_train': mides, 'f1_train': puntuacions_train.mean(axis=1),
                      'f1_validacio': puntuacions_val.mean(axis=1)}))

# %% [markdown]
# ### 4.4. El problema dels passatgers que comparteixen bitllet
#
# Una partició per passatger pot posar companys de viatge en train i validació.
# Provem StratifiedGroupKFold agrupant per Ticket. És una pregunta més exigent:
# com funciona el model en bitllets no vistos? Ticket és un indicador imperfecte
# de família; no detecta tots els vincles. El test principal continua mesurant
# passatgers aleatoris del mateix fitxer, no famílies completament noves.
# No canviem de model després d'aquesta prova: és un diagnòstic de sensibilitat.

# %%
grups = desenvolupament['Ticket'].astype(str)
cv_grups = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=LLAVOR)
folds_grups = list(cv_grups.split(X, y, groups=grups))
for index_train, index_val in folds_grups:
    assert set(grups.iloc[index_train]).isdisjoint(set(grups.iloc[index_val]))
sensibilitat = pd.DataFrame([
    avaluar_cv('Passatgers aleatoris', model_final),
    avaluar_cv('Bitllets separats', model_final, folds_grups)])
display(sensibilitat[['experiment', 'cv_accuracy', 'std_accuracy', 'cv_f1', 'std_f1']])
sensibilitat.to_csv(sortida / 'validacio_per_bitllet.csv', index=False)
comparar_grafic(sensibilitat, '21_grups_bitllet', 'Canviar la unitat de validació canvia la pregunta')
comentar(f"F1 per passatgers: {sensibilitat.iloc[0]['cv_f1']:.3f}; "
         f"per bitllets separats: {sensibilitat.iloc[1]['cv_f1']:.3f}. "
         'A més dels grups canvien els folds, de manera que la diferència no aïlla '
         'una única causa. La configuració havia estat triada amb CV per passatgers: '
         'la prova de grups no és una estimació de tot el procés de selecció per grups.')

# %% [markdown]
# ## 5. Test final: obrim les dades reservades
#
# Model, variables, hiperparàmetres, mètrica i política de decisió ja estan fixats.
# Entrenem amb tot desenvolupament i prediem el test una vegada. A partir
# d'aquestes mateixes prediccions fem el report, corbes i auditoria d'errors.
# No tornem a ajustar el model per millorar aquests números.
# Si guanya SVM, fem servir decision_function per a ROC/PR: és un score,
# no una probabilitat. La classificació es fa amb `predict`, llindar intern 0.
# Per als altres models el llindar habitual de probabilitat és 0.5 (amb les
# convencions pròpies d'empat de cada implementació).

# %%
X_test = crear_variables(test_reservat.drop(columns='Survived'))
y_test = test_reservat['Survived'].copy()
model_final.fit(X, y)
pred_test = model_final.predict(X_test)
if hasattr(model_final, 'predict_proba'):
    score_test = model_final.predict_proba(X_test)[:, 1]
    tipus_score = 'probabilitat estimada, no calibrada explícitament'
else:
    score_test = model_final.decision_function(X_test)
    tipus_score = 'decision_function, no probabilitat'
final = resum_metriques(y_test, pred_test, score_test)
display(pd.DataFrame([final], index=[nom_final + ' | test reservat']))
print(classification_report(y_test, pred_test, target_names=['No sobreviu', 'Sobreviu'], zero_division=0))
mostrar_corbes(y_test, score_test, nom_final + ' test', '22_final_roc_pr')
mostrar_confusio(y_test, pred_test, nom_final + ': test final', '23_final_confusio')
tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
comentar(f"Al test de **{len(y_test)} passatgers**, el model encerta "
         f"**{int((pred_test == y_test).sum())} casos ({final['accuracy']:.1%})**. "
         f"Detecta **{tp} de {tp+fn} supervivents** (recall {final['recall']:.1%}); "
         f"de les {tp+fp} supervivències predites, {tp} són correctes "
         f"(precision {final['precision']:.1%}). Hi ha **{fp} falsos positius i {fn} falsos negatius**. "
         f"F1 és {final['f1']:.3f}, ROC-AUC {final['roc_auc']:.3f} i AP "
         f"{final['average_precision']:.3f}, davant una prevalença de {y_test.mean():.3f}.")
comentar('La corba ROC mostra fins a quin punt ordenem supervivents per damunt '
         'de no supervivents. La corba PR mostra el compromís quan volem recuperar '
         'més supervivents: cal llegir si la precision baixa massa. Una àrea alta '
         'no implica haver encertat totes les persones ni tenir probabilitats calibrades.')

# %% [markdown]
# ### 5.1. Incertesa: no tractem 179 persones com si fossin tota la població
#
# Fem 1.000 remostrejos bootstrap de parelles (etiqueta, predicció) del test,
# mantenint el model fix. Els percentils 2.5 i 97.5 donen intervals orientatius.
# No inclouen la incertesa de tornar a entrenar ni de triar models. A més,
# el bootstrap per passatger assumeix independència, limitada pels companys
# de bitllet. No són garanties per a altres vaixells o poblacions.

# %%
rng = np.random.default_rng(2026)
files_bootstrap = []
real = y_test.to_numpy()
for repeticio in range(1000):
    indexs = rng.integers(0, len(real), size=len(real))
    if len(np.unique(real[indexs])) == 2:
        files_bootstrap.append(resum_metriques(real[indexs], pred_test[indexs], score_test[indexs]))
bootstrap = pd.DataFrame(files_bootstrap)
intervals = pd.DataFrame({'resultat': pd.Series(final),
                          'percentil_2_5': bootstrap.quantile(0.025),
                          'percentil_97_5': bootstrap.quantile(0.975)})
display(intervals)
intervals.to_csv(sortida / 'intervals_test.csv')
fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
for ax, metrica in zip(eixos, ['accuracy', 'f1']):
    ax.hist(bootstrap[metrica], bins=25, alpha=0.8)
    ax.axvline(final[metrica], color='black', linestyle='--')
    ax.set(title='Bootstrap: ' + metrica, xlabel='Mètrica', ylabel='Remostrejos')
guardar_grafic('24_incertesa_bootstrap')

# %% [markdown]
# ### 5.2. On ens equivoquem? Sexe, classe, edat coneguda i infants
#
# La mitjana global pot amagar diferències. Mostrem N, positius reals, TP,
# FP, FN, TN i mètriques. Si un grup no té positius, el recall és indefinit:
# el mostrem com NaN, no com un zero interpretat com mal rendiment.
# Aquesta auditoria no s'utilitza per reajustar el model. Els grups petits
# tenen molta incertesa; comparar-los no estableix causes ni una prova de justícia.

# %%
prediccions = test_reservat[['PassengerId', 'Sex', 'Pclass', 'Age', 'Cabin', 'Survived']].copy()
prediccions['predit'] = pred_test
prediccions['score'] = score_test
prediccions['error'] = prediccions['predit'] != prediccions['Survived']
prediccions['EdatConeguda'] = np.where(prediccions['Age'].notna(), 'Sí', 'No')
prediccions['GrupEdat'] = 'Adult (16 o més)'
prediccions.loc[prediccions['Age'] < 16, 'GrupEdat'] = 'Infant (<16)'
prediccions.loc[prediccions['Age'].isna(), 'GrupEdat'] = 'Desconeguda'
files_grups = []
for columna in ['Sex', 'Pclass', 'EdatConeguda', 'GrupEdat']:
    for valor, grup in prediccions.groupby(columna):
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(grup['Survived'], grup['predit'], labels=[0, 1]).ravel()
        recall_g = np.nan
        precision_g = np.nan
        if tp_g + fn_g > 0:
            recall_g = tp_g / (tp_g + fn_g)
        if tp_g + fp_g > 0:
            precision_g = tp_g / (tp_g + fp_g)
        files_grups.append({'variable': columna, 'grup': str(valor), 'n': len(grup),
                            'positius': int(grup['Survived'].sum()), 'TP': tp_g, 'FP': fp_g,
                            'FN': fn_g, 'TN': tn_g, 'recall': recall_g, 'precision': precision_g,
                            'accuracy': accuracy_score(grup['Survived'], grup['predit'])})
subgrups = pd.DataFrame(files_grups)
display(subgrups)
subgrups.to_csv(sortida / 'errors_per_subgrup.csv', index=False)
fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
for columna, ax in zip(['Sex', 'Pclass'], eixos):
    part = subgrups[subgrups['variable'] == columna]
    ax.bar(part['grup'], part['recall'])
    ax.set(title='Recall segons ' + columna, ylim=(0, 1.15), ylabel='Recall de supervivents')
    for posicio, (_, fila) in enumerate(part.iterrows()):
        ax.text(posicio, fila['recall'] + 0.03,
                str(int(fila['TP'])) + '/' + str(int(fila['positius'])), ha='center')
guardar_grafic('25_errors_subgrups')
display(prediccions.loc[prediccions['error']].head(12))
comentar('Els exemples d’error permeten veure casos concrets, però no explicar '
         'per què aquella persona va sobreviure. No tenim posició exacta, accés '
         'als bots ni totes les circumstàncies del naufragi. El model només '
         'resumeix patrons de les variables disponibles.')

# %% [markdown]
# ### 5.3. Quines variables utilitza el model?
#
# Fem importància per permutació al test com a explicació **posterior**, no
# com a selecció de variables. Barregem una columna original i observem la
# caiguda de F1. Recalculem les variables derivades abans de cada predicció:
# si barregem Sex però deixéssim una còpia derivada de Sex, la prova seria enganyosa.
# Les variables correlacionades es poden substituir; una importància baixa
# no vol dir que una variable sigui irrellevant. Les combinacions permutades
# poden ser poc realistes, així que no són efectes causals.

# %%
# Codi explícit per tornar a calcular totes les derivades en cada permutació.
files_importancia = []
rng = np.random.default_rng(42)
for columna in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'Name']:
    caigudes = []
    for repeticio in range(10):
        barrejat = test_reservat.drop(columns='Survived').copy()
        barrejat[columna] = rng.permutation(barrejat[columna].to_numpy())
        pred_barrejat = model_final.predict(crear_variables(barrejat))
        caigudes.append(final['f1'] - f1_score(y_test, pred_barrejat))
    files_importancia.append({'variable': columna, 'caiguda_f1': np.mean(caigudes),
                              'std': np.std(caigudes, ddof=1)})
importancies = pd.DataFrame(files_importancia).sort_values('caiguda_f1')
display(importancies)
plt.barh(importancies['variable'], importancies['caiguda_f1'], xerr=importancies['std'])
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Caiguda de F1 en barrejar la variable (mitjana i desviació)')
plt.title('Dependència predictiva, no causalitat')
guardar_grafic('26_importancia_permutacio')
importancies.to_csv(sortida / 'importancies.csv', index=False)

# %% [markdown]
# ### 5.4. Conclusió i límits
#
# El resultat principal no és haver trobat un model perfecte. Hem passat d'un
# CSV amb buits i categories a un protocol que es pot explicar i repetir.
# Hem comprovat l'efecte d'escalar, eliminar files, imputar, afegir informació,
# canviar complexitat, canviar folds i canviar la mètrica de selecció.
#
# La cerca només cobreix les graelles declarades. Provar molts models amb els
# mateixos folds fa optimistes els resultats de selecció; el test independent
# redueix aquest problema, però té poques files i pot compartir bitllets amb
# desenvolupament. La prova per grups ensenya per què importa definir què és un cas nou.
#
# El títol, la classe i la cabina poden actuar com a indicadors de condicions
# socials i de com es van registrar les dades. L'absència de cabina pot no
# reproduir-se en un altre registre. No sabem si la mostra representa tota
# la població del vaixell. No atribuïm els errors a defectes personals.
#
# **Ús possible:** una demostració educativa o una eina per explorar patrons
# d'aquest registre històric. **Condició de no ús:** no extrapolar a seguretat
# marítima actual ni a decisions de rescat o valoració individual de persones.
#
# **Millores que tindrien sentit:** validar per grups des del principi si la
# pregunta són famílies noves; fer validació creuada niada de tota la selecció;
# aconseguir més dades i millor documentació dels nuls; calibrar probabilitats
# amb validació separada si s'han d'interpretar com risc; fixar costos FP/FN
# abans d'optimitzar un llindar. No hem aplicat aquestes millores després de veure
# el test, perquè llavors el test deixaria de ser independent.

# %%
resum = {'model': nom_final, 'representacio': guanyador_f1['representacio'],
         'parametres': guanyador_f1['parametres'], 'n_desenvolupament': len(X),
         'n_test': len(X_test), 'configuracions_graella': len(graella_total),
         'ajustos_graella': len(graella_total) * 5,
         'cv_f1_seleccio': float(guanyador_f1['mean_test_f1']),
         'cv_accuracy_seleccio': float(guanyador_f1['mean_test_accuracy']),
         'test': final, 'confusio': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
         'tipus_score': tipus_score, 'sha256_dades': empremta, 'versions': versions}
(sortida / 'resum.json').write_text(json.dumps(resum, indent=2, ensure_ascii=False), encoding='utf-8')
prediccions.to_csv(sortida / 'prediccions_test.csv', index=False)
comentar(f"**Resposta final:** escollim {nom_final} amb variables {guanyador_f1['representacio']}. "
         f"El test dona accuracy **{final['accuracy']:.3f}** i F1 **{final['f1']:.3f}**. "
         f"Hem comparat **{len(graella_total)} configuracions** en la cerca principal "
         f"(**{len(graella_total)*5} ajustos de CV**), a més de les proves de preprocessament "
         'i estabilitat. La decisió es basa en F1 de desenvolupament; els errors i '
         'els intervals del test mostren el límit d’aquesta conclusió.')

# %% [markdown]
# ## Referències i transparència
#
# Materials locals: enunciat `APC_Practica_1_2026.pdf`, notebook de primers
# passos, notebooks P1-P3 i sessions S01-S08 indicades al principi.
# Les definicions del problema i de les variables s'han contrastat amb el
# material proporcionat i el mateix CSV.
#
# Documentació oficial consultada:
# [Pipeline i ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html),
# [mètriques](https://scikit-learn.org/stable/modules/model_evaluation.html),
# [cerca d'hiperparàmetres](https://scikit-learn.org/stable/modules/grid_search.html).
# Les versions exactes de l'execució es mostren al principi i a `resultats/resum.json`.
#
# **Ús d'IA:** aquesta versió s'ha elaborat amb assistència de Codex per revisar
# materials, escriure codi, executar experiments i redactar explicacions.
# Els resultats numèrics provenen de l'execució del codi sobre el CSV indicat.
# Abans del lliurament, els autors han de revisar el contingut, completar els
# noms i NIU i poder explicar les decisions. Aquesta declaració no afirma
# que aquesta revisió personal ja s'hagi fet.
