# Materials de la carpeta i ús en la pràctica

## Identificació de l'encàrrec

`APC_Practica_1_2026/APC_Practica_1_2026.pdf` és l'enunciat que coincideix amb l'encàrrec: classificació de supervivència del Titanic, cinc apartats, mínim quatre famílies, CV i cerca, mètriques i anàlisi final.

`APC_Practica_1_2026/00_Practica1_primers_passos_colab.ipynb` dona el punt de partida: cabina registrada, distinció entre absència i dada perduda, exclusió d'identificadors, baseline, KNN escalat, títol del nom i variabilitat de particions. El fitxer de treball és `titanic_train.csv`: 891 files i 12 columnes.

## Sessions pràctiques

| Carpeta | Contingut | Relació amb la solució |
|---|---|---|
| APC_P1_Encoding_i_NaNs | Notebook explicatiu, plantilla Problema 1, Ames `train.csv`, descripció i `sensor.csv` | Semàntica dels buits, encoding, imputació ajustada amb train i ocultació de valors coneguts per mesurar MAE |
| APC_P2_Regressio_v2 | Notebook explicatiu, plantilla Problema 2, Ames `train.csv` i descripció | Diferència entre ajustar i generalitzar, regularització, complexitat i polinomis |
| APC_P3_Classificacio | Notebook explicatiu, plantilla Problema 3, `diabetes.csv` i descripció | Classificació, mètriques, errors i restriccions de la població |

Ames té 1.460 files i 81 columnes en els dos fitxers. El sensor té 10.080 lectures i dues columnes. Diabetis té 768 files i nou columnes. Aquests fitxers s'han identificat com a exemples d'altres exercicis: no formen part de X o y del Titanic. «Problema 1» d'encoding i «Pràctica 1» de classificació són encàrrecs diferents.

## Teoria, PDF i notebooks associats

| Sessió | Fitxers | Comprovació concreta |
|---|---|---|
| S01 | `APC_S01_From_Atoms_to_Coordinates_Aula90_v6.pdf`, `01_De_l_objecte_al_vector_v3_alineat_colab.ipynb` | Diccionari de dades i diferència entre tipus numèric i significat |
| S02 | `APC_S02_KNN_Geometric_Design_Aula90_v6.pdf`, `02_Distancies_escala_i_KNN_v3_alineat_colab.ipynb` | KNN amb quatre escales, distància p=1/2 i nombre de veïns |
| S03 | `APC_S03_Regression_Cost_Topogrophy_v6.1.pdf`, `03_Regressio_model_residu_i_cost_v3_alineat_colab.ipynb` | Funció d'error i diferència entre reconstruir edat i classificar supervivència |
| S04 | `APC_S04_Optimizing_Generalization_v6.1.pdf`, `04_Gradient_regularitzacio_i_generalitzacio_v3_alineat_colab.ipynb` | Regularització C, train contra CV, polinomis i corba d'aprenentatge |
| S05 | `APC_S05_Defensible_Model_Validation_v6.0.pdf`, `05_Validacio_v5.0.ipynb` | Split reservat, pipeline dins dels folds, CV manual i sensibilitat a grups |
| S06 | `APC_S06_Decision_Engineering_v6.0.pdf`, `06_Llindars_v5.0.ipynb` | Prediccions OOF i efecte del llindar amb scores fixos |
| S07 | `APC_S07_Anatomy_of_Error_v6.3.pdf`, `07_Metriques_imbalance_i_subgrups_v3_alineat_colab.ipynb` | Denominadors de precision/recall, ROC/PR, recompte d'errors per grup |
| S08 | `APC_S08_The_SVM_Blueprint_v6.0.pdf`, `08_SVM_marge_regularitzacio_i_kernels_v3_alineat_colab.ipynb` | C, gamma, marge, kernel lineal/RBF i score diferent de probabilitat |

S01-S02 són a `Documentació de la sessió de teoria-20261012`, S03-S04 a la variant `(1)`, S05-S06 a `(2)` i S07-S08 a `(3)`. Els vuit ZIP corresponen a aquestes carpetes i a les quatre carpetes pràctiques. Se n'ha inspeccionat el contingut per evitar confondre còpies amb materials addicionals.

No hi ha el llibre complet entre aquests fitxers. Les mencions al llibre dins dels apunts no s'han presentat com si s'hagués consultat un document absent.

## Documentació externa

S'han contrastat els conceptes de pipeline, cerca i mètriques amb la documentació oficial de scikit-learn enllaçada al notebook. La biblioteca real de l'execució és la que figura a `resultats/resum.json`; no s'ha suposat que coincideixi amb l'última versió de la web.
