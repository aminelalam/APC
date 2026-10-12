# Guió oral de la pràctica 1

Sis diapositives. Temps orientatiu: 3 minuts i 40 segons, deixant 20 segons de marge. Cal assajar-lo amb la veu pròpia i completar els autors. Les dades són les de l'execució, no números inventats.

## 1. Pregunta i objectiu · 20 segons

En aquesta pràctica intentem predir la supervivència dels passatgers del Titanic. Però no volíem quedar-nos amb el model que dona el número més alt. Hem investigat què representen les dades, com preparar-les i quins errors fa el resultat final.

## 2. Dades i absències · 35 segons

Tenim 891 passatgers i dotze columnes, inclòs el target. Hi ha 177 edats desconegudes i 687 cabines sense registrar. Ens va semblar important no interpretar un buit com un zero. La cabina registrada està relacionada amb la classe. Per això hem mirat taxes i recomptes dins dels grups. Sexe i classe mostren associacions fortes, però les dades no demostren causalitat.

## 3. Preparació i protocol · 40 segons

Hem reservat 179 passatgers per al test i hem investigat amb els 712 restants. Fem cinc folds estratificats i ajustem imputació i escalat només amb les files d'entrenament de cada fold. Això evita filtracions d'informació. Una comprovació clara dels apunts és KNN: passa del 71.8% sense escalar al 79.8% amb StandardScaler. També hem provat idees que no milloren, com eliminar la meitat de variables.

## 4. Models i selecció · 45 segons

Comparem KNN, logística, SVM, arbre i Random Forest, a més d'una baseline que sempre prediu la classe majoritària. La cerca principal té 184 configuracions i 920 ajustos. Primer comparem accuracy i després tornem a ordenar tots els candidats per F1. Escollim F1 perquè volem equilibrar la precisió de les supervivències predites amb els supervivents reals que detectem. Els resultats de la taula són de selecció i poden ser optimistes.

## 5. Resultat final · 45 segons

Amb les decisions tancades, el model Random Forest dona una accuracy del 81.0% i F1 de 0.742 al test. Detecta 49 de 69 supervivents. Hi ha 14 falsos positius i 20 falsos negatius. Hem mirat també ROC i precision-recall, intervals orientatius i errors per sexe, classe i edat. Això explica més que dir només el percentatge d'encert.

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
