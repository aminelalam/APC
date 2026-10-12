## 6. Discussió dels resultats: què defensem i què no?

Aquesta lectura complementa les explicacions de cada experiment. Tots els números provenen de les taules de l'execució. Les decisions de selecció es prenen amb desenvolupament. Les observacions del test són diagnòstics posteriors i no han provocat canvis del model.

### 6.1. Les dades suggereixen interaccions, no una única regla

El creuament sexe/classe és més informatiu que mirar només una mitjana. A les figures veiem que les dones de primera i segona classe tenen taxes altes, mentre que les de tercera queden força per sota. Entre els homes també canvia la taxa segons la classe. Això dona sentit a provar models amb interaccions, però no permet reconstruir les circumstàncies de cada persona.

La matriu de correlació mostra una relació forta entre classe i cabina registrada. Per tant, dues columnes predictives poden estar recollint informació semblant. En el cas d'Age i EdatDesconeguda apareix una casella buida: quan Age té valor, l'indicador sempre és zero. No hi ha variació per calcular aquesta correlació per parelles. **El blanc és una correlació indefinida, no una correlació igual a zero.**

MidaFamilia tampoc compta tota la xarxa social: només suma els familiars recollits a SibSp i Parch, més el passatger. Ticket pot compartir-se entre persones, però no identifica totes les famílies. Fare és la tarifa registrada i no n'hem inventat un preu individual dividint-lo pel nombre de files del mateix bitllet, que pot estar incomplet.

### 6.2. Perdre files no és una solució neutra

Entrenar només amb files completes de les variables base deixa de mitjana 458.4 files per fold, davant 569.6 amb imputació. En la mateixa validació completa, les accuracies són 0.787 i 0.802, respectivament. El filtre de casos complets només cobreix el 80.5% de les files de validació. El seu resultat parcial respondria a una població diferent.

Preferim conservar passatgers i imputar. És una decisió que combina resultat, cobertura i mida de mostra. Si exigíssim també Cabin, perdríem moltes més files. No hem convertit una cabina desconeguda en una falsa cabina coneguda: n'hem conservat l'indicador.

### 6.3. Reconstruir millor una edat no garanteix classificar millor

En deu simulacions, el MAE mitjà és 10.70 anys amb KNNImputer, 11.45 amb mediana i 11.50 amb mitjana. KNN reconstrueix millor les edats ocultades en aquesta prova. Però la logística amb imputació KNN dona accuracy 0.796, davant 0.802 amb mediana.

No és una contradicció. El MAE penalitza la distància en anys, mentre que la classificació depèn de com aquests valors afecten una frontera conjunta amb altres atributs. A més, hem amagat edats conegudes de manera aleatòria; els nuls reals podrien seguir un altre patró. Per això no podem afirmar que hem descobert les edats reals ni que KNN sigui sempre millor.

### 6.4. La prova de geometria sí que té un efecte gran

Amb k=5 fixat, KNN sense escalar obté 0.718 d'accuracy; amb StandardScaler, 0.798. Són 8.01 punts percentuals de diferència. Les unitats de Fare estaven condicionant la noció de proximitat.

MinMax dona 0.798 i RobustScaler 0.794. Aquí la diferència important és escalar respecte a no escalar; no hi ha evidència suficient per proclamar un guanyador universal entre els escaladors. RobustScaler redueix la influència dels extrems en l'escala, però no elimina passatgers.

### 6.5. Més transformacions no sempre ajuden

La logística base amb mediana dona 0.802. Afegir només el títol puja a 0.819; la representació ampliada arriba a 0.824. El títol pot resumir edat, sexe i convencions socials. No l'interpretem com una causa de supervivència.

Afegir només família dona 0.796, i substituir tarifa per log de tarifa dona 0.791. Són exemples d'idees raonables que no milloren aquesta logística per si soles. El resultat de combinar-les pot ser diferent, i per això comparem representacions completes dins de la cerca.

PCA al 95% dona 0.803, molt a prop de la base. No necessitem comprimir una representació tan petita si no hi ha un benefici clar i perdrem interpretació. Seleccionar el 50% dona 0.777: una regla univariant pot perdre informació complementària. Grau 2 puja l'accuracy a 0.822, però l'F1 és 0.746 i la dispersió entre folds augmenta. No l'escollim només perquè train millori.

L'encoding ordinal de totes les categories dona 0.798, davant 0.802 amb one-hot. La diferència és petita; el motiu principal per mantenir one-hot és que Embarked no té un ordre numèric justificat.

### 6.6. Sobreajust i cerca: no confondre un màxim amb una victòria clara

L'arbre per defecte arriba a accuracy de train 0.982, però CV baixa a 0.771. Això fa visible el sobreajust. Limitar la profunditat i el nombre mínim de casos per fulla és una manera comprensible de controlar-lo. El Random Forest per defecte també té una separació gran entre train i CV: combinar arbres no elimina tots els problemes.

Amb la cerca, Random Forest i una SVM RBF empaten en accuracy CV arrodonida a sis decimals: 0.831498. Per F1, Random Forest dona 0.771066, la millor SVM 0.770871 i logística 0.769302. La diferència entre RF i SVM és de només 0.000195. **Escollim el màxim segons la regla fixada, però no afirmem una superioritat estadística clara del Random Forest.** Una logística més senzilla continuaria sent una alternativa defensable si la interpretació fos prioritària. No l'hem avaluat després al test per triar la que hi quedés millor.

Canviar de mètrica sí que canvia la configuració preferida de SVM: la millor per F1 utilitza pesos de classe equilibrats, mentre que la millor per accuracy no. Aquesta és la resposta pràctica a «la comparació amb accuracy aguanta?»: el primer model pot continuar destacant, però les configuracions i el compromís d'errors no són els mateixos.

La logística fixada dona mitjanes d'accuracy entre 0.791 i 0.808 quan variem folds i llavor. Aquesta prova no és un test estadístic de RF contra SVM, però recorda que les particions aporten variabilitat i que una quarta xifra decimal no és una garantia.

Grid SVM prova 48 candidats i dona 0.831, amb 52.4 segons. Random prova 12 candidats i dona 0.826, amb 6.7 segons. Aquesta comparació il·lustra un pressupost de candidats menor, no una prova amb temps igual ni una conclusió universal sobre els mètodes.

### 6.7. El test és menys bo que la selecció i això és part del resultat

L'F1 baixa de 0.771 en CV de selecció a 0.742 en el test. No hem intentat arreglar aquesta diferència buscant una altra llavor o altres paràmetres després de veure-la. Pot intervenir l'optimisme de selecció i la variabilitat d'una mostra petita.

La matriu és TN=96, FP=14, FN=20 i TP=49. El model detecta 49 dels 69 supervivents. F1 resumeix aquest compromís, però no explica els 96 no supervivents ben classificats, que sí que contribueixen a accuracy. Per això presentem les dues mètriques.

El bootstrap situa accuracy entre 0.754 i 0.866, i F1 entre 0.643 i 0.818. Són intervals orientatius condicionats al model fix i a la mostra, amb la limitació d'independència entre passatgers. No demostren un rendiment mínim garantit.

### 6.8. La mitjana global amaga un problema amb els homes supervivents

El recall entre dones és 41/45 = 91.1%; entre homes, 8/24 = 33.3%. Dels 20 falsos negatius globals, 16 són homes. L'accuracy d'homes encara és 78.8%, perquè hi ha molts no supervivents ben classificats. Aquest és un exemple concret de per què accuracy sola no ens basta.

A tercera classe, l'accuracy és 84.0%, però només detectem 12 de 24 supervivents. Una accuracy alta dins d'un subgrup tampoc garanteix bon recall. Amb infants només tenim 17 casos i 10 positius al test: les conclusions d'aquest grup són especialment inestables.

Amb validació per bitllet, F1 passa de 0.771 a 0.747. No podem atribuir tota la diferència als vincles perquè també canvien els folds, però la prova assenyala un límit del protocol per passatgers. Si volguéssim predir famílies noves, caldria seleccionar i avaluar tot el procés amb grups.

### 6.9. La cabina no aporta el mateix en totes les proves

En la permutació del model final, la caiguda mitjana de F1 en barrejar Cabin és -0.022: és negativa, és a dir, aquesta alteració millora lleugerament l'F1 en aquesta mostra. No ho amaguem. Pot intervenir soroll, redundància amb classe i la mida petita del test. No justifica eliminar Cabin i tornar a provar el mateix test, perquè estaríem adaptant el model a l'avaluació final.

Pclass i Sex tenen caigudes de 0.145 i 0.144, respectivament. Name influeix a través del títol, no perquè s'aprenguin noms complets. Les importàncies descriuen dependència predictiva amb aquesta prova i aquest model; les correlacions i les combinacions poc realistes impedeixen llegir-les com efectes causals.

### 6.10. Decisió que podem defensar

Hem seleccionat un model mitjançant una regla explícita i hem conservat les proves que no milloraven. El resultat és útil per aprendre a classificar aquest registre, amb errors concrets i límits clars. La millora més rellevant per a un projecte posterior seria definir bé la població futura i la unitat de validació, abans d'ampliar la graella o provar un model més complex.
