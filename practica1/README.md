# Pràctica 1: classificació de passatgers del Titanic

Memòria i codi en català, amb experiments executats sobre el CSV proporcionat a l'assignatura.

## Què cal obrir

- **[Practica_1_Titanic.ipynb](Practica_1_Titanic.ipynb)**: entrega principal, amb explicacions, codi, taules i gràfics. Es pot obrir a Jupyter o Google Colab.
- **[Practica_1_Titanic.html](Practica_1_Titanic.html)**: la mateixa memòria per llegir al navegador sense instal·lar Python. Cal descarregar el fitxer; GitHub no l'executa en la vista del repositori.
- **[Presentacio_4_minuts.html](Presentacio_4_minuts.html)**: presentació breu. Fletxes dreta/esquerra per canviar de diapositiva. També permet imprimir.
- **[GUIO_PRESENTACIO.md](GUIO_PRESENTACIO.md)**: guió oral, aproximadament 3 minuts i 40 segons.
- **[RESULTATS_COMENTATS.md](RESULTATS_COMENTATS.md)**: lectura concreta dels resultats i decisions.
- **[practica1.py](practica1.py)**: el mateix codi amb cel·les `# %%`, sense classes pròpies ni expressions comprimides.
- **[resultats/](resultats/)**: 26 figures PNG, taules CSV, particions, prediccions del test i resum JSON.

La memòria dins del notebook és una de les modalitats acceptades per l'enunciat. No cal generar un PDF per utilitzar aquesta entrega.

## Com reproduir-ho

Des d'aquesta carpeta, amb Python 3.13:

```bash
python -m pip install -r requirements.txt
python generar_notebook.py
python preparar_entrega.py
```

La primera ordre instal·la dependències. La segona executa totes les cel·les i genera notebook i HTML. La tercera genera la presentació i el guió a partir dels resultats. L'execució completa pot trigar uns minuts segons l'ordinador. Fa servir dos processos per a la validació creuada.

També es pot executar `python practica1.py`, però per conservar gràfics i text dins del notebook convé utilitzar `generar_notebook.py`. A Colab, pugeu el notebook i `titanic_train.csv` al mateix entorn i executeu totes les cel·les. No cal descarregar dades d'Internet.

La llavor principal és 42. El test final queda separat abans de l'EDA predictiva i no es fa servir per escollir models, hiperparàmetres, variables o llindars. Les columnes `mean_test_*` de GridSearchCV són de **validació creuada**, no del test final.

## Contingut de la investigació

1. Mida, significat i tipus de les dades, target, valors nuls, associacions i correlacions.
2. Eliminació de files contra imputació, reconstrucció d'edats, quatre escales, dos encodings, noves variables, PCA, selecció de variables i grau 2.
3. Dummy, KNN, logística, SVM, arbre i Random Forest; bucle K-fold manual; 3/5/10 folds i quatre llavors; cerca exhaustiva i comparació amb random search.
4. Accuracy, F1, balanced accuracy, ROC-AUC, average precision, matrius de confusió, llindars i reconsideració de tota la graella segons F1.
5. Corba d'aprenentatge, sensibilitat a bitllets compartits, test final, intervals bootstrap, errors per subgrups i importàncies per permutació.

Els resultats de CV que s'han utilitzat per seleccionar són exploratoris. El test és l'avaluació final del procediment fixat. Ni el test ni el bootstrap demostren que el model serveixi per a altres poblacions.

## Materials, autoria i lliurament

Els materials originals es conserven a la carpeta local de treball. Aquest repositori conté la solució i les dades necessàries, no còpies dels apunts del professorat. Consulteu [MATERIALS_REVISATS.md](MATERIALS_REVISATS.md).

Cal completar els noms i NIU dels membres abans de lliurar. No s'han inventat autors ni revisions personals. Aquesta versió s'ha elaborat amb assistència d'IA, declarada també al notebook. Els autors han de revisar-la i poder explicar-la durant la defensa.
