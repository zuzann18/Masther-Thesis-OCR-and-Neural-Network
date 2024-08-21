# Praca Magisterska

Temat pracy : 
"Analiza ropoznawania liter za pomocą sztucznych sieci neuronowych" 

Celem pracy będzie rozpoznanie za pomocą Sztucznej sieci Neuronowej: 
1)	liter utworzonych na matrycach 28 pikseli x 28 pikseli -  są wykorzystawane do zbirou testowego 
2)	liter z bazy danych https://huggingface.co/datasets/pittawat/letter_recognition - są wykorzystywane do zbioru treningowego 

1. **Opis problemu**:
Moja praca magisterska koncentruje się na rozpoznawaniu wzorców znaków przy użyciu sztucznych sieci neuronowych (SSN). Zajmuję się opracowaniem i treningiem modeli SSN zdolnych do identyfikacji liter utworzonych na matrycach o różnych rozmiarach, a także rozpoznawaniem pisma odręcznego. Problemem biznesowym, który ten model ma rozwiązać, jest potrzeba efektywnego i automatycznego przetwarzania informacji wizualnych na dane cyfrowe, które mogą być łatwo przeszukiwane i analizowane przez systemy komputerowe.

2. **Uzasadnienie biznesowe**:
Automatyzacja procesu rozpoznawania tekstu ma szerokie zastosowanie w różnych branżach. Na przykład, w bankowości można automatyzować procesy wprowadzania danych czeków, w medycynie - cyfryzować ręcznie zapisane recepty, a w logistyce - usprawniać odczytywanie adresów na paczkach. 

3. **Hipoteza badawcza**:
Hipoteza, którą zamierzam zweryfikować, brzmi: "Sztuczne sieci neuronowe są w stanie efektywnie rozpoznać wzorce znaków oraz pismo odręczne w różnych rozmiarach matrycy i różnym stylu pisma, dostarczając wyniki porównywalne lub lepsze niż tradycyjne metody OCR".

4. **Wykorzystane technologie**:
W pracy wykorzystuję język programowania Python, biblioteki do przetwarzania obrazów takie jak OpenCV oraz frameworki do głębokiego uczenia się jak TensorFlow, które są idealne do projektowania, trenowania i weryfikacji modeli SSN.

5. **Algorytmy/ modele**:
Głównym elementem mojej pracy jest projektowanie, implementacja i trening sieci neuronowych z użyciem różnorodnych architektur, takich jak konwolucyjne sieci neuronowe (CNN), które są szczególnie przydatne w rozpoznawaniu wzorców wizualnych.

**Struktura pracy**:
- Wstęp: przedstawienie motywacji oraz znaczenia biznesowego projektu, definicja celów i hipotez, metodologia.
- Rozdział I: Przegląd teoretycznych podstaw SSN oraz technik rozpoznawania wzorców.
- Rozdział II: Opis realizacji modeli SSN, w tym wyboru architektury i procesu treningu.
- Rozdział III: Prezentacja i analiza wyników rozpoznawania znaków na różnych matrycach oraz pisma odręcznego.
- Rozdział IV: Dyskusja nad potencjalnymi korzyściami biznesowymi wynikającymi z zastosowania SSN w rozpoznawaniu pisma ręcznego i wnioski końcowe.
Praca w pythonie. 

# Todo

- train models
  - config_experiment
  - saving the results & models
- visualization/reporting
script_train -> (test_results) CSV -> Visulaization.ipynb -> JPG -> Report.md keras.callback.CSVLogger CSV:
```bash
python -m run_experiment --id 2
```

# Download data

```bash
python download_data.py
```

https://keras.io/api/applications/


