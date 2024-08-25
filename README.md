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
Wstęp

1.1. Cel i Zakres Pracy
Opis celów i zakresu pracy dyplomowej.
1.2. Przegląd Technologii OCR i Sieci Neuronowych
Przegląd technologii OCR oraz sieci neuronowych.
1.3. Przegląd Literatury
Przegląd literatury podsumowujący kluczowe badania i osiągnięcia.
Rozdział I: Rozpoznawanie Wzorców i OCR

2.1. Wprowadzenie do Rozpoznawania Wzorców
Wprowadzenie do pojęcia rozpoznawania wzorców.
2.2. Historia i Rozwój OCR
Historia rozwoju i ewolucji technologii OCR.
2.3. Zastosowania OCR
Praktyczne zastosowania OCR w różnych dziedzinach.
2.4. Techniki Przetwarzania Obrazów w OCR
Techniki przetwarzania obrazów stosowane w OCR.
2.4.1. Ekstrakcja Cech
Metody ekstrakcji cech w OCR.
2.4.2. Klasyfikacja
Techniki klasyfikacji znaków.
2.4.3. Narzędzia i Biblioteki OCR
Przegląd narzędzi i bibliotek wykorzystywanych w rozwoju OCR.
2.5. Intelligent Character Recognition (ICR)
Zaawansowane techniki OCR i inteligentne rozpoznawanie znaków.
2.5.1. Zastosowanie ICR
Zastosowania i przypadki użycia ICR.
2.5.2. Trendy w Rozwoju Technologii ICR
Aktualne trendy i przyszłe kierunki rozwoju technologii ICR.
Rozdział II: Sieci Neuronowe w OCR

3.1. Wprowadzenie do Sieci Neuronowych
Wprowadzenie do sieci neuronowych i ich roli w OCR.
3.2. Konwolucyjne Sieci Neuronowe (CNN)
Szczegółowe omówienie konwolucyjnych sieci neuronowych w OCR.
3.2.1. Architektura CNN
Architektura CNN i zasady projektowania.
3.2.2. Zastosowanie Małych Filtów Konwolucyjnych
Wykorzystanie małych filtrów konwolucyjnych w CNN do OCR.
3.3. Skuteczność Technik w OCR
Ocena technik poprawiających wydajność OCR.
3.3.1. Augmentacja Danych
Techniki augmentacji danych dla poprawy dokładności OCR.
3.3.2. Metody Ansamblowe
Metody ansamblowe w modelach OCR.
3.3.3. Porównanie z Innymi Modelami
Porównanie CNN z innymi modelami OCR.
3.4. Przykłady Implementacji Sieci Neuronowych w OCR
Studium przypadków i przykłady implementacji sieci neuronowych w systemach OCR.
Rozdział III: Projektowanie i Budowa Sieci Neuronowych

4.1. Szerokie i Głębokie Sieci Neuronowe
Projektowanie i rozwój szerokich i głębokich sieci neuronowych.
4.2. Unikanie Overfittingu
Strategie zapobiegania overfittingowi w sieciach neuronowych.
4.3. Funkcje Aktywacji
Przegląd funkcji aktywacji używanych w sieciach neuronowych.
4.4. Warstwy Strided Convolution
Wprowadzenie do warstw strided convolution i ich zalet.
4.4.1. Zachowanie Detali i Efektywność Pamięciowa
Równowaga między zachowaniem detali a efektywnością pamięciową.
4.4.2. Szybkość i Dokładność
Kompromis między szybkością a dokładnością w projektowaniu sieci.
4.4.3. Optymalna Konfiguracja Warstw
Wytyczne dotyczące optymalnej konfiguracji warstw w sieciach neuronowych.
Rozdział IV: Proces Trenowania Sztucznej Sieci Neuronowej

5.1. Przygotowanie Środowiska
Przygotowanie środowiska do trenowania sieci neuronowej.
5.1.1. Konfiguracja Środowiska Wirtualnego
Kroki konfiguracji środowiska wirtualnego.
5.1.2. Narzędzia i Biblioteki
Podstawowe narzędzia i biblioteki do rozwoju sieci neuronowych.
5.2. Budowa Sieci Neuronowej przy Użyciu TensorFlow
Budowa sieci neuronowej z użyciem TensorFlow.
5.2.1. Zbiór Danych
Wybór i przygotowanie zbioru danych do trenowania.
5.2.2. Konstrukcja Sieci
Budowa sieci i projektowanie modelu.
5.3. Funkcje Aktywacji
Głębsza analiza funkcji aktywacji stosowanych w modelach OCR.
5.3.1. ReLU
Rola funkcji ReLU w głębokim uczeniu.
5.3.2. Softmax
Zrozumienie funkcji Softmax w zadaniach klasyfikacyjnych.
5.4. Warstwy i Ich Funkcje
Analiza różnych warstw sieci i ich roli w OCR.
Zakończenie

6.1. Podsumowanie Wyników
Podsumowanie wyników badań zawartych w pracy.
6.2. Wnioski i Kierunki Przyszłych Badań
Wnioski końcowe i sugerowane kierunki przyszłych badań.
Literatura

Kompleksowa lista źródeł wykorzystanych w pracy.

-

# Download data

```bash
python download_data.py
```

# Uruchamianie Eksperymentów
Eksperymenty można uruchomić za pomocą skryptu run_experiments.py. Przykładowe uruchomienie eksperymentu:

python run_experiments.py --experiment_id 0 --epochs 10
Po uruchomieniu skryptu, TensorBoard automatycznie uruchomi się i zapisze logi z treningu modelu w katalogu
results/tensorboard_logs_{experiment_id}_{timestamp}.

Monitorowanie Wyników z TensorBoard
Uruchomienie TensorBoard
Aby monitorować wyniki eksperymentów w czasie rzeczywistym, otwórz terminal i uruchom TensorBoard:

tensorboard --logdir=results
Po uruchomieniu TensorBoard, otwórz przeglądarkę internetową i przejdź do adresu http://localhost:6006/. Tam będziesz
mógł zobaczyć wykresy i wizualizacje wyników, takie jak:

# Run Experiments
# DNN

- [x] Experiment 0: `python run_experiments.py --experiment_id 0 --epochs 50`
  - Conclusions: Test Acc: +-25%, uczy się ale jest overfitting i na 34 epoce EarlyStopping
- [x] Experiment 1: `python run_experiments.py --experiment_id 1 --epochs 50`
  - Conlusions: Test Acc: +- 19%,
  - Overtfitting na 2/3 epoce
  - Wartość funkcji straty (loss) zaczyna się systemtycznie oddalac mięzyzbiorem testowym,a treningowym
  - na 8 epoce EarlyStopping

# CNN

- [X] Experiment 2: `python run_experiments.py --experiment_id 2 --epochs 50`
  - Test Acc: +- 37 % Train Acc +- 74,5 %
    -overfitting : True
  - Wartość funkcji straty (loss) zaczyna się systemtycznie oddalac mięzyzbiorem testowym,a treningowym
  - na 8 epoce EarlyStopping
- [x] Experiment 3: `python run_experiments.py --experiment_id 3 --epochs 50`
  - Test Acc: 39 %
  - Train Acc: 80,8 %
  - overfitting: True
  - Wartość funkcji straty (loss) zaczyna się systemtycznie oddalac mięzyzbiorem testowym,a treningowym

- [x] Experiment 4: `python run_experiments.py --experiment_id 4 --epochs 50`
  - Test Acc: 20,9%
  - Train Acc: 25,4%
  - overfitting: False
  - underfitting: True
  - Loss: Cykliczne zmiany na zbiorze treningowym osycalacja,
  - W związku z zastosowaniem augumentacji danych, wynik na zbiorze testowym jest wyższy, niż na treningowym.
  - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie
    większej ilości danych,
  - przez co jest trudniejsze nauczenie się tak powiększoneg zbioru
  - augumentacja nie jest aktywna w trakcie walidacji. Niezaugumentowane zbiór jest łatwiejszy do nauczenia,
  -
- [x] Experiment 5: `python run_experiments.py --experiment_id 5 --epochs 50`
  - Test Acc: 25,88%
  - Train Acc: 23,5%
  - overfitting:
  - underfitting:
  - Loss: np. Cykliczne zmiany na zbiorze treningowym osycalacja,
  - W związku z zastosowaniem augumentacji danych, wynik na zbiorze testowym jest wyższy, niż na treningowym.
  - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie
    większej ilości danych,
  - przez co jest trudniejsze nauczenie się tak powiększoneg zbioru
  - augumentacja nie jest aktywna w trakcie walidacji. Niezaugumentowane zbiór jest łatwiejszy do nauczenia,
- [x] Experiment 6:`python run_experiments.py --experiment_id 6 --epochs 50`
  - Train Acc:40,5%
  - Test Acc: 23,5%
  - Loss: Cykliczne zmiany na zbiorze treningowym osycalacja,
  - 40 epok EarlyStopping
- `python run_experiments.py --experiment_id 7 --epochs 50`
  - Test Acc:
    - Train Acc:
    - overfitting:
    - underfitting:
    - dropout:
    - Loss: np. Cykliczne zmiany na zbiorze treningowym osycalacja,
    - W związku z zastosowaniem augumentacji danych, wynik na zbiorze testowym jest wyższy, niż na treningowym.
    - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie
      większej ilości danych,
    - przez co jest trudniejsze nauczenie się tak powiększoneg zbioru
    - augumentacja nie jest aktywna w trakcie walidacji. Niezaugumentowane zbiór jest łatwiejszy do nauczenia,
- `python run_experiments.py --experiment_id 8 --epochs 50`
- Epoch 11/50
- accuracy: 0.0367 - loss: 3.2597 - val_accuracy: 0.0385 - val_loss: 3.2581
- Najprawdopodobniej kernel 5x5 jest za duży, sieć się nie uczy
- `python run_experiments.py --experiment_id 9 --epochs 350`

Best result

![img.png](img.png)

# Todo
- train models
  - config_experiment
- visualization/reporting
script_train -> (test_results) CSV -> Visulaization.ipynb -> JPG -> Report.md keras.callback.CSVLogger CSV:
```bash
python -m run_experiment --id 2
```



https://keras.io/api/applications/


