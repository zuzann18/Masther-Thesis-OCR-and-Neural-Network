# Master's Thesis: OCR and Neural Networks

**Title:** Analysis of Letter Recognition Using Artificial Neural Networks

**Objective:** 
To recognize letters using Artificial Neural Networks (ANN):
1. Letters created on 28x28 pixel matrices - used for the test set.
2. Letters from the dataset [HuggingFace Letter Recognition](https://huggingface.co/datasets/pittawat/letter_recognition) - used for the training set.

## Table of Contents

1. [Introduction](#introduction)
2. [Business Justification](#business-justification)
3. [Research Hypothesis](#research-hypothesis)
4. [Technologies Used](#technologies-used)
5. [Algorithms/Models](#algorithmsmodels)
6. [Data Download](#data-download)
7. [Running Experiments](#running-experiments)
8. [Monitoring Results](#monitoring-results)
9. [Experiment Results](#experiment-results)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Introduction

My master's thesis focuses on pattern recognition of characters using artificial neural networks (ANN). The goal is to develop and train ANN models capable of identifying various characters.

## Business Justification

Automating text recognition has broad applications across different industries. For example, in banking, it can automate check data entry; in medicine, digitize handwritten prescriptions; and in logistics, streamline address reading on packages.

## Research Hypothesis

The hypothesis to be verified is: "Artificial neural networks can effectively recognize character patterns and handwriting of different matrix sizes and writing styles, providing results comparable or superior to traditional OCR methods."

## Technologies Used

- **Programming Language:** Python
- **Image Processing Libraries:** OpenCV
- **Deep Learning Frameworks:** TensorFlow

## Algorithms/Models

The main focus is on designing, implementing, and training neural networks using various architectures, such as Convolutional Neural Networks (CNN), which are particularly useful for visual pattern recognition.

## Data Download

To download the data, run:

```bash
python download_data.py
```

Running Experiments
You can run experiments using the run_experiments.py script. Example:

python run_experiments.py --experiment_id 0 --epochs 10
After running the script, TensorBoard will automatically start and save logs in the directory results/tensorboard_logs_{experiment_id}_{timestamp}.

Monitoring Results
To monitor experiment results in real-time, open a terminal and run:

tensorboard --logdir=results
Then open your browser and go to http://localhost:6006/.

Experiment Results
DNN Experiments
Experiment 0: python run_experiments.py --experiment_id 0 --epochs 50
Conclusions: Test Acc: +-25%, learning but overfitting, EarlyStopping at epoch 34
Experiment 1: python run_experiments.py --experiment_id 1 --epochs 50
Conclusions: Test Acc: +-19%, overfitting at epoch 2/3, systematic loss difference between test and training sets, EarlyStopping at epoch 8
CNN Experiments
Experiment 2: python run_experiments.py --experiment_id 2 --epochs 50
Test Acc: +-37%, Train Acc: +-74.5%, overfitting, systematic loss difference between test and training sets, EarlyStopping at epoch 8
Experiment 3: python run_experiments.py --experiment_id 3 --epochs 50
Test Acc: 39%, Train Acc: 80.8%, overfitting, systematic loss difference between test and training sets
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License
This project is licensed under the MIT License.

Acknowledgements
Special thanks to the creators of the datasets and the open-source libraries used in this project.

# Praca Magisterska

Temat pracy : 
"Analiza ropoznawania liter za pomocą sztucznych sieci neuronowych" 

Celem pracy będzie rozpoznanie za pomocą Sztucznej sieci Neuronowej: 
1)	liter utworzonych na matrycach 28 pikseli x 28 pikseli -  są wykorzystawane do zbioru testowego 
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

 Todo
-słów z bazy danych IAM Handwriting Database https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- train models
  - config_experiment
- visualization/reporting
script_train -> (test_results) CSV -> Visulaization.ipynb -> JPG -> Report.md keras.callback.CSVLogger CSV:
```bash
python -m run_experiment --id 2
```



https://keras.io/api/applications/


