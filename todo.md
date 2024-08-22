# Run Experiments 

- run experiment
- `python visualise_results.py`


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
  - Test Acc: +- 37 % Train Acc +-  74,5 %
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
  - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie większej ilości danych, 
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
  - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie większej ilości danych, 
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
    - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie większej ilości danych, 
    - przez co jest trudniejsze nauczenie się tak powiększoneg zbioru
    - augumentacja nie jest aktywna w trakcie walidacji. Niezaugumentowane zbiór jest łatwiejszy do nauczenia,
- `python run_experiments.py --experiment_id 8 --epochs 50`
- Epoch 11/50
-  accuracy: 0.0367 - loss: 3.2597 - val_accuracy: 0.0385 - val_loss: 3.2581
- Najprawdopodobniej kernel 5x5 jest za duży, sieć się nie uczy 
- `python run_experiments.py --experiment_id 9 --epochs 350`


[//]: # (- [ ] Experiment 6: `python run_experiments.py --experiment_id 6 --epochs 50`)

[//]: # (Szablon do raportu:)
- [x] Experiment X: `python run_experiments.py --experiment_id X --epochs 50`
  - Test Acc: 
  - Train Acc: 
  - overfitting: 
  - underfitting: 
  - dropout:
  - Loss: np. Cykliczne zmiany na zbiorze treningowym osycalacja, 
  - W związku z zastosowaniem augumentacji danych, wynik na zbiorze testowym jest wyższy, niż na treningowym. 
  - Wynika to z tego, że z powodu augumentacji danych, jest aktywna tylko w zbiorze trenigowym, powoduje dodawnie większej ilości danych, 
  - przez co jest trudniejsze nauczenie się tak powiększoneg zbioru
  - augumentacja nie jest aktywna w trakcie walidacji. Niezaugumentowane zbiór jest łatwiejszy do nauczenia, 

# (Improving experiments:) Sprawdźić:

- [x] na cnn2 i cnn3 dropout:0.1-0.4
- [x] batchsize 128
- [x] learning rate: 1e-1 - 1e-5
- [x] zmnieszyć zakres augumentacji - do 10 % albo 15 % 
- [x] sgd, AdaDelta 
- [] sprawdzić AdaDelata na różnych learning_rate e-3 powinno być ok
- [] Batchsize dodać 256, 512
- [] learning rate scheduler callback
- 
- [] **transfer learning** 
- porownac wielkosci danych - VGG jest na obrazkach na wiekszych rozdzelczosciach 258; zreserczeować na 28x28, 

# Kolejność :  na początku bez augumentacji,maskowanie w celu generalizacji modelu, potem z augumentacją,

# TO DO-

- [] raport md zrobic zeby dziala
- [x] co z tymi layers, jak mogę dodać budowę modelu z model do raportu
  -[] # ?Kolejność :  na początku bez augumentacji,maskowanie w celu generalizacji modelu, potem z augumentacją,

- [] sprawdzic jaka wartość po 60 epokach dla learning_rate_scheduler
-
- [] zacząć opisywać eksperymenty w pracy magisterskiej z droputem
- raport md

# check

- [x] experimental_config - jak z obrotami:
- [] sprawdzić czy tmp_images.npy i tmp_labels.npy, które są luzem są potrzebne
- []VGG16 model requires an input size of at least 32x32, but the provided input shape is 28x28. - czy da się w tym
  przykładzie zorbić transfwer learning ?
- [] czy 20 w augumentation to nie jest za duzo ?  Slight rotations such as between 1 and 20 or − 1 to − 20 could be useful on digit recognition tasks such as MNIST, but as the rotation degree increases, the label of the data is no longer preserved post-transformation.
 gdyby dać 4, albo coś takiego ? 
- [] znaleźć jakieś rozszerzenie dla czytania wygodniejszeco csv
- [] visualizing one experiment per time - nie działa
- [] Identify the Problem: The error indicates that the CSV file results/runs_history.csv has inconsistent row lengths,
  with some rows having more fields than expected.

