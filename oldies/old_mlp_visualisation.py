import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
# import json
#
# with open('experiments_config.json', 'r') as config_file:
#     config = json.load(config_file)

images_path = Path(r'C:\Users\zuzan\OneDrive\Pulpit\Dokumenty\GitHub\Praca-Magisterska\dataset\images.npy')
labels_path = Path(r'C:\Users\zuzan\OneDrive\Pulpit\Dokumenty\GitHub\Praca-Magisterska\dataset\labels.npy')

# Wczytywanie danych
images = np.load(images_path)
labels = np.load(labels_path)



# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.20, random_state=42)

# Check the shape of the loaded data
images.shape, labels.shape

# Definicja modelu perceptronu
def build_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(num_classes, activation='softmax')  # Tylko jedna warstwa wyjściowa
    ])

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Budowanie modelu

model = build_model(X_train.shape[1:], y_train.shape[1])

model.summary()

# Trening modelu
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Ewaluacja modelu
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")

# Predykcje i raporty
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print(classification_report(y_true_classes, y_pred_classes))

# Obliczenie dodatkowych metryk
print(classification_report(y_true_classes, y_pred_classes))


# Znajdź indeksy błędnych przewidywań
misclassified_idx = np.where(y_pred_classes != y_true_classes)[0]

# Wyświetl błędnie zaklasyfikowane obrazy
num_images_to_show = 10  # Możesz dostosować liczbę wyświetlanych obrazów
for i, idx in enumerate(misclassified_idx[:num_images_to_show]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.squeeze(X_test[idx]), cmap='gray')
    plt.title(f"True: {y_true_classes[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.show()

class_labels = [str(i) for i in range(26)]  # Tworzy etykiety jako stringi od '0' do '25'

# Obliczenie macierzy pomyłek
cm = confusion_matrix(y_true_classes, y_pred)

# Tworzenie wykresu macierzy pomyłek
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Przewidywane Klasy')
plt.ylabel('Prawdziwe Klasy')
plt.title('Macierz Pomyłek')
plt.show()



# Trening modelu perceptronu
history = model.fit(
    X_train, y_train_categorical,
    validation_data=(X_test, y_test_categorical),
    epochs=10,
    batch_size=32
)
# Wizualizacja wyników treningu
plt.plot(model.history['accuracy'], label='accuracy')
plt.plot(model.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

