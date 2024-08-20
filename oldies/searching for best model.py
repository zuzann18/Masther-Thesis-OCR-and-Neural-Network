from pathlib import Path
import keras
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---- Paths Configuration ----
images_path = Path(r'C:\Users\zuzan\OneDrive\Pulpit\Dokumenty\GitHub\Praca-Magisterska\dataset\28x28 images.npy')
labels_path = Path(r'C:\Users\zuzan\OneDrive\Pulpit\Dokumenty\GitHub\Praca-Magisterska\dataset\28x28 labels.npy')



# ---- Data Loading and Preprocessing ----
images = np.load(images_path)
labels = np.load(labels_path)
images_normalized = images / 255.0
X_train, X_test, y_train, y_test = train_test_split(images_normalized, labels, test_size=0.2, random_state=42)
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

# ---- Model Definition ----
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # First Convolutional Layer
    model.add(Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))

    # Second Convolutional Layer
    model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=32),
                     kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_1_units', min_value=32, max_value=128, step=32),
                    activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---- Keras Tuner Configuration ----
tuner_search = kt.Hyperband(build_model,
                            objective='val_accuracy',
                            max_epochs=10,
                            directory='hyperband',
                            project_name='keras_tuner_trial',
                            overwrite=True)


# ---- Tuner Search ----
tuner_search.search(X_train, y_train, epochs=3, validation_split=0.1, 
                    batch_size=32,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])


# ---- Retrieve the Best Hyperparameters ----
best_hps = tuner_search.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. Here are the best hyperparameters:
- First Conv Layer Filters: {best_hps.get('conv_1_filter')}
- First Conv Layer Kernel Size: {best_hps.get('conv_1_kernel')}
- Second Conv Layer Filters: {best_hps.get('conv_2_filter')}
- Second Conv Layer Kernel Size: {best_hps.get('conv_2_kernel')}
- First Dense Layer Units: {best_hps.get('dense_1_units')}
- First Dropout Rate: {best_hps.get('dropout_1')}
- Second Dropout Rate: {best_hps.get('dropout_2')}
- Third Dropout Rate: {best_hps.get('dropout_3')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# ---- Build the Model with the Optimal Hyperparameters and Train It on the Data ----
model = tuner_search.hypermodel.build(best_hps)

# ---- Data Augmentation Configuration ----
best_data_augmentation = ImageDataGenerator(
    rotation_range=1,  # Example settings
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ---- Trenowanie najlepszego modelu z najlepszymi ustawieniami data augmentation ----
augmented_data = best_data_augmentation.flow(X_train, y_train, batch_size=32)
model.fit(augmented_data, epochs=20, validation_data=(X_test, y_test))

# ---- Evaluate the Best Model ----
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
