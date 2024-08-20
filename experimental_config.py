
EXPERIMENTAL_CONFIG = [
    # test on small and big data
    {
        'experiment_id': 0,
        'model_name': "DNN1",
    },
    {
        'experiment_id': 1,
        'model_name': "DNN2",
    },
    {
        'experiment_id': 2,
        'model_name': "CNN1",
    },
    {
        'experiment_id': 3,
        'model_name': "CNN2",
    },
    {
        "experiment_id": 4,
        "model_name": "CNN3",
    },
    {
        "experiment_id": 5,
        "model_name": "CNN1",
        "augmentation": True,
    },
    {
        "experiment_id": 6,
        "model_name": "CNN1",
        "dropout_rate": 0.1,
    },
    {
        "experiment_id": 7,
        "model_name": "CNN1",
        "dropout_rate": 0.2,
    },
    {
        "experiment_id": 8,
        "model_name": "CNN1",
        "dropout_rate": 0.3,
    },
    {
        "experiment_id": 9,
        "model_name": "CNN1",
        "dropout_rate": 0.4,
    },
    {
        "experiment_id": 10,
        "model_name": "CNN2",
        "dropout_rate": 0.1,
    },
    {
        "experiment_id": 11,
        "model_name": "CNN2",
        "dropout_rate": 0.2,
    },
    {
        "experiment_id": 12,
        "model_name": "CNN2",
        "dropout_rate": 0.3,
    },
    {
        "experiment_id": 13,
        "model_name": "CNN2",
        "dropout_rate": 0.4,
    },
    {
        "experiment_id": 14,
        "model_name": "CNN3",
        "dropout_rate": 0.1,
    },
    {
        "experiment_id": 15,
        "model_name": "CNN3",
        "dropout_rate": 0.2,
    },
    {
        "experiment_id": 16,
        "model_name": "CNN3",
        "dropout_rate": 0.3,
    },
    {
        "experiment_id": 17,
        "model_name": "CNN3",
        "dropout_rate": 0.4,
    },
    # Batch size 128
    {
        "experiment_id": 18,
        "model_name": "CNN1",
        "batch_size": 128,
    },
    {
        "experiment_id": 19,
        "model_name": "CNN2",
        "batch_size": 128,
    },
    {
        "experiment_id": 20,
        "model_name": "CNN3",
        "batch_size": 128,
    },
    # Learning rates 1e-1 to 1e-5 for CNN1
    {
        "experiment_id": 21,
        "model_name": "CNN1",
        "learning_rate": 1e-1,
    },
    {
        "experiment_id": 22,
        "model_name": "CNN1",
        "learning_rate": 1e-2,
    },
    {
        "experiment_id": 23,
        "model_name": "CNN1",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 24,
        "model_name": "CNN1",
        "learning_rate": 1e-4,
    },
    {
        "experiment_id": 25,
        "model_name": "CNN1",
        "learning_rate": 1e-5,
    },
    {
        "experiment_id": 26,
        "model_name": "CNN1",
        "optimizer": "sgd",
    },
    {
        "experiment_id": 27,
        "model_name": "CNN1",
        "optimizer": "adadelta",
    },
    # Optimizers SGD and AdaDelta for CNN2
    {
        "experiment_id": 28,
        "model_name": "CNN2",
        "optimizer": "sgd",
    },
    {
        "experiment_id": 29,
        "model_name": "CNN2",
        "optimizer": "adadelta",
    },
    # Optimizers SGD and AdaDelta for CNN3
    {
        "experiment_id": 30,
        "model_name": "CNN3",
        "optimizer": "sgd",
    },
    {
        "experiment_id": 31,
        "model_name": "CNN3",
        "optimizer": "adadelta",
    },
# Reduced augmentation range to 10%
    {
        "experiment_id": 32,
        "model_name": "CNN1",
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    {
        "experiment_id": 33,
        "model_name": "CNN2",
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    {
        "experiment_id": 34,
        "model_name": "CNN3",
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    # Reduced augmentation range to 15%
    {
        "experiment_id": 35,
        "model_name": "CNN1",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 36,
        "model_name": "CNN2",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 37,
        "model_name": "CNN3",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 38,
        "model_name": "CNN1",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
        "learning_rate_scheduler": True,
    },
    {
        "experiment_id": 39,
        "model_name": "CNN2",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
        "learning_rate_scheduler": True,
    },
    {
        "experiment_id": 40,
        "model_name": "CNN3",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
        "learning_rate_scheduler": True,
    },
    {
        "experiment_id": 41,
        "model_name": "CNN4",
        "augmentation": False,
        "learning_rate_scheduler": True,
    },
    {
        "experiment_id": 42,
        "model_name": "CNN5",
        "augmentation": False,
        "learning_rate_scheduler": True,
    },
    {
        "experiment_id": 43,
        "model_name": "CNN4",
        "batch_size": 128,
    },
    {
        "experiment_id": 44,
        "model_name": "CNN5",
        "batch_size": 128,
    },
    {
        "experiment_id": 45,
        "model_name": "CNN4",
        "learning_rate": 1e-1,
    },
    {
        "experiment_id": 46,
        "model_name": "CNN4",
        "learning_rate": 1e-2,
    },
    {
        "experiment_id": 47,
        "model_name": "CNN4",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 48,
        "model_name": "CNN4",
        "learning_rate": 1e-4,
    },
    {
        "experiment_id": 49,
        "model_name": "CNN4",
        "learning_rate": 1e-5,
    },
    {
        "experiment_id": 50,
        "model_name": "CNN5",
        "learning_rate": 1e-1,
    },
    {
        "experiment_id": 51,
        "model_name": "CNN5",
        "learning_rate": 1e-2,
    },
    {
        "experiment_id": 52,
        "model_name": "CNN5",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 53,
        "model_name": "CNN5",
        "learning_rate": 1e-4,
    },
    {
        "experiment_id": 54,
        "model_name": "CNN5",
        "learning_rate": 1e-5,
    },
    {
        "experiment_id": 55,
        "model_name": "CNN4",
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    {
        "experiment_id": 56,
        "model_name": "CNN5",
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    {
        "experiment_id": 57,
        "model_name": "CNN4",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 58,
        "model_name": "CNN5",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 59,
        "model_name": "CNN4",
        "optimizer": "sgd",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 60,
        "model_name": "CNN4",
        "optimizer": "adadelta",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 61,
        "model_name": "CNN5",
        "optimizer": "sgd",
        "learning_rate": 1e-3,
    },
    {
        "experiment_id": 62,
        "model_name": "CNN5",
        "optimizer": "adadelta",
        "learning_rate": 1e-4,
    },
    {
        "experiment_id": 63,
        "model_name": "CNN5",
        "optimizer": "adadelta",
        "learning_rate": 1e-3,
    },
    {

        "experiment_id": 64,
        "model_name": "CNN5",
        "optimizer": "adadelta",
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,

    },
# dropout_rate and augmentation to CNN4 and CNN5 configurations
    {
        "experiment_id": 65,
        "model_name": "CNN4",
        "epochs": 100,
        "batch_size": 64,
        "dropout_rate": 0.5,  # Add dropout rate
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "augmentation": True  # Add augmentation
    },
    {
        "experiment_id": 66,
        "model_name": "CNN5",
        "epochs": 100,
        "batch_size": 64,
        "dropout_rate": 0.5,  # Add dropout rate
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "augmentation": True  # Add augmentation
    },
    {
        "experiment_id": 67,
        "model_name": "CNN5",
        "dropout_rate": 0.5,
        "augmentation": True,
        "zoom_range": 0.1,
        "rotation_range": 4,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "shear_range": 0.1,
    },
    {
        "experiment_id": 68,
        "model_name": "CNN4",
        "dropout_rate": 0.5,
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 69,
        "model_name": "CNN4",
        "dropout_rate": 0.4,
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 70,
        "model_name": "CNN5",
        "dropout_rate": 0.4,
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 71,
        "model_name": "CNN5",
        "dropout_rate": 0.3,
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
    },
    {
        "experiment_id": 72,
        "model_name": "CNN5",
        "dropout_rate": 0.3,
        "augmentation": True,
        "zoom_range": 0.15,
        "rotation_range": 4,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "shear_range": 0.15,
        "optimizer": "adadelta"
    },
]
    # {
    #     "experiment_id": 43,
    #     "model_name": "ResNet50",
    #     "learning_rate": 0.0001,
    #     "optimizer": "adam",
    #     "augmentation": True,
    #     "epochs": 100
    # }]

# ({
#     "experiment_id": 9,
#     "model_name": "VGG16",
#     "learning_rate": 0.00005,
#     "optimizer": "rmsprop",
#     "augmentation": True,
#     "epochs": 100
# })



    # TODO: Add more experiments
    # Incremental layer addition
    # Hyperparameter tuning
    # Adding Dropout
    # Advanced Architecture

