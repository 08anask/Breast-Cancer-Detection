import matplotlib
matplotlib.use("Agg")  # For saving plots without GUI

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adagrad  # or Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import time

# Training config
NUM_EPOCHS = 50
INIT_LR = 1e-2
BS = 32

# Define a custom callback for logging training details
class TrainingLogger(Callback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        # Extract the logs for loss, accuracy, etc.
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        
        # Write the logs to a text file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{self.params['epochs']} - "
                    f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                    f"train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}\n")

# Load and prepare image paths and labels
trainPaths = list(paths.list_images(config.TRAIN_PATH))
lenTrain = len(trainPaths)
lenVal = len(list(paths.list_images(config.VAL_PATH)))
lenTest = len(list(paths.list_images(config.TEST_PATH)))

trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals
classWeight = {i: classWeight[i] for i in range(len(classWeight))}

# Image augmentation
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

# Data generators
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# Build and compile model
model = CancerNet.build(width=48, height=48, depth=3, classes=2)

opt = Adagrad(learning_rate=INIT_LR)
# opt = Adam(learning_rate=INIT_LR)  # Optional: try Adam too
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
log_file = "training_details.txt"  # Log file for training details
logger = TrainingLogger(log_file)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=40,restore_best_weights=True),
    logger
]

# Train the model
M = model.fit(
    trainGen,
    validation_data=valGen,
    class_weight=classWeight,
    epochs=NUM_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

"""model_filename = f"model_last_epoch_{int(time.time())}.h5"
print(f"Saving model to {model_filename}")
# Check if the file already exists and remove it
if os.path.exists(model_filename):
    os.remove(model_filename)"""

# Save the model
#model.save(model_filename)

# Optionally, save it in the newer .keras format as well
model.save(f"model_last_epoch_{int(time.time())}.keras")

"""if os.path.exists("model.h5"):
    os.remove("model.h5")
model.save("model.h5")  # Save the model to HDF5 format


if os.path.exists("model.keras"):
    os.remove("model.keras")
model.save("model.keras") """

# Evaluate model
print("Now evaluating the model")
testGen.reset()
pred_indices = model.predict(testGen, steps=(lenTest // BS) + 1)
pred_indices = np.argmax(pred_indices, axis=1)

print(classification_report(testGen.classes, pred_indices, target_names=testGen.class_indices.keys()))

# Confusion matrix and metrics
cm = confusion_matrix(testGen.classes, pred_indices)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(cm)
print(f'Accuracy: {accuracy}')
print(f'Sensitivity (Recall for positive class): {sensitivity}')
print(f'Specificity (Recall for negative class): {specificity}')

# Save training plot
N = len(M.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), M.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# Save confusion matrix heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Healthy", "Cancer"], yticklabels=["Healthy", "Cancer"], cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")