import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Input, BatchNormalization, Reshape, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


#Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalize images to [0,1] range
x_train, x_test = x_train / 255.0, x_test / 255.0

#Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


#Gaussian Membership Function
def gaussian_mf(x, mean, sigma):
    mean = tf.cast(mean, tf.float32)  # Ensure mean is float32
    sigma = tf.cast(sigma, tf.float32)  # Ensure sigma is float32
    return tf.exp(-tf.square(x - mean) / (2 * tf.square(sigma)))

#Fuzzy Convolutional Layer
class FuzzyConvLayer(Layer):
    def __init__(self, filters, kernel_size):
        super(FuzzyConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = None
        self.mean = None
        self.sigma = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
                                      initializer="random_normal",
                                      trainable=True,
                                      dtype=tf.float32)
        self.mean = self.add_weight(shape=(1,), initializer="zeros", trainable=True, dtype=tf.float32)
        self.sigma = self.add_weight(shape=(1,), initializer="ones", trainable=True, dtype=tf.float32)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)
        fuzzy_inputs = gaussian_mf(inputs, self.mean, self.sigma)
        fuzzy_output = tf.nn.conv2d(fuzzy_inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return fuzzy_output

#Create Fuzzy CNN Model
model = Sequential([
    Input(shape=(28, 28, 1)),

    FuzzyConvLayer(filters=32, kernel_size=3),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    FuzzyConvLayer(filters=64, kernel_size=3),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax")
])

#Compile Model
model.compile(optimizer=Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


#Train Model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,  
    verbose=1
)


#Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")


#Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")


plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

#Get Model Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax outputs to class labels
y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

#Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

#Plot the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for MNIST Classification")
plt.show()
