# 1. Import libraries and load data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST (auto-downloads if needed)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualize sample
plt.imshow(x_train[0], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()

# 2. Preprocess: Normalize to 0-1, add channel dim for CNN
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# 3. Build CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Prevents overfitting
    keras.layers.Dense(10, activation='softmax')
])

model.summary()  # See layer details

# 4. Compile: Adam optimizer, cross-entropy loss for classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train (5 epochs ~5-10 mins, expect 99% accuracy)
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test),
                    batch_size=128)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Predict sample
predictions = model.predict(x_test[:5])
print("Predicted digits:", np.argmax(predictions, axis=1))
print("Actual digits:", y_test[:5])
