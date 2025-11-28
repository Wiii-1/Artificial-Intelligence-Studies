# MNIST Image Classification (student explanation)

This is a short README written like a 3rd-year CS student explaining what `image_classification.py` does, how to run it, and some easy next steps.

## What this script does (plain talk)
- Loads the MNIST dataset (handwritten digits 0–9). The dataset comes built-in with TensorFlow and will be downloaded automatically if you don't have it.
- Preprocesses the images by scaling pixel values to the 0–1 range and adding a channel dimension so the images fit into a CNN.
- Builds a small Convolutional Neural Network (CNN) with two Conv+Pool blocks, a dense hidden layer, dropout, and an output softmax layer with 10 classes.
- Trains the model for a few epochs and evaluates accuracy on the test set.
- Shows a sample image, prints training progress, and prints final test accuracy and a few example predictions.

## Files
- `image_classification.py` — the Python script that does everything described above.

## How to run (short and copyable)
1. Make sure you have Python 3.8+ and pip installed.
2. Install dependencies (PowerShell example):

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow matplotlib numpy
```

3. Run the script from the `DeepLearning` folder:

```powershell
python image_classification.py
```

You will see a sample MNIST digit plotted, training progress printed to the console, and test accuracy printed at the end. The script also prints a few model predictions for the first 5 test images.

## What the main parts mean (quick guide)
- Data loading: `keras.datasets.mnist.load_data()` gives `(x_train, y_train), (x_test, y_test)`.
- Preprocessing: dividing by 255 converts pixels from 0–255 into floats between 0 and 1. `np.expand_dims(..., -1)` adds the channel dimension for the CNN.
- Model: Conv2D layers learn spatial filters; MaxPooling reduces spatial size; Flatten → Dense builds the classifier; Dropout helps reduce overfitting; final Dense(10, softmax) outputs class probabilities.
- Loss & optimizer: using `sparse_categorical_crossentropy` because labels are integers, and `adam` is a sensible default optimizer.
- Training: `model.fit(...)` runs multiple passes (epochs) over the training data and shows loss/accuracy.
- Evaluation: `model.evaluate(...)` reports final loss and accuracy on the unseen test set.

## Expected results & runtime
- With the current architecture and 5 epochs, expect test accuracy around 98–99% on a typical machine (CPU training may take several minutes; GPU is faster).
- Training time depends on your CPU/GPU. On a typical laptop CPU it might take ~5–15 minutes for 5 epochs; with a GPU it’s much faster.

## Common issues and quick fixes
- If TensorFlow install fails on Windows, ensure you have a compatible Python version and follow TensorFlow's Windows install guide.
- If training is too slow and you don't have a GPU, reduce `batch_size` or `epochs` in the `fit()` call.
- Use a virtual environment to avoid dependency conflicts:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow matplotlib numpy
```

## Small experiments (good for class projects)
1. Change architecture: add/remove Conv layers or change filter counts (32 → 64 → 128).
2. Data augmentation: rotate or shift images during training to make the model more robust.
3. Hyperparameters: try different `batch_size`, `epochs`, or optimizers (SGD with momentum).
4. Evaluation: add confusion matrix and per-class accuracy to see which digits are confused.

## Next steps if you want help
- I can add a small training script that saves the model weights and a `predict.py` that loads the model and predicts on new images.
- I can add data augmentation (TF Keras ImageDataGenerator or `tf.image`) and improve the README with sample commands.

Enjoy — this is a great starting point for learning CNNs and image classification.
