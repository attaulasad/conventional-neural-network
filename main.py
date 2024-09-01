import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models as keras_models
from tensorflow.keras.optimizers import Adam

# Create synthetic dataset
def create_synthetic_data(num_samples=1000, img_size=128):
    X = np.random.rand(num_samples, img_size, img_size, 1)
    Y = (X > 0.5).astype(np.float32)
    return X, Y

X, Y = create_synthetic_data()

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# U-Net model
def unet_model(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x

    def encoder_block(x, filters):
        x = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, skip])
        x = conv_block(x, filters)
        return x

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = keras_models.Model(inputs, outputs)
    return model

# Model compilation and training
def compile_and_train_model(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, learning_rate=1e-4):
    model = unet_model()
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)
    return model, history

# Hyperparameter tuning and cross-validation
kf = KFold(n_splits=5)
histories = []
trained_models = []
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train[train_index], X_train[val_index]
    Y_tr, Y_val = Y_train[train_index], Y_train[val_index]
    model, history = compile_and_train_model(X_tr, Y_tr, X_val, Y_val)
    trained_models.append(model)
    histories.append(history)

# Evaluate the final model on the test set
best_model = trained_models[0]
test_predictions = best_model.predict(X_test)
test_predictions_binary = (test_predictions > 0.5).astype(np.float32)

# Evaluation metrics
accuracy = accuracy_score(Y_test.flatten(), test_predictions_binary.flatten())
precision = precision_score(Y_test.flatten(), test_predictions_binary.flatten())
recall = recall_score(Y_test.flatten(), test_predictions_binary.flatten())
f1 = f1_score(Y_test.flatten(), test_predictions_binary.flatten())

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(histories[0].history['loss'], label='Training Loss')
plt.plot(histories[0].history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(histories[0].history['accuracy'], label='Training Accuracy')
plt.plot(histories[0].history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Visualize predictions
def visualize_predictions(X_test, Y_test, predictions, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(Y_test[i].squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')

        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

visualize_predictions(X_test, Y_test, test_predictions_binary)