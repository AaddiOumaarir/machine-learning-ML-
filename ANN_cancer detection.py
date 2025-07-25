import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

 
image_size = (64, 64)
path = "/content/drive/MyDrive/archive (2)/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/"

# Chargement des données
X, y = [], []
for label in ['Normal cases', 'Bengin cases', 'Malignant cases']:
    folder = os.path.join(path, label)
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert('L')  
            img = img.resize(image_size)
            img = np.array(img) / 255.0

            X.append(img.flatten())  
            y.append(label)


X = np.array(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


loss, acc = model.evaluate(X_test, y_test)
print("Précision sur test :", acc)


plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entraînement')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Fonction de perte (Loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entraînement')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Précision (Accuracy)")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.imsave('sample_image.png', img)
plt.show()