import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Create a simple dummy model for 5 classes
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save dummy model with a different name
model.save('dummy_model.h5')
print("Dummy model saved as dummy_model.h5")