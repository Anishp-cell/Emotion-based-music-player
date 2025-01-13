from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the VGG16 model without the top (classifier) layers
basemodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16 so they are not updated during training
for layer in basemodel.layers:
    layer.trainable = False

# Add custom layers for emotion classification
x = Flatten()(basemodel.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x)  # Assuming 7 emotions

# Create the final model
model = Model(inputs=basemodel.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to check the architecture
model.summary()
