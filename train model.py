from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Reserve 20% of training data for validation
)

train_generator = datagen.flow_from_directory(
    'D:/python/Capstonep/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'D:/python/Capstonep/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'D:/python/Capstonep/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# 2. Load Pre-trained VGG16 Model (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 3. Add Custom Classification Layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Match the number of classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the VGG16 layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# 4. Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Adjust steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = val_generator.samples // val_generator.batch_size

# Check for the actual batch size and dataset size to avoid errors
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# 6. Train the Model
history = model.fit(
    train_generator,
    epochs=20,  # Adjust as needed
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# 7. Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
