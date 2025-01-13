from tensorflow.keras.applications import VGG16

# Load the VGG16 model
model = VGG16(weights='imagenet')
print("VGG16 loaded successfully!")
