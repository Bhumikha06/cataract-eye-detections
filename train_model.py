import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU (if no GPU available)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Dataset path
train_data_dir = "data/"

# Preprocessing & augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # ✅ multi-class
    subset="training"
)

val_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # ✅ multi-class
    subset="validation"
)

# Load ResNet152V2 pretrained model
base_model = ResNet152V2(weights="imagenet", include_top=False, input_shape=(224,224,3))

# Freeze most layers, unfreeze last 30 for fine-tuning
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
output = Dense(3, activation="softmax")(x)  # ✅ 3 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("model/best_resnet_model.h5", monitor="val_accuracy",
                             save_best_only=True, verbose=1)

# Train
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("model/resnet_model.h5", include_optimizer=False)

print("✅ Training complete. Best model saved in model/best_resnet_model.h5")
print("Class Indices:", train_generator.class_indices)