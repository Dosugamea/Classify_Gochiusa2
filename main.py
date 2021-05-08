from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def build_model(num_classes, img_size):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="pred"
    )(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Initialize TPU or GPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

# Specify directory
TRAIN_DATA_DIR = "./gochiusa_dataset/main_dataset"
TEST_DATA_DIR = "./gochiusa_dataset/test_dataset"

# This depends by base model
IMG_SIZE = 224
size = (IMG_SIZE, IMG_SIZE)

# Load train data
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    batch_size=32, image_size=size,
    label_mode='categorical',
    shuffle=True,
    interpolation='bilinear',
    follow_links=False
)

# Load test data
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATA_DIR,
    batch_size=32, image_size=size,
    label_mode='categorical',
    shuffle=True,
    interpolation='bilinear',
    follow_links=False
)

class_names = ds_train.class_names

'''
Check loaded train data if you want
for images, labels in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
'''

'''
Check augmented train data if you want
for images, labels in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
'''

# Define base model
with strategy.scope():
    model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
    model.summary()
    hist = model.fit(ds_train, epochs=25, validation_data=ds_test, verbose=2)
plot_hist(hist)
