# Gerekli kütüphaneleri projemize dahil ettik.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Veri setindeki image'leri eğitimde kullanmak için yolumuzu belirledik.
DIRECTORY = r"C:\Users\bertu\OneDrive\Masaüstü\Face-Mask-Detection\data"
# Bu yoldaki kullanacağımız veri setlerini categories listesine atadık.
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Fotoğraflar yükleniyor...")

data = []  # Image değerlerini tutacak olan data listesi.
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image) # Fotoğrafı modelin gerektirdiği biçime uygun hale getirmek için kullandık.

    	data.append(image)
    	labels.append(category)

# Çok sınıflı etiketleri ikili etiketlere dönüştürmek için kullandık.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Eğitim ve test verilerini iki alt kümeye bölmek için kullandık. Bu fonksiyon ile veri setini manuel olarak bölmeniz gerekmez.
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Veri artırımı kullanarak büyük olan image'leri küçültmek için ImageDataGenerator oluşturduk.
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
	layer.trainable = False

# Modelimizi derledik.
print("[INFO] Model derleniyor...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Head eğitip, doğru tahminler yapmasını sağladık.
print("[INFO] Head eğitiliyor...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Veri setlerinin tahminde bulunmasını sağladık.
print("[INFO] Veriler tahminde bulunuluyor...")
predIdxs = model.predict(testX, batch_size=BS)


predIdxs = np.argmax(predIdxs, axis=1)


print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Maske takip modelimizi oluşturduk ve kaydettik.
print("[INFO] Mask detector modeli kaydediliyor...")
model.save("mask_detector.model", save_format="h5")

# Eğitim kaybını ve eğitimin doğruluğunu çizmek ve göstermek için bu kodları kullandık.
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("graphic.png")