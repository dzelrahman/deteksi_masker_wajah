
# impor package yang diperlukan

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
import argparse
import os


# membuat argumen parser dan lakukan parse terhadap argumen
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# menentukan learning rate awal dan jumlah EPOCHS sebagai parameter ketika melatih model, juga batch size

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# ambil list gambar pada direktori data kita, kemudian buat list berisi data gambar dan kelasnya

print("[INFO] memuat gambar...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop pada path gambar
for imagePath in imagePaths:
    # ambil label kelas dari nama file
    label = imagePath.split(os.path.sep)[-2]

    # muat input gambar (224x224) and lakukan pra-proses
    image = load_img(imagePath, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # perbarui list data dan dabel, secara berurut
    data.append(image)
    labels.append(label)

# konversi data dan label ke NumPy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# melakukan one-hot encoding pada label
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# pisahkan data menjadi training dan testing dengan rasio 80% dari data untuk training dan 20% untuk testing
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.20, stratify=labels, random_state=42)

# buat fungsi image generator pada data train untuk augmentasi data
aug = ImageDataGenerator(
		rotation_range = 20,
		zoom_range = 0.15,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		shear_range = 0.15,
		horizontal_flip = True,
		fill_mode = "nearest"
)

# muat network MobileNetV2, pastikan lapisan head FC dihapus
baseModel = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

# buat kepala dari model yang akan ditempatkan menggantikan kepala model dasar
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)\

#tempatkan model head FC di atas model dasar (ini akan menjadi model aktual yang akan kita latih)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop terhadap seluruh lapisan pada model dasar dan lakukan freeze sehingga tidak akan terupdate ketika proses latihan awal
for layer in baseModel.layers:
	layer.trainable = False

# kompilasi model
print("[INFO] melakukan kompilasi model...")
opt = Adam(lr=INIT_LR,decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

# melatih kepala dari network
print("[INFO] melatih kepala...")
H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch = len(trainX) // BS,
		validation_data = (testX, testY),
		validation_steps = len(testX) // BS,
		epochs = EPOCHS)

# membuat prediksi pada data test
print("[INFO] mengevaluasi network...")
predIdxs = model.predict(testX, batch_size=BS)

# untuk tiap index pada data tes kita perlu mendapatkan index label dengan kemungkinan prediksi terbesar
predIdxs = np.argmax(predIdxs, axis=1)

# menampilkan laporan klasifikasi yang terformat dengan baik
print(classification_report(testY.argmax(axis=1),predIdxs,target_names=lb.classes_))

# serialisasi model ke dalam komputer
print("[INFO] menyimpan model pendeteksi masker...")
model.save(args["model"], save_format="h5")


# plot loss training dan akurasi
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")
plt.title("Loss pada training dan akurasinya")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Akurasi")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
