# import package yang dibutuhkan
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def deteksi_dan_prediksi_masker(frame, faceNet, maskNet):
    # mengambil dimensi dari frame dan buat blob darinya
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # masukkan blob ke dalam network dan dapatkan deteksi muka
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # memasukkan list muka, lokasi yang berhubungan, dan list prediksi dari network face mask kita
    faces = []
    locs = []
    preds = []

    # loop pada deteksi
    for i in range(0, detections.shape[2]):
        # ekstraksi tingkat kepercayaan diri (probabilitas) yang berasosiasi dengan deteksi
        confidence = detections[0, 0, i, 2]

        # keluarkan deteksi lemah dengan memastikan confidence lebih tinggi dari confidence minimum
        if confidence > args["confidence"]:
            # menghitung koordinat (x,y) dari bounding box (kotak batasan) untuk object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # pastikan bounding box berada di dalam dimensi frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # ekstrak ROI muka, konversikan dari saluran BGR menjadi RGB, ubah ukuran menjadi 224x224, dan lakukan pra-proses
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # tambahkan kotak muka dan bounding untuk tiap list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # hanya membuat prediksi jika paling tidak satu muka terdeteksi
    if len(faces) > 0:
        # untuk keputusan/kesimpulan yang lebih cepat, kita akan mengatur prediksi batch dengan "all"
        # artinya semua muka di waktu yang bersamaan bukan prediksi satu demi satu
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # kembalikan 2 tuple dari lokasi muka dan lokasi yang bertautan/bersangkutan
    return(locs, preds)

# Membuat argumen parser dan lakukan parse terhadap argumen
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# muat model face detector yang sudah diserialisasi ke dalam disk
print("[INFO] memuat model face detector...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# memuat model face mask detector dari file/disk
print("[INFO] memuat model face mask detector...")
maskNet = load_model(args["model"])

# memulai video stream dan izinkan sensor kamera untuk bersiap
print("[INFO] memulai video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# melakukan loop terhadap frame dari video stream
while True:
    # ambil frame dari video stream dan rubah ukurannya agar memiliki kelebaran maksima; 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # mendeteksi muka di dalam frame dan menentukan apakah mereka menggunakan masker atau tidak
    (locs,preds) = deteksi_dan_prediksi_masker(frame, faceNet, maskNet)

    # melakukan loop terhadap lokasi muka yang terdeteksi dan lokasi yang berhubungan
    for (box, pred) in zip(locs, preds):
        # membuka bounding box dan prediksi
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # menentukan label kelas dan warna yang akan digunakan untuk gambar bounding box dan teks
        label = "Make Masker" if mask > withoutMask else "ga pake masker"
        color = (0, 255, 0) if label == "Make Masker" else (0, 0, 255)

        # menyertakan probabilitas di dalam label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # menampilkan label dan kotak bounding box pada keluaran frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # menampilkan keluaran frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # jika tombol "q" ditekan, berhenti melakukan loop
    if key == ord("q"):
        break

# melakukan pembersihan (cleanup)
cv2.destroyAllWindows()
vs.stop()
