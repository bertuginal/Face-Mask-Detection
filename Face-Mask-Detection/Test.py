# Kullanacağımız kütüphaneleri çalışmamıza dahil ettik.
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

cascade = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml" # Kullanacağımız cascade dosyamızı projemize dahil ettik.
faceCascade = cv2.CascadeClassifier(cascade)

model = load_model("./mask_detector.model") # Kullanacağımız modeli projemize dahil ettik.

camera = cv2.VideoCapture(0) # Webcam'den veya harici bir kameradan gerçek zamanlı bir görüntü almak için kullandık.

while True:
    ret, frame = camera.read() # Sonsuz bir döngü ile her kareyi(frame) tek tek inceledik.
    frame = cv2.flip(frame, 1, 1) # Kamera frame'lerinin ayna görüntüsünü almak için kullandık.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Haar-like özellikleri kolay algılayabilmek için her bir kareyi boz(gri) tonlara çevirdik.
    faces = faceCascade.detectMultiScale(gray, # Cascade dosyamızı kullanarak her bir kare üzerindeki bir çok yüzün koordinarlarını bulduk.
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    
    faces_list = [] # frame'lerin değerlerini tutmak için bir liste oluşturduk.
    preds = []

    for (x, y, w, h) in faces: # Yüzün koordinatları belirlenen köşe noktalarını çizmek için for döngüsü oluşturduk.
        face_frame = frame[y:y + h, x:x + w] # Yüzün koordinatlarına göre yüzü dikdörtgen içine almak için kullandık.
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB) # frame'leri gerçek renklerine çevirmek için kullandık.
        face_frame = cv2.resize(face_frame, (224, 224)) # frame'leri tekrar boyutlandırmak için kullandık.
        face_frame = img_to_array(face_frame) # her bir frame'in değerini bir diziye, listeye atamak için kullandık.
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame) # Oluşturulan listenin sonuna yeni frame değerini eklemek için kullandık.
        if len(faces_list) > 0:
            for face in faces_list: # 1'den fazla yüzü yakalamak için for döngüsü oluşturduk.
                preds = model.predict(face)  # Verilen test verisinin çıktısını bulduk ve preds listesine atadık.

        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask" # Maske varsa "Mask", maske yoksa "No Mask" yazması için kullandık.
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Maske varsa yeşil, maske yoksa kırmızı renk yapması için bu kodu yazdık.
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # Maske takıldığı ve takılmadığı zamanki yüzdesini görmek için kullandık.
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) # Oluşan frame'deki dikdörtgen şeklin üzerine metin yazmak için bu kodu kullandık.

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # Yüzün etrafına dikdörtgen çizmek için kullandık.

    # İşlediğimiz kareleri görelim.
    cv2.imshow('Video', frame)
    # Programı ESC tuşuna basınca sonlandıracak kodu yazdık.
    if cv2.waitKey(10) == 27:
        break

camera.release() # Kamerayı bırakmak için kullandık.
cv2.destroyAllWindows() # Açık kalan tüm pencereleri programı sonlandırırken kapatmak için kullandık.