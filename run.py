import cv2
import numpy as np
from keras.models import load_model

model = load_model("emotion_detection_model.h5")

CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
IMG_HEIGHT = 48
IMG_WIDTH = 48

def preprocess_image(image):
    resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Create three channels from the grayscale image
    input_image = cv2.merge([gray_image] * 3)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    input_image = preprocess_image(frame)

    predictions = model.predict(input_image)
    emotion_label = CLASS_LABELS[np.argmax(predictions)]

    cv2.putText(frame, emotion_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
