import cv2
import numpy as np
import warnings
# from keras.models import load_model
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os

warnings.filterwarnings("ignore")

my_dir = os.path.dirname(__file__)
model_tflite_path = os.path.join(my_dir, 'model.tflite')

# load model
# model = load_model("best_model.h5")
# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_tflite_path)
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        # captures frame and returns boolean value and captured image
        ret, test_img = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h),
                          (255, 0, 0), thickness=7)
            # cropping region of interest i.e. face area from  image
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            # mengubah gambar menjadi array
            img_pixels = np.asarray(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            # numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'divide' output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'
            # handling error
            img_pixels = img_pixels / 255
            # print(img_pixels)
            img_pixels = np.float32(img_pixels)
            interpreter.set_tensor(input['index'], img_pixels)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output['index'])
            # print(predictions)
            # predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy',
                        'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        return frame

    # buffer = cv2.imshow('Facial Emotion Detection', resized_img)
    # frame = buffer.tobytes()
#     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
#         break
# cap.release()
# cv2.destroyAllWindows
