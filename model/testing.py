import cv2
import io
import base64
import numpy as np
import warnings
import threading
from PIL import Image
from time import sleep
from keras.models import load_model
# from keras.models import load_model
# import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import os

warnings.filterwarnings("ignore")

my_dir = os.path.dirname(__file__)
model_tflite_path = os.path.join(my_dir, 'model.tflite')

# # load model
# # model = load_model("best_model.h5")
# # Load TFLite model and allocate tensors.
# interpreter = tflite.Interpreter(model_path=model_tflite_path)
# interpreter.allocate_tensors()
# output = interpreter.get_output_details()[0]  # Model has single output.
# input = interpreter.get_input_details()[0]  # Model has single input.
# face_haar_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# cap = cv2.VideoCapture(0)
# input_str = self.to_process.pop(0)
# imgdata = base64.b64decode(input_str)
# cap = np.array(Image.open(io.BytesIO(imgdata)))


class VideoCamera(object):
    def __init__(self):

        self.to_process = []
        self.output_image_rgb = []
        self.output_image_bgr = []
        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def generate_frames(self):
        if not self.to_process:
            return
        # gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        input_str = self.to_process.pop(0)
        imgdata = base64.b64decode(input_str)
        input_img = np.array(Image.open(io.BytesIO(imgdata)))

        faces_detected = face_haar_cascade.detectMultiScale(input_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(input_img, (x, y), (x + w, y + h),
                          (255, 0, 0), thickness=7)
            # cropping region of interest i.e. face area from  image
            roi_gray = input_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            # mengubah gambar menjadi array
            img_pixels = np.asarray(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            # numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'divide' output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'
            # handling error
            img_pixels = img_pixels / 255
            # print(img_pixels)
            img_pixels = np.float32(img_pixels)
            # interpreter.set_tensor(input['index'], img_pixels)
            # interpreter.invoke()
            # predictions = interpreter.get_tensor(output['index'])
            # print(predictions)
            # predictions = model.predict(img_pixels)

            predictions = model.predict(img_pixels)
            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy',
                        'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(input_img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(input_img, (1000, 700))
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        return frame

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.output_image_rgb:
            sleep(0.05)
        return self.output_image_rgb.pop(0), self.output_image_bgr.pop(0)

    # buffer = cv2.imshow('Facial Emotion Detection', resized_img)
    # frame = buffer.tobytes()
#     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
#         break
# cap.release()
# cv2.destroyAllWindows
