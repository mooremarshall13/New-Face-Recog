import cv2
import numpy as np
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# initialize id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Keneth']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

recording = False
start_time = None
out = None

while True:
    ret, img = cam.read()
    img = cv2.flip(img, -1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 50 ==> "0" is a perfect match
        if confidence < 55:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10)

    if k == 27:  # Press 'ESC' for exiting the video
        break
    elif k == 32:  # Press spacebar to capture an image
        cv2.imwrite('image/captured_image.jpg', img)
        print("Image captured.")
    elif k == ord('v'):  # Press 'v' to start recording
        recording = True
        start_time = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('video/captured_video.mp4', fourcc, 20.0, (int(cam.get(3)), int(cam.get(4))))
        print("Recording started.")
    elif recording and time.time() - start_time >= 5:  # Automatically stop recording after 5 seconds
        recording = False
        out.release()
        print("Recording stopped.")

    if recording:
        out.write(img)

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
