import cv2
import os

# Constants
COUNT_LIMIT = 30
POS = (30, 60)  # top-left
FONT = cv2.FONT_HERSHEY_COMPLEX  # font type for text overlay
HEIGHT = 1.5  # font_scale
TEXTCOLOR = (0, 0, 255)  # BGR- RED
BOXCOLOR = (255, 0, 255)  # BGR- BLUE
WEIGHT = 3  # font-thickness
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n----Enter User-id and press <return>----')
print("\n [INFO] Initializing face capture. Look at the camera and wait!")

# Create an instance of the VideoCapture object
cam = cv2.VideoCapture(0)

count = 0

while True:
    # Capture a frame from the camera
    ret, frame = cam.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break
    
    # Display count of images taken
    cv2.putText(frame, 'Count:' + str(int(count)), POS, FONT, HEIGHT, TEXTCOLOR, WEIGHT)

    # Convert frame from BGR to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a DS faces- array with 4 elements- x,y coordinates (top-left corner), width and height
    faces = FACE_DETECTOR.detectMultiScale(
        frameGray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        # Create a bounding box across the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOXCOLOR, 3)
        count += 1  # increment count

        # if dataset folder doesnt exist create:
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        # Save the captured bounded-grayscaleimage into the datasets folder only if the same file doesn't exist
        file_path = os.path.join("dataset", f"User.{face_id}.{count}.jpg")
        if os.path.exists(file_path):
            # Move the existing file to the "old_dataset" folder
            old_file_path = file_path.replace("dataset", "old_dataset")
            os.rename(file_path, old_file_path)
        # Write the newer images after moving the old images
        cv2.imwrite(file_path, frameGray[y:y + h, x:x + w])

    # Display the original frame to the user
    cv2.imshow('FaceCapture', frame)
    # Wait for 30 milliseconds for a key event (extract sigfigs) and exit if 'ESC' or 'q' is pressed
    key = cv2.waitKey(100) & 0xff
    # Checking keycode
    if key == 27:  # ESCAPE key
        break
    elif key == 113:  # q key
        break
    elif count >= COUNT_LIMIT:  # Take COUNT_LIMIT face samples and stop video capture
        break

# Release the camera and close all windows
print("\n [INFO] Exiting Program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()
