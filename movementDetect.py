# Pyhton program to implement
# WebCam Motion Detector

# importing OpenCV, time and Pandas library
from _thread import start_new_thread

import cv2, time, pandas, numpy, os
# importing datetime class from datetime library
from datetime import datetime

# Assigning our static_back to None
import pyttsx3 as pyttsx3

static_back = None

# List when any moving object appear
motion_list = [None, None]

# Time of movement
time = []
datNames = []

# Initializing DataFrame, one column is start
# time and other column is end time
df = pandas.DataFrame(columns=["Start", "End"])


# speak function

def robot():
    engine = pyttsx3.init()
    engine.say("Motion detected")
    engine.runAndWait()


# Capturing video

# face detection code start
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'c:/datasets'
# Part 1: Create fisherRecognizer
print('Training...')
spoke = 0
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face_LBPHFaceRecognizer.create()
model.train(images, lables)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
# face detection code end

# video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
    video = cv2.VideoCapture('http://192.168.43.1:8080/shot.jpg')

    # Reading frame(image) from video
    check, frame = video.read()

    # Initializing motion = 0(no motion)
    motion = 0

    # face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 500:

            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            datNames.append(names[prediction[0]])
            if spoke == 0:
                engine = pyttsx3.init()
                engine.say("found " + names[prediction[0]])
                engine.runAndWait()
                spoke = 1
        else:
            cv2.putText(frame, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    # face detection end

    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In first iteration we assign the value
    # of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue

    # Difference between static background
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # robot()  # say motion detected

    # Appending status of motion
    motion_list.append(motion)

    motion_list = motion_list[-2:]

    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

    # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    # Displaying image in gray_scale
    #  cv2.imshow("Gray Frame", gray)

    # Displaying the difference in currentframe to
    # the staticframe(very first_frame)
    # cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and white image in which if
    # intencity difference greater than 30 it will appear white
    # cv2.imshow("Threshold Frame", thresh_frame)

    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        # if something is movingthen it append the end time of movement
        if motion == 1:
            time.append(datetime.now())
        break

# sort distant names

# Appending time of motion in DataFrame
for i in range(0, len(time), 2):
    df = df.append({"Start": time[i], "End": time[i + 1], "Person": datNames}, ignore_index=True)

# Creating a csv file in which time of movements will be saved
df.to_csv("c:/movements/Time_of_movements.csv")

video.release()

# Destroying all the windows
cv2.destroyAllWindows()
