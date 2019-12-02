import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)                                                   # captures frames from a camera/webcam

while 1:

    ret, image = capture.read()                                                 # reads frames from a camera/webcam

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                              # converts to gray scale of each frames

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                         # Detects faces of different sizes from the input image

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)                      # Drawing rectangle on a detected face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)                           # Detects eyes of different sizes in the input image

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)        # Drawing rectangle in detected eyes

    cv2.imshow('image',image)                                                   # Display an image in a window

    a = cv2.waitKey(30) & 0xff                                                  # Wait for Esc key to stop
    if a == 27:
        break

capture.release()                                                               # Close the window

cv2.destroyAllWindows()                                                         # De-allocate any associated memory usage
