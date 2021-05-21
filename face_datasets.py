# Import OpenCV2 for image processing
import cv2
import os

def path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        return 0
    else:
        print("User exists....")
        count = file_count(path)
        return count

def file_count(path):
    list = os.listdir(path) # dir is your directory path
    number_files = len(list)
    print("number of data set captured previouslh is :",number_files)
    return number_files

# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, one face id
folder=input('Enter the user name: ')

# Initialize sample face image
count = 0

path="knn_examples/train/"
path+=(str(folder)+"/")

count=path_exists(path)
capture = False
# Start looping
print('press \'s\' to capture (for 100ms)')
print('press \'p\' to pause (for 100ms)')
print('press \'q\' to stop capturing and exist (for 100ms)')
while(True):

    # Capture video frame
    ret, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # Increment sample face image
        

        # Save the captured image into the datasets folder
        if cv2.waitKey(1)== ord('s'):
            print("Started to capture")
            capture = True

        elif cv2.waitKey(1)== ord('p'):
            print("Stopped capturing")
            capture = False

        if capture:
            image_file_name=str(path + str(folder) + '_' + str(count) + ".jpg")
            cv2.imwrite(image_file_name, image_frame[y:y+h,x:x+w])
            count += 1

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("Images saved under ur Name!")
        print('Note: to recapture enter the same name!')
        print("Thanks! ",str(folder))
        break

    # If image taken reach 100, stop taking video
    elif count>1000:
        
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
