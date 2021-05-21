import pandas as pd
import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders


df= pd.read_csv('Database.csv')
name_list=list(df['name'])
Id_list=list(df['mailid'])
pas_list=list(df['password'])

print("WELCOME! \n please login with ur credentials")
ty_name=input("Enter your name")
ty_pass=input("Enter your password")

#checking if user exits
def mail(msg,ID,attch_file=None):
    print("Sending MAIL to : ",ID)
    fromaddr = "raspberryp087@gmail.com" 
    msg_mail = MIMEMultipart() 
    msg_mail['From'] = fromaddr 
    msg_mail['To'] = ID
    msg_mail['Subject'] = "IGEKKS SECURITY LOGIN STATUS"
    body =msg+"\n\n IGEEKS TECH\nSystem generated mail"
    msg_mail.attach(MIMEText(body, 'plain'))
    if attch_file is not None:
        filename = attch_file
        attachment = open(filename, "rb") 
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read()) 
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg_mail.attach(p) 
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(fromaddr,"Raspberry@123")
        text = msg_mail.as_string()
        s.sendmail(fromaddr,ID, text)
        s.quit()
    else:
        s = smtplib.SMTP('smtp.gmail.com', 587) 
        s.starttls()
        s.login(fromaddr,"Raspberry@123")  
        text = msg_mail.as_string()   
        s.sendmail(fromaddr,ID, text) 
        s.quit()
    return

def Second_auth(Log_name,ID):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    
            if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
                raise Exception("Invalid image path: {}".format(X_img_path))
        
            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        

            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)
        
            # Load image file and find face locations
            X_img = face_recognition.load_image_file(X_img_path)
            X_face_locations = face_recognition.face_locations(X_img)
        
            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []
        
            # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
        
            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        
            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
        
    def show_prediction_labels_on_image(frame, predictions):

            
        
            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(frame,(left, top), (right, bottom),(0, 0, 255),2)
        

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
 
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        

            cv2.imshow('faceRecognisation',frame) 
            
    print("Second Auth! in progress")
    print("Predicting!!!!  ")
    unknown_count=0
    know_count=0
    cap=cv2.VideoCapture(0)
    # STEP 2: Using the trained classifier, make predictions for unknown images
    while(1):
        ret,frame=cap.read()
        save_recent='C:/Users/Pavan R/Desktop/phyton_class/class1/FaceRecognisation_usingKNN/knn_examples/Recent/f1.jpeg'
        cv2.imwrite(save_recent,frame)
        predictions = predict(save_recent, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print(" Found : {} ".format(name))
            if(name=='unknown' or name!= Log_name):
                unknown_count+=1
                if(unknown_count>5):
                    
                    print("UNKNOWN PERSON DTECTED")
                    msg="UNKNOWN PERSON DTECTED, who logged in with ur creden. Please change the password"
                    cv2.imwrite("unknow_face.jpg",frame)
                    print("Image saved")
                    mail(msg,ID,'unknow_face.jpg')
                    return
            elif name == Log_name:
                know_count+=1
                if(know_count>10):
                    unknown_count=0
                    print("LOGGED IN SUCCESFULL!!!!!!! ", name)
                    msg="Logged in successfull with IGEEKS SEcurity"
                    mail(msg,ID)
                    return

                
            print("unknown_count:  ",unknown_count)
            print("Know_count:    ",know_count)

        # Display results overlaid on an image
        show_prediction_labels_on_image(frame, predictions)
        if(cv2.waitKey(1)==ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    pass

if ty_name in name_list:
    index=name_list.index(ty_name)
    print(index)
    if(ty_pass==pas_list[index]):
        print("Authentication succesfull")
        Second_auth(ty_name,Id_list[index])
    else:
        print("Incorrect password:")
        msg="Tried to login with incorrect password"
        ID=Id_list[index]
        mail(msg,ID)
else:
    print("USER DOESNT EXIST!!")
        
