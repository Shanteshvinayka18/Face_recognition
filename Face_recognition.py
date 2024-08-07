import face_recognition
import cv2

known_face_encodings = [] # list of face encodings of known faces. 
known_face_names = [] # list of names of the faces known.

image1 = face_recognition.load_image_file(r"c:\Users\Niti Jain\Downloads\SHANTESH VINAYKA PHOTO.jpg")
face_encoding1 = face_recognition.face_encodings(image1)[0]
known_face_encodings.append(face_encoding1)
known_face_names.append("Shantesh Vinayka.")

face_locations = []# list of face locations of the comparing image.
face_encodings = []# list of encoded face of the comparing image.

img = cv2.imread(r"c:\Users\Niti Jain\Downloads\IMG_0665.jpeg")  #reads the image that is to be compared.
rgb_small_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the image to rgb format.

face_locations = face_recognition.face_locations(rgb_small_frame) # caputre the face locations from the image.
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

for x in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, x)
if matches[0] == True:
    print("Similar Image.")
else:
    print("Different Image.")
    
