import face_recognition
import cv2
import numpy as np 
import boto3
import os
from botocore.exceptions import NoCredentialsError
#Client creation for the AWS Connectivity.
s3client = boto3.client('s3', aws_access_key_id = 'ACCESS_KEY_ID', aws_secret_access_key = 'SECRET_ACCESS_KEY') 


buckets_info = s3client.list_buckets() # Information of all the Buckets in the S3 in Dictionary format.  
print("Name of all the Buckets : ")
print()
for bucket in buckets_info['Buckets']:
    print(bucket['Name'])
print()
bucket_name = input("Enter the bucket name = ")
print()
print(f"list of keys in the bucket {bucket_name} : ")
objects_info = s3client.list_objects(Bucket = bucket_name) #Information of all the objects present in the Specified Bucket in Dictionary format.
for object in objects_info['Contents']:
    print(object['Key'])


 # Upload File.
def aws_file_upload(file_name, bucket_name, object_name=None):
    s3client = boto3.client('s3', aws_access_key_id = 'ACCESS_KEY_ID', aws_secret_access_key = 'SECRET_ACCESS_KEY') #Client creation for the AWS Connectivity.
    
    if object_name is None: 
        object_name = os.path.basename((file_name))     
    try : 
        s3client.upload_file(file_name,bucket_name,object_name)
        print("upload Successful.")
        return True
    except FileNotFoundError:
        print("File not Found Error.")
        return False
    except NoCredentialsError:
        print("Credentials Not Available.")
        return False


# Scanning the image from the specified object.        
def face_encodings(bucket_name , key_name):
    object_info= s3client.get_object(Bucket = bucket_name, Key =key_name)
    object_in_byte = object_info['Body'].read()
    object_in_array = np.frombuffer(object_in_byte, dtype=np.uint8)
    object_img = cv2.imdecode(object_in_array,cv2.IMREAD_COLOR)
    object_in_rgb = cv2.cvtColor(object_img,cv2.COLOR_BGR2RGB)
    object_face_location = face_recognition.face_locations(object_img)
    try:
        object_face_encodings = [face_recognition.face_encodings(object_in_rgb,object_face_location)[0]]#encodings of face 1 in list.
    except IndexError:
        print()
        print("Face not found in the image.")
        exit()
    return object_face_encodings


# prints percentage of similarity if the images are similar person.
def true_images(object2_face_encodings, object1_face_encodings):
    print()
    print("The images are of same person.")
    faces_distance = face_recognition.face_distance(object2_face_encodings, object1_face_encodings)
    matching_percentage =((1-faces_distance)*100) #If the percentage is >= 40%, then the images are similar.
    round_off = np.round(matching_percentage, decimals=2)
    round_off_percentage = round_off[0]
    print(f"Similarty percentage = {round_off_percentage}%")
    print()
    print("\033[1mNote: If the percentage is >= 40%, then the images are similar.")
    print()


# prints percentage of similarity if the images are of different person.
def false_images(object2_face_encodings, object1_face_encodings):
    print()
    print("The images are of two different person.")
    faces_distance = face_recognition.face_distance(object2_face_encodings, object1_face_encodings)
    matching_percentage =((1-faces_distance)*100) #If the percentage is >= 40%, then the images are similar.
    round_off_percentage = round(matching_percentage[0],2)
    print(f"Similarity percentage = {round_off_percentage}%")
    print()


# Enter desired the name of the eys form the list. 
print()
key1_name = input("Enter key name for image - 1 : ") #Calling function that scans the image from the key.
object1_face_encodings = face_encodings(bucket_name, key1_name)
key2_name = input("Enter key name for image - 2 : ")
object2_face_encodings = np.array(face_encodings(bucket_name, key2_name))#Calling function that scans the image from the key.


matches = face_recognition.compare_faces(object1_face_encodings, object2_face_encodings, tolerance= 0.6)
if True in matches :
    true_images(object2_face_encodings, object1_face_encodings)
else:
    false_images(object2_face_encodings, object1_face_encodings)