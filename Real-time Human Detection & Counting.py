# Import Libraries
import cv2    # A strong library used for machine learning
import numpy as np # Used for Scientific Computing. Image is stored in a numpy array
import imutils # To Image Processing
import argparse #Used to give input in command line


#Create a model which will detect Humans
#We will use HOGDescriptor with SVM already implemented in OpenCV
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect() method
#It will take a frame to detect a person in it. Make a box around a person and show the frame.
#and return the frame with person bounded by a green box.
#detectMultiScale(). It returns 2-tuple
#Coordinates of bounding Box of person
#Confidence Value that it is a person
def detect(frame):
    bounding_box , weights = HOG.detectMultiScale(frame, winStride = (4,4), padding = (8,8), scale = 1.03)
    #to build counter
    person = 1
    for x,y,w,h in bounding_box :
        cv2.rectangle(frame , (x, y ), (x+w , y+h), (0,255, 0), 3)
        cv2.putText(frame , f'person{person}', (x, y),cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255, 0, 0) , 1)
        # for updates
        person += 1
        
        cv2.putText(frame , 'Satus : Detecting ', (40,40), cv2.FONT_HERSHEY_SIMPLEX , 0.8 , (0,0,255), 2 )
        cv2.putText(frame, f'Total Persons : {person-1}' , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,255), 2)
        cv2.imshow('Output', frame)
        
        return frame
    
    
    
#HumanDetector() method
#There are two ways of getting Video: 1.Web Camera 2.Path of file stored  
# In this project,we can take images also. So our method will check 
#if a path is given then search for the video or image in the given path and operate. 
#Otherwise, it will open the webCam.
         
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : 
        camera = True 
    else :
        camera = False

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(ouput_path,writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])
  
#DetectByCamera() method

def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    
    
#DetectByPathVideo() method
#This method is very similar to the previous method except we will give a path to the Video.
# First, we check if the video on the provided path is found or not. 

def detectByPathVideo(path, writer):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

#detectByCamera
def detectByCamera(writer):   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()   
    
    
#DetectByPathimage() method
def detectByPathImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width = min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 
# Argparse() method
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=True, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args    


#Main function

if __name__ == "__main__":
    HOG = cv2.HOGDescriptor()
    HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
    
    #detectByPathImage("img2.jpg","testimg.jpg")
    