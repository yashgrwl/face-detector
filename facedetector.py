import cv2

# load pre trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#giving image to a variable
#img=cv2.imread('rdj2.jpg')
webcam = cv2.VideoCapture('video2.mp4')


while True:
    frame_read, frame = webcam.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    face_coordinates= trained_face_data.detectMultiScale(gray_img)
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , 2)
    
    cv2.namedWindow('face detector' , cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('face detector', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('face detector',frame)   

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
'''gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates= trained_face_data.detectMultiScale(gray_img)


#print(face_coordinates)
for (x,y,w,h) in face_coordinates:
   cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2)

cv2.imshow('image',img)   

cv2.waitKey()'''
