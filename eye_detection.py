import cv2 as cv

haar_cascade = cv.CascadeClassifier('data\haarcascades_cuda\haarcascade_frontalface_default.xml')
eye_cascade =  cv.CascadeClassifier('data\haarcascades_cuda\haarcascade_eye.xml')

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    # if not img: break

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    FaceRectangle = haar_cascade.detectMultiScale(gray_img, 1.1, 9)


    # print(rectangle)
    for (x,y,w,h) in FaceRectangle:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        all_gray_faces = gray_img[y:y+h, x:x+w]
        all_real_faces = img[y:y+h, x:x+w]
        eye = eye_cascade.detectMultiScale(all_gray_faces)
        for (a,b,c,d) in eye:
            cv.rectangle(all_real_faces, (a,b), (a+c, b+d), (0, 255, 255), 1)
        # print(eye)

    cv.imshow('My Camera', img)

    if cv.waitKey(1) == ord('q'):
        break


cap.release()
cv.destroyAllWindows()