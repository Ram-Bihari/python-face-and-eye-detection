import cv2 as cv

img = cv.imread('image.jpg')

grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


haar_cascade = cv.CascadeClassifier('data\haarcascades_cuda\haarcascade_frontalface_default.xml')
eye_cascade =  cv.CascadeClassifier('data\haarcascades_cuda\haarcascade_eye.xml')

rectangle = haar_cascade.detectMultiScale(grayImg, 1.1, 9)
# rectangle2 = eye_cascade.detectMultiScale(grayImg, 1.1, 9)

# print(rectangle2)

print(rectangle)
for (x,y,w,h) in rectangle:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv.imshow('picture' ,img)

cv.waitKey(0)


