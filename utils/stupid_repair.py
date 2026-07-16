import cv2 

path = "data/input/original.jpeg"
img = cv2.imread(path)

img= cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("data/input/original_2.jpeg", img)