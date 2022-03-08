import cv2

im = cv2.imread(r"E:\Computing Project BSc\Blender Files\Rendering Blend\Renders\Colour\Image0115.png")

x = 523
y = 393
w = 579
h = 545




image = cv2.rectangle(im, (x,y),(w,h),(0,255,0),6)
cv2.imshow("Test", image)
cv2.waitKey(0)