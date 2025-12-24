import cv2
from ultralytics import YOLO

model=YOLO("yolov8n.pt")

IMG=cv2.imread("fruits.jpeg")

result=model(IMG)

annonated_img=result[0].plot()

cv2.imshow("annonated_img",annonated_img)

cv2.imwrite("After_perform_yolo.jpeg",annonated_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 