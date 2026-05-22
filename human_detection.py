import cv2
import sys

# trying raw v4l2 because argus will definitely crash on the v1 cam
# honestly this probably wont even find /dev/video0 without a kernel patch

pipe = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink"
cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("yeah dev/video0 doesnt exist, told u the driver is missing")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame drop")
        break
        
    cv2.imshow("v1_test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()