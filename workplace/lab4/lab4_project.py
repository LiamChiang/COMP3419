import numpy as np
import cv2

cap = cv2.VideoCapture('monkey.avi')
background_capture = cv2.VideoCapture('Quadrangle.mov')

ret, frame = cap.read()
ret2, frame2 = background_capture.read()
count = 0
while(ret and ret2):
    frame2 = cv2.resize(frame2, (frame.shape[1], frame.shape[0]))
    for x in range(frame.shape[1]):
        for y in range(frame.shape[0]):
            if frame[y][x][0] > 120:
                frame[y][x][0] = frame2[y][x][0]
                frame[y][x][1] = frame2[y][x][1]
                frame[y][x][2] = frame2[y][x][2]
    if not ret:
        print('Video Reach End')
        break
    cv2.imshow('frame', frame)
    ret, frame = cap.read()
    ret2, frame2 = background_capture.read()
    name = ("%d.png" % (count))
    cv2.imwrite(name, frame)
    count += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()