import cv2, time

cap = cv2.VideoCapture("http://192.168.0.115:8080", cv2.CAP_FFMPEG)
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("FPS test", frame)
    if cv2.waitKey(1) == 27:
        break
