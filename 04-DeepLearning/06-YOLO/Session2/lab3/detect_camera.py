from ultralytics import YOLO
import cv2

# load trained model
model = YOLO("best.pt")
# model.predict(source=0, show=True, conf=0.15)
# open laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection
    results = model(frame)

    # draw results
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()