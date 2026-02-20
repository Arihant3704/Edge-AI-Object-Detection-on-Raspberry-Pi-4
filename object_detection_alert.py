import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

thres = 0.6  # Increased threshold to reduce false detections

# Load class names
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn.DetectionModel(weightsPath, configPath)
net.setInputSize(300, 300)  # More common input size for this model
net.setInputScale(1.0 / 255.0)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

target_labels = {"person", "dog", "cat"}  # Using set for faster lookups

prev_time = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Exiting...")
        break

    curr_time = time.time()
    if curr_time != prev_time:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0
    prev_time = curr_time

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classNames[classId - 1]
            if label in target_labels:
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, f"{label.upper()} {confidence*100:.2f}%", 
                            (box[0] + 10, box[1] + 30), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                print(f"Detected: {label} with confidence: {confidence * 100:.2f}%")

                # Display alert for detections under the car
                cv2.putText(img, "ALERT!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display FPS on the frame
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

