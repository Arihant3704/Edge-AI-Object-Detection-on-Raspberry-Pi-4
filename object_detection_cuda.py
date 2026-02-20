import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

thres = 0.5  # Threshold to detect object

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn.DetectionModel(weightsPath, configPath)
net.setInputSize(100, 100)
net.setInputScale(1.0 / 255.0)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define target labels

target_labels = ["person", "dog", "cat"]

# Initialize FPS calculation variables
prev_time = 0

while True:
    suc, img = cap.read()
    if not suc:
        print("Failed to capture image. Exiting...")
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Detect objects
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classNames[classId - 1]
            if label in target_labels:
                # Draw bounding box and labels
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, label.upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Print detected label
                print(f"Detected: {label} with confidence: {confidence * 100:.2f}%")

                # Alert for car
                if label == target_labels:
                    print("ALERT:detected under car!")
                    cv2.putText(img, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display FPS on the frame
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Output", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

