import cv2
import config
import supervision as sv
from ultralytics import YOLO

model = YOLO(f"yolov8{config.MODEL_TYPE}.pt")
people_id = 0
escape_key = 27

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


def process_frame(frame):
    results = model(frame, verbose=config.VERBOSE)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.class_id == people_id) & (detections.confidence > config.CONFIDENCE_THRESHOLD)]

    labels = [
        f"#{model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id
        in zip(detections.confidence, detections.class_id)
    ]

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    if labels:
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

    cv2.imshow("YOLOv8 AI Detection", annotated_frame)

    return detections


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    while True:
        ret, frame = cap.read()

        detections = process_frame(frame)

        if len(detections) > 0:
            print("Mensen gevonden")
            pass
        else:
            print("Geen mensen")

        if cv2.waitKey(30) == escape_key:
            break


if __name__ == '__main__':
    main()
