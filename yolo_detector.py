# working
import cv2
import numpy as np
from processing import Processing

class Detector():
    """
    YOLO detector only for people
    """
    def __init__(self, path: str, wieghts: str = "yolov3.weights", cfg: str = "yolov3.cfg", names: str = "coco.names"):
        self.weights = wieghts
        self.conf = cfg
        self.names = names
        self.path = path
        self.Processing = Processing(path)

    # Load yolo
    def load_yolo(self):
        self.net = cv2.dnn.readNet(self.weights, self.conf)
        with open(self.names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        self.layers_names = self.net.getLayerNames()
        self.output_layers = [self.layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # return net, classes, colors, output_layers

    def display_blob(self):
        """
            Three images each for RED, GREEN, BLUE channel
        """
        for b in self.blob:
            for n, imgb in enumerate(b):
                cv2.imshow(str(n), imgb)

    def detect_objects(self, img):
        self.blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(320, 320), swapRB=True, crop=False)
        self.net.setInput(self.blob)
        self.outputs = self.net.forward(self.output_layers)
        # return blob, outputs

    def get_box_dimensions(self, height, width):
        # TODO: change in class type
        boxes = []
        confs = []
        class_ids = []
        for output in self.outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                if class_id == 0:
                    conf = scores[class_id]
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confs.append(float(conf))
                        class_ids.append(class_id)
        return boxes, confs, class_ids

    def draw_labels(self, img, boxes, confs, class_ids):
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.names[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
        self.Processing.write_frame(img)


    def start_video(self):
        self.load_yolo()
        self.Processing.open_video()
        self.Processing.open_writer('output.avi')
        while True:
            frame = self.Processing.get_frame()
            if frame is None:
                break
            height, width, channels = frame.shape
            self.detect_objects(frame)
            boxes, confs, class_ids = self.get_box_dimensions(height, width)
            self.draw_labels(frame, boxes, confs, class_ids)
            if cv2.waitKey(1) == 27:
                break
        self.Processing.close_video()


if __name__ == '__main__':
    path = 'pedestrians.mp4'
    check = Detector(path)
    check.start_video()

    cv2.destroyAllWindows()
