#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import cv2
import darknet
import random
import time

CONFIG = "/home/vtol/catkin_ws/src/vision/scripts/file/yolov4-tiny-tubitak.cfg"
DATA_FILE = "/home/vtol/catkin_ws/src/vision/scripts/file/obj.data"
EXT_OUTPUT = "store_true"
THRESHOLD_DETECTION = 0.5
WEIGHT = "/home/vtol/catkin_ws/src/vision/scripts/file/final.weights"

vid = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")

network, class_names, class_colors = darknet.load_network(
            CONFIG,
            DATA_FILE,
            WEIGHT,
            batch_size=1
        )

darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)

pub = rospy.Publisher('vision', Int32MultiArray, queue_size=1)
rospy.init_node('vision', anonymous=True)

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def drawing(frame, detections, fps):
    random.seed(3)  # deterministic bbox colors
    detections_adjusted = []
    if frame is not None:
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        cv2.imshow('Inference', image)
        # cv2.waitKey(1)

def publish(detections):
    status = 0
    x = 0
    y = 0
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        status = 1
    if (status == 0):
        data = [int(status),int(x),int(y)]
        data_to_send = Int32MultiArray()
        data_to_send.data = data
    else :
        data = [int(status),int(x-208),int(y-208)]
        data_to_send = Int32MultiArray()
        data_to_send.data = data

    pub.publish(data_to_send)    
        

# count = 0
while (True) :
    # count+=1
    prev_time = time.time()
    ret, frame = vid.read()

    if frame is None :
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)

    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    detections = darknet.detect_image(network, class_names, img_for_detect,
                                        THRESHOLD_DETECTION)
        # detections_queue.put(detections)

    fps = int(1/(time.time() - prev_time))

    # print("FPS: {}".format(fps))

    # darknet.print_detections(detections, EXT_OUTPUT)
    darknet.free_image(img_for_detect)

    publish(detections)

    # uncomment to display result
    # drawing(frame,detections,fps)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
  