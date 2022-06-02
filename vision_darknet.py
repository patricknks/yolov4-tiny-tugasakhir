#!/usr/bin/env python3
from numpy import inner, outer
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import cv2
import darknet
import random
import time
 
CONFIG = "/home/moon/pat/src/yolo_vision/data/yolov4-tiny-custom.cfg"
DATA_FILE = "/home/moon/pat/src/yolo_vision/data/obj.data"
EXT_OUTPUT = "store_true"
THRESHOLD_DETECTION = 0.5
WEIGHT = "/home/moon/pat/src/yolo_vision/weights/yolov4-tiny-custom_final.weights"
INNER = "inner"
MIDDLE = "middle"
OUTER = "outer"
 
# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1280,
#     capture_height=720,
#     display_width=960,
#     display_height=540,
#     framerate=30,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc sensor-id=%d !"
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             sensor_id,
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )
 
# vid = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
 
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
network, class_names, class_colors = darknet.load_network(
            CONFIG,
            DATA_FILE,
            WEIGHT,
            batch_size=1
        )
 
darknet_width = darknet.network_width(network)
darknet_height = darknet.network_height(network)
 
cam_bool_inner_pub = rospy.Publisher("camera/bool/inner", String, queue_size=1)
cam_bool_middle_pub = rospy.Publisher("camera/bool/middle", String, queue_size=1)
cam_bool_outer_pub = rospy.Publisher("camera/bool/outer", String, queue_size=1)
 
cam_val_inner_pub = rospy.Publisher("camera/data/inner", PointStamped, queue_size=1)
cam_val_middle_pub = rospy.Publisher("camera/data/middle", PointStamped, queue_size=1)
cam_val_outer_pub = rospy.Publisher("camera/data/outer", PointStamped, queue_size=1)
 
rospy.init_node("color_detection")
rate = rospy.Rate(16)
 
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
    inner_detected = "false"
    middle_detected = "false"
    outer_detected = "false"
 
    inner_data = PointStamped()
    inner_data.header.stamp = rospy.Time.now()
    inner_data.header.frame_id = "map"
 
    middle_data = PointStamped()
    middle_data.header.stamp = rospy.Time.now()
    middle_data.header.frame_id = "map"
 
    outer_data = PointStamped()
    outer_data.header.stamp = rospy.Time.now()
    outer_data.header.frame_id = "map"
 
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
 
        if(label == INNER):
            inner_detected = "true"
            inner_data.point.x = y - 208  
            inner_data.point.y = x - 208
 
        elif (label == MIDDLE):
            middle_detected = "true"
            middle_data.point.x = y - 208
            middle_data.point.y = x - 208
 
        elif (label == OUTER):
            outer_detected = "true"
            outer_data.point.x = y - 208
            outer_data.point.y = x - 208
 
    cam_val_inner_pub.publish(inner_data)    
    cam_val_middle_pub.publish(middle_data)
    cam_val_outer_pub.publish(outer_data)
 
    cam_bool_inner_pub.publish(inner_detected)
    cam_bool_middle_pub.publish(middle_detected)
    cam_bool_outer_pub.publish(outer_detected)
 
 
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
    drawing(frame,detections,fps)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
 
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
