#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import *
import shlex
import subprocess




def object_counting_webcam(video,detection_graph, category_index, is_color_recognition_enabled,targeted_object):

   
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            cap = cv2.VideoCapture(video)
            (ret, frame) = cap.read()

            # for all the frames that are extracted from input video
            while True:
                # Capture frame-by-frame
                (ret, frame) = cap.read()          

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                

                # Visualization of the results of a detection.

                
                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    temp = "0"
                    with open("detection.csv", 'a') as log:
                            
                            
                            log.write("{0},{1}\n".format(datetime.now().strftime("%H:%M:%S"),str(temp)))
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    
                    c=''.join([n for n in the_result if n.isdigit()])
                    
                    print(the_result)
                    temp = c
                    with open("detection.csv", 'a') as log:
                            
                            
                            log.write("{0},{1}\n".format(datetime.now().strftime("%H:%M:%S"),str(temp)))
                    if int(temp) > 5: # sent a trigger to the manager if the people count exceeds 5
                            
                            
                            cmd = '''curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/vnd.pagerduty+json;version=2' --header 'From: <mail id>' --header 'Authorization: Token token=<token id>' -d '{
                          "incident": {
                            "type": "incident",
                            "title": "trigger",
                            "service": {
                              "id": "<service id>",
                              "type": "service_reference"
                            }
                          }
                        }
                        ' 'https://api.pagerduty.com/incidents' '''
                            args = shlex.split(cmd)
                            process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            stdout, stderr = process.communicate()

            
               
                

                                    
                
                cv2.imshow('object counting',input_frame)
                
                

            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            
            cap.release()
            cv2.destroyAllWindows()


