#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th July 2019
#----------------------------------------------

# Imports
import gi
gi.require_version('Gtk', '2.0')
from api import object_counting_api
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import *

def main():
    
    
    # Object detection imports
    from utils import backbone
    from api import object_counting_api

    # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')
    

    targeted_objects = "person" # (for counting targeted objects) change it with your targeted objects
    is_color_recognition_enabled = 0
    video = "restaurent.mp4"
  

    object_counting_api.object_counting_webcam(video,detection_graph, category_index, is_color_recognition_enabled,targeted_objects)

def graph():
    
    dateconv = lambda s: datetime.strptime(s, "%H:%M:%S")
    col_names = ["T", "V"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
            
            

            with open('detection.csv','rb') as f:
                    lines = f.readlines()

            mydata = np.genfromtxt(lines[-24:], delimiter=',', names=col_names, dtype=[('T', 'O'), ('V', 'u1')], converters={"Time": dateconv})
            ax.clear()
            ax.bar(mydata['T'], mydata['V'],width=0.8,color="red")

            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.title('People count')
            plt.ylabel('Number')
            plt.xlabel('Time')
    ani = animation.FuncAnimation(fig, animate, frames=100, interval=10, blit= False)

     
      
    # show the plot 
    plt.show() 
if __name__ == "__main__": 
    # creating processes 
    p1 = multiprocessing.Process(target=main) 
    p2 = multiprocessing.Process(target=graph) 
  
    # starting process 1 
    p1.start() 
    # starting process 2 
    p2.start() 
  
    # wait until process 1 is finished 
    
    # wait until process 1 is finished 
    p1.join() 
    # wait until process 2 is finished 
    p2.join() 
  
    # both processes finished 
    print("Done!") 
    
