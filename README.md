# People-counting-and-alert-system-using-TensorFlow-and-PagerDuty
# Demo video

![Alt text](https://img.youtube.com/vi/jQlTwzy6tN0/0.jpg)](https://www.youtube.com/watch?v=jQlTwzy6tN0&feature=emb_logo)
# Working using Nvidia Jetson Nano
![alt text](https://www.waveshare.com/img/devkit/accBoard/Fan-4010-12V/Fan-4010-12V-3_800.jpg)

![alt text](https://github.com/kishorkuttan/People-counting-and-alert-system-using-TensorFlow-and-PagerDuty-/blob/master/snapshot.png?raw=true)
### People counting and real-time animated graph using TensorFlow object counting API

* The counting uses upgraded Tensorflow object counting API and the counted graph is plotted using matplotlib animation
* When the count exceeds a threshold(say people count >5) it will trigger our incident
* The triggering script is on api/object_counting_api.py

![alt text](https://github.com/kishorkuttan/People-counting-and-alert-system-using-TensorFlow-and-PagerDuty-/blob/master/android_1.jpg?raw=true)

![alt text](https://github.com/kishorkuttan/People-counting-and-alert-system-using-TensorFlow-and-PagerDuty-/blob/master/android_2.jpg?raw=true)

### Alert
# Run
* If you are already installed Anaconda python you can skip this step
https://www.anaconda.com/distribution/

* conda create -n yolo pip python=3.7
* conda activate yolo
* pip install -r requirements.txt
* python main.py

# Nvidia Jetson Nano features
## GPU: 128-core NVIDIA Maxwell™ architecture-based GPU
## CPU: Quad-core ARM® A57
## Video: 4K @ 30 fps (H.264/H.265) / 4K @ 60 fps (H.264/H.265) encode and decode
## Camera: MIPI CSI-2 DPHY lanes, 12x (Module) and 1x (Developer Kit)
## Memory: 4 GB 64-bit LPDDR4; 25.6 gigabytes/second
## Connectivity: Gigabit Ethernet
## OS Support: Linux for Tegra®
## Module Size: 70mm x 45mm
## Developer Kit Size: 100mm x 80mm
# Reference
https://github.com/ahmetozlu/tensorflow_object_counting_api


