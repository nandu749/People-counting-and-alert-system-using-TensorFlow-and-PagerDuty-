# People-counting-and-alert-system-using-TensorFlow-and-PagerDuty
# Working using Nvidia Jetson Nano

* The counting uses upgraded Tensorflow object counting API and the counted graph is plotted using matplotlib animation
* When the count exceeds a threshold(say people count >5) it will trigger our incident

# Run
* If you are already installed Anaconda python you can skip this step
https://www.anaconda.com/distribution/

* conda create -n yolo pip python=3.7
* conda activate yolo
* pip install -r requirements.txt
* python main.py

