# Embolus-detection
This project cunstruct a nervours network based on F-RCNN model which can identify cancer embolus from pathological slides.
# Environment
CUDA 10.1  
python 3.7.16

# How to use it
First you need to download the model from the following url.  
  Baidu: https://pan.baidu.com/s/1flw3bzz2fuRtkyE9oMfZDA?pwd=5ndv  
Then you can execute the program with the following instruction.
```
    python .\detect.py
        --save_folder=='default: save_folder' 
        --slide_path=='slide_path' 
        --model_path=='default: models/FPNrcnn.pth'
```
You will see results in 'save_folder'.
