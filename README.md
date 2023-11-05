# Embolus-detection
This project cunstruct a nervours network based on F-RCNN model which can identify cancer embolus from pathological slides.
# Environment
CUDA 10.1  
python 3.7.16

# How to use it
```
    python .\detect.py
        --save_folder=='default: save_folder' 
        --slide_path=='slide_path' 
        --model_path=='default: models/FPNrcnn.pth'
```
检测结果存放在save_folder中.
