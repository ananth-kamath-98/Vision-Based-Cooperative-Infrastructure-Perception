# Vision-Based-CIP

## Dataset
This project uses the LUMPI dataset to validate the system against its 
multi-view time synchronized videos of an intersection. 

LUMPI: https://data.uni-hannover.de/gl/dataset/lumpi <br>
The dataset can be downloaded here: https://data.uni-hannover.de:8080/dataset/upload/users/ikg/busch/LUMPI/Measurement6/cam.zip.001

The following directions assume that the repository has been cloned and the 
environment has been set up and activated.

### Setting up the dataset for processing

1. **Extract the videos** <br>
   Extract the files (using 7-zip) of the dataset to the directory 
   "/dataset/". Our solution will be tested on 10 second clips of each 
   perspective. The assumption is that if it works on the 10-second clips, 
   it should work relatively well on the full-length video (if not 
   real-time). To extract 10-second clips, run all the cells in the file -
   "dataset_setup/video_clippings.ipynb". This should generate new video 
   clips under the directory - "/dataset/video_clips/"
2. **Extracting Frames (Optional)** <br>
    This step is necessary if you require individual clips from the videos 
   that we have in our dataset. This script is here to aid in creating the 
   custom dataset for generating ground-truths. These ground-truths would 
   be used for evaluating the system and also to aid in transfer-learning 
   (if required). Run all the cells in the file: 
   "dataset_setup/extracting_frames.ipynb".
3. **Generating Bounding Boxes using object detector** <br>
   Currently, YOLOv9 is used to generate bounding boxes of all the vehicles 
   in each scene. We are using the model and weights shipped by ultralytics 
   to perform object detection. Download the pretrained weights from the 
   [ultraanalytics](https://shorturl.at/tKq1Z) website for the YOLOv9c 
   model. Place these weights in the directory: "object_detection/". To generate bounding boxes, run the scrip in the 
     file - "object_detection/yolo_testing.ipynb". The annotated images can 
     then be found in the directory - "dataset/frames/annotated_frames/".