import os
import cv2
from base_camera import BaseCamera
from video_util import WebcamVideoStream
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
# from config import headers, body_class_model_pt, face_class_model_pt, body_classes, face_classes
import json
import numpy as np
from numpy import array, exp
import torch
from torchvision import transforms
import requests
from tempfile import NamedTemporaryFile
from ultralytics import YOLO
from datetime import datetime 

FONT_SIZE = 15
FONT = ImageFont.load_default() 

# Load detector models
yolo_model = YOLO('weights/yolov8m.pt')

class Camera(BaseCamera):

    def person_draw(img, bboxes):
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.convert("RGBA")

        points_list = []

        if bboxes.boxes.shape[0] > 0:
            for bbox in bboxes.boxes.xyxy:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                v_offset = int((y2-y1)*0.2)
                h_offset = int((x2-x1)*0.1)

                x1 = max(0,x1-h_offset)
                x2 = min(pil_image.size[0],x2+h_offset)
                y1 = max(0,y1-v_offset)
                y2 = min(pil_image.size[1],y2+v_offset)

                outline_color = (255, 0, 0, 255)  # Red in RGBA format

                # Define the rectangle's fill color (RGBA: 255, 255, 255, 0.25)
                fill_color = (255, 255, 255, int(255 * 0.25))  # Reduces opacity to 25%

                # Create a transparent white background for the rectangle
                background = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(background) 
                draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=5)
                pil_image = Image.alpha_composite(pil_image, background)
                points_list.append([x1,y1,x2,y2])

        return pil_image, points_list

    @staticmethod
    def frames():

        vs = WebcamVideoStream(src='rtmp://media.inference.asia/live/terra1').start()

        line_x1,line_y1,line_x2,line_y2 = 1250,200, 750,1000
        vector_line = np.array([line_x2 - line_x1, line_y2 - line_y1])

        position_dict = {}
        status_dict = {'Enter':0,'Exit':0}

        start_time = time.time()-3
        print('camera')
        
        while True:
            
            img = vs.read()
            # img = cv2.resize(img, (1920,1200))
            try:
                img_fr = img.copy()
            except:
                continue

            cv2.line(img, (1250,200), (750,1080), (0, 255, 255), 4)

            # Extract people 
            people_detected = yolo_model.track(img, verbose=False, classes=[0], conf=0.6, persist=True)

            pil_image, people_boxes = Camera.person_draw(img, people_detected[0])
                
            pil_image = pil_image.convert("RGB")
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            yield cv2.imencode('.jpg', opencv_image)[1].tobytes()
            
            # infer every 3 second
            if time.time() - start_time < 1:
                continue
            else:
                start_time = time.time()

            if people_boxes:                
                # Loop for every person detected
                for i, bbox in enumerate(people_boxes):
                    
                    try:
                        person_id = int(people_detected[0].boxes.id[i])
                    except:
                        continue
                    
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    xc = int((x1+x2)/2)
                    
                    cropped_body = img_fr[y1:y2,x1:x2]     
                    vector_point = np.array([xc - line_x1, y2 - line_y1])
                    cross_product = np.cross(vector_line, vector_point)
                    
                    if cross_product > 0:
                        position = 'left'
                    else:
                        position = 'right'
                        
                    if person_id not in position_dict:
                        position_dict[person_id] = position
                    else:
                        if position_dict[person_id] != position:
                            position_dict[person_id] = position
                            if position == 'right':
                                status_dict['Enter'] += 1
                                status = 'Enter'
                            else:
                                status_dict['Exit'] += 1
                                status = 'Exit'

                            retval, buffer = cv2.imencode('.jpg', cropped_body)
                            base64String = base64.b64encode(buffer).decode('utf-8')
                            data = {
                                "image": base64String, #base64 string of one cropped person
                                "xmin": x1,
                                "ymin": y1,
                                "xmax": x2,
                                "ymax": y2,
                                "name": status
                            }

                            requests.post('http://localhost:8418',json = data)





           
            

            

            

                                  


        