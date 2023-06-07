'''
Version: v1.0
Author: 東DONG
Mail: cv_yang@126.com
Date: 2023-03-28 09:48:54
LastEditTime: 2023-06-07 14:28:59
FilePath: /pytrt/Process_cpu_decode.py
Description: 
Copyright (c) 2023 by ${東}, All Rights Reserved. 
'''
# 多进程 infer
import cv2
from src.trt_infer import TRT_inference
from src.draw_result import Colors, DrawResults
import pycuda.autoinit 
import pycuda.driver as cuda 
import multiprocessing as mp
import time


class trtMprocess():
    
    def __init__(self, camera_ip_l):
        
        self.camera_ip_l = camera_ip_l
        self.classes_name = list(map(lambda x:x.strip(), open('./workspace/coco.names', 'r').readlines()))
        self.color = Colors()
        
        self.engine_path = "./workspace/yolov8s_fp16.engine"
       
        
    def image_put(self, q, camera_ip):
        cap = cv2.VideoCapture(camera_ip)
    
        while True:
            decode_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
        
            q.put(frame)
                    
            if q.qsize() > 1:
                q.get()
                    
            print(f'MProcess -{camera_ip}- Decode time: {round((time.time() - decode_time) * 1000, 2) }ms \n')  
                   
        cap.release()
        
    def image_get(self, q, window_name):
    
        inference = TRT_inference(self.engine_path)
        
        while True:
            frame = q.get()
            
            infer_time = time.time()
            outputs = inference(frame)
            print(f'MProcess -{window_name}- Detection spend time: {round((time.time() - infer_time) * 1000, 2) }ms')
            
            draw_fun = DrawResults(outputs, self.classes_name, self.color)
            draw_det = draw_fun.draw_det(frame)
    
            cv2.imshow(window_name, draw_det)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
            
            
    def run_multi_camera(self):
        
        mp.set_start_method(method='spawn')  # init
        queues = [mp.Queue(maxsize=4) for _ in self.camera_ip_l]
        processes = []
        
        for queue, camera_ip in zip(queues, self.camera_ip_l):
            
            processes.append(mp.Process(target=self.image_put, args=(queue, camera_ip,)))
            processes.append(mp.Process(target=self.image_get, args=(queue, camera_ip,)))

        for process in processes:
            process.daemon = True
            process.start()
            
        for process in processes:
            process.join()
  
if __name__=='__main__':
    
    camera_ip_l = [
            
        # "workspace/test.mp4",
        
        ]
    
    mp_run = trtMprocess(camera_ip_l)
    
    mp_run.run_multi_camera()