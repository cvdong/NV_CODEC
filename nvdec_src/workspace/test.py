'''
Version: v1.0
Author: 東DONG
Mail: cv_yang@126.com
Date: 2023-02-28 09:39:14
LastEditTime: 2023-03-09 01:17:52
FilePath: /nvdec/workspace/test.py
Description: 编解码封装py
Copyright (c) 2023 by ${東}, All Rights Reserved. 
'''

import cv2
import time


def nv_decode():
    
    import libffhdd as ffhdd
    
    start_time = time.time()

    demuxer = ffhdd.FFmpegDemuxer("tt.mp4")
    output_type = ffhdd.FrameType.BGR
    decoder = ffhdd.CUVIDDecoder(bUseDeviceFrame=True, codec=demuxer.get_video_codec(), max_cache=-1, gpu_id=0, output_type=output_type)
    pps, size = demuxer.get_extra_data()

    decoder.decode(pps, size, 0)

    nframe = 0
    

    while True:
        pdata, pbytes, time_pts, iskey, ok = demuxer.demux()
        nframe_decoded = decoder.decode(pdata, pbytes, time_pts)
        for i in range(nframe_decoded):
            ptr, pts, idx, image = decoder.get_frame(return_numpy=True)
            nframe += 1
            # cv2.imwrite(f"nv_decode/data_{nframe:05d}.jpg", image) 
       
        if pbytes <= 0:
            break
    print(f"-----------Total nv decode Frame: {nframe}----------------")
    
    return time.time() - start_time


def soft_decode():
    
    start_time = time.time()
    cap = cv2.VideoCapture("tt.mp4")
    nframe = 0
    while (cap.isOpened()):
        _, image = cap.read()
        if not _:
            break
        nframe += 1
        # cv2.imwrite(f"soft_decode/data_{nframe:05d}.jpg", image)     
    
    print(f"-----------Total soft decode Frame: {nframe}--------------")  
    cap.release()
    
    return time.time() - start_time


if __name__=='__main__':
    
    nv_time = nv_decode()
    soft_time = soft_decode()
    
    print(f"-----------nv_time:    {round(nv_time, 3)}s------------------------")
    print(f"-----------soft_time:  {round(soft_time, 3)}s------------------------")
    print(f"-----------硬解码速度软解码的:  {round(soft_time/nv_time, 3)}倍---------------")