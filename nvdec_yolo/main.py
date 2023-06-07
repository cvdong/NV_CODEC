import cv2
from src.trt_infer import TRT_inference
from src.draw_result import Colors, DrawResults

import pycuda.autoinit 
import pycuda.driver as cuda 

import time 

# IMAGE 推理
def image_inference():
    
    classes_name = list(map(lambda x:x.strip(), open('./workspace/coco.names', 'r').readlines()))
    color = Colors()
    engine_path = "./workspace/yolov5s_fp16.engine"
    inference = TRT_inference(engine_path)

    img = cv2.imread("./workspace/bus.jpg")
    
    # 计算时间
    import time
    infer_time = time.time()

    for _ in range(10000):
        det_outputs = inference(img)
        
        print(f'-----------{_}----------')

    print(f'{engine_path.split("/")[-1]} end to end detection (pre+infer+post) spend time: { round((time.time() - infer_time)/10, 3)} ms')

    # 画框渲染
    draw_fun = DrawResults(det_outputs, classes_name, color)
    draw_det = draw_fun.draw_det(img)


    # 显示+存储
    # cv2.imshow("test_show", img) 
    # cv2.waitKey(0)
    cv2.imwrite("workspace/bus_results.jpg", draw_det)
    print(f'{engine_path.split("/")[-1]} faster deploy end to end detection (pre+infer+post) finished! ')
    

# 软解码
def soft_decode():
    
    classes_name = list(map(lambda x:x.strip(), open('./workspace/coco.names', 'r').readlines()))
    color = Colors()
    engine_path = "./workspace/yolov5s_fp16.engine"
    inference = TRT_inference(engine_path)

    img = cv2.imread("./workspace/bus.jpg")
    # 模型预热
    for _ in range(10):
        det_outputs = inference(img)
    
    cap = cv2.VideoCapture("rtsp://admin:dftaike888999@10.10.1.3:554/streaming/channels/101")
    
    while (cap.isOpened()): 
        ret, frame = cap.read()
        if not ret:
            break
        
        import time
        start_infer = time.time()
        det_outputs = inference(frame)
        
        print(f'{engine_path.split("/")[-1]} end to end detection (pre+infer+post) spend time: { round((time.time() - start_infer) *1000, 3)} ms')

        draw_fun = DrawResults(det_outputs, classes_name, color)
        draw_det = draw_fun.draw_det(frame)
        
        cv2.imshow("ts-soft", draw_det)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()
  
  
# 硬解码
def hard_decode():
    
    classes_name = list(map(lambda x:x.strip(), open('./workspace/coco.names', 'r').readlines()))
    color = Colors()
    engine_path = "./workspace/yolov8s_fp16.engine"
   
    import libffhdd as ffhdd
    demuxer = ffhdd.FFmpegDemuxer("rtsp://admin:dftaike888999@10.10.1.3:554/streaming/channels/101")
    output_type = ffhdd.FrameType.BGR
    decoder = ffhdd.CUVIDDecoder(
        bUseDeviceFrame=True,
        codec=demuxer.get_video_codec(),
        max_cache=-1,
        gpu_id=0,
        output_type=output_type
    )
    pps, size = demuxer.get_extra_data()
    decoder.decode(pps, size, 0)
    
    nframe = 0
    
    inference = TRT_inference(engine_path)
    
    img = cv2.imread("./workspace/bus.jpg")
    # 模型预热
    for _ in range(10):
        det_outputs = inference(img)

    while True:
        
        pdata, pbytes, time_pts, iskey, ok = demuxer.demux()
        nframe_decoded = decoder.decode(pdata, pbytes, time_pts)
        
        for i in range(nframe_decoded):
            ptr, pts, idx, frame = decoder.get_frame(return_numpy=True)
            nframe += 1
            
            import time
            start_infer = time.time()
            det_outputs = inference(frame)
            print(f'{engine_path.split("/")[-1]} end to end detection (pre+infer+post) spend time: { round((time.time() - start_infer) *1000, 2)} ms')
            
            draw_fun = DrawResults(det_outputs, classes_name, color)
            draw_det = draw_fun.draw_det(frame)
            cv2.imshow("ts-hard", draw_det)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        if pbytes <= 0:
            break
        
    print(f"Frame. {nframe}")
         
         
if __name__=="__main__":
    
    # image_inference()
    # soft_decode()
    hard_decode()