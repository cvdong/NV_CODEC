# trt_infer 接口

import tensorrt as trt
from src.pre_process import Warpaffine
from src.post_process import gpu_decode
import pycuda.autoinit  
import pycuda.driver as cuda  
import os

class TRT_inference(object):
    def __init__(self, engine_path):
        super(TRT_inference, self).__init__()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        # 读入engine
        if os.path.isfile(engine_path):
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
                
        # engine反序列化
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        
        # 1 3 640 640 
        dst_size = self.context.get_binding_shape(0)[2:]
        # 1 8400 84
        rows, cols = self.context.get_binding_shape(1)[1:]
        
        # v5 v7 float 4字节 1 * 25200 * 85 * 4 
        # v6 v8 float 4字节 1 * 8400 * 84/85 * 4 anchor free
        self.d_output = cuda.mem_alloc(trt.volume(self.context.get_binding_shape(1)) * 4)
        self.stream = cuda.Stream()
        
        # pre_process
        self.warpaffine = Warpaffine(dst_size=dst_size,  stream=self.stream)
        # post_process
        self.postprocess = gpu_decode(rows=rows, cols=cols, stream=self.stream)

    def __call__(self, img):
        # pre
        pdst_device_out, affine = self.warpaffine(img)
        # infer
        self.context.execute_async_v2(bindings=[int(pdst_device_out), int(self.d_output)], stream_handle=self.stream.handle)
        # post
        det_boxs = self.postprocess(self.d_output, affine)
        
        return det_boxs