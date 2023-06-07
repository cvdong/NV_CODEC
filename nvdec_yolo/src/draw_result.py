# 结果渲染

import cv2

# 颜色选择
class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
   
# 画框
class DrawResults():
         
    def __init__(self, det_results, classes_name, color):
        
        self.det_results = det_results
        self.classes_name = classes_name
        self.color = color
        

    def draw_det(self, image):
             
        for obj in self.det_results:
            left, top, right, bottom = map(int, obj[:4])
            confidence = obj[4]
            label = self.classes_name[int(obj[5])]
        
            label_size = cv2.getTextSize(label + '00000', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (left, top-label_size[1]-3), (left + label_size[0], top-3), self.color(int(obj[6])), -1)
            
            cv2.rectangle(image, (left, top), (right, bottom), self.color(int(obj[5]), True), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (left, top-3), 0, 0.5, (0, 0, 0), 2, 8)
            
            cv2.line(image, (left, top), (left+16, top), self.color(int(obj[5])+6,True), 2)
            cv2.line(image, (left, top), (left, top+16), self.color(int(obj[5])+6,True), 2)
            
            cv2.line(image, (right, top), (right-16, top), self.color(int(obj[5])+6,True), 2)
            cv2.line(image, (right, top), (right, top+16), self.color(int(obj[5])+6,True), 2)
            
            cv2.line(image, (left, bottom), (left+16, bottom), self.color(int(obj[5])+6,True), 2)
            cv2.line(image, (left, bottom), (left, bottom-16), self.color(int(obj[5])+6,True), 2)
            
            cv2.line(image, (right, bottom), (right-16, bottom), self.color(int(obj[5])+6,True), 2)
            cv2.line(image, (right, bottom), (right, bottom-16), self.color(int(obj[5])+6,True), 2)
            
        return image