import cv2
import numpy as np

def resize_with_aspect_ratio(image, width=1600):
    """비율을 유지하면서 이미지 크기 조절"""
    h, w = image.shape[:2]
    aspect_ratio = width / w
    new_height = int(h * aspect_ratio)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_CUBIC)

def display_resized(win_name, image, max_width=700):
    """이미지를 화면에 맞게 리사이즈하여 보여줌"""
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_size = (max_width, int(h * ratio))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized = image
    cv2.imshow(win_name, resized)




def preprocess_image(image):
    """이미지 전처리"""
    # Gray 스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응적 히스토그램 평활화 (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 블러 처리
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # OTSU 방식의 전역 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return gray, binary

def detect_edges(binary):
    """에지 검출"""
    
    # Canny 에지 검출
    edged = cv2.Canny(binary, 30, 100)
    
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) # 모폴로지 닫힘 연산
    
    return edged
