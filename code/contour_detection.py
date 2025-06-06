import cv2
import numpy as np

def find_contours(edged):
    """외곽선 찾기"""
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def filter_document_contours(cnts, img_shape, min_area_ratio=0.05):
    """문서 외곽선 필터링"""
    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * min_area_ratio
    
    # 면적순으로 정렬
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    valid_contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            valid_contours.append(c)
    
    return valid_contours

def find_rectangle_contour(contours):
    """사각형 외곽선 찾기"""
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        
        print(f"외곽선 {i}: 면적={area:.0f}, 꼭짓점={len(approx)}")
        
        if len(approx) == 4:
            print("사각형 문서 영역을 찾았습니다")
            return approx
    
    return None

def visualize_contours(image, contours):
    """외곽선 시각화"""
    draw = image.copy()
    cv2.drawContours(draw, contours, -1, (0, 255, 0), 2)
    return draw
