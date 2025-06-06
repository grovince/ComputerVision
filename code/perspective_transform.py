import cv2
import numpy as np

def order_points(pts):
    """개선된 꼭짓점 정렬 함수"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # x+y 합으로 좌상, 우하 결정
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # x-y 차이로 우상, 좌하 결정
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect.astype(np.float32)

def four_point_transform(image, pts):
    """화질 개선된 투시 변환 함수"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 너비 계산
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    # 높이 계산
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # 목적지 좌표
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]
    ], dtype="float32")
    
    # 보간법 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                               flags=cv2.INTER_LANCZOS4)
    
    # 화질 개선 샤프닝
    warped = cv2.detailEnhance(warped, sigma_s=10, sigma_r=0.15)
    
    return warped

def visualize_detection(image, points):
    """개선된 시각화 함수"""
    ordered_pts = order_points(points)
    draw = image.copy()
    
    # 윤곽선 표시
    cv2.polylines(draw, [ordered_pts.astype(int)], True, (0, 0, 255), 5)
    
    # 꼭짓점 표시
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (x, y) in enumerate(ordered_pts.astype(int)):
        cv2.circle(draw, (x, y), 15, (255, 0, 0), -1)
        cv2.putText(draw, f"{labels[i]} ({x},{y})", (x+20, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return draw, ordered_pts
