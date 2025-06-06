import cv2
import numpy as np
from image_utils import resize_with_aspect_ratio, display_resized, preprocess_image, detect_edges
from contour_detection import find_contours, filter_document_contours, find_rectangle_contour, visualize_contours
from perspective_transform import four_point_transform, visualize_detection

class DocumentScanner:
    def __init__(self, image_path, target_width=1600):
        self.image_path = image_path
        self.target_width = target_width
        self.original_image = None
        self.processed_image = None
        
    def load_image(self):
        """이미지 로드 및 크기 조절"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {self.image_path}")
        
        # 이미지 크기 조절
        self.original_image = resize_with_aspect_ratio(self.original_image, self.target_width)
        print(f"이미지 로드 완료: {self.original_image.shape}")
        return self.original_image
    
    def process_image(self, show_steps=True):
        """이미지 전처리"""
        if self.original_image is None:
            raise ValueError("먼저 이미지를 로드해주세요.")
        
        # 전처리
        gray, binary = preprocess_image(self.original_image)
        
        if show_steps:
            display_resized('original', self.original_image)
            cv2.waitKey(0)
            display_resized('binary', binary)
            cv2.waitKey(0)
        
        # 에지 검출
        edged = detect_edges(binary)
        
        if show_steps:
            display_resized('edges', edged)
            cv2.waitKey(0)
        
        return edged
    
    def detect_document(self, edged, show_steps=True):
        """문서 영역 검출"""
        # 외곽선 찾기
        contours = find_contours(edged)
        print(f"검출된 외곽선 개수: {len(contours)}")
        
        # 문서 외곽선 필터링
        valid_contours = filter_document_contours(contours, self.original_image.shape)
        
        if show_steps:
            contour_img = visualize_contours(self.original_image, contours)
            display_resized('contours', contour_img)
            cv2.waitKey(0)
        
        # 사각형 찾기
        vertices = find_rectangle_contour(valid_contours)
        
        if vertices is None:
            raise RuntimeError("사각형 문서를 찾을 수 없습니다!")
        
        return vertices
    
    def transform_document(self, vertices, show_steps=True):
        """문서 투시 변환"""
        # 꼭짓점 시각화
        pts = vertices.reshape(4, 2)
        detection_img, ordered_pts = visualize_detection(self.original_image, pts)
        
        if show_steps:
            merged = np.hstack((self.original_image, detection_img))
            display_resized('detected', merged)
            cv2.waitKey(0)
        
        # 투시 변환 적용
        result = four_point_transform(self.original_image, ordered_pts)
        
        if show_steps:
            display_resized('result', result)
            cv2.waitKey(0)
        
        self.processed_image = result
        return result
    
    def save_result(self, output_path='scanned_result.png'):
        """결과 저장"""
        if self.processed_image is None:
            raise ValueError("처리된 이미지가 없습니다.")
        
        cv2.imwrite(output_path, self.processed_image)
        print(f"스캔 결과가 '{output_path}'로 저장되었습니다!")
    
    def scan_document(self, show_steps=True, output_path='scanned_result.png'):
        """전체 스캔 프로세스 실행"""
        try:
            print("문서 스캔을 시작합니다")
            
            # 이미지 로드
            self.load_image()
            
            # 이미지 전처리
            edged = self.process_image(show_steps)
            
            # 문서 영역 검출
            vertices = self.detect_document(edged, show_steps)
            
            # 투시 변환
            result = self.transform_document(vertices, show_steps)
            
            # 결과 저장
            self.save_result(output_path)
            
            print("문서 스캔이 완료되었습니다!")
            return result
            
        except Exception as e:
            print(f"스캔 중 오류 발생: {e}")
            print("다음을 시도해보세요:")
            print("1. 조명 개선")
            print("2. 배경과 문서의 대비 향상")
            print("3. 다른 각도에서 촬영")
            return None
        
        finally:
            cv2.destroyAllWindows()
