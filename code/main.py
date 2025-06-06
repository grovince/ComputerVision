import sys
import os
import logging
from typing import Optional
from PIL import Image
import pytesseract
from document_scanner import DocumentScanner

# Tesseract 경로
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def validate_image_path(image_path: str) -> bool:
    """이미지 경로 유효성 검사"""
    if not os.path.exists(image_path):
        logging.error(f"파일을 찾을 수 없습니다: {image_path}")
        return False
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        logging.error("지원하지 않는 파일 형식입니다. PNG/JPG 파일만 가능합니다.")
        return False
    return True

def perform_ocr(image_path: str, lang: str = 'kor+eng') -> Optional[str]:
    """OCR 텍스트 추출"""
    try:
        if not validate_image_path(image_path):
            return None

        logging.info(f"OCR 처리 시작: {os.path.basename(image_path)}")
        
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img, lang=lang)
            
            if not text.strip():
                logging.warning("추출된 텍스트가 없습니다.")
                return None

            base_name = os.path.splitext(image_path)[0]
            txt_path = f"{base_name}_ocr.txt"

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            logging.info(f"OCR 텍스트 저장 완료: {txt_path}")
            return txt_path

    except Exception as e:
        logging.error(f"OCR 처리 실패: {str(e)}")
        return None

def create_searchable_pdf(image_path: str, output_path: Optional[str] = None, lang: str = 'kor+eng') -> Optional[str]:
    """검색 가능 PDF 생성"""
    try:
        if not validate_image_path(image_path):
            return None

        if not output_path:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_searchable.pdf"

        logging.info(f"PDF 생성 시작: {os.path.basename(output_path)}")

        with Image.open(image_path) as img:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang=lang)
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        logging.info(f"PDF 생성 완료: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"PDF 생성 실패: {str(e)}")
        return None

def main():
    # 명령행 인자 처리
    if len(sys.argv) < 2:
        print("사용법: python main.py <이미지_경로> [출력_경로]")
        print("예시: python main.py input.jpg output.png")
        image_path = 'D:/project/computerVision/code/input/5.jpg'
        output_path = 'D:/project/computerVision/code/output/scanned_result.png'
        print(f"기본 설정으로 실행: {image_path} -> {output_path}")
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'scanned_result.png'
    
    # 입력 검증
    if not validate_image_path(image_path):
        return
    
    # 출력 디렉토리
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
    
    # 문서 스캔 실행
    scanner = DocumentScanner(image_path)
    print(f"입력 이미지: {image_path}")
    print(f"출력 경로: {output_path}")
    print("-" * 50)
    
    try:
        # 문서 스캔
        result = scanner.scan_document(
            show_steps=True,
            output_path=output_path
        )
        
        if result is None:
            print("스캔 실패")
            return

        print(f"🎉 스캔 완료! 결과 크기: {result.shape}")
        
        # OCR 처리
        print("\nOCR 처리 시작")
        txt_path = perform_ocr(output_path)
        if txt_path:
            print(f"추출된 텍스트 파일: {txt_path}")
        else:
            print("OCR 실패 - 텍스트 추출 불가")
        
        # PDF 생성
        print("\n검색 가능한 PDF 생성 시작")
        pdf_path = create_searchable_pdf(output_path)
        if pdf_path:
            print(f"PDF 생성 완료: {pdf_path}")
        else:
            print("PDF 생성 실패")

    except Exception as e:
        logging.error(f"전체 프로세스 실패: {str(e)}")
        print("다음을 확인:")
        print("1. 이미지가 명확하게 촬영되었는지")
        print("2. Tesseract OCR이 정상 설치되었는지")
        print("3. 출력 경로에 쓰기 권한이 있는지")

if __name__ == "__main__":
    main()
