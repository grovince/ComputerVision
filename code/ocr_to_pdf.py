import pytesseract
from PIL import Image
import os
import logging
from typing import Optional

# Tesseract 경로
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_image_path(image_path: str) -> bool:
    """이미지 파일 경로 유효성 검사"""
    if not os.path.exists(image_path):
        logging.error(f"파일을 찾을 수 없습니다: {image_path}")
        return False
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        logging.error("지원하지 않는 파일 형식입니다. PNG/JPG 파일만 가능합니다.")
        return False
    return True

def perform_ocr(image_path: str, lang: str = 'kor+eng') -> Optional[str]:
    """
    이미지에서 텍스트 추출 및 텍스트 파일 저장 (한글+영어)
    :param image_path: 입력 이미지 경로
    :param lang: OCR 언어 설정 (기본: 한국어+영어)
    :return: 생성된 텍스트 파일 경로 또는 None
    """
    try:
        if not validate_image_path(image_path):
            return None

        logging.info(f"OCR 처리 시작: {os.path.basename(image_path)}")
        
        with Image.open(image_path) as img:
            # 한글+영어 Tesseract 설정
            custom_config = r'--psm 6 --oem 3 -l kor+eng'
            
            # OCR 수행
            text = pytesseract.image_to_string(img, config=custom_config)
            
            if not text.strip():
                logging.warning("추출된 텍스트가 없습니다.")
                return None

            # 텍스트 파일 저장 경로 생성
            base_name = os.path.splitext(image_path)[0]
            txt_path = f"{base_name}_ocr.txt"

            # 텍스트 파일 저장
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            logging.info(f"OCR 텍스트 저장 완료: {txt_path}")
            return txt_path

    except Exception as e:
        logging.error(f"OCR 처리 실패: {str(e)}")
        return None

def create_searchable_pdf(image_path: str, output_path: Optional[str] = None, lang: str = 'kor+eng') -> Optional[str]:
    """
    검색 가능한 PDF 생성 (한글+영어)
    :param image_path: 입력 이미지 경로
    :param output_path: 출력 PDF 경로 (None인 경우 자동 생성)
    :param lang: OCR 언어 설정
    :return: 생성된 PDF 경로 또는 None
    """
    try:
        if not validate_image_path(image_path):
            return None

        # 출력 경로 자동 생성
        if not output_path:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_searchable.pdf"

        logging.info(f"PDF 생성 시작: {os.path.basename(output_path)}")

        # PDF 생성 (한글+영어 혼재 최적화 설정)
        with Image.open(image_path) as img:
            # 한글+영어 혼재 문서에 최적화된 설정
            custom_config = r'--psm 6 --oem 3 -l kor+eng'
            
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                img, 
                extension='pdf',
                lang=lang,
                config=custom_config
            )
            
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        logging.info(f"PDF 생성 완료: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"PDF 생성 실패: {str(e)}")
        return None

def batch_process_directory(input_dir: str, output_dir: str, lang: str = 'kor+eng'):
    """
    디렉토리 단위 배치 처리 (한글+영어)
    :param input_dir: 입력 디렉토리
    :param output_dir: 출력 디렉토리
    :param lang: OCR 언어 설정
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if perform_ocr(input_path, lang):
                create_searchable_pdf(input_path, 
                                    os.path.splitext(output_path)[0] + '.pdf',
                                    lang)
                processed += 1

    logging.info(f"처리 완료: 총 {processed}개 파일")

if __name__ == "__main__":
    setup_logger()
    
    # 단일 파일 처리 예제
    image_file = "example.jpg"
    if perform_ocr(image_file):
        create_searchable_pdf(image_file)
