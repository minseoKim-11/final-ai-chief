# DeepL API
import requests
# 에러 로깅용
import json 
# env 로드용
from dotenv import load_dotenv
import os
load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") 
DEEPL_API_URL = os.getenv("DEEPL_API_URL")

def deepl_translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    DeepL API를 호출하여 텍스트를 번역하는 핵심 함수.
    """
    if not text.strip():
        return text

    # API 요청 데이터 구성
    data = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }

    try:
        # API POST 요청
        res = requests.post(DEEPL_API_URL, data=data, timeout=10)
        # 4xx, 5xx 에러 발생 시 예외 발생
        res.raise_for_status() 
    
    except requests.RequestException as e:
        # 네트워크 오류 또는 API 에러 처리
        print(f"!!!!! [DeepL ERROR] 네트워크/API 오류: {e} !!!!!")
        # (오류 발생 시, 번역 실패 태그와 함께 원본 텍스트 반환)
        return f"[DeepL 오류: 번역 실패]\n{text}"

    # 응답 JSON 파싱
    try:
        result = res.json()
    except json.JSONDecodeError as e:
        print(f"!!!!! [DeepL ERROR] JSON 파싱 오류: {e} !!!!!")
        return f"[DeepL 오류: 응답 파싱 실패]\n{text}"

    translations = result.get("translations", [])
    if not translations:
        print("!!!!! [DeepL ERROR] API 응답에 'translations' 키가 없습니다. !!!!!")
        return f"[DeepL 오류: 번역 결과 없음]\n{text}"

    # 최종 번역된 텍스트 반환
    return translations[0].get("text", text)

# --- 헬퍼 함수 ---

def ko_to_en(text: str) -> str:
    """
    한국어 -> 영어(미국식) 번역
    (주의: 'EN' 대신 'EN-US'를 권장합니다)
    """
    return deepl_translate(text, source_lang="KO", target_lang="EN-US")

def en_to_ko(text: str) -> str:
    """
    영어 -> 한국어 번역
    """
    return deepl_translate(text, source_lang="EN", target_lang="KO")