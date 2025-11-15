import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 모델 이름 정의
MODEL_NAME = "heegyu/polyglot-ko-1.3b-chat"

def load_llm_model():
    """
    RTX 5090의 강력한 VRAM을 활용하여
    4비트 양자화 없이, 모델의 원본 성능(float16)으로 로드합니다.
    (Colab T4보다 훨씬 빠르고 안정적입니다.)
    
    :return: (model, tokenizer) 튜플
    """
    
    print(f"--- [Task 4] LLM 로더 모듈 (RTX 5090 서버용) ---")
    print(f"대상 모델: {MODEL_NAME} (1.3B 파라미터)")
    print("4비트 양자화(bitsandbytes) 설정 제거 -> 'float16'으로 로드합니다.")

    try:
        start_time = time.time()
        
        # 'torch_dtype=torch.float16'을 추가하여 VRAM을 효율적으로 사용
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16, # (float32 대신 16비트로 로드)
            device_map="auto",        # (RTX 5090 GPU에 자동으로 할당)
            trust_remote_code=True
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        end_time = time.time()
        print(f"모델 및 토크나이저 로드 완료! (소요 시간: {end_time - start_time:.2f}초)")
        
        return model, tokenizer

    except Exception as e:
        print(f"!!!!! [오류] 모델 로드에 실패했습니다. !!!!!")
        print(f"!!!!! 1. 'transformers', 'torch', 'accelerate'가 설치되었는지 확인하세요. !!!!!")
        print(f"!!!!! 2. NVIDIA 드라이버 및 CUDA가 올바르게 설정되었는지 확인하세요. (터미널에서 'nvidia-smi' 입력) !!!!!")
        print(f"오류 메시지: {e}")
        return None, None

# --- 독립 테스트용 코드 ---
# (main_pipeline.py에서 import할 때는 실행되지 않음)
if __name__ == "__main__":
    print("--- llm_loader.py 독립 실행 테스트 ---")
    model, tokenizer = load_llm_model()
    
    if model and tokenizer:
        print("\n[테스트 성공] LLM 모듈이 정상적으로 로드되었습니다.")
    else:
        print("\n[테스트 실패] LLM 모듈 로드에 실패했습니다.")