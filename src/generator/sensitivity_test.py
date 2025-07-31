import os
import json
import time
from llm_generator import generate_response
from config import DUMMY_RFP_TEXT, DUMMY_QUERY_LIST 

# 실험 결과를 저장할 디렉터리
RESULTS_DIR = "src/generator/test_results"

# 테스트할 Temperature 값 리스트 (0.1 ~ 1.0)
TEMPERATURES_TO_TEST = [round(i * 0.1, 1) for i in range(1, 11)]

# 추가 테스트할 Temperature 값 리스트 (1.1 ~ 2.0)
ADDITIONAL_TEMPERATURES = [round(i * 0.1, 1) for i in range(11, 21)]

# 프롬프트 당 반복 생성 횟수
NUM_REPETITIONS = 1
DUMMY_QUERY_LIST = DUMMY_QUERY_LIST[:1]

def run_sensitivity_test():
    """
    Temperature 민감도 테스트를 실행하고 모든 결과를 파일로 저장
    """
    # 결과 저장 디렉터리가 없으면 생성
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("Temperature 민감도 테스트를 시작합니다...")
    print(f"테스트할 프롬프트 개수: {len(DUMMY_QUERY_LIST)}개")
    print(f"테스트할 Temperature 값: {TEMPERATURES_TO_TEST}")
    print(f"프롬프트 당 반복 횟수: {NUM_REPETITIONS}회")
    print("-" * 50)

    # 전체 실험 결과물을 담을 딕셔너리
    all_results = []

    # 각 프롬프트를 순회
    for i, query in enumerate(DUMMY_QUERY_LIST[:1]):
        prompt_text = query
        
        # 각 Temperature 값을 순회
        for temp in TEMPERATURES_TO_TEST:
            
            print(f"[{i+1}/{len(DUMMY_QUERY_LIST)}] 프롬프트 테스트 중 (Temp: {temp})...")
            
            # 특정 프롬프트와 Temperature 조합의 결과물을 저장할 리스트
            responses_for_this_setting = []
            
            # N회 반복 생성
            for j in range(NUM_REPETITIONS):
                try:
                    # lll_generator에서 수정한 함수를 호출
                    response_text, _ = generate_response(
                        query=prompt_text,
                        retrieved_rfp_text=DUMMY_RFP_TEXT,
                        temperature=temp
                    )
                    responses_for_this_setting.append(response_text)
                    print(f"  - 반복 {j+1}/{NUM_REPETITIONS} 생성 완료")

                except Exception as e:
                    print(f"  - 반복 {j+1} 생성 중 오류 발생: {e}")
                    responses_for_this_setting.append(f"ERROR: {e}")
                
                # API 과부하 방지를 위한 약간의 지연 시간 (필요에 따라 조절)
                time.sleep(1)

            # 한 조합(프롬프트+Temperature)의 테스트 결과 기록
            result_entry = {
                "prompt_text": prompt_text,
                "temperature": temp,
                "responses": responses_for_this_setting
            }
            all_results.append(result_entry)

    # 모든 실험이 끝나면 전체 결과를 하나의 JSON 파일로 저장
    result_filename = os.path.join(RESULTS_DIR, "temperature_sensitivity_results.json")
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    print("-" * 50)
    print(f"모든 테스트가 완료되었습니다! 결과가 '{result_filename}' 파일에 저장되었습니다.")

def run_additional_sensitivity_test():
    """
    1.1~2.0 범위의 Temperature 민감도 테스트를 실행하고 기존 결과 파일에 추가
    """
    result_filename = os.path.join(RESULTS_DIR, "temperature_sensitivity_results.json")
    
    # 기존 결과 파일이 있는지 확인
    if not os.path.exists(result_filename):
        print("기존 결과 파일을 찾을 수 없습니다. 먼저 run_sensitivity_test()를 실행해주세요.")
        return
    
    # 기존 결과 불러오기
    with open(result_filename, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    
    print("추가 Temperature 민감도 테스트를 시작합니다...")
    print(f"테스트할 프롬프트 개수: {len(DUMMY_QUERY_LIST)}개")
    print(f"추가 테스트할 Temperature 값: {ADDITIONAL_TEMPERATURES}")
    print(f"프롬프트 당 반복 횟수: {NUM_REPETITIONS}회")
    print("-" * 50)

    # 추가 실험 결과물을 담을 리스트
    additional_results = []

    # 각 프롬프트를 순회
    for i, query in enumerate(DUMMY_QUERY_LIST[:1]):
        prompt_text = query
        
        # 각 추가 Temperature 값을 순회
        for temp in ADDITIONAL_TEMPERATURES:
            
            print(f"[{i+1}/{len(DUMMY_QUERY_LIST)}] 프롬프트 테스트 중 (Temp: {temp})...")
            
            # 특정 프롬프트와 Temperature 조합의 결과물을 저장할 리스트
            responses_for_this_setting = []
            
            # N회 반복 생성
            for j in range(NUM_REPETITIONS):
                try:
                    # lll_generator에서 수정한 함수를 호출
                    response_text, _ = generate_response(
                        query=prompt_text,
                        retrieved_rfp_text=DUMMY_RFP_TEXT,
                        temperature=temp
                    )
                    responses_for_this_setting.append(response_text)
                    print(f"  - 반복 {j+1}/{NUM_REPETITIONS} 생성 완료")

                except Exception as e:
                    print(f"  - 반복 {j+1} 생성 중 오류 발생: {e}")
                    responses_for_this_setting.append(f"ERROR: {e}")
                
                # API 과부하 방지를 위한 약간의 지연 시간 (필요에 따라 조절)
                time.sleep(1)

            # 한 조합(프롬프트+Temperature)의 테스트 결과 기록
            result_entry = {
                "prompt_text": prompt_text,
                "temperature": temp,
                "responses": responses_for_this_setting
            }
            additional_results.append(result_entry)

    # 기존 결과와 새로운 결과를 합치기
    combined_results = existing_results + additional_results
    
    # 합친 결과를 파일에 저장
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)
        
    print("-" * 50)
    print(f"추가 테스트가 완료되었습니다! 결과가 기존 파일 '{result_filename}'에 추가되었습니다.")
    print(f"총 결과 개수: {len(combined_results)}개 (기존: {len(existing_results)}개 + 추가: {len(additional_results)}개)")

if __name__ == "__main__":
    # 기존 0.1~1.0 테스트 실행
    # run_sensitivity_test()
    
    # 추가 1.1~2.0 테스트 실행
    run_additional_sensitivity_test()