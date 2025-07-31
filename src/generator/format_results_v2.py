import json
import os

def format_json_responses(input_file, output_file):
    """
    JSON 파일의 responses 내용을 실제 줄바꿈으로 변환하여 가독성을 개선
    """
    # 기존 JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 실제 줄바꿈을 찾아서 처리
    # JSON 문자열 내부의 \n을 실제 줄바꿈으로 변환
    formatted_content = content.replace('\\n\\n', '\n\n')
    formatted_content = formatted_content.replace('\\n', '\n')
    
    # 포맷팅된 내용을 새 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    print(f"포맷팅 완료: {input_file} -> {output_file}")

def main():
    # 파일 경로 설정
    input_file = "src/generator/test_results/temperature_sensitivity_results.json"
    output_file = "src/generator/test_results/temperature_sensitivity_results_formatted.json"
    
    # 파일이 존재하는지 확인
    if not os.path.exists(input_file):
        print(f"입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # 포맷팅 실행
    format_json_responses(input_file, output_file)
    
    # 원본 파일 백업 및 새 파일로 교체
    backup_file = input_file + ".backup"
    if os.path.exists(backup_file):
        os.remove(backup_file)
    os.rename(input_file, backup_file)
    os.rename(output_file, input_file)
    
    print(f"원본 파일이 백업되었습니다: {backup_file}")
    print(f"포맷팅된 파일이 저장되었습니다: {input_file}")

if __name__ == "__main__":
    main() 