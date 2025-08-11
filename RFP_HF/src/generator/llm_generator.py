from openai import OpenAI
from config import PROMPT_TEMPLATE, DUMMY_QUERY_LIST, DUMMY_RFP_TEXT
from pprint import pprint

def generate_response(user_query: str, retrieved_rfp_text: str, previous_response_id: str = None, temperature: float = 1.0, model: str = "gpt-4.1-nano"):
    """
    OpenAI API를 사용하여 주어진 쿼리와 RFP 문서 내용을 기반으로 응답을 생성한다.

    user_query: 사용자가 입력한 질문
    retrieved_rfp_text: RFP 문서 내용
    temperature: 모델의 샘플링 온도로, 0~2 사이 값. (기본값: 1.0)
    model: 사용할 모델 이름
    반환값: 생성된 응답 텍스트와 이전 응답 ID
    """
    client = OpenAI()
    # 프롬프트 템플릿에 사용자 질문과 RFP 문서 내용을 삽입
    completed_prompt = PROMPT_TEMPLATE.format(
        retrieved_rfp_text=retrieved_rfp_text,
        new_user_text=user_query,
    )

    # OpenAI API를 사용하여 응답 생성
    response = client.responses.create(
        model=model,
        input=[
                {
                    "role": "user",
                    "content": completed_prompt
                }
        ],
        # --- 대화 및 상태 관리 ---
        previous_response_id=previous_response_id, #"이전 응답의 고유 ID이다. 여러 턴에 걸친 대화를 만들 때 사용한다.",
        store=True,  # boolean | 생성된 응답을 나중에 검색할 수 있도록 저장할지 여부이다. 기본값: true
        background=False,  # boolean | 모델 응답을 백그라운드에서 실행할지 여부이다. 기본값: false

        # --- 출력 제어 ---
        max_output_tokens=1024, # integer | 생성될 수 있는 토큰의 최대 상한선이다.
        stream=False, # boolean | true로 설정 시, 응답 데이터가 생성되는 대로 스트리밍된다. 기본값: false
        # 여기를 하드코딩된 1.0 대신, 함수 인자로 받은 temperature 값으로 변경
        # number | 샘플링 온도로, 0~2 사이 값이다. 높을수록 무작위성이 커진다. 기본값: 1
        temperature=temperature, 
        top_p=1.0, # number | temperature 대신 사용하는 핵 샘플링(nucleus sampling) 방식이다. 기본값: 1
        truncation="auto", # string | 컨텍스트 창 초과 시 입력을 자르는 전략이다. 'auto' 또는 'disabled'. 기본값: 'disabled'
    )

    # 이전 응답 ID 
    previous_response_id = response.id
    response_text = response.output_text

    # 응답 텍스트와 이전 응답 ID 반환
    return response_text, previous_response_id


if __name__ == "__main__":
    response_text, previous_response_id = generate_response(
        user_query=DUMMY_QUERY_LIST[0], 
        retrieved_rfp_text=DUMMY_RFP_TEXT
    )

    print("응답 메시지", response_text)
    print("이전 응답 ID", previous_response_id)