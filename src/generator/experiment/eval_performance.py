from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from pprint import pp
import time
import json


from src.generator.config import DUMMY_RFP_TEXT

def create_eval_obj(client, name):
    # evals.create 함수를 호출하여 새로운 평가(evaluation) 작업을 생성한다.
    eval_obj = client.evals.create(
        name=name,  # 평가 작업 이름
        data_source_config={
            "type": "custom",  # 데이터 소스의 유형을 지정한. 'custom'은 사용자가 직접 데이터를 제공하는 방식.
            "item_schema": {
                "type": "object",  # 각 데이터 항목이 객체(object)
                "properties": {
                    "user_query": {"type": "string"},
                    "label": {"type": "string"},
                },
                # 각 데이터 항목에 'prompt'와 'label' 필드가 반드시 포함되어야 함을 명시
                "required": ["user_query", "label"],
            },
            # True로 설정 시, 데이터 소스 설정에 샘플 스키마가 포함
            "include_sample_schema": True,
        },
        # 평가 작업의 성공/실패 여부를 판단하는 기준 목록이다.
        testing_criteria=[
             {
                "type": "text_similarity",
                "name": 'meteor grader',
                "input": "{{  sample.output_text  }}",
                "reference": "{{  item.label  }}",
                "evaluation_metric": "meteor" # "fuzzy_match" | "bleu" | "gleu" | "meteor" | "cosine" | "rouge_1" | "rouge_2" | "rouge_3" | "rouge_4" | "rouge_5" | "rouge_l" 
            }
        ],
    )

    return eval_obj

def upload_data(client, file_path):
    file = client.files.create(
			    file=open(file_path, "rb"),
			    purpose="evals"
		   )

    return file

def run_eval(client, eval_obj, file_obj, name, model, retrieved_rfp_text):
    run = client.evals.runs.create(
        eval_obj.id,  # string | 평가(eval) 작업의 고유 ID이다.
        name=name,  # string | 생성될 실행(run)의 이름을 지정한다.
        data_source={  # object | 평가에 사용될 데이터 소스 설정
            "type": "responses",  # string | 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
            "model": model,  # string | 평가 대상이 되는 모델의 이름이다.
            "input_messages": {  # object | 모델에 제공될 입력 메시지 템플릿
                "type": "template",  # string | 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
                "template": [  # array | 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
                    {
                        "role": "user",  # string | 사용자 역할을 정의한다.
                        "content": f"""
<prompt>
    <identity>
        너는 데이터 분석가이다. 너의 유일한 임무는 컨설턴트가 고객사에게 적합한 입찰 기회를 놓치지 않도록, 매일 쏟아지는 RFP의 핵심 정보를 가장 효율적으로 처리하여 제공하는 것이다. 너는 사용자의 질문에 대해, RFP 문서에서 객관적인 사실(Fact)만을 추출하여 한눈에 파악하기 쉬운 형태로 요약하고 구조화한다. 너의 목표는 컨설턴트가 RFP의 전체 내용을 읽지 않고도, 단 몇 분 안에 해당 입찰의 참여 여부를 판단할 수 있는 핵심 근거를 제공하는 것이다.
    </identity>
    <instructions>
        ## 목표
        - 사용자의 질문에 대해 주어진 RFP 문서 내용에만 근거하여 명확하고 사실적인 답변을 생성한다.

        ## 핵심 규칙
        1.  사실 기반 응답: 오직 아래 <context>에 제공된 RFP 문서 내용만을 사용하여 답변해야 한다. 추론, 가정, 또는 외부 지식을 절대 사용해서는 안 된다.
        2.  출처 명시: 모든 주장의 근거는 반드시 문서에서 인용하여 신뢰성을 높인다.
        3.  종합적 요약: 관련된 여러 문서 조각이 있다면, 이를 논리적으로 종합하여 하나의 일관된 답변으로 재구성한다. 사용자가 전체 맥락을 파악할 수 있도록 핵심 정보를 요약하여 제공한다.
        4.  쉬운 언어 사용: RFP의 전문 용어나 복잡한 문장은 사용자가 이해하기 쉬운 평이한 표현으로 바꾸어 설명한다.

        ## 답변 형식
        - '예/아니오' 질문: 질문이 '예/아니오'로 답변될 수 있는 경우, 문서 내용을 근거로 '예' 또는 '아니오'로 명확하게 먼저 답변하고, 그 이유가 되는 문장을 함께 제시한다.
        - 비교 질문: 두 개 이상의 RFP 문서를 비교해달라는 요청에는, 지정된 기준에 따라 각 문서의 핵심 내용을 표(Table) 형식으로 요약하여 명확하게 비교한다.
        - 전문가적 조언: 입찰 컨설턴트로서, RFP 문서 내용을 바탕으로 전문적인 조언을 포함하여 답변을 구성한다.
    </instructions> 
    <context>
        <retrieved_rfp_documents>
            {retrieved_rfp_text}
        </retrieved_rfp_documents>

        <user_question>
            {{{{item.user_query}}}}
        </user_question>
    </context>
</prompt>
"""  # string | 'user_query' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
                    },
                ],
            },
            "source": {  # object | 실제 평가 데이터가 포함된 파일 소스 정보
                "type": "file_id",  # string | 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
                "id": file_obj.id  # string | 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
            },
        },
    )

    return run

def retrieve_eval_result(client, eval_obj, run):
    result = client.evals.runs.retrieve(eval_id=eval_obj.id, run_id=run.id)
    return result

def eval_llm(model: str = 'gpt-4.1-nano', 
            file_path: str = "/home/eojin-kim/AI-Engineer/src/generator/experiment/test_dataset/prompt.jsonl",
            retrieved_rfp_text: str = DUMMY_RFP_TEXT
            ):
    eval_name = "RFP_Evaluation"

    client = OpenAI()

    eval_obj = create_eval_obj(client, eval_name)
    file = upload_data(client, file_path)
    run = run_eval(client, eval_obj, file, eval_name, model, retrieved_rfp_text)

    print("평가 작업이 제출되었습니다. 완료될 때까지 대기합니다.")
    while True:
        # 매 루프마다 평가 실행의 최신 상태를 가져옴
        result = retrieve_eval_result(client, eval_obj, run)

        # 현재 상태를 출력하여 진행 상황을 보여줌
        print(f"현재 상태: {result.status}... 확인 중...")

        # 상태가 'completed'이면 루프를 종료
        if result.status == 'completed':
            print("평가가 성공적으로 완료되었습니다.")
            break
        # 실패하거나 취소된 경우에도 루프를 종료
        elif result.status in ['failed', 'cancelled', 'errored']:
            print(f"평가가 {result.status} 상태로 종료되었습니다.")
            break

        # 아직 진행 중이면 10초간 대기한 후 다시 확인 (API 요청 제한을 피하기 위함)
        time.sleep(10)

    # 루프가 종료된 후, 최종 결과를 출력
    print("\n--- 최종 평가 결과 ---")
    output_items = client.evals.runs.output_items.list(eval_id=eval_obj.id, run_id=run.id)
    output_items = output_items.model_dump()
    print('output_items: ', output_items)

    # score, response 출력
    for v in output_items['data']:
        score = v['results'][0]['score'] 
        response = v['sample']['output'][0]['content']
        print('score:', score)
        print('response:', response)
    
    return output_items

if __name__ == "__main__":
    output_items = eval_llm(model = 'gpt-4.1-nano', 
            file_path = "/home/eojin-kim/AI-Engineer/src/generator/experiment/test_dataset/prompt.jsonl",
            retrieved_rfp_text = DUMMY_RFP_TEXT)