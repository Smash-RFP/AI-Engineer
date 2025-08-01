from openai import OpenAI

def create_eval_obj(client, name, data_source_config, testing_criteria):
    # evals.create 함수를 호출하여 새로운 평가(evaluation) 작업을 생성한다.
    eval_obj = client.evals.create(
        name="RFP_Evaluation",  # 평가 작업 이름
        data_source_config={
            "type": "custom",  # 데이터 소스의 유형을 지정한. 'custom'은 사용자가 직접 데이터를 제공하는 방식.
            "item_schema": {
                "type": "object",  # 각 데이터 항목이 객체(object)
                "properties": {
                    "prompt": {"type": "string"},
                    "label": {"type": "string"},
                },
                # 각 데이터 항목에 'prompt'와 'label' 필드가 반드시 포함되어야 함을 명시
                "required": ["prompt", "label"],
            },
            # True로 설정 시, 데이터 소스 설정에 샘플 스키마가 포함
            "include_sample_schema": True,
        },
        # 평가 작업의 성공/실패 여부를 판단하는 기준 목록이다.
        testing_criteria=[
            {
                "type": "string_check",  # 평가 기준의 유형을 지정한다. 'string_check'는 문자열 비교를 수행한다.
                "name": "Match output to human label",  # 평가 기준의 이름이다.
                "input": "{{ sample.output_text }}", # 평가할 대상 문자열이다. 모델의 출력 텍스트를 나타내는 템플릿 변수
                "operation": "eq",  # 수행할 비교 연산이다. 'eq'는 'equal'을 의미하며, 값이 동일한지 확인한다.
                "reference": "{{ item.correct_label }}", # 정답 레이블을 나타내는 템플릿 변수
            }
        ],
    )

    return eval_obj

def upload_data(client):
    file = client.files.create(
			    file=open("/home/eojin-kim/AI-Engineer/src/generator/experiment/test_dataset/prompt.jsonl", "rb"),
			    purpose="evals"
		   )

    return file

def run_eval(client, eval_obj, file):
    run = client.evals.runs.create(
        "YOUR_EVAL_ID",  # string | 평가(eval) 작업의 고유 ID이다.
        name="Categorization text run",  # string | 생성될 실행(run)의 이름을 지정한다.
        data_source={  # object | 평가에 사용될 데이터 소스 설정
            "type": "responses",  # string | 데이터 소스의 유형. 'responses'는 모델의 응답을 평가함을 의미한다.
            "model": "gpt-4.1",  # string | 평가 대상이 되는 모델의 이름이다.
            "input_messages": {  # object | 모델에 제공될 입력 메시지 템플릿
                "type": "template",  # string | 입력 메시지의 유형. 'template'은 미리 정의된 템플릿을 사용함을 의미한다.
                "template": [  # array | 역할(role)과 내용(content)을 포함하는 메시지 템플릿 목록
                    {
                        "role": "developer",  # string | 메시지를 보내는 주체의 역할이다.
                        "content": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of 'Hardware', 'Software', or 'Other'. Respond with only one of those words."  # string | 역할에 해당하는 메시지 내용이다.
                    },
                    {
                        "role": "user",  # string | 사용자 역할을 정의한다.
                        "content": "{{ item.prompt }}"  # string | 'ticket_text' 필드의 데이터를 동적으로 삽입하는 템플릿 문법이다.
                    },
                ],
            },
            "source": {  # object | 실제 평가 데이터가 포함된 파일 소스 정보
                "type": "file_id",  # string | 소스의 유형. 'file_id'는 파일 ID를 통해 데이터를 참조함을 의미한다.
                "id": "YOUR_FILE_ID"  # string | 평가 데이터가 저장된 파일의 고유 식별자(ID)이다.
            },
        },
    )

    return run

def get_eval_result(client, eval_obj, run):
    result = client.evals.runs.get(eval_obj.id, run.id)
    return result