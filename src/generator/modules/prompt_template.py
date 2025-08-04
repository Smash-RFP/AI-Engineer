from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage

def few_shot_prompt(verbose=False):
    '''
    FewShotChatMessagePromptTemplate을 활용하여 Few-shot 예시를 관리하고
    chat history 기능으로 이전 대화 내용을 context로 추가하는 함수
    '''
    # Few-shot 예시 데이터 정의 
    examples = [
        # 요약형 질문 예시
        {
            "question": "한국바이오협회가 발주한 바이오산업 정보시스템 관련 사업 요구사항을 정리해줘.",
            "answer": """
    1. 사업명: 바이오산업 정보시스템 운영 및 개선  

    2. 주요 요구 조건: 
        1) 해외 바이오기업 정보 서비스 신규 구축
        - KBIOIS 내 현재 운영 중인 국내 바이오기업 정보 서비스와 동일하게 해외 바이오기업 정보 서비스 신규 구축
        - 용역 착수 이후 해외 바이오기업 정보 데이터 공유
        2) 국내 바이오기업 연구개발 현황조사 OALP 신규 개발
        - 국내 바이오기업 연구개발 현황조사 결과에 대한 데이터 마트 구축
        - 국내 바이오기업 연구개발 현황조사 초기 데이터 

    3. 주관 기관: 한국바이오협회

    4. 예산: 42,000천원(VAT포함) 

    5. 기간: 계약체결일로부터 2021년 4월 30일까지
"""},
        # 항목 추출형 예시
        {
            "question": "바이오산업 정보시스템 관련 사업의 입찰 참가자격은?",
            "answer": """
    바이오 산업 정보시스템 사업의 입찰 참가자격은 다음과 같습니다.

    가.「국가를 당사자로 하는 계약에 관한 법률 시행령」제12조(경쟁입찰의 참가자격) 및 동법 시행규칙 제14조(입찰참가자격요건의 증명) 규정에 의한 자격을 갖춘 업체
    나.「국가를 당사자로 하는 계약에 관한 법률 시행령」제76조(부정당업자의 입찰 참가자격 제한) 규정에 의한 부정당업자에 해당하지 않는 자
    다.「소프트웨어산업 진흥법」제24조(소프트웨어사업자의 신고) 규정에 의한 소프트웨어사업자[컴퓨터관련서비스산업] 면허를 보유한 업체
"""},
        # 정보 확인형 예시
        {
            "question": "바이오산업 정보시스템 관련 사업의 시스템의 요구사항 중 산출물 관련 내용이 있나?",
            "answer": """
    필수 사업 산출물 목록은 다음과 같습니다.
    1. 관리 산출물
        - 사업 수행 시: 보안서약서
        - 종료 시: 완료 보고서, 지원 현황 보고서
    
    2. 개발 산출물
        - 시스템 설계: 테이블목록, 테이블정의서
        - 구현 및 테스트: 단위 테스트 결과, 통합 테스트 결과

    사업종료일 20일 이전에 사업결과의 최종 산출물에 대한 초안을 협의한 후 제출하여야 합니다.
"""},
        # 목적/배경형 예시
        {
            "question": "한국바이오협회에서 발주한 바이오 정보 시스템 관련사업을 발주하였는데, 이 사업이 추진된 목적을 알려줘.",
            "answer": """
    본 사업의 사업 목적과 필요성은 다음과 같습니다.

    1. 2020년 구축 완료한 바이오산업 정보시스템(내부 분석시스템 및 대국민 웹포털)의 안정적 운영 필요
    2. 웹포털 내 글로벌 바이오기업 정보 서비스 추가 구축 및 메인화면 개편 등을 통한 서비스 고도화
"""},
        # 여러 문서 비교 예시
        {
            "question": "포항상공회의소 FTA 전략지역 마케팅 지원사업과 기회재정부의 새정부 경제정책 패러다임 홍보영상물 제작 사업의 차이점은?",
            "answer": """
    두 사업은 크게 사업 목적, 사업 대상에서 차이점이 있습니다. 

    1. 사업 목적
        1) 포항상공회의소 FTA 전략지역 마케팅 지원사업
        - 수출 기업의 해외마케팅 지원
        - 바이어 발굴 및 시장조사를 통한 수출역량 강화

        2) 기회재정부의 새정부 경제정책 패러다임 홍보영상물 제작 사업
        - 새정부 경제정책의 대국민 이해 증진
        - 정책 내용을 쉽고 재미있게 전달하는 영상 콘텐츠 제작

    2. 사업 대상
        1) 포항상공회의소 FTA 전략지역 마케팅 지원사업
        - 해외마케팅 참가 기업 (주로 중소기업)

        2) 기회재정부의 새정부 경제정책 패러다임 홍보영상물 제작 사업
        - 전 국민 (정책 수요자)
"""},
        # 후속 질문의 첫 번째 턴
        {
            "question": "서울서교초등학교 늘봄학교 프로그램 운영 용역의 운영 계획은?",
            "answer": """
    서울서교초등학교 늘봄학교 프로그램 운영 용역의 운영 계획은 다음과 같습니다.

    1. 원어민영어 A반
    - 대상: 1~2학년
    - 예상인원: 10명
    - 운영기간: 2025.3.17. ~ 2026.2.5.
    - 운영시간: 주 5회 (월~금), 1일 200분

    2. 원어민영어 B반
    - 대상: 1~3학년
    - 예상인원: 10명
    - 운영기간: 2025.3.17. ~ 2026.2.5.
    - 운영시간: 주 5회 (월~금), 1일 200분
"""},       

        # 후속 질문의 두 번째 턴
        {
            "question": "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
            "answer": """
    교육 및 학습과 관련한 사업으로 인천구월초등학교에서 발주한 방과후학교 프로그램 운영 용역 사업은 다음과 같습니다.

    1. 사업명 : 2024학년도 인천구월초등학교 방과후학교 운영 용역
    2. 사업목적
        1) 창의적인 교육경험의 제공을 통해 창의융합형 인재 육성
        2) 학생․학부모의 요구에 부합하는 다양한 방과후학교 프로그램 운영
        3) 교육격차를 완화 및 사교육비 경감 
    3. 사업기간 : 2024. 3. 1.~ 2024. 11. 30
    4. 사업예산 : 금 145,169,100원
"""},

        # 미포함 문서 예시
        {
            "question": "태권도진흥재단 발주한 태권도원 마케팅 모바일 앱 개발 관련 사업에 대해 요약해줘.",
            "answer": "죄송합니다. 해당 문서에는 정보가 없습니다."
        }
    ]

    # Few-shot 예시를 위한 프롬프트 템플릿 정의 
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}"),
        ("ai", "{answer}"),
    ])

    # FewShotChatMessagePromptTemplate 생성
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 시스템 메시지 템플릿 정의 
    system_template = """
당신은 '입찰메이트' 기업의 입찰 전문 컨설턴트 AI입니다.
제안요청서(RFP)를 기반으로 사용자의 질문에 신뢰도 높은 정보를 제공합니다.

[당신의 역할]
- 당신은 정부·공공기관의 RFP 문서를 분석하여 고객사에게 필요한 정보를 제공하는 전문 컨설턴트입니다.
- 주요 요구조건, 대상 기관, 예산, 제출 방식, 기술 요건 등 문서에 근거한 내용을 요약합니다.

[지시사항]
1. 반드시 문서(context)에서만 정보를 추출하여 응답하세요.
2. 유추하거나 생성하지 말고, 문서에 명시되지 않은 정보는 "해당 문서에는 정보가 없습니다."라고 안내하세요.
3. 응답은 간결하고 핵심 요약 위주로 작성하세요.
"""
    system_message = SystemMessagePromptTemplate.from_template(system_template)

    # 대화 기록 및 사용자 메시지 템플릿 정의
    history_placeholder = MessagesPlaceholder(variable_name="chat_history")
    user_template = """
[질문]
{question}

[문서 내용]
{context}

[답변]
"""
    user_message = HumanMessagePromptTemplate.from_template(user_template)

    # Few-shot 예시를 포함하는 최종 ChatPromptTemplate 생성
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        few_shot_prompt,  # FewShotChatMessagePromptTemplate 객체를 직접 삽입
        history_placeholder,
        user_message
    ])

    if verbose:
        print("✓ FewShotChatMessagePromptTemplate 적용 완료")
    
    return chat_prompt

# 테스트용 main
if __name__ == "__main__":
    prompt = few_shot_prompt(verbose=True)

    # 예시용 대화 기록 리스트
    chat_history = [
        HumanMessage(content="안녕하세요, 시스템의 주요 역할이 뭐죠?"),
        AIMessage(content="RFP 기반 정보를 요약하여 제공하는 컨설턴트 AI입니다."),
    ]

    # `format_messages`를 사용하여 모든 메시지 객체를 포함하는 리스트로 변환
    sample = prompt.format_messages(
        question="출입관리 기능에는 어떤 내용이 포함되나요?",
        context="이 시스템은 출입관리 기능과 방문자 기록 기능을 포함하고 있습니다.",
        chat_history=chat_history
    )

    print("==== FewShotChatMessagePromptTemplate 적용 테스트 ====")
    print(sample)
