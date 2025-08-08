## 프로젝트 구조

AI-Engineer/
│
├── main.py                  # 실행 진입점
├── environment.yaml              # conda 환경 파일 
├── data/                    # 문서 및 벡터DB 저장 폴더
├── src/
│   ├── loader/              # 문서 로딩 및 전처리
│   ├── retriever/           # 문서 검색기
│   ├── generator/           # 응답 생성기
│   └── vectordb/               # 임베딩 및 VectorDB 생성      
├── main.py                   # 실행 스크립트
└── README.md          


main.py: 전체 RAG 파이프라인 실행의 진입점입니다.
environment.yaml: 프로젝트 실행에 필요한 Conda 가상환경 설정 파일입니다.
data/: 원문 문서, 생성된 벡터DB 등이 저장됩니다.
src/loader: PDF, HWP 문서를 텍스트로 추출하고 의미 단위로 분할합니다.
src/embedding: 텍스트 임베딩 벡터를 생성하고 FAISS/Chroma DB를 구축합니다.
src/retriever: 사용자 질문에 대한 관련 문서를 벡터DB에서 검색합니다.
src/generator: 검색된 문서 기반으로 LLM이 응답을 생성합니다.
src/vectordb: 전처리된 문서를 임베딩 및 VectorDB에 저장합니다

<br>

## 배포 링크
[웹 서비스 링크](https://codeit-rfp-ai.netlify.app/)

<br>

## 보고서 PDF 링크
[보고서 PDF 링크](https://drive.google.com/drive/u/0/folders/1GEZdSbp-1DeFFZcOvhO3g0gv_ip3Zutk)

<br>

## 협업 일지
<div align="center">

|     | 이름&nbsp;&nbsp; |                                                     Notion                                                     |
| :-: | :--------------: | :------------------------------------------------------------------------------------------------------------: |
| 🧑‍💻  |    **김어진**    | [협업일지 🚀](https://efficient-saver-88c.notion.site/Daily-238880186a9f8076a5cce4d0fb21e783?source=copy_link) |
| 👩‍💻  |    **전혜정**    |             [협업일지 🚀](https://www.notion.so/23926cc555b4819180a5d94818e700b3?source=copy_link)             |
| 🧑‍💻  |    **이대석**    |             [협업일지 🚀](https://typhoon-friend-92d.notion.site/2378dd9e8ab380d082eedca33ac04203)             |
| 🧑‍💻  |    **이현도**    |                     [협업일지 🚀](https://www.notion.so/1fc5a0ce825e80278ec2e8b670db03fe)                      |
| 👩‍💻  |    **김하늘**    |                     [협업일지 🚀](https://www.notion.so/23806aa61633804eabd8f6ec466e277d)                      |
