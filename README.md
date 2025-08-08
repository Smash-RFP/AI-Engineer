## 📁 프로젝트 구조

AI-Engineer/
│
├── main.py              # 전체 RAG 파이프라인 실행의 진입점
├── environment.yaml     # 프로젝트 실행에 필요한 Conda 가상환경 설정 파일
├── data/                # 원문 문서 및 생성된 벡터DB 저장 폴더
├── src/
│   ├── loader/          # PDF, HWP 문서 로딩 및 텍스트/청크 전처리
│   ├── retriever/       # 사용자 질문에 대한 관련 문서 검색 모듈
│   ├── generator/       # 검색된 문서를 기반으로 LLM 응답 생성
│   └── vectordb/        # 텍스트 임베딩 생성 및 VectorDB 저장 모듈
└── README.md            # 프로젝트 소개 및 설명 문서
📄 디렉토리 및 파일 설명
main.py
전체 RAG 파이프라인 실행의 진입점입니다.

environment.yaml
필요한 패키지가 정의된 Conda 가상환경 설정 파일입니다.

data/
원문 문서, 처리된 텍스트, 생성된 벡터DB 등이 저장되는 디렉토리입니다.

src/loader/
PDF 및 HWP 문서를 로딩하고, 텍스트 추출 및 의미 단위 청킹을 수행합니다.

src/vectordb/
추출된 텍스트를 임베딩하고 FAISS/Chroma 등의 벡터DB로 저장합니다.

src/retriever/
사용자의 질의에 대해 벡터 유사도를 기반으로 관련 문서를 검색합니다.

src/generator/
검색된 문서를 기반으로 LLM을 활용해 최종 응답을 생성합니다.

README.md
프로젝트에 대한 개요와 사용법을 설명한 문서입니다.

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
