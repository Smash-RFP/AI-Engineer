
[시연 영상](https://youtu.be/AqXHLaE00Hs)

https://github.com/Smash-RFP/AI-Engineer/issues/44#issue-3303446380

<br>

💡 프로젝트 개요
---
- **B2G 입찰지원 전문 컨설팅 스타트업   'Smash-RFP'**
- **RFP 내용을 분석·요약과 즉각적인 질의응답 기능을 제공하는 AI 챗봇 서비스 구현**

>매일 수백 건의 RFP가 쏟아지지만, 수십 페이지씩 일일이 읽기는 버겁습니다. 
><br>
>**Smash-RFP**는 관련 문서를 빠르게 찾아 실시간으로 질문에 답해 문서 검토 시간을 대폭 줄여드립니다.
---
<br>

## 팀원 소개
|       | 이름           | Notion                                                                                                     | 역할                                | 담당 업무                                                                                                    |
| :----- | :----- | :----- | :----- | :----- |
| 🧑‍💻 | **김어진** (팀장) | [협업일지 🚀](https://efficient-saver-88c.notion.site/Daily-238880186a9f8076a5cce4d0fb21e783?source=copy_link) | LLM & Prompt Engineer             | - LLM 기반 질의응답 설계 및 프롬프트 엔지니어링<br>- FrontEnd / BackEnd 개발                                                 |
| 👩‍💻 | **김하늘**      | [협업일지 🚀](https://www.notion.so/23806aa61633804eabd8f6ec466e277d)                                          | LLM & Prompt Engineer             | - LLM 기반 질의응답 설계 <br>- 프롬프트 엔지니어링 및 응답 최적화                                                               |
| 🧑‍💻 | **이대석**      | [협업일지 🚀](https://typhoon-friend-92d.notion.site/2378dd9e8ab380d082eedca33ac04203)                         | RFP Document Structuring Engineer | - RFP 문서 데이터 EDA 및 통계 분석<br>- 텍스트/표/이미지 등 구조적 요소 추출<br>- 마크다운 기반 청킹 파이프라인 설계 및 구현<br>- 각 청크에 대한 메타데이터 생성 |
| 🧑‍💻 | **이현도**      | [협업일지 🚀](https://www.notion.so/1fc5a0ce825e80278ec2e8b670db03fe)                                          | Vector Database Engineer          | - 임베딩 및 Vector DB 구축 (FAISS/ChromaDB)<br>- 쿼리 매칭 및 효율적 검색 구현                                             |
| 👩‍💻 | **전혜정**      | [협업일지 🚀](https://www.notion.so/23926cc555b4819180a5d94818e700b3?source=copy_link)                         | Retrieval Engineer                | - HyDE, Hybrid, rerank 기반 Retriever 개발<br>- 검색 성능 평가 및 개선                                                |

<br>

## 기술 스택
**개발 환경** : ![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white) ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white) ![Visual Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)

**언어** : ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)

**라이브러리** : ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) 

<br>


## 📁 프로젝트 구조

```
AI-Engineer/
│
├── main.py              
├── environment.yaml     
├── data/                
├── src/
│   ├── loader/          
│   ├── retriever/       
│   ├── generator/       
│   └── vectordb/        
└── README.md            
```

---

## 📄 디렉토리 및 파일 설명

- **`main.py`**  :   전체 RAG 파이프라인 실행 코드
- **`environment.yaml`**  :   필요한 패키지가 정의된 Conda 가상환경 설정 파일입니다.
- **`data/`**  :   원문 문서, 처리된 텍스트, 생성된 벡터DB 등이 저장되는 디렉토리입니다.
- **`src/loader/`**  :   PDF 및 HWP 문서를 로딩하고, 텍스트 추출 및 의미 단위 청킹을 수행합니다.
- **`src/vectordb/`**  :   추출된 텍스트를 임베딩하고 FAISS 또는 Chroma 등의 벡터DB로 저장합니다.
- **`src/retriever/`**  :   사용자의 질의에 대해 벡터 유사도를 기반으로 관련 문서를 검색합니다.
- **`src/generator/`**  :   검색된 문서를 기반으로 LLM을 활용해 최종 응답을 생성합니다.
- **`README.md`**  :   프로젝트에 대한 개요와 사용법을 설명한 문서입니다.
  
---

## 🤖 배포 링크
[🔗 Smash-RFP 챗 서비스 바로가기](https://codeit-rfp-ai.netlify.app/)

---

## 📜 보고서 PDF 링크
[🔗 Smash-RFP 최종 보고서](https://drive.google.com/drive/u/0/folders/1GEZdSbp-1DeFFZcOvhO3g0gv_ip3Zutk)

---
