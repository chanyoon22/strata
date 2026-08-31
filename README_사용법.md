# 정년연장·계속고용 DSS — 설치·실행 가이드

## 0. 준비물
Python 3.10 이상. 설치 여부 확인: 터미널(윈도우는 PowerShell)에서 `python --version`

## 1. 폴더 만들기
아무 곳에나 폴더 하나를 만든다. 예: `C:\Users\내이름\retirement_dss`
내가 준 파일 **전부**를 이 폴더 하나에 넣는다. 하위 폴더는 딱 하나, `.streamlit`만 만들고 그 안에 config.toml을 넣는다. 나머지는 전부 최상위에.

필수 파일 체크리스트 (하나라도 빠지면 실행 안 됨):
```
engine_retirement_v2.py      ← 시뮬레이션 엔진
dss_layer.py                 ← 시나리오·파레토·민감도
validation_v2.py             ← 검증 6종
llm_report.py                ← AI 해석 층
report_generator.py          ← PDF 생성기
run_pipeline.py              ← 전체 실행 스크립트
app.py                       ← 웹앱 (STRATA)
.streamlit/config.toml       ← 테마 설정 (★반드시 .streamlit 하위폴더에)
params_v2.json               ← 실측 파라미터 (필수 데이터)
big_firm_sample_2023.csv     ← KLIPS 대기업 실표본 (필수 데이터)
did_results.json             ← DiD 결과
refit_coefficients.json      ← 재적합 계수
event_study.csv              ← 이벤트 스터디
```

## 2. 라이브러리 설치 (최초 1회)
터미널에서 방금 만든 폴더로 이동한 뒤:
```
cd C:\Users\내이름\retirement_dss
pip install numpy pandas streamlit plotly matplotlib reportlab statsmodels
```

## 3. 실행 방법 3가지

**웹앱 (주력)**
```
streamlit run app.py
```
브라우저가 자동으로 열린다 (안 열리면 http://localhost:8501 접속).
좌측에서 자사 프로파일 입력 → "시나리오 분석 실행".

**PDF 리포트 일괄 생성**
```
python run_pipeline.py
```
약 2~3분 뒤 같은 폴더에 `executive_report.pdf` 생성.

**검증 스위트**
```
python validation_v2.py
```
6종 검증 결과가 JSON 한 줄씩 출력된다. 전부 `"passed": true`면 정상.

## 4. 흔한 오류
- `ModuleNotFoundError` → 2번 pip install 다시
- `FileNotFoundError: params_v2.json` → 파일이 코드와 같은 폴더에 없음
- 웹앱 화면이 하얗게 멈춤 → 반복 횟수를 50으로 낮추고 재실행
- 한글 깨진 PDF → `pip install reportlab --upgrade`

## 5. 자사 데이터로 쓰는 법
사이드바 입력만으로 충분하지만, 인구 자체를 자사 명부로 바꾸려면
`big_firm_sample_2023.csv`와 같은 형식(age, tenure, wage, hours, edu 5개 열)의
CSV로 교체하면 엔진이 그대로 자사 인력 기반으로 부트스트랩한다.
