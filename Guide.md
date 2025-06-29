# 파이썬 가상 환경 초기 세팅 (macos 기준)

1. 파이썬 가상환경 생성
   python3 -m venv venv

2. 가상환경 활성화
   source venv/bin/activaete

3. 필수 패키지 설치
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter

4. 패키지 리스트 공유
   pip freeze > requirements.txt

5. requirements.txt로 패키지 일괄 설치
   pip install -r requirements.txt

6. 가상환경 비활성화
   deactivate
