"""외관 검사를 위한 이상 탐지와, 그 성능을 미리 등록한 규칙으로 재는 코드를 담은 패키지."""

# [파일 역할] defect_inspect 패키지의 입구다. 실제 기능은 하위 모듈(데이터 받기, PatchCore·Dinomaly 실행,
# 분석, 검사 서비스)에 있고, 대부분 `python -m defect_inspect.<모듈>` 명령으로 따로 실행한다.
# 여기에는 torch 같은 무거운 의존성을 두지 않는다. 기본 패키지만 설치한 CI에서도 `import defect_inspect`가
# 되어야 하기 때문이다.
# __version__: 패키지 버전 문자열. tests/test_package.py가 패키지를 불러올 수 있는지와 이 값이 비어 있지
# 않은지를 확인한다.
__version__ = "0.1.0"
