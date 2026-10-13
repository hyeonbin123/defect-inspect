# [파일 역할] 패키지 자체가 import되는지 확인하는 가장 작은 연기(smoke) 테스트다.
# [왜 필요한가] CI(.github/workflows/ci.yml)는 torch 없이 기본·serve 그룹만 설치해 돈다. 이 테스트가 실패하면
# src/ 배치나 hatchling 빌드 설정(pyproject.toml의 packages)이 깨져 다른 테스트 이전에 패키지가 설치되지
# 않았다는 뜻이라, 원인을 가장 먼저 좁혀 준다.

import defect_inspect


# src/defect_inspect/__init__.py의 __version__이 비어 있지 않은지 본다. 값 자체(0.1.0)는 고정하지 않아 버전을
# 올려도 이 테스트는 고칠 필요가 없다.
def test_version():
    assert defect_inspect.__version__
