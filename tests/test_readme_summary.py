# [파일 역할] README 맨 위 '짧은 답' 절(소개의 질문 넷에 답을 한 줄씩 적은 글)이 제자리에 있고, 그 절의 숫자가
# 절 밖 본문에 그대로 있는지 보는 테스트 묶음이다.
# [왜 필요한가] '짧은 답'은 바로 아래 '한눈에' 표와 그 뒤 본문에 기록된 수치를 다시 적은 글이다. 옮겨 적다
# 틀리면 기록에 없는 새 수치가 README 첫 화면에 서게 되므로, 절에 나온 숫자마다 절 밖 본문에 같은 숫자가
# 있는지 본다. 숫자 경계를 지켜 찾기 때문에 1.6이 21.6이나 1.65의 일부로 맞는 일은 없다.
# [한계] 값이 옳은지는 보지 않는다. 그것은 본문 기록과 docs/experiments.md의 몫이다. 같은 숫자가 본문
# 어딘가에 있기만 하면 통과하므로 그 숫자가 알맞은 문맥에 쓰였는지도 보지 않는다. 한 자리 숫자는
# 어디에나 있어 건너뛴다.
# [주의] 표준 라이브러리만 쓰고 torch를 import하지 않는다(CI는 torch 없이 돈다). README는 이 파일의 위치에서
# 바로 읽는다. paths.ROOT는 tests/test_ledger.py처럼 임시 저장소로 바꿔 쓰는 시험이 있어 기대지 않는다.

import re
from pathlib import Path
from typing import NamedTuple

# 저장소 루트의 README.md. 이 파일이 tests/ 아래에 있으므로 한 단계 위가 저장소 루트다.
README = Path(__file__).resolve().parents[1] / "README.md"

# '짧은 답' 절의 제목 줄과, 그 바로 뒤에 와야 하는 '한눈에' 표 절의 제목 줄. 줄 전체가 같아야 한다.
SECTION_HEADING = "## 짧은 답"
TABLE_HEADING = "## 한눈에"

# 번호 붙은 목록 줄("1. ...")의 번호. 줄 맨 앞에서만 센다.
LIST_ITEM = re.compile(r"^(\d+)\. ", re.MULTILINE)

# 숫자 낱말: 소수(96.5), 분수 꼴(20/20), 쉼표 든 수(3,848)를 한 낱말로 잡는다. %p·ms·장 같은 단위와 부호는
# 낱말에 들어가지 않는다.
NUMBER = re.compile(r"\d+(?:[.,/]\d+)*")

# 낱말 길이가 이보다 짧은 숫자(한 자리 정수)는 어디에나 있어 있는지 보는 뜻이 없으므로 건너뛴다.
MIN_NUMBER_LEN = 2


class ReadmeParts(NamedTuple):
    """README를 이 시험들이 쓰는 조각으로 나눈 것."""

    headings: list[str]  # 줄 맨 앞이 '## '인 제목 줄을 나온 순서대로(줄 끝 공백은 뗀 것)
    intro: str  # 첫 '## ' 제목 줄 앞의 글: 맨 위 제목, 배지, 소개와 질문 넷, 용어 문단
    section: str  # '## 짧은 답' 제목 줄부터 다음 '## ' 제목 줄 앞까지
    body: str  # 절을 뺀 README 나머지 전체(절 앞의 소개와 절 뒤의 모든 글)


def load_readme() -> ReadmeParts:
    """README.md를 제목 목록, 소개, '짧은 답' 절, 절 밖 본문으로 나눈다. 절이 없으면 단언 오류를 낸다.

    절이 없을 때 빈 구간을 돌려주면 숫자 시험이 아무것도 검사하지 않고 통과하므로 여기서 먼저 막는다.
    제목은 줄 맨 앞이 '## '(해시 둘과 공백)인 줄만 센다. '### '로 시작하는 줄은 제목으로 세지 않는다. 코드
    울타리 안에 '## '로 시작하는 줄이 생기면 그것도 제목으로 세므로, 앞쪽(소개와 이 절)에는 두지 않는다.
    """
    lines = README.read_text(encoding="utf-8").split("\n")
    heading_at = [i for i, line in enumerate(lines) if line.startswith("## ")]
    starts = [i for i in heading_at if lines[i].rstrip() == SECTION_HEADING]
    assert starts, f"README.md에 '{SECTION_HEADING}' 절이 없다"
    start = starts[0]
    end = next((i for i in heading_at if i > start), len(lines))
    return ReadmeParts(
        headings=[lines[i].rstrip() for i in heading_at],
        intro="\n".join(lines[: heading_at[0]]),
        section="\n".join(lines[start:end]),
        body="\n".join(lines[:start] + lines[end:]),
    )


def list_numbers(text: str) -> list[int]:
    """글에서 줄 맨 앞이 '번호. '인 줄의 번호를 나온 순서대로 돌려준다(1. 2. 3. 4. 줄을 세는 데 쓴다)."""
    return [int(m.group(1)) for m in LIST_ITEM.finditer(text)]


def numbers_in(text: str) -> list[str]:
    """글에 나온 숫자 낱말을 나온 순서대로, 같은 낱말은 한 번만, MIN_NUMBER_LEN 이상인 것만 돌려준다."""
    return list(dict.fromkeys(t for t in NUMBER.findall(text) if len(t) >= MIN_NUMBER_LEN))


def appears_in(token: str, body: str) -> bool:
    """body에 token이 다른 숫자의 일부가 아닌 채로 나오는가.

    앞에 숫자나 '.'·','·'/'가 붙어 있으면(21.6의 1.6) 다른 수의 뒷부분이고, 뒤에 숫자나 '.'·','·'/' 다음
    숫자가 붙어 있으면(1.65의 1.6) 다른 수의 앞부분이라 맞는 것으로 치지 않는다. 문장 끝의 마침표(96.5.)는
    뒤에 숫자가 이어지지 않으므로 경계로 본다.
    """
    pattern = re.compile(r"(?<![\d.,/])" + re.escape(token) + r"(?!\d|[.,/]\d)")
    return pattern.search(body) is not None


# 확인: README의 첫 '## ' 제목이 '## 짧은 답'이고, 바로 다음 '## ' 제목이 '## 한눈에'다.
# 왜: 질문 넷을 읽은 사람이 긴 표보다 먼저 답을 한 줄씩 보게 하려는 절이다. 절이 표 아래로 밀리거나 둘 사이에
# 다른 절이 끼면 그 자리가 깨진다. 절이 아예 없으면 load_readme가 먼저 단언 오류를 낸다.
def test_short_answers_come_before_the_summary_table():
    first_two = load_readme().headings[:2]
    assert first_two == [SECTION_HEADING, TABLE_HEADING], f"README의 처음 두 '## ' 제목이 {first_two}이다"


# 확인: '짧은 답' 절 안에 '1. ', '2. ', '3. ', '4. '로 시작하는 줄이 이 순서로 정확히 넷 있고, 소개의 번호
# 붙은 질문(첫 '## ' 제목 앞의 '1. '~'4. ' 줄)도 정확히 넷이다.
# 왜: 답은 질문마다 한 줄씩 있어야 한다. 소개의 질문이 늘거나 줄었는데 답을 맞추지 않았거나, 답 줄이
# 빠지거나 하나 더 붙으면 번호로 바로 드러난다. 질문과 답의 내용이 서로 맞는지는 사람이 읽어 본다.
def test_short_answers_match_the_four_questions():
    parts = load_readme()
    questions = list_numbers(parts.intro)
    answers = list_numbers(parts.section)
    assert questions == [1, 2, 3, 4], f"소개의 번호 붙은 질문 줄의 번호가 {questions}이다"
    assert answers == [1, 2, 3, 4], f"'짧은 답' 절의 번호 붙은 줄의 번호가 {answers}이다"


# 확인: '짧은 답' 절의 숫자 낱말(두 글자 이상)마다 절 밖 본문에 숫자 경계를 지켜 나온다. 없는 숫자가 있으면 그
# 목록을 단언 메시지에 싣는다. 절에 비교할 숫자가 하나도 없는 경우도 실패로 본다.
# 왜: 절의 숫자는 본문에 기록된 수치를 다시 적은 것이어야 한다(파일 머리 주석). 본문은 절을 뺀 README 전체라
# 절의 숫자끼리 서로를 증명하는 일이 없다. 숫자가 하나도 없는 절은 이 시험이 아무것도 보지 않고 통과하는
# 구멍이 되므로 막는다.
def test_short_answer_numbers_appear_in_the_body():
    parts = load_readme()
    tokens = numbers_in(parts.section)
    assert tokens, "'짧은 답' 절에 확인할 숫자가 하나도 없다"
    missing = [token for token in tokens if not appears_in(token, parts.body)]
    assert not missing, f"'짧은 답' 절의 숫자가 절 밖 본문에 없다: {missing}"
