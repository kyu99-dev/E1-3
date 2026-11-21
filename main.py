from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

EPSILON = 1e-9
BENCHMARK_REPEATS = 100  # 미션 최소 기준(10회) 이상, 안정적인 평균 측정을 위해 100회 반복
DEFAULT_DATA_FILENAME = "data.json"
DEFAULT_BENCHMARK_SIZES = [3, 5, 13, 25]


# =============================================================================
# 1. 데이터 구조 (2차원 배열 및 1차원 변환 지원)
# =============================================================================
class Matrix2D:
    """N×N 크기의 2차원 패턴 및 필터를 저장하고 관리하는 데이터 구조."""

    def __init__(self, data: List[List[float]]) -> None:
        if not data or not isinstance(data, list):
            raise ValueError("2차원 배열 데이터가 비어 있습니다.")
        row_count = len(data)
        validated: List[List[float]] = []
        for r, row in enumerate(data):
            if not isinstance(row, list) or len(row) != row_count:
                raise ValueError(
                    f"정방 행렬(N×N)이 아닙니다: {r + 1}행 길이 불일치 "
                    f"(기대: {row_count}, 실제: {len(row) if isinstance(row, list) else 'N/A'})"
                )
            converted_row: List[float] = []
            for c, val in enumerate(row):
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError(
                        f"숫자 데이터가 아닙니다: ({r}, {c}) 위치 값 = {val!r}"
                    )
                num = float(val)
                if not math.isfinite(num):
                    raise ValueError(
                        f"유한한 숫자가 아닙니다: ({r}, {c}) 위치 값 = {val!r}"
                    )
                converted_row.append(num)
            validated.append(converted_row)

        self._size = row_count
        self._data = validated

    @property
    def size(self) -> int:
        return self._size

    @property
    def data(self) -> List[List[float]]:
        return self._data

    def get(self, row: int, col: int) -> float:
        """특정 위치 (row, col)의 값을 읽어옵니다."""
        self._check_bounds(row, col)
        return self._data[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        """특정 위치 (row, col)에 값을 저장합니다."""
        self._check_bounds(row, col)
        num = float(value)
        if not math.isfinite(num):
            raise ValueError(f"유한한 숫자가 아닙니다: {value!r}")
        self._data[row][col] = num

    def _check_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self._size and 0 <= col < self._size):
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col}), 허용 범위: 0..{self._size - 1}"
            )

    def to_1d(self) -> List[float]:
        """보너스 과제: 2차원 배열을 길이 N²의 1차원 배열(row-major)로 변환합니다."""
        flat: List[float] = []
        for row in self._data:
            flat.extend(row)
        return flat

    def format_grid(self, max_display: int = 13) -> str:
        """콘솔 출력용 문자열로 변환합니다."""
        if self._size > max_display:
            return f"<{self._size}×{self._size} 행렬 (크기가 커서 생략)>"
        lines = []
        for row in self._data:
            formatted = [
                str(int(v)) if v.is_integer() else f"{v:g}" for v in row
            ]
            lines.append(" ".join(formatted))
        return "\n".join(lines)


# =============================================================================
# 2. 보너스 과제: 패턴 생성기 (Cross / X 자동 생성)
# =============================================================================
def generate_cross_pattern(n: int) -> Matrix2D:
    """크기 N의 N×N 십자가(Cross) 패턴을 자동 생성합니다."""
    if n < 1:
        raise ValueError("패턴 크기 N은 1 이상의 정수여야 합니다.")
    grid = [[0.0 for _ in range(n)] for _ in range(n)]
    if n % 2 == 1:
        centers = {n // 2}
    else:
        centers = {n // 2 - 1, n // 2}

    for r in range(n):
        for c in range(n):
            if r in centers or c in centers:
                grid[r][c] = 1.0
    return Matrix2D(grid)


def generate_x_pattern(n: int) -> Matrix2D:
    """크기 N의 N×N X 패턴을 자동 생성합니다."""
    if n < 1:
        raise ValueError("패턴 크기 N은 1 이상의 정수여야 합니다.")
    grid = [[0.0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if r == c or r + c == n - 1:
                grid[r][c] = 1.0
    return Matrix2D(grid)


# =============================================================================
# 3. 라벨 정규화 (표준 라벨: Cross, X)
# =============================================================================
def normalize_filter_key(raw_key: str) -> str:
    """filters 내 키('cross', 'x')를 내부 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(raw_key, str):
        raise ValueError(f"필터 키가 문자열이 아닙니다: {raw_key!r}")
    key = raw_key.strip().lower()
    if key == "cross":
        return "Cross"
    if key == "x":
        return "X"
    raise ValueError(f"지원하지 않는 필터 키입니다: {raw_key!r}")


def normalize_expected_label(raw_label: str) -> str:
    """expected 값('+', 'x')을 내부 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(raw_label, str):
        raise ValueError(f"expected 라벨이 문자열이 아닙니다: {raw_label!r}")
    label = raw_label.strip().lower()
    if label in ("+", "cross"):
        return "Cross"
    if label == "x":
        return "X"
    raise ValueError(f"지원하지 않는 expected 라벨입니다: {raw_label!r}")


# =============================================================================
# 4. MAC 연산 (외부 라이브러리 금지 — 2D 기본 구현 & 1D 메모리 접근 최적화)
# =============================================================================
def mac_2d(pattern: List[List[float]], kernel: List[List[float]]) -> float:
    """2차원 배열 기반 MAC(Multiply-Accumulate) 연산."""
    n = len(pattern)
    if n != len(kernel):
        raise ValueError(
            f"패턴과 필터의 크기가 일치하지 않습니다: {n}×{n} vs {len(kernel)}×{len(kernel)}"
        )
    total = 0.0
    for r in range(n):
        row_p = pattern[r]
        row_k = kernel[r]
        if len(row_p) != n or len(row_k) != n:
            raise ValueError(f"{r + 1}행의 열 크기가 일치하지 않습니다.")
        for c in range(n):
            total += row_p[c] * row_k[c]
    return total


def mac_1d(pattern_1d: List[float], kernel_1d: List[float]) -> float:
    """보너스 과제: 1차원 배열(길이 N²) 기반 MAC 연산 (단일 루프/인덱싱 단순화)."""
    length = len(pattern_1d)
    if length != len(kernel_1d):
        raise ValueError(
            f"1차원 배열 길이가 일치하지 않습니다: {length} vs {len(kernel_1d)}"
        )
    total = 0.0
    for i in range(length):
        total += pattern_1d[i] * kernel_1d[i]
    return total


# =============================================================================
# 5. 점수 비교 정책 (허용오차 epsilon 기반) 및 출력 포맷
# =============================================================================
def compare_scores(
    score_a: float,
    score_b: float,
    label_a: str,
    label_b: str,
    tie_label: str = "UNDECIDED",
    epsilon: float = EPSILON,
) -> Tuple[str, bool]:
    """허용오차(epsilon) 기준으로 두 점수를 비교해 (판정 결과, 동점 여부)를 반환합니다."""
    if abs(score_a - score_b) < epsilon:
        return tie_label, True
    if score_a > score_b:
        return label_a, False
    return label_b, False


def format_score(score: float, high_precision: bool = False) -> str:
    """동점 근접 시 부동소수점 차이를 확인할 수 있도록 소수점 자릿수를 조절합니다."""
    if high_precision:
        return f"{score:.16f}"
    return f"{round(score, 6)}"


# =============================================================================
# 6. 성능 분석 (I/O 제외, 반복 평균 측정 및 2D vs 1D 최적화 비교)
# =============================================================================
def benchmark_mac(
    pattern: Matrix2D,
    kernel: Matrix2D,
    repeats: int = BENCHMARK_REPEATS,
) -> Dict[str, Any]:
    """동일 입력·동일 반복 횟수로 2D MAC과 1D 최적화 MAC의 평균 시간(ms) 및 정합성을 측정합니다."""
    p_2d = pattern.data
    k_2d = kernel.data
    p_1d = pattern.to_1d()
    k_1d = kernel.to_1d()

    # 2D vs 1D 연산 결과 일치 검증
    score_2d = mac_2d(p_2d, k_2d)
    score_1d = mac_1d(p_1d, k_1d)
    verified = abs(score_2d - score_1d) < 1e-12

    # 2D MAC 순수 연산 구간 측정
    start_2d = time.perf_counter()
    for _ in range(repeats):
        mac_2d(p_2d, k_2d)
    elapsed_2d_ms = ((time.perf_counter() - start_2d) / repeats) * 1000.0

    # 1D MAC 순수 연산 구간 측정 (배열 평탄화 완료 후 순수 연산만 비교)
    start_1d = time.perf_counter()
    for _ in range(repeats):
        mac_1d(p_1d, k_1d)
    elapsed_1d_ms = ((time.perf_counter() - start_1d) / repeats) * 1000.0

    improvement_pct = (
        ((elapsed_2d_ms - elapsed_1d_ms) / elapsed_2d_ms) * 100.0
        if elapsed_2d_ms > 0
        else 0.0
    )

    return {
        "size": pattern.size,
        "ops": pattern.size * pattern.size,
        "avg_2d_ms": elapsed_2d_ms,
        "avg_1d_ms": elapsed_1d_ms,
        "improvement_pct": improvement_pct,
        "verified": verified,
    }


def print_performance_table(
    results: List[Dict[str, Any]],
    repeats: int = BENCHMARK_REPEATS,
    title: str = "[3] 성능 분석",
) -> None:
    """크기(N×N) / 평균 시간(ms) / 연산 횟수(N²) 표를 출력합니다."""
    print("#---------------------------------------")
    print(f"# {title} (평균/{repeats}회)")
    print("#---------------------------------------")
    header = f"{'크기(N×N)':<10} {'평균 시간(ms)':>14} {'연산 횟수(N²)':>14}"
    print(header)
    print("-" * len(header))
    for item in results:
        n = item["size"]
        ops = item["ops"]
        size_label = f"{n}×{n}"
        print(f"{size_label:<10} {item['avg_2d_ms']:>14.6f} {ops:>14d}")


def print_optimization_table(
    results: List[Dict[str, Any]],
    repeats: int = BENCHMARK_REPEATS,
) -> None:
    """보너스 과제: 2D vs 1D 메모리 접근 최적화 전/후 성능 및 검증 결과를 출력합니다."""
    print("\n#---------------------------------------")
    print(f"# [보너스] 메모리 접근 최적화 비교 (2D vs 1D, 평균/{repeats}회)")
    print("#---------------------------------------")
    header = (
        f"{'크기':<8} {'2D 평균(ms)':>13} {'1D 평균(ms)':>13} "
        f"{'개선율(%)':>11} {'검증':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        n = item["size"]
        size_label = f"{n}x{n}"
        status = "PASS" if item["verified"] else "FAIL"
        print(
            f"{size_label:<8} "
            f"{item['avg_2d_ms']:>13.6f} "
            f"{item['avg_1d_ms']:>13.6f} "
            f"{item['improvement_pct']:>10.2f}% "
            f"{status:>8}"
        )


# =============================================================================
# 7. 모드 1: 사용자 입력 (3×3)
# =============================================================================
def read_matrix_from_console(name: str, size: int = 3) -> Matrix2D:
    """콘솔에서 N×N 행렬을 한 줄씩(공백 구분) 입력받아 검증합니다.

    보너스 기능: 첫 줄에 'gen cross' 또는 'gen x'를 입력하면 패턴 생성기로 자동 생성합니다.
    """
    print(f"{name} ({size}줄 입력, 공백 구분 / 자동 생성: 'gen cross' 또는 'gen x')")
    rows: List[List[float]] = []
    row_idx = 0

    while row_idx < size:
        raw = input(f"  [{row_idx + 1}/{size}행]: ").strip()

        # 첫 줄에서 패턴 생성기 재활용 단축 명령 허용
        if row_idx == 0:
            lowered = raw.lower()
            if lowered == "gen cross":
                mat = generate_cross_pattern(size)
                print(f"  → {size}×{size} Cross 패턴을 자동 생성했습니다.")
                return mat
            if lowered == "gen x":
                mat = generate_x_pattern(size)
                print(f"  → {size}×{size} X 패턴을 자동 생성했습니다.")
                return mat

        tokens = raw.split()
        if len(tokens) != size:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. "
                f"(입력된 개수: {len(tokens)})"
            )
            continue

        try:
            parsed_row = [float(tok) for tok in tokens]
            if any(not math.isfinite(val) for val in parsed_row):
                raise ValueError("non-finite float")
        except ValueError:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. "
                "(숫자 변환 실패)"
            )
            continue

        rows.append(parsed_row)
        row_idx += 1

    return Matrix2D(rows)


def run_manual_mode() -> None:
    """모드 1 실행 흐름: 필터 A/B 입력 → 저장 확인 → 패턴 입력 → MAC/판정 → 성능 분석(3×3)."""
    size = 3
    print("\n#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")
    filter_a = read_matrix_from_console("필터 A", size)
    print()
    filter_b = read_matrix_from_console("필터 B", size)

    print("\n#---------------------------------------")
    print("# [1-1] 필터 저장 확인")
    print("#---------------------------------------")
    print("[저장된 필터 A]")
    print(filter_a.format_grid())
    print("\n[저장된 필터 B]")
    print(filter_b.format_grid())

    print("\n#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")
    pattern = read_matrix_from_console("패턴", size)

    score_a = mac_2d(pattern.data, filter_a.data)
    score_b = mac_2d(pattern.data, filter_b.data)
    verdict, is_tie = compare_scores(
        score_a, score_b, label_a="A", label_b="B", tie_label="판정 불가"
    )
    high_prec = is_tie or abs(score_a - score_b) < 1e-6

    perf_a = benchmark_mac(pattern, filter_a, BENCHMARK_REPEATS)
    perf_b = benchmark_mac(pattern, filter_b, BENCHMARK_REPEATS)
    avg_2d_ms = (perf_a["avg_2d_ms"] + perf_b["avg_2d_ms"]) / 2.0
    avg_1d_ms = (perf_a["avg_1d_ms"] + perf_b["avg_1d_ms"]) / 2.0
    improvement_pct = (
        ((avg_2d_ms - avg_1d_ms) / avg_2d_ms) * 100.0 if avg_2d_ms > 0 else 0.0
    )

    print("\n#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")
    print(f"A 점수: {format_score(score_a, high_prec)}")
    print(f"B 점수: {format_score(score_b, high_prec)}")
    print(f"연산 시간(평균/{BENCHMARK_REPEATS}회): {avg_2d_ms:.6f} ms")
    if is_tie:
        diff = abs(score_a - score_b)
        print(f"판정: {verdict} (|A-B| = {diff:.12g} < {EPSILON:g})")
    else:
        print(f"판정: {verdict}")

    perf_summary = [
        {
            "size": size,
            "ops": size * size,
            "avg_2d_ms": avg_2d_ms,
            "avg_1d_ms": avg_1d_ms,
            "improvement_pct": improvement_pct,
            "verified": perf_a["verified"] and perf_b["verified"],
        }
    ]
    print()
    print_performance_table(perf_summary, BENCHMARK_REPEATS, title="[4] 성능 분석")
    print_optimization_table(perf_summary, BENCHMARK_REPEATS)


# =============================================================================
# 8. 모드 2: data.json 로드, 스키마 검증, 일괄 분석 및 리포트
# =============================================================================
def resolve_data_json_path(filename: str = DEFAULT_DATA_FILENAME) -> str:
    """main.py 위치와 현재 작업 디렉터리에서 data.json 경로를 탐색합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def load_and_validate_filters(
    raw_filters: Any,
) -> Tuple[Dict[int, Dict[str, Matrix2D]], Dict[int, str]]:
    """filters 섹션을 파싱하고 라벨 정규화 및 크기 검증을 수행합니다."""
    valid_filters: Dict[int, Dict[str, Matrix2D]] = {}
    filter_errors: Dict[int, str] = {}

    if not isinstance(raw_filters, dict):
        return valid_filters, filter_errors

    for size_key, pair in raw_filters.items():
        match = re.fullmatch(r"size_(\d+)", str(size_key))
        if not match:
            continue
        expected_n = int(match.group(1))

        if not isinstance(pair, dict):
            filter_errors[expected_n] = f"{size_key} 필터 구조가 객체(dict)가 아닙니다."
            continue

        normalized_pair: Dict[str, Matrix2D] = {}
        try:
            for raw_k, matrix_data in pair.items():
                std_label = normalize_filter_key(raw_k)
                mat = Matrix2D(matrix_data)
                if mat.size != expected_n:
                    raise ValueError(
                        f"{size_key}.{raw_k} 크기 불일치 "
                        f"(기대: {expected_n}×{expected_n}, 실제: {mat.size}×{mat.size})"
                    )
                normalized_pair[std_label] = mat

            if "Cross" not in normalized_pair or "X" not in normalized_pair:
                raise ValueError(f"{size_key}에 Cross 또는 X 필터가 누락되었습니다.")

            valid_filters[expected_n] = normalized_pair
        except ValueError as exc:
            filter_errors[expected_n] = str(exc)

    return valid_filters, filter_errors


def extract_pattern_size(pattern_key: str) -> Optional[int]:
    """size_{N}_{idx} 형태의 패턴 키에서 N을 추출합니다."""
    match = re.fullmatch(r"size_(\d+)_.+", str(pattern_key))
    if not match:
        return None
    return int(match.group(1))


def run_json_mode(data_path: Optional[str] = None) -> None:
    """모드 2 실행 흐름: 필터 로드 → 패턴 로드/검증 → MAC/판정 → 성능 분석 → 결과 요약."""
    path = data_path or resolve_data_json_path()
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except FileNotFoundError:
        print(f"[오류] data.json 파일을 찾을 수 없습니다: {path}")
        return
    except json.JSONDecodeError as exc:
        print(f"[오류] data.json 파싱 실패: {exc}")
        return

    if not isinstance(payload, dict):
        print("[오류] data.json 최상위 구조가 JSON 객체가 아닙니다.")
        return

    raw_filters = payload.get("filters")
    raw_patterns = payload.get("patterns")

    print("\n#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")
    filters_by_size, filter_errors = load_and_validate_filters(raw_filters)
    for n in sorted(set(filters_by_size.keys()) | set(filter_errors.keys())):
        if n in filters_by_size:
            labels = ", ".join(
                lbl for lbl in ("Cross", "X") if lbl in filters_by_size[n]
            )
            print(f"✓ size_{n:<2d} 필터 로드 완료 ({labels})")
        else:
            print(f"✗ size_{n:<2d} 필터 로드 실패: {filter_errors[n]}")

    print("\n#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")
    if not isinstance(raw_patterns, dict) or not raw_patterns:
        print("[오류] patterns 항목이 비어 있거나 올바르지 않습니다.")
        return

    total_tests = 0
    passed_tests = 0
    failed_cases: List[Tuple[str, str]] = []
    benchmark_samples: Dict[int, Tuple[Matrix2D, Matrix2D]] = {}

    for case_id, entry in raw_patterns.items():
        total_tests += 1
        print(f"--- {case_id} ---")

        # 1) 패턴 키에서 N 추출
        expected_n = extract_pattern_size(case_id)
        if expected_n is None:
            reason = f"패턴 키 형식 오류 (size_{{N}}_{{idx}} 형식 아님): {case_id}"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        # 2) 해당 크기 필터 존재/유효성 확인
        if expected_n not in filters_by_size:
            detail = filter_errors.get(
                expected_n, f"size_{expected_n} 필터가 존재하지 않습니다."
            )
            reason = f"필터 스키마/크기 오류 ({detail})"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        # 3) 패턴 엔트리 구조 및 expected 라벨 검증
        if not isinstance(entry, dict) or "input" not in entry or "expected" not in entry:
            reason = "패턴 스키마 오류 ('input' 또는 'expected' 필드 누락)"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        try:
            expected_label = normalize_expected_label(entry["expected"])
        except ValueError as exc:
            reason = f"라벨 정규화 실패 ({exc})"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        # 4) 패턴 배열 구조 및 필터와의 크기 일치 검증
        try:
            pattern_mat = Matrix2D(entry["input"])
        except ValueError as exc:
            reason = f"패턴 데이터 오류 ({exc})"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        cross_filter = filters_by_size[expected_n]["Cross"]
        x_filter = filters_by_size[expected_n]["X"]

        if pattern_mat.size != cross_filter.size or pattern_mat.size != x_filter.size:
            reason = (
                f"필터와 패턴 크기 불일치 "
                f"(필터: {cross_filter.size}×{cross_filter.size}, "
                f"패턴: {pattern_mat.size}×{pattern_mat.size})"
            )
            print(f"판정: ERROR | expected: {expected_label} | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        # 성능 분석용 크기별 대표 샘플 저장
        if expected_n not in benchmark_samples:
            benchmark_samples[expected_n] = (pattern_mat, cross_filter)

        # 5) MAC 연산 및 허용오차 기반 판정
        score_cross = mac_2d(pattern_mat.data, cross_filter.data)
        score_x = mac_2d(pattern_mat.data, x_filter.data)

        # 1D 최적화 결과와 정합성 검증
        score_cross_1d = mac_1d(pattern_mat.to_1d(), cross_filter.to_1d())
        score_x_1d = mac_1d(pattern_mat.to_1d(), x_filter.to_1d())
        if (
            abs(score_cross - score_cross_1d) > 1e-12
            or abs(score_x - score_x_1d) > 1e-12
        ):
            reason = "2D MAC과 1D MAC 연산 결과 불일치"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        verdict, is_tie = compare_scores(
            score_cross,
            score_x,
            label_a="Cross",
            label_b="X",
            tie_label="UNDECIDED",
        )
        diff = abs(score_cross - score_x)
        high_prec = is_tie or diff < 1e-6

        print(f"Cross 점수: {format_score(score_cross, high_prec)}")
        print(f"X 점수: {format_score(score_x, high_prec)}")

        if verdict == expected_label:
            passed_tests += 1
            print(f"판정: {verdict} | expected: {expected_label} | PASS")
        else:
            if is_tie:
                reason = (
                    f"예상={expected_label}, 실제=UNDECIDED, "
                    f"점수 차이={diff:.12g} < EPSILON({EPSILON:g})로 동점 처리"
                )
                print(
                    f"판정: {verdict} | expected: {expected_label} | FAIL (동점 규칙)"
                )
            else:
                reason = f"예상={expected_label}, 실제={verdict} (판정 불일치)"
                print(f"판정: {verdict} | expected: {expected_label} | FAIL")
            failed_cases.append((case_id, reason))

    # 6) 크기별 성능 분석 (3×3 포함 + data.json 크기 5×5, 13×13, 25×25)
    print()
    perf_results: List[Dict[str, Any]] = []

    # 3×3은 패턴 생성기(보너스)를 재활용해 벤치마크 포함
    pattern_3 = generate_cross_pattern(3)
    filter_3 = generate_cross_pattern(3)
    perf_results.append(benchmark_mac(pattern_3, filter_3, BENCHMARK_REPEATS))

    all_benchmark_sizes = sorted(
        (set(benchmark_samples.keys()) | set(filters_by_size.keys())) - {3}
    )
    for n in all_benchmark_sizes:
        if n in benchmark_samples:
            pat, flt = benchmark_samples[n]
        else:
            pat = generate_cross_pattern(n)
            flt = filters_by_size[n]["Cross"]
        perf_results.append(benchmark_mac(pat, flt, BENCHMARK_REPEATS))

    print_performance_table(perf_results, BENCHMARK_REPEATS, title="[3] 성능 분석")
    print_optimization_table(perf_results, BENCHMARK_REPEATS)

    # 7) 최종 결과 요약
    failed_count = len(failed_cases)
    print("\n#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total_tests}개")
    print(f"통과: {passed_tests}개")
    print(f"실패: {failed_count}개")

    if failed_cases:
        print("\n실패 케이스:")
        for cid, reason in failed_cases:
            print(f"- {cid}: {reason}")
    else:
        print("\n실패 케이스: 없음 (모든 테스트 통과)")


# =============================================================================
# 9. 모드 3 (보너스): 자동 생성 Cross/X 패턴 성능 분석
# =============================================================================
def parse_sizes_input(raw: str) -> List[int]:
    """공백 또는 쉼표로 구분된 크기 목록을 파싱합니다. 빈 입력이면 기본 크기를 반환합니다."""
    if not raw.strip():
        return list(DEFAULT_BENCHMARK_SIZES)
    tokens = [tok for tok in re.split(r"[\s,]+", raw.strip()) if tok]
    sizes: List[int] = []
    for tok in tokens:
        n = int(tok)
        if n < 1:
            raise ValueError("패턴 크기 N은 1 이상의 정수여야 합니다.")
        if n not in sizes:
            sizes.append(n)
    if not sizes:
        raise ValueError("최소 1개 이상의 크기를 입력하세요.")
    return sizes


def run_generator_mode() -> None:
    """보너스 과제: 임의 크기 N(또는 다중 크기)의 Cross/X 패턴 자동 생성, MAC 판정 및 2D vs 1D 성능 비교."""
    print("\n#---------------------------------------")
    print("# [보너스] 자동 생성 Cross/X 패턴 성능 분석")
    print("#---------------------------------------")
    while True:
        raw = input(
            "생성할 패턴 크기 N 입력 (예: 5 또는 3,5,13,25 / Enter=기본 3,5,13,25): "
        )
        try:
            sizes = parse_sizes_input(raw)
            break
        except ValueError:
            print("입력 형식 오류: 1 이상의 정수를 공백 또는 쉼표로 구분해 입력하세요.")

    perf_results: List[Dict[str, Any]] = []

    if len(sizes) == 1:
        n = sizes[0]
        cross_mat = generate_cross_pattern(n)
        x_mat = generate_x_pattern(n)

        print(f"\n[자동 생성된 {n}×{n} Cross 패턴/필터]")
        print(cross_mat.format_grid())
        print(f"\n[자동 생성된 {n}×{n} X 패턴/필터]")
        print(x_mat.format_grid())

        print("\n테스트할 입력 패턴을 선택하세요:")
        print("1. 생성된 Cross 패턴")
        print("2. 생성된 X 패턴")
        choice = input("선택 (1 또는 2, 기본=1): ").strip()
        test_pattern = x_mat if choice == "2" else cross_mat
        expected = "X" if choice == "2" else "Cross"

        score_cross = mac_2d(test_pattern.data, cross_mat.data)
        score_x = mac_2d(test_pattern.data, x_mat.data)
        verdict, is_tie = compare_scores(
            score_cross, score_x, label_a="Cross", label_b="X", tie_label="UNDECIDED"
        )

        print("\n[MAC 판정 결과]")
        print(f"Cross 점수: {format_score(score_cross, is_tie)}")
        print(f"X 점수: {format_score(score_x, is_tie)}")
        print(
            f"판정: {verdict} | expected: {expected} | "
            f"{'PASS' if verdict == expected else 'FAIL'}"
        )

        perf_results.append(benchmark_mac(test_pattern, cross_mat, BENCHMARK_REPEATS))
    else:
        print(f"\n선택된 크기 목록: {', '.join(f'{n}×{n}' for n in sizes)}")
        for n in sizes:
            cross_mat = generate_cross_pattern(n)
            x_mat = generate_x_pattern(n)

            # 생성된 Cross/X 패턴 분류 자가 검증
            s_c_on_c = mac_2d(cross_mat.data, cross_mat.data)
            s_x_on_c = mac_2d(cross_mat.data, x_mat.data)
            v_cross, _ = compare_scores(s_c_on_c, s_x_on_c, "Cross", "X")

            s_c_on_x = mac_2d(x_mat.data, cross_mat.data)
            s_x_on_x = mac_2d(x_mat.data, x_mat.data)
            v_x, _ = compare_scores(s_c_on_x, s_x_on_x, "Cross", "X")

            print(
                f"✓ {n}×{n} 패턴 생성 및 판정 검증: "
                f"Cross→{v_cross} ({'PASS' if v_cross == 'Cross' else 'FAIL'}), "
                f"X→{v_x} ({'PASS' if v_x == 'X' else 'FAIL'})"
            )
            perf_results.append(benchmark_mac(cross_mat, cross_mat, BENCHMARK_REPEATS))

    print()
    print_performance_table(perf_results, BENCHMARK_REPEATS, title="성능 분석")
    print_optimization_table(perf_results, BENCHMARK_REPEATS)


# =============================================================================
# 10. 메인 진입점 (반복 메뉴 지원)
# =============================================================================
def main() -> None:
    while True:
        print("\n=== Mini NPU Simulator ===")
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 자동 생성 Cross/X 패턴 성능 분석 (보너스)")
        print("0. 종료")

        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            return

        if choice == "0":
            print("프로그램을 종료합니다.")
            return
        if choice == "1":
            try:
                run_manual_mode()
            except (EOFError, KeyboardInterrupt):
                print("\n입력이 중단되어 메뉴로 돌아갑니다.")
        elif choice == "2":
            run_json_mode()
        elif choice == "3":
            try:
                run_generator_mode()
            except (EOFError, KeyboardInterrupt):
                print("\n입력이 중단되어 메뉴로 돌아갑니다.")
        else:
            print("입력 오류: 0, 1, 2, 3 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
