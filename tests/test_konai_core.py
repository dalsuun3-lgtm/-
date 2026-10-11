"""KONAI v9.1.99 핵심 산식 단위/경계/회귀 테스트.
※ 모든 입력은 합성(synthetic) 데이터이며 실제 시장·DART 데이터가 아니다."""
import math
import random

import numpy as np
import pandas as pd
import pytest


# ───────── Z'' ─────────
def test_zdart_hand_calc(k):
    # X1=.3 X2=.3 X3=.1 X4=400/600 → 6.56*.3+3.26*.3+6.72*.1+1.05*(2/3)=4.318
    z, v, _ = k._calc_zscore_dart(dict(ta=1000, te=400, ca=500, cl=200, ebit=100, re=300, year=2024))
    assert z == 4.32 and v == "○안전"


@pytest.mark.parametrize("ta,te,ca,cl,ebit,re,exp", [
    (1000, 400, 0, 0, 0, 0, "✕위험"),     # Z''=1.05*.667=.70
])
def test_zdart_zones(k, ta, te, ca, cl, ebit, re, exp):
    assert k._calc_zscore_dart(dict(ta=ta, te=te, ca=ca, cl=cl, ebit=ebit, re=re))[1] == exp


def test_zdart_missing_or_invalid_returns_none(k):
    assert k._calc_zscore_dart(dict(ta=0, te=0, ca=0, cl=0, ebit=0, re=0)) is None
    assert k._calc_zscore_dart(dict(ta=1000, te=400, ca=None, cl=1, ebit=1, re=1)) is None
    assert k._calc_zscore_dart(dict(ta=1000, te=float("nan"), ca=1, cl=1, ebit=1, re=1)) is None
    assert k._calc_zscore_dart(dict(ta=1000, te=1500, ca=1, cl=1, ebit=1, re=1)) is None  # 부채<0 데이터이상
    assert k._calc_zscore_dart(None) is None


def test_zdart_negative_equity_forced_distress(k):
    assert k._calc_zscore_dart(dict(ta=1000, te=-100, ca=900, cl=100, ebit=300, re=900))[1] == "✕위험"


def test_zscore_approx_edges(k):
    assert k._calc_zscore([1], [1], [1], None, [10], 1.0)[1] == "판단불가"
    assert k._calc_zscore([1], [1], [1], -5.0, [10], 1.0)[1] == "✕위험"      # 완전자본잠식
    assert k._calc_zscore([1], [1], [1], 50.0, [float("nan")], 1.0)[0] is None
    z = k._calc_zscore([1], [1], [1], 0.0, [10], 1.0)
    assert z[0] is not None and math.isfinite(z[0])                           # 무차입 X4 캡 10


# ───────── FCF ─────────
def test_fcf_rows_no_zero_fill(k):
    r = k.fcf_rows([100, None, 130], [50, 60, 20], [2022, 2023, 2024], 2024)
    assert r[1]["fcf"] is None and r[2]["fcf"] == 110


def test_fcf_grade_cases(k):
    g = lambda c, x: k.fcf_grade(k.fcf_rows(c, x, None, 2024))[0]
    assert g([100, 100, 100], [0, 0, 0]) == "훌륭함"
    assert g([-10, -20, -30], [5, 5, 5]) == "위험"
    assert g([100, 120, 130], [50, 60, 200]) == "보통"       # CFO 양호 + 일시 적자
    assert g([100, None, 130], [50, 60, 20]) == "좋음"
    assert g([None, None, None], [1, 1, 1]) is None


def test_fcf_grade_latest_missing_is_not_computed(k):
    assert k.fcf_grade(k.fcf_rows([100, 100, 100], [10, 10, None], None, 2024))[0] is None


def test_capex_like(k):
    def row(nm, v):
        return dict(sj_div="CF", account_nm=nm, thstrm_amount=v, thstrm_add_amount=None,
                    frmtrm_q_amount=None, frmtrm_add_amount=None, frmtrm_amount=None)
    df = pd.DataFrame([row("유형자산의 취득", "-300"), row("무형자산의 취득", "-50"), row("리스부채의 상환", "-20")])
    assert k._capex_like(df) == 370.0
    assert k._capex_like(pd.DataFrame([row("무형자산의 취득", "-50")])) is None   # PPE 없으면 0 위장 금지


# ───────── 분기 단독/누적 변환 ─────────
def _row(sj, nm, th=None, add=None, fq=None, fa=None, fr=None):
    return dict(sj_div=sj, account_nm=nm, thstrm_amount=th, thstrm_add_amount=add,
                frmtrm_q_amount=fq, frmtrm_add_amount=fa, frmtrm_amount=fr)


def _fixture_quarters():
    q1 = pd.DataFrame([_row("IS", "매출액", "1,000", "1,000", "900", "900"), _row("IS", "영업이익", "100", "100", "80", "80"),
                       _row("IS", "당기순이익", "70", "70", "50", "50"), _row("CF", "영업활동현금흐름", "100", fr="60")])
    q2 = pd.DataFrame([_row("IS", "매출액", "1,100", "2,100", "950", "1,850"), _row("IS", "영업이익", "110", "210", "90", "170"),
                       _row("IS", "당기순이익", "75", "145", "55", "105"), _row("CF", "영업활동현금흐름", "250", fr="130")])
    q3 = pd.DataFrame([_row("IS", "매출액", "1,200", "3,300", "1,000", "2,850"), _row("IS", "영업이익", "120", "330", "95", "265"),
                       _row("IS", "당기순이익", "80", "225", "60", "165"), _row("CF", "영업활동현금흐름", "330", fr="200")])
    return q1, q2, q3


def test_quarter_series_cfo_differenced_and_is_standalone(k):
    q1, q2, q3 = _fixture_quarters()
    o = k.build_quarter_series({1: q1, 2: q2, 3: q3}, 2024)
    assert o["filed"] == [1, 2, 3] and o["notes"] == []
    assert o["q"][2]["cfo"] == (150.0, 70.0) and o["q"][3]["cfo"] == (80.0, 70.0)   # 누적 차분
    assert o["q"][2]["sales"] == (1100.0, 950.0)                                      # 3개월 단독 칸


def test_quarter_series_missing_quarter_is_none_not_zero(k):
    q1, _, q3 = _fixture_quarters()
    o = k.build_quarter_series({1: q1, 3: q3}, 2024)
    assert o["q"][3]["cfo"] == (None, None)


def test_quarter_series_integrity_note_on_restatement(k):
    q1, q2, _ = _fixture_quarters()
    q2.loc[0, "thstrm_add_amount"] = "3,000"
    assert any("분기합" in n for n in k.build_quarter_series({1: q1, 2: q2}, 2024)["notes"])


def test_qd_dir_boundaries(k):
    assert k._qd_dir(None, 1) == ("na", None)
    assert k._qd_dir(5, -3)[0] == "turn_profit"
    assert k._qd_dir(-5, 3)[0] == "turn_loss"
    assert k._qd_dir(5, 0)[0] == "up" and k._qd_dir(0, 0)[0] == "flat"


# ───────── 12항목 / 분류 ─────────
def _fd(txt="", **slots):
    fd = [None] * 60
    fd[10] = txt
    for i, v in slots.items():
        fd[int(i[1:])] = v
    return fd


def test_fin_good_breakdown_has_twelve_items(k):
    good, gradeable, items = k.fin_good_breakdown(_fd())
    assert len(items) == 12 and good == 0 and gradeable == 0
    assert [n for n, _ in items] == ["신호등", "Z''", "현금흐름", "CFO×매출채권", "CFO/순이익", "M-Score",
                                     "당해년도", "업종별", "감사의견", "그레이엄", "이자보상배율", "배당"]


def test_fin_good_breakdown_text_parsing(k):
    txt = "○양호 F:7/9 🏦Z'':5.72(○안전) 이익질○좋음 💹💰저평가 [그레이엄18.5 | PEG 배당성장주"
    _, _, items = k.fin_good_breakdown(_fd(txt, s41=("○", "좋음"), s54=("◎", "훌륭함"), s45=("○", "낮음"),
                                           s51=("적정",), s56=("⚠", "위험")))
    st = dict(items)
    assert st["신호등"] == 1 and st["Z''"] == 1 and st["현금흐름"] == 1 and st["그레이엄"] == 1
    assert st["배당"] == 1 and st["감사의견"] == 1 and st["이자보상배율"] == -1 and st["M-Score"] == 1


def test_signal_anchor_not_confused_by_z_distress(k):
    # 신호등 △ + Z'' ✕위험 이 같이 있어도 신호등을 치명(-1)으로 읽지 않는다 (v9.1.37 회귀)
    _, _, items = k.fin_good_breakdown(_fd("△보통 F:4/9 🏦Z'':0.5(✕위험)"))
    st = dict(items)
    assert st["신호등"] == 0 and st["Z''"] == -1


@pytest.mark.parametrize("good,gr,tp,mark_bad,exp", [
    (10, 12, False, False, "재무우량"), (9, 12, False, False, "재무관찰"), (8, 12, False, False, "재무관찰"),
    (7, 12, False, False, "재무미달"), (7, 7, False, False, "재무우량"), (7, 8, False, False, "재무관찰"),
    (5, 8, False, False, "재무미달"), (10, 12, False, True, "재무관찰"), (3, 12, True, False, "재무우량"),
])
def test_lai_classify_thresholds(k, good, gr, tp, mark_bad, exp):
    assert k.lai_classify_fin(good, True, gr, tp, mark_bad) == exp


def test_lai_unverified_never_good(k):
    assert k.lai_classify_fin(12, False, 12, True) == "데이터부족"
    assert k.lai_classify_fin(None, True, 12) == "데이터부족"


def test_top_pick_requires_core_axes_and_no_fatal(k):
    full = "○양호 F:9/9 🏦Z'':5(○안전) 이익질○좋음 💹저평가 [그레이엄 배당성장주"
    fd = _fd(full, s41=("○", "좋음"), s54=("○", "좋음"), s45=("○",), s51=("적정",), s56=("○", "좋음"),
             s53={"eval": {"grade": "좋음"}})
    # 업종별(None)=결측 → 최상단 불가(미확인을 정상 취급 금지)
    assert k.is_fin_top_pick(fd) is True        # 11/11 판정가능 전부 양호(업종별은 비BIO라 구조적 해당없음)
    fd2 = _fd(full.replace("(○안전)", "(△회색지대)"), s41=("○", "좋음"), s54=("○", "좋음"), s45=("○",), s51=("적정",), s56=("○", "좋음"), s53={"eval": {"grade": "좋음"}})
    assert k.is_fin_top_pick(fd2) is False      # Z'' 가 ○가 아니면 최상단 불가(핵심 2축)
    fd3 = _fd(full, s41=("○", "좋음"), s54=("○", "좋음"), s45=("○",), s56=("○", "좋음"), s53={"eval": {"grade": "좋음"}})
    assert k.is_fin_top_pick(fd3) is False      # 감사의견 미확인 → 최상단 불가
    fd[56] = ("⚠", "위험")
    assert k.is_fin_top_pick(fd) is False       # 치명 1개면 무조건 제외


# ───────── 퍼징: 예외·NaN/inf 누출 없음 ─────────
_V = [None, 0, 0.0, -1, 1, -1e9, 1e9, float("nan"), float("inf"), float("-inf"), "abc", "1,234", True]


def _rv(r):
    return r.choice([-1, 1]) * r.random() * 10 ** r.randint(0, 12) if r.random() < .6 else r.choice(_V)


def _rl(r):
    return [_rv(r) for _ in range(r.choice([0, 1, 2, 3, 3, 3, 4]))]


def test_grade_functions_never_raise_or_leak_nan(k):
    r = random.Random(7)
    yrs = [2022, 2023, 2024]
    fns = [
        lambda: k.cfo_ar_grade(_rl(r), _rl(r), _rl(r)),
        lambda: k.cfo_ni_grade(_rl(r), _rl(r), yrs, r.choice([None, "BIO", "BANK"])),
        lambda: k.interest_coverage_grade(_rl(r), _rl(r), yrs, None),
        lambda: k.income_stmt_grade(_rl(r), _rl(r), _rl(r), yrs, None),
        lambda: k.fcf_grade(k.fcf_rows(_rl(r), _rl(r), None, 2024)),
        lambda: k._calc_zscore([1], [1], [1], _rv(r), [_rv(r) for _ in range(r.randint(0, 3))], _rv(r)),
        lambda: k._calc_zscore_dart({c: _rv(r) for c in ("ta", "te", "ca", "cl", "ebit", "re")}),
    ]
    for f in fns:
        for _ in range(1500):
            out = repr(f()).lower()
            assert "nan" not in out and "inf" not in out


# ───────── 기술 스크린(LAI) ─────────
def _ohlcv(n, start, drift, vol=1_000_000):
    idx = pd.bdate_range(end="2024-06-28", periods=n)
    c = start * np.exp(np.cumsum(np.full(n, drift)))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * .99, "Close": c, "Volume": np.full(n, vol)}, index=idx)


def test_lai_tech_pass_fail_and_guards(k):
    scr = lambda df: k.lai_tech_screen("000000", ohlcv_fn=lambda c, s: df)
    assert scr(_ohlcv(300, 50000, -.003))["st"] == "PASS"        # MA5<MA112
    assert scr(_ohlcv(300, 10000, .003))["st"] == "FAIL"
    assert scr(_ohlcv(300, 50000, 0.0))["st"] == "FAIL"          # 동일(경계) → 통과 아님
    assert scr(_ohlcv(100, 50000, -.003))["st"] == "NODATA"      # 봉부족
    assert scr(_ohlcv(300, 50000, -.003, vol=1))["st"] == "ILLIQ"
    d = _ohlcv(300, 50000, -.003); d.iloc[-1, d.columns.get_loc("Close")] = np.nan
    assert scr(d)["st"] == "NODATA"                              # 마지막 종가 NaN ≠ PASS
    assert scr(pd.DataFrame())["st"] == "NODATA" and scr(None)["st"] == "NODATA"


def test_lai_tech_no_lookahead(k):
    full = _ohlcv(320, 50000, -.002)
    a = k.lai_tech_screen("x", ohlcv_fn=lambda c, s: full.iloc[:-10])
    b = k.lai_tech_screen("x", ohlcv_fn=lambda c, s: full.iloc[:-10].copy())
    assert a == b                                                 # 동일 입력 → 동일 결과(재실행 결정성)
    assert a["date"] == full.index[-11].strftime("%Y-%m-%d")      # 마지막 봉 이후 데이터 미사용


def test_intraday_volume_fraction(k):
    from datetime import datetime
    now = datetime(2024, 6, 27, 12, 15)                           # 목요일, 09:00+195분
    assert k._intraday_volume_fraction(datetime(2024, 6, 27), now) == pytest.approx(0.5)
    assert k._intraday_volume_fraction(datetime(2024, 6, 26), now) == 1.0
    assert k._intraday_volume_fraction(datetime(2024, 6, 27), datetime(2024, 6, 27, 16, 0)) == 1.0
    assert k._intraday_volume_fraction(datetime(2024, 6, 27), datetime(2024, 6, 27, 9, 1)) == 0.1


def test_divergence_pct(k):
    assert k.lai_divergence_pct(90, 100) == pytest.approx(-10.0)
    assert k.lai_divergence_pct(1, 0) is None and k.lai_divergence_pct(float("nan"), 1) is None


# ───────── 신호 로그 원자적 저장/손상 격리 ─────────
def test_signal_log_atomic_and_corrupt_quarantine(k, tmp_path, monkeypatch):
    import os
    p = tmp_path / "sig.json"
    monkeypatch.setattr(k, "SIGNAL_LOG_FILE", str(p))
    k._save_signal_log([{"a": 1}])
    assert k.load_signal_log() == [{"a": 1}]
    p.write_text("{broken", encoding="utf-8")
    assert k.load_signal_log() == []
    assert any(".corrupt-" in f for f in os.listdir(tmp_path))    # 원본 보존
