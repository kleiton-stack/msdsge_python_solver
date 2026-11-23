"""timeseries.py, display.py and report.py. RISE references in validation/ref/tools.mat."""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from msdsge import Model, load
from msdsge import timeseries as tsm
from msdsge.display import check, print_solution, resid, solution_table
from test_filter import MS

ROOT = Path(__file__).parents[1]
REF = ROOT / "validation" / "ref"


@pytest.fixture(scope="module")
def ref():
    return sio.loadmat(REF / "tools.mat", squeeze_me=True, struct_as_record=False)


# ----------------------------------------------------------------- time series
def test_dates():
    assert tsm.to_rise_date(tsm.to_period("1990Q1") + 5) == "1991Q2"
    assert tsm.to_rise_date(tsm.to_period("1990M11") + 3) == "1991M2"
    assert tsm.to_rise_date(tsm.to_period("1990H2") + 1) == "1991H1"
    assert tsm.to_rise_date(tsm.to_period(1990) + 1) == "1991"
    assert [tsm.to_rise_date(p) for p in tsm.date_range("1990Q3", 3)] == ["1990Q3", "1990Q4", "1991Q1"]
    assert len(tsm.date_range("1990M1", end="1991M12")) == 24
    assert tsm.frequency("1990Q1") == "Q" and tsm.frequency(tsm.ts("1990M1", np.zeros(3))) == "M"
    for bad in ("1990Q5", "90Q1", "1990M13", "abc"):
        with pytest.raises(ValueError):
            tsm.to_period(bad)
    df = tsm.ts("1990Q1", np.arange(8.0).reshape(4, 2), ["a", "b"])
    assert list(df.columns) == ["a", "b"] and tsm.to_rise_date(df.index[-1]) == "1990Q4"
    assert tsm.dummy("1990Q1", "1990Q4", "1990Q3").tolist() == [0, 0, 1, 0]
    assert tsm.step_dummy("1990Q1", "1990Q4", "1990Q3").tolist() == [0, 0, 1, 1]
    c = tsm.collect(tsm.ts("1990Q1", [1.0, 2.0], "a"), {"b": tsm.ts("1990Q2", [5.0, 6.0])})
    assert c.shape == (3, 2) and np.isnan(c["b"].iloc[0])


def test_filters_and_aggregation_match_rise(ref):
    t = tsm.ts("1990Q1", ref["ts_v"])
    trend, cycle = tsm.hpfilter(t, 1600)
    assert trend.to_numpy() == pytest.approx(ref["ts_hp_trend"], abs=1e-8)
    assert cycle.to_numpy() == pytest.approx(ref["ts_hp_cycle"], abs=1e-8)
    a = tsm.aggregate(t, "H")
    assert a.to_numpy() == pytest.approx(ref["ts_agg_H"]) and tsm.to_rise_date(a.index[0]) == ref["ts_agg_H_start"]
    tm = tsm.ts("1990M1", ref["ts_vm"])
    a = tsm.aggregate(tm, "Q")
    assert a.to_numpy() == pytest.approx(ref["ts_agg_Q"]) and tsm.to_rise_date(a.index[0]) == ref["ts_agg_Q_start"]
    assert tsm.aggregate(tm, "Q", "interpolation").to_numpy() == pytest.approx(ref["ts_agg_Qi"])
    assert tsm.aggregate(tm, "Y", "sum").iloc[0] == pytest.approx(ref["ts_vm"][:12].sum())
    assert tsm.aggregate(tm.iloc[1:], "Q").index[0] == tsm.to_period("1990Q2")     # incomplete quarter is dropped
    with pytest.raises(ValueError):
        tsm.aggregate(t, "M")
    tr, det = tsm.ma_filter(t, 2)
    assert tr.to_numpy() == pytest.approx(ref["ts_ma_trend"]) and tsm.to_rise_date(tr.index[0]) == ref["ts_ma_start"]
    assert (tr + det).to_numpy() == pytest.approx(t.to_numpy()[2:-2])
    assert tsm.ma_filter(t, 2, True)[0].to_numpy() == pytest.approx(ref["ts_ma_trend_ext"])
    assert tsm.ma_filter(t)[0].to_numpy() == pytest.approx(ref["ts_ma_default"])
    w = tsm.window(t, "1991Q2:1992Q1")
    assert w.to_numpy() == pytest.approx(ref["ts_window"]) and tsm.to_rise_date(w.index[0]) == ref["ts_window_start"]
    assert np.log(t).to_numpy() == pytest.approx(ref["ts_log"]) and t.cumsum().to_numpy() == pytest.approx(ref["ts_cumsum"])


def test_hp_bk_properties_and_transforms():
    n = np.arange(120.0)
    line = tsm.ts("1990Q1", 2 + 0.3 * n)
    assert tsm.hpfilter(line)[1].abs().max() < 1e-8              # a linear trend passes through whole
    df = tsm.ts("1990Q1", np.column_stack([n, n ** 2]), ["a", "b"])
    assert tsm.hpfilter(df)[0].shape == (120, 2)
    w = tsm.bk_weights(6, 32, 12)
    assert w.size == 25 and w.sum() == pytest.approx(0, abs=1e-12) and w == pytest.approx(w[::-1])
    cyc = tsm.bkfilter(tsm.ts("1990Q1", 5 + 0.1 * n + np.sin(2 * np.pi * n / 16)))
    assert cyc.isna().sum() == 24
    assert cyc.dropna().to_numpy() == pytest.approx(np.sin(2 * np.pi * n[12:-12] / 16), abs=0.1)   # a 16-quarter cycle passes
    assert tsm.bkfilter(tsm.ts("1990Q1", np.sin(2 * np.pi * n / 3))).abs().max() < 0.1              # 3 quarters does not pass
    x = tsm.ts("1990Q1", 100 * 1.01 ** np.arange(12))
    assert tsm.transform(x, "pct_ch").to_numpy() == pytest.approx(1.0)
    assert tsm.transform(x, 2).to_numpy() == pytest.approx(100 * (1.01 ** 4 - 1))
    assert tsm.transform(x, "yoy_pct_ch").to_numpy() == pytest.approx(100 * (1.01 ** 4 - 1))
    assert tsm.transform(x, "log_ch_ar").to_numpy() == pytest.approx(400 * np.log(1.01))
    assert tsm.transform(x, "yoy_log_ch").to_numpy() == pytest.approx(400 * np.log(1.01))
    assert tsm.transform(x, "yoy_diff").index[0] == tsm.to_period("1991Q1") and len(tsm.transform(x, "diff")) == 11
    assert tsm.transform(x, 1) is x
    xm = tsm.ts("1990M1", 100 * 1.01 ** np.arange(14))
    assert tsm.transform(xm, "pct_ch_ar").to_numpy() == pytest.approx(100 * (1.01 ** 12 - 1))
    with pytest.raises(ValueError):
        tsm.transform(x, "nothing")


# ----------------------------------------------------------------- display
def _norm(text):
    return [re.sub(r"_+", "_", re.sub(r"\s+", " ", l)).strip() for l in text.splitlines()
            if l.strip() and not l.startswith("SOLVER")]


def test_print_solution_matches_rise_text(ref):
    s = Model.from_text(MS).solve()
    sa = load(ROOT / "examples" / "refA.mod").solve()
    assert _norm(print_solution(s, file=io.StringIO())) == _norm(ref["flt_ms"].print)
    assert _norm(print_solution(s, "x", file=io.StringIO())) == _norm(ref["flt_ms"].print_x)
    assert _norm(print_solution(sa, file=io.StringIO())) == _norm(ref["refA"].print)   # with steady state row
    tab = solution_table(sa, regime=2)
    assert tab.loc["steady state", "Z"] == pytest.approx(-2.5) and list(tab.index) == ["steady state", "X{-1}", "Z{-1}", "E"]
    # anticipated shocks enter as rows E{+k}
    assert "E{+2}" in solution_table(sa.model.solve(shock_horizon=2)).index
    assert print_solution(s, regimes=[2], file=io.StringIO()).count("Regime") == 1


def test_resid_and_check(ref, capsys):
    sa = load(ROOT / "examples" / "refA.mod").solve()
    r = resid(sa.model)
    assert r.shape == np.shape(ref["refA"].resid_val) and np.abs(r).max() < 1e-9
    out = capsys.readouterr().out
    assert out.startswith(" \nequation #1 : ") and "equation #2 : " in out
    # at the starting point (zeros) the equations of the model with a mean do not hold
    r0 = resid(sa.model, at="initval", show=False)
    assert np.abs(r0[0]).max() == pytest.approx(0.2)                 # |mu (1 - rho)| = 1 * 0.2
    info = check(sa.model, sa, show=False)
    assert info["counts"] == {"static": 0, "predetermined": 1, "forward": 0, "both": 1}
    assert info["types"] == {"X": "predetermined", "Z": "both"}
    assert (info["equations"], info["endogenous"], info["exogenous"], info["regimes"]) == (2, 2, 1, 2)
    assert info["is_stable"] and info["mss_radius"] == pytest.approx(sa.mss_radius)
    check(sa.model)
    assert "Variables by type" in capsys.readouterr().out


# ----------------------------------------------------------------- report and plots
def test_report_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from msdsge.forecasting import forecast
    from msdsge.report import (Report, model_report, parameter_table, plot_decomposition, plot_forecast, plot_irf,
                               plot_probabilities)
    s = Model.from_text(MS).solve()
    data = pd.read_csv(REF / "filter_ms_data.csv").iloc[:60]
    data.index = tsm.date_range("1990Q1", 60)
    f = s.filter(data)
    pt = parameter_table(s)
    assert pt.loc["a", "regime_2"] == pytest.approx(0.1) and pt.loc["a", "switching"] == "c"
    rep = model_report(s, f, title="Test & report")
    hd = s.historical_decomposition(f)
    rep.section("Decomposition").figure(plot_decomposition(hd["x"], "x"), caption="HD of x", name="hd_x")
    fc = forecast(s, f, steps=8, draws=100)
    assert tsm.to_rise_date(fc.mean.index[0]) == "2005Q1"
    rep.figure(plot_forecast(fc, history=f.smoothed_states.iloc[-12:]), caption="Forecast", name="fc")
    rep.table(hd["x"].head(3), caption="First rows").text("End.").pagebreak()
    assert len(plot_irf(s.irf(periods=8), shock="e").axes) == 2
    assert len(plot_probabilities(f, "filtered").axes) == 2
    with pytest.raises(ValueError):
        plot_irf(s.irf(periods=4))                                  # two shocks: one must be chosen
    md = rep.write(tmp_path / "r.md").read_text()
    tex = rep.write(tmp_path / "r.tex").read_text()
    htm = rep.write(tmp_path / "r.html").read_text()
    assert "| parameter |" in md and "![HD of x](r_figures/hd_x.png)" in md and "Regime    2" in md
    assert r"\begin{tabular}" in tex and r"Test \& report" in tex and r"\includegraphics" in tex
    assert "<table>" in htm and "Test &amp; report" in htm and 'src="r_figures/fc.png"' in htm
    assert (tmp_path / "r_figures" / "irf_e.png").stat().st_size > 1000
    with pytest.raises(ValueError):
        rep.write(tmp_path / "r.docx")
    # more than 8 series: the smallest become "Other"
    big = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 11)), columns=[f"s{k}" for k in range(11)])
    ax = plot_decomposition(big).axes[0]
    assert "Other" in ax.get_legend_handles_labels()[1] and len(ax.get_legend_handles_labels()[1]) == 9
    plt.close("all")


def test_estimation_tables_in_report(tmp_path):
    from msdsge.report import Report, estimation_statistics, estimation_table
    m = Model.from_text("""var x; varexo e; parameters a; a = 0.7; varobs x;
        model; x = a*x(-1) + e; end; shocks; var e; stderr 0.5; end;
        estimated_params; a, 0.5, 0, 0.99, beta_pdf, 0.6, 0.15; end;""")
    data = m.solve().simulate(periods=150, seed=1)[["x"]]
    r = m.estimate(data)
    tab, st = estimation_table(r), estimation_statistics(r)
    assert tab.loc["a", "mode"] == pytest.approx(0.7, abs=0.15) and st.loc["observations", "value"] == 150
    text = Report("Estimation").table(st, digits=8).table(tab).render("md")
    assert "| a |" in text and "log MDD (Laplace)" in text
