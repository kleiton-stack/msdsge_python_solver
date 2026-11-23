"""Time series utilities on top of pandas, in place of RISE's ``ts``/``rseries`` classes.

Only what the RISE examples use: dates in the RISE format ('1990Q1', '1990M3', '1990H2', '1990'),
series construction, windowing, frequency aggregation, filters (HP, Baxter-King, moving average)
and the transformations of ``ts/transform`` (Haver definitions). Series are ``pandas.Series`` or
``DataFrame`` with a ``PeriodIndex``; arithmetic operations, ``log``, ``cumsum`` and lags are those
of pandas itself.

Validated against RISE (validation/matlab/ref_tools.m): ``hpfilter``, ``aggregate``, ``transform``
and ``ma_filter``. RISE has no Baxter-King filter; ``bkfilter`` follows Baxter and King (1999).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

_FREQ = {"Q": "Q-DEC", "M": "M", "H": "2Q-DEC", "A": "Y-DEC", "Y": "Y-DEC", "": "Y-DEC"}
_PER_YEAR = {"Y": 1, "A": 1, "H": 2, "Q": 4, "M": 12, "W": 52, "D": 365}
_DATE = re.compile(r"^\s*(\d{4})\s*(?:([QqMmHh])\s*(\d{1,2}))?\s*$")

TRANSFORMS = {1: "level", 2: "pct_ch_ar", 3: "pct_ch", 4: "yoy_pct_ch", 5: "diff", 6: "yoy_diff",
              7: "log_ch_ar", 8: "log_ch", 9: "yoy_log_ch"}


def frequency(x) -> str:
    """Frequency letter ('Y', 'H', 'Q', 'M') of a RISE date, a Period or an index."""
    if isinstance(x, str):
        mt = _DATE.match(x)
        if not mt:
            raise ValueError(f"invalid date: {x!r}; use '1990', '1990H1', '1990Q1' or '1990M1'")
        return (mt.group(2) or "Y").upper()
    idx = x.index if isinstance(x, (pd.Series, pd.DataFrame)) else x
    name = idx.freqstr if hasattr(idx, "freqstr") else str(idx.freq)
    if name.startswith("2Q"):
        return "H"
    return {"Q": "Q", "M": "M", "Y": "Y", "A": "Y"}[name[0]]


def to_period(date) -> pd.Period:
    """Convert '1990Q1', '1990M3', '1990H2' or '1990' (or an integer year) to ``pandas.Period``."""
    if isinstance(date, pd.Period):
        return date
    mt = _DATE.match(str(date))
    if not mt:
        raise ValueError(f"invalid date: {date!r}; use '1990', '1990H1', '1990Q1' or '1990M1'")
    year, f, k = int(mt.group(1)), (mt.group(2) or "").upper(), int(mt.group(3) or 1)
    if f and not 1 <= k <= _PER_YEAR[f]:
        raise ValueError(f"period out of range in {date!r}")
    if f == "Q":
        return pd.Period(year=year, quarter=k, freq="Q-DEC")
    if f == "M":
        return pd.Period(year=year, month=k, freq="M")
    if f == "H":
        return pd.Period(year=year, quarter=2 * k, freq="2Q-DEC")
    return pd.Period(year=year, freq="Y-DEC")


def to_rise_date(period) -> str:
    """Write a period in the RISE format: '1990Q1', '1990M3', '1990H2', '1990'."""
    p = to_period(period)
    f = frequency(pd.PeriodIndex([p]))
    if f == "Q":
        return f"{p.year}Q{p.quarter}"
    if f == "M":
        return f"{p.year}M{p.month}"
    if f == "H":
        return f"{p.year}H{(p.quarter + 1) // 2}"
    return str(p.year)


def date_range(start, periods: int | None = None, end=None) -> pd.PeriodIndex:
    """Period index starting from a RISE date (``periods`` or ``end``)."""
    p0 = to_period(start)
    if end is not None:
        return pd.period_range(p0, to_period(end), freq=p0.freq)
    return pd.period_range(p0, periods=periods, freq=p0.freq)


def ts(start, data, names=None):
    """Series (or DataFrame, if ``data`` has more than one column) starting at ``start``.

    Equivalent to RISE's ``ts('1990Q1', data, names)``.
    """
    arr = np.asarray(data, float)
    idx = date_range(start, arr.shape[0])
    if arr.ndim == 1:
        return pd.Series(arr, index=idx, name=names if isinstance(names, str) else None)
    cols = list(names) if names is not None else [f"v{k + 1}" for k in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


def window(x, start=None, end=None):
    """Slice between two RISE dates (inclusive), like ``x('1990Q1:1995Q4')``.

    ``start`` may also be the string '1990Q1:1995Q4'.
    """
    if isinstance(start, str) and ":" in start:
        start, end = start.split(":")
    a = to_period(start) if start is not None else x.index[0]
    b = to_period(end) if end is not None else x.index[-1]
    return x.loc[a:b]


def _freq_letter(x) -> str:
    if not isinstance(x.index, pd.PeriodIndex):
        raise TypeError("the series needs a PeriodIndex (use timeseries.ts)")
    return frequency(x)


def aggregate(x, newfreq: str, method: str = "distribution"):
    """Convert to a lower frequency ('Q', 'H', 'Y').

    ``method``: 'distribution' (period mean, the RISE default), 'interpolation' (first
    observation of the period, RISE's definition), 'sum' or 'last'. Incomplete periods at the
    ends are dropped.
    """
    old, new = _freq_letter(x), newfreq.upper().replace("A", "Y")
    if _PER_YEAR[new] >= _PER_YEAR[old]:
        raise ValueError("the new frequency must be lower than that of the series")
    if _PER_YEAR[old] % _PER_YEAR[new]:
        raise ValueError(f"there is no exact conversion from {old} to {new}")
    k = _PER_YEAR[old] // _PER_YEAR[new]
    fn = {"distribution": "mean", "interpolation": "first", "sum": "sum", "last": "last"}
    if method not in fn:
        raise ValueError(f"unknown method: {method!r}; options: {sorted(fn)}")
    sub = (x.index.quarter if old in ("Q", "H") else x.index.month) - 1      # position in the year, 0-based
    if old == "H":
        sub = sub // 2
    letter = "" if new == "Y" else new
    labels = [to_period(f"{y}{letter}{s // k + 1}" if letter else str(y)) for y, s in zip(x.index.year, sub)]
    target = pd.PeriodIndex(labels, freq=labels[0].freq)
    g = x.groupby(target)
    out = getattr(g, fn[method])() if method != "sum" else g.sum(min_count=1)
    full = g.size() == k
    return out[full.to_numpy()]


def hpfilter(x, lamb: float = 1600.0):
    """Hodrick-Prescott filter. Returns (trend, cycle). ``lamb=1600`` for quarterly data."""
    def one(v):
        v = np.asarray(v, float)
        n = v.size
        if n < 3:
            raise ValueError("the HP filter needs at least 3 observations")
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
        return spsolve((sparse.eye(n) + lamb * (D.T @ D)).tocsc(), v)
    return _apply_trend(x, one)


def _apply_trend(x, one):
    if isinstance(x, pd.DataFrame):
        trend = pd.DataFrame({c: one(x[c].to_numpy()) for c in x.columns}, index=x.index)
    elif isinstance(x, pd.Series):
        trend = pd.Series(one(x.to_numpy()), index=x.index, name=x.name)
    else:
        trend = one(np.asarray(x, float))
    return trend, x - trend


def bkfilter(x, low: int = 6, high: int = 32, K: int = 12):
    """Band-pass filter of Baxter and King (1999): cycles between ``low`` and ``high`` periods.

    Symmetric moving average of 2K+1 terms with weights that sum to zero; the first K and last K
    observations are NaN. Defaults for quarterly data (6, 32, 12). Returns the cycle.
    """
    w = bk_weights(low, high, K)

    def one(v):
        v = np.asarray(v, float)
        out = np.full(v.size, np.nan)
        if v.size > 2 * K:
            out[K:v.size - K] = np.convolve(v, w, mode="valid")
        return out
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame({c: one(x[c].to_numpy()) for c in x.columns}, index=x.index)
    if isinstance(x, pd.Series):
        return pd.Series(one(x.to_numpy()), index=x.index, name=x.name)
    return one(x)


def bk_weights(low: int = 6, high: int = 32, K: int = 12) -> np.ndarray:
    """Weights of the Baxter-King filter (vector of 2K+1 terms, symmetric, zero sum)."""
    w1, w2 = 2 * np.pi / high, 2 * np.pi / low
    j = np.arange(1, K + 1)
    b = np.r_[(w2 - w1) / np.pi, (np.sin(w2 * j) - np.sin(w1 * j)) / (np.pi * j)]
    full = np.r_[b[:0:-1], b]
    return full - full.sum() / (2 * K + 1)


def ma_filter(x, q: int | None = None, extend: bool = False):
    """Centred moving average, like RISE's ``ts/ma_filter``. Returns (trend, detrended series).

    ``q`` is the number of periods on each side (default: half the annual frequency). With a
    window of 2q+1 terms, the two end points have weight 1/(4q) and the others 1/(2q) (a 2 x 2q
    moving average, which removes seasonality when 2q is the frequency). Without ``extend`` the
    first q and last q observations are lost; with ``extend`` the first and the last value of the
    trend are replicated at the ends (as RISE does).
    """
    if q is None:
        q = _PER_YEAR[_freq_letter(x)] // 2
    if q < 1:
        raise ValueError("q must be >= 1")
    w = np.r_[0.5, np.ones(2 * q - 1), 0.5] / (2 * q)

    def one(v):
        v = np.asarray(v, float)
        tr = np.convolve(v, w, mode="valid")
        return np.r_[np.full(q, tr[0]), tr, np.full(q, tr[-1])] if extend else tr
    if isinstance(x, (pd.Series, pd.DataFrame)):
        idx = x.index if extend else x.index[q:len(x) - q]
        base = x if extend else x.iloc[q:len(x) - q]
        if isinstance(x, pd.DataFrame):
            trend = pd.DataFrame({c: one(x[c].to_numpy()) for c in x.columns}, index=idx)
        else:
            trend = pd.Series(one(x.to_numpy()), index=idx, name=x.name)
        return trend, base - trend
    v = np.asarray(x, float)
    trend = one(v)
    return trend, (v if extend else v[q:v.size - q]) - trend


def transform(x, kind="level"):
    """Transformations of RISE's ``ts/transform`` (Haver definitions), by name or code 1 to 9.

    level; pct_ch_ar (compound annualised % change); pct_ch (% change over the period); yoy_pct_ch
    (% change over 12 months/4 quarters); diff; yoy_diff; log_ch_ar (annualised log change,
    in %); log_ch (log change, in %); yoy_log_ch. The initial observations without a lag are
    dropped.
    """
    kind = TRANSFORMS.get(kind, kind)
    if kind not in TRANSFORMS.values():
        raise ValueError(f"unknown transformation: {kind!r}; options: {list(TRANSFORMS.values())}")
    if kind == "level":
        return x
    n = _PER_YEAR[_freq_letter(x)]
    if kind == "pct_ch_ar":
        out = 100 * ((x / x.shift(1)) ** n - 1)
    elif kind == "pct_ch":
        out = 100 * (x / x.shift(1) - 1)
    elif kind == "yoy_pct_ch":
        out = 100 * (x / x.shift(n) - 1)
    elif kind == "diff":
        out = x - x.shift(1)
    elif kind == "yoy_diff":
        out = x - x.shift(n)
    elif kind == "log_ch_ar":
        out = 100 * n * np.log(x / x.shift(1))
    elif kind == "log_ch":
        out = 100 * np.log(x / x.shift(1))
    else:
        out = 100 * np.log(x / x.shift(n))
    return out.iloc[n if kind.startswith("yoy") else 1:]


def dummy(start, end, date, name: str | None = None) -> pd.Series:
    """Impulse dummy: 1 at ``date`` and 0 in the rest of the sample from ``start`` to ``end``."""
    idx = date_range(start, end=end)
    return pd.Series((idx == to_period(date)).astype(float), index=idx, name=name)


def step_dummy(start, end, date, name: str | None = None) -> pd.Series:
    """Step dummy: 1 from ``date`` onwards (inclusive)."""
    idx = date_range(start, end=end)
    return pd.Series((idx >= to_period(date)).astype(float), index=idx, name=name)


def collect(*series, names=None) -> pd.DataFrame:
    """Join series (or dictionaries of series) into a DataFrame aligned by date (``ts.collect``)."""
    parts = {}
    for k, s in enumerate(series):
        if isinstance(s, dict):
            parts.update(s)
        elif isinstance(s, pd.DataFrame):
            parts.update({c: s[c] for c in s.columns})
        else:
            parts[s.name if s.name is not None else f"v{k + 1}"] = s
    df = pd.DataFrame(parts)
    if names is not None:
        df.columns = list(names)
    return df
