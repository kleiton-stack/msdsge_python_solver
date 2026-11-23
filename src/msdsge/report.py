"""Minimal report (equivalent to ``rise_report``) and plots of IRFs, probabilities and decompositions.

``Report`` accumulates sections, text, tables (``DataFrame``) and figures and writes Markdown,
LaTeX or HTML, according to the file extension. Figures are saved as PNG next to the report.
``parameter_table`` and ``estimation_table`` build the usual parameter and estimation tables.

The plots use matplotlib (imported only when a plotting function is called). Colours: fixed
categorical palette, in the same order in every plot (the colour follows the series, not the
position); with more than 8 series the smallest are grouped into "Other". One axis per panel, no
dual axis.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

# categorical palette (fixed order) and neutrals
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec")       # sequential bands (light = widest interval)
INK, MUTED, GRID, SURFACE, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9992"


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False) if not matplotlib.get_backend() else None
        import matplotlib.pyplot as plt
    except ImportError as err:      # pragma: no cover
        raise ImportError("the plots need matplotlib (pip install matplotlib)") from err
    return plt


def _style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def _grid(n, ncols, width=3.4, height=2.4):
    plt = _plt()
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _x(index):
    """Positions and labels for an arbitrary index (RISE dates in a PeriodIndex)."""
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp().to_numpy()
    return np.asarray(index)


def plot_irf(irf: pd.DataFrame, variables=None, shock: str | None = None, ncols: int = 3, scale: float = 1.0):
    """IRFs in small panels, one line per regime.

    ``irf``: output of ``Solution.irf`` (index shock, regime, period) or of ``girf`` (index
    period). ``shock``: picks one shock (required if there is more than one in the table).
    """
    if isinstance(irf.index, pd.MultiIndex):
        shocks = list(irf.index.get_level_values("shock").unique())
        if shock is None:
            if len(shocks) > 1:
                raise ValueError(f"choose a shock: {shocks}")
            shock = shocks[0]
        data = irf.xs(shock, level="shock")
        regs = list(data.index.get_level_values("regime").unique())
        series = {f"regime {r}": data.xs(r, level="regime") for r in regs}
    else:
        series = {"": irf}
    names = list(variables) if variables is not None else [c for c in irf.columns if not c.startswith("prob_")]
    fig, axes = _grid(len(names), ncols)
    for ax, v in zip(axes, names):
        _style(ax, v)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        for k, (lab, df) in enumerate(series.items()):
            ax.plot(np.asarray(df.index), scale * df[v].to_numpy(), color=PALETTE[k % len(PALETTE)], linewidth=2, label=lab)
    if len(series) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(series), frameon=False, fontsize=8)
    if shock:
        fig.suptitle(f"Response to {shock}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 if len(series) > 1 else 0, 1, 0.95 if shock else 1))
    return fig


def plot_probabilities(filtration, which: str = "smoothed", regimes=None, ncols: int = 1):
    """Regime probabilities ('smoothed', 'filtered' or 'predicted'), one panel per regime."""
    df = getattr(filtration, f"{which}_probs")
    cols = list(df.columns) if regimes is None else [f"regime_{r}" for r in regimes]
    fig, axes = _grid(len(cols), ncols, width=7.0, height=1.9)
    x = _x(df.index)
    for k, (ax, c) in enumerate(zip(axes, cols)):
        _style(ax, f"Pr({c.replace('_', ' ')}), {which}")
        color = PALETTE[list(df.columns).index(c) % len(PALETTE)]
        ax.fill_between(x, 0, df[c].to_numpy(), color=color, alpha=0.25, linewidth=0)
        ax.plot(x, df[c].to_numpy(), color=color, linewidth=1.5)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def _fold(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Keep the ``limit - 1`` columns with the largest absolute contribution and sum the rest into 'Other'."""
    if df.shape[1] <= limit:
        return df
    order = df.abs().sum().sort_values(ascending=False).index
    keep = [c for c in df.columns if c in set(order[:limit - 1])]
    out = df[keep].copy()
    out["Other"] = df.drop(columns=keep).sum(axis=1)
    return out


def plot_decomposition(decomp: pd.DataFrame, title: str | None = None, colors: dict | None = None):
    """Stacked bars of the historical decomposition of a variable, with the total line.

    ``decomp``: one of the DataFrames of ``historical_decomposition`` (columns: ``init`` and shocks).
    Positive and negative contributions are stacked on opposite sides of zero. ``colors`` fixes
    the colour of each series (to keep the same colour across plots).
    """
    plt = _plt()
    df = _fold(decomp)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, title)
    pos, neg = np.zeros(len(df)), np.zeros(len(df))
    cmap = dict(colors or {})
    free = [c for c in PALETTE if c not in cmap.values()]
    for c in df.columns:
        if c not in cmap:
            cmap[c] = OTHER if c in ("Other", "init") else (free.pop(0) if free else OTHER)
        v = df[c].to_numpy()
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=base, width=0.85, color=cmap[c], label=c, linewidth=0.4, edgecolor=SURFACE,
               hatch="//" if c == "init" else None)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(x, df.sum(axis=1).to_numpy(), color=INK, linewidth=1.5, label="total")
    ax.axhline(0, color=MUTED, linewidth=0.6)
    step = max(1, len(df) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(i) for i in df.index[::step]])
    ax.legend(frameon=False, fontsize=8, ncol=min(5, df.shape[1] + 1), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def plot_forecast(result, variables=None, history: pd.DataFrame | None = None, ncols: int = 2,
                  bands=((0.05, 0.95), (0.16, 0.84))):
    """Forecast fan chart: mean and quantile bands of the draws; ``history`` adds the past."""
    names = list(variables) if variables is not None else list(result.variables)
    fig, axes = _grid(len(names), ncols, width=4.2, height=2.6)
    idx = result.mean.index
    xf = _x(idx)
    for ax, v in zip(axes, names):
        _style(ax, v)
        k = result.variables.index(v)
        if history is not None and v in history:
            ax.plot(_x(history.index), history[v].to_numpy(), color=INK, linewidth=1.5, label="observed")
        if result.draws.shape[0] > 1:
            for j, (lo, hi) in enumerate(bands):
                q = np.quantile(result.draws[:, :, k], [lo, hi], axis=0)
                ax.fill_between(xf, q[0], q[1], color=BLUES[min(j, len(BLUES) - 1)], linewidth=0,
                                label=f"{round(100 * (hi - lo))}%")
        ax.plot(xf, result.mean[v].to_numpy(), color=PALETTE[0], linewidth=2, label="mean")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# --------------------------------------------------------------------------- tables
def parameter_table(solution_or_model, params: dict | None = None) -> pd.DataFrame:
    """Value of each parameter in each regime (one column per regime)."""
    if hasattr(solution_or_model, "params"):
        model, p = solution_or_model.model, solution_or_model.params
    else:
        model, p = solution_or_model, solution_or_model.parameter_values(params)
    df = pd.DataFrame(p.T, index=pd.Index(model.parameters, name="parameter"),
                      columns=[f"regime_{r + 1}" for r in range(model.h)])
    df["switching"] = [model.spec.switching.get(n, "") for n in model.parameters]
    return df


def estimation_table(result, level: float = 0.90) -> pd.DataFrame:
    """Estimation table: prior, mode, Hessian-based standard deviation and, if there are draws, the posterior."""
    return result.summary(level)


def estimation_statistics(result) -> pd.DataFrame:
    """Estimation statistics: log posterior and likelihood at the mode, Laplace MDD, observations."""
    rows = {"log posterior (mode)": result.logpost_mode, "log likelihood (mode)": result.loglik_mode,
            "log MDD (Laplace)": result.log_mdd_laplace, "estimated parameters": float(result.mode.size)}
    data = getattr(result.problem, "data", None)
    if data is not None:
        rows["observations"] = float(len(data))
    return pd.DataFrame({"value": rows})


# --------------------------------------------------------------------------- report
def _num(x, digits):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "" if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return f"{x:.{digits}g}"
    return str(x)


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b) if a != "\\" else s.replace(a, "\0")
    return s.replace("\0", r"\textbackslash{}")


class Report:
    """Simple report. Usage::

        rep = Report("Model", author="...")
        rep.section("Parameters"); rep.table(parameter_table(sol), caption="Calibration")
        rep.figure(plot_irf(sol.irf("e")), caption="IRFs"); rep.write("output/report.md")

    The format is given by the extension in ``write`` (.md, .tex, .html).
    """

    def __init__(self, title: str, author: str | None = None, date: str | None = None):
        self.title, self.author, self.date = title, author, date
        self.items: list[tuple] = []

    def section(self, title: str, level: int = 1) -> "Report":
        self.items.append(("section", title, level))
        return self

    def subsection(self, title: str) -> "Report":
        return self.section(title, 2)

    def text(self, text: str) -> "Report":
        self.items.append(("text", text))
        return self

    def itemize(self, items) -> "Report":
        self.items.append(("list", list(items)))
        return self

    def verbatim(self, text: str) -> "Report":
        self.items.append(("verbatim", text))
        return self

    def table(self, df: pd.DataFrame, caption: str | None = None, digits: int = 4) -> "Report":
        self.items.append(("table", df, caption, digits))
        return self

    def figure(self, fig, caption: str | None = None, name: str | None = None) -> "Report":
        """``fig``: matplotlib figure or path of an already saved image."""
        self.items.append(("figure", fig, caption, name))
        return self

    def pagebreak(self) -> "Report":
        self.items.append(("pagebreak",))
        return self

    # ---- writing
    def _table_cells(self, df, digits):
        idx = df.index
        flat = idx.to_flat_index() if isinstance(idx, pd.MultiIndex) else idx
        head = [" / ".join(str(n) for n in idx.names if n) if any(idx.names) else ""] + [str(c) for c in df.columns]
        rows = [[", ".join(map(str, i)) if isinstance(i, tuple) else str(i)] + [_num(v, digits) for v in row]
                for i, row in zip(flat, df.to_numpy(dtype=object))]
        return head, rows

    def _figure_path(self, fig, name, outdir: Path, count: int) -> str:
        if isinstance(fig, (str, Path)):
            return str(fig)
        fname = f"{name or f'fig_{count:02d}'}.png"
        figdir = outdir / f"{self._stem}_figures"
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / fname, dpi=150, facecolor=fig.get_facecolor())
        return f"{figdir.name}/{fname}"

    def render(self, fmt: str = "md", outdir: Path | None = None, stem: str = "report") -> str:
        if fmt not in ("md", "tex", "html"):
            raise ValueError("format must be 'md', 'tex' or 'html'")
        self._stem = stem
        outdir = Path(outdir) if outdir is not None else Path(".")
        esc = {"md": lambda s: s, "tex": _tex_escape, "html": html.escape}[fmt]
        out, nfig = [], 0
        meta = " · ".join(x for x in (self.author, self.date) if x)
        if fmt == "md":
            out += [f"# {self.title}", ""] + ([meta, ""] if meta else [])
        elif fmt == "tex":
            out += [r"\documentclass{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage{booktabs,graphicx,float}",
                    rf"\title{{{esc(self.title)}}}", rf"\author{{{esc(self.author or '')}}}",
                    rf"\date{{{esc(self.date or '')}}}", r"\begin{document}", r"\maketitle"]
        else:
            out += ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
                    f"<title>{esc(self.title)}</title>",
                    "<style>body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;"
                    f"color:{INK};background:{SURFACE}}}table{{border-collapse:collapse;margin:1rem 0}}"
                    f"th,td{{padding:.25rem .7rem;border-bottom:1px solid {GRID};text-align:right}}"
                    "th:first-child,td:first-child{text-align:left}caption,figcaption{"
                    f"color:{MUTED};font-size:.9rem;text-align:left}}img{{max-width:100%}}</style></head><body>",
                    f"<h1>{esc(self.title)}</h1>"] + ([f"<p>{esc(meta)}</p>"] if meta else [])
        for item in self.items:
            kind = item[0]
            if kind == "section":
                _, title, level = item
                if fmt == "md":
                    out += ["#" * (level + 1) + f" {title}", ""]
                elif fmt == "tex":
                    out.append(rf"\{'sub' * (level - 1)}section{{{esc(title)}}}")
                else:
                    out.append(f"<h{level + 1}>{esc(title)}</h{level + 1}>")
            elif kind == "text":
                out += [item[1], ""] if fmt == "md" else [esc(item[1]), ""] if fmt == "tex" else [f"<p>{esc(item[1])}</p>"]
            elif kind == "list":
                if fmt == "md":
                    out += [f"- {x}" for x in item[1]] + [""]
                elif fmt == "tex":
                    out += [r"\begin{itemize}"] + [rf"\item {esc(str(x))}" for x in item[1]] + [r"\end{itemize}"]
                else:
                    out += ["<ul>"] + [f"<li>{esc(str(x))}</li>" for x in item[1]] + ["</ul>"]
            elif kind == "verbatim":
                if fmt == "md":
                    out += ["```", item[1].rstrip("\n"), "```", ""]
                elif fmt == "tex":
                    out += [r"\begin{verbatim}", item[1].rstrip("\n"), r"\end{verbatim}"]
                else:
                    out.append(f"<pre>{esc(item[1])}</pre>")
            elif kind == "table":
                _, df, caption, digits = item
                head, rows = self._table_cells(df, digits)
                if fmt == "md":
                    if caption:
                        out += [f"**{caption}**", ""]
                    out.append("| " + " | ".join(head) + " |")
                    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
                    out += ["| " + " | ".join(r) + " |" for r in rows] + [""]
                elif fmt == "tex":
                    out += [r"\begin{table}[H]\centering"] + ([rf"\caption{{{esc(caption)}}}"] if caption else [])
                    out += [r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule",
                            " & ".join(esc(x) for x in head) + r" \\", r"\midrule"]
                    out += [" & ".join(esc(x) for x in r) + r" \\" for r in rows]
                    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
                else:
                    out.append("<table>" + (f"<caption>{esc(caption)}</caption>" if caption else ""))
                    out.append("<tr>" + "".join(f"<th>{esc(x)}</th>" for x in head) + "</tr>")
                    out += ["<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>" for r in rows]
                    out.append("</table>")
            elif kind == "figure":
                _, fig, caption, name = item
                nfig += 1
                path = self._figure_path(fig, name, outdir, nfig)
                if fmt == "md":
                    out += [f"![{caption or ''}]({path})", ""] + ([f"*{caption}*", ""] if caption else [])
                elif fmt == "tex":
                    out += [r"\begin{figure}[H]\centering", rf"\includegraphics[width=\textwidth]{{{path}}}"]
                    out += ([rf"\caption{{{esc(caption)}}}"] if caption else []) + [r"\end{figure}"]
                else:
                    out.append(f'<figure><img src="{esc(path)}" alt="{esc(caption or "")}">'
                               + (f"<figcaption>{esc(caption)}</figcaption>" if caption else "") + "</figure>")
            elif kind == "pagebreak":
                out.append({"md": "\n---\n", "tex": r"\clearpage", "html": "<hr>"}[fmt])
        if fmt == "tex":
            out.append(r"\end{document}")
        elif fmt == "html":
            out.append("</body></html>")
        return "\n".join(out) + "\n"

    def write(self, path) -> Path:
        """Write the report; the format comes from the extension (.md, .tex, .html)."""
        path = Path(path)
        fmt = {".md": "md", ".tex": "tex", ".html": "html", ".htm": "html"}.get(path.suffix.lower())
        if fmt is None:
            raise ValueError("extension must be .md, .tex or .html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(fmt, path.parent, path.stem), encoding="utf-8")
        return path


def model_report(solution, filtration=None, estimation=None, title: str = "Model report",
                 irf_periods: int = 20, figures: bool = True) -> Report:
    """Standard report: counts, parameters, solution, stability, IRFs and, if available, filter and estimation."""
    import io

    from .display import check, print_solution
    m = solution.model
    rep = Report(title)
    info = check(m, solution, show=False)
    rep.section("Model")
    rep.itemize([f"{info['equations']} equations, {info['endogenous']} endogenous variables, {info['exogenous']} shocks, "
                 f"{info['parameters']} parameters, {info['regimes']} regime(s)",
                 "variables by type: " + ", ".join(f"{k} = {v}" for k, v in info["counts"].items()),
                 f"mean-square stability radius: {info['mss_radius']:.4f} "
                 f"({'stable' if info['is_stable'] else 'unstable'})"])
    rep.section("Parameters").table(parameter_table(solution), caption="Parameters by regime")
    buf = io.StringIO()
    print_solution(solution, file=buf)
    rep.section("Solution").verbatim(buf.getvalue())
    if estimation is not None:
        rep.section("Estimation")
        rep.table(estimation_statistics(estimation), caption="Statistics", digits=8)
        rep.table(estimation_table(estimation), caption="Estimated parameters")
    if figures:
        rep.section("Impulse response functions")
        irf = solution.irf(periods=irf_periods)
        for s in m.exogenous:
            rep.figure(plot_irf(irf, shock=s), caption=f"Response to a one standard deviation {s} shock", name=f"irf_{s}")
        if filtration is not None:
            rep.section("Regime probabilities")
            rep.figure(plot_probabilities(filtration), caption="Smoothed probabilities", name="probs_smoothed")
    return rep
