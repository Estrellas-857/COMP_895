"""Static presentation figures from the unpooled integer-budget histograms."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .analysis import LABELS, METHODS, OUT, RF, TARGETS, WEATHERS

STYLE = {
    "raw_msp": ("#26394a", "-", "o"), "raw_entropy": ("#478fb6", (0, (5, 2)), "s"),
    "lr3": ("#9163a2", (0, (1.3, 1.5)), "^"), "lr3_class": ("#bd691c", "-", "D"),
    "lr_class_msp_interaction": ("#007e79", (0, (5, 2)), "v"), RF: ("#b52246", "-", "o"),
}
SCOPE_STYLE = {
    "Overall": ("#222e3b", "-"), "Fog": ("#008179", (0, (5, 2))),
    "Night": ("#b52246", "-"), "Rain": ("#4b75af", (0, (2, 2))),
    "Snow": ("#aa621a", (0, (6, 2, 1.5, 2))),
}


def lines(ax, coverage, values, methods=METHODS):
    markers = np.flatnonzero(np.isin(np.round(coverage, 8), TARGETS))
    for m in methods:
        color, style, marker = STYLE[m]
        ax.plot(coverage, values[m], color=color, linestyle=style, marker=marker, markevery=markers,
                ms=4.6 if m == RF else 3.7, mfc=color if m == RF else "white", mew=.9,
                lw=2.25 if m == RF else 1.5, label=LABELS[m], zorder=4 if m == RF else 2)


def format_axis(ax, lo=0, hi=100):
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(0 if lo == 0 else 10, hi + 1, 10))
    ax.grid(axis="y", color="#dae0e5", lw=.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#96a2aa")
    ax.tick_params(color="#96a2aa", labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    ax.set_axisbelow(True)


def decorate(fig, title, subtitle, handles, top, caption):
    fig.text(.075, .967, title, ha="left", va="top", fontsize=18, weight="bold", color="#192a37")
    fig.text(.075, .913, subtitle, ha="left", va="top", fontsize=10.5, color="#425664")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.068, top),
               ncol=len(handles), borderaxespad=0, frameon=False, columnspacing=1.3,
               handlelength=2.1, fontsize=9.1)
    fig.text(.075, .019, caption, ha="left", va="bottom", fontsize=8.6, color="#455863", linespacing=1.4)


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / name, dpi=260, facecolor="white")
    plt.close(fig)


def make_figures(grid, curves, weather):
    coverage = grid * 100
    groups = {"Overall": list(weather), **{w.title(): [g for g in weather if weather[g] == w] for w in WEATHERS}}
    risks = {s: {m: 100 * np.mean([curves[g, m] for g in members], axis=0) for m in METHODS}
             for s, members in groups.items()}
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 10, "axes.labelcolor": "#223443",
                                "xtick.color": "#314351", "ytick.color": "#314351", "axes.titlecolor": "#223443",
                                "axes.formatter.useoffset": False, "path.simplify": False})
    full_caption = ("Symbols mark the nine requested budgets. Curves join direct histogram evaluations at 0.1 percentage-point spacing, "
                    "with extra points below 0.1%.\nThe first budget accepts one pixel per group. Risk is undefined at zero coverage.")
    fig, ax = plt.subplots(figsize=(10.4, 6))
    fig.subplots_adjust(left=.09, right=.974, top=.785, bottom=.185)
    lines(ax, coverage, risks["Overall"])
    format_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="Valid pixels accepted (%)", ylabel="Expected accepted error (%)")
    inset = ax.inset_axes([.075, .53, .39, .405])
    subset = (grid >= .1) & (grid <= .4)
    lines(inset, coverage[subset], {m: risks["Overall"][m][subset] for m in METHODS})
    format_axis(inset, 10, 40)
    inset.set_ylim(bottom=0)
    inset.tick_params(labelsize=7)
    inset.set_title("10–40% coverage", loc="left", fontsize=8.3, pad=2)
    decorate(fig, "Overall risk–coverage comparison", "Mean of 17 held-out groups, with equal weight per group.",
             ax.lines[:6], .865, full_caption)
    save(fig, "overall_risk_coverage.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.9), sharex=True)
    fig.subplots_adjust(left=.085, right=.976, top=.795, bottom=.145, wspace=.24, hspace=.35)
    for i, (ax, (w, n)) in enumerate(zip(axes.flat, WEATHERS.items())):
        lines(ax, coverage, risks[w.title()])
        format_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{w.title()}  ·  {n} groups", loc="left", fontsize=11.5, weight="bold")
        if i % 2 == 0:
            ax.set_ylabel("Expected accepted error (%)")
        if i >= 2:
            ax.set_xlabel("Valid pixels accepted (%)")
    decorate(fig, "Risk–coverage by weather", "Equal group weights within each condition. Each panel has its own vertical scale.",
             axes.flat[0].lines[:6], .861, full_caption)
    save(fig, "weather_risk_coverage.png")

    fig, (ax, overview) = plt.subplots(1, 2, figsize=(10.8, 5.95), gridspec_kw={"width_ratios": (3.15, 1)})
    fig.subplots_adjust(left=.09, right=.97, top=.745, bottom=.19, wspace=.18)
    for panel in (ax, overview):
        panel.axhline(0, color="#52616f", lw=1.1)
    handles = []
    focus = (grid >= .1) & (grid <= .9)
    markers = np.searchsorted(grid[focus], np.asarray(TARGETS) / 100)
    for scope, (color, style) in SCOPE_STYLE.items():
        d = risks[scope][RF] - risks[scope]["raw_msp"]
        line, = ax.plot(coverage[focus], d[focus], color=color, linestyle=style, marker="o", markevery=markers,
                        mfc=color if scope == "Overall" else "white", ms=4.0,
                        lw=2.35 if scope == "Overall" else 1.75, label=scope)
        handles.append(line)
        overview.plot(coverage, d, color=color, linestyle=style, lw=1.35)
    format_axis(ax, 10, 90)
    format_axis(overview)
    overview.set_xticks([0, 50, 100])
    for panel, name in ((ax, "Requested budgets"), (overview, "Full coverage range")):
        panel.set_title(name, loc="left", fontsize=10.5, weight="bold")
        panel.set_xlabel("Valid pixels accepted (%)")
    ax.set(xlabel="Valid pixels accepted (%)", ylabel="RF minus MSP accepted error (pp)")
    decorate(fig, "When does RF accept fewer errors than MSP?",
             "Paired differences at identical per-group pixel budgets. Values below zero favor RF.",
             handles, .858, "pp = percentage points. Vertical scales differ. The full-range panel includes the one-pixel budget and the common endpoint at 100%.\n"
             "Group weights are equal within each curve. Markers show 10%–90% in ten-point increments.")
    save(fig, "rf_vs_msp.png")

    fig, axes = plt.subplots(2, 3, figsize=(12.3, 6.75), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": (2.55, 1)})
    fig.subplots_adjust(left=.081, right=.979, top=.775, bottom=.168, wspace=.15, hspace=.14)
    subset = (grid >= .1) & (grid <= .9)
    x = coverage[subset]
    methods = (METHODS[0], *METHODS[3:])
    night_upper = 104 * max(curves[g, m][subset].max() for g in groups["Night"] for m in methods)
    for i, g in enumerate(groups["Night"]):
        ax, difference = axes[:, i]
        values = {m: 100 * curves[g, m][subset] for m in methods}
        lines(ax, x, values, methods)
        format_axis(ax, 10, 90)
        ax.set_ylim(0, night_upper)
        ax.set_title(g, fontsize=11.5, loc="left", weight="bold")
        d = values[RF] - values["raw_msp"]
        difference.axhline(0, color="#314653", lw=.85)
        difference.fill_between(x, d, 0, where=d <= 0, color="#67b6ad", alpha=.65)
        difference.fill_between(x, d, 0, where=d >= 0, color="#e6a29b", alpha=.75)
        difference.plot(x, d, color="#344351", lw=1.05)
        idx = np.searchsorted(grid[subset], np.asarray(TARGETS) / 100)
        difference.scatter(x[idx], d[idx], s=12, c=np.where(d[idx] <= 0, "#00746d", "#ad3d37"), zorder=3)
        format_axis(difference, 10, 90)
        difference.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        difference.set_xlabel("Valid pixels accepted (%)")
    axes[0, 0].set_ylabel("Expected accepted error (%)")
    axes[1, 0].set_ylabel("RF − MSP (pp)")
    decorate(fig, "Night risk–coverage within each held-out group",
             "The lower panels compare RF with MSP at the same budget. Green lies below zero (RF lower), red above (RF higher).",
             axes[0, 0].lines[:4], .853,
             "Each column uses a single held-out group and its original fold models. Symbols mark the nine requested budgets.\n"
             "Lines join calculated 0.1 percentage-point grid values over 10%–90%. Boundary-bin counts are included in expectation.")
    save(fig, "night_group_coverage.png")
