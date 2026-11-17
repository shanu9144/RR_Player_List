"""Report charts (saved as PNG into outputs/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calibration(calib, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect prediction")
    ax.plot(calib.mean_pred, calib.observed, "o-", label="my model")
    ax.set(xlabel="Predicted chance of winning", ylabel="How often the team actually won",
           title="Does the win predictor tell the truth?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leverage_heatmap(b, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, inn in zip(axes, (1, 2)):
        d = b[b.innings == inn]
        piv = d.pivot_table(index="wkts_before", columns="over_num", values="leverage", aggfunc="mean")
        piv = piv.loc[piv.index <= 9]
        im = ax.imshow(piv, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=3)
        ax.set(title=f"Mean Leverage Index - innings {inn}", xlabel="Over", ylabel="Wickets down")
        ax.set_xticks(range(0, 20, 2), [str(i) for i in range(1, 21, 2)])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Leverage (1 = average ball)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pressure_by_over(b, path):
    """Average pressure per over: batting first vs chasing, plus share of overs that are high-pressure."""
    g = b.groupby(["innings", "over_num"]).agg(avg=("leverage", "mean"), high=("li_pct", lambda s: (s >= 0.8).mean()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for inn, lab, c in ((1, "Batting first", "#1B2A4A"), (2, "Chasing", "#EA1A85")):
        d = g.loc[inn]
        axes[0].plot(d.index, d.avg, marker="o", c=c, label=lab)
        axes[1].bar(d.index + (-0.2 if inn == 1 else 0.2), 100 * d.high, width=0.4, color=c, label=lab)
    axes[0].axhline(1, c="grey", ls="--", lw=1)
    axes[0].text(1, 1.05, "average ball = 1", fontsize=8, color="grey")
    axes[0].set(title="Average pressure in each over", xlabel="Over", ylabel="Pressure score")
    axes[1].set(title="% of balls that are high-pressure", xlabel="Over", ylabel="% of balls")
    for ax in axes:
        ax.set_xticks(range(1, 21, 2))
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chase_li_curve(b, path):
    d = b[(b.innings == 2) & (b.over_num >= 15)].copy()
    d["rrr_bin"] = pd.cut(d.rrr, [0, 6, 8, 10, 12, 14, 18, 36])
    piv = d.pivot_table(index="rrr_bin", columns="wkts_in_hand", values="leverage", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in [2, 4, 6, 8]:
        if w in piv:
            ax.plot(piv.index.astype(str), piv[w], marker="o", label=f"{w} wkts in hand")
    ax.set(title="Last 6 overs of a chase: when is it most tense?", xlabel="Runs needed per over",
           ylabel="Pressure score (1 = average ball)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pressure_vs_normal(tab, role, path, top=10):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tab.n_rv_eb, tab.p_rv_eb, s=12, c="grey", alpha=0.6)
    t = tab.nsmallest(top, "final_rank") if "final_rank" in tab else tab.nsmallest(top, "rank")
    ax.scatter(t.n_rv_eb, t.p_rv_eb, s=40, c="crimson")
    for _, r in t.iterrows():
        ax.annotate(r["name"], (r.n_rv_eb, r.p_rv_eb), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", lw=1)
    ax.axhline(0, c="k", lw=0.5)
    ax.axvline(0, c="k", lw=0.5)
    label = "Batting" if role == "bat" else "Bowling"
    m = "Extra runs per ball" if role == "bat" else "Runs saved per ball"
    ax.set(xlabel=f"{m} vs a typical player - normal moments",
           ylabel=f"{m} vs a typical player - pressure moments",
           title=f"{label}: pressure vs normal (above the line = better under pressure)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top10_bars(tab, role, path):
    t = tab.sort_values("final_rank").head(10).iloc[::-1]
    raw = t.p_rv / t.p_w
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t["name"], raw, color="steelblue", label="actual extra runs per ball under pressure")
    ax.errorbar(raw, y, xerr=[(raw - t.boot_p10).clip(lower=0), (t.boot_p90 - raw).clip(lower=0)],
                fmt="none", ecolor="k", capsize=3, label="likely range (good vs bad days)")
    ax.scatter(t.p_rv_eb, y, marker="D", c="crimson", zorder=3, label="fair value after sample-size check")
    ax.axvline(0, c="k", lw=0.5)
    label = "batters" if role == "bat" else "bowlers"
    xl = "Extra runs scored per ball" if role == "bat" else "Runs saved per ball"
    ax.set(title=f"Top-10 {label} under pressure (vs a typical player)", xlabel=xl)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def data_flow(counts, path):
    """Overview diagram: data files -> clean table -> models -> player ratings -> report."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    navy, pink, grey, light = "#1B2A4A", "#EA1A85", "#8A93A6", "#E8ECF4"
    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(8.2, 72), aspect="equal")
    ax.axis("off")

    def box(cx, cy, w, h, title, sub="", tag="", fill="white", edge=navy, tcol=navy, dashed=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=0.9, ls="--" if dashed else "-", zorder=2))
        lines = [(title, 7.4, "bold", tcol, 1.9)]
        lines += [(s, 6.2, "normal", tcol, 1.65) for s in sub.split("\n")] if sub else []
        if tag:
            lines.append((tag, 5.8, "bold", pink, 1.5))
        y = cy + sum(ln[4] for ln in lines) / 2
        for text, fs, fw, col, lh in lines:
            ax.text(cx, y - lh / 2, text, ha="center", va="center", fontsize=fs, fontweight=fw, color=col, zorder=3)
            y -= lh

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color=grey,
                                     zorder=1, shrinkA=0, shrinkB=0))

    def note(x, y, text):
        ax.text(x, y, text, fontsize=5.4, color=grey, ha="left", va="center", style="italic")

    n = {k: f"{v:,}" for k, v in counts.items()}
    ax.text(0, 70.8, "THE SIX DATA FILES", fontsize=6, color=grey, fontweight="bold", va="center")
    files = [(7.6, "Ball-by-ball", f"{n['balls']} deliveries"), (24.5, "Matches", f"{n['matches']} matches"),
             (41.4, "Competitions", f"{n['comps']} tournaments"), (58.3, "Teams", f"{n['teams']} team entries"),
             (75.2, "Squad lists", f"{n['ctp']} entries")]
    for cx, t, s in files:
        box(cx, 66, 15.2, 7, t, s, fill=light)
    box(92.1, 66, 15.2, 7, "Playing XI", f"{n['xi']} rows\nnot used", fill="white", edge=grey, tcol=grey, dashed=True)

    box(14, 53, 25, 8.6, "Clean ball table", "situation before every ball", "Section 1")
    box(44, 53, 22, 8.6, "Match table", "year, IPL flag,\nweights, winner", "Section 1")
    box(14, 40, 25, 8.6, "Win predictor", "match impact and\npressure score", "Section 2")
    box(42, 40, 26, 8.6, "Typical-player model", "extra runs or runs saved\nper ball", "Section 3")
    box(28, 27, 38, 8.6, "Player tables", "pressure vs normal, small-sample\ncheck, consistency", "Sections 3-4")
    box(85, 27, 24, 8.6, "RR fit", "RR gaps and each\npick's fit", "Section 7")
    box(16, 14, 28, 8.6, "108 test versions", "average rank gives\nthe top 10", "Sections 5-6")
    box(50, 14, 22, 8.6, "Names + teams", "name from ball data,\nteam from teams file", "Section 5")
    box(85, 14, 26, 8.6, "Charts + PDF report", "this document", fill=navy, tcol="white")

    arrow((7.6, 62.5), (7.6, 57.3))
    arrow((24.5, 62.5), (36, 57.3))
    arrow((41.4, 62.5), (46, 57.3))
    arrow((33, 53), (26.5, 53))
    note(27.4, 55.2, "match_id")
    arrow((14, 48.7), (14, 44.3))
    arrow((20, 48.7), (36, 44.3))
    arrow((14, 35.7), (21, 31.3))
    arrow((42, 35.7), (35, 31.3))
    arrow((20, 22.7), (16, 18.3))
    arrow((30, 14), (39, 14))
    arrow((61, 14), (72, 14))
    arrow((58.3, 62.5), (58.3, 18.3))
    note(59.5, 45, "team names")
    arrow((75.2, 62.5), (75.2, 31.3))
    note(76.4, 45, "RR 2026 squad")
    arrow((85, 22.7), (85, 18.3))
    fig.savefig(path, dpi=200)
    plt.close(fig)
