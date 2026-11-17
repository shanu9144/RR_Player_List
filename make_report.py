"""Build the methodology report PDF from outputs/: python make_report.py"""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from pressure import config as C

OUT = C.OUT_DIR
PDF = OUT / "SupeRR_Selector_R1_Methodology_Report.pdf"
PINK, NAVY = colors.HexColor("#EA1A85"), colors.HexColor("#1B2A4A")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=16, spaceBefore=6, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=PINK, fontSize=12, spaceBefore=8, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.4, leading=11.2)
CAP = ParagraphStyle("C", parent=SMALL, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
TITLE = ParagraphStyle("T", parent=ss["Title"], textColor=NAVY, fontSize=22, leading=27)
BOX = ParagraphStyle("X", parent=BODY, backColor=colors.HexColor("#FDEFF6"), borderColor=PINK, borderWidth=0.6,
                     borderPadding=6, spaceBefore=6, spaceAfter=10)


def P(text, st=BODY):
    return Paragraph(text, st)


def bullets(items, st=BODY):
    return [Paragraph(f"&bull;&nbsp;{t}", ParagraphStyle("li", parent=st, leftIndent=10, firstLineIndent=-8)) for t in items]


def img(name, width=16.5, caption=None):
    path = OUT / name
    w, h = ImageReader(str(path)).getSize()
    flow = [Image(str(path), width=width * cm, height=width * cm * h / w)]
    if caption:
        flow.append(P(caption, CAP))
    return KeepTogether(flow)


def table(df, col_widths=None, font=7.6):
    hdr = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=font, leading=font + 1.5, textColor=colors.white)
    data = [[Paragraph(str(c), hdr) for c in df.columns]] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9CED8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def f(x, d=2):
    return "-" if pd.isna(x) else f"{x:.{d}f}"


def pct(x):
    return "-" if pd.isna(x) else f"{100 * x:.0f}%"


def per_over(x):
    """Runs per ball -> runs per over, signed, plain text."""
    return "-" if pd.isna(x) else f"{6 * x:+.1f}"


def load():
    r = {k: pd.read_csv(OUT / f"05_{k}_ratings_full.csv", index_col=0) for k in ("bat", "bowl")}
    prof = {k: pd.read_csv(OUT / f"08_{k}_pick_profiles.csv", index_col=0) for k in ("bat", "bowl")}
    wp = pd.read_csv(OUT / "02_wp_model_metrics.csv", index_col=0).astype(float)
    agr = pd.read_csv(OUT / "04_pressure_agreement.csv", index_col=0).iloc[:, 0]
    bands = pd.read_csv(OUT / "04_leverage_bands.csv", index_col=0)
    lam = pd.read_csv(OUT / "03_wicket_value_by_phase.csv", index_col=0).iloc[:, 0]
    oat = {k: pd.read_csv(OUT / f"06_{k}_one_at_a_time.csv") for k in ("bat", "bowl")}
    gap = pd.read_csv(OUT / "08_rr_pressure_gaps.csv")
    val = (OUT / "01_data_validation.txt").read_text().splitlines()
    return r, prof, wp, agr, bands, lam, oat, gap, val


def best_phase(row, min_balls=30):
    cand = [(p, row[f"rv_{p}"]) for p in ("powerplay", "middle", "death")
            if row.get(f"balls_{p}", 0) >= min_balls and pd.notna(row.get(f"rv_{p}"))]
    return max(cand, key=lambda x: x[1]) if cand else (None, None)


PHASE = {"powerplay": "powerplay (overs 1-6)", "middle": "middle overs (7-15)", "death": "death overs (16-20)"}


def rr_fit_text(role, row, prof, gap):
    """Plain-English fit statement for one pick, built only from his numbers and RR's gaps."""
    pr = prof.loc[row.name]
    phase, val = best_phase(pr)
    g = gap[gap.side == ("batting" if role == "bat" else "bowling")].set_index("phase")
    parts = []
    if pr["in_rr_2026_squad"]:
        parts.append("<b>Already an RR player - keep him.</b>")
    if role == "bat":
        pos = pr["median_bat_pos"]
        slot = "an opener" if pos <= 2 else ("a top-order anchor (No. 3-4)" if pos <= 4 else "a finisher")
        parts.append(f"Usually bats as {slot}.")
        if phase and val > 0:
            parts.append(f"He is at his best under pressure in the <b>{PHASE[phase]}</b>, "
                         f"scoring about {per_over(val)} runs per over more than a typical batter.")
        if pr.get("balls_death", 0) >= 30 and pr["rv_death"] > 0:
            parts.append(f"This directly helps RR's weakest area - finishing under pressure, where RR were "
                         f"{abs(6 * g.loc['death', 'rr_gap_per_ball']):.1f} runs per over behind the league.")
        elif pr["share_powerplay"] >= 0.45:
            parts.append("RR are already strong at the top, so he adds depth rather than fixing a gap.")
        sp, pc = pr.get("rv_vs_spin"), pr.get("rv_vs_pace")
        if pd.notna(sp) and pd.notna(pc):
            better = "spin" if sp > pc else "pace"
            parts.append(f"Under pressure he handles {better} slightly better.")
    else:
        kind = "spinner" if pr["pace_or_spin"] == "Spin" else "pace bowler"
        parts.append(f"{pr['bowling_type']} {kind}.")
        if phase and val > 0:
            parts.append(f"Best under pressure in the <b>{PHASE[phase]}</b>, giving away about "
                         f"{abs(6 * val):.1f} runs per over fewer than a typical bowler.")
        fixes = []
        if pr.get("balls_powerplay", 0) >= 30 and pr["rv_powerplay"] > 0:
            fixes.append(f"the powerplay ({per_over(g.loc['powerplay', 'rr_gap_per_ball'])} runs/over for RR)")
        if pr.get("balls_death", 0) >= 30 and pr["rv_death"] > 0:
            fixes.append(f"the death overs ({per_over(g.loc['death', 'rr_gap_per_ball'])} runs/over for RR)")
        if fixes:
            parts.append("Helps RR where they struggle: " + " and ".join(fixes) + ".")
        elif pr.get("balls_middle", 0) >= 30 and pr["rv_middle"] > 0:
            parts.append("RR are already good in the middle overs, so he is extra strength rather than a fix.")
    return f"<b>{int(row['final_rank'])}. {row['name']}</b> ({row['latest_team']}) - " + " ".join(parts)


def build():
    r, prof, wp, agr, bands, lam, oat, gap, val = load()
    top = {k: r[k].sort_values("final_rank").head(10) for k in r}
    gbd = gap[(gap.side == "batting") & (gap.phase == "death")].rr_gap_per_ball.iat[0]
    gwp = gap[(gap.side == "bowling") & (gap.phase == "powerplay")].rr_gap_per_ball.iat[0]
    S = []

    # ---------- Cover + summary
    S += [P("SupeRR Selector Season 2 - Round 1", ParagraphStyle("k", parent=H2, alignment=TA_CENTER)),
          P("Who Delivers When It Matters?<br/>Finding the Most Reliable Players Under Pressure", TITLE),
          P("Methodology report - IPL and T20 International ball-by-ball data, 2022-2026", CAP), Spacer(1, 6)]
    S += [P("Summary in one minute", H1)]
    S += bullets([
        "<b>I measured pressure instead of guessing it.</b> For every one of about 200,000 balls, I worked out how much that "
        "single ball could change which team wins. The balls that could swing the game the most (top 20%) are my "
        "<b>pressure moments</b>.",
        "<b>I judged players fairly.</b> Each ball is compared with what an average player would do in exactly the same "
        "situation - same over, same score, same wickets down, same target, pace or spin.",
        "<b>I rewarded reliability, not one big night.</b> A player ranks high only if he does well under pressure, does "
        "better than his own normal level, and does it again and again across many matches.",
        "<b>I checked the answer 108 different ways.</b> I changed the settings (how pressure is defined, how strict the "
        "cut-offs are, how much recent seasons count) and ranked players by their average position across all versions.",
    ])
    rows = [[i + 1, b, bw] for i, (b, bw) in enumerate(zip(top["bat"]["name"], top["bowl"]["name"]))]
    S += [Spacer(1, 4), table(pd.DataFrame(rows, columns=["Rank", "Top 10 batters under pressure",
                                                          "Top 10 bowlers under pressure"]),
                              col_widths=[1.4 * cm, 7.3 * cm, 7.3 * cm], font=9)]
    S += [P(f"<b>What it means for RR:</b> in tight moments in IPL 2025-26, RR scored about {abs(6 * gbd):.1f} runs per over "
            f"less than other teams when batting at the death, and gave away about {abs(6 * gwp):.1f} runs per over more "
            "when bowling in the powerplay. Section 7 shows which picks fix these gaps.", BOX)]
    S.append(PageBreak())

    # ---------- Big picture
    S += [P("The big picture", H1),
          P("Read the diagram from top to bottom: six data files become one clean table (Section 1); each ball gets a "
            "pressure score (Section 2) and is compared with a typical player (Section 3); players are rated on their "
            "pressure balls and re-tested 108 ways (Sections 3-6); the picks are then matched to RR's needs (Section 7).")]
    S += [img("01_data_flow.png", 15.8, "Fig 1. From six data files to the final ranking. Each box shows the section that "
                                        "explains it; the dashed box is not used.")]

    # ---------- 1 Data
    S += [P("1. The data and how I cleaned it", H1)]
    S += bullets([
        "I used five of the six files: ball-by-ball, matches, competitions, teams and squad lists. The playing-XI file was not "
        "needed because batting position is already in the ball-by-ball data. The data covers IPL 2022-2026, "
        "international T20s (including smaller cricket nations) and two 'A-team' tournaments.",
        "I removed super overs, because they are a different kind of contest.",
        "Wides are not counted as balls faced, but no-balls are. Byes and leg-byes are not charged to the bowler. "
        "Run-outs are not counted as the bowler's wicket.",
        "For every ball I rebuilt the match situation: score, wickets, balls left, target, runs needed and the required "
        "rate. For rain-shortened games I used the revised overs and target.",
    ])
    nice = {"Balls (innings 1-2)": "Balls analysed", "Matches with balls": "Matches with ball data",
            "Matches with both innings and a recorded winner": "Completed matches with a winner",
            "Winner reconstructed correctly (ties excluded)": "Winner matched from my rebuilt scores",
            "Tied (super-over decided)": "Tied games (decided by super over)",
            "Matches without winner (no result/abandoned/missing)": "No-result / abandoned matches",
            "Legal balls per completed over == 6": "Overs with exactly 6 legal balls"}
    vrows = [(nice.get(a, a), b) for a, b in (l.split(": ", 1) for l in val)]
    S += [KeepTogether([
        P("<b>Quality check</b> - my rebuilt scores picked the same winner as the official result almost every time:"),
        table(pd.DataFrame(vrows, columns=["Check", "Result"]), col_widths=[10 * cm, 4 * cm], font=8.5)])]

    # ---------- 2 Pressure
    S += [P("2. What counts as pressure?", H1)]
    S += [P("My idea is simple: <b>a ball is under pressure when what happens on it can change who wins.</b> "
            "A dot ball when 2 runs are needed off 3 balls is huge. A dot ball when 60 are needed off 6 means very little - the "
            "game is already gone. I turned this idea into a number in three steps.")]
    S += [P("Step 1 - A win predictor", H2),
          P("I built a model that looks at the match situation and gives the batting team's chance of winning. "
            "For the first innings it uses score, wickets, balls left and the average first-innings score for that "
            "tournament and year. For a chase it uses runs needed, balls left, wickets in hand and the target. "
            "The model always obeys cricket common sense: for example, needing more runs can never increase your chances. "
            "To keep it honest, every match was scored by a version of the model that had never seen that match.")]
    S += [P("Step 2 - Match impact of every ball", H2),
          P("Match impact = the team's chance of winning <i>after</i> the ball minus the chance <i>before</i> it. "
            "A last-over six that takes a chase from 30% to 70% gives the batter +40% and the bowler -40%.")]
    S += [P("Step 3 - Pressure score", H2),
          P("Before a ball is bowled, I look at everything that could happen (dot, single, two, four, six, wicket, wide), "
            "how likely each one is, and how much each would move the win chance. The average size of that move is the "
            "<b>pressure score</b>. I scale it so that <b>1 = an average ball</b>; a score of 3 means that ball matters "
            "three times more than usual. <b>Pressure moments = the top 20% of balls by this score.</b>")]
    b2 = bands.reindex(["low", "normal", "high", "extreme"]).reset_index()
    S += [table(pd.DataFrame({
        "Type of ball": ["Low pressure", "Normal", "High pressure", "Extreme pressure"],
        "Share of all balls": ["50%", "30%", "15%", "5%"],
        "How much it matters (1 = average)": b2.mean_lev.map(lambda x: f"{x:.1f}x"),
        "Happens in a chase": b2.inn2_share.map(pct), "Happens in overs 16-20": b2.death_share.map(pct),
    }), col_widths=[3.4 * cm, 2.8 * cm, 4.4 * cm, 2.8 * cm, 3.4 * cm], font=8.2), Spacer(1, 4)]
    S += [img("04_pressure_by_over.png", 16.5, "Fig 2. Left: average pressure in each over. Right: how many balls in each "
                                               "over are high-pressure. Chases build up to the most tense finishes.")]
    S += [img("04_chase_leverage_curve.png", 11, "Fig 3. In the last 6 overs of a chase, tension is highest when the target "
                                                 "is hard but still possible. Once a chase is easy or hopeless, pressure drops.")]

    S += [P("Is the win predictor accurate?", H2)]
    acc_rows = [
        ["How close the predictions are (0 = perfect, 0.25 = coin toss)", f(wp.loc["brier", "out_of_fold"], 3),
         f(wp.loc["brier", "holdout_2026"], 3)],
        ["Ranks the eventual winner higher (50% = guessing, 100% = perfect)", pct(wp.loc["auc", "out_of_fold"]),
         pct(wp.loc["auc", "holdout_2026"])],
        ["Same closeness score - chases only", f(wp.loc["brier_inn2", "out_of_fold"], 3),
         f(wp.loc["brier_inn2", "holdout_2026"], 3)],
    ]
    S += [table(pd.DataFrame(acc_rows, columns=["Question", "All matches", "Trained 2022-25, tested on 2026"]),
                col_widths=[9.8 * cm, 2.6 * cm, 4.6 * cm], font=8.2),
          P("It is much better than a coin toss, and it still works on the 2026 season it never trained on. It is most "
            "accurate in chases, where the target makes the picture clearer.", SMALL)]
    S += [img("02_wp_calibration.png", 7.5, "Fig 4. When the model says '70% chance', the team really wins about 70% of "
                                             "the time - the dots sit close to the dashed line.")]

    S += [P("Why I think this definition is meaningful", H2)]
    S += [P("I tested the things people usually call 'pressure' - wickets falling, death overs, a collapse, a run of dot "
            "balls. On their own, none of them tells you much, because many of those situations are already decided "
            "(7 down chasing 80 off 18 balls is not pressure - it is a lost game). The one thing that really matters is "
            "<b>how close the contest still is</b>. My pressure score captures exactly that, and it also grows towards the "
            "end of a close game, which matches what players and fans feel.")]
    S += [P(f"As a second opinion, I also built a simple rule-based score from those everyday signals (required rate, "
            f"wickets, phase, collapse, dot balls, knockouts), switched on only when the game is still close. The two scores "
            f"broadly agree (correlation {f(agr['spearman_li_vs_rule'])}), and my final ranking uses both.")]
    S.append(PageBreak())

    # ---------- 3 Performance
    S += [P("3. How I measured performance", H1)]
    S += bullets([
        "<b>What would a typical player do here?</b> Using all the data, I predicted for every ball the runs and the chance "
        "of a wicket for an average player in the same situation (over, score, wickets, target, pace or spin, left- or "
        "right-handed batter, IPL or T20I).",
        "<b>A wicket is worth runs.</b> Losing a wicket early costs more than losing one at the end. From the data, a wicket "
        + ", ".join(f"in the {k} costs about {v:.0f} runs" for k, v in lam.reindex(["powerplay", "middle", "death"]).items())
        + ".",
        "<b>Extra runs vs a typical player.</b> For a batter: runs he scored minus what a typical batter would score, minus "
        "the cost of getting out. For a bowler: runs a typical bowler would concede minus what he conceded, plus the value "
        "of his wickets. Zero means 'exactly average'.",
        "<b>His own normal level.</b> I measured each player in normal moments too, so I can see whether he lifts or drops "
        "when the pressure comes on.",
    ])
    S += [P("<b>The final rating is built from four simple questions:</b>"),
          table(pd.DataFrame([
              ["Does he change results?", "30%", "Average match impact on pressure balls"],
              ["Is he good in tight moments?", "20%", "Extra runs (or runs saved) vs a typical player on pressure balls"],
              ["Does he lift his game?", "25%", "Pressure performance minus his own normal performance"],
              ["Can you count on him?", "25%", "How often his pressure games are good + how bad his bad days are"],
          ], columns=["Question", "Weight", "What I measure"]), col_widths=[4.6 * cm, 1.6 * cm, 10.8 * cm], font=8.4),
          P("Each part is put on the same scale before adding them up, so no single number dominates.", SMALL)]

    # ---------- 4 Fairness
    S += [P("4. Keeping the comparison fair", H1)]
    S += bullets([
        f"<b>Enough evidence:</b> a player needs at least {C.MIN_PRESSURE_BALLS_BAT} pressure balls across at least "
        f"{C.MIN_PRESSURE_MATCHES} matches. That leaves {len(r['bat'])} batters and {len(r['bowl'])} bowlers.",
        "<b>No lucky small samples:</b> if a player has only a few pressure balls, I pull his numbers towards the average. "
        "The more balls he has, the more his own numbers count. So a player with one hot month cannot beat a proven performer.",
        f"<b>IPL vs internationals:</b> IPL and matches between top nations count fully. Games against smaller cricket "
        f"nations count {C.W_FULL_V_ASSOC:.0%}, games between two smaller nations {C.W_ASSOC_V_ASSOC:.0%}, and A-team "
        f"tournaments {C.W_A_TEAMS:.0%}. The 'typical player' model also knows whether it is an IPL game, so the higher IPL "
        "scoring of recent years does not inflate anyone.",
        "<b>Recent form matters more:</b> 2026 counts fully, 2024 counts half and 2022 counts a quarter, because we are picking "
        "players for the future. I also tested the ranking without this.",
        "<b>Same yardstick for every role:</b> an opener in the powerplay and a finisher in the last over are each compared "
        "with a typical player in their own situation.",
    ])
    S.append(PageBreak())

    # ---------- 5 Evidence
    S += [P("5. Evidence for my top 10", H1), P("5.1 Batters", H2)]
    tb = top["bat"]
    S += [table(pd.DataFrame({
        "#": tb.final_rank, "Player": tb.name, "Pressure balls": tb.p_balls.astype(int),
        "Matches": tb.pressure_apps.astype(int),
        "Strike rate (pressure)": tb.p_sr.map(lambda x: f(x, 0)), "Strike rate (normal)": tb.n_sr.map(lambda x: f(x, 0)),
        "Extra runs / over (pressure)": tb.p_rv_raw.map(per_over),
        "Good games": tb.pos_share.map(pct), "In top 10 in": tb.top10_rate.map(pct),
    }), col_widths=[0.7 * cm, 3.6 * cm, 1.7 * cm, 1.5 * cm, 2 * cm, 2 * cm, 2.3 * cm, 1.6 * cm, 1.7 * cm], font=7.2),
          P("<b>Extra runs / over</b> = runs above a typical batter in the same situations. <b>Good games</b> = share of his "
            "pressure matches where he beat the typical batter. <b>In top 10 in</b> = share of my 108 test versions where he "
            "made the top 10.", SMALL)]
    S += [img("07_bat_pressure_vs_normal.png", 10.5, "Fig 5. Every qualified batter. Players above the dashed line do "
                                                     "better under pressure than in normal moments. Red = my top 10."),
          img("07_bat_top10.png", 12, "Fig 6. Top-10 batters: bar = actual extra runs per ball under pressure; black line = "
                                      "likely range across his good and bad matches; red diamond = fair value after the "
                                      "small-sample check (this is what the rating uses).")]
    S += [P("5.2 Bowlers", H2)]
    tw = top["bowl"]
    S += [table(pd.DataFrame({
        "#": tw.final_rank, "Player": tw.name, "Pressure balls": tw.p_balls.astype(int),
        "Matches": tw.pressure_apps.astype(int),
        "Economy (pressure)": tw.p_econ.map(f), "Economy (normal)": tw.n_econ.map(f),
        "Dot balls (pressure)": tw.p_dot_pct.map(lambda x: f"{x:.0f}%"),
        "Runs saved / over (pressure)": tw.p_rv_raw.map(per_over),
        "Good games": tw.pos_share.map(pct), "In top 10 in": tw.top10_rate.map(pct),
    }), col_widths=[0.7 * cm, 3.1 * cm, 1.6 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.2 * cm, 1.5 * cm, 1.6 * cm],
        font=7.0),
          P("<b>Runs saved / over</b> includes the value of wickets taken, compared with a typical bowler in the same "
            "situations.", SMALL)]
    S += [img("07_bowl_pressure_vs_normal.png", 10.5, "Fig 7. Every qualified bowler: pressure vs normal performance."),
          img("07_bowl_top10.png", 12, "Fig 8. Top-10 bowlers: runs saved per ball under pressure, with likely range.")]
    S += [P("<b>How I ordered the list.</b> Rank is each player's average position across all 108 versions of my method. "
            "A player who is brilliant in one version but drops out in another is placed below someone who is near the top "
            "every time - that is what 'reliable' means.")]
    S.append(PageBreak())

    # ---------- 6 Reliability
    S += [P("6. How reliable is this, and what are the limits?", H1)]
    S += bullets([
        "<b>Across matches:</b> for each player I re-ran his numbers 300 times, each time with a random mix of his matches. "
        "If his rating holds up even on his bad mixes, he is reliable. This feeds the 'Can you count on him?' part.",
        "<b>Across methods:</b> I compared my main ranking with versions where I changed one setting at a time. "
        "1.00 means the ranking stays exactly the same, 0 means it is completely different:",
    ])
    rows = []
    for k, lab in (("bat", "Batters"), ("bowl", "Bowlers")):
        g = oat[k].groupby("changed").kendall_tau.median()
        rows.append([lab] + [f(g.get(c), 2) for c in ("min_balls", "weights", "weight_mode", "pctl", "definition")])
    S += [table(pd.DataFrame(rows, columns=["", "Minimum balls", "Rating weights", "Recent-form boost",
                                            "Pressure cut-off", "Pressure definition"]),
                col_widths=[2.2 * cm] + [2.9 * cm] * 5, font=8.2), Spacer(1, 4)]
    S += [P("The ranking barely moves when I change the minimum balls or the weights. It moves more when I switch between "
            "the two pressure definitions, because they look at slightly different moments. That is exactly why my final "
            "list uses both instead of picking the one that suits a particular player.")]
    S += [P("<b>Where my method is less certain</b>", BODY)]
    S += bullets([
        "First-innings pressure is harder to read because there is no target yet.",
        "Some international players have smaller samples (about 200 pressure balls). The ranges in Figs 6 and 8 show this, "
        "and bowlers ranked 5-10 are closer together than the batters.",
        "There is no fielding or ball-tracking data, so catches, run-outs and line/length are not credited.",
        "Each ball's impact goes to the striker and the bowler; the non-striker's role is not separated.",
        "Grounds are not modelled one by one; I use the average score for each tournament and year instead.",
        "The weights given to smaller nations are my judgement. When I halved them, or made every competition count the "
        "same, 9 or 10 of my top 10 stayed the same for both batters and bowlers.",
    ])

    # ---------- 7 RR fit
    S.append(PageBreak())
    S += [P("7. How each pick makes Rajasthan Royals stronger", H1)]
    S += [P("First I looked at RR's own pressure moments in IPL 2025-26 and compared them with the rest of the league. "
            "A plus number means RR did better than other teams; a minus number means RR fell behind.")]
    gt = gap.copy()
    gt["verdict"] = gt.rr_gap_per_over.map(lambda x: "Strength" if x > 0.5 else ("Weakness" if x < -0.3 else "About average"))
    S += [table(pd.DataFrame({
        "RR's role": gt.side.str.capitalize(), "Phase": gt.phase.map(PHASE),
        "RR vs league (runs per over)": gt.rr_gap_per_ball.map(per_over), "Verdict": gt.verdict,
    }), col_widths=[2.6 * cm, 4.6 * cm, 4.6 * cm, 3.4 * cm], font=8.4), Spacer(1, 4)]
    S += [P("<b>RR's needs under pressure:</b> (1) a batter who can finish chases and close out innings, (2) a bowler who can "
            "take control in the powerplay, and (3) a bowler for the death overs. Their top-order batting in the powerplay is "
            "already a big strength.", BOX)]
    S += [P("7.1 Batters", H2)]
    for _, row in top["bat"].iterrows():
        S.append(P(rr_fit_text("bat", row, prof["bat"], gap), SMALL))
    S += [P("7.2 Bowlers", H2)]
    for _, row in top["bowl"].iterrows():
        S.append(P(rr_fit_text("bowl", row, prof["bowl"], gap), SMALL))
    S += [P("7.3 Separate batter auction watchlist", H2),
          P("This second list preserves the original performance rankings above. It includes batters with evidence of being "
            "unsold at the IPL 2026 auction, after excluding players in the supplied IPL 2026 squads, the requested names, "
            "and Pakistan players. An earlier unsold result does not confirm registration or availability for a future auction. "
            "The RR death-fit column is each player's pressure-ball run value in overs 16-20; balls shows the sample size.", SMALL)]
    auction = pd.read_csv(OUT / "09_bat_auction_watchlist.csv")
    if len(auction):
        auction_view = auction[["auction_watchlist_rank", "name", "final_rank", "p_balls", "p_rv_eb",
                                "p_sr", "p_avg", "top10_rate", "rv_death", "balls_death"]].copy()
        auction_view.columns = ["Auc.\nrank", "Batter", "Perf.\nrank", "P balls", "EB RV/ball",
                                "P SR", "P avg", "Top 10", "Death RV", "Death n"]
        for col in ("EB RV/ball", "Death RV"):
            auction_view[col] = auction_view[col].map(lambda x: f(x, 3))
        auction_view["P SR"] = auction_view["P SR"].map(lambda x: f(x, 1))
        auction_view["P avg"] = auction_view["P avg"].map(lambda x: f(x, 1))
        auction_view["Top 10"] = auction_view["Top 10"].map(pct)
        S.append(table(auction_view, col_widths=[1.1*cm, 3.2*cm, 1.2*cm, 1.3*cm, 1.5*cm,
                                                   1.3*cm, 1.3*cm, 1.3*cm, 1.4*cm, 1.2*cm], font=7.1))
    S += [P("7.4 Separate bowler auction watchlist", H2),
          P("This list applies the same availability screen to bowlers. Players in the supplied IPL 2026 squads and "
            "Pakistan players are excluded. Overall rank and stability remain the primary quality signals; death-over "
            "value and its ball count help assess RR's death-bowling gap. Past unsold status does not confirm entry into "
            "the 2027 auction. The IPL 2027 retention announcement is pending at this report's data cutoff.", SMALL)]
    bowl_auction = pd.read_csv(OUT / "09_bowl_auction_watchlist.csv")
    if len(bowl_auction):
        bowl_view = bowl_auction[["auction_watchlist_rank", "name", "final_rank", "p_balls", "p_rv_eb",
                                  "p_econ", "p_sr_bowl", "top10_rate", "rv_death", "balls_death"]].copy()
        bowl_view.columns = ["Auc.\nrank", "Bowler", "Perf.\nrank", "P balls", "EB RV/ball",
                             "P econ", "P SR", "Top 10", "Death RV", "Death n"]
        for col in ("EB RV/ball", "Death RV"):
            bowl_view[col] = bowl_view[col].map(lambda x: f(x, 3))
        for col in ("P econ", "P SR"):
            bowl_view[col] = bowl_view[col].map(lambda x: f(x, 1))
        bowl_view["Top 10"] = bowl_view["Top 10"].map(pct)
        S.append(table(bowl_view, col_widths=[1.1*cm, 3.1*cm, 1.2*cm, 1.3*cm, 1.5*cm,
                                                 1.3*cm, 1.3*cm, 1.3*cm, 1.4*cm, 1.2*cm], font=7.1))
    S += [P("Auction shortlist reading", H2)]
    batting_lead = auction.iloc[0] if len(auction) else None
    bowling_lead = bowl_auction.iloc[0] if len(bowl_auction) else None
    if batting_lead is not None:
        S.append(P(f"<b>Batter:</b> {batting_lead['name']} leads the provisional past-unsold shortlist "
                   f"(historical pressure rank {int(batting_lead['final_rank'])}). His death-over value is "
                   f"{f(batting_lead['rv_death'], 3)} across {int(batting_lead['balls_death'])} pressure balls, "
                   "so the overall rank should not be read as a death-finisher endorsement.", SMALL))
    if bowling_lead is not None:
        S.append(P(f"<b>Bowler:</b> {bowling_lead['name']} leads the provisional past-unsold shortlist "
                   f"(historical pressure rank {int(bowling_lead['final_rank'])}, "
                   f"{int(bowling_lead['p_balls'])} pressure balls). His death-over value uses "
                   f"{int(bowling_lead['balls_death'])} balls; compare this with the phase-specific values and samples "
                   "in the table before treating him as a death-overs solution.", SMALL))

    # ---------- Appendix
    S += [P("Appendix - how to repeat my work", H1)]
    S += bullets([
        "All code is in the <font face='Courier'>pressure/</font> folder. Run <font face='Courier'>python run_pipeline.py "
        "--refresh</font> and then <font face='Courier'>python make_report.py</font>. The notebook "
        "<font face='Courier'>analysis.ipynb</font> walks through every step.",
        f"Fixed random seed ({C.SEED}), so the results come out the same every time. Every table and chart here is saved as "
        "a file in <font face='Courier'>outputs/</font>.",
        "Technical names, for readers who want them: win predictor = gradient-boosted classifier with monotonic "
        "constraints; match impact = Win Probability Added (WPA); pressure score = Leverage Index; small-sample "
        "adjustment = empirical-Bayes shrinkage; likely range = match-level bootstrap; ranking agreement = Kendall's tau.",
        "The rankings come only from the supplied data. Player reputation was not used.",
    ], SMALL)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 1 * cm, "SupeRR Selector S2 - Round 1 - Most reliable players under pressure")
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.6 * cm, title="SupeRR Selector S2 - Round 1 Methodology")
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print("Wrote", PDF)


if __name__ == "__main__":
    build()
