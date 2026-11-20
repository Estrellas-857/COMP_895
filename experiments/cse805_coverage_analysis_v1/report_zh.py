"""Render the Chinese interpretation with numerical tables read from the exported CSVs."""
from itertools import groupby

from .analysis import LABELS, METHODS, OUT, RF, TARGETS, TIE_PP, WEATHERS, read_csv


def percent_table(rows):
    header = f"{'Coverage':>8}" + "".join(f"{LABELS[m]:>18}" for m in METHODS)
    lines = [header]
    for r in rows:
        lines.append(f"{r['target_coverage_pct']:>7}%" + "".join(f"{100 * float(r[m]):>18.4f}" for m in METHODS))
    return "\n".join(lines)


def winning_targets(rows):
    targets = [int(r["target_coverage_pct"]) for r in rows if float(r["rf_minus_reference_pp"]) < -TIE_PP]
    runs = []
    for _, items in groupby(enumerate(targets), key=lambda v: v[1]-10*v[0]):
        values = [p for _, p in items]
        runs.append(str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}")
    return ", ".join(runs) or "—"


def write_report():
    overall = read_csv(OUT/"coverage_overall.csv")
    weather = read_csv(OUT/"coverage_by_weather.csv")
    pairs = read_csv(OUT/"rf_paired_differences.csv")
    group = {(r["group"], r["method"], int(r["target_coverage_pct"])):r for r in read_csv(OUT/"coverage_by_group.csv")}
    sections = ["""CSE 805　六种分割错误预测方法的风险–覆盖率比较

把一张分割图中预测错误风险最低的像素保留下来，这些像素中还有多少是错的？在 17 个留出组等权平均时，RF + Class 在 10%、20%、…、90% 九个固定预算上都取得最低的预期接受错误率。Night 给出了不同的答案。在 10% 预算下，MSP 为 0.837%，RF 为 2.332%。到 60% 时 RF 的平均错误率开始在要求的九个点中低于 MSP，但其中一个 Night 组从 10% 到 90% 始终偏向 MSP。

一、如何读这些数字

本报告用相同整数预算补充原研究的回顾性 80% 主分析。使用四种天气各 400 张 ACDC 标注图像，来自 6 个 Fog、3 个 Night、3 个 Rain、5 个 Snow 文件名分组。各方法使用对应留出折的冻结预测、相同的 3,059,663,748 个有效像素及 523,653,212 个分割错误。原 406 张验证图像独立于本分析。MSP 的错误分数为 1−MSP，Entropy 使用归一化预测熵。LR3 组合 MSP、熵和前两类概率差。LR + Class 加入预测类别，交互 LR 再加入类别与 MSP 的乘积，RF + Class 对相同的置信度和预测类别学习非线性排序。

每组按错误分数从低到高接受像素。设组内有效像素数为 N、预算比例为 q，K 沿用历史代码的 float64 运算 floor(q×N+0.5)，至少 1 个、至多 N 个。六种方法在同一组同一目标下都使用这个整数 K，实际覆盖率为 K/N。旧实现的二进制浮点在 70% 恰处于半像素时，0351、0376、0478 三组会取相邻两个等距整数中较小的一个，本次逐像素沿用。所有方法的预算相同。

原直方图把错误分数按反正弦平方根变换分成 100,000 箱。若还需要 r 个像素，边界箱共有 B 个像素，其中 E 个错误，就从该箱等概率、不放回地选取 r 个，期望增加 r×E/B 个错误。接受错误率等于“之前各箱的错误数加上该期望”除以 K。K 是实际整数预算，错误数可以是小数。同分处理与像素的真值无关，原始分数在同一箱内的精细顺序无法从计数恢复。

表中先算每组的错误率，再等权平均。总体 17 组各占 1/17，四种天气按各自组数贡献总体均值。以 100% 覆盖率作校验，六种方法在每组都接受全部有效像素，得到相同错误率。17 组等权均值为 15.290%，合并像素后的错误率为 17.115%。下文各表的错误率单位为 %，差值单位为百分点。

二、总体结果""", percent_table(overall)]
    sections.append("""RF 相对 MSP 的平均改善在九个点上为 0.572–0.943 个百分点。在 10% 时 RF 赢 12/17 组，在 50%、60%、70%、80%、90% 五个点分别赢 16/17 组。对类别 LR，均值和胜组数讲述了两件不同的事。10% 时，RF 比 LR + Class 平均低 3.031 个百分点，逐组却是 6 胜、1 平、10 负。20% 时是 5 胜、0 平、12 负。少数大幅改善的组能抵消更多组的微小劣势，尤其是 Night。80% 时，相对两版类别 LR 均为 11 胜、0 平、6 负，平均改善均约 0.143 个百分点。

三、RF 的平均错误率在哪些预算低于各对照

下表只编码九个指定预算，10–60 表示 10、20、…、60 六个已经计算的点。比较采用未四舍五入的数值。各组胜/平/负及组名见 rf_paired_differences.csv。verification.json 另给出 10.0%–90.0% 内逐 0.1 个百分点计算的有利区间，只对该网格作判断。""")
    refs = METHODS[:-1]
    lines = [f"{'Scope':<10}"+"".join(f"{LABELS[m]:>20}" for m in refs)]
    for scope in ("overall", *WEATHERS):
        runs = [winning_targets([r for r in pairs if r["scope"] == scope and r["reference"] == m]) for m in refs]
        lines.append(f"{scope.title():<10}"+"".join(f"{v:>20}" for v in runs))
    sections.extend(["\n".join(lines), """相对 MSP、熵及 LR3，Fog、Rain、Snow 的九个天气均值均偏向 RF。相对 MSP，10% 至 40% 的四个指定点以 Snow 的绝对改善最大，10% 时达 2.073 个百分点。50% 至 90% 的五个指定点以 Fog 最大，80% 时达 1.728 个百分点。若对照改为类别 LR，低覆盖率下绝对改善最大的天气是 Night。Fog 在 20%、Rain 在 10%、20%、80%、Snow 在 10%、20% 都有类别 LR 的天气均值更低。"""])
    sections.append("""四、Night 的三个留出组

Night 平均在 10%、20%、30%、40%、50% 五个预算上偏向 MSP。RF 相对 MSP 在 10% 高 1.494 个百分点，在 50% 高 0.045 个百分点。60%、70%、80%、90% 的差值依次约为 −0.251、−0.339、−0.164、−0.080 个百分点。0.1 个百分点网格上，50.7% 仍偏向 MSP，50.8% 首次偏向 RF，此后到 90.0% 的网格点均偏向 RF。这个交叉发生在三个留出组的平均曲线上。

相对 LR + Class，RF 在 Night 的 10%–60% 六个指定点和 90% 更好，70%、80% 更差。相对交互 LR，RF 仅在 80% 这个指定点更差。10% 时两版类别 LR 分别为 19.504%、18.884%，RF 为 2.332%，MSP 为 0.837%。三组的相同预算合计 74,391,824 个像素。RF 预期接受约 182.34 万个错误，MSP 约 58.09 万个，LR + Class 约 1,584.71 万个。若用这些总数相除，RF 为 2.451%，这是由组大小加权的另一个量。

逐组的 RF−MSP 变化说明了均值转向的来源。0351 在 20% 优于 MSP，到 30% 又略差 0.014 个百分点，40%–90% 六个指定点再次优于 MSP。细网格中有利的读数为 19.5%–26.6% 和 30.7%–90.0%。0376 在九个指定点、也在整个 10%–90% 的细网格上都落后于 MSP，90% 仍高 0.240 个百分点。0397 则从 42.3% 网格点起低于 MSP。因而 60%–90% 的 Night 平均优势都来自两组胜、一组负。50% 虽然也是两胜一负，0376 的劣势仍足以抵消另外两组的收益。""")
    for g in ("gopro_0351", "gopro_0376", "gopro_0397"):
        gs = [dict(target_coverage_pct=p, **{m:group[g,m,p]["accepted_error_rate"] for m in METHODS}) for p in TARGETS]
        sections.append(g + "　组内固定预算的预期接受错误率（%）\n" + percent_table(gs))
    sections.append("""五、同分与分箱对解释有多大影响

在 Night 的 10% 预算处，若只接受完整分数箱，0351、0376、0397 的 RF 实际覆盖率分别为 18.761%、3.501%、1.400%。0376 只接受最接近的整箱时，接受错误率为 6.331%，补足相同整数预算后的期望为 2.824%。同一边界分数导致覆盖率大幅跳跃，因此跨模型比较使用同预算期望。已有原始分数抽样显示，三组在 10% 边界上分数确实相等的像素权重分别为 17.55%、21.95%、23.38%。

在全部有效像素上把相邻十个原分数箱合并成一箱，再以完全相同的 K 重算。与 100,000 箱相比，10,000 箱在 225 个“总体/天气 × 对照 × 预算”的平均差值上均给出相同胜负方向。Night 的 RF−MSP 平均差值最多变化 0.0017 个百分点，单组 RF 错误率最多变化 0.0033 个百分点。其他方法有更大的数值敏感性，例如 0351 的交互 LR 在 20% 的错误率变化达 0.600 个百分点。

以原始分数排序、用与标签无关的次序拆开真同分的保存核查样本，在可共同核对的 10%、20%、50%、60%、70%、80%、90% 七个点上，Night 的 RF−MSP 平均差值与全像素主表方向一致。10% 时核查样本为 +1.521 个百分点，50% 为 +0.060，60% 为 −0.251。此项核查同时涉及抽样和同分顺序，只用来检验方向。30%、40% 仍有完整的全像素直方图结果。""")
    sections.append("""六、原 80% 操作点与课程汇报结论

逐组重建原来的“最接近 80% 的完整分数箱”，102 个组–方法组合的实际接受像素数及错误数都与原始记录一致。RF 的历史总体接受错误率为 7.522754%，同预算补充结果为 7.522778%，相差 +0.000025 个百分点。所有组和方法中，两种规则的 80% 错误率最大差异为 0.000294 个百分点。历史主分析继续使用整箱结果，其他预算和本报告的同预算 80% 结果属于补充分析。

建议汇报时用图 1 给出全局排序，用图 3 说明与 MSP 的差距怎么随预算变化，再用图 4 的三个 Night 组检验平均曲线的解释。80% 能代表本数据中 RF 的总体平均优势，却会漏掉 Night 低预算时 MSP 的优势，也会漏掉两个类别 LR 在很低预算时接受大量错误的现象。相反，若只报告 Night 的 10%，又会漏掉其与 MSP 在中段的交叉。

这次分析描述的是同一冻结分割模型对已经产生的像素预测进行排序和筛选的效果。分组来自文件名，Night 只有三个组，记录来源的独立性无法从文件名证实。上述胜负是逐组描述，未用相关像素数充当独立样本量。每个操作点使用已观察留出组的分数分布确定预算。原研究把训练组确定的 RF 阈值直接迁移到 Night 时，实际平均覆盖率为 67.73%。在新夜间路线上能达到怎样的覆盖率与错误率，仍需独立验证。

附表　各天气固定预算下的组等权接受错误率（%）""")
    for w, n in WEATHERS.items():
        sections.append(f"{w.title()}　{n} 组\n"+percent_table([r for r in weather if r["weather"] == w]))
    sections.append("""图表与复现

图 1　figures/overall_risk_coverage.png。17 组等权，插图放大 10%–40% 区域。
图 2　figures/weather_risk_coverage.png。各天气内组等权，四个面板各用适合该天气的纵轴。
图 3　figures/rf_vs_msp.png。RF−MSP，单位为百分点。零线下方表示 RF 错误率更低。右侧保留全范围。
图 4　figures/night_group_coverage.png。各 Night 组的四种方法及组内 RF−MSP 差值。绿色表示 RF 接受的错误更少。
图中连线来自直接查询直方图的密集预算点，没有平滑拟合。全范围从每组至少保留一个像素开始，到 100% 为止。零覆盖率的接受错误率无定义。10%–90% 的大标记对应四份 CSV 中的九个指定预算。

coverage_overall.csv、coverage_by_weather.csv 采用宽表，每个方法列为 0–1 的错误率。coverage_by_group.csv 给出 918 行完整分母、整数预算、期望错误数、边界箱和整箱对照。rf_paired_differences.csv 给出 225 个按组配对的差值、胜平负次数及组名。仅 target_coverage_pct 使用 0–100 的覆盖率单位，带 _pp 的差值使用百分点。平局容差为 10⁻¹⁰ 个百分点，所有比较在四舍五入前完成。

在仓库根目录用原项目的 .venv 运行下面两条命令。第一条从已保存直方图生成 CSV、PNG 和本报告。第二条独立复算整数与期望计数，验证来源、历史端点及产物的 SHA-256，并写入 verification.json。
PYTHONPATH=experiments/cse805_error_prediction_v1/runtime/packages:. MPLCONFIGDIR=/tmp/cse805-coverage-figcache .venv/bin/python -B -m experiments.cse805_coverage_analysis_v1.analysis
PYTHONPATH=experiments/cse805_error_prediction_v1/runtime/packages:. MPLCONFIGDIR=/tmp/cse805-coverage-figcache .venv/bin/python -B -m experiments.cse805_coverage_analysis_v1.verify""")
    path = OUT/"coverage_analysis_report_zh.txt"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    write_report()
