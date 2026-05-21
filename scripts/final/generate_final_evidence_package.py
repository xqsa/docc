import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
FINAL_ROOT = ARTIFACTS_ROOT / "final_evidence"


LIGHTWEIGHT_FILES = [
    ("main_performance", "arac_lite_v0_7_generalization_report.md"),
    ("main_performance", "arac_lite_v0_7_generalization_summary.csv"),
    ("main_performance", "arac_lite_v0_7_generalization_run_details.csv"),
    ("main_performance", "arac_lite_v0_7_generalization_rank_summary.csv"),
    ("main_performance", "arac_lite_v0_8_mechanism_ablation_report.md"),
    ("main_performance", "arac_lite_v0_8_mechanism_ablation_summary.csv"),
    ("main_performance", "arac_lite_v0_8_mechanism_ablation_run_details.csv"),
    ("paired_robustness", "arac_lite_v0_7_generalization_robustness.csv"),
    ("paired_robustness", "arac_lite_v0_8_mechanism_ablation_robustness.csv"),
    ("paired_robustness", "arac_lite_v0_8_delta_stress_robustness.csv"),
    ("action_distribution", "arac_lite_v0_7_generalization_action_distribution.csv"),
    ("action_distribution", "arac_lite_v0_8_mechanism_ablation_action_distribution.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_run_details.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_summary.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_robustness.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_probe_metrics.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_recovery_metrics.csv"),
    ("recovery_probe", "arac_lite_v0_7_generalization_probe_metrics.csv"),
    ("recovery_probe", "arac_lite_v0_8_mechanism_ablation_probe_metrics.csv"),
    ("recovery_probe", "arac_lite_v0_8_mechanism_ablation_budget_alignment.csv"),
    ("caveats", "arac_lite_v0_8_mechanism_ablation_report.md"),
]


LARGE_AUDIT_FILES = [
    ("paired_robustness", "arac_lite_v0_7_generalization_relation_action_audit.csv"),
    ("paired_robustness", "arac_lite_v0_8_mechanism_ablation_relation_action_audit.csv"),
    ("delta_gate_stress", "arac_lite_v0_8_delta_stress_relation_action_audit.csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a lightweight final evidence package for ARAC-lite.")
    parser.add_argument(
        "--include-large-audits",
        action="store_true",
        help="Copy relation-action audit CSV files that are hundreds of MB each.",
    )
    return parser.parse_args()


def copy_artifact(category, filename):
    source = ARTIFACTS_ROOT / filename
    destination = FINAL_ROOT / category / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "missing"
    size = 0
    if source.exists():
        shutil.copy2(source, destination)
        status = "copied"
        size = source.stat().st_size
    return {
        "category": category,
        "filename": filename,
        "source": str(source.relative_to(REPO_ROOT)),
        "destination": str(destination.relative_to(REPO_ROOT)),
        "status": status,
        "bytes": size,
    }


def reference_artifact(category, filename):
    source = ARTIFACTS_ROOT / filename
    status = "referenced" if source.exists() else "missing"
    size = source.stat().st_size if source.exists() else 0
    return {
        "category": category,
        "filename": filename,
        "source": str(source.relative_to(REPO_ROOT)),
        "destination": "",
        "status": status,
        "bytes": size,
    }


def write_manifest_csv(rows):
    path = FINAL_ROOT / "MANIFEST.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "filename", "source", "destination", "status", "bytes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_caveats():
    path = FINAL_ROOT / "caveats" / "CAVEATS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# ARAC-lite Final Caveats",
                "",
                "- 日期：2026-05-21",
                "- 执行者：Codex",
                "",
                "## 结论边界",
                "",
                "- V0.7 支持 V0.6-targeted-probe 作为稳定候选：6/6 问题相对 disable-fast 为 majority non-worse，额外 FE 低于 1%。",
                "- V0.8 没有证明 targeted probe 明显优于 same-budget random probe；targeted selection 不能作为主贡献。",
                "- V0.8 delta gate stress test 显示 accept-only recovery 会显著放大 R6 bad recovery；delta gate 是当前最强机制证据。",
                "- 当前版本定位为 ARAC-lite，不是完整 ARAC、UCB 或 bandit 版本。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_manifest_markdown(rows):
    copied = [row for row in rows if row["status"] == "copied"]
    referenced = [row for row in rows if row["status"] == "referenced"]
    generated = [row for row in rows if row["status"] == "generated"]
    missing = [row for row in rows if row["status"] == "missing"]
    lines = [
        "# ARAC-lite Final Evidence Manifest",
        "",
        "- 日期：2026-05-21",
        "- 执行者：Codex",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Package Policy",
        "",
        "- 轻量 summary、robustness、action distribution、probe metrics 和报告文件复制到 `artifacts/final_evidence/`。",
        "- relation-action audit CSV 为数百 MB 级，默认只在 manifest 中引用原路径；需要完整复制时运行 `--include-large-audits`。",
        "- 该证据包不移动、不删除、不重写原始 V0.7/V0.8 产物。",
        "",
        "## Copied Files",
        "",
        "| category | file | bytes |",
        "| --- | --- | ---: |",
    ]
    for row in copied:
        lines.append(f"| {row['category']} | `{row['destination']}` | {row['bytes']} |")
    lines.extend(["", "## Referenced Large Files", "", "| category | source | bytes |", "| --- | --- | ---: |"])
    for row in referenced:
        lines.append(f"| {row['category']} | `{row['source']}` | {row['bytes']} |")
    if generated:
        lines.extend(["", "## Generated Files", "", "| category | file | bytes |", "| --- | --- | ---: |"])
        for row in generated:
            lines.append(f"| {row['category']} | `{row['destination']}` | {row['bytes']} |")
    if missing:
        lines.extend(["", "## Missing Files", "", "| category | source |", "| --- | --- |"])
        for row in missing:
            lines.append(f"| {row['category']} | `{row['source']}` |")
    lines.append("")
    path = FINAL_ROOT / "MANIFEST.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    args = parse_args()
    rows = []
    for category, filename in LIGHTWEIGHT_FILES:
        rows.append(copy_artifact(category, filename))
    for category, filename in LARGE_AUDIT_FILES:
        if args.include_large_audits:
            rows.append(copy_artifact(category, filename))
        else:
            rows.append(reference_artifact(category, filename))
    caveats_path = write_caveats()
    rows.append(
        {
            "category": "caveats",
            "filename": caveats_path.name,
            "source": "",
            "destination": str(caveats_path.relative_to(REPO_ROOT)),
            "status": "generated",
            "bytes": caveats_path.stat().st_size,
        }
    )
    csv_path = write_manifest_csv(rows)
    md_path = write_manifest_markdown(rows)
    print(f"manifest csv -> {csv_path}")
    print(f"manifest md -> {md_path}")
    print(f"copied/referenced rows -> {len(rows)}")


if __name__ == "__main__":
    main()
