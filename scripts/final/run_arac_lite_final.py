import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROBLEMS = ["E4", "E6", "S4", "S6", "A6", "R6"]
DEFAULT_STRESS_PROBLEMS = ["R6", "S6", "E6"]
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_TFES = [5000, 10000, 20000]
DEFAULT_CC_PASS_GROUP_FES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen ARAC-lite final evidence workflow without changing algorithm code."
    )
    parser.add_argument(
        "--stage",
        choices=("v0.7", "v0.8", "package", "all"),
        default="package",
        help="Which final workflow stage to execute.",
    )
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS))
    parser.add_argument("--stress-problems", nargs="+", default=list(DEFAULT_STRESS_PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--tfes", nargs="+", type=int, default=list(DEFAULT_TFES))
    parser.add_argument("--cc-pass-group-fes", type=int, default=DEFAULT_CC_PASS_GROUP_FES)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--include-large-audits",
        action="store_true",
        help="Copy very large relation-action audit CSV files into final_evidence.",
    )
    return parser.parse_args()


def format_values(values):
    return [str(value) for value in values]


def build_v07_command(args):
    command = [
        sys.executable,
        "scripts/generate_arac_lite_v0_7_generalization.py",
        "--problems",
        *format_values(args.problems),
        "--seeds",
        *format_values(args.seeds),
        "--tfes",
        *format_values(args.tfes),
        "--cc-pass-group-fes",
        str(args.cc_pass_group_fes),
        "--workers",
        str(max(1, int(args.workers))),
    ]
    if args.resume:
        command.append("--resume")
    return command


def build_v08_command(args):
    command = [
        sys.executable,
        "scripts/generate_arac_lite_v0_8_mechanism_ablation.py",
        "--problems",
        *format_values(args.problems),
        "--stress-problems",
        *format_values(args.stress_problems),
        "--seeds",
        *format_values(args.seeds),
        "--tfes",
        *format_values(args.tfes),
        "--cc-pass-group-fes",
        str(args.cc_pass_group_fes),
        "--workers",
        str(max(1, int(args.workers))),
    ]
    if args.resume:
        command.append("--resume")
    return command


def build_package_command(args):
    command = [sys.executable, "scripts/final/generate_final_evidence_package.py"]
    if args.include_large_audits:
        command.append("--include-large-audits")
    return command


def selected_commands(args):
    if args.stage == "v0.7":
        return [build_v07_command(args)]
    if args.stage == "v0.8":
        return [build_v08_command(args)]
    if args.stage == "package":
        return [build_package_command(args)]
    return [build_v07_command(args), build_v08_command(args), build_package_command(args)]


def main():
    args = parse_args()
    commands = selected_commands(args)
    for command in commands:
        printable = " ".join(command)
        print(printable)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
