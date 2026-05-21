from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


BASELINE_NAME = "Large-Scale-Overlapping-Optimization"
DEFAULT_METHOD = "CBCCO"
DEFAULT_TEST_ROUNDS = 100
CMAES_REPO = "https://github.com/CMA-ES/c-cmaes"
CMAES_COMMIT = "4450d3deccf2aacb6aa955d8216cfc4461699c60"


@dataclass(frozen=True)
class Toolchain:
    cxx: str
    cc: str


@dataclass(frozen=True)
class CompilePlan:
    baseline_root: Path
    cmaes_root: Path
    output_path: Path
    object_dir: Path
    cxx_sources: tuple[Path, ...]
    c_sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    cxx_flags: tuple[str, ...]
    c_flags: tuple[str, ...]
    link_flags: tuple[str, ...]

    def all_inputs(self) -> tuple[Path, ...]:
        headers = tuple(self.baseline_root.glob("*.h")) + tuple(self.cmaes_root.glob("*.h"))
        return (
            self.cxx_sources
            + self.c_sources
            + headers
            + (
                Path(__file__).resolve(),
            )
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def baseline_root(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "external_baselines" / BASELINE_NAME


def cmaes_root(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "external_baselines" / "third_party" / "c-cmaes" / "src"


def default_output_path(root: Path | None = None) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    base = root or repo_root()
    return base / "external_baselines" / "build" / f"lsgo_cbcco{suffix}"


def default_object_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_objects"


def default_run_root(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "tmp" / "external_cpp_baseline_runs"


def function_cpp_sources(root: Path) -> tuple[Path, ...]:
    return tuple(root / f"F{function_id}.cpp" for function_id in range(1, 13))


def build_compile_plan(
    baseline: Path,
    cmaes: Path,
    output_path: Path,
    object_dir: Path | None = None,
) -> CompilePlan:
    cxx_sources = (
        baseline / "Benchmarks.cpp",
        baseline / "CBOCC.cpp",
        baseline / "CBOG_CBD.cpp",
        baseline / "CMAESO.cpp",
        *function_cpp_sources(baseline),
    )
    c_sources = (
        cmaes / "cmaes.c",
        cmaes / "boundary_transformation.c",
    )
    include_dirs = (baseline, cmaes)

    for path in (*cxx_sources, *c_sources, *include_dirs):
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline build input: {path}")

    return CompilePlan(
        baseline_root=baseline,
        cmaes_root=cmaes,
        output_path=output_path,
        object_dir=object_dir or default_object_dir(output_path),
        cxx_sources=tuple(cxx_sources),
        c_sources=tuple(c_sources),
        include_dirs=include_dirs,
        cxx_flags=("-std=c++17", "-O2", "-include", "algorithm"),
        c_flags=("-O2",),
        link_flags=("-lm",),
    )


def detect_toolchain(cxx_override: str | None = None, cc_override: str | None = None) -> Toolchain:
    if cxx_override and cc_override:
        return Toolchain(cxx=cxx_override, cc=cc_override)

    candidates: list[Toolchain] = []
    if cxx_override:
        candidates.append(Toolchain(cxx=cxx_override, cc=cc_override or "gcc"))
        candidates.append(Toolchain(cxx=cxx_override, cc=cc_override or "clang"))
    elif cc_override:
        candidates.append(Toolchain(cxx="g++", cc=cc_override))
        candidates.append(Toolchain(cxx="clang++", cc=cc_override))
    else:
        candidates.extend(
            (
                Toolchain(cxx="g++", cc="gcc"),
                Toolchain(cxx="clang++", cc="clang"),
            )
        )

    for candidate in candidates:
        if shutil.which(candidate.cxx) and shutil.which(candidate.cc):
            return candidate

    raise RuntimeError(
        "Could not find a supported C/C++ toolchain. "
        "Install gcc/g++ (or clang/clang++) or pass --cc/--cxx explicitly."
    )


def object_path(object_dir: Path, source_path: Path) -> Path:
    return object_dir / f"{source_path.name}.o"


def compile_commands(plan: CompilePlan, toolchain: Toolchain) -> list[list[str]]:
    include_args: list[str] = []
    for include_dir in plan.include_dirs:
        include_args.extend(("-I", str(include_dir)))

    commands: list[list[str]] = []
    for source_path in plan.cxx_sources:
        commands.append(
            [
                toolchain.cxx,
                *plan.cxx_flags,
                *include_args,
                "-c",
                str(source_path),
                "-o",
                str(object_path(plan.object_dir, source_path)),
            ]
        )
    for source_path in plan.c_sources:
        commands.append(
            [
                toolchain.cc,
                *plan.c_flags,
                *include_args,
                "-c",
                str(source_path),
                "-o",
                str(object_path(plan.object_dir, source_path)),
            ]
        )
    object_paths = [
        str(object_path(plan.object_dir, source_path))
        for source_path in (*plan.cxx_sources, *plan.c_sources)
    ]
    commands.append(
        [
            toolchain.cxx,
            "-std=c++17",
            "-O2",
            *object_paths,
            "-o",
            str(plan.output_path),
            *plan.link_flags,
        ]
    )
    return commands


def needs_rebuild(plan: CompilePlan) -> bool:
    if not plan.output_path.exists():
        return True
    output_mtime = plan.output_path.stat().st_mtime
    return any(path.stat().st_mtime > output_mtime for path in plan.all_inputs())


def run_command(command: Sequence[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_executable(
    plan: CompilePlan,
    toolchain: Toolchain,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    commands = compile_commands(plan, toolchain)
    if dry_run:
        for command in commands:
            print(" ".join(command))
        return plan.output_path

    if not force and not needs_rebuild(plan):
        return plan.output_path

    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.object_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        run_command(command)
    return plan.output_path


def resolve_partition_files(
    baseline: Path,
    function_id: int,
    group_file: Path | None,
    overlap_file: Path | None,
) -> tuple[Path, Path]:
    if group_file is None and overlap_file is None:
        if function_id != 1:
            raise ValueError(
                f"No bundled partition files exist for F{function_id}. "
                "Pass --group-file and --overlap-file."
            )
        group_file = baseline / "1po.txt"
        overlap_file = baseline / "1oo.txt"
        if not group_file.exists():
            raise FileNotFoundError(f"Missing bundled group file: {group_file}")
        if not overlap_file.exists():
            raise FileNotFoundError(f"Missing bundled overlap file: {overlap_file}")
        return group_file, overlap_file

    if group_file is None or overlap_file is None:
        raise ValueError("Pass both --group-file and --overlap-file together.")

    resolved_group = group_file.resolve()
    resolved_overlap = overlap_file.resolve()
    if not resolved_group.exists():
        raise FileNotFoundError(f"Missing group file: {resolved_group}")
    if not resolved_overlap.exists():
        raise FileNotFoundError(f"Missing overlap file: {resolved_overlap}")
    return resolved_group, resolved_overlap


def resolve_weight_override(
    baseline: Path,
    weight_profile: int | None,
    weight_file: Path | None,
) -> Path | None:
    if weight_profile is not None and weight_file is not None:
        raise ValueError("Use either --weight-profile or --weight-file, not both.")
    if weight_profile is not None:
        path = baseline / f"weight{weight_profile}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing bundled weight file: {path}")
        return path
    if weight_file is not None:
        resolved = weight_file.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Missing custom weight file: {resolved}")
        return resolved
    return None


def default_run_dir(root: Path, function_id: int, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return default_run_root(root) / f"F{function_id}_seed{seed}_{timestamp}"


def prepare_run_workspace(
    baseline: Path,
    run_dir: Path,
    function_id: int,
    group_file: Path,
    overlap_file: Path,
    weight_override: Path | None = None,
) -> Path:
    if run_dir.exists():
        if any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copytree(baseline / "cdatafiles", run_dir / "cdatafiles")
    shutil.copy2(group_file, run_dir / f"{function_id}po.txt")
    shutil.copy2(overlap_file, run_dir / f"{function_id}oo.txt")
    if weight_override is not None:
        shutil.copy2(weight_override, run_dir / "cdatafiles" / f"F{function_id}-w.txt")
    return run_dir


def expected_result_file(run_dir: Path, function_id: int, seed: int) -> Path:
    return run_dir / f"{function_id}.{seed}.{DEFAULT_TEST_ROUNDS}.CBOG-CBD.result.txt"


def last_nonempty_line(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    return lines[-1] if lines else None


def run_baseline(
    executable: Path,
    run_dir: Path,
    function_id: int,
    method: str,
    seed: int,
    maxfes: int,
    dry_run: bool = False,
) -> None:
    command = [
        str(executable),
        str(function_id),
        method,
        str(seed),
        str(maxfes),
    ]
    if dry_run:
        print(" ".join(command))
        print(f"cwd={run_dir}")
        return
    run_command(command, cwd=run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run the vendored Large-Scale-Overlapping-Optimization C++ baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Compile the C++ baseline executable.")
    build_parser.add_argument("--output", type=Path, default=default_output_path())
    build_parser.add_argument("--object-dir", type=Path, default=None)
    build_parser.add_argument("--cc", default=None)
    build_parser.add_argument("--cxx", default=None)
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser("run", help="Prepare a run directory and execute the baseline.")
    run_parser.add_argument("--func", type=int, required=True, choices=range(1, 13))
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--maxfes", type=int, required=True)
    run_parser.add_argument("--method", default=DEFAULT_METHOD, choices=(DEFAULT_METHOD,))
    run_parser.add_argument("--group-file", type=Path, default=None)
    run_parser.add_argument("--overlap-file", type=Path, default=None)
    run_parser.add_argument("--weight-profile", type=int, choices=range(0, 6), default=None)
    run_parser.add_argument("--weight-file", type=Path, default=None)
    run_parser.add_argument("--run-dir", type=Path, default=None)
    run_parser.add_argument("--executable", type=Path, default=default_output_path())
    run_parser.add_argument("--object-dir", type=Path, default=None)
    run_parser.add_argument("--cc", default=None)
    run_parser.add_argument("--cxx", default=None)
    run_parser.add_argument("--rebuild", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = repo_root()
    baseline = baseline_root(root)
    cmaes = cmaes_root(root)

    if args.command == "build":
        toolchain = detect_toolchain(args.cxx, args.cc)
        output_path = args.output.resolve()
        plan = build_compile_plan(
            baseline=baseline,
            cmaes=cmaes,
            output_path=output_path,
            object_dir=args.object_dir.resolve() if args.object_dir else None,
        )
        executable = build_executable(plan, toolchain, force=args.force, dry_run=args.dry_run)
        print(executable)
        return

    toolchain = detect_toolchain(args.cxx, args.cc)
    executable = args.executable.resolve()
    plan = build_compile_plan(
        baseline=baseline,
        cmaes=cmaes,
        output_path=executable,
        object_dir=args.object_dir.resolve() if args.object_dir else None,
    )
    group_file, overlap_file = resolve_partition_files(
        baseline=baseline,
        function_id=args.func,
        group_file=args.group_file,
        overlap_file=args.overlap_file,
    )
    weight_override = resolve_weight_override(
        baseline=baseline,
        weight_profile=args.weight_profile,
        weight_file=args.weight_file,
    )
    run_dir = args.run_dir.resolve() if args.run_dir else default_run_dir(root, args.func, args.seed)
    if args.dry_run:
        if args.rebuild or not executable.exists():
            build_executable(plan, toolchain, force=args.rebuild, dry_run=True)
        print(f"run_dir={run_dir}")
        print(f"group_file={group_file}")
        print(f"overlap_file={overlap_file}")
        if weight_override is not None:
            print(f"weight_override={weight_override}")
        run_baseline(
            executable=executable,
            run_dir=run_dir,
            function_id=args.func,
            method=args.method,
            seed=args.seed,
            maxfes=args.maxfes,
            dry_run=True,
        )
        return

    executable = build_executable(plan, toolchain, force=args.rebuild, dry_run=False)
    prepare_run_workspace(
        baseline=baseline,
        run_dir=run_dir,
        function_id=args.func,
        group_file=group_file,
        overlap_file=overlap_file,
        weight_override=weight_override,
    )
    run_baseline(
        executable=executable,
        run_dir=run_dir,
        function_id=args.func,
        method=args.method,
        seed=args.seed,
        maxfes=args.maxfes,
        dry_run=args.dry_run,
    )

    result_path = expected_result_file(run_dir, args.func, args.seed)
    print(f"run_dir={run_dir}")
    print(f"result_file={result_path}")
    checkpoint = last_nonempty_line(result_path)
    if checkpoint is not None:
        print(f"last_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
