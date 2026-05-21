from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_REPO = "https://github.com/Flyki/Large-Scale-Overlapping-Optimization"
BASELINE_NAME = "Large-Scale-Overlapping-Optimization"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def baseline_root() -> Path:
    return repo_root() / "external_baselines" / BASELINE_NAME


def cdata_root() -> Path:
    return baseline_root() / "cdatafiles"


def output_path() -> Path:
    return baseline_root() / "benchmark_manifest.json"


def count_scalar_values(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(",") if part.strip()]
            count += len(parts)
    return count


def read_int_vector(path: Path) -> list[int]:
    values: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            values.append(int(float(stripped)))
    return values


def infer_base_function(function_id: int) -> str:
    if 1 <= function_id <= 4:
        return "schwefel"
    if 5 <= function_id <= 8:
        return "elliptic"
    if 9 <= function_id <= 12:
        return "rastrigin"
    raise ValueError(f"Unsupported function id: {function_id}")


def infer_mode(function_id: int) -> str:
    return "conforming" if function_id % 2 == 1 else "conflicting"


def relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(repo_root().resolve()).as_posix()


def build_function_entry(function_id: int) -> dict:
    data_dir = cdata_root()
    xopt_path = data_dir / f"F{function_id}-xopt.txt"
    p_path = data_dir / f"F{function_id}-p.txt"
    s_path = data_dir / f"F{function_id}-s.txt"
    w_path = data_dir / f"F{function_id}-w.txt"
    rotation_paths = sorted(data_dir.glob(f"F{function_id}-R*.txt"))
    subgroup_sizes = read_int_vector(s_path)
    unique_rotation_sizes = sorted(
        int(path.stem.split("-R", 1)[1]) for path in rotation_paths
    )
    decision_dim = count_scalar_values(p_path)
    shift_dim = count_scalar_values(xopt_path)
    expanded_dim = sum(subgroup_sizes)
    overlap_edges = max(0, len(subgroup_sizes) - 1)
    overlap_size = 0 if overlap_edges == 0 else (expanded_dim - decision_dim) // overlap_edges

    file_paths = [xopt_path, p_path, s_path, w_path, *rotation_paths]
    return {
        "id": function_id,
        "label": f"F{function_id}",
        "base_function": infer_base_function(function_id),
        "mode": infer_mode(function_id),
        "decision_dimension": decision_dim,
        "shift_vector_dimension": shift_dim,
        "expanded_subspace_dimension": expanded_dim,
        "subcomponent_count": len(subgroup_sizes),
        "subcomponent_sizes": subgroup_sizes,
        "rotation_sizes": unique_rotation_sizes,
        "adjacent_overlap_size": overlap_size,
        "files": [relative_to_repo(path) for path in file_paths],
    }


def build_manifest(upstream_commit: str | None) -> dict:
    return {
        "name": BASELINE_NAME,
        "kind": "external_cpp_baseline",
        "source_repo": SOURCE_REPO,
        "upstream_commit": upstream_commit,
        "snapshot_root": relative_to_repo(baseline_root()),
        "examples": [
            relative_to_repo(baseline_root() / "1po.txt"),
            relative_to_repo(baseline_root() / "1oo.txt"),
        ],
        "weight_files": [
            relative_to_repo(path)
            for path in sorted(baseline_root().glob("weight*.txt"))
        ],
        "functions": [build_function_entry(function_id) for function_id in range(1, 13)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a manifest for the vendored Large-Scale-Overlapping-Optimization baseline."
    )
    parser.add_argument("--upstream-commit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.upstream_commit)
    path = output_path()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
