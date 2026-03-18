from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_TRACKED_PATHS = {
    "transcribator.json",
}
FORBIDDEN_TRACKED_PREFIXES = (
    "input/",
    "output/",
)


def test_user_artifact_paths_are_not_tracked():
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    tracked_paths = {
        line.strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    offenders = sorted(
        path
        for path in tracked_paths
        if path in FORBIDDEN_TRACKED_PATHS
        or any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)
    )

    assert offenders == []
