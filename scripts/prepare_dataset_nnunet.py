from pathlib import Path
import os
import subprocess
import sys

import typer

app = typer.Typer(help="Prepare all MVAA datasets in nnU-Net format.")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PREPARATION_DIR = PROJECT_ROOT / "data_preparation"

TASK_SCRIPTS = {
    "ct": DATA_PREPARATION_DIR / "data_task1_CT_nnunet.py",
    "tee": DATA_PREPARATION_DIR / "data_task2_TEE_nnunet.py",
    "video": DATA_PREPARATION_DIR / "data_task3_VIDEO_nnunet.py",
}


def log_section(title):
    typer.echo()
    typer.echo("=" * 80)
    typer.secho(title, fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 80)


def log_info(message):
    typer.secho(f"[INFO] {message}", fg=typer.colors.CYAN)


def log_ok(message):
    typer.secho(f"[OK] {message}", fg=typer.colors.GREEN)


def run_command(cmd, env=None):
    log_section("Running command")
    typer.echo(" ".join(str(x) for x in cmd))
    typer.echo()

    subprocess.run(cmd, env=env, check=True)


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("dirs/data_storage/nnUNet/MVAA_nnUNET"),
        "--data-root",
        help="MVAA root directory containing reference_data/.",
    ),
    output_dir: Path = typer.Option(
        Path("dirs/data_storage/nnUNet"),
        "--output-dir",
        help="Final nnU-Net root containing nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.",
    ),
    num_processes: int = typer.Option(
        64, "--num-processes", "-np", help="Number of workers for raw writing and nnU-Net preprocessing."
    ),
    test: bool = typer.Option(False, "--test", help="Prepare only a few samples per split."),
    only: str = typer.Option("all", "--only", help="Which dataset to prepare: all, ct, tee, or video."),
):
    project_root = PROJECT_ROOT
    data_preparation_dir = DATA_PREPARATION_DIR

    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    num_processes = max(1, int(num_processes))

    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(data_preparation_dir), str(project_root)]

    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    log_section("MVAA nnU-Net dataset preparation")
    typer.echo(f"project_root         : {project_root}")
    typer.echo(f"data_preparation_dir : {data_preparation_dir}")
    typer.echo(f"data_root            : {data_root}")
    typer.echo(f"output_dir           : {output_dir}")
    typer.echo(f"num_processes        : {num_processes}")
    typer.echo(f"test                 : {test}")
    typer.echo(f"only                 : {only}")
    typer.echo(f"python               : {sys.executable}")
    typer.echo(f"PYTHONPATH           : {env['PYTHONPATH']}")

    if only == "all":
        selected_tasks = ["ct", "tee", "video"]
    else:
        only = only.lower()

        if only not in TASK_SCRIPTS:
            raise ValueError(f"Unknown --only value: {only}. Use one of: all, ct, tee, video.")

        selected_tasks = [only]

    for task_name in selected_tasks:
        script_path = TASK_SCRIPTS[task_name]

        if not script_path.exists():
            raise FileNotFoundError(f"Task script not found: {script_path}")

        task_title = {"ct": "Preparing Task 1 CT", "tee": "Preparing Task 2 TEE", "video": "Preparing Task 3 VIDEO"}[
            task_name
        ]

        log_section(task_title)

        cmd = [
            sys.executable,
            str(script_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "-np",
            str(num_processes),
        ]

        if test:
            cmd.append("--test")

        run_command(cmd, env=env)

    typer.echo()
    typer.secho("=" * 80, fg=typer.colors.GREEN, bold=True)
    typer.secho("All requested MVAA nnU-Net datasets prepared successfully.", fg=typer.colors.GREEN, bold=True)
    typer.secho("=" * 80, fg=typer.colors.GREEN, bold=True)

    log_ok(f"nnUNet_raw          : {output_dir / 'nnUNet_raw'}")
    log_ok(f"nnUNet_preprocessed : {output_dir / 'nnUNet_preprocessed'}")
    log_ok(f"nnUNet_results      : {output_dir / 'nnUNet_results'}")


if __name__ == "__main__":
    app()
