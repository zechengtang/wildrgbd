import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://huggingface.co/hongchi/wildrgbd/resolve/main"

categories = {
    'bottle': ['bottle.z01', 'bottle.z02', 'bottle.zip'],
    'cup': ['cup.z01', 'cup.z02', 'cup.z03', 'cup.zip'],
    'tooth_brush': ['tooth_brush.z01', 'tooth_brush.z02', 'tooth_brush.z03', 'tooth_brush.zip'],
    'book': ['book.z01', 'book.zip'],
    'box': ['box.z01', 'box.zip'],
    'knife': ['knife.z01', 'knife.z02', 'knife.zip'],
    'remote_control': ['remote_control.z01', 'remote_control.zip'],
    'razor': ['razor.z01', 'razor.zip'],
    'keyboard': ['keyboard.z01', 'keyboard.zip'],
    'bowl': ['bowl.z01', 'bowl.z02', 'bowl.zip'],
    'scissor': ['scissor.z01', 'scissor.z02', 'scissor.z03', 'scissor.zip'],
    'kettle': ['kettle.z01', 'kettle.z02', 'kettle.zip'],
    'bucket': ['bucket.zip'],
    'pliers': ['pliers.z01', 'pliers.z02', 'pliers.zip'],
    'ball': ['ball.z01', 'ball.zip'],
    'mouse': ['mouse.z01', 'mouse.zip'],
    'handbag': ['handbag.z01', 'handbag.z02', 'handbag.zip'],
    'cellphone': ['cellphone.zip'],
    'microwave': ['microwave.z01', 'microwave.zip'],
    'hat': ['hat.z01', 'hat.z02', 'hat.zip'],
    'clock': ['clock.z01', 'clock.zip'],
    'shoe': ['shoe.z01', 'shoe.z02', 'shoe.z03', 'shoe.zip'],
    'flower_pot': ['flower_pot.z01', 'flower_pot.z02', 'flower_pot.zip'],
    'detergent': ['detergent.z01', 'detergent.z02', 'detergent.zip'],
    'backpack': ['backpack.z01', 'backpack.z02', 'backpack.zip'],
    'chair': ['chair.z01', 'chair.z02', 'chair.zip'],
    'TV': ['TV.zip'],
    'pineapple': ['pineapple.z01', 'pineapple.zip'],
    'potato': ['potato.zip'],
    'cucumber': ['cucumber.zip'],
    'apple': ['apple.zip'],
    'banana': ['banana.zip'],
    'pear': ['pear.zip'],
    'tomato': ['tomato.zip'],
    'peach': ['peach.zip'],
    'orange': ['orange.z01', 'orange.zip'],
    'carrot': ['carrot.zip'],
    'donut': ['donut.zip'],
    'cake': ['cake.zip'],
    'stuffed_toy': ['stuffed_toy.z01', 'stuffed_toy.z02', 'stuffed_toy.zip'],
    'train': ['train.zip'],
    'truck': ['truck.zip'],
    'boat': ['boat.zip'],
    'bus': ['bus.zip'],
    'plane': ['plane.zip'],
    'car': ['car.zip'],
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def require_aria2c() -> None:
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c not found. Please install aria2c first.")


def build_aria2_input(file_names: list[str], output_dir: Path) -> Path:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".aria2", dir=str(output_dir)) as f:
        for file_name in file_names:
            f.write(f"{BASE_URL}/{file_name}?download=true\n")
            f.write(f" out={file_name}\n")
        return Path(f.name)


def download_with_aria2(file_names: list[str], workers: int, output_dir: Path) -> None:
    require_aria2c()
    input_file = build_aria2_input(file_names, output_dir)
    print(f"[Download] queued {len(file_names)} files with aria2c")

    cmd = [
        "aria2c",
        "--enable-color=false",
        "--continue=true",
        "--max-tries=20",
        "--retry-wait=2",
        "--max-concurrent-downloads",
        str(workers),
        "--split=8",
        "--min-split-size=10M",
        "--summary-interval=1",
        "--input-file",
        str(input_file),
        "--dir",
        str(output_dir),
    ]

    try:
        run_cmd(cmd, cwd=output_dir)
    finally:
        input_file.unlink(missing_ok=True)


def merge_extract_cleanup(cat: str, parts: list[str], output_dir: Path) -> None:
    print(f"[Extract] {cat}")
    if len(parts) > 1:
        merged_zip = f"{cat}-single.zip"
        run_cmd(["zip", "-F", f"{cat}.zip", "--out", merged_zip], cwd=output_dir)
        run_cmd(["unzip", "-o", merged_zip], cwd=output_dir)
        run_cmd(["rm", "-f", merged_zip], cwd=output_dir)
        run_cmd(["rm", "-f", *parts], cwd=output_dir)
    else:
        run_cmd(["unzip", "-o", f"{cat}.zip"], cwd=output_dir)
        run_cmd(["rm", "-f", f"{cat}.zip"], cwd=output_dir)
    print(f"[Done] {cat}")


def extract_categories(selected_cats: list[str], extract_workers: int, output_dir: Path) -> None:
    failed_cats = set()
    with ThreadPoolExecutor(max_workers=extract_workers) as extract_executor:
        futures = {
            extract_executor.submit(merge_extract_cleanup, cat, categories[cat], output_dir): cat
            for cat in selected_cats
        }
        for future in as_completed(futures):
            cat = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed_cats.add(cat)
                print(f"[Failed Extract] {cat}: {exc}")

    if failed_cats:
        failed_desc = ", ".join(sorted(failed_cats))
        raise RuntimeError(f"Extraction failed for categories: {failed_desc}")


def run_pipeline(
    selected_cats: list[str],
    download_workers: int,
    extract_workers: int,
    download_only: bool,
    output_dir: Path,
) -> None:
    file_names = [file_name for cat in selected_cats for file_name in categories[cat]]
    print(
        f"Starting pipeline: {len(file_names)} files, "
        f"download_workers={download_workers}, extract_workers={extract_workers}, "
        f"output_dir={output_dir}"
    )

    download_with_aria2(file_names, download_workers, output_dir)
    if download_only:
        print("[Skip] --download-only enabled; skip extraction and cleanup.")
        return

    extract_categories(selected_cats, extract_workers, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="aria2c 并行下载数（默认: 32）。",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="并行解压/清理线程数（默认: 4）。",
    )
    parser.add_argument(
        "--download-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否仅下载不解压（默认: 开启）。使用 --no-download-only 可启用解压与清理。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="下载与解压输出目录（默认: 当前工作目录）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    if args.extract_workers < 1:
        raise ValueError("--extract-workers 必须 >= 1")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cat == 'all':
        selected = sorted(categories.keys())
    else:
        if args.cat not in categories:
            available = ", ".join(sorted(categories.keys()))
            raise ValueError(f"Unknown category: {args.cat}. Available: {available}")
        selected = [args.cat]

    run_pipeline(selected, args.workers, args.extract_workers, args.download_only, output_dir)


if __name__ == "__main__":
    main()
