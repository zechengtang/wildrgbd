import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def download_file(file_name: str) -> None:
    url = f"{BASE_URL}/{file_name}?download=true"
    print(f"[Download] {file_name}")
    run_cmd([
        "wget",
        "--continue",
        "--tries=20",
        "--retry-connrefused",
        "--waitretry=2",
        "-O",
        file_name,
        url,
    ])


def merge_extract_cleanup(cat: str, parts: list[str]) -> None:
    if len(parts) > 1:
        merged_zip = f"{cat}-single.zip"
        run_cmd(["zip", "-F", f"{cat}.zip", "--out", merged_zip])
        run_cmd(["unzip", "-o", merged_zip])
        run_cmd(["rm", "-f", merged_zip])
        run_cmd(["rm", "-f", *parts])
    else:
        run_cmd(["unzip", "-o", f"{cat}.zip"])
        run_cmd(["rm", "-f", f"{cat}.zip"])


def download_category(cat: str, workers: int) -> None:
    parts = categories[cat]
    print(f"\n=== Processing category: {cat} ===")
    with ThreadPoolExecutor(max_workers=min(workers, len(parts))) as executor:
        futures = [executor.submit(download_file, file_name) for file_name in parts]
        for future in as_completed(futures):
            future.result()

    merge_extract_cleanup(cat, parts)
    print(f"[Done] {cat}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", type=str, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行下载线程数（默认: 4）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")

    if args.cat == 'all':
        selected = sorted(categories.keys())
    else:
        if args.cat not in categories:
            available = ", ".join(sorted(categories.keys()))
            raise ValueError(f"Unknown category: {args.cat}. Available: {available}")
        selected = [args.cat]

    for cat in selected:
        download_category(cat, args.workers)


if __name__ == "__main__":
    main()
