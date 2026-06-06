import json
import argparse
import subprocess
from pathlib import Path


def _load_pair_whitelist(config_path: Path) -> list[str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get("exchange", {}).get("pair_whitelist", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download futures data through the freqtrade CLI.")
    parser.add_argument(
        "--freqtrade-bin",
        default="freqtrade",
        help="freqtrade executable. Use 3rdparty/freqtrade/.venv/bin/freqtrade if you install the submodule locally.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/alpha101.json",
        help="Freqtrade config containing exchange.pair_whitelist.",
    )
    parser.add_argument(
        "-t",
        "--timeframes",
        nargs="+",
        default=["1h", "1d", "15m", "4h"],
        help="Timeframes passed to freqtrade download-data.",
    )
    parser.add_argument(
        "--timerange",
        default="20250101-20260330",
        help="Freqtrade timerange, for example 20250101-20260330.",
    )
    parser.add_argument("--prepend", action="store_true", help="Pass --prepend to freqtrade.")
    args = parser.parse_args()

    config_path = Path(args.config)
    pair_list = _load_pair_whitelist(config_path)
    if not pair_list:
        raise ValueError(f"No exchange.pair_whitelist found in {config_path}")

    cmd = [
        args.freqtrade_bin,
        "download-data",
        "-p",
        *pair_list,
        "-t",
        *args.timeframes,
        "--timerange",
        args.timerange,
        "-c",
        str(config_path),
        "--trading-mode",
        "futures",
    ]
    if args.prepend:
        cmd.append("--prepend")

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
