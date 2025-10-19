import argparse
import json
import sys
from pathlib import Path

from .metric import Theta3V2

def cmd_validate(args):
    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse config JSON: {exc}")
        sys.exit(1)

    schema_path = Path(args.schema) if args.schema is not None else None
    if schema_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        schema_path = repo_root / "theta3bo_schema_v2_1_0.json"
        if not schema_path.exists():
            schema_path = None

    if schema_path is not None:
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                schema = json.load(fh)
        except FileNotFoundError:
            print(f"[ERROR] Schema file not found: {schema_path}")
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Failed to parse schema JSON: {exc}")
            sys.exit(1)
    else:
        schema = {"required": ["parameters", "objectives", "batch_size", "warm_start"]}

    m = Theta3V2(cfg)
    try:
        m.validate_config(schema)
        schema_info = str(schema_path.resolve()) if schema_path is not None else "inline fallback schema"
        print(f"[OK] Config validated against {schema_info}")
    except Exception as e:
        print("[ERROR] Config validation failed:")
        print(str(e))
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(prog="theta3bo")
    sub = ap.add_subparsers(dest="cmd")

    ap_v = sub.add_parser("validate", help="Validate a config against the JSON Schema")
    ap_v.add_argument("--config", required=True)
    ap_v.add_argument("--schema", default=None)
    ap_v.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); sys.exit(2)
    args.func(args)

if __name__ == "__main__":
    main()
