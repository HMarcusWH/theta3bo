import argparse, json, os, sys, hashlib, time
from .metric import Theta3V2

def cmd_validate(args):
    cfg = json.load(open(args.config, "r"))
    schema = json.load(open(args.schema, "r")) if args.schema else {"required":["parameters","objectives","batch_size","warm_start"]}
    m = Theta3V2(cfg)
    try:
        m.validate_config(schema)
        print("[OK] Config validated against schema")
    except Exception as e:
        print("[ERROR] Config validation failed:", e)
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
