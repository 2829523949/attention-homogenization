import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from harness.configs.paths import ensure_d_drive_environment
ensure_d_drive_environment()
from harness import phase1a

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','acquire','regions','pixel','siglip','report'])
    args=parser.parse_args()
    if args.stage=='prepare':phase1a.prepare()
    elif args.stage=='acquire':phase1a.acquire()
    elif args.stage=='regions':phase1a.evaluate_regions()
    elif args.stage in ('pixel','siglip'):phase1a.colors(args.stage)
    else:
        from harness.phase1a_report import report
        report()
