#!/bin/bash
#set -e
#
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode a
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode b
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode eval
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode r
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode eval_r
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode diff_a
#python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode diff_b
python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode clip
python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode cache_fusion
python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode fusion
python main.py --config cfgs/LAG_TRIBRANCH.yaml --mode eval_all
