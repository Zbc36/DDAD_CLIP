python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode a;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode b;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode diff_a;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode diff_b;
python main.py --config cfgs/Vin_CLIP_DDAD_SAFD.yaml --mode clip;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode cache_fusion;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode fusion;
python main.py --config cfgs/Vin_AE_3BandTopk.yaml --mode eval_all;
