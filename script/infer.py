'''
推理val(-corruption)
'''
import os
import yaml 

domains = ['defocus_blur', 'motion_blur', 'fog', 'frost', 'brightness', 'contrast', 'jpeg_compression', 'gaussian_blur', 'spatter', 'saturate']

cfg_path = 'outputs/memotr_bdd100k/train/config.yaml'
# cfg_path = 'outputs/memotr_bdd100k_gam/train/config.yaml'
temp_save_path = 'script/temp.yaml'

for d in domains:
    with open(f"{cfg_path}", "r") as file:
        cfg = yaml.safe_load(file)
        cfg['CORRUPTION'] = d  # None
        cfg['SEVERITY'] = 3
    with open(f"{temp_save_path}", "w") as file:
        yaml.safe_dump(cfg, file, default_flow_style=False)

    cmd = f'python main.py --mode eval --data-root /root --config-path {temp_save_path} --eval-mode specific --eval-dir ./outputs/memotr_bdd100k/ --eval-model checkpoint_10.pth --eval-threads 8'
    # cmd = f'python main.py --mode eval --data-root /root --config-path {temp_save_path} --eval-mode specific --eval-dir ./outputs/memotr_bdd100k_gam/ --eval-model checkpoint_12.pth --eval-threads 8'

    try:
        os.system(cmd)
    except:
        pass