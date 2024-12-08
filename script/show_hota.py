import os

category = ['bicycle']
domains = ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_blur', 'jpeg_compression', 'motion_blur', 'saturate', 'spatter']

for ctg in category:
    # val上的结果
    path = '/root/MeMOTR/outputs/memotr_bdd100k/val/checkpoint_10_tracker'
    path = os.path.join(path, f'{ctg}_summary.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
        second_line = lines[1].strip().split()
        first_number = float(second_line[0])
        print(first_number)
    # corruption结果
    for c in domains:
        severity = 3
        path = f'/root/MeMOTR/outputs/memotr_bdd100k/val-corruption_checkpoint_10_tracker/{c}-{severity}'
        # path = f'/root/MeMOTR/outputs/memotr_bdd100k_gam/val-corruption_checkpoint_12_tracker/{c}-{severity}'
        path = os.path.join(path, f'{ctg}_summary.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
            second_line = lines[1].strip().split()
            first_number = float(second_line[0])
            print(first_number)
