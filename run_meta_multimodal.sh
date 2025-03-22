# Uni-modal Learning
python3 -W ignore Unimodal_train.py --modal=audio --lr=1e-2 --gpu_id=0 --dataset=KS --epoch=200 --lr_decay_epoch=150 --our_model=audio 
python3 -W ignore Unimodal_train.py --modal=video --lr=1e-2 --gpu_id=0 --dataset=KS --epoch=200 --lr_decay_epoch=150 --our_model=video 
python3 -W ignore Unimodal_train.py --modal=audio --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=200 --lr_decay_epoch=150 --our_model=audio 
python3 -W ignore Unimodal_train.py --modal=video --lr=1e-2 --gpu_id=0 --dataset=CREMA --epoch=200 --lr_decay_epoch=150 --our_model=video 

# Baseline 
python3 -W ignore Baseline_train.py  --lr=1e-2 --gpu_id=3 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=base --alpha=1 --batch-size=64 
python3 -W ignore Baseline_train.py  --lr=1e-2 --gpu_id=3 --dataset=KS --epoch=150 --lr_decay_epoch=120 --our_model=base --alpha=1 --batch-size=64 

# Naive Weight 
python3 -W ignore Weight_train.py  --lr=1e-2 --gpu_id=3 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=weight --fusion_method=concat
python3 -W ignore Weight_train.py  --lr=1e-2 --gpu_id=3 --dataset=KS --epoch=150 --lr_decay_epoch=120 --our_model=weight --fusion_method=concat

# MCUT
python3 -W ignore MCUT_train.py  --lr=1e-2 --gpu_id=2 --dataset=CREMA --epoch=150 --lr_decay_epoch=120 --our_model=MCUT --T=0.5 --batch-size=64 --fusion_method=concat --modulation=1 --caculate_cosine=1
python3 -W ignore MCUT_train.py  --lr=1e-2 --gpu_id=2 --dataset=KS --epoch=150 --lr_decay_epoch=120 --our_model=MCUT --T=0.5 --batch-size=64 --fusion_method=concat --modulation=1 --caculate_cosine=1


