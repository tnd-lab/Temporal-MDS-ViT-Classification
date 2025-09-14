# Temporal MDS ViT
![temporal_mds_vit](/images/fig_vit.png)
## Prerequistes
Change configure at `src/config/classifier.py`
```
cd ml
export PYTHONPATH=`pwd`
pip install -r requirements.txt
```
## Training k_fold
```
python infer/train_kfold.py
```
## Training TemporalMDSViT
```
python infer/train.py
```

## Testing

Show confusion matrix
```
python infer/test.py
```
Eval the gradcam
```
python infer/gradcam_vit.py
