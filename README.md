# Folk from TransMIL, with some modification.

train.py
- changed some pytorch-lightning trainer args

models/TransMIL.py
- change model(check the .pt tensor shape before use it)

logs
- under Camelyon (simple resnet_50 model, with BCNB data)
- under BCNB (modified resnet_50 model by CLAM, with BCNB data)

datasets/camel_data.py
- minor modification for .pt paths

dataset_csv/bcnb
- label

BCNB/TransMIL.yaml
- parameters
I change the precision of apex to 32, somehow making it 16 will have some problems.(also fp16 = False) I think it's just make the model slower, wont change the result.

Run `nohup python -u train.py --stage='train' --config='BCNB/TransMIL.yaml' --fold=0`, can specified gpus

Current result:
simple resnet_50 model
- AUC: .48
- val_acc: .77

modified resnet_50 model
- AUC: .68
- val_acc: .75
