import numpy as np
from numpy import array
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def main():
    configs = list()
    accs = list()
    with open('configrecord.csv') as f:
        for line in f:
            line = line[line.index("array"):]
            acc = float(line[line.index(')')+2:])
            config = line[:line.index(')')+1]
            config = eval(config)
            configs.append(config)
            accs.append(acc)
    np.random.seed(0)
    assert len(configs) == len(accs)
    shuffle = np.random.permutation(len(configs))
    configs = [configs[idx] for idx in shuffle]
    accs = [accs[idx] for idx in shuffle]

    orig_len = len(configs)
    test_idx = int((1.0-0.3) * orig_len)
    train_configs = configs[:test_idx]
    train_accs = accs[:test_idx]
    test_configs = configs[test_idx:]
    test_accs = accs[test_idx:]
    val_idx = int((1.0-0.1) * len(train_configs))
    val_configs = train_configs[val_idx:]
    val_accs = train_accs[val_idx:]
    train_configs = train_configs[:val_idx]
    train_accs = train_accs[:val_idx]
    print("{0:d} = {1:d} + {2:d} + {3:d}".format(orig_len, len(train_configs), len(val_configs), len(test_configs)))
    assert orig_len == len(train_configs) + len(val_configs) + len(test_configs)

    dtrain = xgb.DMatrix(train_configs, label=train_accs)
    dval = xgb.DMatrix(val_configs, label=val_accs)
    dtest = xgb.DMatrix(test_configs, label=test_accs) 
    param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'reg:linear' }
    bst = xgb.train(param, dtrain, 1000, evals=[(dval, 'val')], early_stopping_rounds=10)

    preds = bst.predict(dtest)
    for i in range(0, len(preds)):
        print(preds[i], test_accs[i])
    plt.scatter(preds, test_accs) 
    plt.ylabel('true accuracy')
    plt.xlabel('predicted accuracy')
    plt.savefig('out.pdf')
    
if __name__ == '__main__':
    main()
