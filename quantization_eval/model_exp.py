import xgboost as xgb
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt

def configs_to_feats0(configs, high=8):
    feats = list()
    for config in configs:
        feat = list()
        feat += [a/high for a in config[0]]
        feat += [w/high for w in config[1]]
        feats.append(feat) 
    return feats

def configs_to_feats1(configs, high=8):
     feats = list()
     for config in configs:
        feat = list()
        feat += [a/high for a in config[0]]
        feat += [w/high for w in config[1]]
        feat += [(config[0][i-1] - config[0][i])/high for i in range(1, len(config[0]))]
        feat += [(config[1][i-1] - config[1][i])/high for i in range(1, len(config[1]))]
        feats.append(feat) 
     return feats

def main():
    with open('record.csv') as f:
        lines = f.readlines()
        configs = list()
        accs = list()
        for line in lines:
            net_name, config_str, acc = line.split('!')
            configs.append(eval(config_str))
            accs.append(float(acc))
        print("loaded", len(lines), "records...")
        feats0 = configs_to_feats0(configs)
        feats1 = configs_to_feats1(configs)
        print(feats0[-1])
        print(feats1[-1])
        np.random.seed(42)
        train_len = int(len(lines) * 0.8)
        val_len = (len(lines) - int(len(lines) * 0.8)) //2
        shuffle = np.random.permutation(len(lines))
        feats0 = [feats0[idx] for idx in shuffle]
        feats1 = [feats1[idx] for idx in shuffle]
        accs = [accs[idx] for idx in shuffle]
        acctest = accs[(train_len+val_len):]
        dtrain0 = xgb.DMatrix(feats0[:train_len], label=accs[:train_len])
        dtrain1 = xgb.DMatrix(feats1[:train_len], label=accs[:train_len])
        
        dval0 = xgb.DMatrix(feats0[train_len:(train_len+val_len)], label=accs[train_len:(train_len+val_len)])
        dval1 = xgb.DMatrix(feats1[train_len:(train_len+val_len)], label=accs[train_len:(train_len+val_len)])
        dtest0 = xgb.DMatrix(feats0[(train_len+val_len):], label=accs[(train_len+val_len):])
        dtest1 = xgb.DMatrix(feats1[(train_len+val_len):], label=accs[(train_len+val_len):])

        params = param = {'max_depth': 3, 'eta': 0.1, 'silent': 1}
        watchlist = [(dtrain0, 'train'), (dval0, 'valid')]
        watchlist1 = [(dtrain1, 'train'), (dval1, 'valid')]
        bst = xgb.train(params, dtrain0, 1000, evals=watchlist, early_stopping_rounds=10)
        ypred = bst.predict(dtest0)
        plt.scatter(acctest, ypred)
        plt.ylabel('predicted accuracy')
        plt.xlabel('true accuracy')
        plt.title('quantization accuracy model')
        plt.savefig('model.pdf')

        bst1 = xgb.train(params, dtrain1, 1000, evals=watchlist1, early_stopping_rounds=10)
        ypred1 = bst1.predict(dtest1)
        plt.scatter(acctest, ypred1)
        plt.ylabel('predicted accuracy')
        plt.xlabel('true accuracy')
        plt.title('quantization accuracy model')
        plt.savefig('model1.pdf')
        #for i in range(0, len(acctest)):
        #    print(ypred[i], acctest[i])
      
        

if __name__ == '__main__':
    main()
