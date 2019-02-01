import numpy as np
from evaluate import evaluate_standalone
from model_exp import configs_to_feats0
import xgboost as xgb
from multiprocessing import Process, Queue

class Proposer(object):
    def __init__(self):
        self.visited_configs = set()

    def propose(self):
        raise NotImplementedError

    def visited(self, config):
        return repr(config) in self.visited_configs

class RandomQuantProposer(Proposer):
    def __init__(self, choices, n_layers):
        super().__init__()
        self.choices = choices
        self.n_layers = n_layers

    def random_config(self):
        config = list()
        for i in range(0, 2):
            config.append([v for v in np.random.choice(self.choices, self.n_layers)])
        config.append([32]*self.n_layers)
        return config
 
    def propose(self):
        config = self.random_config()
        while self.visited(config):
            config = self.random_config()
        self.visited_configs.add(repr(config))
        return config

class Evaluator(object):
    def measure(self, config):
        raise NotImplementedError 

#class QuantEvaluator(Evaluator):
#    def __init__(self):
#        
#
#    def measure(self, config):    

class Tuner(object):
    def tune(batch_size=64):
        raise NotImplementedError

def process_wrapper(queue, configs):
    for config in configs:
        queue.put(evaluate_standalone(config))

class XGBTuner(Tuner):
    def __init__(self, proposer, evaluator):
        super().__init__()
        self.configs = list()
        self.accs = list()
        self.proposer = proposer
        self.evaluator = evaluator
        self.bst = None

    def refit(self):
        feats0 = configs_to_feats0(self.configs)
        train_len = int(len(self.configs) * 0.9)

        shuffle = np.random.permutation(len(self.configs))
        feats0 = [feats0[idx] for idx in shuffle]
        accs0 = [self.accs[idx] for idx in shuffle]

        #print(accs0)
        #print(feats0)
        print(len(accs0))
        print(len(feats0))
        dtrain0 = xgb.DMatrix(feats0[:train_len], label=accs0[:train_len])
        dval0 = xgb.DMatrix(feats0[train_len:], label=accs0[train_len:])

        params = param = {'max_depth': 3, 'eta': 0.1, 'silent': 1}
        watchlist = [(dtrain0, 'train'), (dval0, 'valid')]
        self.bst = xgb.train(params, dtrain0, 1000, evals=watchlist, early_stopping_rounds=10)
        

    def tune(self, batch_size=64, mode='test'):
        if mode == 'test':
            queue = Queue()
            proposed_configs = list()
            for i in range(0, batch_size):
                proposed_configs.append(self.proposer.propose())

            p = Process(target=self.evaluator, args=(queue,
                                                     proposed_configs))
            p.start()
            p.join()
            for i in range(0, batch_size):
                result = queue.get()
                print("test result:", result) 
            #self.evaluator(self.proposer.propose())
            return
            
        if len(self.configs) <= 0:
            print("first batch, collecting measurements...")
            cur_batch = list()
            cur_batch_acc = list()
            for i in range(0, batch_size):
                cur_batch.append(self.proposer.propose())
            # MEASUREMENT
            p = Process(target=self.evaluator, args=(queue,
                                                    cur_batch))
            p.start()
            p.join()
            for i in range(0, batch_size):
                result = queue.get()
                cur_batch_acc.append(result)

            cur_batch_acc.append(self.evaluator(cur_batch[i]))
            # TRAIN
            self.configs += cur_batch
            self.accs += cur_batch_acc
        else:
            # PROPOSE
            proposed_configs = list()
            for i in range(0, 131072):
                proposed_configs.append(self.proposer.propose())
            print("proposed 131072 configs...")
            # EVAL
            eval_feats = configs_to_feats0(proposed_configs)
            eval_data = xgb.DMatrix(eval_feats)
            eval_accs = self.bst.predict(eval_data)
            # PRUNE
            best_idx = sorted([i for i in range(0, 131072)], key=lambda idx:eval_accs[idx], reverse=True)
            #top = list()
            cur_batch = list()
            cur_batch_acc = list()
            
            # MEASUREMENT
            queue = Queue()
            for i in range(0, batch_size):
                cur_batch.append(proposed_configs[best_idx[i]])
            p = Process(target=self.evaluator, args=(queue,
                                                    cur_batch))
            p.start()
            p.join()
            for i in range(0, batch_size):
                result = queue.get()
                cur_batch_acc.append(result)
            # TRAIN
            self.configs += cur_batch
            self.accs += cur_batch_acc
            with open('configs', 'w') as f:
                for config in self.configs:
                    f.write(repr(config) + '\n')
            with open('accs', 'w') as f:
                for acc in self.accs:
                    f.write(repr(acc) + '\n') 
            sorted_idx = sorted([idx for idx in range(0, len(self.configs))], key=lambda idx: self.accs[idx], reverse=True)
            print("top 10:")
            for i in range(0, 10):
                idx = sorted_idx[i]
                print(self.accs[idx], self.configs[idx])
        self.refit()

def main():
    rqp = RandomQuantProposer([b for b in range(5, 9)], 21) 
    xgbtuner = XGBTuner(rqp, process_wrapper)
    for i in range(0, 100):
        xgbtuner.tune()

if __name__ == '__main__':
    np.random.seed(42)
    main()
