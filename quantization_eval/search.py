import numpy as np
import heapq

import xgboost as xgb

from evaluate import get_val_data, evaluate_scale

from mxnet import gluon
from mxnet.gluon.model_zoo import vision

from tvm.autotvm.util import sample_ints, get_rank
from tvm.autotvm.tuner.metric import average_recall
from tvm.autotvm.tuner.xgboost_cost_model import custom_callback, xgb_average_recalln_curve_score

class SAKnob(object):
    def __init__(self, name, choices):
        self.name = name
        self.choices = choices
        self.size = len(self.choices)
        self.state = self.choices[0]
        self.pos = 0

    def set(self, idx):
        self.state = self.choices[idx]
        self.pos = idx

    def __str__(self):
        return str(self.name) + ":" + str(self.state)

class FlatSAKnob(SAKnob):
    def __init__(self, name, choices):
        super().__init__(name, choices)

    def step(self):
        # flip coin
        raise NotImplementedError()

class OrderedSAKnob(SAKnob):
    def __init__(self, name, choices):
        super().__init__(name, choices)

    def step(self):
        # flip coin for up or down, step
        up = np.random.choice([True, False])
        if up:
            new_idx = self.pos + 1 if self.pos + 1 < self.size else 0
            self.set(new_idx)
        else:
            new_idx = self.pos - 1 if self.pos - 1 > -self.size else self.size - 1
            self.set(new_idx)

class SASpace(object):
    def __init__(self, knobs):
        self.knobs = knobs
        self.size = 1
        for knob in self.knobs:
            self.size *= knob.size

    def point2config(self, point):
        for idx in range(-1, -1 - len(self.knobs), -1):
            pos = point % self.knobs[idx].size
            self.knobs[idx].set(pos)
            point = point // self.knobs[idx].size

    def config2point(self):
        point = 0
        scale = 1
        for idx in range(-1, -1 - len(self.knobs), -1):
            point = point + scale*self.knobs[idx].pos
            scale = scale * self.knobs[idx].size
        return point

    def step(self):
        knob_idx = sample_ints(0, len(self.knobs), 1)[0]
        self.knobs[knob_idx].step()

    def __str__(self):
        s = ""
        for knob in self.knobs:
            s += knob.__str__() + ','
        return s

    def vector(self):
        vec = [None]*len(self.knobs)
        for i in range(0, len(self.knobs)):
            vec[i] = self.knobs[i].state
        return np.array(vec)

class ScaleSASpace(SASpace):
    def __init__(self, choices, length):
        knobs = list()
        for i in range(0, length):
            knobs.append(OrderedSAKnob('scale'+str(i), choices)) 
        super().__init__(knobs)

class SAOptimizer(object):
    def __init__(self, space, n_iter=500, temp=(1,0), persistent=True,
                 parallel_size=512, early_stop=50, log_interval=50):
        self.space = space
        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, space.size)
        self.early_stop = early_stop or np.inf 
        self.log_interval = log_interval
        # initialize space
        self.points = np.array(sample_ints(0, space.size, parallel_size), dtype=object)

    def crank(self, model, plan_size=64, exclude=None):
        # reset walk
        if not self.persistent:
            self.points = np.array(sample_ints(0, space.size, parallel_size), dtype=object)

        exclude = set(exclude) if exclude is not None else set()
        covered_heap = list()
        heapq.heapify(covered_heap)

        features = list()
        for p in range(0, self.parallel_size):
            self.space.point2config(self.points[p])
            features.append(self.space.vector())
        features = xgb.DMatrix(features)
        scores = model.predict(features)

        for p in range(0, self.parallel_size):
            point = self.points[p]
            score = scores[p]
            if point in exclude:
                continue
            else:
                exclude.add(point)
                heapq.heappush(covered_heap, (score, point)) 

        if isinstance(self.temp, (tuple, list, np.ndarray)):
            t = self.temp[0]
            cool = 1.0 * (self.temp[0] - self.temp[1]) / (self.n_iter + 1)
        else:
            t = self.temp
            cool = 0

        new_points = np.array([2**128]*self.parallel_size)

        for i in range(1, self.n_iter):
            features = [None]*self.parallel_size
            if i % self.log_interval == 0:
                print("sa iter:", i, "(", i*self.parallel_size,")")
            for p in range(0, self.parallel_size):
                self.space.point2config(self.points[p])
                # STEP 
                self.space.step()
                new_points[p] = self.space.config2point()
                features[p] = self.space.vector()
            features = xgb.DMatrix(features)
            new_scores = model.predict(features)
            probs = np.exp(np.minimum((new_scores - scores)/(t + 1e-5), 1))
            for new_score, new_point in zip(new_scores, new_points):
                if new_point in exclude:
                    continue
                else:
                    exclude.add(new_point)
                    heapq.heappush(covered_heap, (new_score, new_point))

            # accept/reject
            accept = np.random.random(self.parallel_size) < probs

            self.points[accept] = new_points[accept]
            scores[accept] = new_scores[accept]
            # save
            t -= cool
            proposals = [None]*self.parallel_size

        top_n = heapq.nlargest(plan_size, covered_heap)
        top_points = [item[1] for item in top_n]
        return top_points

class DummyModel(object):
    def predict(self, features):
        return np.random.uniform(0, 1, len(features))

config_sizes = {'resnet18_v1': 40, 'inceptionv3': 188}

def searchexp(model='resnet18_v1', batch_size=128):
    plan_size = 64
    #boilerplate
    gluon_model = vision.get_model(model, pretrained=True)
    rec_val = "/scratch/tqchen/imagenet/val.rec"
    val_data, batch_fn = get_val_data(model, rec_val, batch_size)

    ssas = ScaleSASpace([1.0, 2.0, 4.0, 8.0, 16.0], config_sizes[model])
    global_scale = 8.0

    features = list()
    scores = list()
    exclude = set()

    #fit model with initial points
    for i in range(0, plan_size):
        point = sample_ints(0, ssas.size, 1)[0]
        ssas.point2config(point)
        scale_config = ssas.vector()
        features.append(scale_config)
        top1 = evaluate_scale(gluon_model, model, val_data, batch_fn, scale_config, global_scale, batch_size)
        #top1 = np.random.random()
        scores.append(top1)
        exclude.add(point)

    bst = fit_xgb_model(features, scores, plan_size)
    saopt = SAOptimizer(ssas, n_iter=100)
    for i in range(0, 100):
        print("iter:", i)
        top_points = saopt.crank(bst, plan_size, exclude)
        for point in top_points:
            ssas.point2config(point)
            scale_config = ssas.vector()
            features.append(scale_config)
            top1 = evaluate_scale(gluon_model, model, val_data, batch_fn, scale_config, global_scale, batch_size)
            #top1 = np.random.random()
            scores.append(top1)
            exclude.add(point)
    with open('salog.csv', 'w') as f:
        for feature, score in zip(features, scores):
            f.write("{0}, {1}".format(np.array_repr(np.array(feature), max_line_width=np.inf), score))
    print("done.")

def fit_xgb_model(features, scores, plan_size):
    params = {
        'max_depth': 3,
        'gamma': 0.0001,
        'min_child_weight': 1,

        'subsample': 1.0,

        'eta': 0.3,
        'lambda': 1.00,
        'alpha': 0,

        'objective': 'rank:pairwise',
    }
    #nice meme
    data = xgb.DMatrix(features, label=scores)
    bst = xgb.train(params, data, num_boost_round=8000,
                    callbacks=[custom_callback(stopping_rounds=20,
                                               metric='tr-a-recall@%d' % plan_size,
                                               evals=[(data, 'tr')],
                                               fevals=[xgb_average_recalln_curve_score(plan_size)])],
                    verbose_eval=50)
    return bst

def main():
    #ssas = ScaleSASpace([1.0, 2.0, 4.0, 8.0, 16.0], 188)
    #saopt = SAOptimizer(ssas, n_iter=50)
    #top = saopt.crank(DummyModel(), 64, set())
    #print("1", top)
    #top = saopt.crank(DummyModel(), 64, set())
    #print("2", top)
    searchexp()

if __name__ == '__main__':
    main() 
