import logging
import argparse
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import numpy as np
from tvm._ffi.base import TVMError
from tvm import relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm.autotvm.task import relay_integration
import os
import sys

# Two functions for reading data from record file or raw images
def get_val_data(rec_val,
                 batch_size,
                 num_workers=16,
                 args=None):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    if args is not None:
        img_size = 299 if args.model == 'inceptionv3' else 224
    else:
        img_size = 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = 16,
        shuffle             = False,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def evaluate(graph, lib, params, ctx, custom_mix_config, args=None, val_data=None):
    """Evaluate on the validation set."""
    import tvm
    from tvm.contrib import graph_runtime

    # setup dataset.
    if args is not None:
        batch_size = args.batch_size
        val_data, batch_fn = get_val_data(args.rec_val, batch_size, args=args)
    else:
        batch_size = 64
        val_data, batch_fn = get_val_data('/scratch/tqchen/imagenet/val.rec', batch_size, args=None)
    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    if args is not None:
        oshape = (batch_size, args.num_classes)
    else:
        oshape = (batch_size, 1000)
    out_arr = tvm.nd.empty(oshape, "float32")
    # setup evaluaiton metric
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    # Execute
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.run(data=data[0].asnumpy())
        m.get_output(0, out_arr)
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])

        if args is not None and args.log_interval and not (i + 1) % args.log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
            if nsamples >= 500:
                break
        if args is None and (i+1)*batch_size >= 500:
            _, top1 = acc_top1.get()
            return top1
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    if args is not None:
        with open('record.csv', "a") as f:
            f.write('{0}! {1}! {2}\n'.format(
                args.model, repr(custom_mix_config), top1))
    else:
        return top1


def evaluate_standalone(config):
    """Evaluate on the validation set."""
    
    gluon_model = vision.get_model('resnet18_v1', pretrained=True)
    try:
        graph, lib, params, ctx = build_model(gluon_model, config, args=None)
    except TVMError as e:
        if 'shift_nbit' in str(e):
            return 0.0
        else:
            raise e
    logging.info("Finish building model...")
    return evaluate(graph, lib, params, ctx, config, args=None)


def generate_custom_mix_config(size=21):
    input_choices = [i for i in range(6, 9)]
    weight_choices = [w for w in range(6, 9)]
    act_choices = [32]
    config = list()

    config.append(list(np.random.choice(input_choices, size)))
    config.append(list(np.random.choice(weight_choices, size)))
    config.append(list(np.random.choice(act_choices, size)))

    return config


def build_model(gluon_model, custom_mix_config, args=None):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 224
    batch_size = 64
    #target = 'llvm -mcpu=core-avx2'
    target = 'cuda -model=1080ti'
    if args is not None:
        img_size = 299 if args.model == 'inceptionv3' else 224
        batch_size = args.batch_size
        target = args.target
    print("ARGS TARGET:", target)
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})

    if args is not None and args.original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    # constant folding and scale folding.
    print('original')
    #print(net.astext(show_meta_data=False))
    with relay.build_config(opt_level=3):
        qgraph = relay.optimize(net, target, params)
        # qgraph = relay.optimize(qgraph)
    print('after optimize')
    #print(qgraph.astext(show_meta_data=False))

    if args is not None:
        nbit_input=args.nbit_input          
        nbit_weight=args.nbit_input
        global_scale=args.global_scale
        dtype_input=args.dtype_input
        dtype_weight=args.dtype_input
        dtype_activation=args.dtype_output
    else:
        nbit_input=32
        nbit_weight=8
        global_scale=8.0
        dtype_input='int8'
        dtype_weight='int8'
        dtype_activation='int32'
        print("?????????????????")
  
    with qtz.qconfig(skip_k_conv=0,
                     nbit_input=nbit_input,
                     nbit_weight=nbit_weight,
                     global_scale=global_scale,
                     dtype_input=dtype_input,
                     dtype_weight=dtype_weight,
                     dtype_activation=dtype_activation,
                     store_lowbit_output=False,
                     debug_enabled_ops=None):
        #print(qtz.current_qconfig())
        qgraph = qtz.annotate(qgraph)
        print('after annotate')
        #print(qgraph.astext(show_meta_data=False))
        qgraph = qtz.calibrate(qgraph, custom_mix_config=custom_mix_config)
        print('after calibrate\n')
        #print(qgraph.astext(show_meta_data=False))
        if True:#args.simulated:
            qgraph = qtz.realize(qgraph)
            qgraph = relay.ir_pass.infer_type(qgraph)
            print('after realize\n')
            #print(qgraph.astext(show_meta_data=False))

    if args is not None and args.tune_workload:
        tasks = relay_integration.extract_from_program(qgraph, params=params, target=target, ops=(relay.op.nn.conv2d, relay.op.nn.dense))
        print(tasks)
        tuning_option = {
            'log_filename': 'tuning_log.txt',
            'tuner': 'xgb',
            'n_trial': 4000,
            'early_stopping': 600,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.RPCRunner(
                    '1080ti',  # change the device key to your key
                    'fleet', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150),
            ),
        }
        print("RESETTING AFFINITY... yikes...")
        os.system("taskset -p 0xffffffff %d" % os.getpid())
        print("Tuning tasks...")
        tune_tasks(tasks, **tuning_option)
        sys.exit()

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)
    ctx = tvm.nd.context(target, 0)
    #print('CUSTOM MIX CONFIG:', custom_mix_config)
    return graph, lib, params, ctx

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=False,
               try_int8=True):
    new_tasks = list()
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    new_tasks.append(tsk)
            except Exception:
                pass
    if try_int8:
        for i in range(len(tasks)):
            try:  # try int8 template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'int8')
                new_tasks.append(tsk)
            except Exception:
                pass

    tasks += new_tasks
    tasks = tasks[:2]
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        print(tsk)
        print(len(tsk.space))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def main(args):
    mix_configs = set()
    for i in range(0, 5000):
        print(i)
        custom_mix_config = generate_custom_mix_config()
        while repr(custom_mix_config) in mix_configs:
            custom_mix_config = generate_custom_mix_config()
        mix_configs.add(repr(custom_mix_config))
        gluon_model = vision.get_model(args.model, pretrained=True)
        try:
            with autotvm.apply_history_best('tuning_log.txt'):
                graph, lib, params, ctx = build_model(gluon_model, custom_mix_config, args=args)
        except TVMError as e:
            raise e
            continue
        logging.info("Finish building model %s...", args.model)
        # raise ValueError
        evaluate(graph, lib, params, ctx, custom_mix_config, args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--tune-workload", action="store_true", help="just do workload tuning")
    parser.add_argument("--rec-val", type=str, default="/scratch/tqchen/imagenet/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet18_v1",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--target", type=str, default="llvm -mcpu=core-avx2",
                        help="target option")
    parser.add_argument("--nbit-input", type=int, default=8,
                        help="number of input bits")
    parser.add_argument("--nbit-output", type=int, default=32,
                        help="number of output bits")
    parser.add_argument("--dtype-input", type=str, default="int8",
                        help="number of input bits")
    parser.add_argument("--dtype-output", type=str, default="int32",
                        help="number of output bits")
    parser.add_argument("--global-scale", type=float, default=8.0,
                        help="global activation scale")
    parser.add_argument("--original", action="store_true",
                        help='whether to use original graph')
    parser.add_argument("--simulated", action="store_true",
                        help='whether to use simulated graph')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)