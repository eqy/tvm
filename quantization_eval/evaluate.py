import logging
import argparse
import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

tuning_option = {
    'log_filename': '1080ti32.log',

    'tuner': 'xgb',
    'n_trial': 8000,
    'early_stopping': 2000,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
            '1080ti',  # change the device key to your key
            'fleet', 9190,
            number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=False):
    for i in range(len(tasks)):
        try:  # try winograd template
            print("try")
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host, 'int8')
            input_channel = tsk.workload[1][1]
            #if input_channel >= 64:
            tasks[i] = tsk
            print("replaced with int8")
        except Exception:
            pass

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


# Two functions for reading data from record file or raw images
def get_val_data(model,
                 rec_val,
                 batch_size,
                 num_workers=4):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if model == 'inceptionv3' else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,
        resize              = 256 if model != 'inceptionv3' else 299,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def evaluate_args(args, graph, lib, params, ctx, val_data, batch_fn):
    num_classes = args.num_classes
    log_interval = args.log_interval
    model = args.model
    nbit_input = args.nbit_input
    nbit_output = args.nbit_output
    scale_config = [args.global_scale]
    batch_size = args.batch_size
    evaluate(graph, lib, params, ctx, batch_size, val_data, batch_fn, num_classes,
             log_interval, model, nbit_input, nbit_output, scale_config)


def evaluate(graph, lib, params, ctx, batch_size, val_data, batch_fn, num_classes,
             log_interval, model_name, nbit_input, nbit_output, scale_config, early_stopping=512):
    """Evaluate on the validation set."""
    import tvm
    from tvm.contrib import graph_runtime

    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    oshape = (batch_size, num_classes)
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

        if log_interval and not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
        if (i+1)*batch_size >= early_stopping:
            print("TOP1:", top1)
            with open('configrecord.csv', "a") as f:
                f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
                    model_name, args.nbit_input, args.nbit_output, np.array_repr(scale_config, max_line_width=np.inf), top1))
            return top1
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    with open('configrecord.csv', "a") as f:
        f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
            model_name, args.nbit_input, args.nbit_output, np.array_repr(scale_config, max_line_width=np.inf), top1))
    return top1

def build_model_args(args, gluon_model):
    model = args.model
    batch_size = args.batch_size
    target = args.target
    original = args.original
    nbit_input = args.nbit_input
    global_scale = args.global_scale
    dtype_input = args.dtype_input
    dtype_output = args.dtype_output
    simulated = args.simulated
    tuning = args.tuning
    scale_config = None
    return build_model(model, batch_size, target, original, nbit_input,
                       global_scale, dtype_input, dtype_output, simulated,
                       tuning, gluon_model, scale_config)


def build_model(model, batch_size, target, original, nbit_input, global_scale,
                dtype_input, dtype_output , simulated,
                tuning, gluon_model, scale_config):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 299 if model == 'inceptionv3' else 224
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    target = tvm.target.create(target)

    if original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    # constant folding and scale folding.
    #print('original')
    #print(net.astext(show_meta_data=False))
    with relay.build_config(opt_level=2):
        qgraph = relay.optimize(net, target, params)
        #qgraph = relay.optimize(qgraph)
    #print('after optimize')
    #print(qgraph.astext(show_meta_data=False))

    with qtz.qconfig(skip_k_conv=0,
                     nbit_input=nbit_input,
                     nbit_weight=nbit_input,
                     global_scale=global_scale,
                     dtype_input=dtype_input,
                     dtype_weight=dtype_input,
                     dtype_activation=dtype_output,
                     store_lowbit_output=False,
                     dom_scale_counter=0,
                     debug_enabled_ops=None):
        #print(qtz.current_qconfig())
        qgraph = qtz.annotate(qgraph)
        #print('after annotate')
        #print(qgraph.astext(show_meta_data=False))
        qgraph = qtz.calibrate(qgraph, custom_scale=scale_config)
        #print('after calibrate\n')
        #print(qgraph.astext(show_meta_data=False))
        if not simulated:
            qgraph = qtz.realize(qgraph)
            qgraph = relay.ir_pass.infer_type(qgraph)
            #print('after realize\n')
            #print(qgraph.astext(show_meta_data=False))
    if tuning:
        print("TUNING ONLY")
        tasks = autotvm.task.extract_from_program(qgraph, target=target, params=params, ops=(relay.op.nn.conv2d,))
        print(tasks)
        import os
        os.system("taskset -p 0xffffffff %d" % os.getpid())
        with relay.build_config(opt_level=3):
            tune_tasks(tasks, **tuning_option)
        import sys
        sys.exit(0)

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)
    ctx = tvm.nd.context(str(target), 0)
    return graph, lib, params, ctx


def evaluate_scale(model, model_name, val_data, batch_fn, config, batch_size=64):
    #pass
    graph, lib, params, ctx = build_model(args.model, batch_size, 'llvm -mcpu=core-avx2', False, 8, args.global_scale, 'int8', 'int32', False, False, gluon_model, scale_config)
    evaluate(graph, lib, params, ctx, batch_size, val_data, batch_fn, 1000,
    128, model_name, 8, 32, scale_config)


def main(args):
    gluon_model = vision.get_model(args.model, pretrained=True)
    val_data, batch_fn = get_val_data(args.model, args.rec_val, args.batch_size)
    if args.test_scale_config:
        for i in range(0, 1000):
            skip = False
            scale_config = np.random.choice([1.0, 2.0, 4.0, 8.0, 16.0], 40)
            try:
                graph, lib, params, ctx = build_model(args.model, 1, 'llvm -mcpu=core-avx2', False, args.nbit_input, args.global_scale,
                args.dtype_input, args.dtype_output, False, False, gluon_model, scale_config)
            except TVMError as e:
                skip = True
                top1 = 0.0
            if not skip:
                evaluate(graph, lib, params, ctx, args.batch_size, val_data, batch_fn, args.num_classes, args.log_interval, args.model, args.nbit_input, args.nbit_output, scale_config)
            else:
                with open('configrecord.csv', "a") as f:
                    f.write('{0}, {1}, {2}, {3}, {4}\n'.format(
                        model_name, args.nbit_input, args.nbit_output, np.array_repr(scale_config, max_line_width=np.inf), top1))
    else:
        graph, lib, params, ctx = build_model_args(args, gluon_model)
        val_data, batch_fn = get_val_data(args.model, args.rec_val, args.batch_size)
        logging.info("Finish building model %s...", args.model)
        # raise ValueError
        evaluate_args(args, graph, lib, params, ctx, val_data, batch_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="/scratch/tqchen/imagenet/val.rec",
                        help="the validation data")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet18_v1",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--target", type=str, default="llvm",
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
    parser.add_argument("--test-scale-config", action="store_true",
                        help='test scale config')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)
