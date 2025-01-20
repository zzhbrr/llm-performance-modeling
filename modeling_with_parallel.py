result = {"prefill":{}, "decode":{}}

Operations = ["Q", "K", "V", "qk_matmul", "softmax", "sv_matmul", "O", "FFN_up", "FFN_gate", "FFN_down"]

weight_size = {}

# Hardware Spec
hardware = { # H100, 3TB/s，989 TFLOPS
    "bandwidth": 3072e9 / 5, # bytes/s 
    "FP16": 1979e12 / 2 / 5 # OPs/s
}

def record(stage, operation, OPs, load_weight, load_act, store_act, load_kv_cache, store_kv_cache):
    t = max((load_weight + load_act + store_act + load_kv_cache + store_kv_cache) / hardware["bandwidth"], OPs / hardware["FP16"])
    result[stage][operation] = {
        "name": operation,
        "time": t*1e6, # us
        "OPs": OPs,
        "load_weight": load_weight,
        "load_act": load_act,
        "store_act": store_act,
        "load_kv_cache": load_kv_cache,
        "store_kv_cache": store_kv_cache
    }
    weight_size[operation] = load_weight
    
# Model Spec
model_name = "Llama2-70B" 
L = 80 # number of layers
d = 8192 # hidden dimension
d_ffn = 28672 # feedforward dimension
h = 64 # number of attention heads
h_kv = 8 # number of key-value heads
a_byte = 2
w_byte = 2
kv_byte = 2
headsize = d // h

# Workload Spec
batchsize = 1
seql = 5502 # sequence length

# Parallelization Spec
tp_size = 8
pp_size = 1
# intra_bw = 50 # GB/s
intra_bw = 400 # GB/s
intra_lat = 1.5 # us
# inter_bw = 50 # GB/s
# inter_lat = 2 # us


def ModelSpec(model_name):
    global d, d_ffn, h, h_kv, a_byte, w_byte, kv_byte, headsize, L
    if model_name == "Llama2-70B":
        L = 80
        d = 8192 
        d_ffn = 28672 
        h = 64 
        h_kv = 8 
        a_byte = 2 
        w_byte = 2
        kv_byte = 2
        headsize = d // h
    elif model_name == "OPT-30B":
        L = 48
        d = 7168
        d_ffn = 28672
        h = 56
        h_kv = 56
        a_byte = 2
        w_byte = 2
        kv_byte = 2
    elif model_name == "7B":
        L = 32
        d = 4096
        d_ffn = 11008
        h = 32
        h_kv = 32
        a_byte = 2
        w_byte = 2
        kv_byte = 2
    else:
        raise ValueError("Invalid model name")

def PrintConfig():
    print("Configuration: {")
    print("Hardware: ")
    print("    bandwidth: ", hardware["bandwidth"])
    print("    FP16: ", hardware["FP16"])
    print("Workload: ")
    print("    batchsize: ", batchsize)
    print("    seql: ", seql)
    print("Model: ")
    print("    model_name: ", model_name)
    print("    d: ", d)
    print("    d_ffn: ", d_ffn)
    print("    h: ", h)
    print("    h_kv: ", h_kv)
    print("    a_byte: ", a_byte)
    print("    w_byte: ", w_byte)
    print("    kv_byte: ", kv_byte)
    print("    headsize: ", headsize)
    print("Parallelization: ")
    print("    tp_size: ", tp_size)
    print("    pp_size: ", pp_size)
    print("}")  

def analyze():
    for op in Operations:
        if op == "Q":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * d / tp_size, 
                   load_weight = d * d * w_byte / tp_size, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = seql * batchsize * d * a_byte / tp_size, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d / tp_size, 
                   load_weight = d * d * w_byte / tp_size, 
                   load_act = batchsize * d * a_byte, 
                   store_act = batchsize * d * a_byte / tp_size, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
        elif op == "K" or op == "V":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * h_kv * headsize / tp_size, 
                   load_weight = d * h_kv * headsize * w_byte / tp_size, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = seql * batchsize * h_kv * headsize * kv_byte / tp_size)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * h_kv * headsize / tp_size, 
                   load_weight = d * h_kv * headsize * w_byte / tp_size, 
                   load_act = batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = batchsize * h_kv * headsize * kv_byte / tp_size)
        elif op == "qk_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * h * headsize / tp_size, 
                   load_weight = 0, 
                   load_act = seql * batchsize * h * headsize * a_byte / tp_size, 
                   store_act = batchsize * seql * seql * h * a_byte / tp_size, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte / tp_size, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * h * headsize / tp_size, 
                   load_weight=0, 
                   load_act = batchsize * h * headsize * a_byte / tp_size, 
                   store_act = batchsize * seql * h * a_byte / tp_size, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte / tp_size, 
                   store_kv_cache=0)
        elif op == "softmax":
            record("prefill", 
                   op, 
                   OPs = 5 * batchsize * seql * seql * h / tp_size, 
                   load_weight=0,
                   load_act=batchsize * seql * seql * h * a_byte / tp_size, 
                   store_act=batchsize * seql * seql * h * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 5 * batchsize * seql * h / tp_size, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte / tp_size, 
                   store_act=batchsize * seql * h * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "sv_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * headsize * h / tp_size,
                   load_weight=0, 
                   load_act=batchsize * seql * seql * h * a_byte / tp_size, 
                   store_act=batchsize * seql * h * headsize * a_byte / tp_size, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte / tp_size, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * headsize * h / tp_size, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte / tp_size, 
                   store_act=batchsize * h * headsize * a_byte / tp_size, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte / tp_size, 
                   store_kv_cache=0)
        elif op == "O":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * d / tp_size, 
                   load_weight=d * d * w_byte / tp_size, 
                   load_act=batchsize * seql * d * a_byte / tp_size, 
                   store_act=batchsize * seql * d * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d / tp_size, 
                   load_weight=d * d * w_byte / tp_size, 
                   load_act=batchsize * d * a_byte / tp_size, 
                   store_act=batchsize * d * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "FFN_up" or op == "FFN_gate":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * d_ffn / tp_size, 
                   load_weight=d * d_ffn * w_byte / tp_size, 
                   load_act=batchsize * seql * d * a_byte, 
                   store_act=batchsize * seql * d_ffn * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d_ffn / tp_size, 
                   load_weight= d * d_ffn * w_byte / tp_size, 
                   load_act= batchsize * d * a_byte, 
                   store_act= batchsize * d_ffn * a_byte / tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "FFN_down":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d_ffn * d / tp_size, 
                   load_weight=d_ffn * d * w_byte / tp_size, 
                   load_act=batchsize * seql * d_ffn * a_byte / tp_size, 
                   store_act=batchsize * seql * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d_ffn * d / tp_size, 
                   load_weight=d_ffn * d * w_byte / tp_size, 
                   load_act=batchsize * d_ffn * a_byte / tp_size, 
                   store_act=batchsize * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)

def GetKVCacheSizePerLayer(): # bytes
    return 2 * seql * batchsize * h_kv * headsize * kv_byte

    
if __name__ == "__main__":
    model_name = "Llama2-70B"
    # model_name = "OPT-30B"
    # model_name = "7B"
    ModelSpec(model_name)
    analyze()
    
    VERBOSE = False
    PrintConfig()
    t = {"prefill":{}, "decode":{}}
    for stage in ['prefill', 'decode']:
        print("Stage:", stage)
        t[stage] = 0
        if tp_size != 1:
            if stage == 'prefill':
                volume = 2 * seql * batchsize * d * a_byte # 一次allreduce的bytes
                # MLP 和 attention 分别有一次 allreduce
                networktime = 2 * (tp_size - 1) * (intra_lat + volume / tp_size / 1024 / 1024 / 1024 / intra_bw * 1e6)
            elif stage == 'decode':
                volume = 2 * batchsize * d * a_byte 
                networktime = 2 * (tp_size - 1) * (intra_lat + volume / tp_size / 1024 / 1024 / 1024 / intra_bw * 1e6)
            t[stage] += networktime
            print("TP communication overhead in", stage, "stage: ", round(networktime, 3), "us")
        for op in Operations:
            if VERBOSE:
                print('{')
                print("    operation: ", result[stage][op]["name"])
                print("    time: ", result[stage][op]["time"])
                print("    OPs: ", result[stage][op]["OPs"])
                print("    load_weight: ", result[stage][op]["load_weight"])
                print("    load_act: ", result[stage][op]["load_act"])
                print("    store_act: ", result[stage][op]["store_act"])
                print("    load_kv_cache: ", result[stage][op]["load_kv_cache"])
                print("    store_kv_cache: ", result[stage][op]["store_kv_cache"])
                print('}')
            t[stage] += result[stage][op]["time"]
        print("Total time: ", round(t[stage], 2), "us")
    print("")


    
    kvsize = GetKVCacheSizePerLayer()
    print("KV Cache Size per layer: ", round(kvsize / 1024 / 1024 / 1024, 3), "GB")

    # print("activation size in decode stage: ", round(batchsize * d * a_byte / 1024 / 1024 / 1024, 3), "GB")
    # print("activation size in prefill stage: ", round(batchsize * seql * d * a_byte / 1024 / 1024 / 1024, 3), "GB")
    # print("QK intermediate size in decode stage: ", round(batchsize * seql * h * headsize * a_byte / 1024 / 1024 / 1024, 3), "GB")
    # print("QK intermediate size in prefill stage: ", round(batchsize * seql * seql * h * headsize * a_byte / 1024 / 1024 / 1024, 3), "GB")
    # print("")

    # for op in Operations:
        # if op not in ['qk_matmul', 'softmax', 'sv_matmul']:
        #     print(op, "weight size: ", round(weight_size[op] / 1024 / 1024 / 1024, 3), "GB")
    
    # print("")
    # # MemoryBW = 50 
    # MemoryBW = 25 
    # print("can prefetch", round(t["decode"] * MemoryBW * 1e-6, 3), "GB every layer, in decode stage, if network bandwidth is", MemoryBW, "GB/s")
    # print("can prefetch", round(t["prefill"] * MemoryBW * 1e-6, 3), "GB every layer, in prefill stage, if network bandwidth is", MemoryBW, "GB/s")
    # print("")

    print("GPU Memory Consumption: ")
    mem_consum = 0
    for op in Operations:
        if op not in ['qk_matmul', 'softmax', 'sv_matmul']:
            mem_consum += weight_size[op] / 1024 / 1024 / 1024
    mem_consum += GetKVCacheSizePerLayer() / 1024 / 1024 / 1024 / tp_size
    mem_consum *= L / pp_size
    print("Prefill stage: ", round(mem_consum + seql*batchsize*max(2*d_ffn/tp_size, d)*a_byte/1024/1024/1024, 3), "GB")
    print("Decode stage: ", round(mem_consum + batchsize*max(2*d_ffn/tp_size, d)*a_byte/1024/1024/1024, 3), "GB")
    print("")



    iteration_t = {"prefill":0, "decode":0}
    
    comm_time = intra_lat + seql * batchsize * d * a_byte / 1024 / 1024 / 1024 / intra_bw * 1e6
    iteration_t["prefill"] = L * t["prefill"] + (pp_size - 1) * comm_time
    # print("Prefill Iteration latency: ", round(L * t["prefill"] + (pp_size - 1) * comm_time, 3), "us")
    print("Prefill Iteration latency: ", round(iteration_t["prefill"] / 1000, 3), "ms")
    comm_time = intra_lat + batchsize * d * a_byte / 1024 / 1024 / 1024 / intra_bw * 1e6
    iteration_t["decode"] = L * t["decode"] + (pp_size - 1) * comm_time
    # print("Decode Iteration latency: ", round(L * t["decode"] + (pp_size - 1) * comm_time, 3), "us")
    print("Decode Iteration latency: ", round(iteration_t["decode"] / 1000, 3), "ms")

    print("")
    print("Throughput: ")
    if pp_size == 1:
        print("prefill throughput: ", round(batchsize * seql / iteration_t["prefill"] * 1e6, 3), " tokens/s")
        print("decode throughput: ", round(batchsize / iteration_t["decode"] * 1e6, 3), " tokens/s")
    elif pp_size != 1: 
        micro_batch_num = 100 
        print("prefill throughput: ", round(batchsize * micro_batch_num * seql / (iteration_t["prefill"] + iteration_t["prefill"] / pp_size * (micro_batch_num - 1)) * 1e6, 3), " tokens/s")
        print("decode throughput: ", round(batchsize * micro_batch_num / (iteration_t["decode"] + iteration_t["decode"] / pp_size * (micro_batch_num - 1)) * 1e6, 3), " tokens/s")

# Decode Scale Results 
# seql = 512, batchsize = 100, micro-batch-num = 100
# GPU num: 1, 2, 3, 4, 5, 6, 7, 8
# Throughput
# TP     : OOM, OOM, 3467, 4483, 5411, 6426, 6987, 7633
# PP     : OOM, OOM, 3561, 4702, 5820, 6917, 7994, 9050
# Latency
# TP     : OOM, OOM, 28, 22, 18, 16, 14, 13
# PP     : OOM, OOM, 82, 82, 82, 82, 82, 82

# Prefill Scale Results 
# seql = 512, batchsize = 1, micro-batch-num = 100
# GPU num: 1, 2, 3, 4, 5, 6, 7, 8
# Throughput
# TP     : 7122(OOM), 13156, 18229, 22463, 25974, 28865, 31226, 33023
# PP     : 7122(OOM), 14100, 20938, 27639, 34208, 40648, 46963, 53157
# Latency
# TP     : 71, 38, 28, 22, 19, 17, 16, 15
# PP     : 71, 71, 71, 71, 71, 71, 71, 72
