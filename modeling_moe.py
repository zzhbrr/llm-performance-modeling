result = {"prefill":{}, "decode":{}}
communication_result = {"prefill":{}, "decode":{}}

Operations = ["Q", "K", "V", "qk_matmul", "softmax", "sv_matmul", "O", "gate", "FFN_up", "FFN_gate", "FFN_down"]

weight_size = {}

# Hardware Spec
hardware = { # H100, 3TB/s，989 TFLOPS
    "bandwidth": 3072e9, # bytes/s 
    "FP16": 1979e12 / 2 # OPs/s
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
model_name = "Hunyuan-Large"
L = 64 # number of layers
d = 6400 # hidden dimension
d_ffn = 18304 # feedforward dimension
h = 80 # number of attention heads
h_kv = 8 # number of key-value heads
expert_num = 16 # number of experts
topk = 1 # top-k experts to route
shared_expert_num = 1 # number of shared experts
cla = 2 # cla share factor
a_byte = 2
w_byte = 2
kv_byte = 2
headsize = d // h
kv_lora_rank = 0
q_lora_rank = 0

# Workload Spec
batchsize = 256 * 10
seql = 1024 # sequence length
global_bachsize = batchsize

# Parallelization Spec
pp_size = 1
nonmoe_dp_size = 2
nonmoe_tp_size = 4
moe_dp_size = 1
moe_tp_size = 1
moe_ep_size = 8

intra_bw = 400 # GB/s
# intra_lat = 1.5 # us
# inter_bw = 50 # GB/s
# inter_lat = 2 # us


def ModelSpec(model_name):
    global d, d_ffn, h, h_kv, a_byte, w_byte, kv_byte, headsize, L, expert_num, topk, shared_expert_num, cla, kv_lora_rank, q_lora_rank
    if model_name == "Hunyuan-Large":
        L = 64
        d = 6400
        d_ffn = 18304
        h = 80
        h_kv = 8
        expert_num = 16
        topk = 1
        shared_expert_num = 1
        cla = 2
        a_byte = 2
        w_byte = 2
        kv_byte = 2
        headsize = d // h
        kv_lora_rank = 0
        q_lora_rank = 0
    elif model_name == "DeepSeek-V3":
        L = 61
        d = 7168
        d_ffn = 18432
        h = 128
        h_kv = 128
        expert_num = 256
        topk = 8
        shared_expert_num = 1
        cla = 0
        a_byte = 2
        w_byte = 2
        kv_byte = 2
        headsize = d // h
        kv_lora_rank = 512
        q_lora_rank = 1536
    else:
        raise ValueError("Invalid model name")

def PrintConfig():
    assert nonmoe_dp_size * nonmoe_tp_size == moe_dp_size * moe_tp_size * moe_ep_size 
    assert moe_ep_size <= expert_num
    assert moe_dp_size == 1
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
    print("    Expert_num: ", expert_num)
    print("    topk: ", topk)
    print("    shared_expert_num: ", shared_expert_num)
    if cla != 0:
        print("    cla: ", cla)
    if kv_lora_rank != 0:   
        print("    kv_lora_rank: ", kv_lora_rank)
    if q_lora_rank != 0:
        print("    q_lora_rank: ", q_lora_rank)
    print("Parallelization: ")
    print("    nonmoe_dp_size: ", nonmoe_dp_size)
    print("    nonmoe_tp_size: ", nonmoe_tp_size)
    print("    moe_dp_size: ", moe_dp_size)
    print("    moe_tp_size: ", moe_tp_size)
    print("    moe_ep_size: ", moe_ep_size)
    print("    pp_size: ", pp_size)
    print("}")  

def analyze():
    global global_bachsize, batchsize
    global_bachsize = batchsize
    
    for op in Operations:
        # Non MoE part
        batchsize = global_bachsize // nonmoe_dp_size
        # One AllReduce in Attention Calculation if TP used
        communication_result["prefill"]["Attention_AllReduce"] = 2 * batchsize * seql * d * a_byte * (nonmoe_tp_size - 1) / nonmoe_tp_size
        communication_result["decode"]["Attention_AllReduce"] = 2 * batchsize * d * a_byte * (nonmoe_tp_size - 1) / nonmoe_tp_size
        if op == "Q":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * d / nonmoe_tp_size, 
                   load_weight = d * d * w_byte / nonmoe_tp_size, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = seql * batchsize * d * a_byte / nonmoe_tp_size, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d / nonmoe_tp_size, 
                   load_weight = d * d * w_byte / nonmoe_tp_size, 
                   load_act = batchsize * d * a_byte, 
                   store_act = batchsize * d * a_byte / nonmoe_tp_size, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
        elif op == "K" or op == "V":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * h_kv * headsize / nonmoe_tp_size, 
                   load_weight = d * h_kv * headsize * w_byte / nonmoe_tp_size, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = seql * batchsize * h_kv * headsize * kv_byte / nonmoe_tp_size)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * h_kv * headsize / nonmoe_tp_size, 
                   load_weight = d * h_kv * headsize * w_byte / nonmoe_tp_size, 
                   load_act = batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = batchsize * h_kv * headsize * kv_byte / nonmoe_tp_size)
        elif op == "qk_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * h * headsize / nonmoe_tp_size, 
                   load_weight = 0, 
                   load_act = seql * batchsize * h * headsize * a_byte / nonmoe_tp_size, 
                   store_act = batchsize * seql * seql * h * a_byte / nonmoe_tp_size, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte / nonmoe_tp_size, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * h * headsize / nonmoe_tp_size, 
                   load_weight=0, 
                   load_act = batchsize * h * headsize * a_byte / nonmoe_tp_size, 
                   store_act = batchsize * seql * h * a_byte / nonmoe_tp_size, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte / nonmoe_tp_size, 
                   store_kv_cache=0)
        elif op == "softmax":
            record("prefill", 
                   op, 
                   OPs = 5 * batchsize * seql * seql * h / nonmoe_tp_size, 
                   load_weight=0,
                   load_act=batchsize * seql * seql * h * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * seql * seql * h * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 5 * batchsize * seql * h / nonmoe_tp_size, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * seql * h * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "sv_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * headsize * h / nonmoe_tp_size,
                   load_weight=0, 
                   load_act=batchsize * seql * seql * h * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * seql * h * headsize * a_byte / nonmoe_tp_size, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte / nonmoe_tp_size, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * headsize * h / nonmoe_tp_size, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * h * headsize * a_byte / nonmoe_tp_size, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte / nonmoe_tp_size, 
                   store_kv_cache=0)
        elif op == "O":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * d / nonmoe_tp_size, 
                   load_weight=d * d * w_byte / nonmoe_tp_size, 
                   load_act=batchsize * seql * d * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * seql * d * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d / nonmoe_tp_size, 
                   load_weight=d * d * w_byte / nonmoe_tp_size, 
                   load_act=batchsize * d * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * d * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "gate":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * expert_num / nonmoe_tp_size, 
                   load_weight=d * expert_num * w_byte / nonmoe_tp_size, 
                   load_act=batchsize * seql * d * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * seql * expert_num * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * expert_num / nonmoe_tp_size, 
                   load_weight=d * expert_num * w_byte / nonmoe_tp_size, 
                   load_act=batchsize * d * a_byte / nonmoe_tp_size, 
                   store_act=batchsize * expert_num * a_byte / nonmoe_tp_size, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        else: 
            # MoE part
            batchsize = global_bachsize
            prefill_average_activated_expertnum = min(batchsize * seql * topk, expert_num) + shared_expert_num
            decode_average_activated_expertnum = min(batchsize * topk, expert_num) + shared_expert_num

            if op == "FFN_up" or op == "FFN_gate":
                record("prefill", 
                    op, 
                    OPs = 2 * batchsize * seql * d * d_ffn / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_weight=d * d_ffn * w_byte / moe_tp_size / moe_ep_size * prefill_average_activated_expertnum, 
                    load_act=batchsize * seql * d * a_byte / moe_ep_size * (topk + shared_expert_num),
                    store_act=batchsize * seql * d_ffn * a_byte / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_kv_cache=0, 
                    store_kv_cache=0)
                record("decode", 
                    op, 
                    OPs = 2 * batchsize * d * d_ffn / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_weight= d * d_ffn * w_byte / moe_tp_size / moe_ep_size * decode_average_activated_expertnum, 
                    load_act= batchsize * d * a_byte / moe_ep_size * (topk + shared_expert_num), 
                    store_act= batchsize * d_ffn * a_byte / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_kv_cache=0, 
                    store_kv_cache=0)
            elif op == "FFN_down":
                record("prefill", 
                    op, 
                    OPs = 2 * batchsize * seql * d_ffn * d / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_weight=d_ffn * d * w_byte / moe_tp_size / moe_ep_size * decode_average_activated_expertnum, 
                    load_act=batchsize * seql * d_ffn * a_byte / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    store_act=batchsize * seql * d * a_byte / moe_ep_size * (topk + shared_expert_num), 
                    load_kv_cache=0, 
                    store_kv_cache=0)
                record("decode", 
                    op, 
                    OPs = 2 * batchsize * d_ffn * d / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    load_weight=d_ffn * d * w_byte / moe_tp_size / moe_ep_size * decode_average_activated_expertnum, 
                    load_act=batchsize * d_ffn * a_byte / moe_tp_size / moe_ep_size * (topk + shared_expert_num), 
                    store_act=batchsize * d * a_byte / moe_ep_size * (topk + shared_expert_num), 
                    load_kv_cache=0, 
                    store_kv_cache=0)
        # communication calculation
        communication_result["prefill"]["MoE_AllReduce"] = 2 * global_bachsize * seql * topk * (moe_tp_size - 1) * d / moe_tp_size / moe_ep_size * a_byte
        communication_result["decode"]["MoE_AllReduce"] = 2 * global_bachsize * topk * (moe_tp_size - 1) * d / moe_tp_size / moe_ep_size * a_byte
        if moe_ep_size > 1:
            communication_result["prefill"]["MoE_All2All_1"] = global_bachsize * seql * topk * d * (nonmoe_dp_size - 1) / nonmoe_dp_size / moe_ep_size * a_byte
            communication_result["decode"]["MoE_All2All_1"] = global_bachsize * topk * d * (nonmoe_dp_size - 1) / nonmoe_dp_size / moe_ep_size * a_byte
        else:
            communication_result["prefill"]["MoE_All2All_1"] = global_bachsize * seql * d * (nonmoe_dp_size - 1) / nonmoe_dp_size * a_byte
            communication_result["decode"]["MoE_All2All_1"] = global_bachsize * d * (nonmoe_dp_size - 1) / nonmoe_dp_size * a_byte
        communication_result["prefill"]["MoE_All2All_2"] = communication_result["prefill"]["MoE_All2All_1"]
        communication_result["decode"]["MoE_All2All_2"] = communication_result["decode"]["MoE_All2All_1"]
            
def GetKVCacheSizePerLayer(): # bytes
    if kv_lora_rank == 0:
        pass
    res = 2 * seql * batchsize * h_kv * headsize * kv_byte
    if cla:
        res /= cla
    return res


if __name__ == "__main__":
    model_name = "Hunyuan-Large"
    # model_name = "DeepSeek-V3"
    ModelSpec(model_name)
    analyze()

    VERBOSE = False
    PrintConfig()
    t = {"prefill":{}, "decode":{}}
    for stage in ['prefill', 'decode']:
        print("Stage:", stage)
        t[stage] = 0
        # Communication Time
        networktime = (
            ( communication_result[stage]["Attention_AllReduce"]
                + communication_result[stage]["MoE_AllReduce"] 
                + communication_result[stage]["MoE_All2All_1"]
                + communication_result[stage]["MoE_All2All_2"]
            ) / 1024 / 1024 / 1024 / intra_bw * 1e6)
        t[stage] += networktime
        print("intra-op communication overhead in", stage, "stage: ", round(networktime, 3), "us")
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
    print("KV Cache Size per layer: ", round(kvsize / 1024 / 1024 / 1024, 4), "GB")
    print("")

    # # MemoryBW = 50
    # MemoryBW = 25
    # print("can prefetch", round(t["decode"] * MemoryBW * 1e-6, 3), "GB every layer, in decode stage, if network bandwidth is", MemoryBW, "GB/s")
    # print("can prefetch", round(t["prefill"] * MemoryBW * 1e-6, 3), "GB every layer, in prefill stage, if network bandwidth is", MemoryBW, "GB/s")
    # print("")

    print("GPU Memory Consumption: ")
    mem_consum = 0
    for op in Operations:
        if op not in ['qk_matmul', 'softmax', 'sv_matmul', 'FFN_up', 'FFN_gate', 'FFN_down']: 
            mem_consum += weight_size[op] / 1024 / 1024 / 1024
    # expert weight单独计算
    mem_consum += (expert_num + shared_expert_num) * d * d_ffn * 3 * w_byte / 1024 / 1024 / 1024 / moe_tp_size / moe_ep_size
    mem_consum += GetKVCacheSizePerLayer() / 1024 / 1024 / 1024 / nonmoe_tp_size # KV Cache
    mem_consum *= L / pp_size
    # activation size = max(attention_part, moe_part)
    bs1, bs2 = global_bachsize / nonmoe_dp_size, global_bachsize / moe_dp_size
    print("Prefill stage(w/o activation): ", round(mem_consum, 3), "GB")
    print("Decode stage(w/o activation): ", round(mem_consum, 3), "GB")
    print("Prefill stage(w/ activation): ", round(mem_consum + max(max(bs1*seql*seql, bs1*seql*d), max(bs2*seql*d_ffn*2/moe_tp_size/moe_ep_size, bs2*seql*d/moe_ep_size)) * a_byte/1024/1024/1024, 3), "GB")
    print("Decode stage(w/ activation): ", round(mem_consum + max(bs1*d, max(bs2*d_ffn*2/moe_tp_size/moe_ep_size, bs2*d/moe_ep_size))*a_byte/1024/1024/1024, 3), "GB")
    print("")

    iteration_t = {"prefill":0, "decode":0}

    comm_time = seql * bs1 * d * a_byte / 1024 / 1024 / 1024 / intra_bw * 1e6
    iteration_t["prefill"] = L * t["prefill"] + (pp_size - 1) * comm_time
    print("Prefill Iteration latency: ", round(iteration_t["prefill"] / 1000, 3), "ms")
    comm_time = bs1 * d * a_byte / 1024 / 1024 / 1024 / intra_bw * 1e6
    iteration_t["decode"] = L * t["decode"] + (pp_size - 1) * comm_time
    print("Decode Iteration latency: ", round(iteration_t["decode"] / 1000, 3), "ms")

    print("")
    print("Throughput: ")
    if pp_size == 1:
        print("prefill throughput: ", round(global_bachsize * seql / iteration_t["prefill"] * 1e6, 3), " tokens/s")
        print("decode throughput: ", round(global_bachsize / iteration_t["decode"] * 1e6, 3), " tokens/s")
    elif pp_size != 1: 
        micro_batch_num = 100 
        print("prefill throughput: ", round(global_bachsize * micro_batch_num * seql / (iteration_t["prefill"] + iteration_t["prefill"] / pp_size * (micro_batch_num - 1)) * 1e6, 3), " tokens/s")
        print("decode throughput: ", round(global_bachsize * micro_batch_num / (iteration_t["decode"] + iteration_t["decode"] / pp_size * (micro_batch_num - 1)) * 1e6, 3), " tokens/s")


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
