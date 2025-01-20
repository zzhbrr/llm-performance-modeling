result = {"prefill":{}, "decode":{}}

Operations = ["Q", "K", "V", "qk_matmul", "softmax", "sv_matmul", "O", "FFN_up", "FFN_gate", "FFN_down"]

weight_size = {}

# Hardware Spec
hardware = { # H100, 3TB/s，989 TFLOPS
    "bandwidth": 3072e9/5 , # bytes/s 
    "FP16": 1979e12 / 2/5  # OPs/s
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
    
# Workload Spec
batchsize = 1 # bachsize
seql = 1024 # sequence length

# Model Spec
model_name = "Llama2-70B" 
d = 8192 # hidden dimension
d_ffn = 28672 # feedforward dimension
h = 64 # number of attention heads
h_kv = 8 # number of key-value heads
a_byte = 2
w_byte = 2
kv_byte = 2
headsize = d // h

def ModelSpec(model_name):
    global d, d_ffn, h, h_kv, a_byte, w_byte, kv_byte, headsize
    if model_name == "Llama2-70B":
        d = 8192 
        d_ffn = 28672 
        h = 64 
        h_kv = 8 
        a_byte = 2
        w_byte = 2
        kv_byte = 2
        headsize = d // h
    elif model_name == "OPT-30B":
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
    print("}")  

def analyze():
    for op in Operations:
        if op == "Q":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * d, 
                   load_weight = d * d * w_byte, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = seql * batchsize * d * a_byte, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d, 
                   load_weight = d * d * w_byte, 
                   load_act = batchsize * d * a_byte, 
                   store_act = batchsize * d * a_byte, 
                   load_kv_cache = 0, 
                   store_kv_cache = 0)
        elif op == "K" or op == "V":
            record("prefill", 
                   op, 
                   OPs = 2 * seql * batchsize * d * h_kv * headsize, 
                   load_weight = d * h_kv * headsize * w_byte, 
                   load_act = seql * batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = seql * batchsize * h_kv * headsize * kv_byte)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * h_kv * headsize, 
                   load_weight = d * h_kv * headsize * w_byte, 
                   load_act = batchsize * d * a_byte, 
                   store_act = 0, 
                   load_kv_cache = 0, 
                   store_kv_cache = batchsize * h_kv * headsize * kv_byte)
        elif op == "qk_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * h * headsize, 
                   load_weight = 0, 
                   load_act = seql * batchsize * h * headsize * a_byte, 
                   store_act = batchsize * seql * seql * h * a_byte, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * h * headsize, 
                   load_weight=0, 
                   load_act = batchsize * h * headsize * a_byte, 
                   store_act = batchsize * seql * h * a_byte, 
                   load_kv_cache = seql * headsize * batchsize * h_kv * kv_byte, 
                   store_kv_cache=0)
        elif op == "softmax":
            record("prefill", 
                   op, 
                   OPs = 5 * batchsize * seql * seql * h, 
                   load_weight=0,
                   load_act=batchsize * seql * seql * h * a_byte, 
                   store_act=batchsize * seql * seql * h * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 5 * batchsize * seql * h, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte, 
                   store_act=batchsize * seql * h * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "sv_matmul":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * seql * headsize * h,
                   load_weight=0, 
                   load_act=batchsize * seql * seql * h * a_byte, 
                   store_act=batchsize * seql * h * headsize * a_byte, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * seql * headsize * h, 
                   load_weight=0, 
                   load_act=batchsize * seql * h * a_byte, 
                   store_act=batchsize * h * headsize * a_byte, 
                   load_kv_cache=batchsize * seql * headsize * h_kv * kv_byte, 
                   store_kv_cache=0)
        elif op == "O":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * d, 
                   load_weight=d * d * w_byte, 
                   load_act=batchsize * seql * d * a_byte, 
                   store_act=batchsize * seql * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d, 
                   load_weight=d * d * w_byte, 
                   load_act=batchsize * d * a_byte, 
                   store_act=batchsize * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "FFN_up" or op == "FFN_gate":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d * d_ffn, 
                   load_weight=d * d_ffn * w_byte, 
                   load_act=batchsize * seql * d * a_byte, 
                   store_act=batchsize * seql * d_ffn * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d * d_ffn, 
                   load_weight= d * d_ffn * w_byte, 
                   load_act= batchsize * d * a_byte, 
                   store_act= batchsize * d_ffn * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
        elif op == "FFN_down":
            record("prefill", 
                   op, 
                   OPs = 2 * batchsize * seql * d_ffn * d, 
                   load_weight=d_ffn * d * w_byte, 
                   load_act=batchsize * seql * d_ffn * a_byte, 
                   store_act=batchsize * seql * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)
            record("decode", 
                   op, 
                   OPs = 2 * batchsize * d_ffn * d, 
                   load_weight=d_ffn * d * w_byte, 
                   load_act=batchsize * d_ffn * a_byte, 
                   store_act=batchsize * d * a_byte, 
                   load_kv_cache=0, 
                   store_kv_cache=0)

def GetKVCacheSizePerLayer(): # bytes
    return 2 * seql * batchsize * h_kv * headsize * kv_byte

    
if __name__ == "__main__":
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
    print("KV Cache Size: ", round(kvsize / 1024 / 1024 / 1024, 3), "GB")

    print("activation size in decode stage: ", round(batchsize * d * a_byte / 1024 / 1024 / 1024, 3), "GB")
    print("activation size in prefill stage: ", round(batchsize * seql * d * a_byte / 1024 / 1024 / 1024, 3), "GB")
    print("QK intermediate size in decode stage: ", round(batchsize * seql * h * headsize * a_byte / 1024 / 1024 / 1024, 3), "GB")
    print("QK intermediate size in prefill stage: ", round(batchsize * seql * seql * h * headsize * a_byte / 1024 / 1024 / 1024, 3), "GB")
    print("")

    for op in Operations:
        if op not in ['qk_matmul', 'softmax', 'sv_matmul']:
            print(op, "weight size: ", round(weight_size[op] / 1024 / 1024 / 1024, 3), "GB")
    
    print("")
    MemoryBW = 50 
    print("can prefetch", round(t["decode"] * MemoryBW * 1e-6, 3), "GB every layer if network bandwidth is", MemoryBW, "GB/s")