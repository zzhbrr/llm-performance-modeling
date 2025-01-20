# llm-performance-modeling

Use roofline model to predict distributed LLMs inference performance (latency, throughput, memory usage and so on). MoE's modeling supports different parallelism in Attention-part and MoE-part

Change 'workload spec', 'hardware spec', 'parallelization spec' and 'model spec' in codes.

`python modeling_with_parallel.py`

`python modeling_moe.py`

Note: haven't modeling different network bandwidth of intra-node and inter-node.

Note: asserts moe_dp=1 for simplicity.