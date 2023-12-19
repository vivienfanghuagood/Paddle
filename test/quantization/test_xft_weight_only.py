import time
import paddle
from paddle.nn.quant import weight_only_linear
from paddle._C_ops import xft_weight_only_linear, xft_weight_quantize

paddle.seed(82)
x = paddle.ones([1, 4, 1024], dtype=paddle.float32).cpu()
weight = paddle.ones([1024, 1024], dtype='float32').cpu() / 10
# for i in range(weight.shape[0]):
#     weight[i] = i

weight_quant, scale, zero_point = xft_weight_quantize(weight, "weight_only_int8")
# 
# abs_max = paddle.max(weight, axis=-1)

# weight_quant, scale = weight_quantize(weight.astype(paddle.bfloat16), "int8")
# scale = paddle.randn([32], dtype='float32')
bias = paddle.cast(paddle.zeros([32]), dtype='float32')
# zero_point = paddle.zeros([32], dtype='float32')

for i in range(5):
    out = xft_weight_only_linear(x, weight_quant, None, scale, zero_point, "int8")
    ref_out = paddle.matmul(x=x, y=weight)

TIMES = 1000
s_time = time.perf_counter()
for i in range(TIMES):
    out = xft_weight_only_linear(x, weight_quant, None, scale, zero_point, "int8")
e_time = time.perf_counter()
ellapse = (e_time - s_time) / TIMES
print(f"xft_weight_only_linear time: {ellapse:4.5f}")


s_time = time.perf_counter()
for i in range(TIMES):
    ref_out = paddle.matmul(x=x, y=weight)
e_time = time.perf_counter()
ellapse = (e_time - s_time) / TIMES
print(f"fp32 linear time:  {ellapse:4.5f}")
# import pdb;pdb.set_trace()
# weight_dequant = weight.T.astype(paddle.float32) * scale
ref_out = paddle.matmul(x=x, y=weight)

# print(out)
# print(ref_out)
# print(out - ref_out)