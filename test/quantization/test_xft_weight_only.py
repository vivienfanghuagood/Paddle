import paddle
from paddle.nn.quant import weight_only_linear, weight_quantize
from paddle._C_ops import xft_weight_only_linear

paddle.seed(82)
x = paddle.cast(paddle.randn([1, 10, 64]), dtype='float32') / 100
weight = paddle.randn([32, 64], dtype='float32')
abs_max = paddle.max(weight, axis=-1)

weight_quant, scale = weight_quantize(weight.astype(paddle.bfloat16), "int8")
scale = paddle.randn([32], dtype='float32')
bias = paddle.cast(paddle.zeros([32]), dtype='float32')
zero_point = paddle.zeros([32], dtype='float32')
out = xft_weight_only_linear(x, weight_quant.cpu(), None, scale.cpu(), zero_point, "int8")

# weight_dequant = weight.T.astype(paddle.float32) * scale
ref_out = paddle.matmul(x=x, y=weight) + bias

print(out)
print(ref_out)
print(out - ref_out)