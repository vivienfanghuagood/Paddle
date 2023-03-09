#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test set_value op in static graph mode

import unittest
import numpy as np

import paddle
from paddle.fluid.layer_helper import LayerHelper
from op_test import OpTest, convert_float_to_uint16


class TestSetValueBF16Base(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.set_value()
        self.set_shape()
        self.data = np.ones(self.shape)
        self.data = convert_float_to_uint16(self.data.astype('float32'))
        self.data[:] = 1
        self.program = paddle.static.Program()

    def set_shape(self):
        self.shape = [2, 3, 4]

    def set_value(self):
        self.value = 6

    def set_dtype(self):
        self.dtype = "uint16"

    def _call_setitem(self, x):
        x[0, 0] = self.value

    def _get_answer(self):
        self.data[0, 0] = self.value


class TestSetValueApi(TestSetValueBF16Base):
    def _run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            self._call_setitem(x)

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        out = exe.run(self.program, fetch_list=[x])
        paddle.disable_static()
        return out

    def _run_dynamic(self):
        paddle.disable_static()
        x = paddle.ones(shape=self.shape, dtype=self.dtype)
        self._call_setitem(x)
        out = x.astype(paddle.float32).numpy().astype(np.uint16)
        paddle.enable_static()
        return out

    def test_api(self):
        dynamic_out = self._run_dynamic()
        self._get_answer()

        error_msg = (
            "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        )
        self.assertTrue(
            (self.data == dynamic_out).all(),
            msg=error_msg.format("dynamic", self.data, dynamic_out),
        )


# 1. Test different type of item: int, Python slice, Paddle Tensor
# 1.1 item is int
class TestSetValueItemInt(TestSetValueApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _get_answer(self):
        self.data[0] = self.value


# 1.2 item is slice
# 1.2.1 step is 1
class TestSetValueItemSlice(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemSlice2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1] = self.value

    def _get_answer(self):
        self.data[0:-1] = self.value


class TestSetValueItemSlice3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = self.value

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemSlice4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:2, :] = self.value


class TestSetValueItemSlice5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:1, :] = self.value

    def _get_answer(self):
        self.data[0:, 1:1, :] = self.value


def create_test_value_bf16(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 125

        def set_dtype(self):
            self.dtype = paddle.bfloat16

    cls_name = "{0}_{1}".format(parent.__name__, "Valuebf16")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_bf16(TestSetValueItemInt)
create_test_value_bf16(TestSetValueItemSlice)
create_test_value_bf16(TestSetValueItemSlice2)
create_test_value_bf16(TestSetValueItemSlice3)
create_test_value_bf16(TestSetValueItemSlice4)


if __name__ == '__main__':
    unittest.main()
