
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHFT_LIGHTHOUSE'] = 'http://localhost:29510'
os.environ['TORCH_TRACE'] = '/home/ivankobzarev/local/b/torchtitan-autoparallel/torchtitan/outputs/torch_traces/251009-1341'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_LOGS_FORMAT'] = '%(levelname)s: %(message)s'
os.environ['TORCHX_CONFIG_DIR'] = '/home/ivankobzarev/fbsource/fbcode/ai_codesign/oss_infra_launch/torchtitan'
os.environ['TORCHELASTIC_SIGNALS_TO_HANDLE'] = 'SIGTERM,SIGINT,SIGHUP,SIGQUIT'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = 'none'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '3'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_yq8l3v0d/none_s8c6s42c/attempt_0/0/error.json'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_ivankobzarev/tmpnpoku6bi'
os.environ['TORCH_FR_BUFFER_SIZE'] = '20000'
os.environ['TORCH_NCCL_DUMP_ON_TIMEOUT'] = '1'
os.environ['TORCH_NCCL_DEBUG_INFO_TEMP_FILE'] = './outputs/comm_traces/rank_'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_ivankobzarev/tmpnpoku6bi/triton'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore


import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.dont_skip_tracing = False
torch._dynamo.config.allow_rnn = False
torch._dynamo.config.install_free_tensors = False
torch._dynamo.config.constant_fold_autograd_profiler_enabled = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.reorder_for_compute_comm_overlap = False
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['sink_waits_iterative', 'reorder_communication_preserving_peak_memory']
torch._inductor.config.reorder_prefetch_limit = None
torch._inductor.config.bucket_all_gathers_fx = 'none'
torch._inductor.config.bucket_reduce_scatters_fx = 'none'
torch._inductor.config.max_autotune = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.comprehensive_padding = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True



isolate_fails_code_str = None




# torch version: 2.10.0a0+git9cf5200
# torch cuda version: 12.6
# torch git version: 9cf52002cbc6dfb205f828cd09bf3f8ce0b6ec0b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, mm, mm_2, getitem_4, getitem_5, getitem_10, getitem_11, mm_4, add_3, mm_8, getitem_13, getitem_14, getitem_19, getitem_20, mm_10, mm_12, add_7, mm_14, mm_16, getitem_22, getitem_23, getitem_28, getitem_29, mm_18, add_11, mm_22, getitem_31, getitem_32, getitem_37, getitem_38, mm_24, mm_26, add_15, mm_28, mm_30, getitem_40, getitem_41, getitem_46, getitem_47, mm_32, mm_34, mm_36, getitem_49, getitem_50, getitem_55, getitem_56, add_21, mm_40, mm_42, mm_44, getitem_58, getitem_59, getitem_64, getitem_65, add_25, mm_46, mm_48, mm_50, getitem_67, getitem_68, getitem_73, getitem_74, add_29, mm_54, mm_56, mm_58, getitem_76, getitem_77, getitem_82, getitem_83, add_33, mm_60, mm_62, mm_64, getitem_85, getitem_86, getitem_91, getitem_92, add_37, mm_68, mm_70, mm_72, getitem_94, getitem_95, getitem_100, getitem_101, add_41, mm_74, mm_76, mm_78, getitem_103, getitem_104, getitem_109, getitem_110, add_45, mm_82, mm_84, mm_86, getitem_112, getitem_113, getitem_118, getitem_119, add_49, mm_88, mm_90, mm_92, getitem_121, getitem_122, getitem_127, getitem_128, add_53, mm_96, mm_98, mm_100, getitem_130, getitem_131, getitem_136, getitem_137, add_57, mm_102, mm_104, mm_106, getitem_139, getitem_140, getitem_145, getitem_146, mm_108, mm_110, tangents_1):
        view_402 = torch.ops.aten.view.default(tangents_1, [16384, 64128]);  tangents_1 = None
        permute_177 = torch.ops.aten.permute.default(view_402, [1, 0])
        view_374 = torch.ops.aten.view.default(mm_104, [2, 8192, 8192]);  mm_104 = None
        reduce_scatter_tensor_14 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_374, 'sum', 2, '3');  view_374 = None
        wait_tensor_167 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_14);  reduce_scatter_tensor_14 = None
        add_59 = torch.ops.aten.add.Tensor(add_57, wait_tensor_167);  wait_tensor_167 = None
        view_393 = torch.ops.aten.view.default(mm_108, [1, 8192, 8192]);  mm_108 = None
        add_61 = torch.ops.aten.add.Tensor(add_59, view_393);  view_393 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_516 = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_516, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_62 = torch.ops.aten.add.Scalar(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_516, rsqrt_31);  convert_element_type_516 = None
        all_gather_into_tensor_158 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_515, 8, '0');  convert_element_type_515 = None
        wait_tensor_173 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_158);  all_gather_into_tensor_158 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, wait_tensor_173)
        convert_element_type_517 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        all_gather_into_tensor_159 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_518, 4, '1');  convert_element_type_518 = None
        wait_tensor_174 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_159);  all_gather_into_tensor_159 = None
        permute_173 = torch.ops.aten.permute.default(wait_tensor_174, [1, 0]);  wait_tensor_174 = None
        all_gather_into_tensor_160 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_517, 2, '3');  convert_element_type_517 = None
        wait_tensor_175 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_160);  all_gather_into_tensor_160 = None
        view_394 = torch.ops.aten.view.default(wait_tensor_175, [16384, 8192]);  wait_tensor_175 = None
        mm_109 = torch.ops.aten.mm.default(view_394, permute_173)
        view_395 = torch.ops.aten.view.default(mm_109, [2, 8192, 14336]);  mm_109 = None
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_521)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_521, sigmoid_15);  sigmoid_15 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        view_397 = torch.ops.aten.view.default(mm_110, [2, 8192, 14336]);  mm_110 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_522, view_397)
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        permute_175 = torch.ops.aten.permute.default(convert_element_type_526, [1, 0]);  convert_element_type_526 = None
        view_398 = torch.ops.aten.view.default(mul_127, [16384, 14336]);  mul_127 = None
        clone_63 = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        all_gather_into_tensor_162 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_63, 4, '1');  clone_63 = None
        wait_tensor_177 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_162);  all_gather_into_tensor_162 = None
        mm_111 = torch.ops.aten.mm.default(view_398, wait_tensor_177)
        view_399 = torch.ops.aten.view.default(mm_111, [2, 8192, 8192]);  mm_111 = None
        reduce_scatter_tensor_15 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_399, 'sum', 2, '3');  view_399 = None
        wait_tensor_178 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_15);  reduce_scatter_tensor_15 = None
        add_63 = torch.ops.aten.add.Tensor(add_61, wait_tensor_178);  add_61 = wait_tensor_178 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_530, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_33, [2], True);  pow_33 = None
        add_64 = torch.ops.aten.add.Scalar(mean_32, 1e-05);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_32);  convert_element_type_530 = None
        all_gather_into_tensor_163 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_529, 8, '0');  convert_element_type_529 = None
        wait_tensor_179 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_163);  all_gather_into_tensor_163 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, wait_tensor_179)
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        view_400 = torch.ops.aten.view.default(convert_element_type_531, [8192, 8192]);  convert_element_type_531 = None
        all_gather_into_tensor_165 = torch.ops._c10d_functional.all_gather_into_tensor.default(view_400, 2, '3');  view_400 = None
        wait_tensor_181 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_165);  all_gather_into_tensor_165 = None
        mm_113 = torch.ops.aten.mm.default(permute_177, wait_tensor_181);  permute_177 = wait_tensor_181 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        all_gather_into_tensor_164 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_532, 4, '1');  convert_element_type_532 = None
        wait_tensor_180 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_164);  all_gather_into_tensor_164 = None
        permute_176 = torch.ops.aten.permute.default(wait_tensor_180, [1, 0]);  wait_tensor_180 = None
        permute_179 = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_114 = torch.ops.aten.mm.default(view_402, permute_179);  view_402 = permute_179 = None
        view_403 = torch.ops.aten.view.default(mm_114, [2, 8192, 8192]);  mm_114 = None
        reduce_scatter_tensor_16 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_113, 'sum', 4, '1');  mm_113 = None
        wait_tensor_182 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_16);  reduce_scatter_tensor_16 = None
        convert_element_type_539 = torch.ops.prims.convert_element_type.default(wait_tensor_182, torch.float32);  wait_tensor_182 = None
        reduce_scatter_tensor_17 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_403, 'sum', 2, '3');  view_403 = None
        wait_tensor_183 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_17);  reduce_scatter_tensor_17 = None
        convert_element_type_540 = torch.ops.prims.convert_element_type.default(wait_tensor_183, torch.float32);  wait_tensor_183 = None
        convert_element_type_542 = torch.ops.prims.convert_element_type.default(wait_tensor_179, torch.float32);  wait_tensor_179 = None
        mul_130 = torch.ops.aten.mul.Tensor(convert_element_type_540, convert_element_type_542);  convert_element_type_542 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_128, mul_130)
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
        div = torch.ops.aten.div.Tensor(mul_128, 8192)
        mul_133 = torch.ops.aten.mul.Tensor(div, sum_1);  div = sum_1 = None
        sub = torch.ops.aten.sub.Tensor(mul_130, mul_133);  mul_130 = mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(sub, rsqrt_32);  sub = rsqrt_32 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_540, mul_128);  convert_element_type_540 = mul_128 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
        convert_element_type_543 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_default_33 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        reduce_scatter_tensor_18 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_33, 'sum', 8, '0');  convert_element_type_default_33 = None
        wait_tensor_184 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_18);  reduce_scatter_tensor_18 = None
        all_gather_into_tensor_166 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_543, 2, '3')
        wait_tensor_185 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_166);  all_gather_into_tensor_166 = None
        view_404 = torch.ops.aten.view.default(wait_tensor_185, [16384, 8192]);  wait_tensor_185 = None
        permute_181 = torch.ops.aten.permute.default(view_404, [1, 0])
        mm_115 = torch.ops.aten.mm.default(permute_181, view_398);  permute_181 = view_398 = None
        permute_182 = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
        permute_183 = torch.ops.aten.permute.default(wait_tensor_177, [1, 0]);  wait_tensor_177 = None
        mm_116 = torch.ops.aten.mm.default(view_404, permute_183);  view_404 = permute_183 = None
        view_405 = torch.ops.aten.view.default(mm_116, [2, 8192, 14336]);  mm_116 = None
        clone_64 = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        reduce_scatter_tensor_19 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_64, 'sum', 4, '1');  clone_64 = None
        wait_tensor_186 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_19);  reduce_scatter_tensor_19 = None
        permute_184 = torch.ops.aten.permute.default(wait_tensor_186, [1, 0]);  wait_tensor_186 = None
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(permute_184, torch.float32);  permute_184 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_405, convert_element_type_522);  convert_element_type_522 = None
        mul_137 = torch.ops.aten.mul.Tensor(view_405, view_397);  view_405 = view_397 = None
        view_406 = torch.ops.aten.view.default(mul_136, [16384, 14336]);  mul_136 = None
        permute_185 = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_185, view_394);  permute_185 = None
        reduce_scatter_tensor_20 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_117, 'sum', 4, '1');  mm_117 = None
        wait_tensor_187 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_20);  reduce_scatter_tensor_20 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        all_gather_into_tensor_161 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_523, 4, '1');  convert_element_type_523 = None
        wait_tensor_176 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_161);  all_gather_into_tensor_161 = None
        permute_174 = torch.ops.aten.permute.default(wait_tensor_176, [1, 0]);  wait_tensor_176 = None
        permute_187 = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_118 = torch.ops.aten.mm.default(view_406, permute_187);  view_406 = permute_187 = None
        view_407 = torch.ops.aten.view.default(mm_118, [2, 8192, 8192]);  mm_118 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(wait_tensor_187, torch.float32);  wait_tensor_187 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(mul_137, torch.float32);  mul_137 = None
        neg = torch.ops.aten.neg.default(convert_element_type_521)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add_65 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal = torch.ops.aten.reciprocal.default(add_65);  add_65 = None
        mul_138 = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_556, mul_138);  convert_element_type_556 = None
        sub_1 = torch.ops.aten.sub.Tensor(1, mul_138);  mul_138 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_521, sub_1);  convert_element_type_521 = sub_1 = None
        add_66 = torch.ops.aten.add.Tensor(mul_140, 1);  mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_66);  mul_139 = add_66 = None
        convert_element_type_558 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        view_408 = torch.ops.aten.view.default(convert_element_type_558, [16384, 14336]);  convert_element_type_558 = None
        permute_189 = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_119 = torch.ops.aten.mm.default(permute_189, view_394);  permute_189 = view_394 = None
        reduce_scatter_tensor_21 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_119, 'sum', 4, '1');  mm_119 = None
        wait_tensor_188 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_21);  reduce_scatter_tensor_21 = None
        permute_191 = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_120 = torch.ops.aten.mm.default(view_408, permute_191);  view_408 = permute_191 = None
        view_409 = torch.ops.aten.view.default(mm_120, [2, 8192, 8192]);  mm_120 = None
        add_67 = torch.ops.aten.add.Tensor(view_407, view_409);  view_407 = view_409 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(wait_tensor_188, torch.float32);  wait_tensor_188 = None
        reduce_scatter_tensor_22 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_67, 'sum', 2, '3');  add_67 = None
        wait_tensor_189 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_22);  reduce_scatter_tensor_22 = None
        convert_element_type_564 = torch.ops.prims.convert_element_type.default(wait_tensor_189, torch.float32);  wait_tensor_189 = None
        convert_element_type_566 = torch.ops.prims.convert_element_type.default(wait_tensor_173, torch.float32);  wait_tensor_173 = None
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_564, convert_element_type_566);  convert_element_type_566 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_124, mul_142)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
        div_1 = torch.ops.aten.div.Tensor(mul_124, 8192)
        mul_145 = torch.ops.aten.mul.Tensor(div_1, sum_3);  div_1 = sum_3 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_142, mul_145);  mul_142 = mul_145 = None
        mul_146 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_31);  sub_2 = rsqrt_31 = None
        mul_147 = torch.ops.aten.mul.Tensor(convert_element_type_564, mul_124);  convert_element_type_564 = mul_124 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
        convert_element_type_567 = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None
        add_68 = torch.ops.aten.add.Tensor(convert_element_type_543, convert_element_type_567);  convert_element_type_543 = convert_element_type_567 = None
        convert_element_type_default_32 = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        reduce_scatter_tensor_23 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_32, 'sum', 8, '0');  convert_element_type_default_32 = None
        wait_tensor_190 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_23);  reduce_scatter_tensor_23 = None
        view_410 = torch.ops.aten.view.default(add_68, [8192, 8192])
        permute_193 = torch.ops.aten.permute.default(view_410, [1, 0])
        permute_171 = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3])
        view_391 = torch.ops.aten.view.default(permute_171, [1, 8192, 8192]);  permute_171 = None
        view_392 = torch.ops.aten.view.default(view_391, [8192, 8192]);  view_391 = None
        mm_121 = torch.ops.aten.mm.default(permute_193, view_392);  permute_193 = view_392 = None
        permute_194 = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        permute_172 = torch.ops.aten.permute.default(convert_element_type_512, [1, 0]);  convert_element_type_512 = None
        clone_62 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        all_gather_into_tensor_157 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_62, 8, '0');  clone_62 = None
        wait_tensor_172 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_157);  all_gather_into_tensor_157 = None
        permute_195 = torch.ops.aten.permute.default(wait_tensor_172, [1, 0]);  wait_tensor_172 = None
        mm_122 = torch.ops.aten.mm.default(view_410, permute_195);  view_410 = permute_195 = None
        view_411 = torch.ops.aten.view.default(mm_122, [1, 8192, 8192]);  mm_122 = None
        clone_65 = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        reduce_scatter_tensor_24 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_65, 'sum', 8, '0');  clone_65 = None
        wait_tensor_191 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_24);  reduce_scatter_tensor_24 = None
        permute_196 = torch.ops.aten.permute.default(wait_tensor_191, [1, 0]);  wait_tensor_191 = None
        convert_element_type_574 = torch.ops.prims.convert_element_type.default(permute_196, torch.float32);  permute_196 = None
        view_412 = torch.ops.aten.view.default(view_411, [1, 8192, 32, 256]);  view_411 = None
        permute_197 = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        view_11 = torch.ops.aten.view.default(primals_148, [1, 8192, 1, 128]);  primals_148 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [2], True);  pow_31 = None
        add_60 = torch.ops.aten.add.Scalar(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_30);  convert_element_type_497 = None
        all_gather_into_tensor_153 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_496, 8, '0');  convert_element_type_496 = None
        wait_tensor_168 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_153);  all_gather_into_tensor_153 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, wait_tensor_168)
        convert_element_type_498 = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        convert_element_type_499 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        all_gather_into_tensor_154 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_499, 8, '0');  convert_element_type_499 = None
        wait_tensor_169 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_154);  all_gather_into_tensor_154 = None
        permute_165 = torch.ops.aten.permute.default(wait_tensor_169, [1, 0]);  wait_tensor_169 = None
        view_375 = torch.ops.aten.view.default(convert_element_type_498, [8192, 8192]);  convert_element_type_498 = None
        mm_105 = torch.ops.aten.mm.default(view_375, permute_165)
        view_376 = torch.ops.aten.view.default(mm_105, [1, 8192, 8192]);  mm_105 = None
        view_378 = torch.ops.aten.view.default(mm_106, [1, 8192, 2048]);  mm_106 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        all_gather_into_tensor_156 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_505, 8, '0');  convert_element_type_505 = None
        wait_tensor_171 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_156);  all_gather_into_tensor_156 = None
        permute_167 = torch.ops.aten.permute.default(wait_tensor_171, [1, 0]);  wait_tensor_171 = None
        mm_107 = torch.ops.aten.mm.default(view_375, permute_167)
        view_380 = torch.ops.aten.view.default(mm_107, [1, 8192, 2048]);  mm_107 = None
        view_381 = torch.ops.aten.view.default(view_376, [1, 8192, 32, 256]);  view_376 = None
        view_382 = torch.ops.aten.view.default(view_378, [1, 8192, 8, 256]);  view_378 = None
        view_383 = torch.ops.aten.view.default(view_380, [1, 8192, 8, 256]);  view_380 = None
        convert_element_type_508 = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        view_384 = torch.ops.aten.view.default(convert_element_type_508, [1, 8192, 32, 128, 2]);  convert_element_type_508 = None
        view_as_complex_30 = torch.ops.aten.view_as_complex.default(view_384);  view_384 = None
        convert_element_type_509 = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        view_385 = torch.ops.aten.view.default(convert_element_type_509, [1, 8192, 8, 128, 2]);  convert_element_type_509 = None
        view_as_complex_31 = torch.ops.aten.view_as_complex.default(view_385);  view_385 = None
        mul_122 = torch.ops.aten.mul.Tensor(view_as_complex_30, view_11);  view_as_complex_30 = None
        view_as_real_30 = torch.ops.aten.view_as_real.default(mul_122);  mul_122 = None
        view_387 = torch.ops.aten.view.default(view_as_real_30, [1, 8192, 32, 256]);  view_as_real_30 = None
        mul_123 = torch.ops.aten.mul.Tensor(view_as_complex_31, view_11);  view_as_complex_31 = None
        view_as_real_31 = torch.ops.aten.view_as_real.default(mul_123);  mul_123 = None
        view_388 = torch.ops.aten.view.default(view_as_real_31, [1, 8192, 8, 256]);  view_as_real_31 = None
        convert_element_type_510 = torch.ops.prims.convert_element_type.default(view_387, torch.bfloat16);  view_387 = None
        convert_element_type_511 = torch.ops.prims.convert_element_type.default(view_388, torch.bfloat16);  view_388 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(convert_element_type_511, 3);  convert_element_type_511 = None
        expand_30 = torch.ops.aten.expand.default(unsqueeze_30, [1, 8192, 8, 4, 256]);  unsqueeze_30 = None
        clone_60 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_389 = torch.ops.aten.view.default(clone_60, [1, 8192, 32, 256]);  clone_60 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(view_383, 3);  view_383 = None
        expand_31 = torch.ops.aten.expand.default(unsqueeze_31, [1, 8192, 8, 4, 256]);  unsqueeze_31 = None
        clone_61 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_390 = torch.ops.aten.view.default(clone_61, [1, 8192, 32, 256]);  clone_61 = None
        permute_168 = torch.ops.aten.permute.default(convert_element_type_510, [0, 2, 1, 3]);  convert_element_type_510 = None
        permute_169 = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
        permute_170 = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_197, permute_168, permute_169, permute_170, getitem_139, getitem_140, None, None, 8192, 8192, 0.0, True, getitem_145, getitem_146, scale = 0.0625);  permute_197 = permute_168 = permute_169 = permute_170 = getitem_139 = getitem_140 = getitem_145 = getitem_146 = None
        getitem_148 = _scaled_dot_product_flash_attention_backward[0]
        getitem_149 = _scaled_dot_product_flash_attention_backward[1]
        getitem_150 = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        permute_198 = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        permute_199 = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        permute_200 = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        view_413 = torch.ops.aten.view.default(permute_198, [1, 8192, 8, 4, 256]);  permute_198 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_413, [3], True);  view_413 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_5, 3);  sum_5 = None
        view_414 = torch.ops.aten.view.default(permute_199, [1, 8192, 8, 4, 256]);  permute_199 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_414, [3], True);  view_414 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_6, 3);  sum_6 = None
        convert_element_type_575 = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(permute_200, torch.float32);  permute_200 = None
        view_415 = torch.ops.aten.view.default(convert_element_type_575, [1, 8192, 8, 128, 2]);  convert_element_type_575 = None
        view_as_complex_32 = torch.ops.aten.view_as_complex.default(view_415);  view_415 = None
        _conj = torch.ops.aten._conj.default(view_11)
        mul_148 = torch.ops.aten.mul.Tensor(view_as_complex_32, _conj);  view_as_complex_32 = None
        view_416 = torch.ops.aten.view.default(convert_element_type_576, [1, 8192, 32, 128, 2]);  convert_element_type_576 = None
        view_as_complex_33 = torch.ops.aten.view_as_complex.default(view_416);  view_416 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_as_complex_33, _conj);  view_as_complex_33 = None
        view_as_real_32 = torch.ops.aten.view_as_real.default(mul_148);  mul_148 = None
        view_417 = torch.ops.aten.view.default(view_as_real_32, [1, 8192, 8, 256]);  view_as_real_32 = None
        convert_element_type_577 = torch.ops.prims.convert_element_type.default(view_417, torch.bfloat16);  view_417 = None
        view_as_real_33 = torch.ops.aten.view_as_real.default(mul_149);  mul_149 = None
        view_418 = torch.ops.aten.view.default(view_as_real_33, [1, 8192, 32, 256]);  view_as_real_33 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419 = torch.ops.aten.view.default(squeeze, [1, 8192, 2048]);  squeeze = None
        view_420 = torch.ops.aten.view.default(convert_element_type_577, [1, 8192, 2048]);  convert_element_type_577 = None
        view_421 = torch.ops.aten.view.default(convert_element_type_578, [1, 8192, 8192]);  convert_element_type_578 = None
        view_422 = torch.ops.aten.view.default(view_419, [8192, 2048]);  view_419 = None
        permute_201 = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_201, view_375);  permute_201 = None
        permute_203 = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_124 = torch.ops.aten.mm.default(view_422, permute_203);  view_422 = permute_203 = None
        view_423 = torch.ops.aten.view.default(mm_124, [1, 8192, 8192]);  mm_124 = None
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        reduce_scatter_tensor_25 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_583, 'sum', 8, '0');  convert_element_type_583 = None
        wait_tensor_192 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_25);  reduce_scatter_tensor_25 = None
        view_424 = torch.ops.aten.view.default(view_420, [8192, 2048]);  view_420 = None
        permute_205 = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_205, view_375);  permute_205 = None
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        all_gather_into_tensor_155 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_502, 8, '0');  convert_element_type_502 = None
        wait_tensor_170 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_155);  all_gather_into_tensor_155 = None
        permute_166 = torch.ops.aten.permute.default(wait_tensor_170, [1, 0]);  wait_tensor_170 = None
        permute_207 = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_126 = torch.ops.aten.mm.default(view_424, permute_207);  view_424 = permute_207 = None
        view_425 = torch.ops.aten.view.default(mm_126, [1, 8192, 8192]);  mm_126 = None
        add_69 = torch.ops.aten.add.Tensor(view_423, view_425);  view_423 = view_425 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        reduce_scatter_tensor_26 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_588, 'sum', 8, '0');  convert_element_type_588 = None
        wait_tensor_193 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_26);  reduce_scatter_tensor_26 = None
        view_426 = torch.ops.aten.view.default(view_421, [8192, 8192]);  view_421 = None
        permute_209 = torch.ops.aten.permute.default(view_426, [1, 0])
        mm_127 = torch.ops.aten.mm.default(permute_209, view_375);  permute_209 = view_375 = None
        permute_211 = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_128 = torch.ops.aten.mm.default(view_426, permute_211);  view_426 = permute_211 = None
        view_427 = torch.ops.aten.view.default(mm_128, [1, 8192, 8192]);  mm_128 = None
        add_70 = torch.ops.aten.add.Tensor(add_69, view_427);  add_69 = view_427 = None
        reduce_scatter_tensor_27 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_127, 'sum', 8, '0');  mm_127 = None
        wait_tensor_194 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_27);  reduce_scatter_tensor_27 = None
        convert_element_type_593 = torch.ops.prims.convert_element_type.default(wait_tensor_194, torch.float32);  wait_tensor_194 = None
        convert_element_type_594 = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(wait_tensor_168, torch.float32);  wait_tensor_168 = None
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_594, convert_element_type_596);  convert_element_type_596 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_120, mul_150)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
        div_2 = torch.ops.aten.div.Tensor(mul_120, 8192)
        mul_153 = torch.ops.aten.mul.Tensor(div_2, sum_7);  div_2 = sum_7 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_150, mul_153);  mul_150 = mul_153 = None
        mul_154 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_30);  sub_3 = rsqrt_30 = None
        mul_155 = torch.ops.aten.mul.Tensor(convert_element_type_594, mul_120);  convert_element_type_594 = mul_120 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        add_71 = torch.ops.aten.add.Tensor(add_68, convert_element_type_597);  add_68 = convert_element_type_597 = None
        convert_element_type_default_31 = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        reduce_scatter_tensor_28 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_31, 'sum', 8, '0');  convert_element_type_default_31 = None
        wait_tensor_195 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_28);  reduce_scatter_tensor_28 = None
        all_gather_into_tensor_167 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_71, 2, '3')
        wait_tensor_196 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_167);  all_gather_into_tensor_167 = None
        view_428 = torch.ops.aten.view.default(wait_tensor_196, [16384, 8192]);  wait_tensor_196 = None
        permute_213 = torch.ops.aten.permute.default(view_428, [1, 0])
        convert_element_type_482 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_58 = torch.ops.aten.add.Scalar(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_29);  convert_element_type_483 = None
        all_gather_into_tensor_148 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_482, 8, '0');  convert_element_type_482 = None
        wait_tensor_162 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_148);  all_gather_into_tensor_148 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, wait_tensor_162)
        convert_element_type_484 = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        all_gather_into_tensor_150 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_484, 2, '3');  convert_element_type_484 = None
        wait_tensor_164 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_150);  all_gather_into_tensor_150 = None
        view_369 = torch.ops.aten.view.default(wait_tensor_164, [16384, 8192]);  wait_tensor_164 = None
        view_370 = torch.ops.aten.view.default(mm_102, [2, 8192, 14336]);  mm_102 = None
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_488)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_488, sigmoid_14);  sigmoid_14 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        all_gather_into_tensor_151 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_490, 4, '1');  convert_element_type_490 = None
        wait_tensor_165 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_151);  all_gather_into_tensor_151 = None
        permute_163 = torch.ops.aten.permute.default(wait_tensor_165, [1, 0]);  wait_tensor_165 = None
        mm_103 = torch.ops.aten.mm.default(view_369, permute_163)
        view_372 = torch.ops.aten.view.default(mm_103, [2, 8192, 14336]);  mm_103 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_489, view_372)
        view_373 = torch.ops.aten.view.default(mul_119, [16384, 14336]);  mul_119 = None
        mm_129 = torch.ops.aten.mm.default(permute_213, view_373);  permute_213 = view_373 = None
        permute_214 = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        permute_164 = torch.ops.aten.permute.default(convert_element_type_493, [1, 0]);  convert_element_type_493 = None
        clone_59 = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        all_gather_into_tensor_152 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_59, 4, '1');  clone_59 = None
        wait_tensor_166 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_152);  all_gather_into_tensor_152 = None
        permute_215 = torch.ops.aten.permute.default(wait_tensor_166, [1, 0]);  wait_tensor_166 = None
        mm_130 = torch.ops.aten.mm.default(view_428, permute_215);  view_428 = permute_215 = None
        view_429 = torch.ops.aten.view.default(mm_130, [2, 8192, 14336]);  mm_130 = None
        clone_68 = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        reduce_scatter_tensor_29 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_68, 'sum', 4, '1');  clone_68 = None
        wait_tensor_197 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_29);  reduce_scatter_tensor_29 = None
        permute_216 = torch.ops.aten.permute.default(wait_tensor_197, [1, 0]);  wait_tensor_197 = None
        convert_element_type_604 = torch.ops.prims.convert_element_type.default(permute_216, torch.float32);  permute_216 = None
        mul_156 = torch.ops.aten.mul.Tensor(view_429, convert_element_type_489);  convert_element_type_489 = None
        mul_157 = torch.ops.aten.mul.Tensor(view_429, view_372);  view_429 = view_372 = None
        view_430 = torch.ops.aten.view.default(mul_156, [16384, 14336]);  mul_156 = None
        permute_217 = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_217, view_369);  permute_217 = None
        reduce_scatter_tensor_30 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_131, 'sum', 4, '1');  mm_131 = None
        wait_tensor_198 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_30);  reduce_scatter_tensor_30 = None
        permute_219 = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_132 = torch.ops.aten.mm.default(view_430, permute_219);  view_430 = permute_219 = None
        view_431 = torch.ops.aten.view.default(mm_132, [2, 8192, 8192]);  mm_132 = None
        convert_element_type_609 = torch.ops.prims.convert_element_type.default(wait_tensor_198, torch.float32);  wait_tensor_198 = None
        convert_element_type_610 = torch.ops.prims.convert_element_type.default(mul_157, torch.float32);  mul_157 = None
        neg_1 = torch.ops.aten.neg.default(convert_element_type_488)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_72 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(add_72);  add_72 = None
        mul_158 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_610, mul_158);  convert_element_type_610 = None
        sub_4 = torch.ops.aten.sub.Tensor(1, mul_158);  mul_158 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_488, sub_4);  convert_element_type_488 = sub_4 = None
        add_73 = torch.ops.aten.add.Tensor(mul_160, 1);  mul_160 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_73);  mul_159 = add_73 = None
        convert_element_type_612 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        view_432 = torch.ops.aten.view.default(convert_element_type_612, [16384, 14336]);  convert_element_type_612 = None
        permute_221 = torch.ops.aten.permute.default(view_432, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_221, view_369);  permute_221 = view_369 = None
        reduce_scatter_tensor_31 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_133, 'sum', 4, '1');  mm_133 = None
        wait_tensor_199 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_31);  reduce_scatter_tensor_31 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        all_gather_into_tensor_149 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_485, 4, '1');  convert_element_type_485 = None
        wait_tensor_163 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_149);  all_gather_into_tensor_149 = None
        permute_162 = torch.ops.aten.permute.default(wait_tensor_163, [1, 0]);  wait_tensor_163 = None
        permute_223 = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_134 = torch.ops.aten.mm.default(view_432, permute_223);  view_432 = permute_223 = None
        view_433 = torch.ops.aten.view.default(mm_134, [2, 8192, 8192]);  mm_134 = None
        add_74 = torch.ops.aten.add.Tensor(view_431, view_433);  view_431 = view_433 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(wait_tensor_199, torch.float32);  wait_tensor_199 = None
        reduce_scatter_tensor_32 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_74, 'sum', 2, '3');  add_74 = None
        wait_tensor_200 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_32);  reduce_scatter_tensor_32 = None
        convert_element_type_618 = torch.ops.prims.convert_element_type.default(wait_tensor_200, torch.float32);  wait_tensor_200 = None
        convert_element_type_620 = torch.ops.prims.convert_element_type.default(wait_tensor_162, torch.float32);  wait_tensor_162 = None
        mul_162 = torch.ops.aten.mul.Tensor(convert_element_type_618, convert_element_type_620);  convert_element_type_620 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_116, mul_162)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_164, [2], True);  mul_164 = None
        div_3 = torch.ops.aten.div.Tensor(mul_116, 8192)
        mul_165 = torch.ops.aten.mul.Tensor(div_3, sum_9);  div_3 = sum_9 = None
        sub_5 = torch.ops.aten.sub.Tensor(mul_162, mul_165);  mul_162 = mul_165 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_29);  sub_5 = rsqrt_29 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_618, mul_116);  convert_element_type_618 = mul_116 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1]);  mul_167 = None
        convert_element_type_621 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        add_75 = torch.ops.aten.add.Tensor(add_71, convert_element_type_621);  add_71 = convert_element_type_621 = None
        convert_element_type_default_30 = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
        reduce_scatter_tensor_33 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_30, 'sum', 8, '0');  convert_element_type_default_30 = None
        wait_tensor_201 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_33);  reduce_scatter_tensor_33 = None
        view_434 = torch.ops.aten.view.default(add_75, [8192, 8192])
        permute_225 = torch.ops.aten.permute.default(view_434, [1, 0])
        permute_160 = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3])
        view_366 = torch.ops.aten.view.default(permute_160, [1, 8192, 8192]);  permute_160 = None
        view_367 = torch.ops.aten.view.default(view_366, [8192, 8192]);  view_366 = None
        mm_135 = torch.ops.aten.mm.default(permute_225, view_367);  permute_225 = view_367 = None
        permute_226 = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        permute_161 = torch.ops.aten.permute.default(convert_element_type_479, [1, 0]);  convert_element_type_479 = None
        clone_58 = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        all_gather_into_tensor_147 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_58, 8, '0');  clone_58 = None
        wait_tensor_161 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_147);  all_gather_into_tensor_147 = None
        permute_227 = torch.ops.aten.permute.default(wait_tensor_161, [1, 0]);  wait_tensor_161 = None
        mm_136 = torch.ops.aten.mm.default(view_434, permute_227);  view_434 = permute_227 = None
        view_435 = torch.ops.aten.view.default(mm_136, [1, 8192, 8192]);  mm_136 = None
        clone_69 = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        reduce_scatter_tensor_34 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_69, 'sum', 8, '0');  clone_69 = None
        wait_tensor_202 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_34);  reduce_scatter_tensor_34 = None
        permute_228 = torch.ops.aten.permute.default(wait_tensor_202, [1, 0]);  wait_tensor_202 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(permute_228, torch.float32);  permute_228 = None
        view_436 = torch.ops.aten.view.default(view_435, [1, 8192, 32, 256]);  view_435 = None
        permute_229 = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
        convert_element_type_449 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_450, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [2], True);  pow_28 = None
        add_54 = torch.ops.aten.add.Scalar(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_27);  convert_element_type_450 = None
        all_gather_into_tensor_138 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_449, 8, '0');  convert_element_type_449 = None
        wait_tensor_151 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_138);  all_gather_into_tensor_138 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, wait_tensor_151)
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        all_gather_into_tensor_139 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_452, 4, '1');  convert_element_type_452 = None
        wait_tensor_152 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_139);  all_gather_into_tensor_139 = None
        permute_151 = torch.ops.aten.permute.default(wait_tensor_152, [1, 0]);  wait_tensor_152 = None
        all_gather_into_tensor_140 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_451, 2, '3');  convert_element_type_451 = None
        wait_tensor_153 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_140);  all_gather_into_tensor_140 = None
        view_344 = torch.ops.aten.view.default(wait_tensor_153, [16384, 8192]);  wait_tensor_153 = None
        mm_95 = torch.ops.aten.mm.default(view_344, permute_151)
        view_345 = torch.ops.aten.view.default(mm_95, [2, 8192, 14336]);  mm_95 = None
        convert_element_type_455 = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_455)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_13);  sigmoid_13 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        view_347 = torch.ops.aten.view.default(mm_96, [2, 8192, 14336]);  mm_96 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_456, view_347)
        convert_element_type_460 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        permute_153 = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        view_348 = torch.ops.aten.view.default(mul_111, [16384, 14336]);  mul_111 = None
        clone_55 = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        all_gather_into_tensor_142 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_55, 4, '1');  clone_55 = None
        wait_tensor_155 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_142);  all_gather_into_tensor_142 = None
        mm_97 = torch.ops.aten.mm.default(view_348, wait_tensor_155)
        view_349 = torch.ops.aten.view.default(mm_97, [2, 8192, 8192]);  mm_97 = None
        reduce_scatter_tensor_13 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_349, 'sum', 2, '3');  view_349 = None
        wait_tensor_156 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_13);  reduce_scatter_tensor_13 = None
        add_55 = torch.ops.aten.add.Tensor(add_53, wait_tensor_156);  add_53 = wait_tensor_156 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_464 = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [2], True);  pow_29 = None
        add_56 = torch.ops.aten.add.Scalar(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_28);  convert_element_type_464 = None
        all_gather_into_tensor_143 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_463, 8, '0');  convert_element_type_463 = None
        wait_tensor_157 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_143);  all_gather_into_tensor_143 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, wait_tensor_157)
        convert_element_type_465 = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        view_350 = torch.ops.aten.view.default(convert_element_type_465, [8192, 8192]);  convert_element_type_465 = None
        view_351 = torch.ops.aten.view.default(mm_98, [1, 8192, 8192]);  mm_98 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        all_gather_into_tensor_145 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_469, 8, '0');  convert_element_type_469 = None
        wait_tensor_159 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_145);  all_gather_into_tensor_145 = None
        permute_155 = torch.ops.aten.permute.default(wait_tensor_159, [1, 0]);  wait_tensor_159 = None
        mm_99 = torch.ops.aten.mm.default(view_350, permute_155)
        view_353 = torch.ops.aten.view.default(mm_99, [1, 8192, 2048]);  mm_99 = None
        view_355 = torch.ops.aten.view.default(mm_100, [1, 8192, 2048]);  mm_100 = None
        view_356 = torch.ops.aten.view.default(view_351, [1, 8192, 32, 256]);  view_351 = None
        view_357 = torch.ops.aten.view.default(view_353, [1, 8192, 8, 256]);  view_353 = None
        view_358 = torch.ops.aten.view.default(view_355, [1, 8192, 8, 256]);  view_355 = None
        convert_element_type_475 = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        view_359 = torch.ops.aten.view.default(convert_element_type_475, [1, 8192, 32, 128, 2]);  convert_element_type_475 = None
        view_as_complex_28 = torch.ops.aten.view_as_complex.default(view_359);  view_359 = None
        convert_element_type_476 = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        view_360 = torch.ops.aten.view.default(convert_element_type_476, [1, 8192, 8, 128, 2]);  convert_element_type_476 = None
        view_as_complex_29 = torch.ops.aten.view_as_complex.default(view_360);  view_360 = None
        mul_114 = torch.ops.aten.mul.Tensor(view_as_complex_28, view_11);  view_as_complex_28 = None
        view_as_real_28 = torch.ops.aten.view_as_real.default(mul_114);  mul_114 = None
        view_362 = torch.ops.aten.view.default(view_as_real_28, [1, 8192, 32, 256]);  view_as_real_28 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_as_complex_29, view_11);  view_as_complex_29 = None
        view_as_real_29 = torch.ops.aten.view_as_real.default(mul_115);  mul_115 = None
        view_363 = torch.ops.aten.view.default(view_as_real_29, [1, 8192, 8, 256]);  view_as_real_29 = None
        convert_element_type_477 = torch.ops.prims.convert_element_type.default(view_362, torch.bfloat16);  view_362 = None
        convert_element_type_478 = torch.ops.prims.convert_element_type.default(view_363, torch.bfloat16);  view_363 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(convert_element_type_478, 3);  convert_element_type_478 = None
        expand_28 = torch.ops.aten.expand.default(unsqueeze_28, [1, 8192, 8, 4, 256]);  unsqueeze_28 = None
        clone_56 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_364 = torch.ops.aten.view.default(clone_56, [1, 8192, 32, 256]);  clone_56 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(view_358, 3);  view_358 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_29, [1, 8192, 8, 4, 256]);  unsqueeze_29 = None
        clone_57 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_365 = torch.ops.aten.view.default(clone_57, [1, 8192, 32, 256]);  clone_57 = None
        permute_157 = torch.ops.aten.permute.default(convert_element_type_477, [0, 2, 1, 3]);  convert_element_type_477 = None
        permute_158 = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        permute_159 = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_229, permute_157, permute_158, permute_159, getitem_130, getitem_131, None, None, 8192, 8192, 0.0, True, getitem_136, getitem_137, scale = 0.0625);  permute_229 = permute_157 = permute_158 = permute_159 = getitem_130 = getitem_131 = getitem_136 = getitem_137 = None
        getitem_151 = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_152 = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_153 = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        permute_230 = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None
        permute_231 = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
        permute_232 = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
        view_437 = torch.ops.aten.view.default(permute_230, [1, 8192, 8, 4, 256]);  permute_230 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_437, [3], True);  view_437 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(sum_11, 3);  sum_11 = None
        view_438 = torch.ops.aten.view.default(permute_231, [1, 8192, 8, 4, 256]);  permute_231 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_438, [3], True);  view_438 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(sum_12, 3);  sum_12 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        convert_element_type_630 = torch.ops.prims.convert_element_type.default(permute_232, torch.float32);  permute_232 = None
        view_439 = torch.ops.aten.view.default(convert_element_type_629, [1, 8192, 8, 128, 2]);  convert_element_type_629 = None
        view_as_complex_34 = torch.ops.aten.view_as_complex.default(view_439);  view_439 = None
        mul_168 = torch.ops.aten.mul.Tensor(view_as_complex_34, _conj);  view_as_complex_34 = None
        view_440 = torch.ops.aten.view.default(convert_element_type_630, [1, 8192, 32, 128, 2]);  convert_element_type_630 = None
        view_as_complex_35 = torch.ops.aten.view_as_complex.default(view_440);  view_440 = None
        mul_169 = torch.ops.aten.mul.Tensor(view_as_complex_35, _conj);  view_as_complex_35 = None
        view_as_real_34 = torch.ops.aten.view_as_real.default(mul_168);  mul_168 = None
        view_441 = torch.ops.aten.view.default(view_as_real_34, [1, 8192, 8, 256]);  view_as_real_34 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(view_441, torch.bfloat16);  view_441 = None
        view_as_real_35 = torch.ops.aten.view_as_real.default(mul_169);  mul_169 = None
        view_442 = torch.ops.aten.view.default(view_as_real_35, [1, 8192, 32, 256]);  view_as_real_35 = None
        convert_element_type_632 = torch.ops.prims.convert_element_type.default(view_442, torch.bfloat16);  view_442 = None
        view_443 = torch.ops.aten.view.default(squeeze_2, [1, 8192, 2048]);  squeeze_2 = None
        view_444 = torch.ops.aten.view.default(convert_element_type_631, [1, 8192, 2048]);  convert_element_type_631 = None
        view_445 = torch.ops.aten.view.default(convert_element_type_632, [1, 8192, 8192]);  convert_element_type_632 = None
        view_446 = torch.ops.aten.view.default(view_443, [8192, 2048]);  view_443 = None
        permute_233 = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_233, view_350);  permute_233 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        all_gather_into_tensor_146 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_472, 8, '0');  convert_element_type_472 = None
        wait_tensor_160 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_146);  all_gather_into_tensor_146 = None
        permute_156 = torch.ops.aten.permute.default(wait_tensor_160, [1, 0]);  wait_tensor_160 = None
        permute_235 = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_138 = torch.ops.aten.mm.default(view_446, permute_235);  view_446 = permute_235 = None
        view_447 = torch.ops.aten.view.default(mm_138, [1, 8192, 8192]);  mm_138 = None
        convert_element_type_637 = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        reduce_scatter_tensor_35 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_637, 'sum', 8, '0');  convert_element_type_637 = None
        wait_tensor_203 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_35);  reduce_scatter_tensor_35 = None
        view_448 = torch.ops.aten.view.default(view_444, [8192, 2048]);  view_444 = None
        permute_237 = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_237, view_350);  permute_237 = None
        permute_239 = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_140 = torch.ops.aten.mm.default(view_448, permute_239);  view_448 = permute_239 = None
        view_449 = torch.ops.aten.view.default(mm_140, [1, 8192, 8192]);  mm_140 = None
        add_76 = torch.ops.aten.add.Tensor(view_447, view_449);  view_447 = view_449 = None
        convert_element_type_642 = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        reduce_scatter_tensor_36 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_642, 'sum', 8, '0');  convert_element_type_642 = None
        wait_tensor_204 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_36);  reduce_scatter_tensor_36 = None
        view_450 = torch.ops.aten.view.default(view_445, [8192, 8192]);  view_445 = None
        permute_241 = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_241, view_350);  permute_241 = view_350 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        all_gather_into_tensor_144 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_466, 8, '0');  convert_element_type_466 = None
        wait_tensor_158 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_144);  all_gather_into_tensor_144 = None
        permute_154 = torch.ops.aten.permute.default(wait_tensor_158, [1, 0]);  wait_tensor_158 = None
        permute_243 = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_142 = torch.ops.aten.mm.default(view_450, permute_243);  view_450 = permute_243 = None
        view_451 = torch.ops.aten.view.default(mm_142, [1, 8192, 8192]);  mm_142 = None
        add_77 = torch.ops.aten.add.Tensor(add_76, view_451);  add_76 = view_451 = None
        reduce_scatter_tensor_37 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_141, 'sum', 8, '0');  mm_141 = None
        wait_tensor_205 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_37);  reduce_scatter_tensor_37 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(wait_tensor_205, torch.float32);  wait_tensor_205 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(wait_tensor_157, torch.float32);  wait_tensor_157 = None
        mul_170 = torch.ops.aten.mul.Tensor(convert_element_type_648, convert_element_type_650);  convert_element_type_650 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_112, mul_170)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        div_4 = torch.ops.aten.div.Tensor(mul_112, 8192)
        mul_173 = torch.ops.aten.mul.Tensor(div_4, sum_13);  div_4 = sum_13 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_170, mul_173);  mul_170 = mul_173 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_28);  sub_6 = rsqrt_28 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_648, mul_112);  convert_element_type_648 = mul_112 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        convert_element_type_651 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        add_78 = torch.ops.aten.add.Tensor(add_75, convert_element_type_651);  add_75 = convert_element_type_651 = None
        convert_element_type_default_29 = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        reduce_scatter_tensor_38 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_29, 'sum', 8, '0');  convert_element_type_default_29 = None
        wait_tensor_206 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_38);  reduce_scatter_tensor_38 = None
        all_gather_into_tensor_168 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_78, 2, '3')
        wait_tensor_207 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_168);  all_gather_into_tensor_168 = None
        view_452 = torch.ops.aten.view.default(wait_tensor_207, [16384, 8192]);  wait_tensor_207 = None
        permute_245 = torch.ops.aten.permute.default(view_452, [1, 0])
        mm_143 = torch.ops.aten.mm.default(permute_245, view_348);  permute_245 = view_348 = None
        permute_246 = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
        permute_247 = torch.ops.aten.permute.default(wait_tensor_155, [1, 0]);  wait_tensor_155 = None
        mm_144 = torch.ops.aten.mm.default(view_452, permute_247);  view_452 = permute_247 = None
        view_453 = torch.ops.aten.view.default(mm_144, [2, 8192, 14336]);  mm_144 = None
        clone_72 = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        reduce_scatter_tensor_39 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_72, 'sum', 4, '1');  clone_72 = None
        wait_tensor_208 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_39);  reduce_scatter_tensor_39 = None
        permute_248 = torch.ops.aten.permute.default(wait_tensor_208, [1, 0]);  wait_tensor_208 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(permute_248, torch.float32);  permute_248 = None
        mul_176 = torch.ops.aten.mul.Tensor(view_453, convert_element_type_456);  convert_element_type_456 = None
        mul_177 = torch.ops.aten.mul.Tensor(view_453, view_347);  view_453 = view_347 = None
        view_454 = torch.ops.aten.view.default(mul_176, [16384, 14336]);  mul_176 = None
        permute_249 = torch.ops.aten.permute.default(view_454, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_249, view_344);  permute_249 = None
        reduce_scatter_tensor_40 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_145, 'sum', 4, '1');  mm_145 = None
        wait_tensor_209 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_40);  reduce_scatter_tensor_40 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        all_gather_into_tensor_141 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_457, 4, '1');  convert_element_type_457 = None
        wait_tensor_154 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_141);  all_gather_into_tensor_141 = None
        permute_152 = torch.ops.aten.permute.default(wait_tensor_154, [1, 0]);  wait_tensor_154 = None
        permute_251 = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_146 = torch.ops.aten.mm.default(view_454, permute_251);  view_454 = permute_251 = None
        view_455 = torch.ops.aten.view.default(mm_146, [2, 8192, 8192]);  mm_146 = None
        convert_element_type_663 = torch.ops.prims.convert_element_type.default(wait_tensor_209, torch.float32);  wait_tensor_209 = None
        convert_element_type_664 = torch.ops.prims.convert_element_type.default(mul_177, torch.float32);  mul_177 = None
        neg_2 = torch.ops.aten.neg.default(convert_element_type_455)
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_79 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(add_79);  add_79 = None
        mul_178 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_179 = torch.ops.aten.mul.Tensor(convert_element_type_664, mul_178);  convert_element_type_664 = None
        sub_7 = torch.ops.aten.sub.Tensor(1, mul_178);  mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(convert_element_type_455, sub_7);  convert_element_type_455 = sub_7 = None
        add_80 = torch.ops.aten.add.Tensor(mul_180, 1);  mul_180 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, add_80);  mul_179 = add_80 = None
        convert_element_type_666 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_456 = torch.ops.aten.view.default(convert_element_type_666, [16384, 14336]);  convert_element_type_666 = None
        permute_253 = torch.ops.aten.permute.default(view_456, [1, 0])
        mm_147 = torch.ops.aten.mm.default(permute_253, view_344);  permute_253 = view_344 = None
        reduce_scatter_tensor_41 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_147, 'sum', 4, '1');  mm_147 = None
        wait_tensor_210 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_41);  reduce_scatter_tensor_41 = None
        permute_255 = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_148 = torch.ops.aten.mm.default(view_456, permute_255);  view_456 = permute_255 = None
        view_457 = torch.ops.aten.view.default(mm_148, [2, 8192, 8192]);  mm_148 = None
        add_81 = torch.ops.aten.add.Tensor(view_455, view_457);  view_455 = view_457 = None
        convert_element_type_671 = torch.ops.prims.convert_element_type.default(wait_tensor_210, torch.float32);  wait_tensor_210 = None
        reduce_scatter_tensor_42 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_81, 'sum', 2, '3');  add_81 = None
        wait_tensor_211 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_42);  reduce_scatter_tensor_42 = None
        convert_element_type_672 = torch.ops.prims.convert_element_type.default(wait_tensor_211, torch.float32);  wait_tensor_211 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(wait_tensor_151, torch.float32);  wait_tensor_151 = None
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_672, convert_element_type_674);  convert_element_type_674 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_108, mul_182)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_184, [2], True);  mul_184 = None
        div_5 = torch.ops.aten.div.Tensor(mul_108, 8192)
        mul_185 = torch.ops.aten.mul.Tensor(div_5, sum_15);  div_5 = sum_15 = None
        sub_8 = torch.ops.aten.sub.Tensor(mul_182, mul_185);  mul_182 = mul_185 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_27);  sub_8 = rsqrt_27 = None
        mul_187 = torch.ops.aten.mul.Tensor(convert_element_type_672, mul_108);  convert_element_type_672 = mul_108 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1]);  mul_187 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None
        add_82 = torch.ops.aten.add.Tensor(add_78, convert_element_type_675);  add_78 = convert_element_type_675 = None
        convert_element_type_default_28 = torch.ops.prims.convert_element_type.default(sum_16, torch.float32);  sum_16 = None
        reduce_scatter_tensor_43 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_28, 'sum', 8, '0');  convert_element_type_default_28 = None
        wait_tensor_212 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_43);  reduce_scatter_tensor_43 = None
        view_458 = torch.ops.aten.view.default(add_82, [8192, 8192])
        permute_257 = torch.ops.aten.permute.default(view_458, [1, 0])
        permute_149 = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3])
        view_341 = torch.ops.aten.view.default(permute_149, [1, 8192, 8192]);  permute_149 = None
        view_342 = torch.ops.aten.view.default(view_341, [8192, 8192]);  view_341 = None
        mm_149 = torch.ops.aten.mm.default(permute_257, view_342);  permute_257 = view_342 = None
        permute_258 = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        permute_150 = torch.ops.aten.permute.default(convert_element_type_446, [1, 0]);  convert_element_type_446 = None
        clone_54 = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        all_gather_into_tensor_137 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_54, 8, '0');  clone_54 = None
        wait_tensor_150 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_137);  all_gather_into_tensor_137 = None
        permute_259 = torch.ops.aten.permute.default(wait_tensor_150, [1, 0]);  wait_tensor_150 = None
        mm_150 = torch.ops.aten.mm.default(view_458, permute_259);  view_458 = permute_259 = None
        view_459 = torch.ops.aten.view.default(mm_150, [1, 8192, 8192]);  mm_150 = None
        clone_73 = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        reduce_scatter_tensor_44 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_73, 'sum', 8, '0');  clone_73 = None
        wait_tensor_213 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_44);  reduce_scatter_tensor_44 = None
        permute_260 = torch.ops.aten.permute.default(wait_tensor_213, [1, 0]);  wait_tensor_213 = None
        convert_element_type_682 = torch.ops.prims.convert_element_type.default(permute_260, torch.float32);  permute_260 = None
        view_460 = torch.ops.aten.view.default(view_459, [1, 8192, 32, 256]);  view_459 = None
        permute_261 = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        view_324 = torch.ops.aten.view.default(mm_90, [2, 8192, 8192]);  mm_90 = None
        reduce_scatter_tensor_12 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_324, 'sum', 2, '3');  view_324 = None
        wait_tensor_145 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_12);  reduce_scatter_tensor_12 = None
        add_51 = torch.ops.aten.add.Tensor(add_49, wait_tensor_145);  wait_tensor_145 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_52 = torch.ops.aten.add.Scalar(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_431, rsqrt_26);  convert_element_type_431 = None
        all_gather_into_tensor_133 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_430, 8, '0');  convert_element_type_430 = None
        wait_tensor_146 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_133);  all_gather_into_tensor_133 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, wait_tensor_146)
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        convert_element_type_433 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        all_gather_into_tensor_134 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_433, 8, '0');  convert_element_type_433 = None
        wait_tensor_147 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_134);  all_gather_into_tensor_134 = None
        permute_143 = torch.ops.aten.permute.default(wait_tensor_147, [1, 0]);  wait_tensor_147 = None
        view_325 = torch.ops.aten.view.default(convert_element_type_432, [8192, 8192]);  convert_element_type_432 = None
        mm_91 = torch.ops.aten.mm.default(view_325, permute_143)
        view_326 = torch.ops.aten.view.default(mm_91, [1, 8192, 8192]);  mm_91 = None
        view_328 = torch.ops.aten.view.default(mm_92, [1, 8192, 2048]);  mm_92 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        all_gather_into_tensor_136 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_439, 8, '0');  convert_element_type_439 = None
        wait_tensor_149 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_136);  all_gather_into_tensor_136 = None
        permute_145 = torch.ops.aten.permute.default(wait_tensor_149, [1, 0]);  wait_tensor_149 = None
        mm_93 = torch.ops.aten.mm.default(view_325, permute_145)
        view_330 = torch.ops.aten.view.default(mm_93, [1, 8192, 2048]);  mm_93 = None
        view_331 = torch.ops.aten.view.default(view_326, [1, 8192, 32, 256]);  view_326 = None
        view_332 = torch.ops.aten.view.default(view_328, [1, 8192, 8, 256]);  view_328 = None
        view_333 = torch.ops.aten.view.default(view_330, [1, 8192, 8, 256]);  view_330 = None
        convert_element_type_442 = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        view_334 = torch.ops.aten.view.default(convert_element_type_442, [1, 8192, 32, 128, 2]);  convert_element_type_442 = None
        view_as_complex_26 = torch.ops.aten.view_as_complex.default(view_334);  view_334 = None
        convert_element_type_443 = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        view_335 = torch.ops.aten.view.default(convert_element_type_443, [1, 8192, 8, 128, 2]);  convert_element_type_443 = None
        view_as_complex_27 = torch.ops.aten.view_as_complex.default(view_335);  view_335 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_as_complex_26, view_11);  view_as_complex_26 = None
        view_as_real_26 = torch.ops.aten.view_as_real.default(mul_106);  mul_106 = None
        view_337 = torch.ops.aten.view.default(view_as_real_26, [1, 8192, 32, 256]);  view_as_real_26 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_as_complex_27, view_11);  view_as_complex_27 = None
        view_as_real_27 = torch.ops.aten.view_as_real.default(mul_107);  mul_107 = None
        view_338 = torch.ops.aten.view.default(view_as_real_27, [1, 8192, 8, 256]);  view_as_real_27 = None
        convert_element_type_444 = torch.ops.prims.convert_element_type.default(view_337, torch.bfloat16);  view_337 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(view_338, torch.bfloat16);  view_338 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(convert_element_type_445, 3);  convert_element_type_445 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_26, [1, 8192, 8, 4, 256]);  unsqueeze_26 = None
        clone_52 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_339 = torch.ops.aten.view.default(clone_52, [1, 8192, 32, 256]);  clone_52 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(view_333, 3);  view_333 = None
        expand_27 = torch.ops.aten.expand.default(unsqueeze_27, [1, 8192, 8, 4, 256]);  unsqueeze_27 = None
        clone_53 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_340 = torch.ops.aten.view.default(clone_53, [1, 8192, 32, 256]);  clone_53 = None
        permute_146 = torch.ops.aten.permute.default(convert_element_type_444, [0, 2, 1, 3]);  convert_element_type_444 = None
        permute_147 = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        permute_148 = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_261, permute_146, permute_147, permute_148, getitem_121, getitem_122, None, None, 8192, 8192, 0.0, True, getitem_127, getitem_128, scale = 0.0625);  permute_261 = permute_146 = permute_147 = permute_148 = getitem_121 = getitem_122 = getitem_127 = getitem_128 = None
        getitem_154 = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_155 = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_156 = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        permute_262 = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        permute_263 = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        permute_264 = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
        view_461 = torch.ops.aten.view.default(permute_262, [1, 8192, 8, 4, 256]);  permute_262 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_461, [3], True);  view_461 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(sum_17, 3);  sum_17 = None
        view_462 = torch.ops.aten.view.default(permute_263, [1, 8192, 8, 4, 256]);  permute_263 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_462, [3], True);  view_462 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(sum_18, 3);  sum_18 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        convert_element_type_684 = torch.ops.prims.convert_element_type.default(permute_264, torch.float32);  permute_264 = None
        view_463 = torch.ops.aten.view.default(convert_element_type_683, [1, 8192, 8, 128, 2]);  convert_element_type_683 = None
        view_as_complex_36 = torch.ops.aten.view_as_complex.default(view_463);  view_463 = None
        mul_188 = torch.ops.aten.mul.Tensor(view_as_complex_36, _conj);  view_as_complex_36 = None
        view_464 = torch.ops.aten.view.default(convert_element_type_684, [1, 8192, 32, 128, 2]);  convert_element_type_684 = None
        view_as_complex_37 = torch.ops.aten.view_as_complex.default(view_464);  view_464 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_as_complex_37, _conj);  view_as_complex_37 = None
        view_as_real_36 = torch.ops.aten.view_as_real.default(mul_188);  mul_188 = None
        view_465 = torch.ops.aten.view.default(view_as_real_36, [1, 8192, 8, 256]);  view_as_real_36 = None
        convert_element_type_685 = torch.ops.prims.convert_element_type.default(view_465, torch.bfloat16);  view_465 = None
        view_as_real_37 = torch.ops.aten.view_as_real.default(mul_189);  mul_189 = None
        view_466 = torch.ops.aten.view.default(view_as_real_37, [1, 8192, 32, 256]);  view_as_real_37 = None
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_466, torch.bfloat16);  view_466 = None
        view_467 = torch.ops.aten.view.default(squeeze_4, [1, 8192, 2048]);  squeeze_4 = None
        view_468 = torch.ops.aten.view.default(convert_element_type_685, [1, 8192, 2048]);  convert_element_type_685 = None
        view_469 = torch.ops.aten.view.default(convert_element_type_686, [1, 8192, 8192]);  convert_element_type_686 = None
        view_470 = torch.ops.aten.view.default(view_467, [8192, 2048]);  view_467 = None
        permute_265 = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_151 = torch.ops.aten.mm.default(permute_265, view_325);  permute_265 = None
        permute_267 = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_152 = torch.ops.aten.mm.default(view_470, permute_267);  view_470 = permute_267 = None
        view_471 = torch.ops.aten.view.default(mm_152, [1, 8192, 8192]);  mm_152 = None
        convert_element_type_691 = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        reduce_scatter_tensor_45 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_691, 'sum', 8, '0');  convert_element_type_691 = None
        wait_tensor_214 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_45);  reduce_scatter_tensor_45 = None
        view_472 = torch.ops.aten.view.default(view_468, [8192, 2048]);  view_468 = None
        permute_269 = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_153 = torch.ops.aten.mm.default(permute_269, view_325);  permute_269 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        all_gather_into_tensor_135 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_436, 8, '0');  convert_element_type_436 = None
        wait_tensor_148 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_135);  all_gather_into_tensor_135 = None
        permute_144 = torch.ops.aten.permute.default(wait_tensor_148, [1, 0]);  wait_tensor_148 = None
        permute_271 = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_154 = torch.ops.aten.mm.default(view_472, permute_271);  view_472 = permute_271 = None
        view_473 = torch.ops.aten.view.default(mm_154, [1, 8192, 8192]);  mm_154 = None
        add_83 = torch.ops.aten.add.Tensor(view_471, view_473);  view_471 = view_473 = None
        convert_element_type_696 = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        reduce_scatter_tensor_46 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_696, 'sum', 8, '0');  convert_element_type_696 = None
        wait_tensor_215 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_46);  reduce_scatter_tensor_46 = None
        view_474 = torch.ops.aten.view.default(view_469, [8192, 8192]);  view_469 = None
        permute_273 = torch.ops.aten.permute.default(view_474, [1, 0])
        mm_155 = torch.ops.aten.mm.default(permute_273, view_325);  permute_273 = view_325 = None
        permute_275 = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_156 = torch.ops.aten.mm.default(view_474, permute_275);  view_474 = permute_275 = None
        view_475 = torch.ops.aten.view.default(mm_156, [1, 8192, 8192]);  mm_156 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, view_475);  add_83 = view_475 = None
        reduce_scatter_tensor_47 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_155, 'sum', 8, '0');  mm_155 = None
        wait_tensor_216 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_47);  reduce_scatter_tensor_47 = None
        convert_element_type_701 = torch.ops.prims.convert_element_type.default(wait_tensor_216, torch.float32);  wait_tensor_216 = None
        convert_element_type_702 = torch.ops.prims.convert_element_type.default(add_84, torch.float32);  add_84 = None
        convert_element_type_704 = torch.ops.prims.convert_element_type.default(wait_tensor_146, torch.float32);  wait_tensor_146 = None
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_702, convert_element_type_704);  convert_element_type_704 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_104, mul_190)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
        div_6 = torch.ops.aten.div.Tensor(mul_104, 8192)
        mul_193 = torch.ops.aten.mul.Tensor(div_6, sum_19);  div_6 = sum_19 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_190, mul_193);  mul_190 = mul_193 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_26);  sub_9 = rsqrt_26 = None
        mul_195 = torch.ops.aten.mul.Tensor(convert_element_type_702, mul_104);  convert_element_type_702 = mul_104 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
        convert_element_type_705 = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        add_85 = torch.ops.aten.add.Tensor(add_82, convert_element_type_705);  add_82 = convert_element_type_705 = None
        convert_element_type_default_27 = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        reduce_scatter_tensor_48 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_27, 'sum', 8, '0');  convert_element_type_default_27 = None
        wait_tensor_217 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_48);  reduce_scatter_tensor_48 = None
        all_gather_into_tensor_169 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_85, 2, '3')
        wait_tensor_218 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_169);  all_gather_into_tensor_169 = None
        view_476 = torch.ops.aten.view.default(wait_tensor_218, [16384, 8192]);  wait_tensor_218 = None
        permute_277 = torch.ops.aten.permute.default(view_476, [1, 0])
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_417, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [2], True);  pow_26 = None
        add_50 = torch.ops.aten.add.Scalar(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_25);  convert_element_type_417 = None
        all_gather_into_tensor_128 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_416, 8, '0');  convert_element_type_416 = None
        wait_tensor_140 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_128);  all_gather_into_tensor_128 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, wait_tensor_140)
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        all_gather_into_tensor_130 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_418, 2, '3');  convert_element_type_418 = None
        wait_tensor_142 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_130);  all_gather_into_tensor_130 = None
        view_319 = torch.ops.aten.view.default(wait_tensor_142, [16384, 8192]);  wait_tensor_142 = None
        view_320 = torch.ops.aten.view.default(mm_88, [2, 8192, 14336]);  mm_88 = None
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_422)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_422, sigmoid_12);  sigmoid_12 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        all_gather_into_tensor_131 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_424, 4, '1');  convert_element_type_424 = None
        wait_tensor_143 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_131);  all_gather_into_tensor_131 = None
        permute_141 = torch.ops.aten.permute.default(wait_tensor_143, [1, 0]);  wait_tensor_143 = None
        mm_89 = torch.ops.aten.mm.default(view_319, permute_141)
        view_322 = torch.ops.aten.view.default(mm_89, [2, 8192, 14336]);  mm_89 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_423, view_322)
        view_323 = torch.ops.aten.view.default(mul_103, [16384, 14336]);  mul_103 = None
        mm_157 = torch.ops.aten.mm.default(permute_277, view_323);  permute_277 = view_323 = None
        permute_278 = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        permute_142 = torch.ops.aten.permute.default(convert_element_type_427, [1, 0]);  convert_element_type_427 = None
        clone_51 = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        all_gather_into_tensor_132 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_51, 4, '1');  clone_51 = None
        wait_tensor_144 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_132);  all_gather_into_tensor_132 = None
        permute_279 = torch.ops.aten.permute.default(wait_tensor_144, [1, 0]);  wait_tensor_144 = None
        mm_158 = torch.ops.aten.mm.default(view_476, permute_279);  view_476 = permute_279 = None
        view_477 = torch.ops.aten.view.default(mm_158, [2, 8192, 14336]);  mm_158 = None
        clone_76 = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
        reduce_scatter_tensor_49 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_76, 'sum', 4, '1');  clone_76 = None
        wait_tensor_219 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_49);  reduce_scatter_tensor_49 = None
        permute_280 = torch.ops.aten.permute.default(wait_tensor_219, [1, 0]);  wait_tensor_219 = None
        convert_element_type_712 = torch.ops.prims.convert_element_type.default(permute_280, torch.float32);  permute_280 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_477, convert_element_type_423);  convert_element_type_423 = None
        mul_197 = torch.ops.aten.mul.Tensor(view_477, view_322);  view_477 = view_322 = None
        view_478 = torch.ops.aten.view.default(mul_196, [16384, 14336]);  mul_196 = None
        permute_281 = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_159 = torch.ops.aten.mm.default(permute_281, view_319);  permute_281 = None
        reduce_scatter_tensor_50 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_159, 'sum', 4, '1');  mm_159 = None
        wait_tensor_220 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_50);  reduce_scatter_tensor_50 = None
        permute_283 = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_160 = torch.ops.aten.mm.default(view_478, permute_283);  view_478 = permute_283 = None
        view_479 = torch.ops.aten.view.default(mm_160, [2, 8192, 8192]);  mm_160 = None
        convert_element_type_717 = torch.ops.prims.convert_element_type.default(wait_tensor_220, torch.float32);  wait_tensor_220 = None
        convert_element_type_718 = torch.ops.prims.convert_element_type.default(mul_197, torch.float32);  mul_197 = None
        neg_3 = torch.ops.aten.neg.default(convert_element_type_422)
        exp_3 = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_86 = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_718, mul_198);  convert_element_type_718 = None
        sub_10 = torch.ops.aten.sub.Tensor(1, mul_198);  mul_198 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_422, sub_10);  convert_element_type_422 = sub_10 = None
        add_87 = torch.ops.aten.add.Tensor(mul_200, 1);  mul_200 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_199, add_87);  mul_199 = add_87 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        view_480 = torch.ops.aten.view.default(convert_element_type_720, [16384, 14336]);  convert_element_type_720 = None
        permute_285 = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_161 = torch.ops.aten.mm.default(permute_285, view_319);  permute_285 = view_319 = None
        reduce_scatter_tensor_51 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_161, 'sum', 4, '1');  mm_161 = None
        wait_tensor_221 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_51);  reduce_scatter_tensor_51 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        all_gather_into_tensor_129 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_419, 4, '1');  convert_element_type_419 = None
        wait_tensor_141 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_129);  all_gather_into_tensor_129 = None
        permute_140 = torch.ops.aten.permute.default(wait_tensor_141, [1, 0]);  wait_tensor_141 = None
        permute_287 = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_162 = torch.ops.aten.mm.default(view_480, permute_287);  view_480 = permute_287 = None
        view_481 = torch.ops.aten.view.default(mm_162, [2, 8192, 8192]);  mm_162 = None
        add_88 = torch.ops.aten.add.Tensor(view_479, view_481);  view_479 = view_481 = None
        convert_element_type_725 = torch.ops.prims.convert_element_type.default(wait_tensor_221, torch.float32);  wait_tensor_221 = None
        reduce_scatter_tensor_52 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_88, 'sum', 2, '3');  add_88 = None
        wait_tensor_222 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_52);  reduce_scatter_tensor_52 = None
        convert_element_type_726 = torch.ops.prims.convert_element_type.default(wait_tensor_222, torch.float32);  wait_tensor_222 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(wait_tensor_140, torch.float32);  wait_tensor_140 = None
        mul_202 = torch.ops.aten.mul.Tensor(convert_element_type_726, convert_element_type_728);  convert_element_type_728 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_100, mul_202)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        div_7 = torch.ops.aten.div.Tensor(mul_100, 8192)
        mul_205 = torch.ops.aten.mul.Tensor(div_7, sum_21);  div_7 = sum_21 = None
        sub_11 = torch.ops.aten.sub.Tensor(mul_202, mul_205);  mul_202 = mul_205 = None
        mul_206 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_25);  sub_11 = rsqrt_25 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_726, mul_100);  convert_element_type_726 = mul_100 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        add_89 = torch.ops.aten.add.Tensor(add_85, convert_element_type_729);  add_85 = convert_element_type_729 = None
        convert_element_type_default_26 = torch.ops.prims.convert_element_type.default(sum_22, torch.float32);  sum_22 = None
        reduce_scatter_tensor_53 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_26, 'sum', 8, '0');  convert_element_type_default_26 = None
        wait_tensor_223 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_53);  reduce_scatter_tensor_53 = None
        view_482 = torch.ops.aten.view.default(add_89, [8192, 8192])
        permute_289 = torch.ops.aten.permute.default(view_482, [1, 0])
        permute_138 = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3])
        view_316 = torch.ops.aten.view.default(permute_138, [1, 8192, 8192]);  permute_138 = None
        view_317 = torch.ops.aten.view.default(view_316, [8192, 8192]);  view_316 = None
        mm_163 = torch.ops.aten.mm.default(permute_289, view_317);  permute_289 = view_317 = None
        permute_290 = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        permute_139 = torch.ops.aten.permute.default(convert_element_type_413, [1, 0]);  convert_element_type_413 = None
        clone_50 = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        all_gather_into_tensor_127 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_50, 8, '0');  clone_50 = None
        wait_tensor_139 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_127);  all_gather_into_tensor_127 = None
        permute_291 = torch.ops.aten.permute.default(wait_tensor_139, [1, 0]);  wait_tensor_139 = None
        mm_164 = torch.ops.aten.mm.default(view_482, permute_291);  view_482 = permute_291 = None
        view_483 = torch.ops.aten.view.default(mm_164, [1, 8192, 8192]);  mm_164 = None
        clone_77 = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        reduce_scatter_tensor_54 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_77, 'sum', 8, '0');  clone_77 = None
        wait_tensor_224 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_54);  reduce_scatter_tensor_54 = None
        permute_292 = torch.ops.aten.permute.default(wait_tensor_224, [1, 0]);  wait_tensor_224 = None
        convert_element_type_736 = torch.ops.prims.convert_element_type.default(permute_292, torch.float32);  permute_292 = None
        view_484 = torch.ops.aten.view.default(view_483, [1, 8192, 32, 256]);  view_483 = None
        permute_293 = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [2], True);  pow_24 = None
        add_46 = torch.ops.aten.add.Scalar(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_23);  convert_element_type_384 = None
        all_gather_into_tensor_118 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_383, 8, '0');  convert_element_type_383 = None
        wait_tensor_129 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_118);  all_gather_into_tensor_118 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, wait_tensor_129)
        convert_element_type_385 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        all_gather_into_tensor_119 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_386, 4, '1');  convert_element_type_386 = None
        wait_tensor_130 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_119);  all_gather_into_tensor_119 = None
        permute_129 = torch.ops.aten.permute.default(wait_tensor_130, [1, 0]);  wait_tensor_130 = None
        all_gather_into_tensor_120 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_385, 2, '3');  convert_element_type_385 = None
        wait_tensor_131 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_120);  all_gather_into_tensor_120 = None
        view_294 = torch.ops.aten.view.default(wait_tensor_131, [16384, 8192]);  wait_tensor_131 = None
        mm_81 = torch.ops.aten.mm.default(view_294, permute_129)
        view_295 = torch.ops.aten.view.default(mm_81, [2, 8192, 14336]);  mm_81 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_389)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_389, sigmoid_11);  sigmoid_11 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        view_297 = torch.ops.aten.view.default(mm_82, [2, 8192, 14336]);  mm_82 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_390, view_297)
        convert_element_type_394 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_131 = torch.ops.aten.permute.default(convert_element_type_394, [1, 0]);  convert_element_type_394 = None
        view_298 = torch.ops.aten.view.default(mul_95, [16384, 14336]);  mul_95 = None
        clone_47 = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        all_gather_into_tensor_122 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_47, 4, '1');  clone_47 = None
        wait_tensor_133 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_122);  all_gather_into_tensor_122 = None
        mm_83 = torch.ops.aten.mm.default(view_298, wait_tensor_133)
        view_299 = torch.ops.aten.view.default(mm_83, [2, 8192, 8192]);  mm_83 = None
        reduce_scatter_tensor_11 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_299, 'sum', 2, '3');  view_299 = None
        wait_tensor_134 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_11);  reduce_scatter_tensor_11 = None
        add_47 = torch.ops.aten.add.Tensor(add_45, wait_tensor_134);  add_45 = wait_tensor_134 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_48 = torch.ops.aten.add.Scalar(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_24);  convert_element_type_398 = None
        all_gather_into_tensor_123 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_397, 8, '0');  convert_element_type_397 = None
        wait_tensor_135 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_123);  all_gather_into_tensor_123 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, wait_tensor_135)
        convert_element_type_399 = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None
        view_300 = torch.ops.aten.view.default(convert_element_type_399, [8192, 8192]);  convert_element_type_399 = None
        view_301 = torch.ops.aten.view.default(mm_84, [1, 8192, 8192]);  mm_84 = None
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        all_gather_into_tensor_125 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_403, 8, '0');  convert_element_type_403 = None
        wait_tensor_137 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_125);  all_gather_into_tensor_125 = None
        permute_133 = torch.ops.aten.permute.default(wait_tensor_137, [1, 0]);  wait_tensor_137 = None
        mm_85 = torch.ops.aten.mm.default(view_300, permute_133)
        view_303 = torch.ops.aten.view.default(mm_85, [1, 8192, 2048]);  mm_85 = None
        view_305 = torch.ops.aten.view.default(mm_86, [1, 8192, 2048]);  mm_86 = None
        view_306 = torch.ops.aten.view.default(view_301, [1, 8192, 32, 256]);  view_301 = None
        view_307 = torch.ops.aten.view.default(view_303, [1, 8192, 8, 256]);  view_303 = None
        view_308 = torch.ops.aten.view.default(view_305, [1, 8192, 8, 256]);  view_305 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        view_309 = torch.ops.aten.view.default(convert_element_type_409, [1, 8192, 32, 128, 2]);  convert_element_type_409 = None
        view_as_complex_24 = torch.ops.aten.view_as_complex.default(view_309);  view_309 = None
        convert_element_type_410 = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        view_310 = torch.ops.aten.view.default(convert_element_type_410, [1, 8192, 8, 128, 2]);  convert_element_type_410 = None
        view_as_complex_25 = torch.ops.aten.view_as_complex.default(view_310);  view_310 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_as_complex_24, view_11);  view_as_complex_24 = None
        view_as_real_24 = torch.ops.aten.view_as_real.default(mul_98);  mul_98 = None
        view_312 = torch.ops.aten.view.default(view_as_real_24, [1, 8192, 32, 256]);  view_as_real_24 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_as_complex_25, view_11);  view_as_complex_25 = None
        view_as_real_25 = torch.ops.aten.view_as_real.default(mul_99);  mul_99 = None
        view_313 = torch.ops.aten.view.default(view_as_real_25, [1, 8192, 8, 256]);  view_as_real_25 = None
        convert_element_type_411 = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(view_313, torch.bfloat16);  view_313 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(convert_element_type_412, 3);  convert_element_type_412 = None
        expand_24 = torch.ops.aten.expand.default(unsqueeze_24, [1, 8192, 8, 4, 256]);  unsqueeze_24 = None
        clone_48 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_314 = torch.ops.aten.view.default(clone_48, [1, 8192, 32, 256]);  clone_48 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(view_308, 3);  view_308 = None
        expand_25 = torch.ops.aten.expand.default(unsqueeze_25, [1, 8192, 8, 4, 256]);  unsqueeze_25 = None
        clone_49 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_315 = torch.ops.aten.view.default(clone_49, [1, 8192, 32, 256]);  clone_49 = None
        permute_135 = torch.ops.aten.permute.default(convert_element_type_411, [0, 2, 1, 3]);  convert_element_type_411 = None
        permute_136 = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        permute_137 = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_293, permute_135, permute_136, permute_137, getitem_112, getitem_113, None, None, 8192, 8192, 0.0, True, getitem_118, getitem_119, scale = 0.0625);  permute_293 = permute_135 = permute_136 = permute_137 = getitem_112 = getitem_113 = getitem_118 = getitem_119 = None
        getitem_157 = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_158 = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_159 = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        permute_294 = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
        permute_295 = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
        permute_296 = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None
        view_485 = torch.ops.aten.view.default(permute_294, [1, 8192, 8, 4, 256]);  permute_294 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_485, [3], True);  view_485 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(sum_23, 3);  sum_23 = None
        view_486 = torch.ops.aten.view.default(permute_295, [1, 8192, 8, 4, 256]);  permute_295 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_486, [3], True);  view_486 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(sum_24, 3);  sum_24 = None
        convert_element_type_737 = torch.ops.prims.convert_element_type.default(squeeze_7, torch.float32);  squeeze_7 = None
        convert_element_type_738 = torch.ops.prims.convert_element_type.default(permute_296, torch.float32);  permute_296 = None
        view_487 = torch.ops.aten.view.default(convert_element_type_737, [1, 8192, 8, 128, 2]);  convert_element_type_737 = None
        view_as_complex_38 = torch.ops.aten.view_as_complex.default(view_487);  view_487 = None
        mul_208 = torch.ops.aten.mul.Tensor(view_as_complex_38, _conj);  view_as_complex_38 = None
        view_488 = torch.ops.aten.view.default(convert_element_type_738, [1, 8192, 32, 128, 2]);  convert_element_type_738 = None
        view_as_complex_39 = torch.ops.aten.view_as_complex.default(view_488);  view_488 = None
        mul_209 = torch.ops.aten.mul.Tensor(view_as_complex_39, _conj);  view_as_complex_39 = None
        view_as_real_38 = torch.ops.aten.view_as_real.default(mul_208);  mul_208 = None
        view_489 = torch.ops.aten.view.default(view_as_real_38, [1, 8192, 8, 256]);  view_as_real_38 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(view_489, torch.bfloat16);  view_489 = None
        view_as_real_39 = torch.ops.aten.view_as_real.default(mul_209);  mul_209 = None
        view_490 = torch.ops.aten.view.default(view_as_real_39, [1, 8192, 32, 256]);  view_as_real_39 = None
        convert_element_type_740 = torch.ops.prims.convert_element_type.default(view_490, torch.bfloat16);  view_490 = None
        view_491 = torch.ops.aten.view.default(squeeze_6, [1, 8192, 2048]);  squeeze_6 = None
        view_492 = torch.ops.aten.view.default(convert_element_type_739, [1, 8192, 2048]);  convert_element_type_739 = None
        view_493 = torch.ops.aten.view.default(convert_element_type_740, [1, 8192, 8192]);  convert_element_type_740 = None
        view_494 = torch.ops.aten.view.default(view_491, [8192, 2048]);  view_491 = None
        permute_297 = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_165 = torch.ops.aten.mm.default(permute_297, view_300);  permute_297 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        all_gather_into_tensor_126 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_406, 8, '0');  convert_element_type_406 = None
        wait_tensor_138 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_126);  all_gather_into_tensor_126 = None
        permute_134 = torch.ops.aten.permute.default(wait_tensor_138, [1, 0]);  wait_tensor_138 = None
        permute_299 = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_166 = torch.ops.aten.mm.default(view_494, permute_299);  view_494 = permute_299 = None
        view_495 = torch.ops.aten.view.default(mm_166, [1, 8192, 8192]);  mm_166 = None
        convert_element_type_745 = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        reduce_scatter_tensor_55 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_745, 'sum', 8, '0');  convert_element_type_745 = None
        wait_tensor_225 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_55);  reduce_scatter_tensor_55 = None
        view_496 = torch.ops.aten.view.default(view_492, [8192, 2048]);  view_492 = None
        permute_301 = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_167 = torch.ops.aten.mm.default(permute_301, view_300);  permute_301 = None
        permute_303 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_168 = torch.ops.aten.mm.default(view_496, permute_303);  view_496 = permute_303 = None
        view_497 = torch.ops.aten.view.default(mm_168, [1, 8192, 8192]);  mm_168 = None
        add_90 = torch.ops.aten.add.Tensor(view_495, view_497);  view_495 = view_497 = None
        convert_element_type_750 = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        reduce_scatter_tensor_56 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_750, 'sum', 8, '0');  convert_element_type_750 = None
        wait_tensor_226 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_56);  reduce_scatter_tensor_56 = None
        view_498 = torch.ops.aten.view.default(view_493, [8192, 8192]);  view_493 = None
        permute_305 = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_169 = torch.ops.aten.mm.default(permute_305, view_300);  permute_305 = view_300 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        all_gather_into_tensor_124 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_400, 8, '0');  convert_element_type_400 = None
        wait_tensor_136 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_124);  all_gather_into_tensor_124 = None
        permute_132 = torch.ops.aten.permute.default(wait_tensor_136, [1, 0]);  wait_tensor_136 = None
        permute_307 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_170 = torch.ops.aten.mm.default(view_498, permute_307);  view_498 = permute_307 = None
        view_499 = torch.ops.aten.view.default(mm_170, [1, 8192, 8192]);  mm_170 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, view_499);  add_90 = view_499 = None
        reduce_scatter_tensor_57 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_169, 'sum', 8, '0');  mm_169 = None
        wait_tensor_227 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_57);  reduce_scatter_tensor_57 = None
        convert_element_type_755 = torch.ops.prims.convert_element_type.default(wait_tensor_227, torch.float32);  wait_tensor_227 = None
        convert_element_type_756 = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        convert_element_type_758 = torch.ops.prims.convert_element_type.default(wait_tensor_135, torch.float32);  wait_tensor_135 = None
        mul_210 = torch.ops.aten.mul.Tensor(convert_element_type_756, convert_element_type_758);  convert_element_type_758 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_96, mul_210)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
        div_8 = torch.ops.aten.div.Tensor(mul_96, 8192)
        mul_213 = torch.ops.aten.mul.Tensor(div_8, sum_25);  div_8 = sum_25 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_210, mul_213);  mul_210 = mul_213 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_24);  sub_12 = rsqrt_24 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_756, mul_96);  convert_element_type_756 = mul_96 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
        convert_element_type_759 = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        add_92 = torch.ops.aten.add.Tensor(add_89, convert_element_type_759);  add_89 = convert_element_type_759 = None
        convert_element_type_default_25 = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        reduce_scatter_tensor_58 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_25, 'sum', 8, '0');  convert_element_type_default_25 = None
        wait_tensor_228 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_58);  reduce_scatter_tensor_58 = None
        all_gather_into_tensor_170 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_92, 2, '3')
        wait_tensor_229 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_170);  all_gather_into_tensor_170 = None
        view_500 = torch.ops.aten.view.default(wait_tensor_229, [16384, 8192]);  wait_tensor_229 = None
        permute_309 = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_171 = torch.ops.aten.mm.default(permute_309, view_298);  permute_309 = view_298 = None
        permute_310 = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
        permute_311 = torch.ops.aten.permute.default(wait_tensor_133, [1, 0]);  wait_tensor_133 = None
        mm_172 = torch.ops.aten.mm.default(view_500, permute_311);  view_500 = permute_311 = None
        view_501 = torch.ops.aten.view.default(mm_172, [2, 8192, 14336]);  mm_172 = None
        clone_80 = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        reduce_scatter_tensor_59 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_80, 'sum', 4, '1');  clone_80 = None
        wait_tensor_230 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_59);  reduce_scatter_tensor_59 = None
        permute_312 = torch.ops.aten.permute.default(wait_tensor_230, [1, 0]);  wait_tensor_230 = None
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(permute_312, torch.float32);  permute_312 = None
        mul_216 = torch.ops.aten.mul.Tensor(view_501, convert_element_type_390);  convert_element_type_390 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_501, view_297);  view_501 = view_297 = None
        view_502 = torch.ops.aten.view.default(mul_216, [16384, 14336]);  mul_216 = None
        permute_313 = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_173 = torch.ops.aten.mm.default(permute_313, view_294);  permute_313 = None
        reduce_scatter_tensor_60 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_173, 'sum', 4, '1');  mm_173 = None
        wait_tensor_231 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_60);  reduce_scatter_tensor_60 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        all_gather_into_tensor_121 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_391, 4, '1');  convert_element_type_391 = None
        wait_tensor_132 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_121);  all_gather_into_tensor_121 = None
        permute_130 = torch.ops.aten.permute.default(wait_tensor_132, [1, 0]);  wait_tensor_132 = None
        permute_315 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_174 = torch.ops.aten.mm.default(view_502, permute_315);  view_502 = permute_315 = None
        view_503 = torch.ops.aten.view.default(mm_174, [2, 8192, 8192]);  mm_174 = None
        convert_element_type_771 = torch.ops.prims.convert_element_type.default(wait_tensor_231, torch.float32);  wait_tensor_231 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(mul_217, torch.float32);  mul_217 = None
        neg_4 = torch.ops.aten.neg.default(convert_element_type_389)
        exp_4 = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_93 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(add_93);  add_93 = None
        mul_218 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_219 = torch.ops.aten.mul.Tensor(convert_element_type_772, mul_218);  convert_element_type_772 = None
        sub_13 = torch.ops.aten.sub.Tensor(1, mul_218);  mul_218 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_389, sub_13);  convert_element_type_389 = sub_13 = None
        add_94 = torch.ops.aten.add.Tensor(mul_220, 1);  mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_219, add_94);  mul_219 = add_94 = None
        convert_element_type_774 = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        view_504 = torch.ops.aten.view.default(convert_element_type_774, [16384, 14336]);  convert_element_type_774 = None
        permute_317 = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_175 = torch.ops.aten.mm.default(permute_317, view_294);  permute_317 = view_294 = None
        reduce_scatter_tensor_61 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_175, 'sum', 4, '1');  mm_175 = None
        wait_tensor_232 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_61);  reduce_scatter_tensor_61 = None
        permute_319 = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_176 = torch.ops.aten.mm.default(view_504, permute_319);  view_504 = permute_319 = None
        view_505 = torch.ops.aten.view.default(mm_176, [2, 8192, 8192]);  mm_176 = None
        add_95 = torch.ops.aten.add.Tensor(view_503, view_505);  view_503 = view_505 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(wait_tensor_232, torch.float32);  wait_tensor_232 = None
        reduce_scatter_tensor_62 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_95, 'sum', 2, '3');  add_95 = None
        wait_tensor_233 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_62);  reduce_scatter_tensor_62 = None
        convert_element_type_780 = torch.ops.prims.convert_element_type.default(wait_tensor_233, torch.float32);  wait_tensor_233 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(wait_tensor_129, torch.float32);  wait_tensor_129 = None
        mul_222 = torch.ops.aten.mul.Tensor(convert_element_type_780, convert_element_type_782);  convert_element_type_782 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_92, mul_222)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        div_9 = torch.ops.aten.div.Tensor(mul_92, 8192)
        mul_225 = torch.ops.aten.mul.Tensor(div_9, sum_27);  div_9 = sum_27 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_222, mul_225);  mul_222 = mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_23);  sub_14 = rsqrt_23 = None
        mul_227 = torch.ops.aten.mul.Tensor(convert_element_type_780, mul_92);  convert_element_type_780 = mul_92 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        convert_element_type_783 = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        add_96 = torch.ops.aten.add.Tensor(add_92, convert_element_type_783);  add_92 = convert_element_type_783 = None
        convert_element_type_default_24 = torch.ops.prims.convert_element_type.default(sum_28, torch.float32);  sum_28 = None
        reduce_scatter_tensor_63 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_24, 'sum', 8, '0');  convert_element_type_default_24 = None
        wait_tensor_234 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_63);  reduce_scatter_tensor_63 = None
        view_506 = torch.ops.aten.view.default(add_96, [8192, 8192])
        permute_321 = torch.ops.aten.permute.default(view_506, [1, 0])
        permute_127 = torch.ops.aten.permute.default(getitem_103, [0, 2, 1, 3])
        view_291 = torch.ops.aten.view.default(permute_127, [1, 8192, 8192]);  permute_127 = None
        view_292 = torch.ops.aten.view.default(view_291, [8192, 8192]);  view_291 = None
        mm_177 = torch.ops.aten.mm.default(permute_321, view_292);  permute_321 = view_292 = None
        permute_322 = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        permute_128 = torch.ops.aten.permute.default(convert_element_type_380, [1, 0]);  convert_element_type_380 = None
        clone_46 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        all_gather_into_tensor_117 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_46, 8, '0');  clone_46 = None
        wait_tensor_128 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_117);  all_gather_into_tensor_117 = None
        permute_323 = torch.ops.aten.permute.default(wait_tensor_128, [1, 0]);  wait_tensor_128 = None
        mm_178 = torch.ops.aten.mm.default(view_506, permute_323);  view_506 = permute_323 = None
        view_507 = torch.ops.aten.view.default(mm_178, [1, 8192, 8192]);  mm_178 = None
        clone_81 = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        reduce_scatter_tensor_64 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_81, 'sum', 8, '0');  clone_81 = None
        wait_tensor_235 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_64);  reduce_scatter_tensor_64 = None
        permute_324 = torch.ops.aten.permute.default(wait_tensor_235, [1, 0]);  wait_tensor_235 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(permute_324, torch.float32);  permute_324 = None
        view_508 = torch.ops.aten.view.default(view_507, [1, 8192, 32, 256]);  view_507 = None
        permute_325 = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        view_274 = torch.ops.aten.view.default(mm_76, [2, 8192, 8192]);  mm_76 = None
        reduce_scatter_tensor_10 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_274, 'sum', 2, '3');  view_274 = None
        wait_tensor_123 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_10);  reduce_scatter_tensor_10 = None
        add_43 = torch.ops.aten.add.Tensor(add_41, wait_tensor_123);  wait_tensor_123 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_365, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [2], True);  pow_23 = None
        add_44 = torch.ops.aten.add.Scalar(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_365, rsqrt_22);  convert_element_type_365 = None
        all_gather_into_tensor_113 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_364, 8, '0');  convert_element_type_364 = None
        wait_tensor_124 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_113);  all_gather_into_tensor_113 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, wait_tensor_124)
        convert_element_type_366 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        all_gather_into_tensor_114 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_367, 8, '0');  convert_element_type_367 = None
        wait_tensor_125 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_114);  all_gather_into_tensor_114 = None
        permute_121 = torch.ops.aten.permute.default(wait_tensor_125, [1, 0]);  wait_tensor_125 = None
        view_275 = torch.ops.aten.view.default(convert_element_type_366, [8192, 8192]);  convert_element_type_366 = None
        mm_77 = torch.ops.aten.mm.default(view_275, permute_121)
        view_276 = torch.ops.aten.view.default(mm_77, [1, 8192, 8192]);  mm_77 = None
        view_278 = torch.ops.aten.view.default(mm_78, [1, 8192, 2048]);  mm_78 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        all_gather_into_tensor_116 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_373, 8, '0');  convert_element_type_373 = None
        wait_tensor_127 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_116);  all_gather_into_tensor_116 = None
        permute_123 = torch.ops.aten.permute.default(wait_tensor_127, [1, 0]);  wait_tensor_127 = None
        mm_79 = torch.ops.aten.mm.default(view_275, permute_123)
        view_280 = torch.ops.aten.view.default(mm_79, [1, 8192, 2048]);  mm_79 = None
        view_281 = torch.ops.aten.view.default(view_276, [1, 8192, 32, 256]);  view_276 = None
        view_282 = torch.ops.aten.view.default(view_278, [1, 8192, 8, 256]);  view_278 = None
        view_283 = torch.ops.aten.view.default(view_280, [1, 8192, 8, 256]);  view_280 = None
        convert_element_type_376 = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None
        view_284 = torch.ops.aten.view.default(convert_element_type_376, [1, 8192, 32, 128, 2]);  convert_element_type_376 = None
        view_as_complex_22 = torch.ops.aten.view_as_complex.default(view_284);  view_284 = None
        convert_element_type_377 = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        view_285 = torch.ops.aten.view.default(convert_element_type_377, [1, 8192, 8, 128, 2]);  convert_element_type_377 = None
        view_as_complex_23 = torch.ops.aten.view_as_complex.default(view_285);  view_285 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_as_complex_22, view_11);  view_as_complex_22 = None
        view_as_real_22 = torch.ops.aten.view_as_real.default(mul_90);  mul_90 = None
        view_287 = torch.ops.aten.view.default(view_as_real_22, [1, 8192, 32, 256]);  view_as_real_22 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_as_complex_23, view_11);  view_as_complex_23 = None
        view_as_real_23 = torch.ops.aten.view_as_real.default(mul_91);  mul_91 = None
        view_288 = torch.ops.aten.view.default(view_as_real_23, [1, 8192, 8, 256]);  view_as_real_23 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(view_287, torch.bfloat16);  view_287 = None
        convert_element_type_379 = torch.ops.prims.convert_element_type.default(view_288, torch.bfloat16);  view_288 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(convert_element_type_379, 3);  convert_element_type_379 = None
        expand_22 = torch.ops.aten.expand.default(unsqueeze_22, [1, 8192, 8, 4, 256]);  unsqueeze_22 = None
        clone_44 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_289 = torch.ops.aten.view.default(clone_44, [1, 8192, 32, 256]);  clone_44 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(view_283, 3);  view_283 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_23, [1, 8192, 8, 4, 256]);  unsqueeze_23 = None
        clone_45 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_290 = torch.ops.aten.view.default(clone_45, [1, 8192, 32, 256]);  clone_45 = None
        permute_124 = torch.ops.aten.permute.default(convert_element_type_378, [0, 2, 1, 3]);  convert_element_type_378 = None
        permute_125 = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        permute_126 = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_325, permute_124, permute_125, permute_126, getitem_103, getitem_104, None, None, 8192, 8192, 0.0, True, getitem_109, getitem_110, scale = 0.0625);  permute_325 = permute_124 = permute_125 = permute_126 = getitem_103 = getitem_104 = getitem_109 = getitem_110 = None
        getitem_160 = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_161 = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_162 = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        permute_326 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        permute_327 = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
        permute_328 = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
        view_509 = torch.ops.aten.view.default(permute_326, [1, 8192, 8, 4, 256]);  permute_326 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(view_509, [3], True);  view_509 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(sum_29, 3);  sum_29 = None
        view_510 = torch.ops.aten.view.default(permute_327, [1, 8192, 8, 4, 256]);  permute_327 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(view_510, [3], True);  view_510 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(sum_30, 3);  sum_30 = None
        convert_element_type_791 = torch.ops.prims.convert_element_type.default(squeeze_9, torch.float32);  squeeze_9 = None
        convert_element_type_792 = torch.ops.prims.convert_element_type.default(permute_328, torch.float32);  permute_328 = None
        view_511 = torch.ops.aten.view.default(convert_element_type_791, [1, 8192, 8, 128, 2]);  convert_element_type_791 = None
        view_as_complex_40 = torch.ops.aten.view_as_complex.default(view_511);  view_511 = None
        mul_228 = torch.ops.aten.mul.Tensor(view_as_complex_40, _conj);  view_as_complex_40 = None
        view_512 = torch.ops.aten.view.default(convert_element_type_792, [1, 8192, 32, 128, 2]);  convert_element_type_792 = None
        view_as_complex_41 = torch.ops.aten.view_as_complex.default(view_512);  view_512 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_as_complex_41, _conj);  view_as_complex_41 = None
        view_as_real_40 = torch.ops.aten.view_as_real.default(mul_228);  mul_228 = None
        view_513 = torch.ops.aten.view.default(view_as_real_40, [1, 8192, 8, 256]);  view_as_real_40 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(view_513, torch.bfloat16);  view_513 = None
        view_as_real_41 = torch.ops.aten.view_as_real.default(mul_229);  mul_229 = None
        view_514 = torch.ops.aten.view.default(view_as_real_41, [1, 8192, 32, 256]);  view_as_real_41 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(view_514, torch.bfloat16);  view_514 = None
        view_515 = torch.ops.aten.view.default(squeeze_8, [1, 8192, 2048]);  squeeze_8 = None
        view_516 = torch.ops.aten.view.default(convert_element_type_793, [1, 8192, 2048]);  convert_element_type_793 = None
        view_517 = torch.ops.aten.view.default(convert_element_type_794, [1, 8192, 8192]);  convert_element_type_794 = None
        view_518 = torch.ops.aten.view.default(view_515, [8192, 2048]);  view_515 = None
        permute_329 = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_179 = torch.ops.aten.mm.default(permute_329, view_275);  permute_329 = None
        permute_331 = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_180 = torch.ops.aten.mm.default(view_518, permute_331);  view_518 = permute_331 = None
        view_519 = torch.ops.aten.view.default(mm_180, [1, 8192, 8192]);  mm_180 = None
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        reduce_scatter_tensor_65 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_799, 'sum', 8, '0');  convert_element_type_799 = None
        wait_tensor_236 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_65);  reduce_scatter_tensor_65 = None
        view_520 = torch.ops.aten.view.default(view_516, [8192, 2048]);  view_516 = None
        permute_333 = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_181 = torch.ops.aten.mm.default(permute_333, view_275);  permute_333 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        all_gather_into_tensor_115 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_370, 8, '0');  convert_element_type_370 = None
        wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_115);  all_gather_into_tensor_115 = None
        permute_122 = torch.ops.aten.permute.default(wait_tensor_126, [1, 0]);  wait_tensor_126 = None
        permute_335 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_182 = torch.ops.aten.mm.default(view_520, permute_335);  view_520 = permute_335 = None
        view_521 = torch.ops.aten.view.default(mm_182, [1, 8192, 8192]);  mm_182 = None
        add_97 = torch.ops.aten.add.Tensor(view_519, view_521);  view_519 = view_521 = None
        convert_element_type_804 = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        reduce_scatter_tensor_66 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_804, 'sum', 8, '0');  convert_element_type_804 = None
        wait_tensor_237 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_66);  reduce_scatter_tensor_66 = None
        view_522 = torch.ops.aten.view.default(view_517, [8192, 8192]);  view_517 = None
        permute_337 = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_183 = torch.ops.aten.mm.default(permute_337, view_275);  permute_337 = view_275 = None
        permute_339 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_184 = torch.ops.aten.mm.default(view_522, permute_339);  view_522 = permute_339 = None
        view_523 = torch.ops.aten.view.default(mm_184, [1, 8192, 8192]);  mm_184 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, view_523);  add_97 = view_523 = None
        reduce_scatter_tensor_67 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_183, 'sum', 8, '0');  mm_183 = None
        wait_tensor_238 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_67);  reduce_scatter_tensor_67 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(wait_tensor_238, torch.float32);  wait_tensor_238 = None
        convert_element_type_810 = torch.ops.prims.convert_element_type.default(add_98, torch.float32);  add_98 = None
        convert_element_type_812 = torch.ops.prims.convert_element_type.default(wait_tensor_124, torch.float32);  wait_tensor_124 = None
        mul_230 = torch.ops.aten.mul.Tensor(convert_element_type_810, convert_element_type_812);  convert_element_type_812 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_88, mul_230)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_232, [2], True);  mul_232 = None
        div_10 = torch.ops.aten.div.Tensor(mul_88, 8192)
        mul_233 = torch.ops.aten.mul.Tensor(div_10, sum_31);  div_10 = sum_31 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_230, mul_233);  mul_230 = mul_233 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_22);  sub_15 = rsqrt_22 = None
        mul_235 = torch.ops.aten.mul.Tensor(convert_element_type_810, mul_88);  convert_element_type_810 = mul_88 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1]);  mul_235 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        add_99 = torch.ops.aten.add.Tensor(add_96, convert_element_type_813);  add_96 = convert_element_type_813 = None
        convert_element_type_default_23 = torch.ops.prims.convert_element_type.default(sum_32, torch.float32);  sum_32 = None
        reduce_scatter_tensor_68 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_23, 'sum', 8, '0');  convert_element_type_default_23 = None
        wait_tensor_239 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_68);  reduce_scatter_tensor_68 = None
        all_gather_into_tensor_171 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_99, 2, '3')
        wait_tensor_240 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_171);  all_gather_into_tensor_171 = None
        view_524 = torch.ops.aten.view.default(wait_tensor_240, [16384, 8192]);  wait_tensor_240 = None
        permute_341 = torch.ops.aten.permute.default(view_524, [1, 0])
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_42 = torch.ops.aten.add.Scalar(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_21);  convert_element_type_351 = None
        all_gather_into_tensor_108 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_350, 8, '0');  convert_element_type_350 = None
        wait_tensor_118 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_108);  all_gather_into_tensor_108 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, wait_tensor_118)
        convert_element_type_352 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        all_gather_into_tensor_110 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_352, 2, '3');  convert_element_type_352 = None
        wait_tensor_120 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_110);  all_gather_into_tensor_110 = None
        view_269 = torch.ops.aten.view.default(wait_tensor_120, [16384, 8192]);  wait_tensor_120 = None
        view_270 = torch.ops.aten.view.default(mm_74, [2, 8192, 14336]);  mm_74 = None
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_356)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_356, sigmoid_10);  sigmoid_10 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        all_gather_into_tensor_111 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_358, 4, '1');  convert_element_type_358 = None
        wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_111);  all_gather_into_tensor_111 = None
        permute_119 = torch.ops.aten.permute.default(wait_tensor_121, [1, 0]);  wait_tensor_121 = None
        mm_75 = torch.ops.aten.mm.default(view_269, permute_119)
        view_272 = torch.ops.aten.view.default(mm_75, [2, 8192, 14336]);  mm_75 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_357, view_272)
        view_273 = torch.ops.aten.view.default(mul_87, [16384, 14336]);  mul_87 = None
        mm_185 = torch.ops.aten.mm.default(permute_341, view_273);  permute_341 = view_273 = None
        permute_342 = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        permute_120 = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        clone_43 = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        all_gather_into_tensor_112 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_43, 4, '1');  clone_43 = None
        wait_tensor_122 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_112);  all_gather_into_tensor_112 = None
        permute_343 = torch.ops.aten.permute.default(wait_tensor_122, [1, 0]);  wait_tensor_122 = None
        mm_186 = torch.ops.aten.mm.default(view_524, permute_343);  view_524 = permute_343 = None
        view_525 = torch.ops.aten.view.default(mm_186, [2, 8192, 14336]);  mm_186 = None
        clone_84 = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
        reduce_scatter_tensor_69 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_84, 'sum', 4, '1');  clone_84 = None
        wait_tensor_241 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_69);  reduce_scatter_tensor_69 = None
        permute_344 = torch.ops.aten.permute.default(wait_tensor_241, [1, 0]);  wait_tensor_241 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(permute_344, torch.float32);  permute_344 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_525, convert_element_type_357);  convert_element_type_357 = None
        mul_237 = torch.ops.aten.mul.Tensor(view_525, view_272);  view_525 = view_272 = None
        view_526 = torch.ops.aten.view.default(mul_236, [16384, 14336]);  mul_236 = None
        permute_345 = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_187 = torch.ops.aten.mm.default(permute_345, view_269);  permute_345 = None
        reduce_scatter_tensor_70 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_187, 'sum', 4, '1');  mm_187 = None
        wait_tensor_242 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_70);  reduce_scatter_tensor_70 = None
        permute_347 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_188 = torch.ops.aten.mm.default(view_526, permute_347);  view_526 = permute_347 = None
        view_527 = torch.ops.aten.view.default(mm_188, [2, 8192, 8192]);  mm_188 = None
        convert_element_type_825 = torch.ops.prims.convert_element_type.default(wait_tensor_242, torch.float32);  wait_tensor_242 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(mul_237, torch.float32);  mul_237 = None
        neg_5 = torch.ops.aten.neg.default(convert_element_type_356)
        exp_5 = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_100 = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(add_100);  add_100 = None
        mul_238 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_826, mul_238);  convert_element_type_826 = None
        sub_16 = torch.ops.aten.sub.Tensor(1, mul_238);  mul_238 = None
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_356, sub_16);  convert_element_type_356 = sub_16 = None
        add_101 = torch.ops.aten.add.Tensor(mul_240, 1);  mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_239, add_101);  mul_239 = add_101 = None
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        view_528 = torch.ops.aten.view.default(convert_element_type_828, [16384, 14336]);  convert_element_type_828 = None
        permute_349 = torch.ops.aten.permute.default(view_528, [1, 0])
        mm_189 = torch.ops.aten.mm.default(permute_349, view_269);  permute_349 = view_269 = None
        reduce_scatter_tensor_71 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_189, 'sum', 4, '1');  mm_189 = None
        wait_tensor_243 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_71);  reduce_scatter_tensor_71 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        all_gather_into_tensor_109 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_353, 4, '1');  convert_element_type_353 = None
        wait_tensor_119 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_109);  all_gather_into_tensor_109 = None
        permute_118 = torch.ops.aten.permute.default(wait_tensor_119, [1, 0]);  wait_tensor_119 = None
        permute_351 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_190 = torch.ops.aten.mm.default(view_528, permute_351);  view_528 = permute_351 = None
        view_529 = torch.ops.aten.view.default(mm_190, [2, 8192, 8192]);  mm_190 = None
        add_102 = torch.ops.aten.add.Tensor(view_527, view_529);  view_527 = view_529 = None
        convert_element_type_833 = torch.ops.prims.convert_element_type.default(wait_tensor_243, torch.float32);  wait_tensor_243 = None
        reduce_scatter_tensor_72 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_102, 'sum', 2, '3');  add_102 = None
        wait_tensor_244 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_72);  reduce_scatter_tensor_72 = None
        convert_element_type_834 = torch.ops.prims.convert_element_type.default(wait_tensor_244, torch.float32);  wait_tensor_244 = None
        convert_element_type_836 = torch.ops.prims.convert_element_type.default(wait_tensor_118, torch.float32);  wait_tensor_118 = None
        mul_242 = torch.ops.aten.mul.Tensor(convert_element_type_834, convert_element_type_836);  convert_element_type_836 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_84, mul_242)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
        div_11 = torch.ops.aten.div.Tensor(mul_84, 8192)
        mul_245 = torch.ops.aten.mul.Tensor(div_11, sum_33);  div_11 = sum_33 = None
        sub_17 = torch.ops.aten.sub.Tensor(mul_242, mul_245);  mul_242 = mul_245 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_21);  sub_17 = rsqrt_21 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_834, mul_84);  convert_element_type_834 = mul_84 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
        convert_element_type_837 = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        add_103 = torch.ops.aten.add.Tensor(add_99, convert_element_type_837);  add_99 = convert_element_type_837 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(sum_34, torch.float32);  sum_34 = None
        reduce_scatter_tensor_73 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_22, 'sum', 8, '0');  convert_element_type_default_22 = None
        wait_tensor_245 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_73);  reduce_scatter_tensor_73 = None
        view_530 = torch.ops.aten.view.default(add_103, [8192, 8192])
        permute_353 = torch.ops.aten.permute.default(view_530, [1, 0])
        permute_116 = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3])
        view_266 = torch.ops.aten.view.default(permute_116, [1, 8192, 8192]);  permute_116 = None
        view_267 = torch.ops.aten.view.default(view_266, [8192, 8192]);  view_266 = None
        mm_191 = torch.ops.aten.mm.default(permute_353, view_267);  permute_353 = view_267 = None
        permute_354 = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        permute_117 = torch.ops.aten.permute.default(convert_element_type_347, [1, 0]);  convert_element_type_347 = None
        clone_42 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        all_gather_into_tensor_107 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_42, 8, '0');  clone_42 = None
        wait_tensor_117 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_107);  all_gather_into_tensor_107 = None
        permute_355 = torch.ops.aten.permute.default(wait_tensor_117, [1, 0]);  wait_tensor_117 = None
        mm_192 = torch.ops.aten.mm.default(view_530, permute_355);  view_530 = permute_355 = None
        view_531 = torch.ops.aten.view.default(mm_192, [1, 8192, 8192]);  mm_192 = None
        clone_85 = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
        reduce_scatter_tensor_74 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_85, 'sum', 8, '0');  clone_85 = None
        wait_tensor_246 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_74);  reduce_scatter_tensor_74 = None
        permute_356 = torch.ops.aten.permute.default(wait_tensor_246, [1, 0]);  wait_tensor_246 = None
        convert_element_type_844 = torch.ops.prims.convert_element_type.default(permute_356, torch.float32);  permute_356 = None
        view_532 = torch.ops.aten.view.default(view_531, [1, 8192, 32, 256]);  view_531 = None
        permute_357 = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_38 = torch.ops.aten.add.Scalar(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_19);  convert_element_type_318 = None
        all_gather_into_tensor_98 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_317, 8, '0');  convert_element_type_317 = None
        wait_tensor_107 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_98);  all_gather_into_tensor_98 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, wait_tensor_107)
        convert_element_type_319 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        all_gather_into_tensor_99 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_320, 4, '1');  convert_element_type_320 = None
        wait_tensor_108 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_99);  all_gather_into_tensor_99 = None
        permute_107 = torch.ops.aten.permute.default(wait_tensor_108, [1, 0]);  wait_tensor_108 = None
        all_gather_into_tensor_100 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_319, 2, '3');  convert_element_type_319 = None
        wait_tensor_109 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_100);  all_gather_into_tensor_100 = None
        view_244 = torch.ops.aten.view.default(wait_tensor_109, [16384, 8192]);  wait_tensor_109 = None
        mm_67 = torch.ops.aten.mm.default(view_244, permute_107)
        view_245 = torch.ops.aten.view.default(mm_67, [2, 8192, 14336]);  mm_67 = None
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_323)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_323, sigmoid_9);  sigmoid_9 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        view_247 = torch.ops.aten.view.default(mm_68, [2, 8192, 14336]);  mm_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_324, view_247)
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        permute_109 = torch.ops.aten.permute.default(convert_element_type_328, [1, 0]);  convert_element_type_328 = None
        view_248 = torch.ops.aten.view.default(mul_79, [16384, 14336]);  mul_79 = None
        clone_39 = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        all_gather_into_tensor_102 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_39, 4, '1');  clone_39 = None
        wait_tensor_111 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_102);  all_gather_into_tensor_102 = None
        mm_69 = torch.ops.aten.mm.default(view_248, wait_tensor_111)
        view_249 = torch.ops.aten.view.default(mm_69, [2, 8192, 8192]);  mm_69 = None
        reduce_scatter_tensor_9 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_249, 'sum', 2, '3');  view_249 = None
        wait_tensor_112 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_9);  reduce_scatter_tensor_9 = None
        add_39 = torch.ops.aten.add.Tensor(add_37, wait_tensor_112);  add_37 = wait_tensor_112 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [2], True);  pow_21 = None
        add_40 = torch.ops.aten.add.Scalar(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_20);  convert_element_type_332 = None
        all_gather_into_tensor_103 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_331, 8, '0');  convert_element_type_331 = None
        wait_tensor_113 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_103);  all_gather_into_tensor_103 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, wait_tensor_113)
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        view_250 = torch.ops.aten.view.default(convert_element_type_333, [8192, 8192]);  convert_element_type_333 = None
        view_251 = torch.ops.aten.view.default(mm_70, [1, 8192, 8192]);  mm_70 = None
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        all_gather_into_tensor_105 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_337, 8, '0');  convert_element_type_337 = None
        wait_tensor_115 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_105);  all_gather_into_tensor_105 = None
        permute_111 = torch.ops.aten.permute.default(wait_tensor_115, [1, 0]);  wait_tensor_115 = None
        mm_71 = torch.ops.aten.mm.default(view_250, permute_111)
        view_253 = torch.ops.aten.view.default(mm_71, [1, 8192, 2048]);  mm_71 = None
        view_255 = torch.ops.aten.view.default(mm_72, [1, 8192, 2048]);  mm_72 = None
        view_256 = torch.ops.aten.view.default(view_251, [1, 8192, 32, 256]);  view_251 = None
        view_257 = torch.ops.aten.view.default(view_253, [1, 8192, 8, 256]);  view_253 = None
        view_258 = torch.ops.aten.view.default(view_255, [1, 8192, 8, 256]);  view_255 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(view_256, torch.float32);  view_256 = None
        view_259 = torch.ops.aten.view.default(convert_element_type_343, [1, 8192, 32, 128, 2]);  convert_element_type_343 = None
        view_as_complex_20 = torch.ops.aten.view_as_complex.default(view_259);  view_259 = None
        convert_element_type_344 = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        view_260 = torch.ops.aten.view.default(convert_element_type_344, [1, 8192, 8, 128, 2]);  convert_element_type_344 = None
        view_as_complex_21 = torch.ops.aten.view_as_complex.default(view_260);  view_260 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_as_complex_20, view_11);  view_as_complex_20 = None
        view_as_real_20 = torch.ops.aten.view_as_real.default(mul_82);  mul_82 = None
        view_262 = torch.ops.aten.view.default(view_as_real_20, [1, 8192, 32, 256]);  view_as_real_20 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_as_complex_21, view_11);  view_as_complex_21 = None
        view_as_real_21 = torch.ops.aten.view_as_real.default(mul_83);  mul_83 = None
        view_263 = torch.ops.aten.view.default(view_as_real_21, [1, 8192, 8, 256]);  view_as_real_21 = None
        convert_element_type_345 = torch.ops.prims.convert_element_type.default(view_262, torch.bfloat16);  view_262 = None
        convert_element_type_346 = torch.ops.prims.convert_element_type.default(view_263, torch.bfloat16);  view_263 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(convert_element_type_346, 3);  convert_element_type_346 = None
        expand_20 = torch.ops.aten.expand.default(unsqueeze_20, [1, 8192, 8, 4, 256]);  unsqueeze_20 = None
        clone_40 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_264 = torch.ops.aten.view.default(clone_40, [1, 8192, 32, 256]);  clone_40 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(view_258, 3);  view_258 = None
        expand_21 = torch.ops.aten.expand.default(unsqueeze_21, [1, 8192, 8, 4, 256]);  unsqueeze_21 = None
        clone_41 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_265 = torch.ops.aten.view.default(clone_41, [1, 8192, 32, 256]);  clone_41 = None
        permute_113 = torch.ops.aten.permute.default(convert_element_type_345, [0, 2, 1, 3]);  convert_element_type_345 = None
        permute_114 = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        permute_115 = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_357, permute_113, permute_114, permute_115, getitem_94, getitem_95, None, None, 8192, 8192, 0.0, True, getitem_100, getitem_101, scale = 0.0625);  permute_357 = permute_113 = permute_114 = permute_115 = getitem_94 = getitem_95 = getitem_100 = getitem_101 = None
        getitem_163 = _scaled_dot_product_flash_attention_backward_5[0]
        getitem_164 = _scaled_dot_product_flash_attention_backward_5[1]
        getitem_165 = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
        permute_358 = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        permute_359 = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None
        permute_360 = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None
        view_533 = torch.ops.aten.view.default(permute_358, [1, 8192, 8, 4, 256]);  permute_358 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(view_533, [3], True);  view_533 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(sum_35, 3);  sum_35 = None
        view_534 = torch.ops.aten.view.default(permute_359, [1, 8192, 8, 4, 256]);  permute_359 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(view_534, [3], True);  view_534 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(sum_36, 3);  sum_36 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(squeeze_11, torch.float32);  squeeze_11 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(permute_360, torch.float32);  permute_360 = None
        view_535 = torch.ops.aten.view.default(convert_element_type_845, [1, 8192, 8, 128, 2]);  convert_element_type_845 = None
        view_as_complex_42 = torch.ops.aten.view_as_complex.default(view_535);  view_535 = None
        mul_248 = torch.ops.aten.mul.Tensor(view_as_complex_42, _conj);  view_as_complex_42 = None
        view_536 = torch.ops.aten.view.default(convert_element_type_846, [1, 8192, 32, 128, 2]);  convert_element_type_846 = None
        view_as_complex_43 = torch.ops.aten.view_as_complex.default(view_536);  view_536 = None
        mul_249 = torch.ops.aten.mul.Tensor(view_as_complex_43, _conj);  view_as_complex_43 = None
        view_as_real_42 = torch.ops.aten.view_as_real.default(mul_248);  mul_248 = None
        view_537 = torch.ops.aten.view.default(view_as_real_42, [1, 8192, 8, 256]);  view_as_real_42 = None
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(view_537, torch.bfloat16);  view_537 = None
        view_as_real_43 = torch.ops.aten.view_as_real.default(mul_249);  mul_249 = None
        view_538 = torch.ops.aten.view.default(view_as_real_43, [1, 8192, 32, 256]);  view_as_real_43 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        view_539 = torch.ops.aten.view.default(squeeze_10, [1, 8192, 2048]);  squeeze_10 = None
        view_540 = torch.ops.aten.view.default(convert_element_type_847, [1, 8192, 2048]);  convert_element_type_847 = None
        view_541 = torch.ops.aten.view.default(convert_element_type_848, [1, 8192, 8192]);  convert_element_type_848 = None
        view_542 = torch.ops.aten.view.default(view_539, [8192, 2048]);  view_539 = None
        permute_361 = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_193 = torch.ops.aten.mm.default(permute_361, view_250);  permute_361 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        all_gather_into_tensor_106 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_340, 8, '0');  convert_element_type_340 = None
        wait_tensor_116 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_106);  all_gather_into_tensor_106 = None
        permute_112 = torch.ops.aten.permute.default(wait_tensor_116, [1, 0]);  wait_tensor_116 = None
        permute_363 = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_194 = torch.ops.aten.mm.default(view_542, permute_363);  view_542 = permute_363 = None
        view_543 = torch.ops.aten.view.default(mm_194, [1, 8192, 8192]);  mm_194 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        reduce_scatter_tensor_75 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_853, 'sum', 8, '0');  convert_element_type_853 = None
        wait_tensor_247 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_75);  reduce_scatter_tensor_75 = None
        view_544 = torch.ops.aten.view.default(view_540, [8192, 2048]);  view_540 = None
        permute_365 = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_195 = torch.ops.aten.mm.default(permute_365, view_250);  permute_365 = None
        permute_367 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_196 = torch.ops.aten.mm.default(view_544, permute_367);  view_544 = permute_367 = None
        view_545 = torch.ops.aten.view.default(mm_196, [1, 8192, 8192]);  mm_196 = None
        add_104 = torch.ops.aten.add.Tensor(view_543, view_545);  view_543 = view_545 = None
        convert_element_type_858 = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        reduce_scatter_tensor_76 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_858, 'sum', 8, '0');  convert_element_type_858 = None
        wait_tensor_248 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_76);  reduce_scatter_tensor_76 = None
        view_546 = torch.ops.aten.view.default(view_541, [8192, 8192]);  view_541 = None
        permute_369 = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_197 = torch.ops.aten.mm.default(permute_369, view_250);  permute_369 = view_250 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        all_gather_into_tensor_104 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_334, 8, '0');  convert_element_type_334 = None
        wait_tensor_114 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_104);  all_gather_into_tensor_104 = None
        permute_110 = torch.ops.aten.permute.default(wait_tensor_114, [1, 0]);  wait_tensor_114 = None
        permute_371 = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_198 = torch.ops.aten.mm.default(view_546, permute_371);  view_546 = permute_371 = None
        view_547 = torch.ops.aten.view.default(mm_198, [1, 8192, 8192]);  mm_198 = None
        add_105 = torch.ops.aten.add.Tensor(add_104, view_547);  add_104 = view_547 = None
        reduce_scatter_tensor_77 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_197, 'sum', 8, '0');  mm_197 = None
        wait_tensor_249 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_77);  reduce_scatter_tensor_77 = None
        convert_element_type_863 = torch.ops.prims.convert_element_type.default(wait_tensor_249, torch.float32);  wait_tensor_249 = None
        convert_element_type_864 = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        convert_element_type_866 = torch.ops.prims.convert_element_type.default(wait_tensor_113, torch.float32);  wait_tensor_113 = None
        mul_250 = torch.ops.aten.mul.Tensor(convert_element_type_864, convert_element_type_866);  convert_element_type_866 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_80, mul_250)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        div_12 = torch.ops.aten.div.Tensor(mul_80, 8192)
        mul_253 = torch.ops.aten.mul.Tensor(div_12, sum_37);  div_12 = sum_37 = None
        sub_18 = torch.ops.aten.sub.Tensor(mul_250, mul_253);  mul_250 = mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_20);  sub_18 = rsqrt_20 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_864, mul_80);  convert_element_type_864 = mul_80 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        convert_element_type_867 = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        add_106 = torch.ops.aten.add.Tensor(add_103, convert_element_type_867);  add_103 = convert_element_type_867 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
        reduce_scatter_tensor_78 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_21, 'sum', 8, '0');  convert_element_type_default_21 = None
        wait_tensor_250 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_78);  reduce_scatter_tensor_78 = None
        all_gather_into_tensor_172 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_106, 2, '3')
        wait_tensor_251 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_172);  all_gather_into_tensor_172 = None
        view_548 = torch.ops.aten.view.default(wait_tensor_251, [16384, 8192]);  wait_tensor_251 = None
        permute_373 = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_199 = torch.ops.aten.mm.default(permute_373, view_248);  permute_373 = view_248 = None
        permute_374 = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
        permute_375 = torch.ops.aten.permute.default(wait_tensor_111, [1, 0]);  wait_tensor_111 = None
        mm_200 = torch.ops.aten.mm.default(view_548, permute_375);  view_548 = permute_375 = None
        view_549 = torch.ops.aten.view.default(mm_200, [2, 8192, 14336]);  mm_200 = None
        clone_88 = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        reduce_scatter_tensor_79 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_88, 'sum', 4, '1');  clone_88 = None
        wait_tensor_252 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_79);  reduce_scatter_tensor_79 = None
        permute_376 = torch.ops.aten.permute.default(wait_tensor_252, [1, 0]);  wait_tensor_252 = None
        convert_element_type_874 = torch.ops.prims.convert_element_type.default(permute_376, torch.float32);  permute_376 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_549, convert_element_type_324);  convert_element_type_324 = None
        mul_257 = torch.ops.aten.mul.Tensor(view_549, view_247);  view_549 = view_247 = None
        view_550 = torch.ops.aten.view.default(mul_256, [16384, 14336]);  mul_256 = None
        permute_377 = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_201 = torch.ops.aten.mm.default(permute_377, view_244);  permute_377 = None
        reduce_scatter_tensor_80 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_201, 'sum', 4, '1');  mm_201 = None
        wait_tensor_253 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_80);  reduce_scatter_tensor_80 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        all_gather_into_tensor_101 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_325, 4, '1');  convert_element_type_325 = None
        wait_tensor_110 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_101);  all_gather_into_tensor_101 = None
        permute_108 = torch.ops.aten.permute.default(wait_tensor_110, [1, 0]);  wait_tensor_110 = None
        permute_379 = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_202 = torch.ops.aten.mm.default(view_550, permute_379);  view_550 = permute_379 = None
        view_551 = torch.ops.aten.view.default(mm_202, [2, 8192, 8192]);  mm_202 = None
        convert_element_type_879 = torch.ops.prims.convert_element_type.default(wait_tensor_253, torch.float32);  wait_tensor_253 = None
        convert_element_type_880 = torch.ops.prims.convert_element_type.default(mul_257, torch.float32);  mul_257 = None
        neg_6 = torch.ops.aten.neg.default(convert_element_type_323)
        exp_6 = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_107 = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(add_107);  add_107 = None
        mul_258 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_259 = torch.ops.aten.mul.Tensor(convert_element_type_880, mul_258);  convert_element_type_880 = None
        sub_19 = torch.ops.aten.sub.Tensor(1, mul_258);  mul_258 = None
        mul_260 = torch.ops.aten.mul.Tensor(convert_element_type_323, sub_19);  convert_element_type_323 = sub_19 = None
        add_108 = torch.ops.aten.add.Tensor(mul_260, 1);  mul_260 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_259, add_108);  mul_259 = add_108 = None
        convert_element_type_882 = torch.ops.prims.convert_element_type.default(mul_261, torch.bfloat16);  mul_261 = None
        view_552 = torch.ops.aten.view.default(convert_element_type_882, [16384, 14336]);  convert_element_type_882 = None
        permute_381 = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_203 = torch.ops.aten.mm.default(permute_381, view_244);  permute_381 = view_244 = None
        reduce_scatter_tensor_81 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_203, 'sum', 4, '1');  mm_203 = None
        wait_tensor_254 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_81);  reduce_scatter_tensor_81 = None
        permute_383 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_204 = torch.ops.aten.mm.default(view_552, permute_383);  view_552 = permute_383 = None
        view_553 = torch.ops.aten.view.default(mm_204, [2, 8192, 8192]);  mm_204 = None
        add_109 = torch.ops.aten.add.Tensor(view_551, view_553);  view_551 = view_553 = None
        convert_element_type_887 = torch.ops.prims.convert_element_type.default(wait_tensor_254, torch.float32);  wait_tensor_254 = None
        reduce_scatter_tensor_82 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_109, 'sum', 2, '3');  add_109 = None
        wait_tensor_255 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_82);  reduce_scatter_tensor_82 = None
        convert_element_type_888 = torch.ops.prims.convert_element_type.default(wait_tensor_255, torch.float32);  wait_tensor_255 = None
        convert_element_type_890 = torch.ops.prims.convert_element_type.default(wait_tensor_107, torch.float32);  wait_tensor_107 = None
        mul_262 = torch.ops.aten.mul.Tensor(convert_element_type_888, convert_element_type_890);  convert_element_type_890 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_76, mul_262)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
        div_13 = torch.ops.aten.div.Tensor(mul_76, 8192)
        mul_265 = torch.ops.aten.mul.Tensor(div_13, sum_39);  div_13 = sum_39 = None
        sub_20 = torch.ops.aten.sub.Tensor(mul_262, mul_265);  mul_262 = mul_265 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_267 = torch.ops.aten.mul.Tensor(convert_element_type_888, mul_76);  convert_element_type_888 = mul_76 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1]);  mul_267 = None
        convert_element_type_891 = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None
        add_110 = torch.ops.aten.add.Tensor(add_106, convert_element_type_891);  add_106 = convert_element_type_891 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(sum_40, torch.float32);  sum_40 = None
        reduce_scatter_tensor_83 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_20, 'sum', 8, '0');  convert_element_type_default_20 = None
        wait_tensor_256 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_83);  reduce_scatter_tensor_83 = None
        view_554 = torch.ops.aten.view.default(add_110, [8192, 8192])
        permute_385 = torch.ops.aten.permute.default(view_554, [1, 0])
        permute_105 = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_241 = torch.ops.aten.view.default(permute_105, [1, 8192, 8192]);  permute_105 = None
        view_242 = torch.ops.aten.view.default(view_241, [8192, 8192]);  view_241 = None
        mm_205 = torch.ops.aten.mm.default(permute_385, view_242);  permute_385 = view_242 = None
        permute_386 = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        permute_106 = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        clone_38 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        all_gather_into_tensor_97 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_38, 8, '0');  clone_38 = None
        wait_tensor_106 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_97);  all_gather_into_tensor_97 = None
        permute_387 = torch.ops.aten.permute.default(wait_tensor_106, [1, 0]);  wait_tensor_106 = None
        mm_206 = torch.ops.aten.mm.default(view_554, permute_387);  view_554 = permute_387 = None
        view_555 = torch.ops.aten.view.default(mm_206, [1, 8192, 8192]);  mm_206 = None
        clone_89 = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        reduce_scatter_tensor_84 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_89, 'sum', 8, '0');  clone_89 = None
        wait_tensor_257 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_84);  reduce_scatter_tensor_84 = None
        permute_388 = torch.ops.aten.permute.default(wait_tensor_257, [1, 0]);  wait_tensor_257 = None
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(permute_388, torch.float32);  permute_388 = None
        view_556 = torch.ops.aten.view.default(view_555, [1, 8192, 32, 256]);  view_555 = None
        permute_389 = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        view_224 = torch.ops.aten.view.default(mm_62, [2, 8192, 8192]);  mm_62 = None
        reduce_scatter_tensor_8 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_224, 'sum', 2, '3');  view_224 = None
        wait_tensor_101 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_8);  reduce_scatter_tensor_8 = None
        add_35 = torch.ops.aten.add.Tensor(add_33, wait_tensor_101);  wait_tensor_101 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [2], True);  pow_19 = None
        add_36 = torch.ops.aten.add.Scalar(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_18);  convert_element_type_299 = None
        all_gather_into_tensor_93 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_298, 8, '0');  convert_element_type_298 = None
        wait_tensor_102 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_93);  all_gather_into_tensor_93 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, wait_tensor_102)
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        all_gather_into_tensor_94 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_301, 8, '0');  convert_element_type_301 = None
        wait_tensor_103 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_94);  all_gather_into_tensor_94 = None
        permute_99 = torch.ops.aten.permute.default(wait_tensor_103, [1, 0]);  wait_tensor_103 = None
        view_225 = torch.ops.aten.view.default(convert_element_type_300, [8192, 8192]);  convert_element_type_300 = None
        mm_63 = torch.ops.aten.mm.default(view_225, permute_99)
        view_226 = torch.ops.aten.view.default(mm_63, [1, 8192, 8192]);  mm_63 = None
        view_228 = torch.ops.aten.view.default(mm_64, [1, 8192, 2048]);  mm_64 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        all_gather_into_tensor_96 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_307, 8, '0');  convert_element_type_307 = None
        wait_tensor_105 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_96);  all_gather_into_tensor_96 = None
        permute_101 = torch.ops.aten.permute.default(wait_tensor_105, [1, 0]);  wait_tensor_105 = None
        mm_65 = torch.ops.aten.mm.default(view_225, permute_101)
        view_230 = torch.ops.aten.view.default(mm_65, [1, 8192, 2048]);  mm_65 = None
        view_231 = torch.ops.aten.view.default(view_226, [1, 8192, 32, 256]);  view_226 = None
        view_232 = torch.ops.aten.view.default(view_228, [1, 8192, 8, 256]);  view_228 = None
        view_233 = torch.ops.aten.view.default(view_230, [1, 8192, 8, 256]);  view_230 = None
        convert_element_type_310 = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        view_234 = torch.ops.aten.view.default(convert_element_type_310, [1, 8192, 32, 128, 2]);  convert_element_type_310 = None
        view_as_complex_18 = torch.ops.aten.view_as_complex.default(view_234);  view_234 = None
        convert_element_type_311 = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        view_235 = torch.ops.aten.view.default(convert_element_type_311, [1, 8192, 8, 128, 2]);  convert_element_type_311 = None
        view_as_complex_19 = torch.ops.aten.view_as_complex.default(view_235);  view_235 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_as_complex_18, view_11);  view_as_complex_18 = None
        view_as_real_18 = torch.ops.aten.view_as_real.default(mul_74);  mul_74 = None
        view_237 = torch.ops.aten.view.default(view_as_real_18, [1, 8192, 32, 256]);  view_as_real_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_as_complex_19, view_11);  view_as_complex_19 = None
        view_as_real_19 = torch.ops.aten.view_as_real.default(mul_75);  mul_75 = None
        view_238 = torch.ops.aten.view.default(view_as_real_19, [1, 8192, 8, 256]);  view_as_real_19 = None
        convert_element_type_312 = torch.ops.prims.convert_element_type.default(view_237, torch.bfloat16);  view_237 = None
        convert_element_type_313 = torch.ops.prims.convert_element_type.default(view_238, torch.bfloat16);  view_238 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(convert_element_type_313, 3);  convert_element_type_313 = None
        expand_18 = torch.ops.aten.expand.default(unsqueeze_18, [1, 8192, 8, 4, 256]);  unsqueeze_18 = None
        clone_36 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_239 = torch.ops.aten.view.default(clone_36, [1, 8192, 32, 256]);  clone_36 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(view_233, 3);  view_233 = None
        expand_19 = torch.ops.aten.expand.default(unsqueeze_19, [1, 8192, 8, 4, 256]);  unsqueeze_19 = None
        clone_37 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_240 = torch.ops.aten.view.default(clone_37, [1, 8192, 32, 256]);  clone_37 = None
        permute_102 = torch.ops.aten.permute.default(convert_element_type_312, [0, 2, 1, 3]);  convert_element_type_312 = None
        permute_103 = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        permute_104 = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_389, permute_102, permute_103, permute_104, getitem_85, getitem_86, None, None, 8192, 8192, 0.0, True, getitem_91, getitem_92, scale = 0.0625);  permute_389 = permute_102 = permute_103 = permute_104 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_166 = _scaled_dot_product_flash_attention_backward_6[0]
        getitem_167 = _scaled_dot_product_flash_attention_backward_6[1]
        getitem_168 = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
        permute_390 = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
        permute_391 = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        permute_392 = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
        view_557 = torch.ops.aten.view.default(permute_390, [1, 8192, 8, 4, 256]);  permute_390 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(view_557, [3], True);  view_557 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(sum_41, 3);  sum_41 = None
        view_558 = torch.ops.aten.view.default(permute_391, [1, 8192, 8, 4, 256]);  permute_391 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_558, [3], True);  view_558 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(sum_42, 3);  sum_42 = None
        convert_element_type_899 = torch.ops.prims.convert_element_type.default(squeeze_13, torch.float32);  squeeze_13 = None
        convert_element_type_900 = torch.ops.prims.convert_element_type.default(permute_392, torch.float32);  permute_392 = None
        view_559 = torch.ops.aten.view.default(convert_element_type_899, [1, 8192, 8, 128, 2]);  convert_element_type_899 = None
        view_as_complex_44 = torch.ops.aten.view_as_complex.default(view_559);  view_559 = None
        mul_268 = torch.ops.aten.mul.Tensor(view_as_complex_44, _conj);  view_as_complex_44 = None
        view_560 = torch.ops.aten.view.default(convert_element_type_900, [1, 8192, 32, 128, 2]);  convert_element_type_900 = None
        view_as_complex_45 = torch.ops.aten.view_as_complex.default(view_560);  view_560 = None
        mul_269 = torch.ops.aten.mul.Tensor(view_as_complex_45, _conj);  view_as_complex_45 = None
        view_as_real_44 = torch.ops.aten.view_as_real.default(mul_268);  mul_268 = None
        view_561 = torch.ops.aten.view.default(view_as_real_44, [1, 8192, 8, 256]);  view_as_real_44 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(view_561, torch.bfloat16);  view_561 = None
        view_as_real_45 = torch.ops.aten.view_as_real.default(mul_269);  mul_269 = None
        view_562 = torch.ops.aten.view.default(view_as_real_45, [1, 8192, 32, 256]);  view_as_real_45 = None
        convert_element_type_902 = torch.ops.prims.convert_element_type.default(view_562, torch.bfloat16);  view_562 = None
        view_563 = torch.ops.aten.view.default(squeeze_12, [1, 8192, 2048]);  squeeze_12 = None
        view_564 = torch.ops.aten.view.default(convert_element_type_901, [1, 8192, 2048]);  convert_element_type_901 = None
        view_565 = torch.ops.aten.view.default(convert_element_type_902, [1, 8192, 8192]);  convert_element_type_902 = None
        view_566 = torch.ops.aten.view.default(view_563, [8192, 2048]);  view_563 = None
        permute_393 = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_207 = torch.ops.aten.mm.default(permute_393, view_225);  permute_393 = None
        permute_395 = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_208 = torch.ops.aten.mm.default(view_566, permute_395);  view_566 = permute_395 = None
        view_567 = torch.ops.aten.view.default(mm_208, [1, 8192, 8192]);  mm_208 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        reduce_scatter_tensor_85 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_907, 'sum', 8, '0');  convert_element_type_907 = None
        wait_tensor_258 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_85);  reduce_scatter_tensor_85 = None
        view_568 = torch.ops.aten.view.default(view_564, [8192, 2048]);  view_564 = None
        permute_397 = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_209 = torch.ops.aten.mm.default(permute_397, view_225);  permute_397 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        all_gather_into_tensor_95 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_304, 8, '0');  convert_element_type_304 = None
        wait_tensor_104 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_95);  all_gather_into_tensor_95 = None
        permute_100 = torch.ops.aten.permute.default(wait_tensor_104, [1, 0]);  wait_tensor_104 = None
        permute_399 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_210 = torch.ops.aten.mm.default(view_568, permute_399);  view_568 = permute_399 = None
        view_569 = torch.ops.aten.view.default(mm_210, [1, 8192, 8192]);  mm_210 = None
        add_111 = torch.ops.aten.add.Tensor(view_567, view_569);  view_567 = view_569 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        reduce_scatter_tensor_86 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_912, 'sum', 8, '0');  convert_element_type_912 = None
        wait_tensor_259 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_86);  reduce_scatter_tensor_86 = None
        view_570 = torch.ops.aten.view.default(view_565, [8192, 8192]);  view_565 = None
        permute_401 = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_211 = torch.ops.aten.mm.default(permute_401, view_225);  permute_401 = view_225 = None
        permute_403 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_212 = torch.ops.aten.mm.default(view_570, permute_403);  view_570 = permute_403 = None
        view_571 = torch.ops.aten.view.default(mm_212, [1, 8192, 8192]);  mm_212 = None
        add_112 = torch.ops.aten.add.Tensor(add_111, view_571);  add_111 = view_571 = None
        reduce_scatter_tensor_87 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_211, 'sum', 8, '0');  mm_211 = None
        wait_tensor_260 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_87);  reduce_scatter_tensor_87 = None
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(wait_tensor_260, torch.float32);  wait_tensor_260 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(add_112, torch.float32);  add_112 = None
        convert_element_type_920 = torch.ops.prims.convert_element_type.default(wait_tensor_102, torch.float32);  wait_tensor_102 = None
        mul_270 = torch.ops.aten.mul.Tensor(convert_element_type_918, convert_element_type_920);  convert_element_type_920 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_72, mul_270)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        div_14 = torch.ops.aten.div.Tensor(mul_72, 8192)
        mul_273 = torch.ops.aten.mul.Tensor(div_14, sum_43);  div_14 = sum_43 = None
        sub_21 = torch.ops.aten.sub.Tensor(mul_270, mul_273);  mul_270 = mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_18);  sub_21 = rsqrt_18 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_918, mul_72);  convert_element_type_918 = mul_72 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        convert_element_type_921 = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None
        add_113 = torch.ops.aten.add.Tensor(add_110, convert_element_type_921);  add_110 = convert_element_type_921 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None
        reduce_scatter_tensor_88 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_19, 'sum', 8, '0');  convert_element_type_default_19 = None
        wait_tensor_261 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_88);  reduce_scatter_tensor_88 = None
        all_gather_into_tensor_173 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_113, 2, '3')
        wait_tensor_262 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_173);  all_gather_into_tensor_173 = None
        view_572 = torch.ops.aten.view.default(wait_tensor_262, [16384, 8192]);  wait_tensor_262 = None
        permute_405 = torch.ops.aten.permute.default(view_572, [1, 0])
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [2], True);  pow_18 = None
        add_34 = torch.ops.aten.add.Scalar(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_285, rsqrt_17);  convert_element_type_285 = None
        all_gather_into_tensor_88 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_284, 8, '0');  convert_element_type_284 = None
        wait_tensor_96 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_88);  all_gather_into_tensor_88 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, wait_tensor_96)
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        all_gather_into_tensor_90 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_286, 2, '3');  convert_element_type_286 = None
        wait_tensor_98 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_90);  all_gather_into_tensor_90 = None
        view_219 = torch.ops.aten.view.default(wait_tensor_98, [16384, 8192]);  wait_tensor_98 = None
        view_220 = torch.ops.aten.view.default(mm_60, [2, 8192, 14336]);  mm_60 = None
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_290)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_290, sigmoid_8);  sigmoid_8 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        all_gather_into_tensor_91 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_292, 4, '1');  convert_element_type_292 = None
        wait_tensor_99 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_91);  all_gather_into_tensor_91 = None
        permute_97 = torch.ops.aten.permute.default(wait_tensor_99, [1, 0]);  wait_tensor_99 = None
        mm_61 = torch.ops.aten.mm.default(view_219, permute_97)
        view_222 = torch.ops.aten.view.default(mm_61, [2, 8192, 14336]);  mm_61 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_291, view_222)
        view_223 = torch.ops.aten.view.default(mul_71, [16384, 14336]);  mul_71 = None
        mm_213 = torch.ops.aten.mm.default(permute_405, view_223);  permute_405 = view_223 = None
        permute_406 = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        permute_98 = torch.ops.aten.permute.default(convert_element_type_295, [1, 0]);  convert_element_type_295 = None
        clone_35 = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
        all_gather_into_tensor_92 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_35, 4, '1');  clone_35 = None
        wait_tensor_100 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_92);  all_gather_into_tensor_92 = None
        permute_407 = torch.ops.aten.permute.default(wait_tensor_100, [1, 0]);  wait_tensor_100 = None
        mm_214 = torch.ops.aten.mm.default(view_572, permute_407);  view_572 = permute_407 = None
        view_573 = torch.ops.aten.view.default(mm_214, [2, 8192, 14336]);  mm_214 = None
        clone_92 = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
        reduce_scatter_tensor_89 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_92, 'sum', 4, '1');  clone_92 = None
        wait_tensor_263 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_89);  reduce_scatter_tensor_89 = None
        permute_408 = torch.ops.aten.permute.default(wait_tensor_263, [1, 0]);  wait_tensor_263 = None
        convert_element_type_928 = torch.ops.prims.convert_element_type.default(permute_408, torch.float32);  permute_408 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_573, convert_element_type_291);  convert_element_type_291 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_573, view_222);  view_573 = view_222 = None
        view_574 = torch.ops.aten.view.default(mul_276, [16384, 14336]);  mul_276 = None
        permute_409 = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_215 = torch.ops.aten.mm.default(permute_409, view_219);  permute_409 = None
        reduce_scatter_tensor_90 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_215, 'sum', 4, '1');  mm_215 = None
        wait_tensor_264 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_90);  reduce_scatter_tensor_90 = None
        permute_411 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_216 = torch.ops.aten.mm.default(view_574, permute_411);  view_574 = permute_411 = None
        view_575 = torch.ops.aten.view.default(mm_216, [2, 8192, 8192]);  mm_216 = None
        convert_element_type_933 = torch.ops.prims.convert_element_type.default(wait_tensor_264, torch.float32);  wait_tensor_264 = None
        convert_element_type_934 = torch.ops.prims.convert_element_type.default(mul_277, torch.float32);  mul_277 = None
        neg_7 = torch.ops.aten.neg.default(convert_element_type_290)
        exp_7 = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_114 = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(add_114);  add_114 = None
        mul_278 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_279 = torch.ops.aten.mul.Tensor(convert_element_type_934, mul_278);  convert_element_type_934 = None
        sub_22 = torch.ops.aten.sub.Tensor(1, mul_278);  mul_278 = None
        mul_280 = torch.ops.aten.mul.Tensor(convert_element_type_290, sub_22);  convert_element_type_290 = sub_22 = None
        add_115 = torch.ops.aten.add.Tensor(mul_280, 1);  mul_280 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_279, add_115);  mul_279 = add_115 = None
        convert_element_type_936 = torch.ops.prims.convert_element_type.default(mul_281, torch.bfloat16);  mul_281 = None
        view_576 = torch.ops.aten.view.default(convert_element_type_936, [16384, 14336]);  convert_element_type_936 = None
        permute_413 = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_217 = torch.ops.aten.mm.default(permute_413, view_219);  permute_413 = view_219 = None
        reduce_scatter_tensor_91 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_217, 'sum', 4, '1');  mm_217 = None
        wait_tensor_265 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_91);  reduce_scatter_tensor_91 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        all_gather_into_tensor_89 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_287, 4, '1');  convert_element_type_287 = None
        wait_tensor_97 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_89);  all_gather_into_tensor_89 = None
        permute_96 = torch.ops.aten.permute.default(wait_tensor_97, [1, 0]);  wait_tensor_97 = None
        permute_415 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_218 = torch.ops.aten.mm.default(view_576, permute_415);  view_576 = permute_415 = None
        view_577 = torch.ops.aten.view.default(mm_218, [2, 8192, 8192]);  mm_218 = None
        add_116 = torch.ops.aten.add.Tensor(view_575, view_577);  view_575 = view_577 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(wait_tensor_265, torch.float32);  wait_tensor_265 = None
        reduce_scatter_tensor_92 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_116, 'sum', 2, '3');  add_116 = None
        wait_tensor_266 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_92);  reduce_scatter_tensor_92 = None
        convert_element_type_942 = torch.ops.prims.convert_element_type.default(wait_tensor_266, torch.float32);  wait_tensor_266 = None
        convert_element_type_944 = torch.ops.prims.convert_element_type.default(wait_tensor_96, torch.float32);  wait_tensor_96 = None
        mul_282 = torch.ops.aten.mul.Tensor(convert_element_type_942, convert_element_type_944);  convert_element_type_944 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_68, mul_282)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        div_15 = torch.ops.aten.div.Tensor(mul_68, 8192)
        mul_285 = torch.ops.aten.mul.Tensor(div_15, sum_45);  div_15 = sum_45 = None
        sub_23 = torch.ops.aten.sub.Tensor(mul_282, mul_285);  mul_282 = mul_285 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
        mul_287 = torch.ops.aten.mul.Tensor(convert_element_type_942, mul_68);  convert_element_type_942 = mul_68 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        convert_element_type_945 = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        add_117 = torch.ops.aten.add.Tensor(add_113, convert_element_type_945);  add_113 = convert_element_type_945 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(sum_46, torch.float32);  sum_46 = None
        reduce_scatter_tensor_93 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_18, 'sum', 8, '0');  convert_element_type_default_18 = None
        wait_tensor_267 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_93);  reduce_scatter_tensor_93 = None
        view_578 = torch.ops.aten.view.default(add_117, [8192, 8192])
        permute_417 = torch.ops.aten.permute.default(view_578, [1, 0])
        permute_94 = torch.ops.aten.permute.default(getitem_76, [0, 2, 1, 3])
        view_216 = torch.ops.aten.view.default(permute_94, [1, 8192, 8192]);  permute_94 = None
        view_217 = torch.ops.aten.view.default(view_216, [8192, 8192]);  view_216 = None
        mm_219 = torch.ops.aten.mm.default(permute_417, view_217);  permute_417 = view_217 = None
        permute_418 = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        permute_95 = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        clone_34 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        all_gather_into_tensor_87 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_34, 8, '0');  clone_34 = None
        wait_tensor_95 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_87);  all_gather_into_tensor_87 = None
        permute_419 = torch.ops.aten.permute.default(wait_tensor_95, [1, 0]);  wait_tensor_95 = None
        mm_220 = torch.ops.aten.mm.default(view_578, permute_419);  view_578 = permute_419 = None
        view_579 = torch.ops.aten.view.default(mm_220, [1, 8192, 8192]);  mm_220 = None
        clone_93 = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        reduce_scatter_tensor_94 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_93, 'sum', 8, '0');  clone_93 = None
        wait_tensor_268 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_94);  reduce_scatter_tensor_94 = None
        permute_420 = torch.ops.aten.permute.default(wait_tensor_268, [1, 0]);  wait_tensor_268 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(permute_420, torch.float32);  permute_420 = None
        view_580 = torch.ops.aten.view.default(view_579, [1, 8192, 32, 256]);  view_579 = None
        permute_421 = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [2], True);  pow_16 = None
        add_30 = torch.ops.aten.add.Scalar(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_15);  convert_element_type_252 = None
        all_gather_into_tensor_78 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_251, 8, '0');  convert_element_type_251 = None
        wait_tensor_85 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_78);  all_gather_into_tensor_78 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, wait_tensor_85)
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        all_gather_into_tensor_79 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_254, 4, '1');  convert_element_type_254 = None
        wait_tensor_86 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_79);  all_gather_into_tensor_79 = None
        permute_85 = torch.ops.aten.permute.default(wait_tensor_86, [1, 0]);  wait_tensor_86 = None
        all_gather_into_tensor_80 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_253, 2, '3');  convert_element_type_253 = None
        wait_tensor_87 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_80);  all_gather_into_tensor_80 = None
        view_194 = torch.ops.aten.view.default(wait_tensor_87, [16384, 8192]);  wait_tensor_87 = None
        mm_53 = torch.ops.aten.mm.default(view_194, permute_85)
        view_195 = torch.ops.aten.view.default(mm_53, [2, 8192, 14336]);  mm_53 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_7);  sigmoid_7 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        view_197 = torch.ops.aten.view.default(mm_54, [2, 8192, 14336]);  mm_54 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_258, view_197)
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        permute_87 = torch.ops.aten.permute.default(convert_element_type_262, [1, 0]);  convert_element_type_262 = None
        view_198 = torch.ops.aten.view.default(mul_63, [16384, 14336]);  mul_63 = None
        clone_31 = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        all_gather_into_tensor_82 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_31, 4, '1');  clone_31 = None
        wait_tensor_89 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_82);  all_gather_into_tensor_82 = None
        mm_55 = torch.ops.aten.mm.default(view_198, wait_tensor_89)
        view_199 = torch.ops.aten.view.default(mm_55, [2, 8192, 8192]);  mm_55 = None
        reduce_scatter_tensor_7 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_199, 'sum', 2, '3');  view_199 = None
        wait_tensor_90 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7);  reduce_scatter_tensor_7 = None
        add_31 = torch.ops.aten.add.Tensor(add_29, wait_tensor_90);  add_29 = wait_tensor_90 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_266 = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_32 = torch.ops.aten.add.Scalar(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_16);  convert_element_type_266 = None
        all_gather_into_tensor_83 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_265, 8, '0');  convert_element_type_265 = None
        wait_tensor_91 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_83);  all_gather_into_tensor_83 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, wait_tensor_91)
        convert_element_type_267 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        view_200 = torch.ops.aten.view.default(convert_element_type_267, [8192, 8192]);  convert_element_type_267 = None
        view_201 = torch.ops.aten.view.default(mm_56, [1, 8192, 8192]);  mm_56 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        all_gather_into_tensor_85 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_271, 8, '0');  convert_element_type_271 = None
        wait_tensor_93 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_85);  all_gather_into_tensor_85 = None
        permute_89 = torch.ops.aten.permute.default(wait_tensor_93, [1, 0]);  wait_tensor_93 = None
        mm_57 = torch.ops.aten.mm.default(view_200, permute_89)
        view_203 = torch.ops.aten.view.default(mm_57, [1, 8192, 2048]);  mm_57 = None
        view_205 = torch.ops.aten.view.default(mm_58, [1, 8192, 2048]);  mm_58 = None
        view_206 = torch.ops.aten.view.default(view_201, [1, 8192, 32, 256]);  view_201 = None
        view_207 = torch.ops.aten.view.default(view_203, [1, 8192, 8, 256]);  view_203 = None
        view_208 = torch.ops.aten.view.default(view_205, [1, 8192, 8, 256]);  view_205 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None
        view_209 = torch.ops.aten.view.default(convert_element_type_277, [1, 8192, 32, 128, 2]);  convert_element_type_277 = None
        view_as_complex_16 = torch.ops.aten.view_as_complex.default(view_209);  view_209 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(view_207, torch.float32);  view_207 = None
        view_210 = torch.ops.aten.view.default(convert_element_type_278, [1, 8192, 8, 128, 2]);  convert_element_type_278 = None
        view_as_complex_17 = torch.ops.aten.view_as_complex.default(view_210);  view_210 = None
        mul_66 = torch.ops.aten.mul.Tensor(view_as_complex_16, view_11);  view_as_complex_16 = None
        view_as_real_16 = torch.ops.aten.view_as_real.default(mul_66);  mul_66 = None
        view_212 = torch.ops.aten.view.default(view_as_real_16, [1, 8192, 32, 256]);  view_as_real_16 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_as_complex_17, view_11);  view_as_complex_17 = None
        view_as_real_17 = torch.ops.aten.view_as_real.default(mul_67);  mul_67 = None
        view_213 = torch.ops.aten.view.default(view_as_real_17, [1, 8192, 8, 256]);  view_as_real_17 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(view_212, torch.bfloat16);  view_212 = None
        convert_element_type_280 = torch.ops.prims.convert_element_type.default(view_213, torch.bfloat16);  view_213 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3);  convert_element_type_280 = None
        expand_16 = torch.ops.aten.expand.default(unsqueeze_16, [1, 8192, 8, 4, 256]);  unsqueeze_16 = None
        clone_32 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_214 = torch.ops.aten.view.default(clone_32, [1, 8192, 32, 256]);  clone_32 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_208, 3);  view_208 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_17, [1, 8192, 8, 4, 256]);  unsqueeze_17 = None
        clone_33 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_215 = torch.ops.aten.view.default(clone_33, [1, 8192, 32, 256]);  clone_33 = None
        permute_91 = torch.ops.aten.permute.default(convert_element_type_279, [0, 2, 1, 3]);  convert_element_type_279 = None
        permute_92 = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        permute_93 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_421, permute_91, permute_92, permute_93, getitem_76, getitem_77, None, None, 8192, 8192, 0.0, True, getitem_82, getitem_83, scale = 0.0625);  permute_421 = permute_91 = permute_92 = permute_93 = getitem_76 = getitem_77 = getitem_82 = getitem_83 = None
        getitem_169 = _scaled_dot_product_flash_attention_backward_7[0]
        getitem_170 = _scaled_dot_product_flash_attention_backward_7[1]
        getitem_171 = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
        permute_422 = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None
        permute_423 = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None
        permute_424 = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None
        view_581 = torch.ops.aten.view.default(permute_422, [1, 8192, 8, 4, 256]);  permute_422 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(view_581, [3], True);  view_581 = None
        squeeze_14 = torch.ops.aten.squeeze.dim(sum_47, 3);  sum_47 = None
        view_582 = torch.ops.aten.view.default(permute_423, [1, 8192, 8, 4, 256]);  permute_423 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(view_582, [3], True);  view_582 = None
        squeeze_15 = torch.ops.aten.squeeze.dim(sum_48, 3);  sum_48 = None
        convert_element_type_953 = torch.ops.prims.convert_element_type.default(squeeze_15, torch.float32);  squeeze_15 = None
        convert_element_type_954 = torch.ops.prims.convert_element_type.default(permute_424, torch.float32);  permute_424 = None
        view_583 = torch.ops.aten.view.default(convert_element_type_953, [1, 8192, 8, 128, 2]);  convert_element_type_953 = None
        view_as_complex_46 = torch.ops.aten.view_as_complex.default(view_583);  view_583 = None
        mul_288 = torch.ops.aten.mul.Tensor(view_as_complex_46, _conj);  view_as_complex_46 = None
        view_584 = torch.ops.aten.view.default(convert_element_type_954, [1, 8192, 32, 128, 2]);  convert_element_type_954 = None
        view_as_complex_47 = torch.ops.aten.view_as_complex.default(view_584);  view_584 = None
        mul_289 = torch.ops.aten.mul.Tensor(view_as_complex_47, _conj);  view_as_complex_47 = None
        view_as_real_46 = torch.ops.aten.view_as_real.default(mul_288);  mul_288 = None
        view_585 = torch.ops.aten.view.default(view_as_real_46, [1, 8192, 8, 256]);  view_as_real_46 = None
        convert_element_type_955 = torch.ops.prims.convert_element_type.default(view_585, torch.bfloat16);  view_585 = None
        view_as_real_47 = torch.ops.aten.view_as_real.default(mul_289);  mul_289 = None
        view_586 = torch.ops.aten.view.default(view_as_real_47, [1, 8192, 32, 256]);  view_as_real_47 = None
        convert_element_type_956 = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None
        view_587 = torch.ops.aten.view.default(squeeze_14, [1, 8192, 2048]);  squeeze_14 = None
        view_588 = torch.ops.aten.view.default(convert_element_type_955, [1, 8192, 2048]);  convert_element_type_955 = None
        view_589 = torch.ops.aten.view.default(convert_element_type_956, [1, 8192, 8192]);  convert_element_type_956 = None
        view_590 = torch.ops.aten.view.default(view_587, [8192, 2048]);  view_587 = None
        permute_425 = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_221 = torch.ops.aten.mm.default(permute_425, view_200);  permute_425 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        all_gather_into_tensor_86 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_274, 8, '0');  convert_element_type_274 = None
        wait_tensor_94 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_86);  all_gather_into_tensor_86 = None
        permute_90 = torch.ops.aten.permute.default(wait_tensor_94, [1, 0]);  wait_tensor_94 = None
        permute_427 = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_222 = torch.ops.aten.mm.default(view_590, permute_427);  view_590 = permute_427 = None
        view_591 = torch.ops.aten.view.default(mm_222, [1, 8192, 8192]);  mm_222 = None
        convert_element_type_961 = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        reduce_scatter_tensor_95 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_961, 'sum', 8, '0');  convert_element_type_961 = None
        wait_tensor_269 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_95);  reduce_scatter_tensor_95 = None
        view_592 = torch.ops.aten.view.default(view_588, [8192, 2048]);  view_588 = None
        permute_429 = torch.ops.aten.permute.default(view_592, [1, 0])
        mm_223 = torch.ops.aten.mm.default(permute_429, view_200);  permute_429 = None
        permute_431 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_224 = torch.ops.aten.mm.default(view_592, permute_431);  view_592 = permute_431 = None
        view_593 = torch.ops.aten.view.default(mm_224, [1, 8192, 8192]);  mm_224 = None
        add_118 = torch.ops.aten.add.Tensor(view_591, view_593);  view_591 = view_593 = None
        convert_element_type_966 = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        reduce_scatter_tensor_96 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_966, 'sum', 8, '0');  convert_element_type_966 = None
        wait_tensor_270 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_96);  reduce_scatter_tensor_96 = None
        view_594 = torch.ops.aten.view.default(view_589, [8192, 8192]);  view_589 = None
        permute_433 = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_225 = torch.ops.aten.mm.default(permute_433, view_200);  permute_433 = view_200 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        all_gather_into_tensor_84 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_268, 8, '0');  convert_element_type_268 = None
        wait_tensor_92 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_84);  all_gather_into_tensor_84 = None
        permute_88 = torch.ops.aten.permute.default(wait_tensor_92, [1, 0]);  wait_tensor_92 = None
        permute_435 = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_226 = torch.ops.aten.mm.default(view_594, permute_435);  view_594 = permute_435 = None
        view_595 = torch.ops.aten.view.default(mm_226, [1, 8192, 8192]);  mm_226 = None
        add_119 = torch.ops.aten.add.Tensor(add_118, view_595);  add_118 = view_595 = None
        reduce_scatter_tensor_97 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_225, 'sum', 8, '0');  mm_225 = None
        wait_tensor_271 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_97);  reduce_scatter_tensor_97 = None
        convert_element_type_971 = torch.ops.prims.convert_element_type.default(wait_tensor_271, torch.float32);  wait_tensor_271 = None
        convert_element_type_972 = torch.ops.prims.convert_element_type.default(add_119, torch.float32);  add_119 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(wait_tensor_91, torch.float32);  wait_tensor_91 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_972, convert_element_type_974);  convert_element_type_974 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_64, mul_290)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        div_16 = torch.ops.aten.div.Tensor(mul_64, 8192)
        mul_293 = torch.ops.aten.mul.Tensor(div_16, sum_49);  div_16 = sum_49 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_290, mul_293);  mul_290 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_295 = torch.ops.aten.mul.Tensor(convert_element_type_972, mul_64);  convert_element_type_972 = mul_64 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        convert_element_type_975 = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None
        add_120 = torch.ops.aten.add.Tensor(add_117, convert_element_type_975);  add_117 = convert_element_type_975 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(sum_50, torch.float32);  sum_50 = None
        reduce_scatter_tensor_98 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_17, 'sum', 8, '0');  convert_element_type_default_17 = None
        wait_tensor_272 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_98);  reduce_scatter_tensor_98 = None
        all_gather_into_tensor_174 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_120, 2, '3')
        wait_tensor_273 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_174);  all_gather_into_tensor_174 = None
        view_596 = torch.ops.aten.view.default(wait_tensor_273, [16384, 8192]);  wait_tensor_273 = None
        permute_437 = torch.ops.aten.permute.default(view_596, [1, 0])
        mm_227 = torch.ops.aten.mm.default(permute_437, view_198);  permute_437 = view_198 = None
        permute_438 = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
        permute_439 = torch.ops.aten.permute.default(wait_tensor_89, [1, 0]);  wait_tensor_89 = None
        mm_228 = torch.ops.aten.mm.default(view_596, permute_439);  view_596 = permute_439 = None
        view_597 = torch.ops.aten.view.default(mm_228, [2, 8192, 14336]);  mm_228 = None
        clone_96 = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
        reduce_scatter_tensor_99 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_96, 'sum', 4, '1');  clone_96 = None
        wait_tensor_274 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_99);  reduce_scatter_tensor_99 = None
        permute_440 = torch.ops.aten.permute.default(wait_tensor_274, [1, 0]);  wait_tensor_274 = None
        convert_element_type_982 = torch.ops.prims.convert_element_type.default(permute_440, torch.float32);  permute_440 = None
        mul_296 = torch.ops.aten.mul.Tensor(view_597, convert_element_type_258);  convert_element_type_258 = None
        mul_297 = torch.ops.aten.mul.Tensor(view_597, view_197);  view_597 = view_197 = None
        view_598 = torch.ops.aten.view.default(mul_296, [16384, 14336]);  mul_296 = None
        permute_441 = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_229 = torch.ops.aten.mm.default(permute_441, view_194);  permute_441 = None
        reduce_scatter_tensor_100 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_229, 'sum', 4, '1');  mm_229 = None
        wait_tensor_275 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_100);  reduce_scatter_tensor_100 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        all_gather_into_tensor_81 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_259, 4, '1');  convert_element_type_259 = None
        wait_tensor_88 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_81);  all_gather_into_tensor_81 = None
        permute_86 = torch.ops.aten.permute.default(wait_tensor_88, [1, 0]);  wait_tensor_88 = None
        permute_443 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_230 = torch.ops.aten.mm.default(view_598, permute_443);  view_598 = permute_443 = None
        view_599 = torch.ops.aten.view.default(mm_230, [2, 8192, 8192]);  mm_230 = None
        convert_element_type_987 = torch.ops.prims.convert_element_type.default(wait_tensor_275, torch.float32);  wait_tensor_275 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(mul_297, torch.float32);  mul_297 = None
        neg_8 = torch.ops.aten.neg.default(convert_element_type_257)
        exp_8 = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_121 = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(add_121);  add_121 = None
        mul_298 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_299 = torch.ops.aten.mul.Tensor(convert_element_type_988, mul_298);  convert_element_type_988 = None
        sub_25 = torch.ops.aten.sub.Tensor(1, mul_298);  mul_298 = None
        mul_300 = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_25);  convert_element_type_257 = sub_25 = None
        add_122 = torch.ops.aten.add.Tensor(mul_300, 1);  mul_300 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_299, add_122);  mul_299 = add_122 = None
        convert_element_type_990 = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16);  mul_301 = None
        view_600 = torch.ops.aten.view.default(convert_element_type_990, [16384, 14336]);  convert_element_type_990 = None
        permute_445 = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_231 = torch.ops.aten.mm.default(permute_445, view_194);  permute_445 = view_194 = None
        reduce_scatter_tensor_101 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_231, 'sum', 4, '1');  mm_231 = None
        wait_tensor_276 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_101);  reduce_scatter_tensor_101 = None
        permute_447 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_232 = torch.ops.aten.mm.default(view_600, permute_447);  view_600 = permute_447 = None
        view_601 = torch.ops.aten.view.default(mm_232, [2, 8192, 8192]);  mm_232 = None
        add_123 = torch.ops.aten.add.Tensor(view_599, view_601);  view_599 = view_601 = None
        convert_element_type_995 = torch.ops.prims.convert_element_type.default(wait_tensor_276, torch.float32);  wait_tensor_276 = None
        reduce_scatter_tensor_102 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_123, 'sum', 2, '3');  add_123 = None
        wait_tensor_277 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_102);  reduce_scatter_tensor_102 = None
        convert_element_type_996 = torch.ops.prims.convert_element_type.default(wait_tensor_277, torch.float32);  wait_tensor_277 = None
        convert_element_type_998 = torch.ops.prims.convert_element_type.default(wait_tensor_85, torch.float32);  wait_tensor_85 = None
        mul_302 = torch.ops.aten.mul.Tensor(convert_element_type_996, convert_element_type_998);  convert_element_type_998 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_60, mul_302)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        div_17 = torch.ops.aten.div.Tensor(mul_60, 8192)
        mul_305 = torch.ops.aten.mul.Tensor(div_17, sum_51);  div_17 = sum_51 = None
        sub_26 = torch.ops.aten.sub.Tensor(mul_302, mul_305);  mul_302 = mul_305 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = rsqrt_15 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_996, mul_60);  convert_element_type_996 = mul_60 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        convert_element_type_999 = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        add_124 = torch.ops.aten.add.Tensor(add_120, convert_element_type_999);  add_120 = convert_element_type_999 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None
        reduce_scatter_tensor_103 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_16, 'sum', 8, '0');  convert_element_type_default_16 = None
        wait_tensor_278 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_103);  reduce_scatter_tensor_103 = None
        view_602 = torch.ops.aten.view.default(add_124, [8192, 8192])
        permute_449 = torch.ops.aten.permute.default(view_602, [1, 0])
        permute_83 = torch.ops.aten.permute.default(getitem_67, [0, 2, 1, 3])
        view_191 = torch.ops.aten.view.default(permute_83, [1, 8192, 8192]);  permute_83 = None
        view_192 = torch.ops.aten.view.default(view_191, [8192, 8192]);  view_191 = None
        mm_233 = torch.ops.aten.mm.default(permute_449, view_192);  permute_449 = view_192 = None
        permute_450 = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        permute_84 = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        clone_30 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        all_gather_into_tensor_77 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_30, 8, '0');  clone_30 = None
        wait_tensor_84 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_77);  all_gather_into_tensor_77 = None
        permute_451 = torch.ops.aten.permute.default(wait_tensor_84, [1, 0]);  wait_tensor_84 = None
        mm_234 = torch.ops.aten.mm.default(view_602, permute_451);  view_602 = permute_451 = None
        view_603 = torch.ops.aten.view.default(mm_234, [1, 8192, 8192]);  mm_234 = None
        clone_97 = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        reduce_scatter_tensor_104 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_97, 'sum', 8, '0');  clone_97 = None
        wait_tensor_279 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_104);  reduce_scatter_tensor_104 = None
        permute_452 = torch.ops.aten.permute.default(wait_tensor_279, [1, 0]);  wait_tensor_279 = None
        convert_element_type_1006 = torch.ops.prims.convert_element_type.default(permute_452, torch.float32);  permute_452 = None
        view_604 = torch.ops.aten.view.default(view_603, [1, 8192, 32, 256]);  view_603 = None
        permute_453 = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
        view_174 = torch.ops.aten.view.default(mm_48, [2, 8192, 8192]);  mm_48 = None
        reduce_scatter_tensor_6 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_174, 'sum', 2, '3');  view_174 = None
        wait_tensor_79 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_6);  reduce_scatter_tensor_6 = None
        add_27 = torch.ops.aten.add.Tensor(add_25, wait_tensor_79);  wait_tensor_79 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_28 = torch.ops.aten.add.Scalar(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_14);  convert_element_type_233 = None
        all_gather_into_tensor_73 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_232, 8, '0');  convert_element_type_232 = None
        wait_tensor_80 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_73);  all_gather_into_tensor_73 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, wait_tensor_80)
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        all_gather_into_tensor_74 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_235, 8, '0');  convert_element_type_235 = None
        wait_tensor_81 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_74);  all_gather_into_tensor_74 = None
        permute_77 = torch.ops.aten.permute.default(wait_tensor_81, [1, 0]);  wait_tensor_81 = None
        view_175 = torch.ops.aten.view.default(convert_element_type_234, [8192, 8192]);  convert_element_type_234 = None
        mm_49 = torch.ops.aten.mm.default(view_175, permute_77)
        view_176 = torch.ops.aten.view.default(mm_49, [1, 8192, 8192]);  mm_49 = None
        view_178 = torch.ops.aten.view.default(mm_50, [1, 8192, 2048]);  mm_50 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        all_gather_into_tensor_76 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_241, 8, '0');  convert_element_type_241 = None
        wait_tensor_83 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_76);  all_gather_into_tensor_76 = None
        permute_79 = torch.ops.aten.permute.default(wait_tensor_83, [1, 0]);  wait_tensor_83 = None
        mm_51 = torch.ops.aten.mm.default(view_175, permute_79)
        view_180 = torch.ops.aten.view.default(mm_51, [1, 8192, 2048]);  mm_51 = None
        view_181 = torch.ops.aten.view.default(view_176, [1, 8192, 32, 256]);  view_176 = None
        view_182 = torch.ops.aten.view.default(view_178, [1, 8192, 8, 256]);  view_178 = None
        view_183 = torch.ops.aten.view.default(view_180, [1, 8192, 8, 256]);  view_180 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        view_184 = torch.ops.aten.view.default(convert_element_type_244, [1, 8192, 32, 128, 2]);  convert_element_type_244 = None
        view_as_complex_14 = torch.ops.aten.view_as_complex.default(view_184);  view_184 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        view_185 = torch.ops.aten.view.default(convert_element_type_245, [1, 8192, 8, 128, 2]);  convert_element_type_245 = None
        view_as_complex_15 = torch.ops.aten.view_as_complex.default(view_185);  view_185 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_as_complex_14, view_11);  view_as_complex_14 = None
        view_as_real_14 = torch.ops.aten.view_as_real.default(mul_58);  mul_58 = None
        view_187 = torch.ops.aten.view.default(view_as_real_14, [1, 8192, 32, 256]);  view_as_real_14 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_as_complex_15, view_11);  view_as_complex_15 = None
        view_as_real_15 = torch.ops.aten.view_as_real.default(mul_59);  mul_59 = None
        view_188 = torch.ops.aten.view.default(view_as_real_15, [1, 8192, 8, 256]);  view_as_real_15 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(view_188, torch.bfloat16);  view_188 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(convert_element_type_247, 3);  convert_element_type_247 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_14, [1, 8192, 8, 4, 256]);  unsqueeze_14 = None
        clone_28 = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_189 = torch.ops.aten.view.default(clone_28, [1, 8192, 32, 256]);  clone_28 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_183, 3);  view_183 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_15, [1, 8192, 8, 4, 256]);  unsqueeze_15 = None
        clone_29 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_190 = torch.ops.aten.view.default(clone_29, [1, 8192, 32, 256]);  clone_29 = None
        permute_80 = torch.ops.aten.permute.default(convert_element_type_246, [0, 2, 1, 3]);  convert_element_type_246 = None
        permute_81 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        permute_82 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_453, permute_80, permute_81, permute_82, getitem_67, getitem_68, None, None, 8192, 8192, 0.0, True, getitem_73, getitem_74, scale = 0.0625);  permute_453 = permute_80 = permute_81 = permute_82 = getitem_67 = getitem_68 = getitem_73 = getitem_74 = None
        getitem_172 = _scaled_dot_product_flash_attention_backward_8[0]
        getitem_173 = _scaled_dot_product_flash_attention_backward_8[1]
        getitem_174 = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
        permute_454 = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
        permute_455 = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
        permute_456 = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_605 = torch.ops.aten.view.default(permute_454, [1, 8192, 8, 4, 256]);  permute_454 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(view_605, [3], True);  view_605 = None
        squeeze_16 = torch.ops.aten.squeeze.dim(sum_53, 3);  sum_53 = None
        view_606 = torch.ops.aten.view.default(permute_455, [1, 8192, 8, 4, 256]);  permute_455 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(view_606, [3], True);  view_606 = None
        squeeze_17 = torch.ops.aten.squeeze.dim(sum_54, 3);  sum_54 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(squeeze_17, torch.float32);  squeeze_17 = None
        convert_element_type_1008 = torch.ops.prims.convert_element_type.default(permute_456, torch.float32);  permute_456 = None
        view_607 = torch.ops.aten.view.default(convert_element_type_1007, [1, 8192, 8, 128, 2]);  convert_element_type_1007 = None
        view_as_complex_48 = torch.ops.aten.view_as_complex.default(view_607);  view_607 = None
        mul_308 = torch.ops.aten.mul.Tensor(view_as_complex_48, _conj);  view_as_complex_48 = None
        view_608 = torch.ops.aten.view.default(convert_element_type_1008, [1, 8192, 32, 128, 2]);  convert_element_type_1008 = None
        view_as_complex_49 = torch.ops.aten.view_as_complex.default(view_608);  view_608 = None
        mul_309 = torch.ops.aten.mul.Tensor(view_as_complex_49, _conj);  view_as_complex_49 = None
        view_as_real_48 = torch.ops.aten.view_as_real.default(mul_308);  mul_308 = None
        view_609 = torch.ops.aten.view.default(view_as_real_48, [1, 8192, 8, 256]);  view_as_real_48 = None
        convert_element_type_1009 = torch.ops.prims.convert_element_type.default(view_609, torch.bfloat16);  view_609 = None
        view_as_real_49 = torch.ops.aten.view_as_real.default(mul_309);  mul_309 = None
        view_610 = torch.ops.aten.view.default(view_as_real_49, [1, 8192, 32, 256]);  view_as_real_49 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(view_610, torch.bfloat16);  view_610 = None
        view_611 = torch.ops.aten.view.default(squeeze_16, [1, 8192, 2048]);  squeeze_16 = None
        view_612 = torch.ops.aten.view.default(convert_element_type_1009, [1, 8192, 2048]);  convert_element_type_1009 = None
        view_613 = torch.ops.aten.view.default(convert_element_type_1010, [1, 8192, 8192]);  convert_element_type_1010 = None
        view_614 = torch.ops.aten.view.default(view_611, [8192, 2048]);  view_611 = None
        permute_457 = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_235 = torch.ops.aten.mm.default(permute_457, view_175);  permute_457 = None
        permute_459 = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_236 = torch.ops.aten.mm.default(view_614, permute_459);  view_614 = permute_459 = None
        view_615 = torch.ops.aten.view.default(mm_236, [1, 8192, 8192]);  mm_236 = None
        convert_element_type_1015 = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        reduce_scatter_tensor_105 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1015, 'sum', 8, '0');  convert_element_type_1015 = None
        wait_tensor_280 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_105);  reduce_scatter_tensor_105 = None
        view_616 = torch.ops.aten.view.default(view_612, [8192, 2048]);  view_612 = None
        permute_461 = torch.ops.aten.permute.default(view_616, [1, 0])
        mm_237 = torch.ops.aten.mm.default(permute_461, view_175);  permute_461 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        all_gather_into_tensor_75 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_238, 8, '0');  convert_element_type_238 = None
        wait_tensor_82 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_75);  all_gather_into_tensor_75 = None
        permute_78 = torch.ops.aten.permute.default(wait_tensor_82, [1, 0]);  wait_tensor_82 = None
        permute_463 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_238 = torch.ops.aten.mm.default(view_616, permute_463);  view_616 = permute_463 = None
        view_617 = torch.ops.aten.view.default(mm_238, [1, 8192, 8192]);  mm_238 = None
        add_125 = torch.ops.aten.add.Tensor(view_615, view_617);  view_615 = view_617 = None
        convert_element_type_1020 = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        reduce_scatter_tensor_106 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1020, 'sum', 8, '0');  convert_element_type_1020 = None
        wait_tensor_281 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_106);  reduce_scatter_tensor_106 = None
        view_618 = torch.ops.aten.view.default(view_613, [8192, 8192]);  view_613 = None
        permute_465 = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_239 = torch.ops.aten.mm.default(permute_465, view_175);  permute_465 = view_175 = None
        permute_467 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_240 = torch.ops.aten.mm.default(view_618, permute_467);  view_618 = permute_467 = None
        view_619 = torch.ops.aten.view.default(mm_240, [1, 8192, 8192]);  mm_240 = None
        add_126 = torch.ops.aten.add.Tensor(add_125, view_619);  add_125 = view_619 = None
        reduce_scatter_tensor_107 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_239, 'sum', 8, '0');  mm_239 = None
        wait_tensor_282 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_107);  reduce_scatter_tensor_107 = None
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(wait_tensor_282, torch.float32);  wait_tensor_282 = None
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        convert_element_type_1028 = torch.ops.prims.convert_element_type.default(wait_tensor_80, torch.float32);  wait_tensor_80 = None
        mul_310 = torch.ops.aten.mul.Tensor(convert_element_type_1026, convert_element_type_1028);  convert_element_type_1028 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_56, mul_310)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        div_18 = torch.ops.aten.div.Tensor(mul_56, 8192)
        mul_313 = torch.ops.aten.mul.Tensor(div_18, sum_55);  div_18 = sum_55 = None
        sub_27 = torch.ops.aten.sub.Tensor(mul_310, mul_313);  mul_310 = mul_313 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_14);  sub_27 = rsqrt_14 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_1026, mul_56);  convert_element_type_1026 = mul_56 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        convert_element_type_1029 = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16);  mul_314 = None
        add_127 = torch.ops.aten.add.Tensor(add_124, convert_element_type_1029);  add_124 = convert_element_type_1029 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(sum_56, torch.float32);  sum_56 = None
        reduce_scatter_tensor_108 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_15, 'sum', 8, '0');  convert_element_type_default_15 = None
        wait_tensor_283 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_108);  reduce_scatter_tensor_108 = None
        all_gather_into_tensor_175 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_127, 2, '3')
        wait_tensor_284 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_175);  all_gather_into_tensor_175 = None
        view_620 = torch.ops.aten.view.default(wait_tensor_284, [16384, 8192]);  wait_tensor_284 = None
        permute_469 = torch.ops.aten.permute.default(view_620, [1, 0])
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [2], True);  pow_14 = None
        add_26 = torch.ops.aten.add.Scalar(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_13);  convert_element_type_219 = None
        all_gather_into_tensor_68 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_218, 8, '0');  convert_element_type_218 = None
        wait_tensor_74 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_68);  all_gather_into_tensor_68 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, wait_tensor_74)
        convert_element_type_220 = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        all_gather_into_tensor_70 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_220, 2, '3');  convert_element_type_220 = None
        wait_tensor_76 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_70);  all_gather_into_tensor_70 = None
        view_169 = torch.ops.aten.view.default(wait_tensor_76, [16384, 8192]);  wait_tensor_76 = None
        view_170 = torch.ops.aten.view.default(mm_46, [2, 8192, 14336]);  mm_46 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_224)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_224, sigmoid_6);  sigmoid_6 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        all_gather_into_tensor_71 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_226, 4, '1');  convert_element_type_226 = None
        wait_tensor_77 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_71);  all_gather_into_tensor_71 = None
        permute_75 = torch.ops.aten.permute.default(wait_tensor_77, [1, 0]);  wait_tensor_77 = None
        mm_47 = torch.ops.aten.mm.default(view_169, permute_75)
        view_172 = torch.ops.aten.view.default(mm_47, [2, 8192, 14336]);  mm_47 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_225, view_172)
        view_173 = torch.ops.aten.view.default(mul_55, [16384, 14336]);  mul_55 = None
        mm_241 = torch.ops.aten.mm.default(permute_469, view_173);  permute_469 = view_173 = None
        permute_470 = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        permute_76 = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        clone_27 = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
        all_gather_into_tensor_72 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_27, 4, '1');  clone_27 = None
        wait_tensor_78 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_72);  all_gather_into_tensor_72 = None
        permute_471 = torch.ops.aten.permute.default(wait_tensor_78, [1, 0]);  wait_tensor_78 = None
        mm_242 = torch.ops.aten.mm.default(view_620, permute_471);  view_620 = permute_471 = None
        view_621 = torch.ops.aten.view.default(mm_242, [2, 8192, 14336]);  mm_242 = None
        clone_100 = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        reduce_scatter_tensor_109 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_100, 'sum', 4, '1');  clone_100 = None
        wait_tensor_285 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_109);  reduce_scatter_tensor_109 = None
        permute_472 = torch.ops.aten.permute.default(wait_tensor_285, [1, 0]);  wait_tensor_285 = None
        convert_element_type_1036 = torch.ops.prims.convert_element_type.default(permute_472, torch.float32);  permute_472 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_621, convert_element_type_225);  convert_element_type_225 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_621, view_172);  view_621 = view_172 = None
        view_622 = torch.ops.aten.view.default(mul_316, [16384, 14336]);  mul_316 = None
        permute_473 = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_243 = torch.ops.aten.mm.default(permute_473, view_169);  permute_473 = None
        reduce_scatter_tensor_110 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_243, 'sum', 4, '1');  mm_243 = None
        wait_tensor_286 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_110);  reduce_scatter_tensor_110 = None
        permute_475 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_244 = torch.ops.aten.mm.default(view_622, permute_475);  view_622 = permute_475 = None
        view_623 = torch.ops.aten.view.default(mm_244, [2, 8192, 8192]);  mm_244 = None
        convert_element_type_1041 = torch.ops.prims.convert_element_type.default(wait_tensor_286, torch.float32);  wait_tensor_286 = None
        convert_element_type_1042 = torch.ops.prims.convert_element_type.default(mul_317, torch.float32);  mul_317 = None
        neg_9 = torch.ops.aten.neg.default(convert_element_type_224)
        exp_9 = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_128 = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(add_128);  add_128 = None
        mul_318 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_319 = torch.ops.aten.mul.Tensor(convert_element_type_1042, mul_318);  convert_element_type_1042 = None
        sub_28 = torch.ops.aten.sub.Tensor(1, mul_318);  mul_318 = None
        mul_320 = torch.ops.aten.mul.Tensor(convert_element_type_224, sub_28);  convert_element_type_224 = sub_28 = None
        add_129 = torch.ops.aten.add.Tensor(mul_320, 1);  mul_320 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_319, add_129);  mul_319 = add_129 = None
        convert_element_type_1044 = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        view_624 = torch.ops.aten.view.default(convert_element_type_1044, [16384, 14336]);  convert_element_type_1044 = None
        permute_477 = torch.ops.aten.permute.default(view_624, [1, 0])
        mm_245 = torch.ops.aten.mm.default(permute_477, view_169);  permute_477 = view_169 = None
        reduce_scatter_tensor_111 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_245, 'sum', 4, '1');  mm_245 = None
        wait_tensor_287 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_111);  reduce_scatter_tensor_111 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        all_gather_into_tensor_69 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_221, 4, '1');  convert_element_type_221 = None
        wait_tensor_75 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_69);  all_gather_into_tensor_69 = None
        permute_74 = torch.ops.aten.permute.default(wait_tensor_75, [1, 0]);  wait_tensor_75 = None
        permute_479 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_246 = torch.ops.aten.mm.default(view_624, permute_479);  view_624 = permute_479 = None
        view_625 = torch.ops.aten.view.default(mm_246, [2, 8192, 8192]);  mm_246 = None
        add_130 = torch.ops.aten.add.Tensor(view_623, view_625);  view_623 = view_625 = None
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(wait_tensor_287, torch.float32);  wait_tensor_287 = None
        reduce_scatter_tensor_112 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_130, 'sum', 2, '3');  add_130 = None
        wait_tensor_288 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_112);  reduce_scatter_tensor_112 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(wait_tensor_288, torch.float32);  wait_tensor_288 = None
        convert_element_type_1052 = torch.ops.prims.convert_element_type.default(wait_tensor_74, torch.float32);  wait_tensor_74 = None
        mul_322 = torch.ops.aten.mul.Tensor(convert_element_type_1050, convert_element_type_1052);  convert_element_type_1052 = None
        mul_324 = torch.ops.aten.mul.Tensor(mul_52, mul_322)
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
        div_19 = torch.ops.aten.div.Tensor(mul_52, 8192)
        mul_325 = torch.ops.aten.mul.Tensor(div_19, sum_57);  div_19 = sum_57 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_322, mul_325);  mul_322 = mul_325 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_13);  sub_29 = rsqrt_13 = None
        mul_327 = torch.ops.aten.mul.Tensor(convert_element_type_1050, mul_52);  convert_element_type_1050 = mul_52 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1]);  mul_327 = None
        convert_element_type_1053 = torch.ops.prims.convert_element_type.default(mul_326, torch.bfloat16);  mul_326 = None
        add_131 = torch.ops.aten.add.Tensor(add_127, convert_element_type_1053);  add_127 = convert_element_type_1053 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(sum_58, torch.float32);  sum_58 = None
        reduce_scatter_tensor_113 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_14, 'sum', 8, '0');  convert_element_type_default_14 = None
        wait_tensor_289 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_113);  reduce_scatter_tensor_113 = None
        view_626 = torch.ops.aten.view.default(add_131, [8192, 8192])
        permute_481 = torch.ops.aten.permute.default(view_626, [1, 0])
        permute_72 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3])
        view_166 = torch.ops.aten.view.default(permute_72, [1, 8192, 8192]);  permute_72 = None
        view_167 = torch.ops.aten.view.default(view_166, [8192, 8192]);  view_166 = None
        mm_247 = torch.ops.aten.mm.default(permute_481, view_167);  permute_481 = view_167 = None
        permute_482 = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        permute_73 = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        clone_26 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        all_gather_into_tensor_67 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_26, 8, '0');  clone_26 = None
        wait_tensor_73 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_67);  all_gather_into_tensor_67 = None
        permute_483 = torch.ops.aten.permute.default(wait_tensor_73, [1, 0]);  wait_tensor_73 = None
        mm_248 = torch.ops.aten.mm.default(view_626, permute_483);  view_626 = permute_483 = None
        view_627 = torch.ops.aten.view.default(mm_248, [1, 8192, 8192]);  mm_248 = None
        clone_101 = torch.ops.aten.clone.default(permute_482, memory_format = torch.contiguous_format);  permute_482 = None
        reduce_scatter_tensor_114 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_101, 'sum', 8, '0');  clone_101 = None
        wait_tensor_290 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_114);  reduce_scatter_tensor_114 = None
        permute_484 = torch.ops.aten.permute.default(wait_tensor_290, [1, 0]);  wait_tensor_290 = None
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(permute_484, torch.float32);  permute_484 = None
        view_628 = torch.ops.aten.view.default(view_627, [1, 8192, 32, 256]);  view_627 = None
        permute_485 = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_22 = torch.ops.aten.add.Scalar(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_186, rsqrt_11);  convert_element_type_186 = None
        all_gather_into_tensor_58 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_185, 8, '0');  convert_element_type_185 = None
        wait_tensor_63 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_58);  all_gather_into_tensor_58 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, wait_tensor_63)
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        all_gather_into_tensor_59 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_188, 4, '1');  convert_element_type_188 = None
        wait_tensor_64 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_59);  all_gather_into_tensor_59 = None
        permute_63 = torch.ops.aten.permute.default(wait_tensor_64, [1, 0]);  wait_tensor_64 = None
        all_gather_into_tensor_60 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_187, 2, '3');  convert_element_type_187 = None
        wait_tensor_65 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_60);  all_gather_into_tensor_60 = None
        view_144 = torch.ops.aten.view.default(wait_tensor_65, [16384, 8192]);  wait_tensor_65 = None
        mm_39 = torch.ops.aten.mm.default(view_144, permute_63)
        view_145 = torch.ops.aten.view.default(mm_39, [2, 8192, 14336]);  mm_39 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_191)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_191, sigmoid_5);  sigmoid_5 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        view_147 = torch.ops.aten.view.default(mm_40, [2, 8192, 14336]);  mm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_192, view_147)
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        permute_65 = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        view_148 = torch.ops.aten.view.default(mul_47, [16384, 14336]);  mul_47 = None
        clone_23 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        all_gather_into_tensor_62 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_23, 4, '1');  clone_23 = None
        wait_tensor_67 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_62);  all_gather_into_tensor_62 = None
        mm_41 = torch.ops.aten.mm.default(view_148, wait_tensor_67)
        view_149 = torch.ops.aten.view.default(mm_41, [2, 8192, 8192]);  mm_41 = None
        reduce_scatter_tensor_5 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_149, 'sum', 2, '3');  view_149 = None
        wait_tensor_68 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_5);  reduce_scatter_tensor_5 = None
        add_23 = torch.ops.aten.add.Tensor(add_21, wait_tensor_68);  add_21 = wait_tensor_68 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [2], True);  pow_13 = None
        add_24 = torch.ops.aten.add.Scalar(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_12);  convert_element_type_200 = None
        all_gather_into_tensor_63 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_199, 8, '0');  convert_element_type_199 = None
        wait_tensor_69 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_63);  all_gather_into_tensor_63 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, wait_tensor_69)
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_150 = torch.ops.aten.view.default(convert_element_type_201, [8192, 8192]);  convert_element_type_201 = None
        view_151 = torch.ops.aten.view.default(mm_42, [1, 8192, 8192]);  mm_42 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        all_gather_into_tensor_65 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_205, 8, '0');  convert_element_type_205 = None
        wait_tensor_71 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_65);  all_gather_into_tensor_65 = None
        permute_67 = torch.ops.aten.permute.default(wait_tensor_71, [1, 0]);  wait_tensor_71 = None
        mm_43 = torch.ops.aten.mm.default(view_150, permute_67)
        view_153 = torch.ops.aten.view.default(mm_43, [1, 8192, 2048]);  mm_43 = None
        view_155 = torch.ops.aten.view.default(mm_44, [1, 8192, 2048]);  mm_44 = None
        view_156 = torch.ops.aten.view.default(view_151, [1, 8192, 32, 256]);  view_151 = None
        view_157 = torch.ops.aten.view.default(view_153, [1, 8192, 8, 256]);  view_153 = None
        view_158 = torch.ops.aten.view.default(view_155, [1, 8192, 8, 256]);  view_155 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(view_156, torch.float32);  view_156 = None
        view_159 = torch.ops.aten.view.default(convert_element_type_211, [1, 8192, 32, 128, 2]);  convert_element_type_211 = None
        view_as_complex_12 = torch.ops.aten.view_as_complex.default(view_159);  view_159 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        view_160 = torch.ops.aten.view.default(convert_element_type_212, [1, 8192, 8, 128, 2]);  convert_element_type_212 = None
        view_as_complex_13 = torch.ops.aten.view_as_complex.default(view_160);  view_160 = None
        mul_50 = torch.ops.aten.mul.Tensor(view_as_complex_12, view_11);  view_as_complex_12 = None
        view_as_real_12 = torch.ops.aten.view_as_real.default(mul_50);  mul_50 = None
        view_162 = torch.ops.aten.view.default(view_as_real_12, [1, 8192, 32, 256]);  view_as_real_12 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_as_complex_13, view_11);  view_as_complex_13 = None
        view_as_real_13 = torch.ops.aten.view_as_real.default(mul_51);  mul_51 = None
        view_163 = torch.ops.aten.view.default(view_as_real_13, [1, 8192, 8, 256]);  view_as_real_13 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(view_162, torch.bfloat16);  view_162 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(view_163, torch.bfloat16);  view_163 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(convert_element_type_214, 3);  convert_element_type_214 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_12, [1, 8192, 8, 4, 256]);  unsqueeze_12 = None
        clone_24 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_164 = torch.ops.aten.view.default(clone_24, [1, 8192, 32, 256]);  clone_24 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_158, 3);  view_158 = None
        expand_13 = torch.ops.aten.expand.default(unsqueeze_13, [1, 8192, 8, 4, 256]);  unsqueeze_13 = None
        clone_25 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_165 = torch.ops.aten.view.default(clone_25, [1, 8192, 32, 256]);  clone_25 = None
        permute_69 = torch.ops.aten.permute.default(convert_element_type_213, [0, 2, 1, 3]);  convert_element_type_213 = None
        permute_70 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        permute_71 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_485, permute_69, permute_70, permute_71, getitem_58, getitem_59, None, None, 8192, 8192, 0.0, True, getitem_64, getitem_65, scale = 0.0625);  permute_485 = permute_69 = permute_70 = permute_71 = getitem_58 = getitem_59 = getitem_64 = getitem_65 = None
        getitem_175 = _scaled_dot_product_flash_attention_backward_9[0]
        getitem_176 = _scaled_dot_product_flash_attention_backward_9[1]
        getitem_177 = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
        permute_486 = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3]);  getitem_177 = None
        permute_487 = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        permute_488 = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
        view_629 = torch.ops.aten.view.default(permute_486, [1, 8192, 8, 4, 256]);  permute_486 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(view_629, [3], True);  view_629 = None
        squeeze_18 = torch.ops.aten.squeeze.dim(sum_59, 3);  sum_59 = None
        view_630 = torch.ops.aten.view.default(permute_487, [1, 8192, 8, 4, 256]);  permute_487 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(view_630, [3], True);  view_630 = None
        squeeze_19 = torch.ops.aten.squeeze.dim(sum_60, 3);  sum_60 = None
        convert_element_type_1061 = torch.ops.prims.convert_element_type.default(squeeze_19, torch.float32);  squeeze_19 = None
        convert_element_type_1062 = torch.ops.prims.convert_element_type.default(permute_488, torch.float32);  permute_488 = None
        view_631 = torch.ops.aten.view.default(convert_element_type_1061, [1, 8192, 8, 128, 2]);  convert_element_type_1061 = None
        view_as_complex_50 = torch.ops.aten.view_as_complex.default(view_631);  view_631 = None
        mul_328 = torch.ops.aten.mul.Tensor(view_as_complex_50, _conj);  view_as_complex_50 = None
        view_632 = torch.ops.aten.view.default(convert_element_type_1062, [1, 8192, 32, 128, 2]);  convert_element_type_1062 = None
        view_as_complex_51 = torch.ops.aten.view_as_complex.default(view_632);  view_632 = None
        mul_329 = torch.ops.aten.mul.Tensor(view_as_complex_51, _conj);  view_as_complex_51 = None
        view_as_real_50 = torch.ops.aten.view_as_real.default(mul_328);  mul_328 = None
        view_633 = torch.ops.aten.view.default(view_as_real_50, [1, 8192, 8, 256]);  view_as_real_50 = None
        convert_element_type_1063 = torch.ops.prims.convert_element_type.default(view_633, torch.bfloat16);  view_633 = None
        view_as_real_51 = torch.ops.aten.view_as_real.default(mul_329);  mul_329 = None
        view_634 = torch.ops.aten.view.default(view_as_real_51, [1, 8192, 32, 256]);  view_as_real_51 = None
        convert_element_type_1064 = torch.ops.prims.convert_element_type.default(view_634, torch.bfloat16);  view_634 = None
        view_635 = torch.ops.aten.view.default(squeeze_18, [1, 8192, 2048]);  squeeze_18 = None
        view_636 = torch.ops.aten.view.default(convert_element_type_1063, [1, 8192, 2048]);  convert_element_type_1063 = None
        view_637 = torch.ops.aten.view.default(convert_element_type_1064, [1, 8192, 8192]);  convert_element_type_1064 = None
        view_638 = torch.ops.aten.view.default(view_635, [8192, 2048]);  view_635 = None
        permute_489 = torch.ops.aten.permute.default(view_638, [1, 0])
        mm_249 = torch.ops.aten.mm.default(permute_489, view_150);  permute_489 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        all_gather_into_tensor_66 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_208, 8, '0');  convert_element_type_208 = None
        wait_tensor_72 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_66);  all_gather_into_tensor_66 = None
        permute_68 = torch.ops.aten.permute.default(wait_tensor_72, [1, 0]);  wait_tensor_72 = None
        permute_491 = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_250 = torch.ops.aten.mm.default(view_638, permute_491);  view_638 = permute_491 = None
        view_639 = torch.ops.aten.view.default(mm_250, [1, 8192, 8192]);  mm_250 = None
        convert_element_type_1069 = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        reduce_scatter_tensor_115 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1069, 'sum', 8, '0');  convert_element_type_1069 = None
        wait_tensor_291 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_115);  reduce_scatter_tensor_115 = None
        view_640 = torch.ops.aten.view.default(view_636, [8192, 2048]);  view_636 = None
        permute_493 = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_251 = torch.ops.aten.mm.default(permute_493, view_150);  permute_493 = None
        permute_495 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_252 = torch.ops.aten.mm.default(view_640, permute_495);  view_640 = permute_495 = None
        view_641 = torch.ops.aten.view.default(mm_252, [1, 8192, 8192]);  mm_252 = None
        add_132 = torch.ops.aten.add.Tensor(view_639, view_641);  view_639 = view_641 = None
        convert_element_type_1074 = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        reduce_scatter_tensor_116 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1074, 'sum', 8, '0');  convert_element_type_1074 = None
        wait_tensor_292 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_116);  reduce_scatter_tensor_116 = None
        view_642 = torch.ops.aten.view.default(view_637, [8192, 8192]);  view_637 = None
        permute_497 = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_253 = torch.ops.aten.mm.default(permute_497, view_150);  permute_497 = view_150 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        all_gather_into_tensor_64 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_202, 8, '0');  convert_element_type_202 = None
        wait_tensor_70 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_64);  all_gather_into_tensor_64 = None
        permute_66 = torch.ops.aten.permute.default(wait_tensor_70, [1, 0]);  wait_tensor_70 = None
        permute_499 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_254 = torch.ops.aten.mm.default(view_642, permute_499);  view_642 = permute_499 = None
        view_643 = torch.ops.aten.view.default(mm_254, [1, 8192, 8192]);  mm_254 = None
        add_133 = torch.ops.aten.add.Tensor(add_132, view_643);  add_132 = view_643 = None
        reduce_scatter_tensor_117 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_253, 'sum', 8, '0');  mm_253 = None
        wait_tensor_293 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_117);  reduce_scatter_tensor_117 = None
        convert_element_type_1079 = torch.ops.prims.convert_element_type.default(wait_tensor_293, torch.float32);  wait_tensor_293 = None
        convert_element_type_1080 = torch.ops.prims.convert_element_type.default(add_133, torch.float32);  add_133 = None
        convert_element_type_1082 = torch.ops.prims.convert_element_type.default(wait_tensor_69, torch.float32);  wait_tensor_69 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_1080, convert_element_type_1082);  convert_element_type_1082 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_48, mul_330)
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        div_20 = torch.ops.aten.div.Tensor(mul_48, 8192)
        mul_333 = torch.ops.aten.mul.Tensor(div_20, sum_61);  div_20 = sum_61 = None
        sub_30 = torch.ops.aten.sub.Tensor(mul_330, mul_333);  mul_330 = mul_333 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_12);  sub_30 = rsqrt_12 = None
        mul_335 = torch.ops.aten.mul.Tensor(convert_element_type_1080, mul_48);  convert_element_type_1080 = mul_48 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        convert_element_type_1083 = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        add_134 = torch.ops.aten.add.Tensor(add_131, convert_element_type_1083);  add_131 = convert_element_type_1083 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(sum_62, torch.float32);  sum_62 = None
        reduce_scatter_tensor_118 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_13, 'sum', 8, '0');  convert_element_type_default_13 = None
        wait_tensor_294 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_118);  reduce_scatter_tensor_118 = None
        all_gather_into_tensor_176 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_134, 2, '3')
        wait_tensor_295 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_176);  all_gather_into_tensor_176 = None
        view_644 = torch.ops.aten.view.default(wait_tensor_295, [16384, 8192]);  wait_tensor_295 = None
        permute_501 = torch.ops.aten.permute.default(view_644, [1, 0])
        mm_255 = torch.ops.aten.mm.default(permute_501, view_148);  permute_501 = view_148 = None
        permute_502 = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
        permute_503 = torch.ops.aten.permute.default(wait_tensor_67, [1, 0]);  wait_tensor_67 = None
        mm_256 = torch.ops.aten.mm.default(view_644, permute_503);  view_644 = permute_503 = None
        view_645 = torch.ops.aten.view.default(mm_256, [2, 8192, 14336]);  mm_256 = None
        clone_104 = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        reduce_scatter_tensor_119 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_104, 'sum', 4, '1');  clone_104 = None
        wait_tensor_296 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_119);  reduce_scatter_tensor_119 = None
        permute_504 = torch.ops.aten.permute.default(wait_tensor_296, [1, 0]);  wait_tensor_296 = None
        convert_element_type_1090 = torch.ops.prims.convert_element_type.default(permute_504, torch.float32);  permute_504 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_645, convert_element_type_192);  convert_element_type_192 = None
        mul_337 = torch.ops.aten.mul.Tensor(view_645, view_147);  view_645 = view_147 = None
        view_646 = torch.ops.aten.view.default(mul_336, [16384, 14336]);  mul_336 = None
        permute_505 = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_257 = torch.ops.aten.mm.default(permute_505, view_144);  permute_505 = None
        reduce_scatter_tensor_120 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_257, 'sum', 4, '1');  mm_257 = None
        wait_tensor_297 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_120);  reduce_scatter_tensor_120 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        all_gather_into_tensor_61 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_193, 4, '1');  convert_element_type_193 = None
        wait_tensor_66 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_61);  all_gather_into_tensor_61 = None
        permute_64 = torch.ops.aten.permute.default(wait_tensor_66, [1, 0]);  wait_tensor_66 = None
        permute_507 = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_258 = torch.ops.aten.mm.default(view_646, permute_507);  view_646 = permute_507 = None
        view_647 = torch.ops.aten.view.default(mm_258, [2, 8192, 8192]);  mm_258 = None
        convert_element_type_1095 = torch.ops.prims.convert_element_type.default(wait_tensor_297, torch.float32);  wait_tensor_297 = None
        convert_element_type_1096 = torch.ops.prims.convert_element_type.default(mul_337, torch.float32);  mul_337 = None
        neg_10 = torch.ops.aten.neg.default(convert_element_type_191)
        exp_10 = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_135 = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(add_135);  add_135 = None
        mul_338 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_339 = torch.ops.aten.mul.Tensor(convert_element_type_1096, mul_338);  convert_element_type_1096 = None
        sub_31 = torch.ops.aten.sub.Tensor(1, mul_338);  mul_338 = None
        mul_340 = torch.ops.aten.mul.Tensor(convert_element_type_191, sub_31);  convert_element_type_191 = sub_31 = None
        add_136 = torch.ops.aten.add.Tensor(mul_340, 1);  mul_340 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_339, add_136);  mul_339 = add_136 = None
        convert_element_type_1098 = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16);  mul_341 = None
        view_648 = torch.ops.aten.view.default(convert_element_type_1098, [16384, 14336]);  convert_element_type_1098 = None
        permute_509 = torch.ops.aten.permute.default(view_648, [1, 0])
        mm_259 = torch.ops.aten.mm.default(permute_509, view_144);  permute_509 = view_144 = None
        reduce_scatter_tensor_121 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_259, 'sum', 4, '1');  mm_259 = None
        wait_tensor_298 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_121);  reduce_scatter_tensor_121 = None
        permute_511 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_260 = torch.ops.aten.mm.default(view_648, permute_511);  view_648 = permute_511 = None
        view_649 = torch.ops.aten.view.default(mm_260, [2, 8192, 8192]);  mm_260 = None
        add_137 = torch.ops.aten.add.Tensor(view_647, view_649);  view_647 = view_649 = None
        convert_element_type_1103 = torch.ops.prims.convert_element_type.default(wait_tensor_298, torch.float32);  wait_tensor_298 = None
        reduce_scatter_tensor_122 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_137, 'sum', 2, '3');  add_137 = None
        wait_tensor_299 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_122);  reduce_scatter_tensor_122 = None
        convert_element_type_1104 = torch.ops.prims.convert_element_type.default(wait_tensor_299, torch.float32);  wait_tensor_299 = None
        convert_element_type_1106 = torch.ops.prims.convert_element_type.default(wait_tensor_63, torch.float32);  wait_tensor_63 = None
        mul_342 = torch.ops.aten.mul.Tensor(convert_element_type_1104, convert_element_type_1106);  convert_element_type_1106 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_44, mul_342)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        div_21 = torch.ops.aten.div.Tensor(mul_44, 8192)
        mul_345 = torch.ops.aten.mul.Tensor(div_21, sum_63);  div_21 = sum_63 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_342, mul_345);  mul_342 = mul_345 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_11);  sub_32 = rsqrt_11 = None
        mul_347 = torch.ops.aten.mul.Tensor(convert_element_type_1104, mul_44);  convert_element_type_1104 = mul_44 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        convert_element_type_1107 = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        add_138 = torch.ops.aten.add.Tensor(add_134, convert_element_type_1107);  add_134 = convert_element_type_1107 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(sum_64, torch.float32);  sum_64 = None
        reduce_scatter_tensor_123 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_12, 'sum', 8, '0');  convert_element_type_default_12 = None
        wait_tensor_300 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_123);  reduce_scatter_tensor_123 = None
        view_650 = torch.ops.aten.view.default(add_138, [8192, 8192])
        permute_513 = torch.ops.aten.permute.default(view_650, [1, 0])
        permute_61 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_141 = torch.ops.aten.view.default(permute_61, [1, 8192, 8192]);  permute_61 = None
        view_142 = torch.ops.aten.view.default(view_141, [8192, 8192]);  view_141 = None
        mm_261 = torch.ops.aten.mm.default(permute_513, view_142);  permute_513 = view_142 = None
        permute_514 = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_62 = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        clone_22 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        all_gather_into_tensor_57 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_22, 8, '0');  clone_22 = None
        wait_tensor_62 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_57);  all_gather_into_tensor_57 = None
        permute_515 = torch.ops.aten.permute.default(wait_tensor_62, [1, 0]);  wait_tensor_62 = None
        mm_262 = torch.ops.aten.mm.default(view_650, permute_515);  view_650 = permute_515 = None
        view_651 = torch.ops.aten.view.default(mm_262, [1, 8192, 8192]);  mm_262 = None
        clone_105 = torch.ops.aten.clone.default(permute_514, memory_format = torch.contiguous_format);  permute_514 = None
        reduce_scatter_tensor_124 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_105, 'sum', 8, '0');  clone_105 = None
        wait_tensor_301 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_124);  reduce_scatter_tensor_124 = None
        permute_516 = torch.ops.aten.permute.default(wait_tensor_301, [1, 0]);  wait_tensor_301 = None
        convert_element_type_1114 = torch.ops.prims.convert_element_type.default(permute_516, torch.float32);  permute_516 = None
        view_652 = torch.ops.aten.view.default(view_651, [1, 8192, 32, 256]);  view_651 = None
        permute_517 = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
        permute_50 = torch.ops.aten.permute.default(getitem_40, [0, 2, 1, 3])
        view_116 = torch.ops.aten.view.default(permute_50, [1, 8192, 8192]);  permute_50 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_51 = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        view_117 = torch.ops.aten.view.default(view_116, [8192, 8192]);  view_116 = None
        clone_18 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        all_gather_into_tensor_47 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_18, 8, '0');  clone_18 = None
        wait_tensor_51 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_47);  all_gather_into_tensor_47 = None
        mm_31 = torch.ops.aten.mm.default(view_117, wait_tensor_51)
        view_118 = torch.ops.aten.view.default(mm_31, [1, 8192, 8192]);  mm_31 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, view_118);  view_118 = None
        view_124 = torch.ops.aten.view.default(mm_34, [2, 8192, 8192]);  mm_34 = None
        reduce_scatter_tensor_4 = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_124, 'sum', 2, '3');  view_124 = None
        wait_tensor_57 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, wait_tensor_57);  wait_tensor_57 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
        add_20 = torch.ops.aten.add.Scalar(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_167, rsqrt_10);  convert_element_type_167 = None
        all_gather_into_tensor_53 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_166, 8, '0');  convert_element_type_166 = None
        wait_tensor_58 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_53);  all_gather_into_tensor_53 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, wait_tensor_58)
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        all_gather_into_tensor_54 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_169, 8, '0');  convert_element_type_169 = None
        wait_tensor_59 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_54);  all_gather_into_tensor_54 = None
        permute_55 = torch.ops.aten.permute.default(wait_tensor_59, [1, 0]);  wait_tensor_59 = None
        view_125 = torch.ops.aten.view.default(convert_element_type_168, [8192, 8192]);  convert_element_type_168 = None
        mm_35 = torch.ops.aten.mm.default(view_125, permute_55)
        view_126 = torch.ops.aten.view.default(mm_35, [1, 8192, 8192]);  mm_35 = None
        view_128 = torch.ops.aten.view.default(mm_36, [1, 8192, 2048]);  mm_36 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        all_gather_into_tensor_56 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_175, 8, '0');  convert_element_type_175 = None
        wait_tensor_61 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_56);  all_gather_into_tensor_56 = None
        permute_57 = torch.ops.aten.permute.default(wait_tensor_61, [1, 0]);  wait_tensor_61 = None
        mm_37 = torch.ops.aten.mm.default(view_125, permute_57)
        view_130 = torch.ops.aten.view.default(mm_37, [1, 8192, 2048]);  mm_37 = None
        view_131 = torch.ops.aten.view.default(view_126, [1, 8192, 32, 256]);  view_126 = None
        view_132 = torch.ops.aten.view.default(view_128, [1, 8192, 8, 256]);  view_128 = None
        view_133 = torch.ops.aten.view.default(view_130, [1, 8192, 8, 256]);  view_130 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        view_134 = torch.ops.aten.view.default(convert_element_type_178, [1, 8192, 32, 128, 2]);  convert_element_type_178 = None
        view_as_complex_10 = torch.ops.aten.view_as_complex.default(view_134);  view_134 = None
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(view_132, torch.float32);  view_132 = None
        view_135 = torch.ops.aten.view.default(convert_element_type_179, [1, 8192, 8, 128, 2]);  convert_element_type_179 = None
        view_as_complex_11 = torch.ops.aten.view_as_complex.default(view_135);  view_135 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_as_complex_10, view_11);  view_as_complex_10 = None
        view_as_real_10 = torch.ops.aten.view_as_real.default(mul_42);  mul_42 = None
        view_137 = torch.ops.aten.view.default(view_as_real_10, [1, 8192, 32, 256]);  view_as_real_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_as_complex_11, view_11);  view_as_complex_11 = None
        view_as_real_11 = torch.ops.aten.view_as_real.default(mul_43);  mul_43 = None
        view_138 = torch.ops.aten.view.default(view_as_real_11, [1, 8192, 8, 256]);  view_as_real_11 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(view_137, torch.bfloat16);  view_137 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(view_138, torch.bfloat16);  view_138 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3);  convert_element_type_181 = None
        expand_10 = torch.ops.aten.expand.default(unsqueeze_10, [1, 8192, 8, 4, 256]);  unsqueeze_10 = None
        clone_20 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_139 = torch.ops.aten.view.default(clone_20, [1, 8192, 32, 256]);  clone_20 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(view_133, 3);  view_133 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_11, [1, 8192, 8, 4, 256]);  unsqueeze_11 = None
        clone_21 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_140 = torch.ops.aten.view.default(clone_21, [1, 8192, 32, 256]);  clone_21 = None
        permute_58 = torch.ops.aten.permute.default(convert_element_type_180, [0, 2, 1, 3]);  convert_element_type_180 = None
        permute_59 = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        permute_60 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_517, permute_58, permute_59, permute_60, getitem_49, getitem_50, None, None, 8192, 8192, 0.0, True, getitem_55, getitem_56, scale = 0.0625);  permute_517 = permute_58 = permute_59 = permute_60 = getitem_49 = getitem_50 = getitem_55 = getitem_56 = None
        getitem_178 = _scaled_dot_product_flash_attention_backward_10[0]
        getitem_179 = _scaled_dot_product_flash_attention_backward_10[1]
        getitem_180 = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
        permute_518 = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        permute_519 = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
        permute_520 = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        view_653 = torch.ops.aten.view.default(permute_518, [1, 8192, 8, 4, 256]);  permute_518 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(view_653, [3], True);  view_653 = None
        squeeze_20 = torch.ops.aten.squeeze.dim(sum_65, 3);  sum_65 = None
        view_654 = torch.ops.aten.view.default(permute_519, [1, 8192, 8, 4, 256]);  permute_519 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(view_654, [3], True);  view_654 = None
        squeeze_21 = torch.ops.aten.squeeze.dim(sum_66, 3);  sum_66 = None
        convert_element_type_1115 = torch.ops.prims.convert_element_type.default(squeeze_21, torch.float32);  squeeze_21 = None
        convert_element_type_1116 = torch.ops.prims.convert_element_type.default(permute_520, torch.float32);  permute_520 = None
        view_655 = torch.ops.aten.view.default(convert_element_type_1115, [1, 8192, 8, 128, 2]);  convert_element_type_1115 = None
        view_as_complex_52 = torch.ops.aten.view_as_complex.default(view_655);  view_655 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_as_complex_52, _conj);  view_as_complex_52 = None
        view_656 = torch.ops.aten.view.default(convert_element_type_1116, [1, 8192, 32, 128, 2]);  convert_element_type_1116 = None
        view_as_complex_53 = torch.ops.aten.view_as_complex.default(view_656);  view_656 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_as_complex_53, _conj);  view_as_complex_53 = None
        view_as_real_52 = torch.ops.aten.view_as_real.default(mul_348);  mul_348 = None
        view_657 = torch.ops.aten.view.default(view_as_real_52, [1, 8192, 8, 256]);  view_as_real_52 = None
        convert_element_type_1117 = torch.ops.prims.convert_element_type.default(view_657, torch.bfloat16);  view_657 = None
        view_as_real_53 = torch.ops.aten.view_as_real.default(mul_349);  mul_349 = None
        view_658 = torch.ops.aten.view.default(view_as_real_53, [1, 8192, 32, 256]);  view_as_real_53 = None
        convert_element_type_1118 = torch.ops.prims.convert_element_type.default(view_658, torch.bfloat16);  view_658 = None
        view_659 = torch.ops.aten.view.default(squeeze_20, [1, 8192, 2048]);  squeeze_20 = None
        view_660 = torch.ops.aten.view.default(convert_element_type_1117, [1, 8192, 2048]);  convert_element_type_1117 = None
        view_661 = torch.ops.aten.view.default(convert_element_type_1118, [1, 8192, 8192]);  convert_element_type_1118 = None
        view_662 = torch.ops.aten.view.default(view_659, [8192, 2048]);  view_659 = None
        permute_521 = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_263 = torch.ops.aten.mm.default(permute_521, view_125);  permute_521 = None
        permute_523 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_264 = torch.ops.aten.mm.default(view_662, permute_523);  view_662 = permute_523 = None
        view_663 = torch.ops.aten.view.default(mm_264, [1, 8192, 8192]);  mm_264 = None
        convert_element_type_1123 = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        reduce_scatter_tensor_125 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1123, 'sum', 8, '0');  convert_element_type_1123 = None
        wait_tensor_302 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_125);  reduce_scatter_tensor_125 = None
        view_664 = torch.ops.aten.view.default(view_660, [8192, 2048]);  view_660 = None
        permute_525 = torch.ops.aten.permute.default(view_664, [1, 0])
        mm_265 = torch.ops.aten.mm.default(permute_525, view_125);  permute_525 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        all_gather_into_tensor_55 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_172, 8, '0');  convert_element_type_172 = None
        wait_tensor_60 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_55);  all_gather_into_tensor_55 = None
        permute_56 = torch.ops.aten.permute.default(wait_tensor_60, [1, 0]);  wait_tensor_60 = None
        permute_527 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_266 = torch.ops.aten.mm.default(view_664, permute_527);  view_664 = permute_527 = None
        view_665 = torch.ops.aten.view.default(mm_266, [1, 8192, 8192]);  mm_266 = None
        add_139 = torch.ops.aten.add.Tensor(view_663, view_665);  view_663 = view_665 = None
        convert_element_type_1128 = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        reduce_scatter_tensor_126 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1128, 'sum', 8, '0');  convert_element_type_1128 = None
        wait_tensor_303 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_126);  reduce_scatter_tensor_126 = None
        view_666 = torch.ops.aten.view.default(view_661, [8192, 8192]);  view_661 = None
        permute_529 = torch.ops.aten.permute.default(view_666, [1, 0])
        mm_267 = torch.ops.aten.mm.default(permute_529, view_125);  permute_529 = view_125 = None
        permute_531 = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_268 = torch.ops.aten.mm.default(view_666, permute_531);  view_666 = permute_531 = None
        view_667 = torch.ops.aten.view.default(mm_268, [1, 8192, 8192]);  mm_268 = None
        add_140 = torch.ops.aten.add.Tensor(add_139, view_667);  add_139 = view_667 = None
        reduce_scatter_tensor_127 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_267, 'sum', 8, '0');  mm_267 = None
        wait_tensor_304 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_127);  reduce_scatter_tensor_127 = None
        convert_element_type_1133 = torch.ops.prims.convert_element_type.default(wait_tensor_304, torch.float32);  wait_tensor_304 = None
        convert_element_type_1134 = torch.ops.prims.convert_element_type.default(add_140, torch.float32);  add_140 = None
        convert_element_type_1136 = torch.ops.prims.convert_element_type.default(wait_tensor_58, torch.float32);  wait_tensor_58 = None
        mul_350 = torch.ops.aten.mul.Tensor(convert_element_type_1134, convert_element_type_1136);  convert_element_type_1136 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_40, mul_350)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        div_22 = torch.ops.aten.div.Tensor(mul_40, 8192)
        mul_353 = torch.ops.aten.mul.Tensor(div_22, sum_67);  div_22 = sum_67 = None
        sub_33 = torch.ops.aten.sub.Tensor(mul_350, mul_353);  mul_350 = mul_353 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_10);  sub_33 = rsqrt_10 = None
        mul_355 = torch.ops.aten.mul.Tensor(convert_element_type_1134, mul_40);  convert_element_type_1134 = mul_40 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        convert_element_type_1137 = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        add_141 = torch.ops.aten.add.Tensor(add_138, convert_element_type_1137);  add_138 = convert_element_type_1137 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(sum_68, torch.float32);  sum_68 = None
        reduce_scatter_tensor_128 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_11, 'sum', 8, '0');  convert_element_type_default_11 = None
        wait_tensor_305 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_128);  reduce_scatter_tensor_128 = None
        all_gather_into_tensor_177 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_141, 2, '3')
        wait_tensor_306 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_177);  all_gather_into_tensor_177 = None
        view_668 = torch.ops.aten.view.default(wait_tensor_306, [16384, 8192]);  wait_tensor_306 = None
        permute_533 = torch.ops.aten.permute.default(view_668, [1, 0])
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_18 = torch.ops.aten.add.Scalar(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_9);  convert_element_type_153 = None
        all_gather_into_tensor_48 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_152, 8, '0');  convert_element_type_152 = None
        wait_tensor_52 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_48);  all_gather_into_tensor_48 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, wait_tensor_52)
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        all_gather_into_tensor_50 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_154, 2, '3');  convert_element_type_154 = None
        wait_tensor_54 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_50);  all_gather_into_tensor_50 = None
        view_119 = torch.ops.aten.view.default(wait_tensor_54, [16384, 8192]);  wait_tensor_54 = None
        view_120 = torch.ops.aten.view.default(mm_32, [2, 8192, 14336]);  mm_32 = None
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(view_120, torch.float32);  view_120 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_158)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_158, sigmoid_4);  sigmoid_4 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        all_gather_into_tensor_51 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_160, 4, '1');  convert_element_type_160 = None
        wait_tensor_55 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_51);  all_gather_into_tensor_51 = None
        permute_53 = torch.ops.aten.permute.default(wait_tensor_55, [1, 0]);  wait_tensor_55 = None
        mm_33 = torch.ops.aten.mm.default(view_119, permute_53)
        view_122 = torch.ops.aten.view.default(mm_33, [2, 8192, 14336]);  mm_33 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_159, view_122)
        view_123 = torch.ops.aten.view.default(mul_39, [16384, 14336]);  mul_39 = None
        mm_269 = torch.ops.aten.mm.default(permute_533, view_123);  permute_533 = view_123 = None
        permute_534 = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_54 = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        clone_19 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        all_gather_into_tensor_52 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_19, 4, '1');  clone_19 = None
        wait_tensor_56 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_52);  all_gather_into_tensor_52 = None
        permute_535 = torch.ops.aten.permute.default(wait_tensor_56, [1, 0]);  wait_tensor_56 = None
        mm_270 = torch.ops.aten.mm.default(view_668, permute_535);  view_668 = permute_535 = None
        view_669 = torch.ops.aten.view.default(mm_270, [2, 8192, 14336]);  mm_270 = None
        clone_108 = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        reduce_scatter_tensor_129 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_108, 'sum', 4, '1');  clone_108 = None
        wait_tensor_307 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_129);  reduce_scatter_tensor_129 = None
        permute_536 = torch.ops.aten.permute.default(wait_tensor_307, [1, 0]);  wait_tensor_307 = None
        convert_element_type_1144 = torch.ops.prims.convert_element_type.default(permute_536, torch.float32);  permute_536 = None
        mul_356 = torch.ops.aten.mul.Tensor(view_669, convert_element_type_159);  convert_element_type_159 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_669, view_122);  view_669 = view_122 = None
        view_670 = torch.ops.aten.view.default(mul_356, [16384, 14336]);  mul_356 = None
        permute_537 = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_271 = torch.ops.aten.mm.default(permute_537, view_119);  permute_537 = None
        reduce_scatter_tensor_130 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_271, 'sum', 4, '1');  mm_271 = None
        wait_tensor_308 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_130);  reduce_scatter_tensor_130 = None
        permute_539 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_272 = torch.ops.aten.mm.default(view_670, permute_539);  view_670 = permute_539 = None
        view_671 = torch.ops.aten.view.default(mm_272, [2, 8192, 8192]);  mm_272 = None
        convert_element_type_1149 = torch.ops.prims.convert_element_type.default(wait_tensor_308, torch.float32);  wait_tensor_308 = None
        convert_element_type_1150 = torch.ops.prims.convert_element_type.default(mul_357, torch.float32);  mul_357 = None
        neg_11 = torch.ops.aten.neg.default(convert_element_type_158)
        exp_11 = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_142 = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_142);  add_142 = None
        mul_358 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_359 = torch.ops.aten.mul.Tensor(convert_element_type_1150, mul_358);  convert_element_type_1150 = None
        sub_34 = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
        mul_360 = torch.ops.aten.mul.Tensor(convert_element_type_158, sub_34);  convert_element_type_158 = sub_34 = None
        add_143 = torch.ops.aten.add.Tensor(mul_360, 1);  mul_360 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_359, add_143);  mul_359 = add_143 = None
        convert_element_type_1152 = torch.ops.prims.convert_element_type.default(mul_361, torch.bfloat16);  mul_361 = None
        view_672 = torch.ops.aten.view.default(convert_element_type_1152, [16384, 14336]);  convert_element_type_1152 = None
        permute_541 = torch.ops.aten.permute.default(view_672, [1, 0])
        mm_273 = torch.ops.aten.mm.default(permute_541, view_119);  permute_541 = view_119 = None
        reduce_scatter_tensor_131 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_273, 'sum', 4, '1');  mm_273 = None
        wait_tensor_309 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_131);  reduce_scatter_tensor_131 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        all_gather_into_tensor_49 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_155, 4, '1');  convert_element_type_155 = None
        wait_tensor_53 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_49);  all_gather_into_tensor_49 = None
        permute_52 = torch.ops.aten.permute.default(wait_tensor_53, [1, 0]);  wait_tensor_53 = None
        permute_543 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_274 = torch.ops.aten.mm.default(view_672, permute_543);  view_672 = permute_543 = None
        view_673 = torch.ops.aten.view.default(mm_274, [2, 8192, 8192]);  mm_274 = None
        add_144 = torch.ops.aten.add.Tensor(view_671, view_673);  view_671 = view_673 = None
        convert_element_type_1157 = torch.ops.prims.convert_element_type.default(wait_tensor_309, torch.float32);  wait_tensor_309 = None
        reduce_scatter_tensor_132 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_144, 'sum', 2, '3');  add_144 = None
        wait_tensor_310 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_132);  reduce_scatter_tensor_132 = None
        convert_element_type_1158 = torch.ops.prims.convert_element_type.default(wait_tensor_310, torch.float32);  wait_tensor_310 = None
        convert_element_type_1160 = torch.ops.prims.convert_element_type.default(wait_tensor_52, torch.float32);  wait_tensor_52 = None
        mul_362 = torch.ops.aten.mul.Tensor(convert_element_type_1158, convert_element_type_1160);  convert_element_type_1160 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_36, mul_362)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_364, [2], True);  mul_364 = None
        div_23 = torch.ops.aten.div.Tensor(mul_36, 8192)
        mul_365 = torch.ops.aten.mul.Tensor(div_23, sum_69);  div_23 = sum_69 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_362, mul_365);  mul_362 = mul_365 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_9);  sub_35 = rsqrt_9 = None
        mul_367 = torch.ops.aten.mul.Tensor(convert_element_type_1158, mul_36);  convert_element_type_1158 = mul_36 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1]);  mul_367 = None
        convert_element_type_1161 = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None
        add_145 = torch.ops.aten.add.Tensor(add_141, convert_element_type_1161);  add_141 = convert_element_type_1161 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None
        reduce_scatter_tensor_133 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_10, 'sum', 8, '0');  convert_element_type_default_10 = None
        wait_tensor_311 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_133);  reduce_scatter_tensor_133 = None
        view_674 = torch.ops.aten.view.default(add_145, [8192, 8192])
        permute_545 = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_275 = torch.ops.aten.mm.default(permute_545, view_117);  permute_545 = view_117 = None
        permute_546 = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
        permute_547 = torch.ops.aten.permute.default(wait_tensor_51, [1, 0]);  wait_tensor_51 = None
        mm_276 = torch.ops.aten.mm.default(view_674, permute_547);  view_674 = permute_547 = None
        view_675 = torch.ops.aten.view.default(mm_276, [1, 8192, 8192]);  mm_276 = None
        clone_109 = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
        reduce_scatter_tensor_134 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_109, 'sum', 8, '0');  clone_109 = None
        wait_tensor_312 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_134);  reduce_scatter_tensor_134 = None
        permute_548 = torch.ops.aten.permute.default(wait_tensor_312, [1, 0]);  wait_tensor_312 = None
        convert_element_type_1168 = torch.ops.prims.convert_element_type.default(permute_548, torch.float32);  permute_548 = None
        view_676 = torch.ops.aten.view.default(view_675, [1, 8192, 32, 256]);  view_675 = None
        permute_549 = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_16 = torch.ops.aten.add.Scalar(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_8);  convert_element_type_134 = None
        all_gather_into_tensor_43 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_133, 8, '0');  convert_element_type_133 = None
        wait_tensor_47 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_43);  all_gather_into_tensor_43 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, wait_tensor_47)
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        view_100 = torch.ops.aten.view.default(convert_element_type_135, [8192, 8192]);  convert_element_type_135 = None
        view_101 = torch.ops.aten.view.default(mm_28, [1, 8192, 8192]);  mm_28 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        all_gather_into_tensor_45 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_139, 8, '0');  convert_element_type_139 = None
        wait_tensor_49 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_45);  all_gather_into_tensor_45 = None
        permute_45 = torch.ops.aten.permute.default(wait_tensor_49, [1, 0]);  wait_tensor_49 = None
        mm_29 = torch.ops.aten.mm.default(view_100, permute_45)
        view_103 = torch.ops.aten.view.default(mm_29, [1, 8192, 2048]);  mm_29 = None
        view_105 = torch.ops.aten.view.default(mm_30, [1, 8192, 2048]);  mm_30 = None
        view_106 = torch.ops.aten.view.default(view_101, [1, 8192, 32, 256]);  view_101 = None
        view_107 = torch.ops.aten.view.default(view_103, [1, 8192, 8, 256]);  view_103 = None
        view_108 = torch.ops.aten.view.default(view_105, [1, 8192, 8, 256]);  view_105 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None
        view_109 = torch.ops.aten.view.default(convert_element_type_145, [1, 8192, 32, 128, 2]);  convert_element_type_145 = None
        view_as_complex_8 = torch.ops.aten.view_as_complex.default(view_109);  view_109 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        view_110 = torch.ops.aten.view.default(convert_element_type_146, [1, 8192, 8, 128, 2]);  convert_element_type_146 = None
        view_as_complex_9 = torch.ops.aten.view_as_complex.default(view_110);  view_110 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_as_complex_8, view_11);  view_as_complex_8 = None
        view_as_real_8 = torch.ops.aten.view_as_real.default(mul_34);  mul_34 = None
        view_112 = torch.ops.aten.view.default(view_as_real_8, [1, 8192, 32, 256]);  view_as_real_8 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_as_complex_9, view_11);  view_as_complex_9 = None
        view_as_real_9 = torch.ops.aten.view_as_real.default(mul_35);  mul_35 = None
        view_113 = torch.ops.aten.view.default(view_as_real_9, [1, 8192, 8, 256]);  view_as_real_9 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(view_112, torch.bfloat16);  view_112 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(view_113, torch.bfloat16);  view_113 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(convert_element_type_148, 3);  convert_element_type_148 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_8, [1, 8192, 8, 4, 256]);  unsqueeze_8 = None
        clone_16 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_114 = torch.ops.aten.view.default(clone_16, [1, 8192, 32, 256]);  clone_16 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_108, 3);  view_108 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_9, [1, 8192, 8, 4, 256]);  unsqueeze_9 = None
        clone_17 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_115 = torch.ops.aten.view.default(clone_17, [1, 8192, 32, 256]);  clone_17 = None
        permute_47 = torch.ops.aten.permute.default(convert_element_type_147, [0, 2, 1, 3]);  convert_element_type_147 = None
        permute_48 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_49 = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_549, permute_47, permute_48, permute_49, getitem_40, getitem_41, None, None, 8192, 8192, 0.0, True, getitem_46, getitem_47, scale = 0.0625);  permute_549 = permute_47 = permute_48 = permute_49 = getitem_40 = getitem_41 = getitem_46 = getitem_47 = None
        getitem_181 = _scaled_dot_product_flash_attention_backward_11[0]
        getitem_182 = _scaled_dot_product_flash_attention_backward_11[1]
        getitem_183 = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
        permute_550 = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        permute_551 = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None
        permute_552 = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_677 = torch.ops.aten.view.default(permute_550, [1, 8192, 8, 4, 256]);  permute_550 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(view_677, [3], True);  view_677 = None
        squeeze_22 = torch.ops.aten.squeeze.dim(sum_71, 3);  sum_71 = None
        view_678 = torch.ops.aten.view.default(permute_551, [1, 8192, 8, 4, 256]);  permute_551 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_678, [3], True);  view_678 = None
        squeeze_23 = torch.ops.aten.squeeze.dim(sum_72, 3);  sum_72 = None
        convert_element_type_1169 = torch.ops.prims.convert_element_type.default(squeeze_23, torch.float32);  squeeze_23 = None
        convert_element_type_1170 = torch.ops.prims.convert_element_type.default(permute_552, torch.float32);  permute_552 = None
        view_679 = torch.ops.aten.view.default(convert_element_type_1169, [1, 8192, 8, 128, 2]);  convert_element_type_1169 = None
        view_as_complex_54 = torch.ops.aten.view_as_complex.default(view_679);  view_679 = None
        mul_368 = torch.ops.aten.mul.Tensor(view_as_complex_54, _conj);  view_as_complex_54 = None
        view_680 = torch.ops.aten.view.default(convert_element_type_1170, [1, 8192, 32, 128, 2]);  convert_element_type_1170 = None
        view_as_complex_55 = torch.ops.aten.view_as_complex.default(view_680);  view_680 = None
        mul_369 = torch.ops.aten.mul.Tensor(view_as_complex_55, _conj);  view_as_complex_55 = None
        view_as_real_54 = torch.ops.aten.view_as_real.default(mul_368);  mul_368 = None
        view_681 = torch.ops.aten.view.default(view_as_real_54, [1, 8192, 8, 256]);  view_as_real_54 = None
        convert_element_type_1171 = torch.ops.prims.convert_element_type.default(view_681, torch.bfloat16);  view_681 = None
        view_as_real_55 = torch.ops.aten.view_as_real.default(mul_369);  mul_369 = None
        view_682 = torch.ops.aten.view.default(view_as_real_55, [1, 8192, 32, 256]);  view_as_real_55 = None
        convert_element_type_1172 = torch.ops.prims.convert_element_type.default(view_682, torch.bfloat16);  view_682 = None
        view_683 = torch.ops.aten.view.default(squeeze_22, [1, 8192, 2048]);  squeeze_22 = None
        view_684 = torch.ops.aten.view.default(convert_element_type_1171, [1, 8192, 2048]);  convert_element_type_1171 = None
        view_685 = torch.ops.aten.view.default(convert_element_type_1172, [1, 8192, 8192]);  convert_element_type_1172 = None
        view_686 = torch.ops.aten.view.default(view_683, [8192, 2048]);  view_683 = None
        permute_553 = torch.ops.aten.permute.default(view_686, [1, 0])
        mm_277 = torch.ops.aten.mm.default(permute_553, view_100);  permute_553 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        all_gather_into_tensor_46 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_142, 8, '0');  convert_element_type_142 = None
        wait_tensor_50 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_46);  all_gather_into_tensor_46 = None
        permute_46 = torch.ops.aten.permute.default(wait_tensor_50, [1, 0]);  wait_tensor_50 = None
        permute_555 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_278 = torch.ops.aten.mm.default(view_686, permute_555);  view_686 = permute_555 = None
        view_687 = torch.ops.aten.view.default(mm_278, [1, 8192, 8192]);  mm_278 = None
        convert_element_type_1177 = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        reduce_scatter_tensor_135 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1177, 'sum', 8, '0');  convert_element_type_1177 = None
        wait_tensor_313 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_135);  reduce_scatter_tensor_135 = None
        view_688 = torch.ops.aten.view.default(view_684, [8192, 2048]);  view_684 = None
        permute_557 = torch.ops.aten.permute.default(view_688, [1, 0])
        mm_279 = torch.ops.aten.mm.default(permute_557, view_100);  permute_557 = None
        permute_559 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_280 = torch.ops.aten.mm.default(view_688, permute_559);  view_688 = permute_559 = None
        view_689 = torch.ops.aten.view.default(mm_280, [1, 8192, 8192]);  mm_280 = None
        add_146 = torch.ops.aten.add.Tensor(view_687, view_689);  view_687 = view_689 = None
        convert_element_type_1182 = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        reduce_scatter_tensor_136 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1182, 'sum', 8, '0');  convert_element_type_1182 = None
        wait_tensor_314 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_136);  reduce_scatter_tensor_136 = None
        view_690 = torch.ops.aten.view.default(view_685, [8192, 8192]);  view_685 = None
        permute_561 = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_281 = torch.ops.aten.mm.default(permute_561, view_100);  permute_561 = view_100 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        all_gather_into_tensor_44 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_136, 8, '0');  convert_element_type_136 = None
        wait_tensor_48 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_44);  all_gather_into_tensor_44 = None
        permute_44 = torch.ops.aten.permute.default(wait_tensor_48, [1, 0]);  wait_tensor_48 = None
        permute_563 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_282 = torch.ops.aten.mm.default(view_690, permute_563);  view_690 = permute_563 = None
        view_691 = torch.ops.aten.view.default(mm_282, [1, 8192, 8192]);  mm_282 = None
        add_147 = torch.ops.aten.add.Tensor(add_146, view_691);  add_146 = view_691 = None
        reduce_scatter_tensor_137 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_281, 'sum', 8, '0');  mm_281 = None
        wait_tensor_315 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_137);  reduce_scatter_tensor_137 = None
        convert_element_type_1187 = torch.ops.prims.convert_element_type.default(wait_tensor_315, torch.float32);  wait_tensor_315 = None
        convert_element_type_1188 = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        convert_element_type_1190 = torch.ops.prims.convert_element_type.default(wait_tensor_47, torch.float32);  wait_tensor_47 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_1188, convert_element_type_1190);  convert_element_type_1190 = None
        mul_372 = torch.ops.aten.mul.Tensor(mul_32, mul_370)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        div_24 = torch.ops.aten.div.Tensor(mul_32, 8192)
        mul_373 = torch.ops.aten.mul.Tensor(div_24, sum_73);  div_24 = sum_73 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_370, mul_373);  mul_370 = mul_373 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_8);  sub_36 = rsqrt_8 = None
        mul_375 = torch.ops.aten.mul.Tensor(convert_element_type_1188, mul_32);  convert_element_type_1188 = mul_32 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        convert_element_type_1191 = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None
        add_148 = torch.ops.aten.add.Tensor(add_145, convert_element_type_1191);  add_145 = convert_element_type_1191 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(sum_74, torch.float32);  sum_74 = None
        reduce_scatter_tensor_138 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_9, 'sum', 8, '0');  convert_element_type_default_9 = None
        wait_tensor_316 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_138);  reduce_scatter_tensor_138 = None
        all_gather_into_tensor_178 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_148, 2, '3')
        wait_tensor_317 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_178);  all_gather_into_tensor_178 = None
        view_692 = torch.ops.aten.view.default(wait_tensor_317, [16384, 8192]);  wait_tensor_317 = None
        permute_565 = torch.ops.aten.permute.default(view_692, [1, 0])
        view_93 = torch.ops.aten.view.default(mm_24, [1, 8192, 8192]);  mm_24 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, view_93);  view_93 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_14 = torch.ops.aten.add.Scalar(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_7);  convert_element_type_120 = None
        all_gather_into_tensor_38 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_119, 8, '0');  convert_element_type_119 = None
        wait_tensor_41 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_38);  all_gather_into_tensor_38 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, wait_tensor_41)
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        all_gather_into_tensor_39 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_122, 4, '1');  convert_element_type_122 = None
        wait_tensor_42 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_39);  all_gather_into_tensor_39 = None
        permute_41 = torch.ops.aten.permute.default(wait_tensor_42, [1, 0]);  wait_tensor_42 = None
        all_gather_into_tensor_40 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_121, 2, '3');  convert_element_type_121 = None
        wait_tensor_43 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_40);  all_gather_into_tensor_40 = None
        view_94 = torch.ops.aten.view.default(wait_tensor_43, [16384, 8192]);  wait_tensor_43 = None
        mm_25 = torch.ops.aten.mm.default(view_94, permute_41)
        view_95 = torch.ops.aten.view.default(mm_25, [2, 8192, 14336]);  mm_25 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_95, torch.float32);  view_95 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_125, sigmoid_3);  sigmoid_3 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        view_97 = torch.ops.aten.view.default(mm_26, [2, 8192, 14336]);  mm_26 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_126, view_97)
        view_98 = torch.ops.aten.view.default(mul_31, [16384, 14336]);  mul_31 = None
        mm_283 = torch.ops.aten.mm.default(permute_565, view_98);  permute_565 = view_98 = None
        permute_566 = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        permute_43 = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        clone_15 = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        all_gather_into_tensor_42 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_15, 4, '1');  clone_15 = None
        wait_tensor_45 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_42);  all_gather_into_tensor_42 = None
        permute_567 = torch.ops.aten.permute.default(wait_tensor_45, [1, 0]);  wait_tensor_45 = None
        mm_284 = torch.ops.aten.mm.default(view_692, permute_567);  view_692 = permute_567 = None
        view_693 = torch.ops.aten.view.default(mm_284, [2, 8192, 14336]);  mm_284 = None
        clone_112 = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        reduce_scatter_tensor_139 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_112, 'sum', 4, '1');  clone_112 = None
        wait_tensor_318 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_139);  reduce_scatter_tensor_139 = None
        permute_568 = torch.ops.aten.permute.default(wait_tensor_318, [1, 0]);  wait_tensor_318 = None
        convert_element_type_1198 = torch.ops.prims.convert_element_type.default(permute_568, torch.float32);  permute_568 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_693, convert_element_type_126);  convert_element_type_126 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_693, view_97);  view_693 = view_97 = None
        view_694 = torch.ops.aten.view.default(mul_376, [16384, 14336]);  mul_376 = None
        permute_569 = torch.ops.aten.permute.default(view_694, [1, 0])
        mm_285 = torch.ops.aten.mm.default(permute_569, view_94);  permute_569 = None
        reduce_scatter_tensor_140 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_285, 'sum', 4, '1');  mm_285 = None
        wait_tensor_319 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_140);  reduce_scatter_tensor_140 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        all_gather_into_tensor_41 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_127, 4, '1');  convert_element_type_127 = None
        wait_tensor_44 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_41);  all_gather_into_tensor_41 = None
        permute_42 = torch.ops.aten.permute.default(wait_tensor_44, [1, 0]);  wait_tensor_44 = None
        permute_571 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_286 = torch.ops.aten.mm.default(view_694, permute_571);  view_694 = permute_571 = None
        view_695 = torch.ops.aten.view.default(mm_286, [2, 8192, 8192]);  mm_286 = None
        convert_element_type_1203 = torch.ops.prims.convert_element_type.default(wait_tensor_319, torch.float32);  wait_tensor_319 = None
        convert_element_type_1204 = torch.ops.prims.convert_element_type.default(mul_377, torch.float32);  mul_377 = None
        neg_12 = torch.ops.aten.neg.default(convert_element_type_125)
        exp_12 = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_149 = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(add_149);  add_149 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_379 = torch.ops.aten.mul.Tensor(convert_element_type_1204, mul_378);  convert_element_type_1204 = None
        sub_37 = torch.ops.aten.sub.Tensor(1, mul_378);  mul_378 = None
        mul_380 = torch.ops.aten.mul.Tensor(convert_element_type_125, sub_37);  convert_element_type_125 = sub_37 = None
        add_150 = torch.ops.aten.add.Tensor(mul_380, 1);  mul_380 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_379, add_150);  mul_379 = add_150 = None
        convert_element_type_1206 = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16);  mul_381 = None
        view_696 = torch.ops.aten.view.default(convert_element_type_1206, [16384, 14336]);  convert_element_type_1206 = None
        permute_573 = torch.ops.aten.permute.default(view_696, [1, 0])
        mm_287 = torch.ops.aten.mm.default(permute_573, view_94);  permute_573 = view_94 = None
        reduce_scatter_tensor_141 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_287, 'sum', 4, '1');  mm_287 = None
        wait_tensor_320 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_141);  reduce_scatter_tensor_141 = None
        permute_575 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_288 = torch.ops.aten.mm.default(view_696, permute_575);  view_696 = permute_575 = None
        view_697 = torch.ops.aten.view.default(mm_288, [2, 8192, 8192]);  mm_288 = None
        add_151 = torch.ops.aten.add.Tensor(view_695, view_697);  view_695 = view_697 = None
        convert_element_type_1211 = torch.ops.prims.convert_element_type.default(wait_tensor_320, torch.float32);  wait_tensor_320 = None
        reduce_scatter_tensor_142 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_151, 'sum', 2, '3');  add_151 = None
        wait_tensor_321 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_142);  reduce_scatter_tensor_142 = None
        convert_element_type_1212 = torch.ops.prims.convert_element_type.default(wait_tensor_321, torch.float32);  wait_tensor_321 = None
        convert_element_type_1214 = torch.ops.prims.convert_element_type.default(wait_tensor_41, torch.float32);  wait_tensor_41 = None
        mul_382 = torch.ops.aten.mul.Tensor(convert_element_type_1212, convert_element_type_1214);  convert_element_type_1214 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_28, mul_382)
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
        div_25 = torch.ops.aten.div.Tensor(mul_28, 8192)
        mul_385 = torch.ops.aten.mul.Tensor(div_25, sum_75);  div_25 = sum_75 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_382, mul_385);  mul_382 = mul_385 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_7);  sub_38 = rsqrt_7 = None
        mul_387 = torch.ops.aten.mul.Tensor(convert_element_type_1212, mul_28);  convert_element_type_1212 = mul_28 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        convert_element_type_1215 = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None
        add_152 = torch.ops.aten.add.Tensor(add_148, convert_element_type_1215);  add_148 = convert_element_type_1215 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(sum_76, torch.float32);  sum_76 = None
        reduce_scatter_tensor_143 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_8, 'sum', 8, '0');  convert_element_type_default_8 = None
        wait_tensor_322 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_143);  reduce_scatter_tensor_143 = None
        view_698 = torch.ops.aten.view.default(add_152, [8192, 8192])
        permute_577 = torch.ops.aten.permute.default(view_698, [1, 0])
        permute_39 = torch.ops.aten.permute.default(getitem_31, [0, 2, 1, 3])
        view_91 = torch.ops.aten.view.default(permute_39, [1, 8192, 8192]);  permute_39 = None
        view_92 = torch.ops.aten.view.default(view_91, [8192, 8192]);  view_91 = None
        mm_289 = torch.ops.aten.mm.default(permute_577, view_92);  permute_577 = view_92 = None
        permute_578 = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_40 = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        clone_14 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        all_gather_into_tensor_37 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_14, 8, '0');  clone_14 = None
        wait_tensor_40 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_37);  all_gather_into_tensor_37 = None
        permute_579 = torch.ops.aten.permute.default(wait_tensor_40, [1, 0]);  wait_tensor_40 = None
        mm_290 = torch.ops.aten.mm.default(view_698, permute_579);  view_698 = permute_579 = None
        view_699 = torch.ops.aten.view.default(mm_290, [1, 8192, 8192]);  mm_290 = None
        clone_113 = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        reduce_scatter_tensor_144 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_113, 'sum', 8, '0');  clone_113 = None
        wait_tensor_323 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_144);  reduce_scatter_tensor_144 = None
        permute_580 = torch.ops.aten.permute.default(wait_tensor_323, [1, 0]);  wait_tensor_323 = None
        convert_element_type_1222 = torch.ops.prims.convert_element_type.default(permute_580, torch.float32);  permute_580 = None
        view_700 = torch.ops.aten.view.default(view_699, [1, 8192, 32, 256]);  view_699 = None
        permute_581 = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_12 = torch.ops.aten.add.Scalar(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_6);  convert_element_type_101 = None
        all_gather_into_tensor_33 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_100, 8, '0');  convert_element_type_100 = None
        wait_tensor_36 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_33);  all_gather_into_tensor_33 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, wait_tensor_36)
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        all_gather_into_tensor_34 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_103, 8, '0');  convert_element_type_103 = None
        wait_tensor_37 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_34);  all_gather_into_tensor_34 = None
        permute_33 = torch.ops.aten.permute.default(wait_tensor_37, [1, 0]);  wait_tensor_37 = None
        view_75 = torch.ops.aten.view.default(convert_element_type_102, [8192, 8192]);  convert_element_type_102 = None
        mm_21 = torch.ops.aten.mm.default(view_75, permute_33)
        view_76 = torch.ops.aten.view.default(mm_21, [1, 8192, 8192]);  mm_21 = None
        view_78 = torch.ops.aten.view.default(mm_22, [1, 8192, 2048]);  mm_22 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        all_gather_into_tensor_36 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_109, 8, '0');  convert_element_type_109 = None
        wait_tensor_39 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_36);  all_gather_into_tensor_36 = None
        permute_35 = torch.ops.aten.permute.default(wait_tensor_39, [1, 0]);  wait_tensor_39 = None
        mm_23 = torch.ops.aten.mm.default(view_75, permute_35)
        view_80 = torch.ops.aten.view.default(mm_23, [1, 8192, 2048]);  mm_23 = None
        view_81 = torch.ops.aten.view.default(view_76, [1, 8192, 32, 256]);  view_76 = None
        view_82 = torch.ops.aten.view.default(view_78, [1, 8192, 8, 256]);  view_78 = None
        view_83 = torch.ops.aten.view.default(view_80, [1, 8192, 8, 256]);  view_80 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        view_84 = torch.ops.aten.view.default(convert_element_type_112, [1, 8192, 32, 128, 2]);  convert_element_type_112 = None
        view_as_complex_6 = torch.ops.aten.view_as_complex.default(view_84);  view_84 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        view_85 = torch.ops.aten.view.default(convert_element_type_113, [1, 8192, 8, 128, 2]);  convert_element_type_113 = None
        view_as_complex_7 = torch.ops.aten.view_as_complex.default(view_85);  view_85 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_as_complex_6, view_11);  view_as_complex_6 = None
        view_as_real_6 = torch.ops.aten.view_as_real.default(mul_26);  mul_26 = None
        view_87 = torch.ops.aten.view.default(view_as_real_6, [1, 8192, 32, 256]);  view_as_real_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_as_complex_7, view_11);  view_as_complex_7 = None
        view_as_real_7 = torch.ops.aten.view_as_real.default(mul_27);  mul_27 = None
        view_88 = torch.ops.aten.view.default(view_as_real_7, [1, 8192, 8, 256]);  view_as_real_7 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(view_87, torch.bfloat16);  view_87 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(view_88, torch.bfloat16);  view_88 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(convert_element_type_115, 3);  convert_element_type_115 = None
        expand_6 = torch.ops.aten.expand.default(unsqueeze_6, [1, 8192, 8, 4, 256]);  unsqueeze_6 = None
        clone_12 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_89 = torch.ops.aten.view.default(clone_12, [1, 8192, 32, 256]);  clone_12 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_83, 3);  view_83 = None
        expand_7 = torch.ops.aten.expand.default(unsqueeze_7, [1, 8192, 8, 4, 256]);  unsqueeze_7 = None
        clone_13 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_90 = torch.ops.aten.view.default(clone_13, [1, 8192, 32, 256]);  clone_13 = None
        permute_36 = torch.ops.aten.permute.default(convert_element_type_114, [0, 2, 1, 3]);  convert_element_type_114 = None
        permute_37 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        permute_38 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        _scaled_dot_product_flash_attention_backward_12 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_581, permute_36, permute_37, permute_38, getitem_31, getitem_32, None, None, 8192, 8192, 0.0, True, getitem_37, getitem_38, scale = 0.0625);  permute_581 = permute_36 = permute_37 = permute_38 = getitem_31 = getitem_32 = getitem_37 = getitem_38 = None
        getitem_184 = _scaled_dot_product_flash_attention_backward_12[0]
        getitem_185 = _scaled_dot_product_flash_attention_backward_12[1]
        getitem_186 = _scaled_dot_product_flash_attention_backward_12[2];  _scaled_dot_product_flash_attention_backward_12 = None
        permute_582 = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None
        permute_583 = torch.ops.aten.permute.default(getitem_185, [0, 2, 1, 3]);  getitem_185 = None
        permute_584 = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None
        view_701 = torch.ops.aten.view.default(permute_582, [1, 8192, 8, 4, 256]);  permute_582 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(view_701, [3], True);  view_701 = None
        squeeze_24 = torch.ops.aten.squeeze.dim(sum_77, 3);  sum_77 = None
        view_702 = torch.ops.aten.view.default(permute_583, [1, 8192, 8, 4, 256]);  permute_583 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(view_702, [3], True);  view_702 = None
        squeeze_25 = torch.ops.aten.squeeze.dim(sum_78, 3);  sum_78 = None
        convert_element_type_1223 = torch.ops.prims.convert_element_type.default(squeeze_25, torch.float32);  squeeze_25 = None
        convert_element_type_1224 = torch.ops.prims.convert_element_type.default(permute_584, torch.float32);  permute_584 = None
        view_703 = torch.ops.aten.view.default(convert_element_type_1223, [1, 8192, 8, 128, 2]);  convert_element_type_1223 = None
        view_as_complex_56 = torch.ops.aten.view_as_complex.default(view_703);  view_703 = None
        mul_388 = torch.ops.aten.mul.Tensor(view_as_complex_56, _conj);  view_as_complex_56 = None
        view_704 = torch.ops.aten.view.default(convert_element_type_1224, [1, 8192, 32, 128, 2]);  convert_element_type_1224 = None
        view_as_complex_57 = torch.ops.aten.view_as_complex.default(view_704);  view_704 = None
        mul_389 = torch.ops.aten.mul.Tensor(view_as_complex_57, _conj);  view_as_complex_57 = None
        view_as_real_56 = torch.ops.aten.view_as_real.default(mul_388);  mul_388 = None
        view_705 = torch.ops.aten.view.default(view_as_real_56, [1, 8192, 8, 256]);  view_as_real_56 = None
        convert_element_type_1225 = torch.ops.prims.convert_element_type.default(view_705, torch.bfloat16);  view_705 = None
        view_as_real_57 = torch.ops.aten.view_as_real.default(mul_389);  mul_389 = None
        view_706 = torch.ops.aten.view.default(view_as_real_57, [1, 8192, 32, 256]);  view_as_real_57 = None
        convert_element_type_1226 = torch.ops.prims.convert_element_type.default(view_706, torch.bfloat16);  view_706 = None
        view_707 = torch.ops.aten.view.default(squeeze_24, [1, 8192, 2048]);  squeeze_24 = None
        view_708 = torch.ops.aten.view.default(convert_element_type_1225, [1, 8192, 2048]);  convert_element_type_1225 = None
        view_709 = torch.ops.aten.view.default(convert_element_type_1226, [1, 8192, 8192]);  convert_element_type_1226 = None
        view_710 = torch.ops.aten.view.default(view_707, [8192, 2048]);  view_707 = None
        permute_585 = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_291 = torch.ops.aten.mm.default(permute_585, view_75);  permute_585 = None
        permute_587 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_292 = torch.ops.aten.mm.default(view_710, permute_587);  view_710 = permute_587 = None
        view_711 = torch.ops.aten.view.default(mm_292, [1, 8192, 8192]);  mm_292 = None
        convert_element_type_1231 = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        reduce_scatter_tensor_145 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1231, 'sum', 8, '0');  convert_element_type_1231 = None
        wait_tensor_324 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_145);  reduce_scatter_tensor_145 = None
        view_712 = torch.ops.aten.view.default(view_708, [8192, 2048]);  view_708 = None
        permute_589 = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_293 = torch.ops.aten.mm.default(permute_589, view_75);  permute_589 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        all_gather_into_tensor_35 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_106, 8, '0');  convert_element_type_106 = None
        wait_tensor_38 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_35);  all_gather_into_tensor_35 = None
        permute_34 = torch.ops.aten.permute.default(wait_tensor_38, [1, 0]);  wait_tensor_38 = None
        permute_591 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_294 = torch.ops.aten.mm.default(view_712, permute_591);  view_712 = permute_591 = None
        view_713 = torch.ops.aten.view.default(mm_294, [1, 8192, 8192]);  mm_294 = None
        add_153 = torch.ops.aten.add.Tensor(view_711, view_713);  view_711 = view_713 = None
        convert_element_type_1236 = torch.ops.prims.convert_element_type.default(mm_293, torch.float32);  mm_293 = None
        reduce_scatter_tensor_146 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1236, 'sum', 8, '0');  convert_element_type_1236 = None
        wait_tensor_325 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_146);  reduce_scatter_tensor_146 = None
        view_714 = torch.ops.aten.view.default(view_709, [8192, 8192]);  view_709 = None
        permute_593 = torch.ops.aten.permute.default(view_714, [1, 0])
        mm_295 = torch.ops.aten.mm.default(permute_593, view_75);  permute_593 = view_75 = None
        permute_595 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_296 = torch.ops.aten.mm.default(view_714, permute_595);  view_714 = permute_595 = None
        view_715 = torch.ops.aten.view.default(mm_296, [1, 8192, 8192]);  mm_296 = None
        add_154 = torch.ops.aten.add.Tensor(add_153, view_715);  add_153 = view_715 = None
        reduce_scatter_tensor_147 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_295, 'sum', 8, '0');  mm_295 = None
        wait_tensor_326 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_147);  reduce_scatter_tensor_147 = None
        convert_element_type_1241 = torch.ops.prims.convert_element_type.default(wait_tensor_326, torch.float32);  wait_tensor_326 = None
        convert_element_type_1242 = torch.ops.prims.convert_element_type.default(add_154, torch.float32);  add_154 = None
        convert_element_type_1244 = torch.ops.prims.convert_element_type.default(wait_tensor_36, torch.float32);  wait_tensor_36 = None
        mul_390 = torch.ops.aten.mul.Tensor(convert_element_type_1242, convert_element_type_1244);  convert_element_type_1244 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_24, mul_390)
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        div_26 = torch.ops.aten.div.Tensor(mul_24, 8192)
        mul_393 = torch.ops.aten.mul.Tensor(div_26, sum_79);  div_26 = sum_79 = None
        sub_39 = torch.ops.aten.sub.Tensor(mul_390, mul_393);  mul_390 = mul_393 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_6);  sub_39 = rsqrt_6 = None
        mul_395 = torch.ops.aten.mul.Tensor(convert_element_type_1242, mul_24);  convert_element_type_1242 = mul_24 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        convert_element_type_1245 = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None
        add_155 = torch.ops.aten.add.Tensor(add_152, convert_element_type_1245);  add_152 = convert_element_type_1245 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(sum_80, torch.float32);  sum_80 = None
        reduce_scatter_tensor_148 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_7, 'sum', 8, '0');  convert_element_type_default_7 = None
        wait_tensor_327 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_148);  reduce_scatter_tensor_148 = None
        all_gather_into_tensor_179 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_155, 2, '3')
        wait_tensor_328 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_179);  all_gather_into_tensor_179 = None
        view_716 = torch.ops.aten.view.default(wait_tensor_328, [16384, 8192]);  wait_tensor_328 = None
        permute_597 = torch.ops.aten.permute.default(view_716, [1, 0])
        permute_28 = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3])
        view_66 = torch.ops.aten.view.default(permute_28, [1, 8192, 8192]);  permute_28 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_29 = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        view_67 = torch.ops.aten.view.default(view_66, [8192, 8192]);  view_66 = None
        clone_10 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        all_gather_into_tensor_27 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_10, 8, '0');  clone_10 = None
        wait_tensor_29 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_27);  all_gather_into_tensor_27 = None
        mm_17 = torch.ops.aten.mm.default(view_67, wait_tensor_29)
        view_68 = torch.ops.aten.view.default(mm_17, [1, 8192, 8192]);  mm_17 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, view_68);  view_68 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_10 = torch.ops.aten.add.Scalar(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_5);  convert_element_type_87 = None
        all_gather_into_tensor_28 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_86, 8, '0');  convert_element_type_86 = None
        wait_tensor_30 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_28);  all_gather_into_tensor_28 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, wait_tensor_30)
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        all_gather_into_tensor_30 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_88, 2, '3');  convert_element_type_88 = None
        wait_tensor_32 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_30);  all_gather_into_tensor_30 = None
        view_69 = torch.ops.aten.view.default(wait_tensor_32, [16384, 8192]);  wait_tensor_32 = None
        view_70 = torch.ops.aten.view.default(mm_18, [2, 8192, 14336]);  mm_18 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_92)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_92, sigmoid_2);  sigmoid_2 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        all_gather_into_tensor_31 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_94, 4, '1');  convert_element_type_94 = None
        wait_tensor_33 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_31);  all_gather_into_tensor_31 = None
        permute_31 = torch.ops.aten.permute.default(wait_tensor_33, [1, 0]);  wait_tensor_33 = None
        mm_19 = torch.ops.aten.mm.default(view_69, permute_31)
        view_72 = torch.ops.aten.view.default(mm_19, [2, 8192, 14336]);  mm_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_93, view_72)
        view_73 = torch.ops.aten.view.default(mul_23, [16384, 14336]);  mul_23 = None
        mm_297 = torch.ops.aten.mm.default(permute_597, view_73);  permute_597 = view_73 = None
        permute_598 = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_32 = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        clone_11 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        all_gather_into_tensor_32 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_11, 4, '1');  clone_11 = None
        wait_tensor_34 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_32);  all_gather_into_tensor_32 = None
        permute_599 = torch.ops.aten.permute.default(wait_tensor_34, [1, 0]);  wait_tensor_34 = None
        mm_298 = torch.ops.aten.mm.default(view_716, permute_599);  view_716 = permute_599 = None
        view_717 = torch.ops.aten.view.default(mm_298, [2, 8192, 14336]);  mm_298 = None
        clone_116 = torch.ops.aten.clone.default(permute_598, memory_format = torch.contiguous_format);  permute_598 = None
        reduce_scatter_tensor_149 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_116, 'sum', 4, '1');  clone_116 = None
        wait_tensor_329 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_149);  reduce_scatter_tensor_149 = None
        permute_600 = torch.ops.aten.permute.default(wait_tensor_329, [1, 0]);  wait_tensor_329 = None
        convert_element_type_1252 = torch.ops.prims.convert_element_type.default(permute_600, torch.float32);  permute_600 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_717, convert_element_type_93);  convert_element_type_93 = None
        mul_397 = torch.ops.aten.mul.Tensor(view_717, view_72);  view_717 = view_72 = None
        view_718 = torch.ops.aten.view.default(mul_396, [16384, 14336]);  mul_396 = None
        permute_601 = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_299 = torch.ops.aten.mm.default(permute_601, view_69);  permute_601 = None
        reduce_scatter_tensor_150 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_299, 'sum', 4, '1');  mm_299 = None
        wait_tensor_330 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_150);  reduce_scatter_tensor_150 = None
        permute_603 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_300 = torch.ops.aten.mm.default(view_718, permute_603);  view_718 = permute_603 = None
        view_719 = torch.ops.aten.view.default(mm_300, [2, 8192, 8192]);  mm_300 = None
        convert_element_type_1257 = torch.ops.prims.convert_element_type.default(wait_tensor_330, torch.float32);  wait_tensor_330 = None
        convert_element_type_1258 = torch.ops.prims.convert_element_type.default(mul_397, torch.float32);  mul_397 = None
        neg_13 = torch.ops.aten.neg.default(convert_element_type_92)
        exp_13 = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_156 = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(add_156);  add_156 = None
        mul_398 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_399 = torch.ops.aten.mul.Tensor(convert_element_type_1258, mul_398);  convert_element_type_1258 = None
        sub_40 = torch.ops.aten.sub.Tensor(1, mul_398);  mul_398 = None
        mul_400 = torch.ops.aten.mul.Tensor(convert_element_type_92, sub_40);  convert_element_type_92 = sub_40 = None
        add_157 = torch.ops.aten.add.Tensor(mul_400, 1);  mul_400 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_399, add_157);  mul_399 = add_157 = None
        convert_element_type_1260 = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16);  mul_401 = None
        view_720 = torch.ops.aten.view.default(convert_element_type_1260, [16384, 14336]);  convert_element_type_1260 = None
        permute_605 = torch.ops.aten.permute.default(view_720, [1, 0])
        mm_301 = torch.ops.aten.mm.default(permute_605, view_69);  permute_605 = view_69 = None
        reduce_scatter_tensor_151 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_301, 'sum', 4, '1');  mm_301 = None
        wait_tensor_331 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_151);  reduce_scatter_tensor_151 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        all_gather_into_tensor_29 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_89, 4, '1');  convert_element_type_89 = None
        wait_tensor_31 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_29);  all_gather_into_tensor_29 = None
        permute_30 = torch.ops.aten.permute.default(wait_tensor_31, [1, 0]);  wait_tensor_31 = None
        permute_607 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_302 = torch.ops.aten.mm.default(view_720, permute_607);  view_720 = permute_607 = None
        view_721 = torch.ops.aten.view.default(mm_302, [2, 8192, 8192]);  mm_302 = None
        add_158 = torch.ops.aten.add.Tensor(view_719, view_721);  view_719 = view_721 = None
        convert_element_type_1265 = torch.ops.prims.convert_element_type.default(wait_tensor_331, torch.float32);  wait_tensor_331 = None
        reduce_scatter_tensor_152 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_158, 'sum', 2, '3');  add_158 = None
        wait_tensor_332 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_152);  reduce_scatter_tensor_152 = None
        convert_element_type_1266 = torch.ops.prims.convert_element_type.default(wait_tensor_332, torch.float32);  wait_tensor_332 = None
        convert_element_type_1268 = torch.ops.prims.convert_element_type.default(wait_tensor_30, torch.float32);  wait_tensor_30 = None
        mul_402 = torch.ops.aten.mul.Tensor(convert_element_type_1266, convert_element_type_1268);  convert_element_type_1268 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_20, mul_402)
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        div_27 = torch.ops.aten.div.Tensor(mul_20, 8192)
        mul_405 = torch.ops.aten.mul.Tensor(div_27, sum_81);  div_27 = sum_81 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_402, mul_405);  mul_402 = mul_405 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_5);  sub_41 = rsqrt_5 = None
        mul_407 = torch.ops.aten.mul.Tensor(convert_element_type_1266, mul_20);  convert_element_type_1266 = mul_20 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        convert_element_type_1269 = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16);  mul_406 = None
        add_159 = torch.ops.aten.add.Tensor(add_155, convert_element_type_1269);  add_155 = convert_element_type_1269 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(sum_82, torch.float32);  sum_82 = None
        reduce_scatter_tensor_153 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_6, 'sum', 8, '0');  convert_element_type_default_6 = None
        wait_tensor_333 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_153);  reduce_scatter_tensor_153 = None
        view_722 = torch.ops.aten.view.default(add_159, [8192, 8192])
        permute_609 = torch.ops.aten.permute.default(view_722, [1, 0])
        mm_303 = torch.ops.aten.mm.default(permute_609, view_67);  permute_609 = view_67 = None
        permute_610 = torch.ops.aten.permute.default(mm_303, [1, 0]);  mm_303 = None
        permute_611 = torch.ops.aten.permute.default(wait_tensor_29, [1, 0]);  wait_tensor_29 = None
        mm_304 = torch.ops.aten.mm.default(view_722, permute_611);  view_722 = permute_611 = None
        view_723 = torch.ops.aten.view.default(mm_304, [1, 8192, 8192]);  mm_304 = None
        clone_117 = torch.ops.aten.clone.default(permute_610, memory_format = torch.contiguous_format);  permute_610 = None
        reduce_scatter_tensor_154 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_117, 'sum', 8, '0');  clone_117 = None
        wait_tensor_334 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_154);  reduce_scatter_tensor_154 = None
        permute_612 = torch.ops.aten.permute.default(wait_tensor_334, [1, 0]);  wait_tensor_334 = None
        convert_element_type_1276 = torch.ops.prims.convert_element_type.default(permute_612, torch.float32);  permute_612 = None
        view_724 = torch.ops.aten.view.default(view_723, [1, 8192, 32, 256]);  view_723 = None
        permute_613 = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_68, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_8 = torch.ops.aten.add.Scalar(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_68, rsqrt_4);  convert_element_type_68 = None
        all_gather_into_tensor_23 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_67, 8, '0');  convert_element_type_67 = None
        wait_tensor_25 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_23);  all_gather_into_tensor_23 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, wait_tensor_25)
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_50 = torch.ops.aten.view.default(convert_element_type_69, [8192, 8192]);  convert_element_type_69 = None
        view_51 = torch.ops.aten.view.default(mm_14, [1, 8192, 8192]);  mm_14 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        all_gather_into_tensor_25 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_73, 8, '0');  convert_element_type_73 = None
        wait_tensor_27 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_25);  all_gather_into_tensor_25 = None
        permute_23 = torch.ops.aten.permute.default(wait_tensor_27, [1, 0]);  wait_tensor_27 = None
        mm_15 = torch.ops.aten.mm.default(view_50, permute_23)
        view_53 = torch.ops.aten.view.default(mm_15, [1, 8192, 2048]);  mm_15 = None
        view_55 = torch.ops.aten.view.default(mm_16, [1, 8192, 2048]);  mm_16 = None
        view_56 = torch.ops.aten.view.default(view_51, [1, 8192, 32, 256]);  view_51 = None
        view_57 = torch.ops.aten.view.default(view_53, [1, 8192, 8, 256]);  view_53 = None
        view_58 = torch.ops.aten.view.default(view_55, [1, 8192, 8, 256]);  view_55 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        view_59 = torch.ops.aten.view.default(convert_element_type_79, [1, 8192, 32, 128, 2]);  convert_element_type_79 = None
        view_as_complex_4 = torch.ops.aten.view_as_complex.default(view_59);  view_59 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        view_60 = torch.ops.aten.view.default(convert_element_type_80, [1, 8192, 8, 128, 2]);  convert_element_type_80 = None
        view_as_complex_5 = torch.ops.aten.view_as_complex.default(view_60);  view_60 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_as_complex_4, view_11);  view_as_complex_4 = None
        view_as_real_4 = torch.ops.aten.view_as_real.default(mul_18);  mul_18 = None
        view_62 = torch.ops.aten.view.default(view_as_real_4, [1, 8192, 32, 256]);  view_as_real_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_as_complex_5, view_11);  view_as_complex_5 = None
        view_as_real_5 = torch.ops.aten.view_as_real.default(mul_19);  mul_19 = None
        view_63 = torch.ops.aten.view.default(view_as_real_5, [1, 8192, 8, 256]);  view_as_real_5 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(view_62, torch.bfloat16);  view_62 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_82, 3);  convert_element_type_82 = None
        expand_4 = torch.ops.aten.expand.default(unsqueeze_4, [1, 8192, 8, 4, 256]);  unsqueeze_4 = None
        clone_8 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_64 = torch.ops.aten.view.default(clone_8, [1, 8192, 32, 256]);  clone_8 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(view_58, 3);  view_58 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_5, [1, 8192, 8, 4, 256]);  unsqueeze_5 = None
        clone_9 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_65 = torch.ops.aten.view.default(clone_9, [1, 8192, 32, 256]);  clone_9 = None
        permute_25 = torch.ops.aten.permute.default(convert_element_type_81, [0, 2, 1, 3]);  convert_element_type_81 = None
        permute_26 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        permute_27 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        _scaled_dot_product_flash_attention_backward_13 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_613, permute_25, permute_26, permute_27, getitem_22, getitem_23, None, None, 8192, 8192, 0.0, True, getitem_28, getitem_29, scale = 0.0625);  permute_613 = permute_25 = permute_26 = permute_27 = getitem_22 = getitem_23 = getitem_28 = getitem_29 = None
        getitem_187 = _scaled_dot_product_flash_attention_backward_13[0]
        getitem_188 = _scaled_dot_product_flash_attention_backward_13[1]
        getitem_189 = _scaled_dot_product_flash_attention_backward_13[2];  _scaled_dot_product_flash_attention_backward_13 = None
        permute_614 = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None
        permute_615 = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
        permute_616 = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None
        view_725 = torch.ops.aten.view.default(permute_614, [1, 8192, 8, 4, 256]);  permute_614 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(view_725, [3], True);  view_725 = None
        squeeze_26 = torch.ops.aten.squeeze.dim(sum_83, 3);  sum_83 = None
        view_726 = torch.ops.aten.view.default(permute_615, [1, 8192, 8, 4, 256]);  permute_615 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(view_726, [3], True);  view_726 = None
        squeeze_27 = torch.ops.aten.squeeze.dim(sum_84, 3);  sum_84 = None
        convert_element_type_1277 = torch.ops.prims.convert_element_type.default(squeeze_27, torch.float32);  squeeze_27 = None
        convert_element_type_1278 = torch.ops.prims.convert_element_type.default(permute_616, torch.float32);  permute_616 = None
        view_727 = torch.ops.aten.view.default(convert_element_type_1277, [1, 8192, 8, 128, 2]);  convert_element_type_1277 = None
        view_as_complex_58 = torch.ops.aten.view_as_complex.default(view_727);  view_727 = None
        mul_408 = torch.ops.aten.mul.Tensor(view_as_complex_58, _conj);  view_as_complex_58 = None
        view_728 = torch.ops.aten.view.default(convert_element_type_1278, [1, 8192, 32, 128, 2]);  convert_element_type_1278 = None
        view_as_complex_59 = torch.ops.aten.view_as_complex.default(view_728);  view_728 = None
        mul_409 = torch.ops.aten.mul.Tensor(view_as_complex_59, _conj);  view_as_complex_59 = None
        view_as_real_58 = torch.ops.aten.view_as_real.default(mul_408);  mul_408 = None
        view_729 = torch.ops.aten.view.default(view_as_real_58, [1, 8192, 8, 256]);  view_as_real_58 = None
        convert_element_type_1279 = torch.ops.prims.convert_element_type.default(view_729, torch.bfloat16);  view_729 = None
        view_as_real_59 = torch.ops.aten.view_as_real.default(mul_409);  mul_409 = None
        view_730 = torch.ops.aten.view.default(view_as_real_59, [1, 8192, 32, 256]);  view_as_real_59 = None
        convert_element_type_1280 = torch.ops.prims.convert_element_type.default(view_730, torch.bfloat16);  view_730 = None
        view_731 = torch.ops.aten.view.default(squeeze_26, [1, 8192, 2048]);  squeeze_26 = None
        view_732 = torch.ops.aten.view.default(convert_element_type_1279, [1, 8192, 2048]);  convert_element_type_1279 = None
        view_733 = torch.ops.aten.view.default(convert_element_type_1280, [1, 8192, 8192]);  convert_element_type_1280 = None
        view_734 = torch.ops.aten.view.default(view_731, [8192, 2048]);  view_731 = None
        permute_617 = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_305 = torch.ops.aten.mm.default(permute_617, view_50);  permute_617 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        all_gather_into_tensor_26 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_76, 8, '0');  convert_element_type_76 = None
        wait_tensor_28 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_26);  all_gather_into_tensor_26 = None
        permute_24 = torch.ops.aten.permute.default(wait_tensor_28, [1, 0]);  wait_tensor_28 = None
        permute_619 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_306 = torch.ops.aten.mm.default(view_734, permute_619);  view_734 = permute_619 = None
        view_735 = torch.ops.aten.view.default(mm_306, [1, 8192, 8192]);  mm_306 = None
        convert_element_type_1285 = torch.ops.prims.convert_element_type.default(mm_305, torch.float32);  mm_305 = None
        reduce_scatter_tensor_155 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1285, 'sum', 8, '0');  convert_element_type_1285 = None
        wait_tensor_335 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_155);  reduce_scatter_tensor_155 = None
        view_736 = torch.ops.aten.view.default(view_732, [8192, 2048]);  view_732 = None
        permute_621 = torch.ops.aten.permute.default(view_736, [1, 0])
        mm_307 = torch.ops.aten.mm.default(permute_621, view_50);  permute_621 = None
        permute_623 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_308 = torch.ops.aten.mm.default(view_736, permute_623);  view_736 = permute_623 = None
        view_737 = torch.ops.aten.view.default(mm_308, [1, 8192, 8192]);  mm_308 = None
        add_160 = torch.ops.aten.add.Tensor(view_735, view_737);  view_735 = view_737 = None
        convert_element_type_1290 = torch.ops.prims.convert_element_type.default(mm_307, torch.float32);  mm_307 = None
        reduce_scatter_tensor_156 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1290, 'sum', 8, '0');  convert_element_type_1290 = None
        wait_tensor_336 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_156);  reduce_scatter_tensor_156 = None
        view_738 = torch.ops.aten.view.default(view_733, [8192, 8192]);  view_733 = None
        permute_625 = torch.ops.aten.permute.default(view_738, [1, 0])
        mm_309 = torch.ops.aten.mm.default(permute_625, view_50);  permute_625 = view_50 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        all_gather_into_tensor_24 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_70, 8, '0');  convert_element_type_70 = None
        wait_tensor_26 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_24);  all_gather_into_tensor_24 = None
        permute_22 = torch.ops.aten.permute.default(wait_tensor_26, [1, 0]);  wait_tensor_26 = None
        permute_627 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_310 = torch.ops.aten.mm.default(view_738, permute_627);  view_738 = permute_627 = None
        view_739 = torch.ops.aten.view.default(mm_310, [1, 8192, 8192]);  mm_310 = None
        add_161 = torch.ops.aten.add.Tensor(add_160, view_739);  add_160 = view_739 = None
        reduce_scatter_tensor_157 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_309, 'sum', 8, '0');  mm_309 = None
        wait_tensor_337 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_157);  reduce_scatter_tensor_157 = None
        convert_element_type_1295 = torch.ops.prims.convert_element_type.default(wait_tensor_337, torch.float32);  wait_tensor_337 = None
        convert_element_type_1296 = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        convert_element_type_1298 = torch.ops.prims.convert_element_type.default(wait_tensor_25, torch.float32);  wait_tensor_25 = None
        mul_410 = torch.ops.aten.mul.Tensor(convert_element_type_1296, convert_element_type_1298);  convert_element_type_1298 = None
        mul_412 = torch.ops.aten.mul.Tensor(mul_16, mul_410)
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_412, [2], True);  mul_412 = None
        div_28 = torch.ops.aten.div.Tensor(mul_16, 8192)
        mul_413 = torch.ops.aten.mul.Tensor(div_28, sum_85);  div_28 = sum_85 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_410, mul_413);  mul_410 = mul_413 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_4);  sub_42 = rsqrt_4 = None
        mul_415 = torch.ops.aten.mul.Tensor(convert_element_type_1296, mul_16);  convert_element_type_1296 = mul_16 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1]);  mul_415 = None
        convert_element_type_1299 = torch.ops.prims.convert_element_type.default(mul_414, torch.bfloat16);  mul_414 = None
        add_162 = torch.ops.aten.add.Tensor(add_159, convert_element_type_1299);  add_159 = convert_element_type_1299 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(sum_86, torch.float32);  sum_86 = None
        reduce_scatter_tensor_158 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_5, 'sum', 8, '0');  convert_element_type_default_5 = None
        wait_tensor_338 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_158);  reduce_scatter_tensor_158 = None
        all_gather_into_tensor_180 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_162, 2, '3')
        wait_tensor_339 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_180);  all_gather_into_tensor_180 = None
        view_740 = torch.ops.aten.view.default(wait_tensor_339, [16384, 8192]);  wait_tensor_339 = None
        permute_629 = torch.ops.aten.permute.default(view_740, [1, 0])
        view_43 = torch.ops.aten.view.default(mm_10, [1, 8192, 8192]);  mm_10 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, view_43);  view_43 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_6 = torch.ops.aten.add.Scalar(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_3);  convert_element_type_54 = None
        all_gather_into_tensor_18 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_53, 8, '0');  convert_element_type_53 = None
        wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_18);  all_gather_into_tensor_18 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, wait_tensor_19)
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        all_gather_into_tensor_19 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_56, 4, '1');  convert_element_type_56 = None
        wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_19);  all_gather_into_tensor_19 = None
        permute_19 = torch.ops.aten.permute.default(wait_tensor_20, [1, 0]);  wait_tensor_20 = None
        all_gather_into_tensor_20 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_55, 2, '3');  convert_element_type_55 = None
        wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_20);  all_gather_into_tensor_20 = None
        view_44 = torch.ops.aten.view.default(wait_tensor_21, [16384, 8192]);  wait_tensor_21 = None
        mm_11 = torch.ops.aten.mm.default(view_44, permute_19)
        view_45 = torch.ops.aten.view.default(mm_11, [2, 8192, 14336]);  mm_11 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_59, sigmoid_1);  sigmoid_1 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_47 = torch.ops.aten.view.default(mm_12, [2, 8192, 14336]);  mm_12 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_60, view_47)
        view_48 = torch.ops.aten.view.default(mul_15, [16384, 14336]);  mul_15 = None
        mm_311 = torch.ops.aten.mm.default(permute_629, view_48);  permute_629 = view_48 = None
        permute_630 = torch.ops.aten.permute.default(mm_311, [1, 0]);  mm_311 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        permute_21 = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        clone_7 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        all_gather_into_tensor_22 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_7, 4, '1');  clone_7 = None
        wait_tensor_23 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_22);  all_gather_into_tensor_22 = None
        permute_631 = torch.ops.aten.permute.default(wait_tensor_23, [1, 0]);  wait_tensor_23 = None
        mm_312 = torch.ops.aten.mm.default(view_740, permute_631);  view_740 = permute_631 = None
        view_741 = torch.ops.aten.view.default(mm_312, [2, 8192, 14336]);  mm_312 = None
        clone_120 = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        reduce_scatter_tensor_159 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_120, 'sum', 4, '1');  clone_120 = None
        wait_tensor_340 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_159);  reduce_scatter_tensor_159 = None
        permute_632 = torch.ops.aten.permute.default(wait_tensor_340, [1, 0]);  wait_tensor_340 = None
        convert_element_type_1306 = torch.ops.prims.convert_element_type.default(permute_632, torch.float32);  permute_632 = None
        mul_416 = torch.ops.aten.mul.Tensor(view_741, convert_element_type_60);  convert_element_type_60 = None
        mul_417 = torch.ops.aten.mul.Tensor(view_741, view_47);  view_741 = view_47 = None
        view_742 = torch.ops.aten.view.default(mul_416, [16384, 14336]);  mul_416 = None
        permute_633 = torch.ops.aten.permute.default(view_742, [1, 0])
        mm_313 = torch.ops.aten.mm.default(permute_633, view_44);  permute_633 = None
        reduce_scatter_tensor_160 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_313, 'sum', 4, '1');  mm_313 = None
        wait_tensor_341 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_160);  reduce_scatter_tensor_160 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        all_gather_into_tensor_21 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_61, 4, '1');  convert_element_type_61 = None
        wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_21);  all_gather_into_tensor_21 = None
        permute_20 = torch.ops.aten.permute.default(wait_tensor_22, [1, 0]);  wait_tensor_22 = None
        permute_635 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_314 = torch.ops.aten.mm.default(view_742, permute_635);  view_742 = permute_635 = None
        view_743 = torch.ops.aten.view.default(mm_314, [2, 8192, 8192]);  mm_314 = None
        convert_element_type_1311 = torch.ops.prims.convert_element_type.default(wait_tensor_341, torch.float32);  wait_tensor_341 = None
        convert_element_type_1312 = torch.ops.prims.convert_element_type.default(mul_417, torch.float32);  mul_417 = None
        neg_14 = torch.ops.aten.neg.default(convert_element_type_59)
        exp_14 = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_163 = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_163);  add_163 = None
        mul_418 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_419 = torch.ops.aten.mul.Tensor(convert_element_type_1312, mul_418);  convert_element_type_1312 = None
        sub_43 = torch.ops.aten.sub.Tensor(1, mul_418);  mul_418 = None
        mul_420 = torch.ops.aten.mul.Tensor(convert_element_type_59, sub_43);  convert_element_type_59 = sub_43 = None
        add_164 = torch.ops.aten.add.Tensor(mul_420, 1);  mul_420 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_419, add_164);  mul_419 = add_164 = None
        convert_element_type_1314 = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16);  mul_421 = None
        view_744 = torch.ops.aten.view.default(convert_element_type_1314, [16384, 14336]);  convert_element_type_1314 = None
        permute_637 = torch.ops.aten.permute.default(view_744, [1, 0])
        mm_315 = torch.ops.aten.mm.default(permute_637, view_44);  permute_637 = view_44 = None
        reduce_scatter_tensor_161 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_315, 'sum', 4, '1');  mm_315 = None
        wait_tensor_342 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_161);  reduce_scatter_tensor_161 = None
        permute_639 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_316 = torch.ops.aten.mm.default(view_744, permute_639);  view_744 = permute_639 = None
        view_745 = torch.ops.aten.view.default(mm_316, [2, 8192, 8192]);  mm_316 = None
        add_165 = torch.ops.aten.add.Tensor(view_743, view_745);  view_743 = view_745 = None
        convert_element_type_1319 = torch.ops.prims.convert_element_type.default(wait_tensor_342, torch.float32);  wait_tensor_342 = None
        reduce_scatter_tensor_162 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_165, 'sum', 2, '3');  add_165 = None
        wait_tensor_343 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_162);  reduce_scatter_tensor_162 = None
        convert_element_type_1320 = torch.ops.prims.convert_element_type.default(wait_tensor_343, torch.float32);  wait_tensor_343 = None
        convert_element_type_1322 = torch.ops.prims.convert_element_type.default(wait_tensor_19, torch.float32);  wait_tensor_19 = None
        mul_422 = torch.ops.aten.mul.Tensor(convert_element_type_1320, convert_element_type_1322);  convert_element_type_1322 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_12, mul_422)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
        div_29 = torch.ops.aten.div.Tensor(mul_12, 8192)
        mul_425 = torch.ops.aten.mul.Tensor(div_29, sum_87);  div_29 = sum_87 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_422, mul_425);  mul_422 = mul_425 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_3);  sub_44 = rsqrt_3 = None
        mul_427 = torch.ops.aten.mul.Tensor(convert_element_type_1320, mul_12);  convert_element_type_1320 = mul_12 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
        convert_element_type_1323 = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        add_166 = torch.ops.aten.add.Tensor(add_162, convert_element_type_1323);  add_162 = convert_element_type_1323 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sum_88, torch.float32);  sum_88 = None
        reduce_scatter_tensor_163 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_4, 'sum', 8, '0');  convert_element_type_default_4 = None
        wait_tensor_344 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_163);  reduce_scatter_tensor_163 = None
        view_746 = torch.ops.aten.view.default(add_166, [8192, 8192])
        permute_641 = torch.ops.aten.permute.default(view_746, [1, 0])
        permute_17 = torch.ops.aten.permute.default(getitem_13, [0, 2, 1, 3])
        view_41 = torch.ops.aten.view.default(permute_17, [1, 8192, 8192]);  permute_17 = None
        view_42 = torch.ops.aten.view.default(view_41, [8192, 8192]);  view_41 = None
        mm_317 = torch.ops.aten.mm.default(permute_641, view_42);  permute_641 = view_42 = None
        permute_642 = torch.ops.aten.permute.default(mm_317, [1, 0]);  mm_317 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        permute_18 = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        clone_6 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        all_gather_into_tensor_17 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_6, 8, '0');  clone_6 = None
        wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_17);  all_gather_into_tensor_17 = None
        permute_643 = torch.ops.aten.permute.default(wait_tensor_18, [1, 0]);  wait_tensor_18 = None
        mm_318 = torch.ops.aten.mm.default(view_746, permute_643);  view_746 = permute_643 = None
        view_747 = torch.ops.aten.view.default(mm_318, [1, 8192, 8192]);  mm_318 = None
        clone_121 = torch.ops.aten.clone.default(permute_642, memory_format = torch.contiguous_format);  permute_642 = None
        reduce_scatter_tensor_164 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_121, 'sum', 8, '0');  clone_121 = None
        wait_tensor_345 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_164);  reduce_scatter_tensor_164 = None
        permute_644 = torch.ops.aten.permute.default(wait_tensor_345, [1, 0]);  wait_tensor_345 = None
        convert_element_type_1330 = torch.ops.prims.convert_element_type.default(permute_644, torch.float32);  permute_644 = None
        view_748 = torch.ops.aten.view.default(view_747, [1, 8192, 32, 256]);  view_747 = None
        permute_645 = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_4 = torch.ops.aten.add.Scalar(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_2);  convert_element_type_35 = None
        all_gather_into_tensor_13 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_34, 8, '0');  convert_element_type_34 = None
        wait_tensor_14 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_13);  all_gather_into_tensor_13 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, wait_tensor_14)
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        all_gather_into_tensor_14 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_37, 8, '0');  convert_element_type_37 = None
        wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        permute_11 = torch.ops.aten.permute.default(wait_tensor_15, [1, 0]);  wait_tensor_15 = None
        view_25 = torch.ops.aten.view.default(convert_element_type_36, [8192, 8192]);  convert_element_type_36 = None
        mm_7 = torch.ops.aten.mm.default(view_25, permute_11)
        view_26 = torch.ops.aten.view.default(mm_7, [1, 8192, 8192]);  mm_7 = None
        view_28 = torch.ops.aten.view.default(mm_8, [1, 8192, 2048]);  mm_8 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        all_gather_into_tensor_16 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_43, 8, '0');  convert_element_type_43 = None
        wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        permute_13 = torch.ops.aten.permute.default(wait_tensor_17, [1, 0]);  wait_tensor_17 = None
        mm_9 = torch.ops.aten.mm.default(view_25, permute_13)
        view_30 = torch.ops.aten.view.default(mm_9, [1, 8192, 2048]);  mm_9 = None
        view_31 = torch.ops.aten.view.default(view_26, [1, 8192, 32, 256]);  view_26 = None
        view_32 = torch.ops.aten.view.default(view_28, [1, 8192, 8, 256]);  view_28 = None
        view_33 = torch.ops.aten.view.default(view_30, [1, 8192, 8, 256]);  view_30 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_46, [1, 8192, 32, 128, 2]);  convert_element_type_46 = None
        view_as_complex_2 = torch.ops.aten.view_as_complex.default(view_34);  view_34 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        view_35 = torch.ops.aten.view.default(convert_element_type_47, [1, 8192, 8, 128, 2]);  convert_element_type_47 = None
        view_as_complex_3 = torch.ops.aten.view_as_complex.default(view_35);  view_35 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_as_complex_2, view_11);  view_as_complex_2 = None
        view_as_real_2 = torch.ops.aten.view_as_real.default(mul_10);  mul_10 = None
        view_37 = torch.ops.aten.view.default(view_as_real_2, [1, 8192, 32, 256]);  view_as_real_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_as_complex_3, view_11);  view_as_complex_3 = None
        view_as_real_3 = torch.ops.aten.view_as_real.default(mul_11);  mul_11 = None
        view_38 = torch.ops.aten.view.default(view_as_real_3, [1, 8192, 8, 256]);  view_as_real_3 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(view_37, torch.bfloat16);  view_37 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_49, 3);  convert_element_type_49 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_2, [1, 8192, 8, 4, 256]);  unsqueeze_2 = None
        clone_4 = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_39 = torch.ops.aten.view.default(clone_4, [1, 8192, 32, 256]);  clone_4 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_33, 3);  view_33 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [1, 8192, 8, 4, 256]);  unsqueeze_3 = None
        clone_5 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_40 = torch.ops.aten.view.default(clone_5, [1, 8192, 32, 256]);  clone_5 = None
        permute_14 = torch.ops.aten.permute.default(convert_element_type_48, [0, 2, 1, 3]);  convert_element_type_48 = None
        permute_15 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        permute_16 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_flash_attention_backward_14 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_645, permute_14, permute_15, permute_16, getitem_13, getitem_14, None, None, 8192, 8192, 0.0, True, getitem_19, getitem_20, scale = 0.0625);  permute_645 = permute_14 = permute_15 = permute_16 = getitem_13 = getitem_14 = getitem_19 = getitem_20 = None
        getitem_190 = _scaled_dot_product_flash_attention_backward_14[0]
        getitem_191 = _scaled_dot_product_flash_attention_backward_14[1]
        getitem_192 = _scaled_dot_product_flash_attention_backward_14[2];  _scaled_dot_product_flash_attention_backward_14 = None
        permute_646 = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3]);  getitem_192 = None
        permute_647 = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3]);  getitem_191 = None
        permute_648 = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None
        view_749 = torch.ops.aten.view.default(permute_646, [1, 8192, 8, 4, 256]);  permute_646 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(view_749, [3], True);  view_749 = None
        squeeze_28 = torch.ops.aten.squeeze.dim(sum_89, 3);  sum_89 = None
        view_750 = torch.ops.aten.view.default(permute_647, [1, 8192, 8, 4, 256]);  permute_647 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(view_750, [3], True);  view_750 = None
        squeeze_29 = torch.ops.aten.squeeze.dim(sum_90, 3);  sum_90 = None
        convert_element_type_1331 = torch.ops.prims.convert_element_type.default(squeeze_29, torch.float32);  squeeze_29 = None
        convert_element_type_1332 = torch.ops.prims.convert_element_type.default(permute_648, torch.float32);  permute_648 = None
        view_751 = torch.ops.aten.view.default(convert_element_type_1331, [1, 8192, 8, 128, 2]);  convert_element_type_1331 = None
        view_as_complex_60 = torch.ops.aten.view_as_complex.default(view_751);  view_751 = None
        mul_428 = torch.ops.aten.mul.Tensor(view_as_complex_60, _conj);  view_as_complex_60 = None
        view_752 = torch.ops.aten.view.default(convert_element_type_1332, [1, 8192, 32, 128, 2]);  convert_element_type_1332 = None
        view_as_complex_61 = torch.ops.aten.view_as_complex.default(view_752);  view_752 = None
        mul_429 = torch.ops.aten.mul.Tensor(view_as_complex_61, _conj);  view_as_complex_61 = None
        view_as_real_60 = torch.ops.aten.view_as_real.default(mul_428);  mul_428 = None
        view_753 = torch.ops.aten.view.default(view_as_real_60, [1, 8192, 8, 256]);  view_as_real_60 = None
        convert_element_type_1333 = torch.ops.prims.convert_element_type.default(view_753, torch.bfloat16);  view_753 = None
        view_as_real_61 = torch.ops.aten.view_as_real.default(mul_429);  mul_429 = None
        view_754 = torch.ops.aten.view.default(view_as_real_61, [1, 8192, 32, 256]);  view_as_real_61 = None
        convert_element_type_1334 = torch.ops.prims.convert_element_type.default(view_754, torch.bfloat16);  view_754 = None
        view_755 = torch.ops.aten.view.default(squeeze_28, [1, 8192, 2048]);  squeeze_28 = None
        view_756 = torch.ops.aten.view.default(convert_element_type_1333, [1, 8192, 2048]);  convert_element_type_1333 = None
        view_757 = torch.ops.aten.view.default(convert_element_type_1334, [1, 8192, 8192]);  convert_element_type_1334 = None
        view_758 = torch.ops.aten.view.default(view_755, [8192, 2048]);  view_755 = None
        permute_649 = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_319 = torch.ops.aten.mm.default(permute_649, view_25);  permute_649 = None
        permute_651 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_320 = torch.ops.aten.mm.default(view_758, permute_651);  view_758 = permute_651 = None
        view_759 = torch.ops.aten.view.default(mm_320, [1, 8192, 8192]);  mm_320 = None
        convert_element_type_1339 = torch.ops.prims.convert_element_type.default(mm_319, torch.float32);  mm_319 = None
        reduce_scatter_tensor_165 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1339, 'sum', 8, '0');  convert_element_type_1339 = None
        wait_tensor_346 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_165);  reduce_scatter_tensor_165 = None
        view_760 = torch.ops.aten.view.default(view_756, [8192, 2048]);  view_756 = None
        permute_653 = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_321 = torch.ops.aten.mm.default(permute_653, view_25);  permute_653 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        all_gather_into_tensor_15 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_40, 8, '0');  convert_element_type_40 = None
        wait_tensor_16 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        permute_12 = torch.ops.aten.permute.default(wait_tensor_16, [1, 0]);  wait_tensor_16 = None
        permute_655 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_322 = torch.ops.aten.mm.default(view_760, permute_655);  view_760 = permute_655 = None
        view_761 = torch.ops.aten.view.default(mm_322, [1, 8192, 8192]);  mm_322 = None
        add_167 = torch.ops.aten.add.Tensor(view_759, view_761);  view_759 = view_761 = None
        convert_element_type_1344 = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None
        reduce_scatter_tensor_166 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1344, 'sum', 8, '0');  convert_element_type_1344 = None
        wait_tensor_347 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_166);  reduce_scatter_tensor_166 = None
        view_762 = torch.ops.aten.view.default(view_757, [8192, 8192]);  view_757 = None
        permute_657 = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_323 = torch.ops.aten.mm.default(permute_657, view_25);  permute_657 = view_25 = None
        permute_659 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_324 = torch.ops.aten.mm.default(view_762, permute_659);  view_762 = permute_659 = None
        view_763 = torch.ops.aten.view.default(mm_324, [1, 8192, 8192]);  mm_324 = None
        add_168 = torch.ops.aten.add.Tensor(add_167, view_763);  add_167 = view_763 = None
        reduce_scatter_tensor_167 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_323, 'sum', 8, '0');  mm_323 = None
        wait_tensor_348 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_167);  reduce_scatter_tensor_167 = None
        convert_element_type_1349 = torch.ops.prims.convert_element_type.default(wait_tensor_348, torch.float32);  wait_tensor_348 = None
        convert_element_type_1350 = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        convert_element_type_1352 = torch.ops.prims.convert_element_type.default(wait_tensor_14, torch.float32);  wait_tensor_14 = None
        mul_430 = torch.ops.aten.mul.Tensor(convert_element_type_1350, convert_element_type_1352);  convert_element_type_1352 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_8, mul_430)
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
        div_30 = torch.ops.aten.div.Tensor(mul_8, 8192)
        mul_433 = torch.ops.aten.mul.Tensor(div_30, sum_91);  div_30 = sum_91 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_430, mul_433);  mul_430 = mul_433 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_2);  sub_45 = rsqrt_2 = None
        mul_435 = torch.ops.aten.mul.Tensor(convert_element_type_1350, mul_8);  convert_element_type_1350 = mul_8 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
        convert_element_type_1353 = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        add_169 = torch.ops.aten.add.Tensor(add_166, convert_element_type_1353);  add_166 = convert_element_type_1353 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sum_92, torch.float32);  sum_92 = None
        reduce_scatter_tensor_168 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_3, 'sum', 8, '0');  convert_element_type_default_3 = None
        wait_tensor_349 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_168);  reduce_scatter_tensor_168 = None
        view_764 = torch.ops.aten.view.default(add_169, [8192, 8192])
        all_gather_into_tensor_181 = torch.ops._c10d_functional.all_gather_into_tensor.default(view_764, 2, '3');  view_764 = None
        wait_tensor_350 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_181);  all_gather_into_tensor_181 = None
        permute_661 = torch.ops.aten.permute.default(wait_tensor_350, [1, 0])
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(primals_149, 4, '1');  primals_149 = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type, 2, '3');  convert_element_type = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        split = torch.ops.aten.split.Tensor(wait_tensor, 4)
        getitem = split[0];  split = None
        embedding = torch.ops.aten.embedding.default(wait_tensor_1, getitem);  wait_tensor_1 = getitem = None
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(embedding, 2, '3');  embedding = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        shard_dim_alltoall = torch.ops._dtensor.shard_dim_alltoall.default(wait_tensor_2, 2, 0, '1');  wait_tensor_2 = None
        split_1 = torch.ops.aten.split.Tensor(shard_dim_alltoall, 1);  shard_dim_alltoall = None
        getitem_2 = split_1[0];  split_1 = None
        permute_6 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3])
        view_16 = torch.ops.aten.view.default(permute_6, [1, 8192, 8192]);  permute_6 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        permute_7 = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        view_17 = torch.ops.aten.view.default(view_16, [8192, 8192]);  view_16 = None
        clone_2 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        all_gather_into_tensor_7 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_2, 8, '0');  clone_2 = None
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        mm_3 = torch.ops.aten.mm.default(view_17, wait_tensor_7)
        view_18 = torch.ops.aten.view.default(mm_3, [1, 8192, 8192]);  mm_3 = None
        add_1 = torch.ops.aten.add.Tensor(getitem_2, view_18);  view_18 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Scalar(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_1);  convert_element_type_21 = None
        all_gather_into_tensor_8 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_20, 8, '0');  convert_element_type_20 = None
        wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, wait_tensor_8)
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        all_gather_into_tensor_10 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_22, 2, '3');  convert_element_type_22 = None
        wait_tensor_10 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        view_19 = torch.ops.aten.view.default(wait_tensor_10, [16384, 8192]);  wait_tensor_10 = None
        view_20 = torch.ops.aten.view.default(mm_4, [2, 8192, 14336]);  mm_4 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_26)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_26, sigmoid);  sigmoid = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        all_gather_into_tensor_11 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_28, 4, '1');  convert_element_type_28 = None
        wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        permute_9 = torch.ops.aten.permute.default(wait_tensor_11, [1, 0]);  wait_tensor_11 = None
        mm_5 = torch.ops.aten.mm.default(view_19, permute_9)
        view_22 = torch.ops.aten.view.default(mm_5, [2, 8192, 14336]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_27, view_22)
        view_23 = torch.ops.aten.view.default(mul_7, [16384, 14336]);  mul_7 = None
        mm_325 = torch.ops.aten.mm.default(permute_661, view_23);  permute_661 = view_23 = None
        permute_662 = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        permute_10 = torch.ops.aten.permute.default(convert_element_type_31, [1, 0]);  convert_element_type_31 = None
        clone_3 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        all_gather_into_tensor_12 = torch.ops._c10d_functional.all_gather_into_tensor.default(clone_3, 4, '1');  clone_3 = None
        wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        permute_663 = torch.ops.aten.permute.default(wait_tensor_12, [1, 0]);  wait_tensor_12 = None
        mm_326 = torch.ops.aten.mm.default(wait_tensor_350, permute_663);  wait_tensor_350 = permute_663 = None
        view_765 = torch.ops.aten.view.default(mm_326, [2, 8192, 14336]);  mm_326 = None
        clone_124 = torch.ops.aten.clone.default(permute_662, memory_format = torch.contiguous_format);  permute_662 = None
        reduce_scatter_tensor_169 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_124, 'sum', 4, '1');  clone_124 = None
        wait_tensor_351 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_169);  reduce_scatter_tensor_169 = None
        permute_664 = torch.ops.aten.permute.default(wait_tensor_351, [1, 0]);  wait_tensor_351 = None
        convert_element_type_1360 = torch.ops.prims.convert_element_type.default(permute_664, torch.float32);  permute_664 = None
        mul_436 = torch.ops.aten.mul.Tensor(view_765, convert_element_type_27);  convert_element_type_27 = None
        mul_437 = torch.ops.aten.mul.Tensor(view_765, view_22);  view_765 = view_22 = None
        view_766 = torch.ops.aten.view.default(mul_436, [16384, 14336]);  mul_436 = None
        permute_665 = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_327 = torch.ops.aten.mm.default(permute_665, view_19);  permute_665 = None
        reduce_scatter_tensor_170 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_327, 'sum', 4, '1');  mm_327 = None
        wait_tensor_352 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_170);  reduce_scatter_tensor_170 = None
        permute_667 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_328 = torch.ops.aten.mm.default(view_766, permute_667);  view_766 = permute_667 = None
        view_767 = torch.ops.aten.view.default(mm_328, [2, 8192, 8192]);  mm_328 = None
        convert_element_type_1365 = torch.ops.prims.convert_element_type.default(wait_tensor_352, torch.float32);  wait_tensor_352 = None
        convert_element_type_1366 = torch.ops.prims.convert_element_type.default(mul_437, torch.float32);  mul_437 = None
        neg_15 = torch.ops.aten.neg.default(convert_element_type_26)
        exp_15 = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_170 = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(add_170);  add_170 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_439 = torch.ops.aten.mul.Tensor(convert_element_type_1366, mul_438);  convert_element_type_1366 = None
        sub_46 = torch.ops.aten.sub.Tensor(1, mul_438);  mul_438 = None
        mul_440 = torch.ops.aten.mul.Tensor(convert_element_type_26, sub_46);  convert_element_type_26 = sub_46 = None
        add_171 = torch.ops.aten.add.Tensor(mul_440, 1);  mul_440 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_171);  mul_439 = add_171 = None
        convert_element_type_1368 = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None
        view_768 = torch.ops.aten.view.default(convert_element_type_1368, [16384, 14336]);  convert_element_type_1368 = None
        permute_669 = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_329 = torch.ops.aten.mm.default(permute_669, view_19);  permute_669 = view_19 = None
        reduce_scatter_tensor_171 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_329, 'sum', 4, '1');  mm_329 = None
        wait_tensor_353 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_171);  reduce_scatter_tensor_171 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        all_gather_into_tensor_9 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_23, 4, '1');  convert_element_type_23 = None
        wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        permute_8 = torch.ops.aten.permute.default(wait_tensor_9, [1, 0]);  wait_tensor_9 = None
        permute_671 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_330 = torch.ops.aten.mm.default(view_768, permute_671);  view_768 = permute_671 = None
        view_769 = torch.ops.aten.view.default(mm_330, [2, 8192, 8192]);  mm_330 = None
        add_172 = torch.ops.aten.add.Tensor(view_767, view_769);  view_767 = view_769 = None
        convert_element_type_1373 = torch.ops.prims.convert_element_type.default(wait_tensor_353, torch.float32);  wait_tensor_353 = None
        reduce_scatter_tensor_172 = torch.ops._c10d_functional.reduce_scatter_tensor.default(add_172, 'sum', 2, '3');  add_172 = None
        wait_tensor_354 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_172);  reduce_scatter_tensor_172 = None
        convert_element_type_1374 = torch.ops.prims.convert_element_type.default(wait_tensor_354, torch.float32);  wait_tensor_354 = None
        convert_element_type_1376 = torch.ops.prims.convert_element_type.default(wait_tensor_8, torch.float32);  wait_tensor_8 = None
        mul_442 = torch.ops.aten.mul.Tensor(convert_element_type_1374, convert_element_type_1376);  convert_element_type_1376 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_4, mul_442)
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
        div_31 = torch.ops.aten.div.Tensor(mul_4, 8192)
        mul_445 = torch.ops.aten.mul.Tensor(div_31, sum_93);  div_31 = sum_93 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_442, mul_445);  mul_442 = mul_445 = None
        mul_446 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_1);  sub_47 = rsqrt_1 = None
        mul_447 = torch.ops.aten.mul.Tensor(convert_element_type_1374, mul_4);  convert_element_type_1374 = mul_4 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
        convert_element_type_1377 = torch.ops.prims.convert_element_type.default(mul_446, torch.bfloat16);  mul_446 = None
        add_173 = torch.ops.aten.add.Tensor(add_169, convert_element_type_1377);  add_169 = convert_element_type_1377 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sum_94, torch.float32);  sum_94 = None
        reduce_scatter_tensor_173 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_2, 'sum', 8, '0');  convert_element_type_default_2 = None
        wait_tensor_355 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_173);  reduce_scatter_tensor_173 = None
        view_770 = torch.ops.aten.view.default(add_173, [8192, 8192])
        permute_673 = torch.ops.aten.permute.default(view_770, [1, 0])
        mm_331 = torch.ops.aten.mm.default(permute_673, view_17);  permute_673 = view_17 = None
        permute_674 = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
        permute_675 = torch.ops.aten.permute.default(wait_tensor_7, [1, 0]);  wait_tensor_7 = None
        mm_332 = torch.ops.aten.mm.default(view_770, permute_675);  view_770 = permute_675 = None
        view_771 = torch.ops.aten.view.default(mm_332, [1, 8192, 8192]);  mm_332 = None
        clone_125 = torch.ops.aten.clone.default(permute_674, memory_format = torch.contiguous_format);  permute_674 = None
        reduce_scatter_tensor_174 = torch.ops._c10d_functional.reduce_scatter_tensor.default(clone_125, 'sum', 8, '0');  clone_125 = None
        wait_tensor_356 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_174);  reduce_scatter_tensor_174 = None
        permute_676 = torch.ops.aten.permute.default(wait_tensor_356, [1, 0]);  wait_tensor_356 = None
        convert_element_type_1384 = torch.ops.prims.convert_element_type.default(permute_676, torch.float32);  permute_676 = None
        view_772 = torch.ops.aten.view.default(view_771, [1, 8192, 32, 256]);  view_771 = None
        permute_677 = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = None
        all_gather_into_tensor_3 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 8, '0');  convert_element_type_1 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, wait_tensor_3)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view = torch.ops.aten.view.default(convert_element_type_3, [8192, 8192]);  convert_element_type_3 = None
        view_1 = torch.ops.aten.view.default(mm, [1, 8192, 8192]);  mm = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        all_gather_into_tensor_5 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_7, 8, '0');  convert_element_type_7 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        permute_1 = torch.ops.aten.permute.default(wait_tensor_5, [1, 0]);  wait_tensor_5 = None
        mm_1 = torch.ops.aten.mm.default(view, permute_1)
        view_3 = torch.ops.aten.view.default(mm_1, [1, 8192, 2048]);  mm_1 = None
        view_5 = torch.ops.aten.view.default(mm_2, [1, 8192, 2048]);  mm_2 = None
        view_6 = torch.ops.aten.view.default(view_1, [1, 8192, 32, 256]);  view_1 = None
        view_7 = torch.ops.aten.view.default(view_3, [1, 8192, 8, 256]);  view_3 = None
        view_8 = torch.ops.aten.view.default(view_5, [1, 8192, 8, 256]);  view_5 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        view_9 = torch.ops.aten.view.default(convert_element_type_13, [1, 8192, 32, 128, 2]);  convert_element_type_13 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_9);  view_9 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        view_10 = torch.ops.aten.view.default(convert_element_type_14, [1, 8192, 8, 128, 2]);  convert_element_type_14 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_10);  view_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_11);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_12 = torch.ops.aten.view.default(view_as_real, [1, 8192, 32, 256]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_11);  view_as_complex_1 = view_11 = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
        view_13 = torch.ops.aten.view.default(view_as_real_1, [1, 8192, 8, 256]);  view_as_real_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(view_13, torch.bfloat16);  view_13 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_16, 3);  convert_element_type_16 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [1, 8192, 8, 4, 256]);  unsqueeze = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_14 = torch.ops.aten.view.default(clone, [1, 8192, 32, 256]);  clone = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_8, 3);  view_8 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [1, 8192, 8, 4, 256]);  unsqueeze_1 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15 = torch.ops.aten.view.default(clone_1, [1, 8192, 32, 256]);  clone_1 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1, 3]);  convert_element_type_15 = None
        permute_4 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        permute_5 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        _scaled_dot_product_flash_attention_backward_15 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_677, permute_3, permute_4, permute_5, getitem_4, getitem_5, None, None, 8192, 8192, 0.0, True, getitem_10, getitem_11, scale = 0.0625);  permute_677 = permute_3 = permute_4 = permute_5 = getitem_4 = getitem_5 = getitem_10 = getitem_11 = None
        getitem_193 = _scaled_dot_product_flash_attention_backward_15[0]
        getitem_194 = _scaled_dot_product_flash_attention_backward_15[1]
        getitem_195 = _scaled_dot_product_flash_attention_backward_15[2];  _scaled_dot_product_flash_attention_backward_15 = None
        permute_678 = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
        permute_679 = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
        permute_680 = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None
        view_773 = torch.ops.aten.view.default(permute_678, [1, 8192, 8, 4, 256]);  permute_678 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(view_773, [3], True);  view_773 = None
        squeeze_30 = torch.ops.aten.squeeze.dim(sum_95, 3);  sum_95 = None
        view_774 = torch.ops.aten.view.default(permute_679, [1, 8192, 8, 4, 256]);  permute_679 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(view_774, [3], True);  view_774 = None
        squeeze_31 = torch.ops.aten.squeeze.dim(sum_96, 3);  sum_96 = None
        convert_element_type_1385 = torch.ops.prims.convert_element_type.default(squeeze_31, torch.float32);  squeeze_31 = None
        convert_element_type_1386 = torch.ops.prims.convert_element_type.default(permute_680, torch.float32);  permute_680 = None
        view_775 = torch.ops.aten.view.default(convert_element_type_1385, [1, 8192, 8, 128, 2]);  convert_element_type_1385 = None
        view_as_complex_62 = torch.ops.aten.view_as_complex.default(view_775);  view_775 = None
        mul_448 = torch.ops.aten.mul.Tensor(view_as_complex_62, _conj);  view_as_complex_62 = None
        view_776 = torch.ops.aten.view.default(convert_element_type_1386, [1, 8192, 32, 128, 2]);  convert_element_type_1386 = None
        view_as_complex_63 = torch.ops.aten.view_as_complex.default(view_776);  view_776 = None
        mul_449 = torch.ops.aten.mul.Tensor(view_as_complex_63, _conj);  view_as_complex_63 = _conj = None
        view_as_real_62 = torch.ops.aten.view_as_real.default(mul_448);  mul_448 = None
        view_777 = torch.ops.aten.view.default(view_as_real_62, [1, 8192, 8, 256]);  view_as_real_62 = None
        convert_element_type_1387 = torch.ops.prims.convert_element_type.default(view_777, torch.bfloat16);  view_777 = None
        view_as_real_63 = torch.ops.aten.view_as_real.default(mul_449);  mul_449 = None
        view_778 = torch.ops.aten.view.default(view_as_real_63, [1, 8192, 32, 256]);  view_as_real_63 = None
        convert_element_type_1388 = torch.ops.prims.convert_element_type.default(view_778, torch.bfloat16);  view_778 = None
        view_779 = torch.ops.aten.view.default(squeeze_30, [1, 8192, 2048]);  squeeze_30 = None
        view_780 = torch.ops.aten.view.default(convert_element_type_1387, [1, 8192, 2048]);  convert_element_type_1387 = None
        view_781 = torch.ops.aten.view.default(convert_element_type_1388, [1, 8192, 8192]);  convert_element_type_1388 = None
        view_782 = torch.ops.aten.view.default(view_779, [8192, 2048]);  view_779 = None
        permute_681 = torch.ops.aten.permute.default(view_782, [1, 0])
        mm_333 = torch.ops.aten.mm.default(permute_681, view);  permute_681 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        all_gather_into_tensor_6 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_10, 8, '0');  convert_element_type_10 = None
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        permute_2 = torch.ops.aten.permute.default(wait_tensor_6, [1, 0]);  wait_tensor_6 = None
        permute_683 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_334 = torch.ops.aten.mm.default(view_782, permute_683);  view_782 = permute_683 = None
        view_783 = torch.ops.aten.view.default(mm_334, [1, 8192, 8192]);  mm_334 = None
        convert_element_type_1393 = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None
        reduce_scatter_tensor_175 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1393, 'sum', 8, '0');  convert_element_type_1393 = None
        wait_tensor_357 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_175);  reduce_scatter_tensor_175 = None
        view_784 = torch.ops.aten.view.default(view_780, [8192, 2048]);  view_780 = None
        permute_685 = torch.ops.aten.permute.default(view_784, [1, 0])
        mm_335 = torch.ops.aten.mm.default(permute_685, view);  permute_685 = None
        permute_687 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_336 = torch.ops.aten.mm.default(view_784, permute_687);  view_784 = permute_687 = None
        view_785 = torch.ops.aten.view.default(mm_336, [1, 8192, 8192]);  mm_336 = None
        add_174 = torch.ops.aten.add.Tensor(view_783, view_785);  view_783 = view_785 = None
        convert_element_type_1398 = torch.ops.prims.convert_element_type.default(mm_335, torch.float32);  mm_335 = None
        reduce_scatter_tensor_176 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1398, 'sum', 8, '0');  convert_element_type_1398 = None
        wait_tensor_358 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_176);  reduce_scatter_tensor_176 = None
        view_786 = torch.ops.aten.view.default(view_781, [8192, 8192]);  view_781 = None
        permute_689 = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_337 = torch.ops.aten.mm.default(permute_689, view);  permute_689 = view = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        all_gather_into_tensor_4 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_4, 8, '0');  convert_element_type_4 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        permute = torch.ops.aten.permute.default(wait_tensor_4, [1, 0]);  wait_tensor_4 = None
        permute_691 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_338 = torch.ops.aten.mm.default(view_786, permute_691);  view_786 = permute_691 = None
        view_787 = torch.ops.aten.view.default(mm_338, [1, 8192, 8192]);  mm_338 = None
        add_175 = torch.ops.aten.add.Tensor(add_174, view_787);  add_174 = view_787 = None
        reduce_scatter_tensor_177 = torch.ops._c10d_functional.reduce_scatter_tensor.default(mm_337, 'sum', 8, '0');  mm_337 = None
        wait_tensor_359 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_177);  reduce_scatter_tensor_177 = None
        convert_element_type_1403 = torch.ops.prims.convert_element_type.default(wait_tensor_359, torch.float32);  wait_tensor_359 = None
        convert_element_type_1404 = torch.ops.prims.convert_element_type.default(add_175, torch.float32);  add_175 = None
        convert_element_type_1406 = torch.ops.prims.convert_element_type.default(wait_tensor_3, torch.float32);  wait_tensor_3 = None
        mul_450 = torch.ops.aten.mul.Tensor(convert_element_type_1404, convert_element_type_1406);  convert_element_type_1406 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul, mul_450)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_452, [2], True);  mul_452 = None
        div_32 = torch.ops.aten.div.Tensor(mul, 8192)
        mul_453 = torch.ops.aten.mul.Tensor(div_32, sum_97);  div_32 = sum_97 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_450, mul_453);  mul_450 = mul_453 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_48, rsqrt);  sub_48 = rsqrt = None
        mul_455 = torch.ops.aten.mul.Tensor(convert_element_type_1404, mul);  convert_element_type_1404 = mul = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1]);  mul_455 = None
        convert_element_type_1407 = torch.ops.prims.convert_element_type.default(mul_454, torch.bfloat16);  mul_454 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, convert_element_type_1407);  add_173 = convert_element_type_1407 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sum_98, torch.float32);  sum_98 = None
        reduce_scatter_tensor_178 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_1, 'sum', 8, '0');  convert_element_type_default_1 = None
        wait_tensor_360 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_178);  reduce_scatter_tensor_178 = None
        all_gather_into_tensor_182 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_176, 2, '3');  add_176 = None
        wait_tensor_361 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_182);  all_gather_into_tensor_182 = None
        shard_dim_alltoall_1 = torch.ops._dtensor.shard_dim_alltoall.default(wait_tensor_361, 0, 2, '1');  wait_tensor_361 = None
        convert_element_type_1410 = torch.ops.prims.convert_element_type.default(shard_dim_alltoall_1, torch.float32);  shard_dim_alltoall_1 = None
        eq = torch.ops.aten.eq.Scalar(wait_tensor, -1)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(unsqueeze_32, scalar_tensor, convert_element_type_1410);  unsqueeze_32 = scalar_tensor = convert_element_type_1410 = None
        full = torch.ops.aten.full.default([128256, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full, [wait_tensor], where, True);  full = wait_tensor = where = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(index_put, torch.float32);  index_put = None
        split_2 = torch.ops.aten.split.Tensor(convert_element_type_default, 64128);  convert_element_type_default = None
        getitem_196 = split_2[0];  split_2 = None
        return (getitem_196, wait_tensor_360, convert_element_type_1403, wait_tensor_358, wait_tensor_357, convert_element_type_1384, wait_tensor_355, convert_element_type_1373, convert_element_type_1365, convert_element_type_1360, wait_tensor_349, convert_element_type_1349, wait_tensor_347, wait_tensor_346, convert_element_type_1330, wait_tensor_344, convert_element_type_1319, convert_element_type_1311, convert_element_type_1306, wait_tensor_338, convert_element_type_1295, wait_tensor_336, wait_tensor_335, convert_element_type_1276, wait_tensor_333, convert_element_type_1265, convert_element_type_1257, convert_element_type_1252, wait_tensor_327, convert_element_type_1241, wait_tensor_325, wait_tensor_324, convert_element_type_1222, wait_tensor_322, convert_element_type_1211, convert_element_type_1203, convert_element_type_1198, wait_tensor_316, convert_element_type_1187, wait_tensor_314, wait_tensor_313, convert_element_type_1168, wait_tensor_311, convert_element_type_1157, convert_element_type_1149, convert_element_type_1144, wait_tensor_305, convert_element_type_1133, wait_tensor_303, wait_tensor_302, convert_element_type_1114, wait_tensor_300, convert_element_type_1103, convert_element_type_1095, convert_element_type_1090, wait_tensor_294, convert_element_type_1079, wait_tensor_292, wait_tensor_291, convert_element_type_1060, wait_tensor_289, convert_element_type_1049, convert_element_type_1041, convert_element_type_1036, wait_tensor_283, convert_element_type_1025, wait_tensor_281, wait_tensor_280, convert_element_type_1006, wait_tensor_278, convert_element_type_995, convert_element_type_987, convert_element_type_982, wait_tensor_272, convert_element_type_971, wait_tensor_270, wait_tensor_269, convert_element_type_952, wait_tensor_267, convert_element_type_941, convert_element_type_933, convert_element_type_928, wait_tensor_261, convert_element_type_917, wait_tensor_259, wait_tensor_258, convert_element_type_898, wait_tensor_256, convert_element_type_887, convert_element_type_879, convert_element_type_874, wait_tensor_250, convert_element_type_863, wait_tensor_248, wait_tensor_247, convert_element_type_844, wait_tensor_245, convert_element_type_833, convert_element_type_825, convert_element_type_820, wait_tensor_239, convert_element_type_809, wait_tensor_237, wait_tensor_236, convert_element_type_790, wait_tensor_234, convert_element_type_779, convert_element_type_771, convert_element_type_766, wait_tensor_228, convert_element_type_755, wait_tensor_226, wait_tensor_225, convert_element_type_736, wait_tensor_223, convert_element_type_725, convert_element_type_717, convert_element_type_712, wait_tensor_217, convert_element_type_701, wait_tensor_215, wait_tensor_214, convert_element_type_682, wait_tensor_212, convert_element_type_671, convert_element_type_663, convert_element_type_658, wait_tensor_206, convert_element_type_647, wait_tensor_204, wait_tensor_203, convert_element_type_628, wait_tensor_201, convert_element_type_617, convert_element_type_609, convert_element_type_604, wait_tensor_195, convert_element_type_593, wait_tensor_193, wait_tensor_192, convert_element_type_574, wait_tensor_190, convert_element_type_563, convert_element_type_555, convert_element_type_550, wait_tensor_184, convert_element_type_539, None, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 525336576, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64128, 2048), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1024,), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024, 8192), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf3, (256, 8192), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256, 8192), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf5, (8192, 1024), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1024,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf7, (3584, 8192), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (3584, 8192), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (8192, 3584), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024, 8192), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256, 8192), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256, 8192), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf14, (8192, 1024), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1024,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3584, 8192), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (3584, 8192), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (8192, 3584), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024, 8192), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 8192), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256, 8192), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf23, (8192, 1024), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1024,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (3584, 8192), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (3584, 8192), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (8192, 3584), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1024,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1024, 8192), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256, 8192), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256, 8192), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf32, (8192, 1024), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1024,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf34, (3584, 8192), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3584, 8192), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf36, (8192, 3584), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1024, 8192), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 8192), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256, 8192), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf41, (8192, 1024), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024,), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf43, (3584, 8192), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf44, (3584, 8192), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (8192, 3584), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024, 8192), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256, 8192), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256, 8192), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf50, (8192, 1024), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024,), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3584, 8192), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (3584, 8192), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf54, (8192, 3584), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf55, (1024,), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf56, (1024, 8192), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256, 8192), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256, 8192), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf59, (8192, 1024), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024,), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (3584, 8192), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3584, 8192), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (8192, 3584), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf64, (1024,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf65, (1024, 8192), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256, 8192), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256, 8192), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf68, (8192, 1024), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (3584, 8192), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf71, (3584, 8192), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf72, (8192, 3584), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1024,), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf74, (1024, 8192), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256, 8192), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256, 8192), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf77, (8192, 1024), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (3584, 8192), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (3584, 8192), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf81, (8192, 3584), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1024,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1024, 8192), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256, 8192), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256, 8192), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf86, (8192, 1024), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1024,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf88, (3584, 8192), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (3584, 8192), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (8192, 3584), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1024,), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1024, 8192), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256, 8192), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256, 8192), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf95, (8192, 1024), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1024,), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (3584, 8192), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (3584, 8192), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (8192, 3584), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1024,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024, 8192), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256, 8192), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256, 8192), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf104, (8192, 1024), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1024,), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf106, (3584, 8192), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (3584, 8192), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (8192, 3584), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024,), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024, 8192), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 8192), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256, 8192), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf113, (8192, 1024), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf114, (1024,), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (3584, 8192), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (3584, 8192), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf117, (8192, 3584), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1024,), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf119, (1024, 8192), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256, 8192), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 8192), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf122, (8192, 1024), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1024,), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf124, (3584, 8192), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf125, (3584, 8192), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf126, (8192, 3584), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1024,), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf128, (1024, 8192), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256, 8192), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256, 8192), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf131, (8192, 1024), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf133, (3584, 8192), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3584, 8192), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf135, (8192, 3584), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024,), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024, 8192), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf138, (256, 8192), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf139, (256, 8192), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf140, (8192, 1024), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024,), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf142, (3584, 8192), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3584, 8192), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 117440512, device=device(type='cuda', index=0))
    reader.tensor(buf144, (8192, 3584), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1024,), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 525336576, device=device(type='cuda', index=0))
    reader.tensor(buf146, (16032, 8192), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf147, (8192, 128), dtype=torch.complex64, is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf148, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf149, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf150 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf150, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf151 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf151, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_4
    buf152 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1, 32, 8192), is_leaf=True)  # getitem_5
    buf153 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf153, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_10
    buf154 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf154, (), dtype=torch.uint64, is_leaf=True)  # getitem_11
    buf155 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf155, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf156 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf156, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_3
    buf157 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf157, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_8
    buf158 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf158, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_13
    buf159 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1, 32, 8192), is_leaf=True)  # getitem_14
    buf160 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf160, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_19
    buf161 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf161, (), dtype=torch.uint64, is_leaf=True)  # getitem_20
    buf162 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf162, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_10
    buf163 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf163, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_12
    buf164 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf164, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_7
    buf165 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf165, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_14
    buf166 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf166, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_16
    buf167 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf167, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_22
    buf168 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1, 32, 8192), is_leaf=True)  # getitem_23
    buf169 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf169, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_28
    buf170 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf170, (), dtype=torch.uint64, is_leaf=True)  # getitem_29
    buf171 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf171, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_18
    buf172 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf172, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_11
    buf173 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf173, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_22
    buf174 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf174, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_31
    buf175 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1, 32, 8192), is_leaf=True)  # getitem_32
    buf176 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf176, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_37
    buf177 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf177, (), dtype=torch.uint64, is_leaf=True)  # getitem_38
    buf178 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf178, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_24
    buf179 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf179, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_26
    buf180 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf180, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_15
    buf181 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf181, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_28
    buf182 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf182, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_30
    buf183 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf183, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_40
    buf184 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1, 32, 8192), is_leaf=True)  # getitem_41
    buf185 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf185, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_46
    buf186 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf186, (), dtype=torch.uint64, is_leaf=True)  # getitem_47
    buf187 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf187, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_32
    buf188 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf188, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_34
    buf189 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf189, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_36
    buf190 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf190, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_49
    buf191 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1, 32, 8192), is_leaf=True)  # getitem_50
    buf192 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf192, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_55
    buf193 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf193, (), dtype=torch.uint64, is_leaf=True)  # getitem_56
    buf194 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf194, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_21
    buf195 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf195, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_40
    buf196 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf196, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_42
    buf197 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf197, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_44
    buf198 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf198, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_58
    buf199 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1, 32, 8192), is_leaf=True)  # getitem_59
    buf200 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf200, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_64
    buf201 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf201, (), dtype=torch.uint64, is_leaf=True)  # getitem_65
    buf202 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf202, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_25
    buf203 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf203, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_46
    buf204 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf204, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_48
    buf205 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf205, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_50
    buf206 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf206, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_67
    buf207 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1, 32, 8192), is_leaf=True)  # getitem_68
    buf208 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf208, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_73
    buf209 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf209, (), dtype=torch.uint64, is_leaf=True)  # getitem_74
    buf210 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf210, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_29
    buf211 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf211, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_54
    buf212 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf212, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_56
    buf213 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf213, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_58
    buf214 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf214, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_76
    buf215 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1, 32, 8192), is_leaf=True)  # getitem_77
    buf216 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf216, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_82
    buf217 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf217, (), dtype=torch.uint64, is_leaf=True)  # getitem_83
    buf218 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf218, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_33
    buf219 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf219, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_60
    buf220 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf220, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_62
    buf221 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf221, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_64
    buf222 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf222, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_85
    buf223 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1, 32, 8192), is_leaf=True)  # getitem_86
    buf224 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf224, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_91
    buf225 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf225, (), dtype=torch.uint64, is_leaf=True)  # getitem_92
    buf226 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf226, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_37
    buf227 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf227, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_68
    buf228 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf228, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_70
    buf229 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf229, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_72
    buf230 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf230, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_94
    buf231 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf231, (1, 32, 8192), is_leaf=True)  # getitem_95
    buf232 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf232, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_100
    buf233 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf233, (), dtype=torch.uint64, is_leaf=True)  # getitem_101
    buf234 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf234, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_41
    buf235 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf235, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_74
    buf236 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf236, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_76
    buf237 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf237, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_78
    buf238 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf238, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_103
    buf239 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf239, (1, 32, 8192), is_leaf=True)  # getitem_104
    buf240 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf240, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_109
    buf241 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf241, (), dtype=torch.uint64, is_leaf=True)  # getitem_110
    buf242 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf242, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_45
    buf243 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf243, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_82
    buf244 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf244, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_84
    buf245 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf245, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_86
    buf246 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf246, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_112
    buf247 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf247, (1, 32, 8192), is_leaf=True)  # getitem_113
    buf248 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf248, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_118
    buf249 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf249, (), dtype=torch.uint64, is_leaf=True)  # getitem_119
    buf250 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf250, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_49
    buf251 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf251, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_88
    buf252 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf252, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_90
    buf253 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf253, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_92
    buf254 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf254, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_121
    buf255 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf255, (1, 32, 8192), is_leaf=True)  # getitem_122
    buf256 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf256, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_127
    buf257 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf257, (), dtype=torch.uint64, is_leaf=True)  # getitem_128
    buf258 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf258, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_53
    buf259 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf259, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_96
    buf260 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf260, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_98
    buf261 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf261, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_100
    buf262 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf262, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_130
    buf263 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1, 32, 8192), is_leaf=True)  # getitem_131
    buf264 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf264, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_136
    buf265 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf265, (), dtype=torch.uint64, is_leaf=True)  # getitem_137
    buf266 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf266, (1, 8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # add_57
    buf267 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf267, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_102
    buf268 = reader.storage(None, 268435456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf268, (16384, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_104
    buf269 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf269, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_106
    buf270 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf270, (1, 32, 8192, 256), (67108864, 256, 8192, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_139
    buf271 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1, 32, 8192), is_leaf=True)  # getitem_140
    buf272 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf272, (2,), dtype=torch.uint64, is_leaf=True)  # getitem_145
    buf273 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.uint64)
    reader.tensor(buf273, (), dtype=torch.uint64, is_leaf=True)  # getitem_146
    buf274 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf274, (8192, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_108
    buf275 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf275, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_110
    buf276 = reader.storage(None, 2101346304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf276, (2, 8192, 64128), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()

if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    from torch.distributed.tensor import DeviceMesh, Shard
    import torch
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
    from torch.distributed.distributed_c10d import _get_group_tag, _register_process_group

    from torch.distributed.device_mesh import init_device_mesh
    from torch.fx.experimental.proxy_tensor import make_fx

    world_size = 8
    tp_size = 2

    store = dist.HashStore()
    dist.init_process_group(
        backend="fake", rank=0, world_size=world_size, store=store
    )
    device_type = "cuda"

    device_mesh = DeviceMesh(
        "cuda", torch.arange(0, world_size).view(-1, tp_size)
    )
    device_mesh = init_device_mesh(
        device_type, (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
    )

    with torch.no_grad():
        mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        mod(*args)
        gm = make_fx(mod)(*args)
        from autoparallel.tools.bubble_estimator import BubbleEstimatorMode
        with BubbleEstimatorMode(
            use_latencies_from_profile_traces = ["/data/users/ivankobzarev/b/autoparallel/tests/test_bubble_estimator_profile.json"],
            visualize=False, 
            chrome_trace_path="trace-nooverlap.json"
        ) as mode:
            fake_args = mode.convert_args(args)
            gm(*fake_args)
        from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
        schedule_overlap_bucketing(gm)
        with BubbleEstimatorMode(visualize=False, chrome_trace_path="trace-overlap.json") as mode:
            fake_args = mode.convert_args(args)
            gm(*fake_args)

    dist.destroy_process_group()
