import torch
from torch.nn import *
from torch import tensor, device


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16)
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type, 64, '0');  convert_element_type = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        embedding = torch.ops.aten.embedding.default(wait_tensor, primals_2);  wait_tensor = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16)
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 64, '0');  convert_element_type_1 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, wait_tensor_1);  mul = wait_tensor_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16)
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_4, 64, '0');  convert_element_type_4 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        permute = torch.ops.aten.permute.default(wait_tensor_2, [1, 0]);  wait_tensor_2 = None
        view_3 = torch.ops.aten.view.default(convert_element_type_3, [16384, 4096]);  convert_element_type_3 = None
        mm = torch.ops.aten.mm.default(view_3, permute);  permute = None
        view_4 = torch.ops.aten.view.default(mm, [2, 8192, 4096])
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16)
        all_gather_into_tensor_3 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_7, 64, '0');  convert_element_type_7 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        permute_1 = torch.ops.aten.permute.default(wait_tensor_3, [1, 0]);  wait_tensor_3 = None
        mm_1 = torch.ops.aten.mm.default(view_3, permute_1);  permute_1 = None
        view_7 = torch.ops.aten.view.default(mm_1, [2, 8192, 1024]);  mm_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16)
        all_gather_into_tensor_4 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_10, 64, '0');  convert_element_type_10 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        permute_2 = torch.ops.aten.permute.default(wait_tensor_4, [1, 0]);  wait_tensor_4 = None
        mm_2 = torch.ops.aten.mm.default(view_3, permute_2);  view_3 = permute_2 = None
        view_10 = torch.ops.aten.view.default(mm_2, [2, 8192, 1024])
        view_11 = torch.ops.aten.view.default(view_4, [2, 8192, -1, 128]);  view_4 = None
        view_12 = torch.ops.aten.view.default(view_7, [2, 8192, -1, 128]);  view_7 = None
        view_13 = torch.ops.aten.view.default(view_10, [2, 8192, -1, 128]);  view_10 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_13, [2, 8192, 32, -1, 2]);  convert_element_type_13 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_14);  view_14 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        view_15 = torch.ops.aten.view.default(convert_element_type_14, [2, 8192, 8, -1, 2]);  convert_element_type_14 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_15);  view_15 = None
        view_16 = torch.ops.aten.view.default(primals_3, [1, 8192, 1, 64])
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_16);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_17 = torch.ops.aten.view.default(view_as_real, [2, 8192, 32, 128]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_16);  view_as_complex_1 = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
        view_18 = torch.ops.aten.view.default(view_as_real_1, [2, 8192, 8, 128]);  view_as_real_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16);  view_17 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_16, 3);  convert_element_type_16 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [2, 8192, 8, 4, 128]);  unsqueeze = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_19 = torch.ops.aten.view.default(clone, [2, 8192, 32, 128]);  clone = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_13, 3);  view_13 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [2, 8192, 8, 4, 128]);  unsqueeze_1 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_20 = torch.ops.aten.view.default(clone_1, [2, 8192, 32, 128]);  clone_1 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1, 3]);  convert_element_type_15 = None
        permute_4 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        permute_5 = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_3, permute_4, permute_5, None, True, 0.0, True);  permute_3 = permute_4 = permute_5 = None
        getitem = _scaled_dot_product_cudnn_attention[0]
        getitem_1 = _scaled_dot_product_cudnn_attention[1]
        getitem_6 = _scaled_dot_product_cudnn_attention[6]
        getitem_7 = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])
        view_21 = torch.ops.aten.view.default(permute_6, [2, 8192, -1]);  permute_6 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16)
        all_gather_into_tensor_5 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_17, 64, '0');  convert_element_type_17 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        permute_7 = torch.ops.aten.permute.default(wait_tensor_5, [1, 0]);  wait_tensor_5 = None
        view_23 = torch.ops.aten.view.default(view_21, [16384, 4096]);  view_21 = None
        mm_3 = torch.ops.aten.mm.default(view_23, permute_7);  view_23 = permute_7 = None
        view_24 = torch.ops.aten.view.default(mm_3, [2, 8192, 4096]);  mm_3 = None
        add_1 = torch.ops.aten.add.Tensor(embedding, view_24);  view_24 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16)
        all_gather_into_tensor_6 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_20, 64, '0');  convert_element_type_20 = None
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Scalar(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_1);  convert_element_type_21 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, wait_tensor_6);  mul_4 = wait_tensor_6 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16)
        all_gather_into_tensor_7 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_23, 64, '0');  convert_element_type_23 = None
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        permute_8 = torch.ops.aten.permute.default(wait_tensor_7, [1, 0]);  wait_tensor_7 = None
        view_27 = torch.ops.aten.view.default(convert_element_type_22, [16384, 4096]);  convert_element_type_22 = None
        mm_4 = torch.ops.aten.mm.default(view_27, permute_8);  permute_8 = None
        view_28 = torch.ops.aten.view.default(mm_4, [2, 8192, 14336])
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_26)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_26, sigmoid);  convert_element_type_26 = sigmoid = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16)
        all_gather_into_tensor_8 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_28, 64, '0');  convert_element_type_28 = None
        wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        permute_9 = torch.ops.aten.permute.default(wait_tensor_8, [1, 0]);  wait_tensor_8 = None
        mm_5 = torch.ops.aten.mm.default(view_27, permute_9);  view_27 = permute_9 = None
        view_31 = torch.ops.aten.view.default(mm_5, [2, 8192, 14336]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_27, view_31);  convert_element_type_27 = view_31 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16)
        all_gather_into_tensor_9 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_31, 64, '0');  convert_element_type_31 = None
        wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        permute_10 = torch.ops.aten.permute.default(wait_tensor_9, [1, 0]);  wait_tensor_9 = None
        view_33 = torch.ops.aten.view.default(mul_7, [16384, 14336]);  mul_7 = None
        mm_6 = torch.ops.aten.mm.default(view_33, permute_10);  view_33 = permute_10 = None
        view_34 = torch.ops.aten.view.default(mm_6, [2, 8192, 4096]);  mm_6 = None
        add_3 = torch.ops.aten.add.Tensor(add_1, view_34);  add_1 = view_34 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16)
        all_gather_into_tensor_10 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_34, 64, '0');  convert_element_type_34 = None
        wait_tensor_10 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_4 = torch.ops.aten.add.Scalar(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_2);  convert_element_type_35 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, wait_tensor_10);  mul_8 = wait_tensor_10 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16)
        all_gather_into_tensor_11 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_37, 64, '0');  convert_element_type_37 = None
        wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        permute_11 = torch.ops.aten.permute.default(wait_tensor_11, [1, 0]);  wait_tensor_11 = None
        view_37 = torch.ops.aten.view.default(convert_element_type_36, [16384, 4096]);  convert_element_type_36 = None
        mm_7 = torch.ops.aten.mm.default(view_37, permute_11);  permute_11 = None
        view_38 = torch.ops.aten.view.default(mm_7, [2, 8192, 4096])
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        all_gather_into_tensor_12 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_40, 64, '0');  convert_element_type_40 = None
        wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        permute_12 = torch.ops.aten.permute.default(wait_tensor_12, [1, 0]);  wait_tensor_12 = None
        mm_8 = torch.ops.aten.mm.default(view_37, permute_12);  permute_12 = None
        view_41 = torch.ops.aten.view.default(mm_8, [2, 8192, 1024]);  mm_8 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16)
        all_gather_into_tensor_13 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_43, 64, '0');  convert_element_type_43 = None
        wait_tensor_13 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_13);  all_gather_into_tensor_13 = None
        permute_13 = torch.ops.aten.permute.default(wait_tensor_13, [1, 0]);  wait_tensor_13 = None
        mm_9 = torch.ops.aten.mm.default(view_37, permute_13);  view_37 = permute_13 = None
        view_44 = torch.ops.aten.view.default(mm_9, [2, 8192, 1024])
        view_45 = torch.ops.aten.view.default(view_38, [2, 8192, -1, 128]);  view_38 = None
        view_46 = torch.ops.aten.view.default(view_41, [2, 8192, -1, 128]);  view_41 = None
        view_47 = torch.ops.aten.view.default(view_44, [2, 8192, -1, 128]);  view_44 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_46, [2, 8192, 32, -1, 2]);  convert_element_type_46 = None
        view_as_complex_2 = torch.ops.aten.view_as_complex.default(view_48);  view_48 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        view_49 = torch.ops.aten.view.default(convert_element_type_47, [2, 8192, 8, -1, 2]);  convert_element_type_47 = None
        view_as_complex_3 = torch.ops.aten.view_as_complex.default(view_49);  view_49 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_as_complex_2, view_16);  view_as_complex_2 = None
        view_as_real_2 = torch.ops.aten.view_as_real.default(mul_10);  mul_10 = None
        view_51 = torch.ops.aten.view.default(view_as_real_2, [2, 8192, 32, 128]);  view_as_real_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_as_complex_3, view_16);  view_as_complex_3 = None
        view_as_real_3 = torch.ops.aten.view_as_real.default(mul_11);  mul_11 = None
        view_52 = torch.ops.aten.view.default(view_as_real_3, [2, 8192, 8, 128]);  view_as_real_3 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(view_51, torch.bfloat16);  view_51 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(view_52, torch.bfloat16);  view_52 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_49, 3);  convert_element_type_49 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_2, [2, 8192, 8, 4, 128]);  unsqueeze_2 = None
        clone_2 = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_53 = torch.ops.aten.view.default(clone_2, [2, 8192, 32, 128]);  clone_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_47, 3);  view_47 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [2, 8192, 8, 4, 128]);  unsqueeze_3 = None
        clone_3 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_54 = torch.ops.aten.view.default(clone_3, [2, 8192, 32, 128]);  clone_3 = None
        permute_14 = torch.ops.aten.permute.default(convert_element_type_48, [0, 2, 1, 3]);  convert_element_type_48 = None
        permute_15 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_16 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_14, permute_15, permute_16, None, True, 0.0, True);  permute_14 = permute_15 = permute_16 = None
        getitem_9 = _scaled_dot_product_cudnn_attention_1[0]
        getitem_10 = _scaled_dot_product_cudnn_attention_1[1]
        getitem_15 = _scaled_dot_product_cudnn_attention_1[6]
        getitem_16 = _scaled_dot_product_cudnn_attention_1[7];  _scaled_dot_product_cudnn_attention_1 = None
        permute_17 = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3])
        view_55 = torch.ops.aten.view.default(permute_17, [2, 8192, -1]);  permute_17 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16)
        all_gather_into_tensor_14 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_50, 64, '0');  convert_element_type_50 = None
        wait_tensor_14 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        permute_18 = torch.ops.aten.permute.default(wait_tensor_14, [1, 0]);  wait_tensor_14 = None
        view_57 = torch.ops.aten.view.default(view_55, [16384, 4096]);  view_55 = None
        mm_10 = torch.ops.aten.mm.default(view_57, permute_18);  view_57 = permute_18 = None
        view_58 = torch.ops.aten.view.default(mm_10, [2, 8192, 4096]);  mm_10 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, view_58);  view_58 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16)
        all_gather_into_tensor_15 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_53, 64, '0');  convert_element_type_53 = None
        wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_6 = torch.ops.aten.add.Scalar(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_3);  convert_element_type_54 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, wait_tensor_15);  mul_12 = wait_tensor_15 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16)
        all_gather_into_tensor_16 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_56, 64, '0');  convert_element_type_56 = None
        wait_tensor_16 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        permute_19 = torch.ops.aten.permute.default(wait_tensor_16, [1, 0]);  wait_tensor_16 = None
        view_61 = torch.ops.aten.view.default(convert_element_type_55, [16384, 4096]);  convert_element_type_55 = None
        mm_11 = torch.ops.aten.mm.default(view_61, permute_19);  permute_19 = None
        view_62 = torch.ops.aten.view.default(mm_11, [2, 8192, 14336])
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_59, sigmoid_1);  convert_element_type_59 = sigmoid_1 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16)
        all_gather_into_tensor_17 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_61, 64, '0');  convert_element_type_61 = None
        wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_17);  all_gather_into_tensor_17 = None
        permute_20 = torch.ops.aten.permute.default(wait_tensor_17, [1, 0]);  wait_tensor_17 = None
        mm_12 = torch.ops.aten.mm.default(view_61, permute_20);  view_61 = permute_20 = None
        view_65 = torch.ops.aten.view.default(mm_12, [2, 8192, 14336]);  mm_12 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_60, view_65);  convert_element_type_60 = view_65 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16)
        all_gather_into_tensor_18 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_64, 64, '0');  convert_element_type_64 = None
        wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_18);  all_gather_into_tensor_18 = None
        permute_21 = torch.ops.aten.permute.default(wait_tensor_18, [1, 0]);  wait_tensor_18 = None
        view_67 = torch.ops.aten.view.default(mul_15, [16384, 14336]);  mul_15 = None
        mm_13 = torch.ops.aten.mm.default(view_67, permute_21);  view_67 = permute_21 = None
        view_68 = torch.ops.aten.view.default(mm_13, [2, 8192, 4096]);  mm_13 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, view_68);  add_5 = view_68 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16)
        all_gather_into_tensor_19 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_67, 64, '0');  convert_element_type_67 = None
        wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_19);  all_gather_into_tensor_19 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_68, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_8 = torch.ops.aten.add.Scalar(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_68, rsqrt_4);  convert_element_type_68 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, wait_tensor_19);  mul_16 = wait_tensor_19 = None
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16)
        all_gather_into_tensor_20 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_70, 64, '0');  convert_element_type_70 = None
        wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_20);  all_gather_into_tensor_20 = None
        permute_22 = torch.ops.aten.permute.default(wait_tensor_20, [1, 0]);  wait_tensor_20 = None
        view_71 = torch.ops.aten.view.default(convert_element_type_69, [16384, 4096]);  convert_element_type_69 = None
        mm_14 = torch.ops.aten.mm.default(view_71, permute_22);  permute_22 = None
        view_72 = torch.ops.aten.view.default(mm_14, [2, 8192, 4096])
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16)
        all_gather_into_tensor_21 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_73, 64, '0');  convert_element_type_73 = None
        wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_21);  all_gather_into_tensor_21 = None
        permute_23 = torch.ops.aten.permute.default(wait_tensor_21, [1, 0]);  wait_tensor_21 = None
        mm_15 = torch.ops.aten.mm.default(view_71, permute_23);  permute_23 = None
        view_75 = torch.ops.aten.view.default(mm_15, [2, 8192, 1024]);  mm_15 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16)
        all_gather_into_tensor_22 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_76, 64, '0');  convert_element_type_76 = None
        wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_22);  all_gather_into_tensor_22 = None
        permute_24 = torch.ops.aten.permute.default(wait_tensor_22, [1, 0]);  wait_tensor_22 = None
        mm_16 = torch.ops.aten.mm.default(view_71, permute_24);  view_71 = permute_24 = None
        view_78 = torch.ops.aten.view.default(mm_16, [2, 8192, 1024])
        view_79 = torch.ops.aten.view.default(view_72, [2, 8192, -1, 128]);  view_72 = None
        view_80 = torch.ops.aten.view.default(view_75, [2, 8192, -1, 128]);  view_75 = None
        view_81 = torch.ops.aten.view.default(view_78, [2, 8192, -1, 128]);  view_78 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        view_82 = torch.ops.aten.view.default(convert_element_type_79, [2, 8192, 32, -1, 2]);  convert_element_type_79 = None
        view_as_complex_4 = torch.ops.aten.view_as_complex.default(view_82);  view_82 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        view_83 = torch.ops.aten.view.default(convert_element_type_80, [2, 8192, 8, -1, 2]);  convert_element_type_80 = None
        view_as_complex_5 = torch.ops.aten.view_as_complex.default(view_83);  view_83 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_as_complex_4, view_16);  view_as_complex_4 = None
        view_as_real_4 = torch.ops.aten.view_as_real.default(mul_18);  mul_18 = None
        view_85 = torch.ops.aten.view.default(view_as_real_4, [2, 8192, 32, 128]);  view_as_real_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_as_complex_5, view_16);  view_as_complex_5 = None
        view_as_real_5 = torch.ops.aten.view_as_real.default(mul_19);  mul_19 = None
        view_86 = torch.ops.aten.view.default(view_as_real_5, [2, 8192, 8, 128]);  view_as_real_5 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(view_85, torch.bfloat16);  view_85 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_82, 3);  convert_element_type_82 = None
        expand_4 = torch.ops.aten.expand.default(unsqueeze_4, [2, 8192, 8, 4, 128]);  unsqueeze_4 = None
        clone_4 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_87 = torch.ops.aten.view.default(clone_4, [2, 8192, 32, 128]);  clone_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(view_81, 3);  view_81 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_5, [2, 8192, 8, 4, 128]);  unsqueeze_5 = None
        clone_5 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_88 = torch.ops.aten.view.default(clone_5, [2, 8192, 32, 128]);  clone_5 = None
        permute_25 = torch.ops.aten.permute.default(convert_element_type_81, [0, 2, 1, 3]);  convert_element_type_81 = None
        permute_26 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        permute_27 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_25, permute_26, permute_27, None, True, 0.0, True);  permute_25 = permute_26 = permute_27 = None
        getitem_18 = _scaled_dot_product_cudnn_attention_2[0]
        getitem_19 = _scaled_dot_product_cudnn_attention_2[1]
        getitem_24 = _scaled_dot_product_cudnn_attention_2[6]
        getitem_25 = _scaled_dot_product_cudnn_attention_2[7];  _scaled_dot_product_cudnn_attention_2 = None
        permute_28 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])
        view_89 = torch.ops.aten.view.default(permute_28, [2, 8192, -1]);  permute_28 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16)
        all_gather_into_tensor_23 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_83, 64, '0');  convert_element_type_83 = None
        wait_tensor_23 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_23);  all_gather_into_tensor_23 = None
        permute_29 = torch.ops.aten.permute.default(wait_tensor_23, [1, 0]);  wait_tensor_23 = None
        view_91 = torch.ops.aten.view.default(view_89, [16384, 4096]);  view_89 = None
        mm_17 = torch.ops.aten.mm.default(view_91, permute_29);  view_91 = permute_29 = None
        view_92 = torch.ops.aten.view.default(mm_17, [2, 8192, 4096]);  mm_17 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, view_92);  view_92 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16)
        all_gather_into_tensor_24 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_86, 64, '0');  convert_element_type_86 = None
        wait_tensor_24 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_24);  all_gather_into_tensor_24 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_10 = torch.ops.aten.add.Scalar(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_5);  convert_element_type_87 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, wait_tensor_24);  mul_20 = wait_tensor_24 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16)
        all_gather_into_tensor_25 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_89, 64, '0');  convert_element_type_89 = None
        wait_tensor_25 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_25);  all_gather_into_tensor_25 = None
        permute_30 = torch.ops.aten.permute.default(wait_tensor_25, [1, 0]);  wait_tensor_25 = None
        view_95 = torch.ops.aten.view.default(convert_element_type_88, [16384, 4096]);  convert_element_type_88 = None
        mm_18 = torch.ops.aten.mm.default(view_95, permute_30);  permute_30 = None
        view_96 = torch.ops.aten.view.default(mm_18, [2, 8192, 14336])
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_92)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_92, sigmoid_2);  convert_element_type_92 = sigmoid_2 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16)
        all_gather_into_tensor_26 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_94, 64, '0');  convert_element_type_94 = None
        wait_tensor_26 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_26);  all_gather_into_tensor_26 = None
        permute_31 = torch.ops.aten.permute.default(wait_tensor_26, [1, 0]);  wait_tensor_26 = None
        mm_19 = torch.ops.aten.mm.default(view_95, permute_31);  view_95 = permute_31 = None
        view_99 = torch.ops.aten.view.default(mm_19, [2, 8192, 14336]);  mm_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_93, view_99);  convert_element_type_93 = view_99 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16)
        all_gather_into_tensor_27 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_97, 64, '0');  convert_element_type_97 = None
        wait_tensor_27 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_27);  all_gather_into_tensor_27 = None
        permute_32 = torch.ops.aten.permute.default(wait_tensor_27, [1, 0]);  wait_tensor_27 = None
        view_101 = torch.ops.aten.view.default(mul_23, [16384, 14336]);  mul_23 = None
        mm_20 = torch.ops.aten.mm.default(view_101, permute_32);  view_101 = permute_32 = None
        view_102 = torch.ops.aten.view.default(mm_20, [2, 8192, 4096]);  mm_20 = None
        add_11 = torch.ops.aten.add.Tensor(add_9, view_102);  add_9 = view_102 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16)
        all_gather_into_tensor_28 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_100, 64, '0');  convert_element_type_100 = None
        wait_tensor_28 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_28);  all_gather_into_tensor_28 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_12 = torch.ops.aten.add.Scalar(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_6);  convert_element_type_101 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, wait_tensor_28);  mul_24 = wait_tensor_28 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16)
        all_gather_into_tensor_29 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_103, 64, '0');  convert_element_type_103 = None
        wait_tensor_29 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_29);  all_gather_into_tensor_29 = None
        permute_33 = torch.ops.aten.permute.default(wait_tensor_29, [1, 0]);  wait_tensor_29 = None
        view_105 = torch.ops.aten.view.default(convert_element_type_102, [16384, 4096]);  convert_element_type_102 = None
        mm_21 = torch.ops.aten.mm.default(view_105, permute_33);  permute_33 = None
        view_106 = torch.ops.aten.view.default(mm_21, [2, 8192, 4096])
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16)
        all_gather_into_tensor_30 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_106, 64, '0');  convert_element_type_106 = None
        wait_tensor_30 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_30);  all_gather_into_tensor_30 = None
        permute_34 = torch.ops.aten.permute.default(wait_tensor_30, [1, 0]);  wait_tensor_30 = None
        mm_22 = torch.ops.aten.mm.default(view_105, permute_34);  permute_34 = None
        view_109 = torch.ops.aten.view.default(mm_22, [2, 8192, 1024]);  mm_22 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16)
        all_gather_into_tensor_31 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_109, 64, '0');  convert_element_type_109 = None
        wait_tensor_31 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_31);  all_gather_into_tensor_31 = None
        permute_35 = torch.ops.aten.permute.default(wait_tensor_31, [1, 0]);  wait_tensor_31 = None
        mm_23 = torch.ops.aten.mm.default(view_105, permute_35);  view_105 = permute_35 = None
        view_112 = torch.ops.aten.view.default(mm_23, [2, 8192, 1024])
        view_113 = torch.ops.aten.view.default(view_106, [2, 8192, -1, 128]);  view_106 = None
        view_114 = torch.ops.aten.view.default(view_109, [2, 8192, -1, 128]);  view_109 = None
        view_115 = torch.ops.aten.view.default(view_112, [2, 8192, -1, 128]);  view_112 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        view_116 = torch.ops.aten.view.default(convert_element_type_112, [2, 8192, 32, -1, 2]);  convert_element_type_112 = None
        view_as_complex_6 = torch.ops.aten.view_as_complex.default(view_116);  view_116 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None
        view_117 = torch.ops.aten.view.default(convert_element_type_113, [2, 8192, 8, -1, 2]);  convert_element_type_113 = None
        view_as_complex_7 = torch.ops.aten.view_as_complex.default(view_117);  view_117 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_as_complex_6, view_16);  view_as_complex_6 = None
        view_as_real_6 = torch.ops.aten.view_as_real.default(mul_26);  mul_26 = None
        view_119 = torch.ops.aten.view.default(view_as_real_6, [2, 8192, 32, 128]);  view_as_real_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_as_complex_7, view_16);  view_as_complex_7 = None
        view_as_real_7 = torch.ops.aten.view_as_real.default(mul_27);  mul_27 = None
        view_120 = torch.ops.aten.view.default(view_as_real_7, [2, 8192, 8, 128]);  view_as_real_7 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(view_119, torch.bfloat16);  view_119 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(view_120, torch.bfloat16);  view_120 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(convert_element_type_115, 3);  convert_element_type_115 = None
        expand_6 = torch.ops.aten.expand.default(unsqueeze_6, [2, 8192, 8, 4, 128]);  unsqueeze_6 = None
        clone_6 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_121 = torch.ops.aten.view.default(clone_6, [2, 8192, 32, 128]);  clone_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_115, 3);  view_115 = None
        expand_7 = torch.ops.aten.expand.default(unsqueeze_7, [2, 8192, 8, 4, 128]);  unsqueeze_7 = None
        clone_7 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_122 = torch.ops.aten.view.default(clone_7, [2, 8192, 32, 128]);  clone_7 = None
        permute_36 = torch.ops.aten.permute.default(convert_element_type_114, [0, 2, 1, 3]);  convert_element_type_114 = None
        permute_37 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        permute_38 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_36, permute_37, permute_38, None, True, 0.0, True);  permute_36 = permute_37 = permute_38 = None
        getitem_27 = _scaled_dot_product_cudnn_attention_3[0]
        getitem_28 = _scaled_dot_product_cudnn_attention_3[1]
        getitem_33 = _scaled_dot_product_cudnn_attention_3[6]
        getitem_34 = _scaled_dot_product_cudnn_attention_3[7];  _scaled_dot_product_cudnn_attention_3 = None
        permute_39 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_123 = torch.ops.aten.view.default(permute_39, [2, 8192, -1]);  permute_39 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16)
        all_gather_into_tensor_32 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_116, 64, '0');  convert_element_type_116 = None
        wait_tensor_32 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_32);  all_gather_into_tensor_32 = None
        permute_40 = torch.ops.aten.permute.default(wait_tensor_32, [1, 0]);  wait_tensor_32 = None
        view_125 = torch.ops.aten.view.default(view_123, [16384, 4096]);  view_123 = None
        mm_24 = torch.ops.aten.mm.default(view_125, permute_40);  view_125 = permute_40 = None
        view_126 = torch.ops.aten.view.default(mm_24, [2, 8192, 4096]);  mm_24 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, view_126);  view_126 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16)
        all_gather_into_tensor_33 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_119, 64, '0');  convert_element_type_119 = None
        wait_tensor_33 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_33);  all_gather_into_tensor_33 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_14 = torch.ops.aten.add.Scalar(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_7);  convert_element_type_120 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, wait_tensor_33);  mul_28 = wait_tensor_33 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16)
        all_gather_into_tensor_34 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_122, 64, '0');  convert_element_type_122 = None
        wait_tensor_34 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_34);  all_gather_into_tensor_34 = None
        permute_41 = torch.ops.aten.permute.default(wait_tensor_34, [1, 0]);  wait_tensor_34 = None
        view_129 = torch.ops.aten.view.default(convert_element_type_121, [16384, 4096]);  convert_element_type_121 = None
        mm_25 = torch.ops.aten.mm.default(view_129, permute_41);  permute_41 = None
        view_130 = torch.ops.aten.view.default(mm_25, [2, 8192, 14336])
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_125, sigmoid_3);  convert_element_type_125 = sigmoid_3 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16)
        all_gather_into_tensor_35 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_127, 64, '0');  convert_element_type_127 = None
        wait_tensor_35 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_35);  all_gather_into_tensor_35 = None
        permute_42 = torch.ops.aten.permute.default(wait_tensor_35, [1, 0]);  wait_tensor_35 = None
        mm_26 = torch.ops.aten.mm.default(view_129, permute_42);  view_129 = permute_42 = None
        view_133 = torch.ops.aten.view.default(mm_26, [2, 8192, 14336]);  mm_26 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_126, view_133);  convert_element_type_126 = view_133 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16)
        all_gather_into_tensor_36 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_130, 64, '0');  convert_element_type_130 = None
        wait_tensor_36 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_36);  all_gather_into_tensor_36 = None
        permute_43 = torch.ops.aten.permute.default(wait_tensor_36, [1, 0]);  wait_tensor_36 = None
        view_135 = torch.ops.aten.view.default(mul_31, [16384, 14336]);  mul_31 = None
        mm_27 = torch.ops.aten.mm.default(view_135, permute_43);  view_135 = permute_43 = None
        view_136 = torch.ops.aten.view.default(mm_27, [2, 8192, 4096]);  mm_27 = None
        add_15 = torch.ops.aten.add.Tensor(add_13, view_136);  add_13 = view_136 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16)
        all_gather_into_tensor_37 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_133, 64, '0');  convert_element_type_133 = None
        wait_tensor_37 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_37);  all_gather_into_tensor_37 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_16 = torch.ops.aten.add.Scalar(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_8);  convert_element_type_134 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, wait_tensor_37);  mul_32 = wait_tensor_37 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16)
        all_gather_into_tensor_38 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_136, 64, '0');  convert_element_type_136 = None
        wait_tensor_38 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_38);  all_gather_into_tensor_38 = None
        permute_44 = torch.ops.aten.permute.default(wait_tensor_38, [1, 0]);  wait_tensor_38 = None
        view_139 = torch.ops.aten.view.default(convert_element_type_135, [16384, 4096]);  convert_element_type_135 = None
        mm_28 = torch.ops.aten.mm.default(view_139, permute_44);  permute_44 = None
        view_140 = torch.ops.aten.view.default(mm_28, [2, 8192, 4096])
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16)
        all_gather_into_tensor_39 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_139, 64, '0');  convert_element_type_139 = None
        wait_tensor_39 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_39);  all_gather_into_tensor_39 = None
        permute_45 = torch.ops.aten.permute.default(wait_tensor_39, [1, 0]);  wait_tensor_39 = None
        mm_29 = torch.ops.aten.mm.default(view_139, permute_45);  permute_45 = None
        view_143 = torch.ops.aten.view.default(mm_29, [2, 8192, 1024]);  mm_29 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16)
        all_gather_into_tensor_40 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_142, 64, '0');  convert_element_type_142 = None
        wait_tensor_40 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_40);  all_gather_into_tensor_40 = None
        permute_46 = torch.ops.aten.permute.default(wait_tensor_40, [1, 0]);  wait_tensor_40 = None
        mm_30 = torch.ops.aten.mm.default(view_139, permute_46);  view_139 = permute_46 = None
        view_146 = torch.ops.aten.view.default(mm_30, [2, 8192, 1024])
        view_147 = torch.ops.aten.view.default(view_140, [2, 8192, -1, 128]);  view_140 = None
        view_148 = torch.ops.aten.view.default(view_143, [2, 8192, -1, 128]);  view_143 = None
        view_149 = torch.ops.aten.view.default(view_146, [2, 8192, -1, 128]);  view_146 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(view_147, torch.float32);  view_147 = None
        view_150 = torch.ops.aten.view.default(convert_element_type_145, [2, 8192, 32, -1, 2]);  convert_element_type_145 = None
        view_as_complex_8 = torch.ops.aten.view_as_complex.default(view_150);  view_150 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(view_148, torch.float32);  view_148 = None
        view_151 = torch.ops.aten.view.default(convert_element_type_146, [2, 8192, 8, -1, 2]);  convert_element_type_146 = None
        view_as_complex_9 = torch.ops.aten.view_as_complex.default(view_151);  view_151 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_as_complex_8, view_16);  view_as_complex_8 = None
        view_as_real_8 = torch.ops.aten.view_as_real.default(mul_34);  mul_34 = None
        view_153 = torch.ops.aten.view.default(view_as_real_8, [2, 8192, 32, 128]);  view_as_real_8 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_as_complex_9, view_16);  view_as_complex_9 = None
        view_as_real_9 = torch.ops.aten.view_as_real.default(mul_35);  mul_35 = None
        view_154 = torch.ops.aten.view.default(view_as_real_9, [2, 8192, 8, 128]);  view_as_real_9 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(view_153, torch.bfloat16);  view_153 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(view_154, torch.bfloat16);  view_154 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(convert_element_type_148, 3);  convert_element_type_148 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_8, [2, 8192, 8, 4, 128]);  unsqueeze_8 = None
        clone_8 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_155 = torch.ops.aten.view.default(clone_8, [2, 8192, 32, 128]);  clone_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_149, 3);  view_149 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_9, [2, 8192, 8, 4, 128]);  unsqueeze_9 = None
        clone_9 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_156 = torch.ops.aten.view.default(clone_9, [2, 8192, 32, 128]);  clone_9 = None
        permute_47 = torch.ops.aten.permute.default(convert_element_type_147, [0, 2, 1, 3]);  convert_element_type_147 = None
        permute_48 = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        permute_49 = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_47, permute_48, permute_49, None, True, 0.0, True);  permute_47 = permute_48 = permute_49 = None
        getitem_36 = _scaled_dot_product_cudnn_attention_4[0]
        getitem_37 = _scaled_dot_product_cudnn_attention_4[1]
        getitem_42 = _scaled_dot_product_cudnn_attention_4[6]
        getitem_43 = _scaled_dot_product_cudnn_attention_4[7];  _scaled_dot_product_cudnn_attention_4 = None
        permute_50 = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3])
        view_157 = torch.ops.aten.view.default(permute_50, [2, 8192, -1]);  permute_50 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16)
        all_gather_into_tensor_41 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_149, 64, '0');  convert_element_type_149 = None
        wait_tensor_41 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_41);  all_gather_into_tensor_41 = None
        permute_51 = torch.ops.aten.permute.default(wait_tensor_41, [1, 0]);  wait_tensor_41 = None
        view_159 = torch.ops.aten.view.default(view_157, [16384, 4096]);  view_157 = None
        mm_31 = torch.ops.aten.mm.default(view_159, permute_51);  view_159 = permute_51 = None
        view_160 = torch.ops.aten.view.default(mm_31, [2, 8192, 4096]);  mm_31 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, view_160);  view_160 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16)
        all_gather_into_tensor_42 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_152, 64, '0');  convert_element_type_152 = None
        wait_tensor_42 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_42);  all_gather_into_tensor_42 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_18 = torch.ops.aten.add.Scalar(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_9);  convert_element_type_153 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, wait_tensor_42);  mul_36 = wait_tensor_42 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16)
        all_gather_into_tensor_43 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_155, 64, '0');  convert_element_type_155 = None
        wait_tensor_43 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_43);  all_gather_into_tensor_43 = None
        permute_52 = torch.ops.aten.permute.default(wait_tensor_43, [1, 0]);  wait_tensor_43 = None
        view_163 = torch.ops.aten.view.default(convert_element_type_154, [16384, 4096]);  convert_element_type_154 = None
        mm_32 = torch.ops.aten.mm.default(view_163, permute_52);  permute_52 = None
        view_164 = torch.ops.aten.view.default(mm_32, [2, 8192, 14336])
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_158)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_158, sigmoid_4);  convert_element_type_158 = sigmoid_4 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16)
        all_gather_into_tensor_44 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_160, 64, '0');  convert_element_type_160 = None
        wait_tensor_44 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_44);  all_gather_into_tensor_44 = None
        permute_53 = torch.ops.aten.permute.default(wait_tensor_44, [1, 0]);  wait_tensor_44 = None
        mm_33 = torch.ops.aten.mm.default(view_163, permute_53);  view_163 = permute_53 = None
        view_167 = torch.ops.aten.view.default(mm_33, [2, 8192, 14336]);  mm_33 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_159, view_167);  convert_element_type_159 = view_167 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16)
        all_gather_into_tensor_45 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_163, 64, '0');  convert_element_type_163 = None
        wait_tensor_45 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_45);  all_gather_into_tensor_45 = None
        permute_54 = torch.ops.aten.permute.default(wait_tensor_45, [1, 0]);  wait_tensor_45 = None
        view_169 = torch.ops.aten.view.default(mul_39, [16384, 14336]);  mul_39 = None
        mm_34 = torch.ops.aten.mm.default(view_169, permute_54);  view_169 = permute_54 = None
        view_170 = torch.ops.aten.view.default(mm_34, [2, 8192, 4096]);  mm_34 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, view_170);  add_17 = view_170 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16)
        all_gather_into_tensor_46 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_166, 64, '0');  convert_element_type_166 = None
        wait_tensor_46 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_46);  all_gather_into_tensor_46 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
        add_20 = torch.ops.aten.add.Scalar(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_167, rsqrt_10);  convert_element_type_167 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, wait_tensor_46);  mul_40 = wait_tensor_46 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16)
        all_gather_into_tensor_47 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_169, 64, '0');  convert_element_type_169 = None
        wait_tensor_47 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_47);  all_gather_into_tensor_47 = None
        permute_55 = torch.ops.aten.permute.default(wait_tensor_47, [1, 0]);  wait_tensor_47 = None
        view_173 = torch.ops.aten.view.default(convert_element_type_168, [16384, 4096]);  convert_element_type_168 = None
        mm_35 = torch.ops.aten.mm.default(view_173, permute_55);  permute_55 = None
        view_174 = torch.ops.aten.view.default(mm_35, [2, 8192, 4096])
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16)
        all_gather_into_tensor_48 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_172, 64, '0');  convert_element_type_172 = None
        wait_tensor_48 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_48);  all_gather_into_tensor_48 = None
        permute_56 = torch.ops.aten.permute.default(wait_tensor_48, [1, 0]);  wait_tensor_48 = None
        mm_36 = torch.ops.aten.mm.default(view_173, permute_56);  permute_56 = None
        view_177 = torch.ops.aten.view.default(mm_36, [2, 8192, 1024]);  mm_36 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16)
        all_gather_into_tensor_49 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_175, 64, '0');  convert_element_type_175 = None
        wait_tensor_49 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_49);  all_gather_into_tensor_49 = None
        permute_57 = torch.ops.aten.permute.default(wait_tensor_49, [1, 0]);  wait_tensor_49 = None
        mm_37 = torch.ops.aten.mm.default(view_173, permute_57);  view_173 = permute_57 = None
        view_180 = torch.ops.aten.view.default(mm_37, [2, 8192, 1024])
        view_181 = torch.ops.aten.view.default(view_174, [2, 8192, -1, 128]);  view_174 = None
        view_182 = torch.ops.aten.view.default(view_177, [2, 8192, -1, 128]);  view_177 = None
        view_183 = torch.ops.aten.view.default(view_180, [2, 8192, -1, 128]);  view_180 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        view_184 = torch.ops.aten.view.default(convert_element_type_178, [2, 8192, 32, -1, 2]);  convert_element_type_178 = None
        view_as_complex_10 = torch.ops.aten.view_as_complex.default(view_184);  view_184 = None
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        view_185 = torch.ops.aten.view.default(convert_element_type_179, [2, 8192, 8, -1, 2]);  convert_element_type_179 = None
        view_as_complex_11 = torch.ops.aten.view_as_complex.default(view_185);  view_185 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_as_complex_10, view_16);  view_as_complex_10 = None
        view_as_real_10 = torch.ops.aten.view_as_real.default(mul_42);  mul_42 = None
        view_187 = torch.ops.aten.view.default(view_as_real_10, [2, 8192, 32, 128]);  view_as_real_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_as_complex_11, view_16);  view_as_complex_11 = None
        view_as_real_11 = torch.ops.aten.view_as_real.default(mul_43);  mul_43 = None
        view_188 = torch.ops.aten.view.default(view_as_real_11, [2, 8192, 8, 128]);  view_as_real_11 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(view_188, torch.bfloat16);  view_188 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3);  convert_element_type_181 = None
        expand_10 = torch.ops.aten.expand.default(unsqueeze_10, [2, 8192, 8, 4, 128]);  unsqueeze_10 = None
        clone_10 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_189 = torch.ops.aten.view.default(clone_10, [2, 8192, 32, 128]);  clone_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(view_183, 3);  view_183 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_11, [2, 8192, 8, 4, 128]);  unsqueeze_11 = None
        clone_11 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_190 = torch.ops.aten.view.default(clone_11, [2, 8192, 32, 128]);  clone_11 = None
        permute_58 = torch.ops.aten.permute.default(convert_element_type_180, [0, 2, 1, 3]);  convert_element_type_180 = None
        permute_59 = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        permute_60 = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_58, permute_59, permute_60, None, True, 0.0, True);  permute_58 = permute_59 = permute_60 = None
        getitem_45 = _scaled_dot_product_cudnn_attention_5[0]
        getitem_46 = _scaled_dot_product_cudnn_attention_5[1]
        getitem_51 = _scaled_dot_product_cudnn_attention_5[6]
        getitem_52 = _scaled_dot_product_cudnn_attention_5[7];  _scaled_dot_product_cudnn_attention_5 = None
        permute_61 = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3])
        view_191 = torch.ops.aten.view.default(permute_61, [2, 8192, -1]);  permute_61 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16)
        all_gather_into_tensor_50 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_182, 64, '0');  convert_element_type_182 = None
        wait_tensor_50 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_50);  all_gather_into_tensor_50 = None
        permute_62 = torch.ops.aten.permute.default(wait_tensor_50, [1, 0]);  wait_tensor_50 = None
        view_193 = torch.ops.aten.view.default(view_191, [16384, 4096]);  view_191 = None
        mm_38 = torch.ops.aten.mm.default(view_193, permute_62);  view_193 = permute_62 = None
        view_194 = torch.ops.aten.view.default(mm_38, [2, 8192, 4096]);  mm_38 = None
        add_21 = torch.ops.aten.add.Tensor(add_19, view_194);  view_194 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16)
        all_gather_into_tensor_51 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_185, 64, '0');  convert_element_type_185 = None
        wait_tensor_51 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_51);  all_gather_into_tensor_51 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_22 = torch.ops.aten.add.Scalar(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_186, rsqrt_11);  convert_element_type_186 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, wait_tensor_51);  mul_44 = wait_tensor_51 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16)
        all_gather_into_tensor_52 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_188, 64, '0');  convert_element_type_188 = None
        wait_tensor_52 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_52);  all_gather_into_tensor_52 = None
        permute_63 = torch.ops.aten.permute.default(wait_tensor_52, [1, 0]);  wait_tensor_52 = None
        view_197 = torch.ops.aten.view.default(convert_element_type_187, [16384, 4096]);  convert_element_type_187 = None
        mm_39 = torch.ops.aten.mm.default(view_197, permute_63);  permute_63 = None
        view_198 = torch.ops.aten.view.default(mm_39, [2, 8192, 14336])
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_191)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_191, sigmoid_5);  convert_element_type_191 = sigmoid_5 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16)
        all_gather_into_tensor_53 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_193, 64, '0');  convert_element_type_193 = None
        wait_tensor_53 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_53);  all_gather_into_tensor_53 = None
        permute_64 = torch.ops.aten.permute.default(wait_tensor_53, [1, 0]);  wait_tensor_53 = None
        mm_40 = torch.ops.aten.mm.default(view_197, permute_64);  view_197 = permute_64 = None
        view_201 = torch.ops.aten.view.default(mm_40, [2, 8192, 14336]);  mm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_192, view_201);  convert_element_type_192 = view_201 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16)
        all_gather_into_tensor_54 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_196, 64, '0');  convert_element_type_196 = None
        wait_tensor_54 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_54);  all_gather_into_tensor_54 = None
        permute_65 = torch.ops.aten.permute.default(wait_tensor_54, [1, 0]);  wait_tensor_54 = None
        view_203 = torch.ops.aten.view.default(mul_47, [16384, 14336]);  mul_47 = None
        mm_41 = torch.ops.aten.mm.default(view_203, permute_65);  view_203 = permute_65 = None
        view_204 = torch.ops.aten.view.default(mm_41, [2, 8192, 4096]);  mm_41 = None
        add_23 = torch.ops.aten.add.Tensor(add_21, view_204);  add_21 = view_204 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16)
        all_gather_into_tensor_55 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_199, 64, '0');  convert_element_type_199 = None
        wait_tensor_55 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_55);  all_gather_into_tensor_55 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [2], True);  pow_13 = None
        add_24 = torch.ops.aten.add.Scalar(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_12);  convert_element_type_200 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, wait_tensor_55);  mul_48 = wait_tensor_55 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16)
        all_gather_into_tensor_56 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_202, 64, '0');  convert_element_type_202 = None
        wait_tensor_56 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_56);  all_gather_into_tensor_56 = None
        permute_66 = torch.ops.aten.permute.default(wait_tensor_56, [1, 0]);  wait_tensor_56 = None
        view_207 = torch.ops.aten.view.default(convert_element_type_201, [16384, 4096]);  convert_element_type_201 = None
        mm_42 = torch.ops.aten.mm.default(view_207, permute_66);  permute_66 = None
        view_208 = torch.ops.aten.view.default(mm_42, [2, 8192, 4096])
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16)
        all_gather_into_tensor_57 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_205, 64, '0');  convert_element_type_205 = None
        wait_tensor_57 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_57);  all_gather_into_tensor_57 = None
        permute_67 = torch.ops.aten.permute.default(wait_tensor_57, [1, 0]);  wait_tensor_57 = None
        mm_43 = torch.ops.aten.mm.default(view_207, permute_67);  permute_67 = None
        view_211 = torch.ops.aten.view.default(mm_43, [2, 8192, 1024]);  mm_43 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16)
        all_gather_into_tensor_58 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_208, 64, '0');  convert_element_type_208 = None
        wait_tensor_58 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_58);  all_gather_into_tensor_58 = None
        permute_68 = torch.ops.aten.permute.default(wait_tensor_58, [1, 0]);  wait_tensor_58 = None
        mm_44 = torch.ops.aten.mm.default(view_207, permute_68);  view_207 = permute_68 = None
        view_214 = torch.ops.aten.view.default(mm_44, [2, 8192, 1024])
        view_215 = torch.ops.aten.view.default(view_208, [2, 8192, -1, 128]);  view_208 = None
        view_216 = torch.ops.aten.view.default(view_211, [2, 8192, -1, 128]);  view_211 = None
        view_217 = torch.ops.aten.view.default(view_214, [2, 8192, -1, 128]);  view_214 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None
        view_218 = torch.ops.aten.view.default(convert_element_type_211, [2, 8192, 32, -1, 2]);  convert_element_type_211 = None
        view_as_complex_12 = torch.ops.aten.view_as_complex.default(view_218);  view_218 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        view_219 = torch.ops.aten.view.default(convert_element_type_212, [2, 8192, 8, -1, 2]);  convert_element_type_212 = None
        view_as_complex_13 = torch.ops.aten.view_as_complex.default(view_219);  view_219 = None
        mul_50 = torch.ops.aten.mul.Tensor(view_as_complex_12, view_16);  view_as_complex_12 = None
        view_as_real_12 = torch.ops.aten.view_as_real.default(mul_50);  mul_50 = None
        view_221 = torch.ops.aten.view.default(view_as_real_12, [2, 8192, 32, 128]);  view_as_real_12 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_as_complex_13, view_16);  view_as_complex_13 = None
        view_as_real_13 = torch.ops.aten.view_as_real.default(mul_51);  mul_51 = None
        view_222 = torch.ops.aten.view.default(view_as_real_13, [2, 8192, 8, 128]);  view_as_real_13 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(view_221, torch.bfloat16);  view_221 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(convert_element_type_214, 3);  convert_element_type_214 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_12, [2, 8192, 8, 4, 128]);  unsqueeze_12 = None
        clone_12 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_223 = torch.ops.aten.view.default(clone_12, [2, 8192, 32, 128]);  clone_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_217, 3);  view_217 = None
        expand_13 = torch.ops.aten.expand.default(unsqueeze_13, [2, 8192, 8, 4, 128]);  unsqueeze_13 = None
        clone_13 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_224 = torch.ops.aten.view.default(clone_13, [2, 8192, 32, 128]);  clone_13 = None
        permute_69 = torch.ops.aten.permute.default(convert_element_type_213, [0, 2, 1, 3]);  convert_element_type_213 = None
        permute_70 = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        permute_71 = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_69, permute_70, permute_71, None, True, 0.0, True);  permute_69 = permute_70 = permute_71 = None
        getitem_54 = _scaled_dot_product_cudnn_attention_6[0]
        getitem_55 = _scaled_dot_product_cudnn_attention_6[1]
        getitem_60 = _scaled_dot_product_cudnn_attention_6[6]
        getitem_61 = _scaled_dot_product_cudnn_attention_6[7];  _scaled_dot_product_cudnn_attention_6 = None
        permute_72 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3])
        view_225 = torch.ops.aten.view.default(permute_72, [2, 8192, -1]);  permute_72 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16)
        all_gather_into_tensor_59 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_215, 64, '0');  convert_element_type_215 = None
        wait_tensor_59 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_59);  all_gather_into_tensor_59 = None
        permute_73 = torch.ops.aten.permute.default(wait_tensor_59, [1, 0]);  wait_tensor_59 = None
        view_227 = torch.ops.aten.view.default(view_225, [16384, 4096]);  view_225 = None
        mm_45 = torch.ops.aten.mm.default(view_227, permute_73);  view_227 = permute_73 = None
        view_228 = torch.ops.aten.view.default(mm_45, [2, 8192, 4096]);  mm_45 = None
        add_25 = torch.ops.aten.add.Tensor(add_23, view_228);  view_228 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16)
        all_gather_into_tensor_60 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_218, 64, '0');  convert_element_type_218 = None
        wait_tensor_60 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_60);  all_gather_into_tensor_60 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [2], True);  pow_14 = None
        add_26 = torch.ops.aten.add.Scalar(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_13);  convert_element_type_219 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, wait_tensor_60);  mul_52 = wait_tensor_60 = None
        convert_element_type_220 = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16)
        all_gather_into_tensor_61 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_221, 64, '0');  convert_element_type_221 = None
        wait_tensor_61 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_61);  all_gather_into_tensor_61 = None
        permute_74 = torch.ops.aten.permute.default(wait_tensor_61, [1, 0]);  wait_tensor_61 = None
        view_231 = torch.ops.aten.view.default(convert_element_type_220, [16384, 4096]);  convert_element_type_220 = None
        mm_46 = torch.ops.aten.mm.default(view_231, permute_74);  permute_74 = None
        view_232 = torch.ops.aten.view.default(mm_46, [2, 8192, 14336])
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_224)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_224, sigmoid_6);  convert_element_type_224 = sigmoid_6 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16)
        all_gather_into_tensor_62 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_226, 64, '0');  convert_element_type_226 = None
        wait_tensor_62 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_62);  all_gather_into_tensor_62 = None
        permute_75 = torch.ops.aten.permute.default(wait_tensor_62, [1, 0]);  wait_tensor_62 = None
        mm_47 = torch.ops.aten.mm.default(view_231, permute_75);  view_231 = permute_75 = None
        view_235 = torch.ops.aten.view.default(mm_47, [2, 8192, 14336]);  mm_47 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_225, view_235);  convert_element_type_225 = view_235 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16)
        all_gather_into_tensor_63 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_229, 64, '0');  convert_element_type_229 = None
        wait_tensor_63 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_63);  all_gather_into_tensor_63 = None
        permute_76 = torch.ops.aten.permute.default(wait_tensor_63, [1, 0]);  wait_tensor_63 = None
        view_237 = torch.ops.aten.view.default(mul_55, [16384, 14336]);  mul_55 = None
        mm_48 = torch.ops.aten.mm.default(view_237, permute_76);  view_237 = permute_76 = None
        view_238 = torch.ops.aten.view.default(mm_48, [2, 8192, 4096]);  mm_48 = None
        add_27 = torch.ops.aten.add.Tensor(add_25, view_238);  add_25 = view_238 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16)
        all_gather_into_tensor_64 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_232, 64, '0');  convert_element_type_232 = None
        wait_tensor_64 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_64);  all_gather_into_tensor_64 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_28 = torch.ops.aten.add.Scalar(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_14);  convert_element_type_233 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, wait_tensor_64);  mul_56 = wait_tensor_64 = None
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16)
        all_gather_into_tensor_65 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_235, 64, '0');  convert_element_type_235 = None
        wait_tensor_65 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_65);  all_gather_into_tensor_65 = None
        permute_77 = torch.ops.aten.permute.default(wait_tensor_65, [1, 0]);  wait_tensor_65 = None
        view_241 = torch.ops.aten.view.default(convert_element_type_234, [16384, 4096]);  convert_element_type_234 = None
        mm_49 = torch.ops.aten.mm.default(view_241, permute_77);  permute_77 = None
        view_242 = torch.ops.aten.view.default(mm_49, [2, 8192, 4096])
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16)
        all_gather_into_tensor_66 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_238, 64, '0');  convert_element_type_238 = None
        wait_tensor_66 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_66);  all_gather_into_tensor_66 = None
        permute_78 = torch.ops.aten.permute.default(wait_tensor_66, [1, 0]);  wait_tensor_66 = None
        mm_50 = torch.ops.aten.mm.default(view_241, permute_78);  permute_78 = None
        view_245 = torch.ops.aten.view.default(mm_50, [2, 8192, 1024]);  mm_50 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16)
        all_gather_into_tensor_67 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_241, 64, '0');  convert_element_type_241 = None
        wait_tensor_67 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_67);  all_gather_into_tensor_67 = None
        permute_79 = torch.ops.aten.permute.default(wait_tensor_67, [1, 0]);  wait_tensor_67 = None
        mm_51 = torch.ops.aten.mm.default(view_241, permute_79);  view_241 = permute_79 = None
        view_248 = torch.ops.aten.view.default(mm_51, [2, 8192, 1024])
        view_249 = torch.ops.aten.view.default(view_242, [2, 8192, -1, 128]);  view_242 = None
        view_250 = torch.ops.aten.view.default(view_245, [2, 8192, -1, 128]);  view_245 = None
        view_251 = torch.ops.aten.view.default(view_248, [2, 8192, -1, 128]);  view_248 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        view_252 = torch.ops.aten.view.default(convert_element_type_244, [2, 8192, 32, -1, 2]);  convert_element_type_244 = None
        view_as_complex_14 = torch.ops.aten.view_as_complex.default(view_252);  view_252 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        view_253 = torch.ops.aten.view.default(convert_element_type_245, [2, 8192, 8, -1, 2]);  convert_element_type_245 = None
        view_as_complex_15 = torch.ops.aten.view_as_complex.default(view_253);  view_253 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_as_complex_14, view_16);  view_as_complex_14 = None
        view_as_real_14 = torch.ops.aten.view_as_real.default(mul_58);  mul_58 = None
        view_255 = torch.ops.aten.view.default(view_as_real_14, [2, 8192, 32, 128]);  view_as_real_14 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_as_complex_15, view_16);  view_as_complex_15 = None
        view_as_real_15 = torch.ops.aten.view_as_real.default(mul_59);  mul_59 = None
        view_256 = torch.ops.aten.view.default(view_as_real_15, [2, 8192, 8, 128]);  view_as_real_15 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(view_255, torch.bfloat16);  view_255 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(convert_element_type_247, 3);  convert_element_type_247 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_14, [2, 8192, 8, 4, 128]);  unsqueeze_14 = None
        clone_14 = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_257 = torch.ops.aten.view.default(clone_14, [2, 8192, 32, 128]);  clone_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_251, 3);  view_251 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_15, [2, 8192, 8, 4, 128]);  unsqueeze_15 = None
        clone_15 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_258 = torch.ops.aten.view.default(clone_15, [2, 8192, 32, 128]);  clone_15 = None
        permute_80 = torch.ops.aten.permute.default(convert_element_type_246, [0, 2, 1, 3]);  convert_element_type_246 = None
        permute_81 = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        permute_82 = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_80, permute_81, permute_82, None, True, 0.0, True);  permute_80 = permute_81 = permute_82 = None
        getitem_63 = _scaled_dot_product_cudnn_attention_7[0]
        getitem_64 = _scaled_dot_product_cudnn_attention_7[1]
        getitem_69 = _scaled_dot_product_cudnn_attention_7[6]
        getitem_70 = _scaled_dot_product_cudnn_attention_7[7];  _scaled_dot_product_cudnn_attention_7 = None
        permute_83 = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3])
        view_259 = torch.ops.aten.view.default(permute_83, [2, 8192, -1]);  permute_83 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16)
        all_gather_into_tensor_68 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_248, 64, '0');  convert_element_type_248 = None
        wait_tensor_68 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_68);  all_gather_into_tensor_68 = None
        permute_84 = torch.ops.aten.permute.default(wait_tensor_68, [1, 0]);  wait_tensor_68 = None
        view_261 = torch.ops.aten.view.default(view_259, [16384, 4096]);  view_259 = None
        mm_52 = torch.ops.aten.mm.default(view_261, permute_84);  view_261 = permute_84 = None
        view_262 = torch.ops.aten.view.default(mm_52, [2, 8192, 4096]);  mm_52 = None
        add_29 = torch.ops.aten.add.Tensor(add_27, view_262);  view_262 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16)
        all_gather_into_tensor_69 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_251, 64, '0');  convert_element_type_251 = None
        wait_tensor_69 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_69);  all_gather_into_tensor_69 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [2], True);  pow_16 = None
        add_30 = torch.ops.aten.add.Scalar(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_15);  convert_element_type_252 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, wait_tensor_69);  mul_60 = wait_tensor_69 = None
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16)
        all_gather_into_tensor_70 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_254, 64, '0');  convert_element_type_254 = None
        wait_tensor_70 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_70);  all_gather_into_tensor_70 = None
        permute_85 = torch.ops.aten.permute.default(wait_tensor_70, [1, 0]);  wait_tensor_70 = None
        view_265 = torch.ops.aten.view.default(convert_element_type_253, [16384, 4096]);  convert_element_type_253 = None
        mm_53 = torch.ops.aten.mm.default(view_265, permute_85);  permute_85 = None
        view_266 = torch.ops.aten.view.default(mm_53, [2, 8192, 14336])
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_7);  convert_element_type_257 = sigmoid_7 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16)
        all_gather_into_tensor_71 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_259, 64, '0');  convert_element_type_259 = None
        wait_tensor_71 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_71);  all_gather_into_tensor_71 = None
        permute_86 = torch.ops.aten.permute.default(wait_tensor_71, [1, 0]);  wait_tensor_71 = None
        mm_54 = torch.ops.aten.mm.default(view_265, permute_86);  view_265 = permute_86 = None
        view_269 = torch.ops.aten.view.default(mm_54, [2, 8192, 14336]);  mm_54 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_258, view_269);  convert_element_type_258 = view_269 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16)
        all_gather_into_tensor_72 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_262, 64, '0');  convert_element_type_262 = None
        wait_tensor_72 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_72);  all_gather_into_tensor_72 = None
        permute_87 = torch.ops.aten.permute.default(wait_tensor_72, [1, 0]);  wait_tensor_72 = None
        view_271 = torch.ops.aten.view.default(mul_63, [16384, 14336]);  mul_63 = None
        mm_55 = torch.ops.aten.mm.default(view_271, permute_87);  view_271 = permute_87 = None
        view_272 = torch.ops.aten.view.default(mm_55, [2, 8192, 4096]);  mm_55 = None
        add_31 = torch.ops.aten.add.Tensor(add_29, view_272);  add_29 = view_272 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16)
        all_gather_into_tensor_73 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_265, 64, '0');  convert_element_type_265 = None
        wait_tensor_73 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_73);  all_gather_into_tensor_73 = None
        convert_element_type_266 = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_32 = torch.ops.aten.add.Scalar(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_16);  convert_element_type_266 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, wait_tensor_73);  mul_64 = wait_tensor_73 = None
        convert_element_type_267 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16)
        all_gather_into_tensor_74 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_268, 64, '0');  convert_element_type_268 = None
        wait_tensor_74 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_74);  all_gather_into_tensor_74 = None
        permute_88 = torch.ops.aten.permute.default(wait_tensor_74, [1, 0]);  wait_tensor_74 = None
        view_275 = torch.ops.aten.view.default(convert_element_type_267, [16384, 4096]);  convert_element_type_267 = None
        mm_56 = torch.ops.aten.mm.default(view_275, permute_88);  permute_88 = None
        view_276 = torch.ops.aten.view.default(mm_56, [2, 8192, 4096])
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16)
        all_gather_into_tensor_75 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_271, 64, '0');  convert_element_type_271 = None
        wait_tensor_75 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_75);  all_gather_into_tensor_75 = None
        permute_89 = torch.ops.aten.permute.default(wait_tensor_75, [1, 0]);  wait_tensor_75 = None
        mm_57 = torch.ops.aten.mm.default(view_275, permute_89);  permute_89 = None
        view_279 = torch.ops.aten.view.default(mm_57, [2, 8192, 1024]);  mm_57 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16)
        all_gather_into_tensor_76 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_274, 64, '0');  convert_element_type_274 = None
        wait_tensor_76 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_76);  all_gather_into_tensor_76 = None
        permute_90 = torch.ops.aten.permute.default(wait_tensor_76, [1, 0]);  wait_tensor_76 = None
        mm_58 = torch.ops.aten.mm.default(view_275, permute_90);  view_275 = permute_90 = None
        view_282 = torch.ops.aten.view.default(mm_58, [2, 8192, 1024])
        view_283 = torch.ops.aten.view.default(view_276, [2, 8192, -1, 128]);  view_276 = None
        view_284 = torch.ops.aten.view.default(view_279, [2, 8192, -1, 128]);  view_279 = None
        view_285 = torch.ops.aten.view.default(view_282, [2, 8192, -1, 128]);  view_282 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        view_286 = torch.ops.aten.view.default(convert_element_type_277, [2, 8192, 32, -1, 2]);  convert_element_type_277 = None
        view_as_complex_16 = torch.ops.aten.view_as_complex.default(view_286);  view_286 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        view_287 = torch.ops.aten.view.default(convert_element_type_278, [2, 8192, 8, -1, 2]);  convert_element_type_278 = None
        view_as_complex_17 = torch.ops.aten.view_as_complex.default(view_287);  view_287 = None
        mul_66 = torch.ops.aten.mul.Tensor(view_as_complex_16, view_16);  view_as_complex_16 = None
        view_as_real_16 = torch.ops.aten.view_as_real.default(mul_66);  mul_66 = None
        view_289 = torch.ops.aten.view.default(view_as_real_16, [2, 8192, 32, 128]);  view_as_real_16 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_as_complex_17, view_16);  view_as_complex_17 = None
        view_as_real_17 = torch.ops.aten.view_as_real.default(mul_67);  mul_67 = None
        view_290 = torch.ops.aten.view.default(view_as_real_17, [2, 8192, 8, 128]);  view_as_real_17 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        convert_element_type_280 = torch.ops.prims.convert_element_type.default(view_290, torch.bfloat16);  view_290 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3);  convert_element_type_280 = None
        expand_16 = torch.ops.aten.expand.default(unsqueeze_16, [2, 8192, 8, 4, 128]);  unsqueeze_16 = None
        clone_16 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_291 = torch.ops.aten.view.default(clone_16, [2, 8192, 32, 128]);  clone_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_285, 3);  view_285 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_17, [2, 8192, 8, 4, 128]);  unsqueeze_17 = None
        clone_17 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_292 = torch.ops.aten.view.default(clone_17, [2, 8192, 32, 128]);  clone_17 = None
        permute_91 = torch.ops.aten.permute.default(convert_element_type_279, [0, 2, 1, 3]);  convert_element_type_279 = None
        permute_92 = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        permute_93 = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_91, permute_92, permute_93, None, True, 0.0, True);  permute_91 = permute_92 = permute_93 = None
        getitem_72 = _scaled_dot_product_cudnn_attention_8[0]
        getitem_73 = _scaled_dot_product_cudnn_attention_8[1]
        getitem_78 = _scaled_dot_product_cudnn_attention_8[6]
        getitem_79 = _scaled_dot_product_cudnn_attention_8[7];  _scaled_dot_product_cudnn_attention_8 = None
        permute_94 = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3])
        view_293 = torch.ops.aten.view.default(permute_94, [2, 8192, -1]);  permute_94 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16)
        all_gather_into_tensor_77 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_281, 64, '0');  convert_element_type_281 = None
        wait_tensor_77 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_77);  all_gather_into_tensor_77 = None
        permute_95 = torch.ops.aten.permute.default(wait_tensor_77, [1, 0]);  wait_tensor_77 = None
        view_295 = torch.ops.aten.view.default(view_293, [16384, 4096]);  view_293 = None
        mm_59 = torch.ops.aten.mm.default(view_295, permute_95);  view_295 = permute_95 = None
        view_296 = torch.ops.aten.view.default(mm_59, [2, 8192, 4096]);  mm_59 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, view_296);  view_296 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16)
        all_gather_into_tensor_78 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_284, 64, '0');  convert_element_type_284 = None
        wait_tensor_78 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_78);  all_gather_into_tensor_78 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [2], True);  pow_18 = None
        add_34 = torch.ops.aten.add.Scalar(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_285, rsqrt_17);  convert_element_type_285 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, wait_tensor_78);  mul_68 = wait_tensor_78 = None
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16)
        all_gather_into_tensor_79 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_287, 64, '0');  convert_element_type_287 = None
        wait_tensor_79 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_79);  all_gather_into_tensor_79 = None
        permute_96 = torch.ops.aten.permute.default(wait_tensor_79, [1, 0]);  wait_tensor_79 = None
        view_299 = torch.ops.aten.view.default(convert_element_type_286, [16384, 4096]);  convert_element_type_286 = None
        mm_60 = torch.ops.aten.mm.default(view_299, permute_96);  permute_96 = None
        view_300 = torch.ops.aten.view.default(mm_60, [2, 8192, 14336])
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_290)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_290, sigmoid_8);  convert_element_type_290 = sigmoid_8 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16)
        all_gather_into_tensor_80 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_292, 64, '0');  convert_element_type_292 = None
        wait_tensor_80 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_80);  all_gather_into_tensor_80 = None
        permute_97 = torch.ops.aten.permute.default(wait_tensor_80, [1, 0]);  wait_tensor_80 = None
        mm_61 = torch.ops.aten.mm.default(view_299, permute_97);  view_299 = permute_97 = None
        view_303 = torch.ops.aten.view.default(mm_61, [2, 8192, 14336]);  mm_61 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_291, view_303);  convert_element_type_291 = view_303 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16)
        all_gather_into_tensor_81 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_295, 64, '0');  convert_element_type_295 = None
        wait_tensor_81 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_81);  all_gather_into_tensor_81 = None
        permute_98 = torch.ops.aten.permute.default(wait_tensor_81, [1, 0]);  wait_tensor_81 = None
        view_305 = torch.ops.aten.view.default(mul_71, [16384, 14336]);  mul_71 = None
        mm_62 = torch.ops.aten.mm.default(view_305, permute_98);  view_305 = permute_98 = None
        view_306 = torch.ops.aten.view.default(mm_62, [2, 8192, 4096]);  mm_62 = None
        add_35 = torch.ops.aten.add.Tensor(add_33, view_306);  add_33 = view_306 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16)
        all_gather_into_tensor_82 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_298, 64, '0');  convert_element_type_298 = None
        wait_tensor_82 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_82);  all_gather_into_tensor_82 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [2], True);  pow_19 = None
        add_36 = torch.ops.aten.add.Scalar(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_18);  convert_element_type_299 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, wait_tensor_82);  mul_72 = wait_tensor_82 = None
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16)
        all_gather_into_tensor_83 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_301, 64, '0');  convert_element_type_301 = None
        wait_tensor_83 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_83);  all_gather_into_tensor_83 = None
        permute_99 = torch.ops.aten.permute.default(wait_tensor_83, [1, 0]);  wait_tensor_83 = None
        view_309 = torch.ops.aten.view.default(convert_element_type_300, [16384, 4096]);  convert_element_type_300 = None
        mm_63 = torch.ops.aten.mm.default(view_309, permute_99);  permute_99 = None
        view_310 = torch.ops.aten.view.default(mm_63, [2, 8192, 4096])
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16)
        all_gather_into_tensor_84 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_304, 64, '0');  convert_element_type_304 = None
        wait_tensor_84 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_84);  all_gather_into_tensor_84 = None
        permute_100 = torch.ops.aten.permute.default(wait_tensor_84, [1, 0]);  wait_tensor_84 = None
        mm_64 = torch.ops.aten.mm.default(view_309, permute_100);  permute_100 = None
        view_313 = torch.ops.aten.view.default(mm_64, [2, 8192, 1024]);  mm_64 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16)
        all_gather_into_tensor_85 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_307, 64, '0');  convert_element_type_307 = None
        wait_tensor_85 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_85);  all_gather_into_tensor_85 = None
        permute_101 = torch.ops.aten.permute.default(wait_tensor_85, [1, 0]);  wait_tensor_85 = None
        mm_65 = torch.ops.aten.mm.default(view_309, permute_101);  view_309 = permute_101 = None
        view_316 = torch.ops.aten.view.default(mm_65, [2, 8192, 1024])
        view_317 = torch.ops.aten.view.default(view_310, [2, 8192, -1, 128]);  view_310 = None
        view_318 = torch.ops.aten.view.default(view_313, [2, 8192, -1, 128]);  view_313 = None
        view_319 = torch.ops.aten.view.default(view_316, [2, 8192, -1, 128]);  view_316 = None
        convert_element_type_310 = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        view_320 = torch.ops.aten.view.default(convert_element_type_310, [2, 8192, 32, -1, 2]);  convert_element_type_310 = None
        view_as_complex_18 = torch.ops.aten.view_as_complex.default(view_320);  view_320 = None
        convert_element_type_311 = torch.ops.prims.convert_element_type.default(view_318, torch.float32);  view_318 = None
        view_321 = torch.ops.aten.view.default(convert_element_type_311, [2, 8192, 8, -1, 2]);  convert_element_type_311 = None
        view_as_complex_19 = torch.ops.aten.view_as_complex.default(view_321);  view_321 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_as_complex_18, view_16);  view_as_complex_18 = None
        view_as_real_18 = torch.ops.aten.view_as_real.default(mul_74);  mul_74 = None
        view_323 = torch.ops.aten.view.default(view_as_real_18, [2, 8192, 32, 128]);  view_as_real_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_as_complex_19, view_16);  view_as_complex_19 = None
        view_as_real_19 = torch.ops.aten.view_as_real.default(mul_75);  mul_75 = None
        view_324 = torch.ops.aten.view.default(view_as_real_19, [2, 8192, 8, 128]);  view_as_real_19 = None
        convert_element_type_312 = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        convert_element_type_313 = torch.ops.prims.convert_element_type.default(view_324, torch.bfloat16);  view_324 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(convert_element_type_313, 3);  convert_element_type_313 = None
        expand_18 = torch.ops.aten.expand.default(unsqueeze_18, [2, 8192, 8, 4, 128]);  unsqueeze_18 = None
        clone_18 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_325 = torch.ops.aten.view.default(clone_18, [2, 8192, 32, 128]);  clone_18 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(view_319, 3);  view_319 = None
        expand_19 = torch.ops.aten.expand.default(unsqueeze_19, [2, 8192, 8, 4, 128]);  unsqueeze_19 = None
        clone_19 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_326 = torch.ops.aten.view.default(clone_19, [2, 8192, 32, 128]);  clone_19 = None
        permute_102 = torch.ops.aten.permute.default(convert_element_type_312, [0, 2, 1, 3]);  convert_element_type_312 = None
        permute_103 = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        permute_104 = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_102, permute_103, permute_104, None, True, 0.0, True);  permute_102 = permute_103 = permute_104 = None
        getitem_81 = _scaled_dot_product_cudnn_attention_9[0]
        getitem_82 = _scaled_dot_product_cudnn_attention_9[1]
        getitem_87 = _scaled_dot_product_cudnn_attention_9[6]
        getitem_88 = _scaled_dot_product_cudnn_attention_9[7];  _scaled_dot_product_cudnn_attention_9 = None
        permute_105 = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3])
        view_327 = torch.ops.aten.view.default(permute_105, [2, 8192, -1]);  permute_105 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16)
        all_gather_into_tensor_86 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_314, 64, '0');  convert_element_type_314 = None
        wait_tensor_86 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_86);  all_gather_into_tensor_86 = None
        permute_106 = torch.ops.aten.permute.default(wait_tensor_86, [1, 0]);  wait_tensor_86 = None
        view_329 = torch.ops.aten.view.default(view_327, [16384, 4096]);  view_327 = None
        mm_66 = torch.ops.aten.mm.default(view_329, permute_106);  view_329 = permute_106 = None
        view_330 = torch.ops.aten.view.default(mm_66, [2, 8192, 4096]);  mm_66 = None
        add_37 = torch.ops.aten.add.Tensor(add_35, view_330);  view_330 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16)
        all_gather_into_tensor_87 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_317, 64, '0');  convert_element_type_317 = None
        wait_tensor_87 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_87);  all_gather_into_tensor_87 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_38 = torch.ops.aten.add.Scalar(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_19);  convert_element_type_318 = rsqrt_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, wait_tensor_87);  mul_76 = wait_tensor_87 = None
        convert_element_type_319 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16)
        all_gather_into_tensor_88 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_320, 64, '0');  convert_element_type_320 = None
        wait_tensor_88 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_88);  all_gather_into_tensor_88 = None
        permute_107 = torch.ops.aten.permute.default(wait_tensor_88, [1, 0]);  wait_tensor_88 = None
        view_333 = torch.ops.aten.view.default(convert_element_type_319, [16384, 4096]);  convert_element_type_319 = None
        mm_67 = torch.ops.aten.mm.default(view_333, permute_107);  permute_107 = None
        view_334 = torch.ops.aten.view.default(mm_67, [2, 8192, 14336])
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(view_334, torch.float32);  view_334 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_323)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_323, sigmoid_9);  convert_element_type_323 = sigmoid_9 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16)
        all_gather_into_tensor_89 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_325, 64, '0');  convert_element_type_325 = None
        wait_tensor_89 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_89);  all_gather_into_tensor_89 = None
        permute_108 = torch.ops.aten.permute.default(wait_tensor_89, [1, 0]);  wait_tensor_89 = None
        mm_68 = torch.ops.aten.mm.default(view_333, permute_108);  view_333 = permute_108 = None
        view_337 = torch.ops.aten.view.default(mm_68, [2, 8192, 14336]);  mm_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_324, view_337);  convert_element_type_324 = view_337 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16)
        all_gather_into_tensor_90 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_328, 64, '0');  convert_element_type_328 = None
        wait_tensor_90 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_90);  all_gather_into_tensor_90 = None
        permute_109 = torch.ops.aten.permute.default(wait_tensor_90, [1, 0]);  wait_tensor_90 = None
        view_339 = torch.ops.aten.view.default(mul_79, [16384, 14336]);  mul_79 = None
        mm_69 = torch.ops.aten.mm.default(view_339, permute_109);  view_339 = permute_109 = None
        view_340 = torch.ops.aten.view.default(mm_69, [2, 8192, 4096]);  mm_69 = None
        add_39 = torch.ops.aten.add.Tensor(add_37, view_340);  add_37 = view_340 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16)
        all_gather_into_tensor_91 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_331, 64, '0');  convert_element_type_331 = None
        wait_tensor_91 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_91);  all_gather_into_tensor_91 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [2], True);  pow_21 = None
        add_40 = torch.ops.aten.add.Scalar(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_20);  convert_element_type_332 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, wait_tensor_91);  mul_80 = wait_tensor_91 = None
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16)
        all_gather_into_tensor_92 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_334, 64, '0');  convert_element_type_334 = None
        wait_tensor_92 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_92);  all_gather_into_tensor_92 = None
        permute_110 = torch.ops.aten.permute.default(wait_tensor_92, [1, 0]);  wait_tensor_92 = None
        view_343 = torch.ops.aten.view.default(convert_element_type_333, [16384, 4096]);  convert_element_type_333 = None
        mm_70 = torch.ops.aten.mm.default(view_343, permute_110);  permute_110 = None
        view_344 = torch.ops.aten.view.default(mm_70, [2, 8192, 4096])
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16)
        all_gather_into_tensor_93 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_337, 64, '0');  convert_element_type_337 = None
        wait_tensor_93 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_93);  all_gather_into_tensor_93 = None
        permute_111 = torch.ops.aten.permute.default(wait_tensor_93, [1, 0]);  wait_tensor_93 = None
        mm_71 = torch.ops.aten.mm.default(view_343, permute_111);  permute_111 = None
        view_347 = torch.ops.aten.view.default(mm_71, [2, 8192, 1024]);  mm_71 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16)
        all_gather_into_tensor_94 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_340, 64, '0');  convert_element_type_340 = None
        wait_tensor_94 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_94);  all_gather_into_tensor_94 = None
        permute_112 = torch.ops.aten.permute.default(wait_tensor_94, [1, 0]);  wait_tensor_94 = None
        mm_72 = torch.ops.aten.mm.default(view_343, permute_112);  view_343 = permute_112 = None
        view_350 = torch.ops.aten.view.default(mm_72, [2, 8192, 1024])
        view_351 = torch.ops.aten.view.default(view_344, [2, 8192, -1, 128]);  view_344 = None
        view_352 = torch.ops.aten.view.default(view_347, [2, 8192, -1, 128]);  view_347 = None
        view_353 = torch.ops.aten.view.default(view_350, [2, 8192, -1, 128]);  view_350 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        view_354 = torch.ops.aten.view.default(convert_element_type_343, [2, 8192, 32, -1, 2]);  convert_element_type_343 = None
        view_as_complex_20 = torch.ops.aten.view_as_complex.default(view_354);  view_354 = None
        convert_element_type_344 = torch.ops.prims.convert_element_type.default(view_352, torch.float32);  view_352 = None
        view_355 = torch.ops.aten.view.default(convert_element_type_344, [2, 8192, 8, -1, 2]);  convert_element_type_344 = None
        view_as_complex_21 = torch.ops.aten.view_as_complex.default(view_355);  view_355 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_as_complex_20, view_16);  view_as_complex_20 = None
        view_as_real_20 = torch.ops.aten.view_as_real.default(mul_82);  mul_82 = None
        view_357 = torch.ops.aten.view.default(view_as_real_20, [2, 8192, 32, 128]);  view_as_real_20 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_as_complex_21, view_16);  view_as_complex_21 = None
        view_as_real_21 = torch.ops.aten.view_as_real.default(mul_83);  mul_83 = None
        view_358 = torch.ops.aten.view.default(view_as_real_21, [2, 8192, 8, 128]);  view_as_real_21 = None
        convert_element_type_345 = torch.ops.prims.convert_element_type.default(view_357, torch.bfloat16);  view_357 = None
        convert_element_type_346 = torch.ops.prims.convert_element_type.default(view_358, torch.bfloat16);  view_358 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(convert_element_type_346, 3);  convert_element_type_346 = None
        expand_20 = torch.ops.aten.expand.default(unsqueeze_20, [2, 8192, 8, 4, 128]);  unsqueeze_20 = None
        clone_20 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_359 = torch.ops.aten.view.default(clone_20, [2, 8192, 32, 128]);  clone_20 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(view_353, 3);  view_353 = None
        expand_21 = torch.ops.aten.expand.default(unsqueeze_21, [2, 8192, 8, 4, 128]);  unsqueeze_21 = None
        clone_21 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_360 = torch.ops.aten.view.default(clone_21, [2, 8192, 32, 128]);  clone_21 = None
        permute_113 = torch.ops.aten.permute.default(convert_element_type_345, [0, 2, 1, 3]);  convert_element_type_345 = None
        permute_114 = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
        permute_115 = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_113, permute_114, permute_115, None, True, 0.0, True);  permute_113 = permute_114 = permute_115 = None
        getitem_90 = _scaled_dot_product_cudnn_attention_10[0]
        getitem_91 = _scaled_dot_product_cudnn_attention_10[1]
        getitem_96 = _scaled_dot_product_cudnn_attention_10[6]
        getitem_97 = _scaled_dot_product_cudnn_attention_10[7];  _scaled_dot_product_cudnn_attention_10 = None
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3])
        view_361 = torch.ops.aten.view.default(permute_116, [2, 8192, -1]);  permute_116 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16)
        all_gather_into_tensor_95 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_347, 64, '0');  convert_element_type_347 = None
        wait_tensor_95 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_95);  all_gather_into_tensor_95 = None
        permute_117 = torch.ops.aten.permute.default(wait_tensor_95, [1, 0]);  wait_tensor_95 = None
        view_363 = torch.ops.aten.view.default(view_361, [16384, 4096]);  view_361 = None
        mm_73 = torch.ops.aten.mm.default(view_363, permute_117);  view_363 = permute_117 = None
        view_364 = torch.ops.aten.view.default(mm_73, [2, 8192, 4096]);  mm_73 = None
        add_41 = torch.ops.aten.add.Tensor(add_39, view_364);  view_364 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16)
        all_gather_into_tensor_96 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_350, 64, '0');  convert_element_type_350 = None
        wait_tensor_96 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_96);  all_gather_into_tensor_96 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_42 = torch.ops.aten.add.Scalar(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_21);  convert_element_type_351 = rsqrt_21 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, wait_tensor_96);  mul_84 = wait_tensor_96 = None
        convert_element_type_352 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16)
        all_gather_into_tensor_97 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_353, 64, '0');  convert_element_type_353 = None
        wait_tensor_97 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_97);  all_gather_into_tensor_97 = None
        permute_118 = torch.ops.aten.permute.default(wait_tensor_97, [1, 0]);  wait_tensor_97 = None
        view_367 = torch.ops.aten.view.default(convert_element_type_352, [16384, 4096]);  convert_element_type_352 = None
        mm_74 = torch.ops.aten.mm.default(view_367, permute_118);  permute_118 = None
        view_368 = torch.ops.aten.view.default(mm_74, [2, 8192, 14336])
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_356)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_356, sigmoid_10);  convert_element_type_356 = sigmoid_10 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16)
        all_gather_into_tensor_98 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_358, 64, '0');  convert_element_type_358 = None
        wait_tensor_98 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_98);  all_gather_into_tensor_98 = None
        permute_119 = torch.ops.aten.permute.default(wait_tensor_98, [1, 0]);  wait_tensor_98 = None
        mm_75 = torch.ops.aten.mm.default(view_367, permute_119);  view_367 = permute_119 = None
        view_371 = torch.ops.aten.view.default(mm_75, [2, 8192, 14336]);  mm_75 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_357, view_371);  convert_element_type_357 = view_371 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16)
        all_gather_into_tensor_99 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_361, 64, '0');  convert_element_type_361 = None
        wait_tensor_99 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_99);  all_gather_into_tensor_99 = None
        permute_120 = torch.ops.aten.permute.default(wait_tensor_99, [1, 0]);  wait_tensor_99 = None
        view_373 = torch.ops.aten.view.default(mul_87, [16384, 14336]);  mul_87 = None
        mm_76 = torch.ops.aten.mm.default(view_373, permute_120);  view_373 = permute_120 = None
        view_374 = torch.ops.aten.view.default(mm_76, [2, 8192, 4096]);  mm_76 = None
        add_43 = torch.ops.aten.add.Tensor(add_41, view_374);  add_41 = view_374 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16)
        all_gather_into_tensor_100 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_364, 64, '0');  convert_element_type_364 = None
        wait_tensor_100 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_100);  all_gather_into_tensor_100 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_365, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [2], True);  pow_23 = None
        add_44 = torch.ops.aten.add.Scalar(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_365, rsqrt_22);  convert_element_type_365 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, wait_tensor_100);  mul_88 = wait_tensor_100 = None
        convert_element_type_366 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16)
        all_gather_into_tensor_101 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_367, 64, '0');  convert_element_type_367 = None
        wait_tensor_101 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_101);  all_gather_into_tensor_101 = None
        permute_121 = torch.ops.aten.permute.default(wait_tensor_101, [1, 0]);  wait_tensor_101 = None
        view_377 = torch.ops.aten.view.default(convert_element_type_366, [16384, 4096]);  convert_element_type_366 = None
        mm_77 = torch.ops.aten.mm.default(view_377, permute_121);  permute_121 = None
        view_378 = torch.ops.aten.view.default(mm_77, [2, 8192, 4096])
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16)
        all_gather_into_tensor_102 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_370, 64, '0');  convert_element_type_370 = None
        wait_tensor_102 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_102);  all_gather_into_tensor_102 = None
        permute_122 = torch.ops.aten.permute.default(wait_tensor_102, [1, 0]);  wait_tensor_102 = None
        mm_78 = torch.ops.aten.mm.default(view_377, permute_122);  permute_122 = None
        view_381 = torch.ops.aten.view.default(mm_78, [2, 8192, 1024]);  mm_78 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16)
        all_gather_into_tensor_103 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_373, 64, '0');  convert_element_type_373 = None
        wait_tensor_103 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_103);  all_gather_into_tensor_103 = None
        permute_123 = torch.ops.aten.permute.default(wait_tensor_103, [1, 0]);  wait_tensor_103 = None
        mm_79 = torch.ops.aten.mm.default(view_377, permute_123);  view_377 = permute_123 = None
        view_384 = torch.ops.aten.view.default(mm_79, [2, 8192, 1024])
        view_385 = torch.ops.aten.view.default(view_378, [2, 8192, -1, 128]);  view_378 = None
        view_386 = torch.ops.aten.view.default(view_381, [2, 8192, -1, 128]);  view_381 = None
        view_387 = torch.ops.aten.view.default(view_384, [2, 8192, -1, 128]);  view_384 = None
        convert_element_type_376 = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        view_388 = torch.ops.aten.view.default(convert_element_type_376, [2, 8192, 32, -1, 2]);  convert_element_type_376 = None
        view_as_complex_22 = torch.ops.aten.view_as_complex.default(view_388);  view_388 = None
        convert_element_type_377 = torch.ops.prims.convert_element_type.default(view_386, torch.float32);  view_386 = None
        view_389 = torch.ops.aten.view.default(convert_element_type_377, [2, 8192, 8, -1, 2]);  convert_element_type_377 = None
        view_as_complex_23 = torch.ops.aten.view_as_complex.default(view_389);  view_389 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_as_complex_22, view_16);  view_as_complex_22 = None
        view_as_real_22 = torch.ops.aten.view_as_real.default(mul_90);  mul_90 = None
        view_391 = torch.ops.aten.view.default(view_as_real_22, [2, 8192, 32, 128]);  view_as_real_22 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_as_complex_23, view_16);  view_as_complex_23 = None
        view_as_real_23 = torch.ops.aten.view_as_real.default(mul_91);  mul_91 = None
        view_392 = torch.ops.aten.view.default(view_as_real_23, [2, 8192, 8, 128]);  view_as_real_23 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(view_391, torch.bfloat16);  view_391 = None
        convert_element_type_379 = torch.ops.prims.convert_element_type.default(view_392, torch.bfloat16);  view_392 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(convert_element_type_379, 3);  convert_element_type_379 = None
        expand_22 = torch.ops.aten.expand.default(unsqueeze_22, [2, 8192, 8, 4, 128]);  unsqueeze_22 = None
        clone_22 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_393 = torch.ops.aten.view.default(clone_22, [2, 8192, 32, 128]);  clone_22 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(view_387, 3);  view_387 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_23, [2, 8192, 8, 4, 128]);  unsqueeze_23 = None
        clone_23 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_394 = torch.ops.aten.view.default(clone_23, [2, 8192, 32, 128]);  clone_23 = None
        permute_124 = torch.ops.aten.permute.default(convert_element_type_378, [0, 2, 1, 3]);  convert_element_type_378 = None
        permute_125 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        permute_126 = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_124, permute_125, permute_126, None, True, 0.0, True);  permute_124 = permute_125 = permute_126 = None
        getitem_99 = _scaled_dot_product_cudnn_attention_11[0]
        getitem_100 = _scaled_dot_product_cudnn_attention_11[1]
        getitem_105 = _scaled_dot_product_cudnn_attention_11[6]
        getitem_106 = _scaled_dot_product_cudnn_attention_11[7];  _scaled_dot_product_cudnn_attention_11 = None
        permute_127 = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3])
        view_395 = torch.ops.aten.view.default(permute_127, [2, 8192, -1]);  permute_127 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16)
        all_gather_into_tensor_104 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_380, 64, '0');  convert_element_type_380 = None
        wait_tensor_104 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_104);  all_gather_into_tensor_104 = None
        permute_128 = torch.ops.aten.permute.default(wait_tensor_104, [1, 0]);  wait_tensor_104 = None
        view_397 = torch.ops.aten.view.default(view_395, [16384, 4096]);  view_395 = None
        mm_80 = torch.ops.aten.mm.default(view_397, permute_128);  view_397 = permute_128 = None
        view_398 = torch.ops.aten.view.default(mm_80, [2, 8192, 4096]);  mm_80 = None
        add_45 = torch.ops.aten.add.Tensor(add_43, view_398);  view_398 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16)
        all_gather_into_tensor_105 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_383, 64, '0');  convert_element_type_383 = None
        wait_tensor_105 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_105);  all_gather_into_tensor_105 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [2], True);  pow_24 = None
        add_46 = torch.ops.aten.add.Scalar(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_23);  convert_element_type_384 = rsqrt_23 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, wait_tensor_105);  mul_92 = wait_tensor_105 = None
        convert_element_type_385 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16)
        all_gather_into_tensor_106 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_386, 64, '0');  convert_element_type_386 = None
        wait_tensor_106 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_106);  all_gather_into_tensor_106 = None
        permute_129 = torch.ops.aten.permute.default(wait_tensor_106, [1, 0]);  wait_tensor_106 = None
        view_401 = torch.ops.aten.view.default(convert_element_type_385, [16384, 4096]);  convert_element_type_385 = None
        mm_81 = torch.ops.aten.mm.default(view_401, permute_129);  permute_129 = None
        view_402 = torch.ops.aten.view.default(mm_81, [2, 8192, 14336])
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_389)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_389, sigmoid_11);  convert_element_type_389 = sigmoid_11 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16)
        all_gather_into_tensor_107 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_391, 64, '0');  convert_element_type_391 = None
        wait_tensor_107 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_107);  all_gather_into_tensor_107 = None
        permute_130 = torch.ops.aten.permute.default(wait_tensor_107, [1, 0]);  wait_tensor_107 = None
        mm_82 = torch.ops.aten.mm.default(view_401, permute_130);  view_401 = permute_130 = None
        view_405 = torch.ops.aten.view.default(mm_82, [2, 8192, 14336]);  mm_82 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_390, view_405);  convert_element_type_390 = view_405 = None
        convert_element_type_394 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16)
        all_gather_into_tensor_108 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_394, 64, '0');  convert_element_type_394 = None
        wait_tensor_108 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_108);  all_gather_into_tensor_108 = None
        permute_131 = torch.ops.aten.permute.default(wait_tensor_108, [1, 0]);  wait_tensor_108 = None
        view_407 = torch.ops.aten.view.default(mul_95, [16384, 14336]);  mul_95 = None
        mm_83 = torch.ops.aten.mm.default(view_407, permute_131);  view_407 = permute_131 = None
        view_408 = torch.ops.aten.view.default(mm_83, [2, 8192, 4096]);  mm_83 = None
        add_47 = torch.ops.aten.add.Tensor(add_45, view_408);  add_45 = view_408 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16)
        all_gather_into_tensor_109 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_397, 64, '0');  convert_element_type_397 = None
        wait_tensor_109 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_109);  all_gather_into_tensor_109 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_48 = torch.ops.aten.add.Scalar(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_24);  convert_element_type_398 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, wait_tensor_109);  mul_96 = wait_tensor_109 = None
        convert_element_type_399 = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16)
        all_gather_into_tensor_110 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_400, 64, '0');  convert_element_type_400 = None
        wait_tensor_110 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_110);  all_gather_into_tensor_110 = None
        permute_132 = torch.ops.aten.permute.default(wait_tensor_110, [1, 0]);  wait_tensor_110 = None
        view_411 = torch.ops.aten.view.default(convert_element_type_399, [16384, 4096]);  convert_element_type_399 = None
        mm_84 = torch.ops.aten.mm.default(view_411, permute_132);  permute_132 = None
        view_412 = torch.ops.aten.view.default(mm_84, [2, 8192, 4096])
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16)
        all_gather_into_tensor_111 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_403, 64, '0');  convert_element_type_403 = None
        wait_tensor_111 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_111);  all_gather_into_tensor_111 = None
        permute_133 = torch.ops.aten.permute.default(wait_tensor_111, [1, 0]);  wait_tensor_111 = None
        mm_85 = torch.ops.aten.mm.default(view_411, permute_133);  permute_133 = None
        view_415 = torch.ops.aten.view.default(mm_85, [2, 8192, 1024]);  mm_85 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16)
        all_gather_into_tensor_112 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_406, 64, '0');  convert_element_type_406 = None
        wait_tensor_112 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_112);  all_gather_into_tensor_112 = None
        permute_134 = torch.ops.aten.permute.default(wait_tensor_112, [1, 0]);  wait_tensor_112 = None
        mm_86 = torch.ops.aten.mm.default(view_411, permute_134);  view_411 = permute_134 = None
        view_418 = torch.ops.aten.view.default(mm_86, [2, 8192, 1024])
        view_419 = torch.ops.aten.view.default(view_412, [2, 8192, -1, 128]);  view_412 = None
        view_420 = torch.ops.aten.view.default(view_415, [2, 8192, -1, 128]);  view_415 = None
        view_421 = torch.ops.aten.view.default(view_418, [2, 8192, -1, 128]);  view_418 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        view_422 = torch.ops.aten.view.default(convert_element_type_409, [2, 8192, 32, -1, 2]);  convert_element_type_409 = None
        view_as_complex_24 = torch.ops.aten.view_as_complex.default(view_422);  view_422 = None
        convert_element_type_410 = torch.ops.prims.convert_element_type.default(view_420, torch.float32);  view_420 = None
        view_423 = torch.ops.aten.view.default(convert_element_type_410, [2, 8192, 8, -1, 2]);  convert_element_type_410 = None
        view_as_complex_25 = torch.ops.aten.view_as_complex.default(view_423);  view_423 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_as_complex_24, view_16);  view_as_complex_24 = None
        view_as_real_24 = torch.ops.aten.view_as_real.default(mul_98);  mul_98 = None
        view_425 = torch.ops.aten.view.default(view_as_real_24, [2, 8192, 32, 128]);  view_as_real_24 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_as_complex_25, view_16);  view_as_complex_25 = None
        view_as_real_25 = torch.ops.aten.view_as_real.default(mul_99);  mul_99 = None
        view_426 = torch.ops.aten.view.default(view_as_real_25, [2, 8192, 8, 128]);  view_as_real_25 = None
        convert_element_type_411 = torch.ops.prims.convert_element_type.default(view_425, torch.bfloat16);  view_425 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(view_426, torch.bfloat16);  view_426 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(convert_element_type_412, 3);  convert_element_type_412 = None
        expand_24 = torch.ops.aten.expand.default(unsqueeze_24, [2, 8192, 8, 4, 128]);  unsqueeze_24 = None
        clone_24 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_427 = torch.ops.aten.view.default(clone_24, [2, 8192, 32, 128]);  clone_24 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(view_421, 3);  view_421 = None
        expand_25 = torch.ops.aten.expand.default(unsqueeze_25, [2, 8192, 8, 4, 128]);  unsqueeze_25 = None
        clone_25 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_428 = torch.ops.aten.view.default(clone_25, [2, 8192, 32, 128]);  clone_25 = None
        permute_135 = torch.ops.aten.permute.default(convert_element_type_411, [0, 2, 1, 3]);  convert_element_type_411 = None
        permute_136 = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        permute_137 = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_135, permute_136, permute_137, None, True, 0.0, True);  permute_135 = permute_136 = permute_137 = None
        getitem_108 = _scaled_dot_product_cudnn_attention_12[0]
        getitem_109 = _scaled_dot_product_cudnn_attention_12[1]
        getitem_114 = _scaled_dot_product_cudnn_attention_12[6]
        getitem_115 = _scaled_dot_product_cudnn_attention_12[7];  _scaled_dot_product_cudnn_attention_12 = None
        permute_138 = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])
        view_429 = torch.ops.aten.view.default(permute_138, [2, 8192, -1]);  permute_138 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16)
        all_gather_into_tensor_113 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_413, 64, '0');  convert_element_type_413 = None
        wait_tensor_113 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_113);  all_gather_into_tensor_113 = None
        permute_139 = torch.ops.aten.permute.default(wait_tensor_113, [1, 0]);  wait_tensor_113 = None
        view_431 = torch.ops.aten.view.default(view_429, [16384, 4096]);  view_429 = None
        mm_87 = torch.ops.aten.mm.default(view_431, permute_139);  view_431 = permute_139 = None
        view_432 = torch.ops.aten.view.default(mm_87, [2, 8192, 4096]);  mm_87 = None
        add_49 = torch.ops.aten.add.Tensor(add_47, view_432);  view_432 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16)
        all_gather_into_tensor_114 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_416, 64, '0');  convert_element_type_416 = None
        wait_tensor_114 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_114);  all_gather_into_tensor_114 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_417, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [2], True);  pow_26 = None
        add_50 = torch.ops.aten.add.Scalar(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_25);  convert_element_type_417 = rsqrt_25 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, wait_tensor_114);  mul_100 = wait_tensor_114 = None
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16)
        all_gather_into_tensor_115 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_419, 64, '0');  convert_element_type_419 = None
        wait_tensor_115 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_115);  all_gather_into_tensor_115 = None
        permute_140 = torch.ops.aten.permute.default(wait_tensor_115, [1, 0]);  wait_tensor_115 = None
        view_435 = torch.ops.aten.view.default(convert_element_type_418, [16384, 4096]);  convert_element_type_418 = None
        mm_88 = torch.ops.aten.mm.default(view_435, permute_140);  permute_140 = None
        view_436 = torch.ops.aten.view.default(mm_88, [2, 8192, 14336])
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_422)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_422, sigmoid_12);  convert_element_type_422 = sigmoid_12 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16)
        all_gather_into_tensor_116 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_424, 64, '0');  convert_element_type_424 = None
        wait_tensor_116 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_116);  all_gather_into_tensor_116 = None
        permute_141 = torch.ops.aten.permute.default(wait_tensor_116, [1, 0]);  wait_tensor_116 = None
        mm_89 = torch.ops.aten.mm.default(view_435, permute_141);  view_435 = permute_141 = None
        view_439 = torch.ops.aten.view.default(mm_89, [2, 8192, 14336]);  mm_89 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_423, view_439);  convert_element_type_423 = view_439 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16)
        all_gather_into_tensor_117 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_427, 64, '0');  convert_element_type_427 = None
        wait_tensor_117 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_117);  all_gather_into_tensor_117 = None
        permute_142 = torch.ops.aten.permute.default(wait_tensor_117, [1, 0]);  wait_tensor_117 = None
        view_441 = torch.ops.aten.view.default(mul_103, [16384, 14336]);  mul_103 = None
        mm_90 = torch.ops.aten.mm.default(view_441, permute_142);  view_441 = permute_142 = None
        view_442 = torch.ops.aten.view.default(mm_90, [2, 8192, 4096]);  mm_90 = None
        add_51 = torch.ops.aten.add.Tensor(add_49, view_442);  add_49 = view_442 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16)
        all_gather_into_tensor_118 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_430, 64, '0');  convert_element_type_430 = None
        wait_tensor_118 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_118);  all_gather_into_tensor_118 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_52 = torch.ops.aten.add.Scalar(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_431, rsqrt_26);  convert_element_type_431 = rsqrt_26 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, wait_tensor_118);  mul_104 = wait_tensor_118 = None
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        convert_element_type_433 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16)
        all_gather_into_tensor_119 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_433, 64, '0');  convert_element_type_433 = None
        wait_tensor_119 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_119);  all_gather_into_tensor_119 = None
        permute_143 = torch.ops.aten.permute.default(wait_tensor_119, [1, 0]);  wait_tensor_119 = None
        view_445 = torch.ops.aten.view.default(convert_element_type_432, [16384, 4096]);  convert_element_type_432 = None
        mm_91 = torch.ops.aten.mm.default(view_445, permute_143);  permute_143 = None
        view_446 = torch.ops.aten.view.default(mm_91, [2, 8192, 4096])
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16)
        all_gather_into_tensor_120 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_436, 64, '0');  convert_element_type_436 = None
        wait_tensor_120 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_120);  all_gather_into_tensor_120 = None
        permute_144 = torch.ops.aten.permute.default(wait_tensor_120, [1, 0]);  wait_tensor_120 = None
        mm_92 = torch.ops.aten.mm.default(view_445, permute_144);  permute_144 = None
        view_449 = torch.ops.aten.view.default(mm_92, [2, 8192, 1024]);  mm_92 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16)
        all_gather_into_tensor_121 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_439, 64, '0');  convert_element_type_439 = None
        wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_121);  all_gather_into_tensor_121 = None
        permute_145 = torch.ops.aten.permute.default(wait_tensor_121, [1, 0]);  wait_tensor_121 = None
        mm_93 = torch.ops.aten.mm.default(view_445, permute_145);  view_445 = permute_145 = None
        view_452 = torch.ops.aten.view.default(mm_93, [2, 8192, 1024])
        view_453 = torch.ops.aten.view.default(view_446, [2, 8192, -1, 128]);  view_446 = None
        view_454 = torch.ops.aten.view.default(view_449, [2, 8192, -1, 128]);  view_449 = None
        view_455 = torch.ops.aten.view.default(view_452, [2, 8192, -1, 128]);  view_452 = None
        convert_element_type_442 = torch.ops.prims.convert_element_type.default(view_453, torch.float32);  view_453 = None
        view_456 = torch.ops.aten.view.default(convert_element_type_442, [2, 8192, 32, -1, 2]);  convert_element_type_442 = None
        view_as_complex_26 = torch.ops.aten.view_as_complex.default(view_456);  view_456 = None
        convert_element_type_443 = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        view_457 = torch.ops.aten.view.default(convert_element_type_443, [2, 8192, 8, -1, 2]);  convert_element_type_443 = None
        view_as_complex_27 = torch.ops.aten.view_as_complex.default(view_457);  view_457 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_as_complex_26, view_16);  view_as_complex_26 = None
        view_as_real_26 = torch.ops.aten.view_as_real.default(mul_106);  mul_106 = None
        view_459 = torch.ops.aten.view.default(view_as_real_26, [2, 8192, 32, 128]);  view_as_real_26 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_as_complex_27, view_16);  view_as_complex_27 = None
        view_as_real_27 = torch.ops.aten.view_as_real.default(mul_107);  mul_107 = None
        view_460 = torch.ops.aten.view.default(view_as_real_27, [2, 8192, 8, 128]);  view_as_real_27 = None
        convert_element_type_444 = torch.ops.prims.convert_element_type.default(view_459, torch.bfloat16);  view_459 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(view_460, torch.bfloat16);  view_460 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(convert_element_type_445, 3);  convert_element_type_445 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_26, [2, 8192, 8, 4, 128]);  unsqueeze_26 = None
        clone_26 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_461 = torch.ops.aten.view.default(clone_26, [2, 8192, 32, 128]);  clone_26 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(view_455, 3);  view_455 = None
        expand_27 = torch.ops.aten.expand.default(unsqueeze_27, [2, 8192, 8, 4, 128]);  unsqueeze_27 = None
        clone_27 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_462 = torch.ops.aten.view.default(clone_27, [2, 8192, 32, 128]);  clone_27 = None
        permute_146 = torch.ops.aten.permute.default(convert_element_type_444, [0, 2, 1, 3]);  convert_element_type_444 = None
        permute_147 = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
        permute_148 = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_146, permute_147, permute_148, None, True, 0.0, True);  permute_146 = permute_147 = permute_148 = None
        getitem_117 = _scaled_dot_product_cudnn_attention_13[0]
        getitem_118 = _scaled_dot_product_cudnn_attention_13[1]
        getitem_123 = _scaled_dot_product_cudnn_attention_13[6]
        getitem_124 = _scaled_dot_product_cudnn_attention_13[7];  _scaled_dot_product_cudnn_attention_13 = None
        permute_149 = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_463 = torch.ops.aten.view.default(permute_149, [2, 8192, -1]);  permute_149 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16)
        all_gather_into_tensor_122 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_446, 64, '0');  convert_element_type_446 = None
        wait_tensor_122 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_122);  all_gather_into_tensor_122 = None
        permute_150 = torch.ops.aten.permute.default(wait_tensor_122, [1, 0]);  wait_tensor_122 = None
        view_465 = torch.ops.aten.view.default(view_463, [16384, 4096]);  view_463 = None
        mm_94 = torch.ops.aten.mm.default(view_465, permute_150);  view_465 = permute_150 = None
        view_466 = torch.ops.aten.view.default(mm_94, [2, 8192, 4096]);  mm_94 = None
        add_53 = torch.ops.aten.add.Tensor(add_51, view_466);  view_466 = None
        convert_element_type_449 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16)
        all_gather_into_tensor_123 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_449, 64, '0');  convert_element_type_449 = None
        wait_tensor_123 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_123);  all_gather_into_tensor_123 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_450, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [2], True);  pow_28 = None
        add_54 = torch.ops.aten.add.Scalar(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_27);  convert_element_type_450 = rsqrt_27 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, wait_tensor_123);  mul_108 = wait_tensor_123 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16)
        all_gather_into_tensor_124 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_452, 64, '0');  convert_element_type_452 = None
        wait_tensor_124 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_124);  all_gather_into_tensor_124 = None
        permute_151 = torch.ops.aten.permute.default(wait_tensor_124, [1, 0]);  wait_tensor_124 = None
        view_469 = torch.ops.aten.view.default(convert_element_type_451, [16384, 4096]);  convert_element_type_451 = None
        mm_95 = torch.ops.aten.mm.default(view_469, permute_151);  permute_151 = None
        view_470 = torch.ops.aten.view.default(mm_95, [2, 8192, 14336])
        convert_element_type_455 = torch.ops.prims.convert_element_type.default(view_470, torch.float32);  view_470 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_455)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_13);  convert_element_type_455 = sigmoid_13 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16)
        all_gather_into_tensor_125 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_457, 64, '0');  convert_element_type_457 = None
        wait_tensor_125 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_125);  all_gather_into_tensor_125 = None
        permute_152 = torch.ops.aten.permute.default(wait_tensor_125, [1, 0]);  wait_tensor_125 = None
        mm_96 = torch.ops.aten.mm.default(view_469, permute_152);  view_469 = permute_152 = None
        view_473 = torch.ops.aten.view.default(mm_96, [2, 8192, 14336]);  mm_96 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_456, view_473);  convert_element_type_456 = view_473 = None
        convert_element_type_460 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16)
        all_gather_into_tensor_126 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_460, 64, '0');  convert_element_type_460 = None
        wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_126);  all_gather_into_tensor_126 = None
        permute_153 = torch.ops.aten.permute.default(wait_tensor_126, [1, 0]);  wait_tensor_126 = None
        view_475 = torch.ops.aten.view.default(mul_111, [16384, 14336]);  mul_111 = None
        mm_97 = torch.ops.aten.mm.default(view_475, permute_153);  view_475 = permute_153 = None
        view_476 = torch.ops.aten.view.default(mm_97, [2, 8192, 4096]);  mm_97 = None
        add_55 = torch.ops.aten.add.Tensor(add_53, view_476);  add_53 = view_476 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16)
        all_gather_into_tensor_127 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_463, 64, '0');  convert_element_type_463 = None
        wait_tensor_127 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_127);  all_gather_into_tensor_127 = None
        convert_element_type_464 = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [2], True);  pow_29 = None
        add_56 = torch.ops.aten.add.Scalar(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_28);  convert_element_type_464 = rsqrt_28 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, wait_tensor_127);  mul_112 = wait_tensor_127 = None
        convert_element_type_465 = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16)
        all_gather_into_tensor_128 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_466, 64, '0');  convert_element_type_466 = None
        wait_tensor_128 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_128);  all_gather_into_tensor_128 = None
        permute_154 = torch.ops.aten.permute.default(wait_tensor_128, [1, 0]);  wait_tensor_128 = None
        view_479 = torch.ops.aten.view.default(convert_element_type_465, [16384, 4096]);  convert_element_type_465 = None
        mm_98 = torch.ops.aten.mm.default(view_479, permute_154);  permute_154 = None
        view_480 = torch.ops.aten.view.default(mm_98, [2, 8192, 4096])
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16)
        all_gather_into_tensor_129 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_469, 64, '0');  convert_element_type_469 = None
        wait_tensor_129 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_129);  all_gather_into_tensor_129 = None
        permute_155 = torch.ops.aten.permute.default(wait_tensor_129, [1, 0]);  wait_tensor_129 = None
        mm_99 = torch.ops.aten.mm.default(view_479, permute_155);  permute_155 = None
        view_483 = torch.ops.aten.view.default(mm_99, [2, 8192, 1024]);  mm_99 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16)
        all_gather_into_tensor_130 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_472, 64, '0');  convert_element_type_472 = None
        wait_tensor_130 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_130);  all_gather_into_tensor_130 = None
        permute_156 = torch.ops.aten.permute.default(wait_tensor_130, [1, 0]);  wait_tensor_130 = None
        mm_100 = torch.ops.aten.mm.default(view_479, permute_156);  view_479 = permute_156 = None
        view_486 = torch.ops.aten.view.default(mm_100, [2, 8192, 1024])
        view_487 = torch.ops.aten.view.default(view_480, [2, 8192, -1, 128]);  view_480 = None
        view_488 = torch.ops.aten.view.default(view_483, [2, 8192, -1, 128]);  view_483 = None
        view_489 = torch.ops.aten.view.default(view_486, [2, 8192, -1, 128]);  view_486 = None
        convert_element_type_475 = torch.ops.prims.convert_element_type.default(view_487, torch.float32);  view_487 = None
        view_490 = torch.ops.aten.view.default(convert_element_type_475, [2, 8192, 32, -1, 2]);  convert_element_type_475 = None
        view_as_complex_28 = torch.ops.aten.view_as_complex.default(view_490);  view_490 = None
        convert_element_type_476 = torch.ops.prims.convert_element_type.default(view_488, torch.float32);  view_488 = None
        view_491 = torch.ops.aten.view.default(convert_element_type_476, [2, 8192, 8, -1, 2]);  convert_element_type_476 = None
        view_as_complex_29 = torch.ops.aten.view_as_complex.default(view_491);  view_491 = None
        mul_114 = torch.ops.aten.mul.Tensor(view_as_complex_28, view_16);  view_as_complex_28 = None
        view_as_real_28 = torch.ops.aten.view_as_real.default(mul_114);  mul_114 = None
        view_493 = torch.ops.aten.view.default(view_as_real_28, [2, 8192, 32, 128]);  view_as_real_28 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_as_complex_29, view_16);  view_as_complex_29 = None
        view_as_real_29 = torch.ops.aten.view_as_real.default(mul_115);  mul_115 = None
        view_494 = torch.ops.aten.view.default(view_as_real_29, [2, 8192, 8, 128]);  view_as_real_29 = None
        convert_element_type_477 = torch.ops.prims.convert_element_type.default(view_493, torch.bfloat16);  view_493 = None
        convert_element_type_478 = torch.ops.prims.convert_element_type.default(view_494, torch.bfloat16);  view_494 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(convert_element_type_478, 3);  convert_element_type_478 = None
        expand_28 = torch.ops.aten.expand.default(unsqueeze_28, [2, 8192, 8, 4, 128]);  unsqueeze_28 = None
        clone_28 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_495 = torch.ops.aten.view.default(clone_28, [2, 8192, 32, 128]);  clone_28 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(view_489, 3);  view_489 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_29, [2, 8192, 8, 4, 128]);  unsqueeze_29 = None
        clone_29 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_496 = torch.ops.aten.view.default(clone_29, [2, 8192, 32, 128]);  clone_29 = None
        permute_157 = torch.ops.aten.permute.default(convert_element_type_477, [0, 2, 1, 3]);  convert_element_type_477 = None
        permute_158 = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
        permute_159 = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_157, permute_158, permute_159, None, True, 0.0, True);  permute_157 = permute_158 = permute_159 = None
        getitem_126 = _scaled_dot_product_cudnn_attention_14[0]
        getitem_127 = _scaled_dot_product_cudnn_attention_14[1]
        getitem_132 = _scaled_dot_product_cudnn_attention_14[6]
        getitem_133 = _scaled_dot_product_cudnn_attention_14[7];  _scaled_dot_product_cudnn_attention_14 = None
        permute_160 = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_497 = torch.ops.aten.view.default(permute_160, [2, 8192, -1]);  permute_160 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16)
        all_gather_into_tensor_131 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_479, 64, '0');  convert_element_type_479 = None
        wait_tensor_131 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_131);  all_gather_into_tensor_131 = None
        permute_161 = torch.ops.aten.permute.default(wait_tensor_131, [1, 0]);  wait_tensor_131 = None
        view_499 = torch.ops.aten.view.default(view_497, [16384, 4096]);  view_497 = None
        mm_101 = torch.ops.aten.mm.default(view_499, permute_161);  view_499 = permute_161 = None
        view_500 = torch.ops.aten.view.default(mm_101, [2, 8192, 4096]);  mm_101 = None
        add_57 = torch.ops.aten.add.Tensor(add_55, view_500);  view_500 = None
        convert_element_type_482 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16)
        all_gather_into_tensor_132 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_482, 64, '0');  convert_element_type_482 = None
        wait_tensor_132 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_132);  all_gather_into_tensor_132 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_58 = torch.ops.aten.add.Scalar(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_29);  convert_element_type_483 = rsqrt_29 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, wait_tensor_132);  mul_116 = wait_tensor_132 = None
        convert_element_type_484 = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16)
        all_gather_into_tensor_133 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_485, 64, '0');  convert_element_type_485 = None
        wait_tensor_133 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_133);  all_gather_into_tensor_133 = None
        permute_162 = torch.ops.aten.permute.default(wait_tensor_133, [1, 0]);  wait_tensor_133 = None
        view_503 = torch.ops.aten.view.default(convert_element_type_484, [16384, 4096]);  convert_element_type_484 = None
        mm_102 = torch.ops.aten.mm.default(view_503, permute_162);  permute_162 = None
        view_504 = torch.ops.aten.view.default(mm_102, [2, 8192, 14336])
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(view_504, torch.float32);  view_504 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_488)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_488, sigmoid_14);  convert_element_type_488 = sigmoid_14 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16)
        all_gather_into_tensor_134 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_490, 64, '0');  convert_element_type_490 = None
        wait_tensor_134 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_134);  all_gather_into_tensor_134 = None
        permute_163 = torch.ops.aten.permute.default(wait_tensor_134, [1, 0]);  wait_tensor_134 = None
        mm_103 = torch.ops.aten.mm.default(view_503, permute_163);  view_503 = permute_163 = None
        view_507 = torch.ops.aten.view.default(mm_103, [2, 8192, 14336]);  mm_103 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_489, view_507);  convert_element_type_489 = view_507 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16)
        all_gather_into_tensor_135 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_493, 64, '0');  convert_element_type_493 = None
        wait_tensor_135 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_135);  all_gather_into_tensor_135 = None
        permute_164 = torch.ops.aten.permute.default(wait_tensor_135, [1, 0]);  wait_tensor_135 = None
        view_509 = torch.ops.aten.view.default(mul_119, [16384, 14336]);  mul_119 = None
        mm_104 = torch.ops.aten.mm.default(view_509, permute_164);  view_509 = permute_164 = None
        view_510 = torch.ops.aten.view.default(mm_104, [2, 8192, 4096]);  mm_104 = None
        add_59 = torch.ops.aten.add.Tensor(add_57, view_510);  add_57 = view_510 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16)
        all_gather_into_tensor_136 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_496, 64, '0');  convert_element_type_496 = None
        wait_tensor_136 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_136);  all_gather_into_tensor_136 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [2], True);  pow_31 = None
        add_60 = torch.ops.aten.add.Scalar(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_30);  convert_element_type_497 = rsqrt_30 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, wait_tensor_136);  mul_120 = wait_tensor_136 = None
        convert_element_type_498 = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        convert_element_type_499 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16)
        all_gather_into_tensor_137 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_499, 64, '0');  convert_element_type_499 = None
        wait_tensor_137 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_137);  all_gather_into_tensor_137 = None
        permute_165 = torch.ops.aten.permute.default(wait_tensor_137, [1, 0]);  wait_tensor_137 = None
        view_513 = torch.ops.aten.view.default(convert_element_type_498, [16384, 4096]);  convert_element_type_498 = None
        mm_105 = torch.ops.aten.mm.default(view_513, permute_165);  permute_165 = None
        view_514 = torch.ops.aten.view.default(mm_105, [2, 8192, 4096])
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16)
        all_gather_into_tensor_138 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_502, 64, '0');  convert_element_type_502 = None
        wait_tensor_138 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_138);  all_gather_into_tensor_138 = None
        permute_166 = torch.ops.aten.permute.default(wait_tensor_138, [1, 0]);  wait_tensor_138 = None
        mm_106 = torch.ops.aten.mm.default(view_513, permute_166);  permute_166 = None
        view_517 = torch.ops.aten.view.default(mm_106, [2, 8192, 1024]);  mm_106 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16)
        all_gather_into_tensor_139 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_505, 64, '0');  convert_element_type_505 = None
        wait_tensor_139 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_139);  all_gather_into_tensor_139 = None
        permute_167 = torch.ops.aten.permute.default(wait_tensor_139, [1, 0]);  wait_tensor_139 = None
        mm_107 = torch.ops.aten.mm.default(view_513, permute_167);  view_513 = permute_167 = None
        view_520 = torch.ops.aten.view.default(mm_107, [2, 8192, 1024])
        view_521 = torch.ops.aten.view.default(view_514, [2, 8192, -1, 128]);  view_514 = None
        view_522 = torch.ops.aten.view.default(view_517, [2, 8192, -1, 128]);  view_517 = None
        view_523 = torch.ops.aten.view.default(view_520, [2, 8192, -1, 128]);  view_520 = None
        convert_element_type_508 = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        view_524 = torch.ops.aten.view.default(convert_element_type_508, [2, 8192, 32, -1, 2]);  convert_element_type_508 = None
        view_as_complex_30 = torch.ops.aten.view_as_complex.default(view_524);  view_524 = None
        convert_element_type_509 = torch.ops.prims.convert_element_type.default(view_522, torch.float32);  view_522 = None
        view_525 = torch.ops.aten.view.default(convert_element_type_509, [2, 8192, 8, -1, 2]);  convert_element_type_509 = None
        view_as_complex_31 = torch.ops.aten.view_as_complex.default(view_525);  view_525 = None
        mul_122 = torch.ops.aten.mul.Tensor(view_as_complex_30, view_16);  view_as_complex_30 = None
        view_as_real_30 = torch.ops.aten.view_as_real.default(mul_122);  mul_122 = None
        view_527 = torch.ops.aten.view.default(view_as_real_30, [2, 8192, 32, 128]);  view_as_real_30 = None
        mul_123 = torch.ops.aten.mul.Tensor(view_as_complex_31, view_16);  view_as_complex_31 = None
        view_as_real_31 = torch.ops.aten.view_as_real.default(mul_123);  mul_123 = None
        view_528 = torch.ops.aten.view.default(view_as_real_31, [2, 8192, 8, 128]);  view_as_real_31 = None
        convert_element_type_510 = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        convert_element_type_511 = torch.ops.prims.convert_element_type.default(view_528, torch.bfloat16);  view_528 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(convert_element_type_511, 3);  convert_element_type_511 = None
        expand_30 = torch.ops.aten.expand.default(unsqueeze_30, [2, 8192, 8, 4, 128]);  unsqueeze_30 = None
        clone_30 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_529 = torch.ops.aten.view.default(clone_30, [2, 8192, 32, 128]);  clone_30 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(view_523, 3);  view_523 = None
        expand_31 = torch.ops.aten.expand.default(unsqueeze_31, [2, 8192, 8, 4, 128]);  unsqueeze_31 = None
        clone_31 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_530 = torch.ops.aten.view.default(clone_31, [2, 8192, 32, 128]);  clone_31 = None
        permute_168 = torch.ops.aten.permute.default(convert_element_type_510, [0, 2, 1, 3]);  convert_element_type_510 = None
        permute_169 = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
        permute_170 = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_168, permute_169, permute_170, None, True, 0.0, True);  permute_168 = permute_169 = permute_170 = None
        getitem_135 = _scaled_dot_product_cudnn_attention_15[0]
        getitem_136 = _scaled_dot_product_cudnn_attention_15[1]
        getitem_141 = _scaled_dot_product_cudnn_attention_15[6]
        getitem_142 = _scaled_dot_product_cudnn_attention_15[7];  _scaled_dot_product_cudnn_attention_15 = None
        permute_171 = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3])
        view_531 = torch.ops.aten.view.default(permute_171, [2, 8192, -1]);  permute_171 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16)
        all_gather_into_tensor_140 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_512, 64, '0');  convert_element_type_512 = None
        wait_tensor_140 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_140);  all_gather_into_tensor_140 = None
        permute_172 = torch.ops.aten.permute.default(wait_tensor_140, [1, 0]);  wait_tensor_140 = None
        view_533 = torch.ops.aten.view.default(view_531, [16384, 4096]);  view_531 = None
        mm_108 = torch.ops.aten.mm.default(view_533, permute_172);  view_533 = permute_172 = None
        view_534 = torch.ops.aten.view.default(mm_108, [2, 8192, 4096]);  mm_108 = None
        add_61 = torch.ops.aten.add.Tensor(add_59, view_534);  view_534 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16)
        all_gather_into_tensor_141 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_515, 64, '0');  convert_element_type_515 = None
        wait_tensor_141 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_141);  all_gather_into_tensor_141 = None
        convert_element_type_516 = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_516, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_62 = torch.ops.aten.add.Scalar(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_516, rsqrt_31);  convert_element_type_516 = rsqrt_31 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, wait_tensor_141);  mul_124 = wait_tensor_141 = None
        convert_element_type_517 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16)
        all_gather_into_tensor_142 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_518, 64, '0');  convert_element_type_518 = None
        wait_tensor_142 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_142);  all_gather_into_tensor_142 = None
        permute_173 = torch.ops.aten.permute.default(wait_tensor_142, [1, 0]);  wait_tensor_142 = None
        view_537 = torch.ops.aten.view.default(convert_element_type_517, [16384, 4096]);  convert_element_type_517 = None
        mm_109 = torch.ops.aten.mm.default(view_537, permute_173);  permute_173 = None
        view_538 = torch.ops.aten.view.default(mm_109, [2, 8192, 14336])
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(view_538, torch.float32);  view_538 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_521)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_521, sigmoid_15);  convert_element_type_521 = sigmoid_15 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16)
        all_gather_into_tensor_143 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_523, 64, '0');  convert_element_type_523 = None
        wait_tensor_143 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_143);  all_gather_into_tensor_143 = None
        permute_174 = torch.ops.aten.permute.default(wait_tensor_143, [1, 0]);  wait_tensor_143 = None
        mm_110 = torch.ops.aten.mm.default(view_537, permute_174);  view_537 = permute_174 = None
        view_541 = torch.ops.aten.view.default(mm_110, [2, 8192, 14336]);  mm_110 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_522, view_541);  convert_element_type_522 = view_541 = None
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16)
        all_gather_into_tensor_144 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_526, 64, '0');  convert_element_type_526 = None
        wait_tensor_144 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_144);  all_gather_into_tensor_144 = None
        permute_175 = torch.ops.aten.permute.default(wait_tensor_144, [1, 0]);  wait_tensor_144 = None
        view_543 = torch.ops.aten.view.default(mul_127, [16384, 14336]);  mul_127 = None
        mm_111 = torch.ops.aten.mm.default(view_543, permute_175);  view_543 = permute_175 = None
        view_544 = torch.ops.aten.view.default(mm_111, [2, 8192, 4096]);  mm_111 = None
        add_63 = torch.ops.aten.add.Tensor(add_61, view_544);  add_61 = view_544 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16)
        all_gather_into_tensor_145 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_529, 64, '0');  convert_element_type_529 = None
        wait_tensor_145 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_145);  all_gather_into_tensor_145 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_530, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_33, [2], True);  pow_33 = None
        add_64 = torch.ops.aten.add.Scalar(mean_32, 1e-05);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_32);  convert_element_type_530 = rsqrt_32 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, wait_tensor_145);  mul_128 = wait_tensor_145 = None
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16)
        all_gather_into_tensor_146 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_532, 64, '0');  convert_element_type_532 = None
        wait_tensor_146 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_146);  all_gather_into_tensor_146 = None
        permute_176 = torch.ops.aten.permute.default(wait_tensor_146, [1, 0]);  wait_tensor_146 = None
        view_547 = torch.ops.aten.view.default(convert_element_type_531, [16384, 4096]);  convert_element_type_531 = None
        mm_112 = torch.ops.aten.mm.default(view_547, permute_176);  permute_176 = None
        view_548 = torch.ops.aten.view.default(mm_112, [2, 8192, 4096])
        convert_element_type_535 = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16)
        all_gather_into_tensor_147 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_535, 64, '0');  convert_element_type_535 = None
        wait_tensor_147 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_147);  all_gather_into_tensor_147 = None
        permute_177 = torch.ops.aten.permute.default(wait_tensor_147, [1, 0]);  wait_tensor_147 = None
        mm_113 = torch.ops.aten.mm.default(view_547, permute_177);  permute_177 = None
        view_551 = torch.ops.aten.view.default(mm_113, [2, 8192, 1024]);  mm_113 = None
        convert_element_type_538 = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16)
        all_gather_into_tensor_148 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_538, 64, '0');  convert_element_type_538 = None
        wait_tensor_148 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_148);  all_gather_into_tensor_148 = None
        permute_178 = torch.ops.aten.permute.default(wait_tensor_148, [1, 0]);  wait_tensor_148 = None
        mm_114 = torch.ops.aten.mm.default(view_547, permute_178);  view_547 = permute_178 = None
        view_554 = torch.ops.aten.view.default(mm_114, [2, 8192, 1024])
        view_555 = torch.ops.aten.view.default(view_548, [2, 8192, -1, 128]);  view_548 = None
        view_556 = torch.ops.aten.view.default(view_551, [2, 8192, -1, 128]);  view_551 = None
        view_557 = torch.ops.aten.view.default(view_554, [2, 8192, -1, 128]);  view_554 = None
        convert_element_type_541 = torch.ops.prims.convert_element_type.default(view_555, torch.float32);  view_555 = None
        view_558 = torch.ops.aten.view.default(convert_element_type_541, [2, 8192, 32, -1, 2]);  convert_element_type_541 = None
        view_as_complex_32 = torch.ops.aten.view_as_complex.default(view_558);  view_558 = None
        convert_element_type_542 = torch.ops.prims.convert_element_type.default(view_556, torch.float32);  view_556 = None
        view_559 = torch.ops.aten.view.default(convert_element_type_542, [2, 8192, 8, -1, 2]);  convert_element_type_542 = None
        view_as_complex_33 = torch.ops.aten.view_as_complex.default(view_559);  view_559 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_as_complex_32, view_16);  view_as_complex_32 = None
        view_as_real_32 = torch.ops.aten.view_as_real.default(mul_130);  mul_130 = None
        view_561 = torch.ops.aten.view.default(view_as_real_32, [2, 8192, 32, 128]);  view_as_real_32 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_as_complex_33, view_16);  view_as_complex_33 = None
        view_as_real_33 = torch.ops.aten.view_as_real.default(mul_131);  mul_131 = None
        view_562 = torch.ops.aten.view.default(view_as_real_33, [2, 8192, 8, 128]);  view_as_real_33 = None
        convert_element_type_543 = torch.ops.prims.convert_element_type.default(view_561, torch.bfloat16);  view_561 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(view_562, torch.bfloat16);  view_562 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(convert_element_type_544, 3);  convert_element_type_544 = None
        expand_32 = torch.ops.aten.expand.default(unsqueeze_32, [2, 8192, 8, 4, 128]);  unsqueeze_32 = None
        clone_32 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_563 = torch.ops.aten.view.default(clone_32, [2, 8192, 32, 128]);  clone_32 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(view_557, 3);  view_557 = None
        expand_33 = torch.ops.aten.expand.default(unsqueeze_33, [2, 8192, 8, 4, 128]);  unsqueeze_33 = None
        clone_33 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_564 = torch.ops.aten.view.default(clone_33, [2, 8192, 32, 128]);  clone_33 = None
        permute_179 = torch.ops.aten.permute.default(convert_element_type_543, [0, 2, 1, 3]);  convert_element_type_543 = None
        permute_180 = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
        permute_181 = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_179, permute_180, permute_181, None, True, 0.0, True);  permute_179 = permute_180 = permute_181 = None
        getitem_144 = _scaled_dot_product_cudnn_attention_16[0]
        getitem_145 = _scaled_dot_product_cudnn_attention_16[1]
        getitem_150 = _scaled_dot_product_cudnn_attention_16[6]
        getitem_151 = _scaled_dot_product_cudnn_attention_16[7];  _scaled_dot_product_cudnn_attention_16 = None
        permute_182 = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3])
        view_565 = torch.ops.aten.view.default(permute_182, [2, 8192, -1]);  permute_182 = None
        convert_element_type_545 = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16)
        all_gather_into_tensor_149 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_545, 64, '0');  convert_element_type_545 = None
        wait_tensor_149 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_149);  all_gather_into_tensor_149 = None
        permute_183 = torch.ops.aten.permute.default(wait_tensor_149, [1, 0]);  wait_tensor_149 = None
        view_567 = torch.ops.aten.view.default(view_565, [16384, 4096]);  view_565 = None
        mm_115 = torch.ops.aten.mm.default(view_567, permute_183);  view_567 = permute_183 = None
        view_568 = torch.ops.aten.view.default(mm_115, [2, 8192, 4096]);  mm_115 = None
        add_65 = torch.ops.aten.add.Tensor(add_63, view_568);  view_568 = None
        convert_element_type_548 = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16)
        all_gather_into_tensor_150 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_548, 64, '0');  convert_element_type_548 = None
        wait_tensor_150 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_150);  all_gather_into_tensor_150 = None
        convert_element_type_549 = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_549, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_34, [2], True);  pow_34 = None
        add_66 = torch.ops.aten.add.Scalar(mean_33, 1e-05);  mean_33 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_132 = torch.ops.aten.mul.Tensor(convert_element_type_549, rsqrt_33);  convert_element_type_549 = rsqrt_33 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, wait_tensor_150);  mul_132 = wait_tensor_150 = None
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        convert_element_type_551 = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16)
        all_gather_into_tensor_151 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_551, 64, '0');  convert_element_type_551 = None
        wait_tensor_151 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_151);  all_gather_into_tensor_151 = None
        permute_184 = torch.ops.aten.permute.default(wait_tensor_151, [1, 0]);  wait_tensor_151 = None
        view_571 = torch.ops.aten.view.default(convert_element_type_550, [16384, 4096]);  convert_element_type_550 = None
        mm_116 = torch.ops.aten.mm.default(view_571, permute_184);  permute_184 = None
        view_572 = torch.ops.aten.view.default(mm_116, [2, 8192, 14336])
        convert_element_type_554 = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(convert_element_type_554)
        mul_134 = torch.ops.aten.mul.Tensor(convert_element_type_554, sigmoid_16);  convert_element_type_554 = sigmoid_16 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16)
        all_gather_into_tensor_152 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_556, 64, '0');  convert_element_type_556 = None
        wait_tensor_152 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_152);  all_gather_into_tensor_152 = None
        permute_185 = torch.ops.aten.permute.default(wait_tensor_152, [1, 0]);  wait_tensor_152 = None
        mm_117 = torch.ops.aten.mm.default(view_571, permute_185);  view_571 = permute_185 = None
        view_575 = torch.ops.aten.view.default(mm_117, [2, 8192, 14336]);  mm_117 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_555, view_575);  convert_element_type_555 = view_575 = None
        convert_element_type_559 = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16)
        all_gather_into_tensor_153 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_559, 64, '0');  convert_element_type_559 = None
        wait_tensor_153 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_153);  all_gather_into_tensor_153 = None
        permute_186 = torch.ops.aten.permute.default(wait_tensor_153, [1, 0]);  wait_tensor_153 = None
        view_577 = torch.ops.aten.view.default(mul_135, [16384, 14336]);  mul_135 = None
        mm_118 = torch.ops.aten.mm.default(view_577, permute_186);  view_577 = permute_186 = None
        view_578 = torch.ops.aten.view.default(mm_118, [2, 8192, 4096]);  mm_118 = None
        add_67 = torch.ops.aten.add.Tensor(add_65, view_578);  add_65 = view_578 = None
        convert_element_type_562 = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16)
        all_gather_into_tensor_154 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_562, 64, '0');  convert_element_type_562 = None
        wait_tensor_154 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_154);  all_gather_into_tensor_154 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_35, [2], True);  pow_35 = None
        add_68 = torch.ops.aten.add.Scalar(mean_34, 1e-05);  mean_34 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_563, rsqrt_34);  convert_element_type_563 = rsqrt_34 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, wait_tensor_154);  mul_136 = wait_tensor_154 = None
        convert_element_type_564 = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        convert_element_type_565 = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16)
        all_gather_into_tensor_155 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_565, 64, '0');  convert_element_type_565 = None
        wait_tensor_155 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_155);  all_gather_into_tensor_155 = None
        permute_187 = torch.ops.aten.permute.default(wait_tensor_155, [1, 0]);  wait_tensor_155 = None
        view_581 = torch.ops.aten.view.default(convert_element_type_564, [16384, 4096]);  convert_element_type_564 = None
        mm_119 = torch.ops.aten.mm.default(view_581, permute_187);  permute_187 = None
        view_582 = torch.ops.aten.view.default(mm_119, [2, 8192, 4096])
        convert_element_type_568 = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16)
        all_gather_into_tensor_156 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_568, 64, '0');  convert_element_type_568 = None
        wait_tensor_156 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_156);  all_gather_into_tensor_156 = None
        permute_188 = torch.ops.aten.permute.default(wait_tensor_156, [1, 0]);  wait_tensor_156 = None
        mm_120 = torch.ops.aten.mm.default(view_581, permute_188);  permute_188 = None
        view_585 = torch.ops.aten.view.default(mm_120, [2, 8192, 1024]);  mm_120 = None
        convert_element_type_571 = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16)
        all_gather_into_tensor_157 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_571, 64, '0');  convert_element_type_571 = None
        wait_tensor_157 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_157);  all_gather_into_tensor_157 = None
        permute_189 = torch.ops.aten.permute.default(wait_tensor_157, [1, 0]);  wait_tensor_157 = None
        mm_121 = torch.ops.aten.mm.default(view_581, permute_189);  view_581 = permute_189 = None
        view_588 = torch.ops.aten.view.default(mm_121, [2, 8192, 1024])
        view_589 = torch.ops.aten.view.default(view_582, [2, 8192, -1, 128]);  view_582 = None
        view_590 = torch.ops.aten.view.default(view_585, [2, 8192, -1, 128]);  view_585 = None
        view_591 = torch.ops.aten.view.default(view_588, [2, 8192, -1, 128]);  view_588 = None
        convert_element_type_574 = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None
        view_592 = torch.ops.aten.view.default(convert_element_type_574, [2, 8192, 32, -1, 2]);  convert_element_type_574 = None
        view_as_complex_34 = torch.ops.aten.view_as_complex.default(view_592);  view_592 = None
        convert_element_type_575 = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        view_593 = torch.ops.aten.view.default(convert_element_type_575, [2, 8192, 8, -1, 2]);  convert_element_type_575 = None
        view_as_complex_35 = torch.ops.aten.view_as_complex.default(view_593);  view_593 = None
        mul_138 = torch.ops.aten.mul.Tensor(view_as_complex_34, view_16);  view_as_complex_34 = None
        view_as_real_34 = torch.ops.aten.view_as_real.default(mul_138);  mul_138 = None
        view_595 = torch.ops.aten.view.default(view_as_real_34, [2, 8192, 32, 128]);  view_as_real_34 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_as_complex_35, view_16);  view_as_complex_35 = None
        view_as_real_35 = torch.ops.aten.view_as_real.default(mul_139);  mul_139 = None
        view_596 = torch.ops.aten.view.default(view_as_real_35, [2, 8192, 8, 128]);  view_as_real_35 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(view_595, torch.bfloat16);  view_595 = None
        convert_element_type_577 = torch.ops.prims.convert_element_type.default(view_596, torch.bfloat16);  view_596 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(convert_element_type_577, 3);  convert_element_type_577 = None
        expand_34 = torch.ops.aten.expand.default(unsqueeze_34, [2, 8192, 8, 4, 128]);  unsqueeze_34 = None
        clone_34 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_597 = torch.ops.aten.view.default(clone_34, [2, 8192, 32, 128]);  clone_34 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(view_591, 3);  view_591 = None
        expand_35 = torch.ops.aten.expand.default(unsqueeze_35, [2, 8192, 8, 4, 128]);  unsqueeze_35 = None
        clone_35 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_598 = torch.ops.aten.view.default(clone_35, [2, 8192, 32, 128]);  clone_35 = None
        permute_190 = torch.ops.aten.permute.default(convert_element_type_576, [0, 2, 1, 3]);  convert_element_type_576 = None
        permute_191 = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
        permute_192 = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_190, permute_191, permute_192, None, True, 0.0, True);  permute_190 = permute_191 = permute_192 = None
        getitem_153 = _scaled_dot_product_cudnn_attention_17[0]
        getitem_154 = _scaled_dot_product_cudnn_attention_17[1]
        getitem_159 = _scaled_dot_product_cudnn_attention_17[6]
        getitem_160 = _scaled_dot_product_cudnn_attention_17[7];  _scaled_dot_product_cudnn_attention_17 = None
        permute_193 = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3])
        view_599 = torch.ops.aten.view.default(permute_193, [2, 8192, -1]);  permute_193 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16)
        all_gather_into_tensor_158 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_578, 64, '0');  convert_element_type_578 = None
        wait_tensor_158 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_158);  all_gather_into_tensor_158 = None
        permute_194 = torch.ops.aten.permute.default(wait_tensor_158, [1, 0]);  wait_tensor_158 = None
        view_601 = torch.ops.aten.view.default(view_599, [16384, 4096]);  view_599 = None
        mm_122 = torch.ops.aten.mm.default(view_601, permute_194);  view_601 = permute_194 = None
        view_602 = torch.ops.aten.view.default(mm_122, [2, 8192, 4096]);  mm_122 = None
        add_69 = torch.ops.aten.add.Tensor(add_67, view_602);  view_602 = None
        convert_element_type_581 = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16)
        all_gather_into_tensor_159 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_581, 64, '0');  convert_element_type_581 = None
        wait_tensor_159 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_159);  all_gather_into_tensor_159 = None
        convert_element_type_582 = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_582, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_36, [2], True);  pow_36 = None
        add_70 = torch.ops.aten.add.Scalar(mean_35, 1e-05);  mean_35 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_582, rsqrt_35);  convert_element_type_582 = rsqrt_35 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, wait_tensor_159);  mul_140 = wait_tensor_159 = None
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        convert_element_type_584 = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16)
        all_gather_into_tensor_160 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_584, 64, '0');  convert_element_type_584 = None
        wait_tensor_160 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_160);  all_gather_into_tensor_160 = None
        permute_195 = torch.ops.aten.permute.default(wait_tensor_160, [1, 0]);  wait_tensor_160 = None
        view_605 = torch.ops.aten.view.default(convert_element_type_583, [16384, 4096]);  convert_element_type_583 = None
        mm_123 = torch.ops.aten.mm.default(view_605, permute_195);  permute_195 = None
        view_606 = torch.ops.aten.view.default(mm_123, [2, 8192, 14336])
        convert_element_type_587 = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(convert_element_type_587)
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_587, sigmoid_17);  convert_element_type_587 = sigmoid_17 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        convert_element_type_589 = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16)
        all_gather_into_tensor_161 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_589, 64, '0');  convert_element_type_589 = None
        wait_tensor_161 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_161);  all_gather_into_tensor_161 = None
        permute_196 = torch.ops.aten.permute.default(wait_tensor_161, [1, 0]);  wait_tensor_161 = None
        mm_124 = torch.ops.aten.mm.default(view_605, permute_196);  view_605 = permute_196 = None
        view_609 = torch.ops.aten.view.default(mm_124, [2, 8192, 14336]);  mm_124 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_588, view_609);  convert_element_type_588 = view_609 = None
        convert_element_type_592 = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16)
        all_gather_into_tensor_162 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_592, 64, '0');  convert_element_type_592 = None
        wait_tensor_162 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_162);  all_gather_into_tensor_162 = None
        permute_197 = torch.ops.aten.permute.default(wait_tensor_162, [1, 0]);  wait_tensor_162 = None
        view_611 = torch.ops.aten.view.default(mul_143, [16384, 14336]);  mul_143 = None
        mm_125 = torch.ops.aten.mm.default(view_611, permute_197);  view_611 = permute_197 = None
        view_612 = torch.ops.aten.view.default(mm_125, [2, 8192, 4096]);  mm_125 = None
        add_71 = torch.ops.aten.add.Tensor(add_69, view_612);  add_69 = view_612 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16)
        all_gather_into_tensor_163 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_595, 64, '0');  convert_element_type_595 = None
        wait_tensor_163 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_163);  all_gather_into_tensor_163 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_596, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_37, [2], True);  pow_37 = None
        add_72 = torch.ops.aten.add.Scalar(mean_36, 1e-05);  mean_36 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_596, rsqrt_36);  convert_element_type_596 = rsqrt_36 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, wait_tensor_163);  mul_144 = wait_tensor_163 = None
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        convert_element_type_598 = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16)
        all_gather_into_tensor_164 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_598, 64, '0');  convert_element_type_598 = None
        wait_tensor_164 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_164);  all_gather_into_tensor_164 = None
        permute_198 = torch.ops.aten.permute.default(wait_tensor_164, [1, 0]);  wait_tensor_164 = None
        view_615 = torch.ops.aten.view.default(convert_element_type_597, [16384, 4096]);  convert_element_type_597 = None
        mm_126 = torch.ops.aten.mm.default(view_615, permute_198);  permute_198 = None
        view_616 = torch.ops.aten.view.default(mm_126, [2, 8192, 4096])
        convert_element_type_601 = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16)
        all_gather_into_tensor_165 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_601, 64, '0');  convert_element_type_601 = None
        wait_tensor_165 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_165);  all_gather_into_tensor_165 = None
        permute_199 = torch.ops.aten.permute.default(wait_tensor_165, [1, 0]);  wait_tensor_165 = None
        mm_127 = torch.ops.aten.mm.default(view_615, permute_199);  permute_199 = None
        view_619 = torch.ops.aten.view.default(mm_127, [2, 8192, 1024]);  mm_127 = None
        convert_element_type_604 = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16)
        all_gather_into_tensor_166 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_604, 64, '0');  convert_element_type_604 = None
        wait_tensor_166 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_166);  all_gather_into_tensor_166 = None
        permute_200 = torch.ops.aten.permute.default(wait_tensor_166, [1, 0]);  wait_tensor_166 = None
        mm_128 = torch.ops.aten.mm.default(view_615, permute_200);  view_615 = permute_200 = None
        view_622 = torch.ops.aten.view.default(mm_128, [2, 8192, 1024])
        view_623 = torch.ops.aten.view.default(view_616, [2, 8192, -1, 128]);  view_616 = None
        view_624 = torch.ops.aten.view.default(view_619, [2, 8192, -1, 128]);  view_619 = None
        view_625 = torch.ops.aten.view.default(view_622, [2, 8192, -1, 128]);  view_622 = None
        convert_element_type_607 = torch.ops.prims.convert_element_type.default(view_623, torch.float32);  view_623 = None
        view_626 = torch.ops.aten.view.default(convert_element_type_607, [2, 8192, 32, -1, 2]);  convert_element_type_607 = None
        view_as_complex_36 = torch.ops.aten.view_as_complex.default(view_626);  view_626 = None
        convert_element_type_608 = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        view_627 = torch.ops.aten.view.default(convert_element_type_608, [2, 8192, 8, -1, 2]);  convert_element_type_608 = None
        view_as_complex_37 = torch.ops.aten.view_as_complex.default(view_627);  view_627 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_as_complex_36, view_16);  view_as_complex_36 = None
        view_as_real_36 = torch.ops.aten.view_as_real.default(mul_146);  mul_146 = None
        view_629 = torch.ops.aten.view.default(view_as_real_36, [2, 8192, 32, 128]);  view_as_real_36 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_as_complex_37, view_16);  view_as_complex_37 = None
        view_as_real_37 = torch.ops.aten.view_as_real.default(mul_147);  mul_147 = None
        view_630 = torch.ops.aten.view.default(view_as_real_37, [2, 8192, 8, 128]);  view_as_real_37 = None
        convert_element_type_609 = torch.ops.prims.convert_element_type.default(view_629, torch.bfloat16);  view_629 = None
        convert_element_type_610 = torch.ops.prims.convert_element_type.default(view_630, torch.bfloat16);  view_630 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(convert_element_type_610, 3);  convert_element_type_610 = None
        expand_36 = torch.ops.aten.expand.default(unsqueeze_36, [2, 8192, 8, 4, 128]);  unsqueeze_36 = None
        clone_36 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_631 = torch.ops.aten.view.default(clone_36, [2, 8192, 32, 128]);  clone_36 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(view_625, 3);  view_625 = None
        expand_37 = torch.ops.aten.expand.default(unsqueeze_37, [2, 8192, 8, 4, 128]);  unsqueeze_37 = None
        clone_37 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_632 = torch.ops.aten.view.default(clone_37, [2, 8192, 32, 128]);  clone_37 = None
        permute_201 = torch.ops.aten.permute.default(convert_element_type_609, [0, 2, 1, 3]);  convert_element_type_609 = None
        permute_202 = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
        permute_203 = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_201, permute_202, permute_203, None, True, 0.0, True);  permute_201 = permute_202 = permute_203 = None
        getitem_162 = _scaled_dot_product_cudnn_attention_18[0]
        getitem_163 = _scaled_dot_product_cudnn_attention_18[1]
        getitem_168 = _scaled_dot_product_cudnn_attention_18[6]
        getitem_169 = _scaled_dot_product_cudnn_attention_18[7];  _scaled_dot_product_cudnn_attention_18 = None
        permute_204 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3])
        view_633 = torch.ops.aten.view.default(permute_204, [2, 8192, -1]);  permute_204 = None
        convert_element_type_611 = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16)
        all_gather_into_tensor_167 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_611, 64, '0');  convert_element_type_611 = None
        wait_tensor_167 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_167);  all_gather_into_tensor_167 = None
        permute_205 = torch.ops.aten.permute.default(wait_tensor_167, [1, 0]);  wait_tensor_167 = None
        view_635 = torch.ops.aten.view.default(view_633, [16384, 4096]);  view_633 = None
        mm_129 = torch.ops.aten.mm.default(view_635, permute_205);  view_635 = permute_205 = None
        view_636 = torch.ops.aten.view.default(mm_129, [2, 8192, 4096]);  mm_129 = None
        add_73 = torch.ops.aten.add.Tensor(add_71, view_636);  view_636 = None
        convert_element_type_614 = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16)
        all_gather_into_tensor_168 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_614, 64, '0');  convert_element_type_614 = None
        wait_tensor_168 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_168);  all_gather_into_tensor_168 = None
        convert_element_type_615 = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_615, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_38, [2], True);  pow_38 = None
        add_74 = torch.ops.aten.add.Scalar(mean_37, 1e-05);  mean_37 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_148 = torch.ops.aten.mul.Tensor(convert_element_type_615, rsqrt_37);  convert_element_type_615 = rsqrt_37 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, wait_tensor_168);  mul_148 = wait_tensor_168 = None
        convert_element_type_616 = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16)
        all_gather_into_tensor_169 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_617, 64, '0');  convert_element_type_617 = None
        wait_tensor_169 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_169);  all_gather_into_tensor_169 = None
        permute_206 = torch.ops.aten.permute.default(wait_tensor_169, [1, 0]);  wait_tensor_169 = None
        view_639 = torch.ops.aten.view.default(convert_element_type_616, [16384, 4096]);  convert_element_type_616 = None
        mm_130 = torch.ops.aten.mm.default(view_639, permute_206);  permute_206 = None
        view_640 = torch.ops.aten.view.default(mm_130, [2, 8192, 14336])
        convert_element_type_620 = torch.ops.prims.convert_element_type.default(view_640, torch.float32);  view_640 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(convert_element_type_620)
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_620, sigmoid_18);  convert_element_type_620 = sigmoid_18 = None
        convert_element_type_621 = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16)
        all_gather_into_tensor_170 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_622, 64, '0');  convert_element_type_622 = None
        wait_tensor_170 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_170);  all_gather_into_tensor_170 = None
        permute_207 = torch.ops.aten.permute.default(wait_tensor_170, [1, 0]);  wait_tensor_170 = None
        mm_131 = torch.ops.aten.mm.default(view_639, permute_207);  view_639 = permute_207 = None
        view_643 = torch.ops.aten.view.default(mm_131, [2, 8192, 14336]);  mm_131 = None
        mul_151 = torch.ops.aten.mul.Tensor(convert_element_type_621, view_643);  convert_element_type_621 = view_643 = None
        convert_element_type_625 = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16)
        all_gather_into_tensor_171 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_625, 64, '0');  convert_element_type_625 = None
        wait_tensor_171 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_171);  all_gather_into_tensor_171 = None
        permute_208 = torch.ops.aten.permute.default(wait_tensor_171, [1, 0]);  wait_tensor_171 = None
        view_645 = torch.ops.aten.view.default(mul_151, [16384, 14336]);  mul_151 = None
        mm_132 = torch.ops.aten.mm.default(view_645, permute_208);  view_645 = permute_208 = None
        view_646 = torch.ops.aten.view.default(mm_132, [2, 8192, 4096]);  mm_132 = None
        add_75 = torch.ops.aten.add.Tensor(add_73, view_646);  add_73 = view_646 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16)
        all_gather_into_tensor_172 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_628, 64, '0');  convert_element_type_628 = None
        wait_tensor_172 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_172);  all_gather_into_tensor_172 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_629, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_39, [2], True);  pow_39 = None
        add_76 = torch.ops.aten.add.Scalar(mean_38, 1e-05);  mean_38 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_629, rsqrt_38);  convert_element_type_629 = rsqrt_38 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, wait_tensor_172);  mul_152 = wait_tensor_172 = None
        convert_element_type_630 = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16)
        all_gather_into_tensor_173 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_631, 64, '0');  convert_element_type_631 = None
        wait_tensor_173 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_173);  all_gather_into_tensor_173 = None
        permute_209 = torch.ops.aten.permute.default(wait_tensor_173, [1, 0]);  wait_tensor_173 = None
        view_649 = torch.ops.aten.view.default(convert_element_type_630, [16384, 4096]);  convert_element_type_630 = None
        mm_133 = torch.ops.aten.mm.default(view_649, permute_209);  permute_209 = None
        view_650 = torch.ops.aten.view.default(mm_133, [2, 8192, 4096])
        convert_element_type_634 = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16)
        all_gather_into_tensor_174 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_634, 64, '0');  convert_element_type_634 = None
        wait_tensor_174 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_174);  all_gather_into_tensor_174 = None
        permute_210 = torch.ops.aten.permute.default(wait_tensor_174, [1, 0]);  wait_tensor_174 = None
        mm_134 = torch.ops.aten.mm.default(view_649, permute_210);  permute_210 = None
        view_653 = torch.ops.aten.view.default(mm_134, [2, 8192, 1024]);  mm_134 = None
        convert_element_type_637 = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16)
        all_gather_into_tensor_175 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_637, 64, '0');  convert_element_type_637 = None
        wait_tensor_175 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_175);  all_gather_into_tensor_175 = None
        permute_211 = torch.ops.aten.permute.default(wait_tensor_175, [1, 0]);  wait_tensor_175 = None
        mm_135 = torch.ops.aten.mm.default(view_649, permute_211);  view_649 = permute_211 = None
        view_656 = torch.ops.aten.view.default(mm_135, [2, 8192, 1024])
        view_657 = torch.ops.aten.view.default(view_650, [2, 8192, -1, 128]);  view_650 = None
        view_658 = torch.ops.aten.view.default(view_653, [2, 8192, -1, 128]);  view_653 = None
        view_659 = torch.ops.aten.view.default(view_656, [2, 8192, -1, 128]);  view_656 = None
        convert_element_type_640 = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        view_660 = torch.ops.aten.view.default(convert_element_type_640, [2, 8192, 32, -1, 2]);  convert_element_type_640 = None
        view_as_complex_38 = torch.ops.aten.view_as_complex.default(view_660);  view_660 = None
        convert_element_type_641 = torch.ops.prims.convert_element_type.default(view_658, torch.float32);  view_658 = None
        view_661 = torch.ops.aten.view.default(convert_element_type_641, [2, 8192, 8, -1, 2]);  convert_element_type_641 = None
        view_as_complex_39 = torch.ops.aten.view_as_complex.default(view_661);  view_661 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_as_complex_38, view_16);  view_as_complex_38 = None
        view_as_real_38 = torch.ops.aten.view_as_real.default(mul_154);  mul_154 = None
        view_663 = torch.ops.aten.view.default(view_as_real_38, [2, 8192, 32, 128]);  view_as_real_38 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_as_complex_39, view_16);  view_as_complex_39 = None
        view_as_real_39 = torch.ops.aten.view_as_real.default(mul_155);  mul_155 = None
        view_664 = torch.ops.aten.view.default(view_as_real_39, [2, 8192, 8, 128]);  view_as_real_39 = None
        convert_element_type_642 = torch.ops.prims.convert_element_type.default(view_663, torch.bfloat16);  view_663 = None
        convert_element_type_643 = torch.ops.prims.convert_element_type.default(view_664, torch.bfloat16);  view_664 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(convert_element_type_643, 3);  convert_element_type_643 = None
        expand_38 = torch.ops.aten.expand.default(unsqueeze_38, [2, 8192, 8, 4, 128]);  unsqueeze_38 = None
        clone_38 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_665 = torch.ops.aten.view.default(clone_38, [2, 8192, 32, 128]);  clone_38 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(view_659, 3);  view_659 = None
        expand_39 = torch.ops.aten.expand.default(unsqueeze_39, [2, 8192, 8, 4, 128]);  unsqueeze_39 = None
        clone_39 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_666 = torch.ops.aten.view.default(clone_39, [2, 8192, 32, 128]);  clone_39 = None
        permute_212 = torch.ops.aten.permute.default(convert_element_type_642, [0, 2, 1, 3]);  convert_element_type_642 = None
        permute_213 = torch.ops.aten.permute.default(view_665, [0, 2, 1, 3]);  view_665 = None
        permute_214 = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_212, permute_213, permute_214, None, True, 0.0, True);  permute_212 = permute_213 = permute_214 = None
        getitem_171 = _scaled_dot_product_cudnn_attention_19[0]
        getitem_172 = _scaled_dot_product_cudnn_attention_19[1]
        getitem_177 = _scaled_dot_product_cudnn_attention_19[6]
        getitem_178 = _scaled_dot_product_cudnn_attention_19[7];  _scaled_dot_product_cudnn_attention_19 = None
        permute_215 = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3])
        view_667 = torch.ops.aten.view.default(permute_215, [2, 8192, -1]);  permute_215 = None
        convert_element_type_644 = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16)
        all_gather_into_tensor_176 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_644, 64, '0');  convert_element_type_644 = None
        wait_tensor_176 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_176);  all_gather_into_tensor_176 = None
        permute_216 = torch.ops.aten.permute.default(wait_tensor_176, [1, 0]);  wait_tensor_176 = None
        view_669 = torch.ops.aten.view.default(view_667, [16384, 4096]);  view_667 = None
        mm_136 = torch.ops.aten.mm.default(view_669, permute_216);  view_669 = permute_216 = None
        view_670 = torch.ops.aten.view.default(mm_136, [2, 8192, 4096]);  mm_136 = None
        add_77 = torch.ops.aten.add.Tensor(add_75, view_670);  view_670 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16)
        all_gather_into_tensor_177 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_647, 64, '0');  convert_element_type_647 = None
        wait_tensor_177 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_177);  all_gather_into_tensor_177 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_648, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_40, [2], True);  pow_40 = None
        add_78 = torch.ops.aten.add.Scalar(mean_39, 1e-05);  mean_39 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_156 = torch.ops.aten.mul.Tensor(convert_element_type_648, rsqrt_39);  convert_element_type_648 = rsqrt_39 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, wait_tensor_177);  mul_156 = wait_tensor_177 = None
        convert_element_type_649 = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16)
        all_gather_into_tensor_178 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_650, 64, '0');  convert_element_type_650 = None
        wait_tensor_178 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_178);  all_gather_into_tensor_178 = None
        permute_217 = torch.ops.aten.permute.default(wait_tensor_178, [1, 0]);  wait_tensor_178 = None
        view_673 = torch.ops.aten.view.default(convert_element_type_649, [16384, 4096]);  convert_element_type_649 = None
        mm_137 = torch.ops.aten.mm.default(view_673, permute_217);  permute_217 = None
        view_674 = torch.ops.aten.view.default(mm_137, [2, 8192, 14336])
        convert_element_type_653 = torch.ops.prims.convert_element_type.default(view_674, torch.float32);  view_674 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(convert_element_type_653)
        mul_158 = torch.ops.aten.mul.Tensor(convert_element_type_653, sigmoid_19);  convert_element_type_653 = sigmoid_19 = None
        convert_element_type_654 = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        convert_element_type_655 = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16)
        all_gather_into_tensor_179 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_655, 64, '0');  convert_element_type_655 = None
        wait_tensor_179 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_179);  all_gather_into_tensor_179 = None
        permute_218 = torch.ops.aten.permute.default(wait_tensor_179, [1, 0]);  wait_tensor_179 = None
        mm_138 = torch.ops.aten.mm.default(view_673, permute_218);  view_673 = permute_218 = None
        view_677 = torch.ops.aten.view.default(mm_138, [2, 8192, 14336]);  mm_138 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_654, view_677);  convert_element_type_654 = view_677 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16)
        all_gather_into_tensor_180 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_658, 64, '0');  convert_element_type_658 = None
        wait_tensor_180 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_180);  all_gather_into_tensor_180 = None
        permute_219 = torch.ops.aten.permute.default(wait_tensor_180, [1, 0]);  wait_tensor_180 = None
        view_679 = torch.ops.aten.view.default(mul_159, [16384, 14336]);  mul_159 = None
        mm_139 = torch.ops.aten.mm.default(view_679, permute_219);  view_679 = permute_219 = None
        view_680 = torch.ops.aten.view.default(mm_139, [2, 8192, 4096]);  mm_139 = None
        add_79 = torch.ops.aten.add.Tensor(add_77, view_680);  add_77 = view_680 = None
        convert_element_type_661 = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16)
        all_gather_into_tensor_181 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_661, 64, '0');  convert_element_type_661 = None
        wait_tensor_181 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_181);  all_gather_into_tensor_181 = None
        convert_element_type_662 = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_662, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_41, [2], True);  pow_41 = None
        add_80 = torch.ops.aten.add.Scalar(mean_40, 1e-05);  mean_40 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_662, rsqrt_40);  convert_element_type_662 = rsqrt_40 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, wait_tensor_181);  mul_160 = wait_tensor_181 = None
        convert_element_type_663 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        convert_element_type_664 = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16)
        all_gather_into_tensor_182 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_664, 64, '0');  convert_element_type_664 = None
        wait_tensor_182 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_182);  all_gather_into_tensor_182 = None
        permute_220 = torch.ops.aten.permute.default(wait_tensor_182, [1, 0]);  wait_tensor_182 = None
        view_683 = torch.ops.aten.view.default(convert_element_type_663, [16384, 4096]);  convert_element_type_663 = None
        mm_140 = torch.ops.aten.mm.default(view_683, permute_220);  permute_220 = None
        view_684 = torch.ops.aten.view.default(mm_140, [2, 8192, 4096])
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16)
        all_gather_into_tensor_183 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_667, 64, '0');  convert_element_type_667 = None
        wait_tensor_183 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_183);  all_gather_into_tensor_183 = None
        permute_221 = torch.ops.aten.permute.default(wait_tensor_183, [1, 0]);  wait_tensor_183 = None
        mm_141 = torch.ops.aten.mm.default(view_683, permute_221);  permute_221 = None
        view_687 = torch.ops.aten.view.default(mm_141, [2, 8192, 1024]);  mm_141 = None
        convert_element_type_670 = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16)
        all_gather_into_tensor_184 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_670, 64, '0');  convert_element_type_670 = None
        wait_tensor_184 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_184);  all_gather_into_tensor_184 = None
        permute_222 = torch.ops.aten.permute.default(wait_tensor_184, [1, 0]);  wait_tensor_184 = None
        mm_142 = torch.ops.aten.mm.default(view_683, permute_222);  view_683 = permute_222 = None
        view_690 = torch.ops.aten.view.default(mm_142, [2, 8192, 1024])
        view_691 = torch.ops.aten.view.default(view_684, [2, 8192, -1, 128]);  view_684 = None
        view_692 = torch.ops.aten.view.default(view_687, [2, 8192, -1, 128]);  view_687 = None
        view_693 = torch.ops.aten.view.default(view_690, [2, 8192, -1, 128]);  view_690 = None
        convert_element_type_673 = torch.ops.prims.convert_element_type.default(view_691, torch.float32);  view_691 = None
        view_694 = torch.ops.aten.view.default(convert_element_type_673, [2, 8192, 32, -1, 2]);  convert_element_type_673 = None
        view_as_complex_40 = torch.ops.aten.view_as_complex.default(view_694);  view_694 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        view_695 = torch.ops.aten.view.default(convert_element_type_674, [2, 8192, 8, -1, 2]);  convert_element_type_674 = None
        view_as_complex_41 = torch.ops.aten.view_as_complex.default(view_695);  view_695 = None
        mul_162 = torch.ops.aten.mul.Tensor(view_as_complex_40, view_16);  view_as_complex_40 = None
        view_as_real_40 = torch.ops.aten.view_as_real.default(mul_162);  mul_162 = None
        view_697 = torch.ops.aten.view.default(view_as_real_40, [2, 8192, 32, 128]);  view_as_real_40 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_as_complex_41, view_16);  view_as_complex_41 = None
        view_as_real_41 = torch.ops.aten.view_as_real.default(mul_163);  mul_163 = None
        view_698 = torch.ops.aten.view.default(view_as_real_41, [2, 8192, 8, 128]);  view_as_real_41 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(view_697, torch.bfloat16);  view_697 = None
        convert_element_type_676 = torch.ops.prims.convert_element_type.default(view_698, torch.bfloat16);  view_698 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(convert_element_type_676, 3);  convert_element_type_676 = None
        expand_40 = torch.ops.aten.expand.default(unsqueeze_40, [2, 8192, 8, 4, 128]);  unsqueeze_40 = None
        clone_40 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_699 = torch.ops.aten.view.default(clone_40, [2, 8192, 32, 128]);  clone_40 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(view_693, 3);  view_693 = None
        expand_41 = torch.ops.aten.expand.default(unsqueeze_41, [2, 8192, 8, 4, 128]);  unsqueeze_41 = None
        clone_41 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_700 = torch.ops.aten.view.default(clone_41, [2, 8192, 32, 128]);  clone_41 = None
        permute_223 = torch.ops.aten.permute.default(convert_element_type_675, [0, 2, 1, 3]);  convert_element_type_675 = None
        permute_224 = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
        permute_225 = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_223, permute_224, permute_225, None, True, 0.0, True);  permute_223 = permute_224 = permute_225 = None
        getitem_180 = _scaled_dot_product_cudnn_attention_20[0]
        getitem_181 = _scaled_dot_product_cudnn_attention_20[1]
        getitem_186 = _scaled_dot_product_cudnn_attention_20[6]
        getitem_187 = _scaled_dot_product_cudnn_attention_20[7];  _scaled_dot_product_cudnn_attention_20 = None
        permute_226 = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3])
        view_701 = torch.ops.aten.view.default(permute_226, [2, 8192, -1]);  permute_226 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16)
        all_gather_into_tensor_185 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_677, 64, '0');  convert_element_type_677 = None
        wait_tensor_185 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_185);  all_gather_into_tensor_185 = None
        permute_227 = torch.ops.aten.permute.default(wait_tensor_185, [1, 0]);  wait_tensor_185 = None
        view_703 = torch.ops.aten.view.default(view_701, [16384, 4096]);  view_701 = None
        mm_143 = torch.ops.aten.mm.default(view_703, permute_227);  view_703 = permute_227 = None
        view_704 = torch.ops.aten.view.default(mm_143, [2, 8192, 4096]);  mm_143 = None
        add_81 = torch.ops.aten.add.Tensor(add_79, view_704);  view_704 = None
        convert_element_type_680 = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16)
        all_gather_into_tensor_186 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_680, 64, '0');  convert_element_type_680 = None
        wait_tensor_186 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_186);  all_gather_into_tensor_186 = None
        convert_element_type_681 = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_681, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_42, [2], True);  pow_42 = None
        add_82 = torch.ops.aten.add.Scalar(mean_41, 1e-05);  mean_41 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_164 = torch.ops.aten.mul.Tensor(convert_element_type_681, rsqrt_41);  convert_element_type_681 = rsqrt_41 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, wait_tensor_186);  mul_164 = wait_tensor_186 = None
        convert_element_type_682 = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16)
        all_gather_into_tensor_187 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_683, 64, '0');  convert_element_type_683 = None
        wait_tensor_187 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_187);  all_gather_into_tensor_187 = None
        permute_228 = torch.ops.aten.permute.default(wait_tensor_187, [1, 0]);  wait_tensor_187 = None
        view_707 = torch.ops.aten.view.default(convert_element_type_682, [16384, 4096]);  convert_element_type_682 = None
        mm_144 = torch.ops.aten.mm.default(view_707, permute_228);  permute_228 = None
        view_708 = torch.ops.aten.view.default(mm_144, [2, 8192, 14336])
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_708, torch.float32);  view_708 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(convert_element_type_686)
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_686, sigmoid_20);  convert_element_type_686 = sigmoid_20 = None
        convert_element_type_687 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        convert_element_type_688 = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16)
        all_gather_into_tensor_188 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_688, 64, '0');  convert_element_type_688 = None
        wait_tensor_188 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_188);  all_gather_into_tensor_188 = None
        permute_229 = torch.ops.aten.permute.default(wait_tensor_188, [1, 0]);  wait_tensor_188 = None
        mm_145 = torch.ops.aten.mm.default(view_707, permute_229);  view_707 = permute_229 = None
        view_711 = torch.ops.aten.view.default(mm_145, [2, 8192, 14336]);  mm_145 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_687, view_711);  convert_element_type_687 = view_711 = None
        convert_element_type_691 = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16)
        all_gather_into_tensor_189 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_691, 64, '0');  convert_element_type_691 = None
        wait_tensor_189 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_189);  all_gather_into_tensor_189 = None
        permute_230 = torch.ops.aten.permute.default(wait_tensor_189, [1, 0]);  wait_tensor_189 = None
        view_713 = torch.ops.aten.view.default(mul_167, [16384, 14336]);  mul_167 = None
        mm_146 = torch.ops.aten.mm.default(view_713, permute_230);  view_713 = permute_230 = None
        view_714 = torch.ops.aten.view.default(mm_146, [2, 8192, 4096]);  mm_146 = None
        add_83 = torch.ops.aten.add.Tensor(add_81, view_714);  add_81 = view_714 = None
        convert_element_type_694 = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16)
        all_gather_into_tensor_190 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_694, 64, '0');  convert_element_type_694 = None
        wait_tensor_190 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_190);  all_gather_into_tensor_190 = None
        convert_element_type_695 = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 2)
        mean_42 = torch.ops.aten.mean.dim(pow_43, [2], True);  pow_43 = None
        add_84 = torch.ops.aten.add.Scalar(mean_42, 1e-05);  mean_42 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_168 = torch.ops.aten.mul.Tensor(convert_element_type_695, rsqrt_42);  convert_element_type_695 = rsqrt_42 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, wait_tensor_190);  mul_168 = wait_tensor_190 = None
        convert_element_type_696 = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        convert_element_type_697 = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16)
        all_gather_into_tensor_191 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_697, 64, '0');  convert_element_type_697 = None
        wait_tensor_191 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_191);  all_gather_into_tensor_191 = None
        permute_231 = torch.ops.aten.permute.default(wait_tensor_191, [1, 0]);  wait_tensor_191 = None
        view_717 = torch.ops.aten.view.default(convert_element_type_696, [16384, 4096]);  convert_element_type_696 = None
        mm_147 = torch.ops.aten.mm.default(view_717, permute_231);  permute_231 = None
        view_718 = torch.ops.aten.view.default(mm_147, [2, 8192, 4096])
        convert_element_type_700 = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16)
        all_gather_into_tensor_192 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_700, 64, '0');  convert_element_type_700 = None
        wait_tensor_192 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_192);  all_gather_into_tensor_192 = None
        permute_232 = torch.ops.aten.permute.default(wait_tensor_192, [1, 0]);  wait_tensor_192 = None
        mm_148 = torch.ops.aten.mm.default(view_717, permute_232);  permute_232 = None
        view_721 = torch.ops.aten.view.default(mm_148, [2, 8192, 1024]);  mm_148 = None
        convert_element_type_703 = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16)
        all_gather_into_tensor_193 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_703, 64, '0');  convert_element_type_703 = None
        wait_tensor_193 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_193);  all_gather_into_tensor_193 = None
        permute_233 = torch.ops.aten.permute.default(wait_tensor_193, [1, 0]);  wait_tensor_193 = None
        mm_149 = torch.ops.aten.mm.default(view_717, permute_233);  view_717 = permute_233 = None
        view_724 = torch.ops.aten.view.default(mm_149, [2, 8192, 1024])
        view_725 = torch.ops.aten.view.default(view_718, [2, 8192, -1, 128]);  view_718 = None
        view_726 = torch.ops.aten.view.default(view_721, [2, 8192, -1, 128]);  view_721 = None
        view_727 = torch.ops.aten.view.default(view_724, [2, 8192, -1, 128]);  view_724 = None
        convert_element_type_706 = torch.ops.prims.convert_element_type.default(view_725, torch.float32);  view_725 = None
        view_728 = torch.ops.aten.view.default(convert_element_type_706, [2, 8192, 32, -1, 2]);  convert_element_type_706 = None
        view_as_complex_42 = torch.ops.aten.view_as_complex.default(view_728);  view_728 = None
        convert_element_type_707 = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None
        view_729 = torch.ops.aten.view.default(convert_element_type_707, [2, 8192, 8, -1, 2]);  convert_element_type_707 = None
        view_as_complex_43 = torch.ops.aten.view_as_complex.default(view_729);  view_729 = None
        mul_170 = torch.ops.aten.mul.Tensor(view_as_complex_42, view_16);  view_as_complex_42 = None
        view_as_real_42 = torch.ops.aten.view_as_real.default(mul_170);  mul_170 = None
        view_731 = torch.ops.aten.view.default(view_as_real_42, [2, 8192, 32, 128]);  view_as_real_42 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_as_complex_43, view_16);  view_as_complex_43 = None
        view_as_real_43 = torch.ops.aten.view_as_real.default(mul_171);  mul_171 = None
        view_732 = torch.ops.aten.view.default(view_as_real_43, [2, 8192, 8, 128]);  view_as_real_43 = None
        convert_element_type_708 = torch.ops.prims.convert_element_type.default(view_731, torch.bfloat16);  view_731 = None
        convert_element_type_709 = torch.ops.prims.convert_element_type.default(view_732, torch.bfloat16);  view_732 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(convert_element_type_709, 3);  convert_element_type_709 = None
        expand_42 = torch.ops.aten.expand.default(unsqueeze_42, [2, 8192, 8, 4, 128]);  unsqueeze_42 = None
        clone_42 = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_733 = torch.ops.aten.view.default(clone_42, [2, 8192, 32, 128]);  clone_42 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(view_727, 3);  view_727 = None
        expand_43 = torch.ops.aten.expand.default(unsqueeze_43, [2, 8192, 8, 4, 128]);  unsqueeze_43 = None
        clone_43 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_734 = torch.ops.aten.view.default(clone_43, [2, 8192, 32, 128]);  clone_43 = None
        permute_234 = torch.ops.aten.permute.default(convert_element_type_708, [0, 2, 1, 3]);  convert_element_type_708 = None
        permute_235 = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
        permute_236 = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_234, permute_235, permute_236, None, True, 0.0, True);  permute_234 = permute_235 = permute_236 = None
        getitem_189 = _scaled_dot_product_cudnn_attention_21[0]
        getitem_190 = _scaled_dot_product_cudnn_attention_21[1]
        getitem_195 = _scaled_dot_product_cudnn_attention_21[6]
        getitem_196 = _scaled_dot_product_cudnn_attention_21[7];  _scaled_dot_product_cudnn_attention_21 = None
        permute_237 = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3])
        view_735 = torch.ops.aten.view.default(permute_237, [2, 8192, -1]);  permute_237 = None
        convert_element_type_710 = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16)
        all_gather_into_tensor_194 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_710, 64, '0');  convert_element_type_710 = None
        wait_tensor_194 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_194);  all_gather_into_tensor_194 = None
        permute_238 = torch.ops.aten.permute.default(wait_tensor_194, [1, 0]);  wait_tensor_194 = None
        view_737 = torch.ops.aten.view.default(view_735, [16384, 4096]);  view_735 = None
        mm_150 = torch.ops.aten.mm.default(view_737, permute_238);  view_737 = permute_238 = None
        view_738 = torch.ops.aten.view.default(mm_150, [2, 8192, 4096]);  mm_150 = None
        add_85 = torch.ops.aten.add.Tensor(add_83, view_738);  view_738 = None
        convert_element_type_713 = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16)
        all_gather_into_tensor_195 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_713, 64, '0');  convert_element_type_713 = None
        wait_tensor_195 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_195);  all_gather_into_tensor_195 = None
        convert_element_type_714 = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_714, 2)
        mean_43 = torch.ops.aten.mean.dim(pow_44, [2], True);  pow_44 = None
        add_86 = torch.ops.aten.add.Scalar(mean_43, 1e-05);  mean_43 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_172 = torch.ops.aten.mul.Tensor(convert_element_type_714, rsqrt_43);  convert_element_type_714 = rsqrt_43 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, wait_tensor_195);  mul_172 = wait_tensor_195 = None
        convert_element_type_715 = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        convert_element_type_716 = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16)
        all_gather_into_tensor_196 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_716, 64, '0');  convert_element_type_716 = None
        wait_tensor_196 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_196);  all_gather_into_tensor_196 = None
        permute_239 = torch.ops.aten.permute.default(wait_tensor_196, [1, 0]);  wait_tensor_196 = None
        view_741 = torch.ops.aten.view.default(convert_element_type_715, [16384, 4096]);  convert_element_type_715 = None
        mm_151 = torch.ops.aten.mm.default(view_741, permute_239);  permute_239 = None
        view_742 = torch.ops.aten.view.default(mm_151, [2, 8192, 14336])
        convert_element_type_719 = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(convert_element_type_719)
        mul_174 = torch.ops.aten.mul.Tensor(convert_element_type_719, sigmoid_21);  convert_element_type_719 = sigmoid_21 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        convert_element_type_721 = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16)
        all_gather_into_tensor_197 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_721, 64, '0');  convert_element_type_721 = None
        wait_tensor_197 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_197);  all_gather_into_tensor_197 = None
        permute_240 = torch.ops.aten.permute.default(wait_tensor_197, [1, 0]);  wait_tensor_197 = None
        mm_152 = torch.ops.aten.mm.default(view_741, permute_240);  view_741 = permute_240 = None
        view_745 = torch.ops.aten.view.default(mm_152, [2, 8192, 14336]);  mm_152 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_720, view_745);  convert_element_type_720 = view_745 = None
        convert_element_type_724 = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16)
        all_gather_into_tensor_198 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_724, 64, '0');  convert_element_type_724 = None
        wait_tensor_198 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_198);  all_gather_into_tensor_198 = None
        permute_241 = torch.ops.aten.permute.default(wait_tensor_198, [1, 0]);  wait_tensor_198 = None
        view_747 = torch.ops.aten.view.default(mul_175, [16384, 14336]);  mul_175 = None
        mm_153 = torch.ops.aten.mm.default(view_747, permute_241);  view_747 = permute_241 = None
        view_748 = torch.ops.aten.view.default(mm_153, [2, 8192, 4096]);  mm_153 = None
        add_87 = torch.ops.aten.add.Tensor(add_85, view_748);  add_85 = view_748 = None
        convert_element_type_727 = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16)
        all_gather_into_tensor_199 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_727, 64, '0');  convert_element_type_727 = None
        wait_tensor_199 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_199);  all_gather_into_tensor_199 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_728, 2)
        mean_44 = torch.ops.aten.mean.dim(pow_45, [2], True);  pow_45 = None
        add_88 = torch.ops.aten.add.Scalar(mean_44, 1e-05);  mean_44 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_176 = torch.ops.aten.mul.Tensor(convert_element_type_728, rsqrt_44);  convert_element_type_728 = rsqrt_44 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, wait_tensor_199);  mul_176 = wait_tensor_199 = None
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        convert_element_type_730 = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16)
        all_gather_into_tensor_200 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_730, 64, '0');  convert_element_type_730 = None
        wait_tensor_200 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_200);  all_gather_into_tensor_200 = None
        permute_242 = torch.ops.aten.permute.default(wait_tensor_200, [1, 0]);  wait_tensor_200 = None
        view_751 = torch.ops.aten.view.default(convert_element_type_729, [16384, 4096]);  convert_element_type_729 = None
        mm_154 = torch.ops.aten.mm.default(view_751, permute_242);  permute_242 = None
        view_752 = torch.ops.aten.view.default(mm_154, [2, 8192, 4096])
        convert_element_type_733 = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16)
        all_gather_into_tensor_201 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_733, 64, '0');  convert_element_type_733 = None
        wait_tensor_201 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_201);  all_gather_into_tensor_201 = None
        permute_243 = torch.ops.aten.permute.default(wait_tensor_201, [1, 0]);  wait_tensor_201 = None
        mm_155 = torch.ops.aten.mm.default(view_751, permute_243);  permute_243 = None
        view_755 = torch.ops.aten.view.default(mm_155, [2, 8192, 1024]);  mm_155 = None
        convert_element_type_736 = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16)
        all_gather_into_tensor_202 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_736, 64, '0');  convert_element_type_736 = None
        wait_tensor_202 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_202);  all_gather_into_tensor_202 = None
        permute_244 = torch.ops.aten.permute.default(wait_tensor_202, [1, 0]);  wait_tensor_202 = None
        mm_156 = torch.ops.aten.mm.default(view_751, permute_244);  view_751 = permute_244 = None
        view_758 = torch.ops.aten.view.default(mm_156, [2, 8192, 1024])
        view_759 = torch.ops.aten.view.default(view_752, [2, 8192, -1, 128]);  view_752 = None
        view_760 = torch.ops.aten.view.default(view_755, [2, 8192, -1, 128]);  view_755 = None
        view_761 = torch.ops.aten.view.default(view_758, [2, 8192, -1, 128]);  view_758 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        view_762 = torch.ops.aten.view.default(convert_element_type_739, [2, 8192, 32, -1, 2]);  convert_element_type_739 = None
        view_as_complex_44 = torch.ops.aten.view_as_complex.default(view_762);  view_762 = None
        convert_element_type_740 = torch.ops.prims.convert_element_type.default(view_760, torch.float32);  view_760 = None
        view_763 = torch.ops.aten.view.default(convert_element_type_740, [2, 8192, 8, -1, 2]);  convert_element_type_740 = None
        view_as_complex_45 = torch.ops.aten.view_as_complex.default(view_763);  view_763 = None
        mul_178 = torch.ops.aten.mul.Tensor(view_as_complex_44, view_16);  view_as_complex_44 = None
        view_as_real_44 = torch.ops.aten.view_as_real.default(mul_178);  mul_178 = None
        view_765 = torch.ops.aten.view.default(view_as_real_44, [2, 8192, 32, 128]);  view_as_real_44 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_as_complex_45, view_16);  view_as_complex_45 = None
        view_as_real_45 = torch.ops.aten.view_as_real.default(mul_179);  mul_179 = None
        view_766 = torch.ops.aten.view.default(view_as_real_45, [2, 8192, 8, 128]);  view_as_real_45 = None
        convert_element_type_741 = torch.ops.prims.convert_element_type.default(view_765, torch.bfloat16);  view_765 = None
        convert_element_type_742 = torch.ops.prims.convert_element_type.default(view_766, torch.bfloat16);  view_766 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(convert_element_type_742, 3);  convert_element_type_742 = None
        expand_44 = torch.ops.aten.expand.default(unsqueeze_44, [2, 8192, 8, 4, 128]);  unsqueeze_44 = None
        clone_44 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_767 = torch.ops.aten.view.default(clone_44, [2, 8192, 32, 128]);  clone_44 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_761, 3);  view_761 = None
        expand_45 = torch.ops.aten.expand.default(unsqueeze_45, [2, 8192, 8, 4, 128]);  unsqueeze_45 = None
        clone_45 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_768 = torch.ops.aten.view.default(clone_45, [2, 8192, 32, 128]);  clone_45 = None
        permute_245 = torch.ops.aten.permute.default(convert_element_type_741, [0, 2, 1, 3]);  convert_element_type_741 = None
        permute_246 = torch.ops.aten.permute.default(view_767, [0, 2, 1, 3]);  view_767 = None
        permute_247 = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_245, permute_246, permute_247, None, True, 0.0, True);  permute_245 = permute_246 = permute_247 = None
        getitem_198 = _scaled_dot_product_cudnn_attention_22[0]
        getitem_199 = _scaled_dot_product_cudnn_attention_22[1]
        getitem_204 = _scaled_dot_product_cudnn_attention_22[6]
        getitem_205 = _scaled_dot_product_cudnn_attention_22[7];  _scaled_dot_product_cudnn_attention_22 = None
        permute_248 = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3])
        view_769 = torch.ops.aten.view.default(permute_248, [2, 8192, -1]);  permute_248 = None
        convert_element_type_743 = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16)
        all_gather_into_tensor_203 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_743, 64, '0');  convert_element_type_743 = None
        wait_tensor_203 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_203);  all_gather_into_tensor_203 = None
        permute_249 = torch.ops.aten.permute.default(wait_tensor_203, [1, 0]);  wait_tensor_203 = None
        view_771 = torch.ops.aten.view.default(view_769, [16384, 4096]);  view_769 = None
        mm_157 = torch.ops.aten.mm.default(view_771, permute_249);  view_771 = permute_249 = None
        view_772 = torch.ops.aten.view.default(mm_157, [2, 8192, 4096]);  mm_157 = None
        add_89 = torch.ops.aten.add.Tensor(add_87, view_772);  view_772 = None
        convert_element_type_746 = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16)
        all_gather_into_tensor_204 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_746, 64, '0');  convert_element_type_746 = None
        wait_tensor_204 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_204);  all_gather_into_tensor_204 = None
        convert_element_type_747 = torch.ops.prims.convert_element_type.default(add_89, torch.float32)
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_747, 2)
        mean_45 = torch.ops.aten.mean.dim(pow_46, [2], True);  pow_46 = None
        add_90 = torch.ops.aten.add.Scalar(mean_45, 1e-05);  mean_45 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_180 = torch.ops.aten.mul.Tensor(convert_element_type_747, rsqrt_45);  convert_element_type_747 = rsqrt_45 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, wait_tensor_204);  mul_180 = wait_tensor_204 = None
        convert_element_type_748 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        convert_element_type_749 = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16)
        all_gather_into_tensor_205 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_749, 64, '0');  convert_element_type_749 = None
        wait_tensor_205 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_205);  all_gather_into_tensor_205 = None
        permute_250 = torch.ops.aten.permute.default(wait_tensor_205, [1, 0]);  wait_tensor_205 = None
        view_775 = torch.ops.aten.view.default(convert_element_type_748, [16384, 4096]);  convert_element_type_748 = None
        mm_158 = torch.ops.aten.mm.default(view_775, permute_250);  permute_250 = None
        view_776 = torch.ops.aten.view.default(mm_158, [2, 8192, 14336])
        convert_element_type_752 = torch.ops.prims.convert_element_type.default(view_776, torch.float32);  view_776 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(convert_element_type_752)
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_752, sigmoid_22);  convert_element_type_752 = sigmoid_22 = None
        convert_element_type_753 = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        convert_element_type_754 = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16)
        all_gather_into_tensor_206 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_754, 64, '0');  convert_element_type_754 = None
        wait_tensor_206 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_206);  all_gather_into_tensor_206 = None
        permute_251 = torch.ops.aten.permute.default(wait_tensor_206, [1, 0]);  wait_tensor_206 = None
        mm_159 = torch.ops.aten.mm.default(view_775, permute_251);  view_775 = permute_251 = None
        view_779 = torch.ops.aten.view.default(mm_159, [2, 8192, 14336]);  mm_159 = None
        mul_183 = torch.ops.aten.mul.Tensor(convert_element_type_753, view_779);  convert_element_type_753 = view_779 = None
        convert_element_type_757 = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16)
        all_gather_into_tensor_207 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_757, 64, '0');  convert_element_type_757 = None
        wait_tensor_207 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_207);  all_gather_into_tensor_207 = None
        permute_252 = torch.ops.aten.permute.default(wait_tensor_207, [1, 0]);  wait_tensor_207 = None
        view_781 = torch.ops.aten.view.default(mul_183, [16384, 14336]);  mul_183 = None
        mm_160 = torch.ops.aten.mm.default(view_781, permute_252);  view_781 = permute_252 = None
        view_782 = torch.ops.aten.view.default(mm_160, [2, 8192, 4096]);  mm_160 = None
        add_91 = torch.ops.aten.add.Tensor(add_89, view_782);  add_89 = view_782 = None
        convert_element_type_760 = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16)
        all_gather_into_tensor_208 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_760, 64, '0');  convert_element_type_760 = None
        wait_tensor_208 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_208);  all_gather_into_tensor_208 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_761, 2)
        mean_46 = torch.ops.aten.mean.dim(pow_47, [2], True);  pow_47 = None
        add_92 = torch.ops.aten.add.Scalar(mean_46, 1e-05);  mean_46 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_184 = torch.ops.aten.mul.Tensor(convert_element_type_761, rsqrt_46);  convert_element_type_761 = rsqrt_46 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, wait_tensor_208);  mul_184 = wait_tensor_208 = None
        convert_element_type_762 = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        convert_element_type_763 = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16)
        all_gather_into_tensor_209 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_763, 64, '0');  convert_element_type_763 = None
        wait_tensor_209 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_209);  all_gather_into_tensor_209 = None
        permute_253 = torch.ops.aten.permute.default(wait_tensor_209, [1, 0]);  wait_tensor_209 = None
        view_785 = torch.ops.aten.view.default(convert_element_type_762, [16384, 4096]);  convert_element_type_762 = None
        mm_161 = torch.ops.aten.mm.default(view_785, permute_253);  permute_253 = None
        view_786 = torch.ops.aten.view.default(mm_161, [2, 8192, 4096])
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16)
        all_gather_into_tensor_210 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_766, 64, '0');  convert_element_type_766 = None
        wait_tensor_210 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_210);  all_gather_into_tensor_210 = None
        permute_254 = torch.ops.aten.permute.default(wait_tensor_210, [1, 0]);  wait_tensor_210 = None
        mm_162 = torch.ops.aten.mm.default(view_785, permute_254);  permute_254 = None
        view_789 = torch.ops.aten.view.default(mm_162, [2, 8192, 1024]);  mm_162 = None
        convert_element_type_769 = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16)
        all_gather_into_tensor_211 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_769, 64, '0');  convert_element_type_769 = None
        wait_tensor_211 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_211);  all_gather_into_tensor_211 = None
        permute_255 = torch.ops.aten.permute.default(wait_tensor_211, [1, 0]);  wait_tensor_211 = None
        mm_163 = torch.ops.aten.mm.default(view_785, permute_255);  view_785 = permute_255 = None
        view_792 = torch.ops.aten.view.default(mm_163, [2, 8192, 1024])
        view_793 = torch.ops.aten.view.default(view_786, [2, 8192, -1, 128]);  view_786 = None
        view_794 = torch.ops.aten.view.default(view_789, [2, 8192, -1, 128]);  view_789 = None
        view_795 = torch.ops.aten.view.default(view_792, [2, 8192, -1, 128]);  view_792 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(view_793, torch.float32);  view_793 = None
        view_796 = torch.ops.aten.view.default(convert_element_type_772, [2, 8192, 32, -1, 2]);  convert_element_type_772 = None
        view_as_complex_46 = torch.ops.aten.view_as_complex.default(view_796);  view_796 = None
        convert_element_type_773 = torch.ops.prims.convert_element_type.default(view_794, torch.float32);  view_794 = None
        view_797 = torch.ops.aten.view.default(convert_element_type_773, [2, 8192, 8, -1, 2]);  convert_element_type_773 = None
        view_as_complex_47 = torch.ops.aten.view_as_complex.default(view_797);  view_797 = None
        mul_186 = torch.ops.aten.mul.Tensor(view_as_complex_46, view_16);  view_as_complex_46 = None
        view_as_real_46 = torch.ops.aten.view_as_real.default(mul_186);  mul_186 = None
        view_799 = torch.ops.aten.view.default(view_as_real_46, [2, 8192, 32, 128]);  view_as_real_46 = None
        mul_187 = torch.ops.aten.mul.Tensor(view_as_complex_47, view_16);  view_as_complex_47 = None
        view_as_real_47 = torch.ops.aten.view_as_real.default(mul_187);  mul_187 = None
        view_800 = torch.ops.aten.view.default(view_as_real_47, [2, 8192, 8, 128]);  view_as_real_47 = None
        convert_element_type_774 = torch.ops.prims.convert_element_type.default(view_799, torch.bfloat16);  view_799 = None
        convert_element_type_775 = torch.ops.prims.convert_element_type.default(view_800, torch.bfloat16);  view_800 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(convert_element_type_775, 3);  convert_element_type_775 = None
        expand_46 = torch.ops.aten.expand.default(unsqueeze_46, [2, 8192, 8, 4, 128]);  unsqueeze_46 = None
        clone_46 = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_801 = torch.ops.aten.view.default(clone_46, [2, 8192, 32, 128]);  clone_46 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(view_795, 3);  view_795 = None
        expand_47 = torch.ops.aten.expand.default(unsqueeze_47, [2, 8192, 8, 4, 128]);  unsqueeze_47 = None
        clone_47 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_802 = torch.ops.aten.view.default(clone_47, [2, 8192, 32, 128]);  clone_47 = None
        permute_256 = torch.ops.aten.permute.default(convert_element_type_774, [0, 2, 1, 3]);  convert_element_type_774 = None
        permute_257 = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None
        permute_258 = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_256, permute_257, permute_258, None, True, 0.0, True);  permute_256 = permute_257 = permute_258 = None
        getitem_207 = _scaled_dot_product_cudnn_attention_23[0]
        getitem_208 = _scaled_dot_product_cudnn_attention_23[1]
        getitem_213 = _scaled_dot_product_cudnn_attention_23[6]
        getitem_214 = _scaled_dot_product_cudnn_attention_23[7];  _scaled_dot_product_cudnn_attention_23 = None
        permute_259 = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3])
        view_803 = torch.ops.aten.view.default(permute_259, [2, 8192, -1]);  permute_259 = None
        convert_element_type_776 = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16)
        all_gather_into_tensor_212 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_776, 64, '0');  convert_element_type_776 = None
        wait_tensor_212 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_212);  all_gather_into_tensor_212 = None
        permute_260 = torch.ops.aten.permute.default(wait_tensor_212, [1, 0]);  wait_tensor_212 = None
        view_805 = torch.ops.aten.view.default(view_803, [16384, 4096]);  view_803 = None
        mm_164 = torch.ops.aten.mm.default(view_805, permute_260);  view_805 = permute_260 = None
        view_806 = torch.ops.aten.view.default(mm_164, [2, 8192, 4096]);  mm_164 = None
        add_93 = torch.ops.aten.add.Tensor(add_91, view_806);  view_806 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16)
        all_gather_into_tensor_213 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_779, 64, '0');  convert_element_type_779 = None
        wait_tensor_213 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_213);  all_gather_into_tensor_213 = None
        convert_element_type_780 = torch.ops.prims.convert_element_type.default(add_93, torch.float32)
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_780, 2)
        mean_47 = torch.ops.aten.mean.dim(pow_48, [2], True);  pow_48 = None
        add_94 = torch.ops.aten.add.Scalar(mean_47, 1e-05);  mean_47 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_188 = torch.ops.aten.mul.Tensor(convert_element_type_780, rsqrt_47);  convert_element_type_780 = rsqrt_47 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, wait_tensor_213);  mul_188 = wait_tensor_213 = None
        convert_element_type_781 = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16)
        all_gather_into_tensor_214 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_782, 64, '0');  convert_element_type_782 = None
        wait_tensor_214 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_214);  all_gather_into_tensor_214 = None
        permute_261 = torch.ops.aten.permute.default(wait_tensor_214, [1, 0]);  wait_tensor_214 = None
        view_809 = torch.ops.aten.view.default(convert_element_type_781, [16384, 4096]);  convert_element_type_781 = None
        mm_165 = torch.ops.aten.mm.default(view_809, permute_261);  permute_261 = None
        view_810 = torch.ops.aten.view.default(mm_165, [2, 8192, 14336])
        convert_element_type_785 = torch.ops.prims.convert_element_type.default(view_810, torch.float32);  view_810 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(convert_element_type_785)
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_785, sigmoid_23);  convert_element_type_785 = sigmoid_23 = None
        convert_element_type_786 = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        convert_element_type_787 = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16)
        all_gather_into_tensor_215 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_787, 64, '0');  convert_element_type_787 = None
        wait_tensor_215 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_215);  all_gather_into_tensor_215 = None
        permute_262 = torch.ops.aten.permute.default(wait_tensor_215, [1, 0]);  wait_tensor_215 = None
        mm_166 = torch.ops.aten.mm.default(view_809, permute_262);  view_809 = permute_262 = None
        view_813 = torch.ops.aten.view.default(mm_166, [2, 8192, 14336]);  mm_166 = None
        mul_191 = torch.ops.aten.mul.Tensor(convert_element_type_786, view_813);  convert_element_type_786 = view_813 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16)
        all_gather_into_tensor_216 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_790, 64, '0');  convert_element_type_790 = None
        wait_tensor_216 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_216);  all_gather_into_tensor_216 = None
        permute_263 = torch.ops.aten.permute.default(wait_tensor_216, [1, 0]);  wait_tensor_216 = None
        view_815 = torch.ops.aten.view.default(mul_191, [16384, 14336]);  mul_191 = None
        mm_167 = torch.ops.aten.mm.default(view_815, permute_263);  view_815 = permute_263 = None
        view_816 = torch.ops.aten.view.default(mm_167, [2, 8192, 4096]);  mm_167 = None
        add_95 = torch.ops.aten.add.Tensor(add_93, view_816);  add_93 = view_816 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16)
        all_gather_into_tensor_217 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_793, 64, '0');  convert_element_type_793 = None
        wait_tensor_217 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_217);  all_gather_into_tensor_217 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_794, 2)
        mean_48 = torch.ops.aten.mean.dim(pow_49, [2], True);  pow_49 = None
        add_96 = torch.ops.aten.add.Scalar(mean_48, 1e-05);  mean_48 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_794, rsqrt_48);  convert_element_type_794 = rsqrt_48 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, wait_tensor_217);  mul_192 = wait_tensor_217 = None
        convert_element_type_795 = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        convert_element_type_796 = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16)
        all_gather_into_tensor_218 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_796, 64, '0');  convert_element_type_796 = None
        wait_tensor_218 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_218);  all_gather_into_tensor_218 = None
        permute_264 = torch.ops.aten.permute.default(wait_tensor_218, [1, 0]);  wait_tensor_218 = None
        view_819 = torch.ops.aten.view.default(convert_element_type_795, [16384, 4096]);  convert_element_type_795 = None
        mm_168 = torch.ops.aten.mm.default(view_819, permute_264);  permute_264 = None
        view_820 = torch.ops.aten.view.default(mm_168, [2, 8192, 4096])
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16)
        all_gather_into_tensor_219 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_799, 64, '0');  convert_element_type_799 = None
        wait_tensor_219 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_219);  all_gather_into_tensor_219 = None
        permute_265 = torch.ops.aten.permute.default(wait_tensor_219, [1, 0]);  wait_tensor_219 = None
        mm_169 = torch.ops.aten.mm.default(view_819, permute_265);  permute_265 = None
        view_823 = torch.ops.aten.view.default(mm_169, [2, 8192, 1024]);  mm_169 = None
        convert_element_type_802 = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16)
        all_gather_into_tensor_220 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_802, 64, '0');  convert_element_type_802 = None
        wait_tensor_220 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_220);  all_gather_into_tensor_220 = None
        permute_266 = torch.ops.aten.permute.default(wait_tensor_220, [1, 0]);  wait_tensor_220 = None
        mm_170 = torch.ops.aten.mm.default(view_819, permute_266);  view_819 = permute_266 = None
        view_826 = torch.ops.aten.view.default(mm_170, [2, 8192, 1024])
        view_827 = torch.ops.aten.view.default(view_820, [2, 8192, -1, 128]);  view_820 = None
        view_828 = torch.ops.aten.view.default(view_823, [2, 8192, -1, 128]);  view_823 = None
        view_829 = torch.ops.aten.view.default(view_826, [2, 8192, -1, 128]);  view_826 = None
        convert_element_type_805 = torch.ops.prims.convert_element_type.default(view_827, torch.float32);  view_827 = None
        view_830 = torch.ops.aten.view.default(convert_element_type_805, [2, 8192, 32, -1, 2]);  convert_element_type_805 = None
        view_as_complex_48 = torch.ops.aten.view_as_complex.default(view_830);  view_830 = None
        convert_element_type_806 = torch.ops.prims.convert_element_type.default(view_828, torch.float32);  view_828 = None
        view_831 = torch.ops.aten.view.default(convert_element_type_806, [2, 8192, 8, -1, 2]);  convert_element_type_806 = None
        view_as_complex_49 = torch.ops.aten.view_as_complex.default(view_831);  view_831 = None
        mul_194 = torch.ops.aten.mul.Tensor(view_as_complex_48, view_16);  view_as_complex_48 = None
        view_as_real_48 = torch.ops.aten.view_as_real.default(mul_194);  mul_194 = None
        view_833 = torch.ops.aten.view.default(view_as_real_48, [2, 8192, 32, 128]);  view_as_real_48 = None
        mul_195 = torch.ops.aten.mul.Tensor(view_as_complex_49, view_16);  view_as_complex_49 = None
        view_as_real_49 = torch.ops.aten.view_as_real.default(mul_195);  mul_195 = None
        view_834 = torch.ops.aten.view.default(view_as_real_49, [2, 8192, 8, 128]);  view_as_real_49 = None
        convert_element_type_807 = torch.ops.prims.convert_element_type.default(view_833, torch.bfloat16);  view_833 = None
        convert_element_type_808 = torch.ops.prims.convert_element_type.default(view_834, torch.bfloat16);  view_834 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(convert_element_type_808, 3);  convert_element_type_808 = None
        expand_48 = torch.ops.aten.expand.default(unsqueeze_48, [2, 8192, 8, 4, 128]);  unsqueeze_48 = None
        clone_48 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_835 = torch.ops.aten.view.default(clone_48, [2, 8192, 32, 128]);  clone_48 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(view_829, 3);  view_829 = None
        expand_49 = torch.ops.aten.expand.default(unsqueeze_49, [2, 8192, 8, 4, 128]);  unsqueeze_49 = None
        clone_49 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_836 = torch.ops.aten.view.default(clone_49, [2, 8192, 32, 128]);  clone_49 = None
        permute_267 = torch.ops.aten.permute.default(convert_element_type_807, [0, 2, 1, 3]);  convert_element_type_807 = None
        permute_268 = torch.ops.aten.permute.default(view_835, [0, 2, 1, 3]);  view_835 = None
        permute_269 = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_267, permute_268, permute_269, None, True, 0.0, True);  permute_267 = permute_268 = permute_269 = None
        getitem_216 = _scaled_dot_product_cudnn_attention_24[0]
        getitem_217 = _scaled_dot_product_cudnn_attention_24[1]
        getitem_222 = _scaled_dot_product_cudnn_attention_24[6]
        getitem_223 = _scaled_dot_product_cudnn_attention_24[7];  _scaled_dot_product_cudnn_attention_24 = None
        permute_270 = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3])
        view_837 = torch.ops.aten.view.default(permute_270, [2, 8192, -1]);  permute_270 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16)
        all_gather_into_tensor_221 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_809, 64, '0');  convert_element_type_809 = None
        wait_tensor_221 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_221);  all_gather_into_tensor_221 = None
        permute_271 = torch.ops.aten.permute.default(wait_tensor_221, [1, 0]);  wait_tensor_221 = None
        view_839 = torch.ops.aten.view.default(view_837, [16384, 4096]);  view_837 = None
        mm_171 = torch.ops.aten.mm.default(view_839, permute_271);  view_839 = permute_271 = None
        view_840 = torch.ops.aten.view.default(mm_171, [2, 8192, 4096]);  mm_171 = None
        add_97 = torch.ops.aten.add.Tensor(add_95, view_840);  view_840 = None
        convert_element_type_812 = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16)
        all_gather_into_tensor_222 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_812, 64, '0');  convert_element_type_812 = None
        wait_tensor_222 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_222);  all_gather_into_tensor_222 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_813, 2)
        mean_49 = torch.ops.aten.mean.dim(pow_50, [2], True);  pow_50 = None
        add_98 = torch.ops.aten.add.Scalar(mean_49, 1e-05);  mean_49 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_196 = torch.ops.aten.mul.Tensor(convert_element_type_813, rsqrt_49);  convert_element_type_813 = rsqrt_49 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, wait_tensor_222);  mul_196 = wait_tensor_222 = None
        convert_element_type_814 = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16);  mul_197 = None
        convert_element_type_815 = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16)
        all_gather_into_tensor_223 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_815, 64, '0');  convert_element_type_815 = None
        wait_tensor_223 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_223);  all_gather_into_tensor_223 = None
        permute_272 = torch.ops.aten.permute.default(wait_tensor_223, [1, 0]);  wait_tensor_223 = None
        view_843 = torch.ops.aten.view.default(convert_element_type_814, [16384, 4096]);  convert_element_type_814 = None
        mm_172 = torch.ops.aten.mm.default(view_843, permute_272);  permute_272 = None
        view_844 = torch.ops.aten.view.default(mm_172, [2, 8192, 14336])
        convert_element_type_818 = torch.ops.prims.convert_element_type.default(view_844, torch.float32);  view_844 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(convert_element_type_818)
        mul_198 = torch.ops.aten.mul.Tensor(convert_element_type_818, sigmoid_24);  convert_element_type_818 = sigmoid_24 = None
        convert_element_type_819 = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16)
        all_gather_into_tensor_224 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_820, 64, '0');  convert_element_type_820 = None
        wait_tensor_224 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_224);  all_gather_into_tensor_224 = None
        permute_273 = torch.ops.aten.permute.default(wait_tensor_224, [1, 0]);  wait_tensor_224 = None
        mm_173 = torch.ops.aten.mm.default(view_843, permute_273);  view_843 = permute_273 = None
        view_847 = torch.ops.aten.view.default(mm_173, [2, 8192, 14336]);  mm_173 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_819, view_847);  convert_element_type_819 = view_847 = None
        convert_element_type_823 = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16)
        all_gather_into_tensor_225 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_823, 64, '0');  convert_element_type_823 = None
        wait_tensor_225 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_225);  all_gather_into_tensor_225 = None
        permute_274 = torch.ops.aten.permute.default(wait_tensor_225, [1, 0]);  wait_tensor_225 = None
        view_849 = torch.ops.aten.view.default(mul_199, [16384, 14336]);  mul_199 = None
        mm_174 = torch.ops.aten.mm.default(view_849, permute_274);  view_849 = permute_274 = None
        view_850 = torch.ops.aten.view.default(mm_174, [2, 8192, 4096]);  mm_174 = None
        add_99 = torch.ops.aten.add.Tensor(add_97, view_850);  add_97 = view_850 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16)
        all_gather_into_tensor_226 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_826, 64, '0');  convert_element_type_826 = None
        wait_tensor_226 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_226);  all_gather_into_tensor_226 = None
        convert_element_type_827 = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_827, 2)
        mean_50 = torch.ops.aten.mean.dim(pow_51, [2], True);  pow_51 = None
        add_100 = torch.ops.aten.add.Scalar(mean_50, 1e-05);  mean_50 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_827, rsqrt_50);  convert_element_type_827 = rsqrt_50 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, wait_tensor_226);  mul_200 = wait_tensor_226 = None
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        convert_element_type_829 = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16)
        all_gather_into_tensor_227 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_829, 64, '0');  convert_element_type_829 = None
        wait_tensor_227 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_227);  all_gather_into_tensor_227 = None
        permute_275 = torch.ops.aten.permute.default(wait_tensor_227, [1, 0]);  wait_tensor_227 = None
        view_853 = torch.ops.aten.view.default(convert_element_type_828, [16384, 4096]);  convert_element_type_828 = None
        mm_175 = torch.ops.aten.mm.default(view_853, permute_275);  permute_275 = None
        view_854 = torch.ops.aten.view.default(mm_175, [2, 8192, 4096])
        convert_element_type_832 = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16)
        all_gather_into_tensor_228 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_832, 64, '0');  convert_element_type_832 = None
        wait_tensor_228 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_228);  all_gather_into_tensor_228 = None
        permute_276 = torch.ops.aten.permute.default(wait_tensor_228, [1, 0]);  wait_tensor_228 = None
        mm_176 = torch.ops.aten.mm.default(view_853, permute_276);  permute_276 = None
        view_857 = torch.ops.aten.view.default(mm_176, [2, 8192, 1024]);  mm_176 = None
        convert_element_type_835 = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16)
        all_gather_into_tensor_229 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_835, 64, '0');  convert_element_type_835 = None
        wait_tensor_229 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_229);  all_gather_into_tensor_229 = None
        permute_277 = torch.ops.aten.permute.default(wait_tensor_229, [1, 0]);  wait_tensor_229 = None
        mm_177 = torch.ops.aten.mm.default(view_853, permute_277);  view_853 = permute_277 = None
        view_860 = torch.ops.aten.view.default(mm_177, [2, 8192, 1024])
        view_861 = torch.ops.aten.view.default(view_854, [2, 8192, -1, 128]);  view_854 = None
        view_862 = torch.ops.aten.view.default(view_857, [2, 8192, -1, 128]);  view_857 = None
        view_863 = torch.ops.aten.view.default(view_860, [2, 8192, -1, 128]);  view_860 = None
        convert_element_type_838 = torch.ops.prims.convert_element_type.default(view_861, torch.float32);  view_861 = None
        view_864 = torch.ops.aten.view.default(convert_element_type_838, [2, 8192, 32, -1, 2]);  convert_element_type_838 = None
        view_as_complex_50 = torch.ops.aten.view_as_complex.default(view_864);  view_864 = None
        convert_element_type_839 = torch.ops.prims.convert_element_type.default(view_862, torch.float32);  view_862 = None
        view_865 = torch.ops.aten.view.default(convert_element_type_839, [2, 8192, 8, -1, 2]);  convert_element_type_839 = None
        view_as_complex_51 = torch.ops.aten.view_as_complex.default(view_865);  view_865 = None
        mul_202 = torch.ops.aten.mul.Tensor(view_as_complex_50, view_16);  view_as_complex_50 = None
        view_as_real_50 = torch.ops.aten.view_as_real.default(mul_202);  mul_202 = None
        view_867 = torch.ops.aten.view.default(view_as_real_50, [2, 8192, 32, 128]);  view_as_real_50 = None
        mul_203 = torch.ops.aten.mul.Tensor(view_as_complex_51, view_16);  view_as_complex_51 = None
        view_as_real_51 = torch.ops.aten.view_as_real.default(mul_203);  mul_203 = None
        view_868 = torch.ops.aten.view.default(view_as_real_51, [2, 8192, 8, 128]);  view_as_real_51 = None
        convert_element_type_840 = torch.ops.prims.convert_element_type.default(view_867, torch.bfloat16);  view_867 = None
        convert_element_type_841 = torch.ops.prims.convert_element_type.default(view_868, torch.bfloat16);  view_868 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(convert_element_type_841, 3);  convert_element_type_841 = None
        expand_50 = torch.ops.aten.expand.default(unsqueeze_50, [2, 8192, 8, 4, 128]);  unsqueeze_50 = None
        clone_50 = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_869 = torch.ops.aten.view.default(clone_50, [2, 8192, 32, 128]);  clone_50 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(view_863, 3);  view_863 = None
        expand_51 = torch.ops.aten.expand.default(unsqueeze_51, [2, 8192, 8, 4, 128]);  unsqueeze_51 = None
        clone_51 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_870 = torch.ops.aten.view.default(clone_51, [2, 8192, 32, 128]);  clone_51 = None
        permute_278 = torch.ops.aten.permute.default(convert_element_type_840, [0, 2, 1, 3]);  convert_element_type_840 = None
        permute_279 = torch.ops.aten.permute.default(view_869, [0, 2, 1, 3]);  view_869 = None
        permute_280 = torch.ops.aten.permute.default(view_870, [0, 2, 1, 3]);  view_870 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_278, permute_279, permute_280, None, True, 0.0, True);  permute_278 = permute_279 = permute_280 = None
        getitem_225 = _scaled_dot_product_cudnn_attention_25[0]
        getitem_226 = _scaled_dot_product_cudnn_attention_25[1]
        getitem_231 = _scaled_dot_product_cudnn_attention_25[6]
        getitem_232 = _scaled_dot_product_cudnn_attention_25[7];  _scaled_dot_product_cudnn_attention_25 = None
        permute_281 = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3])
        view_871 = torch.ops.aten.view.default(permute_281, [2, 8192, -1]);  permute_281 = None
        convert_element_type_842 = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16)
        all_gather_into_tensor_230 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_842, 64, '0');  convert_element_type_842 = None
        wait_tensor_230 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_230);  all_gather_into_tensor_230 = None
        permute_282 = torch.ops.aten.permute.default(wait_tensor_230, [1, 0]);  wait_tensor_230 = None
        view_873 = torch.ops.aten.view.default(view_871, [16384, 4096]);  view_871 = None
        mm_178 = torch.ops.aten.mm.default(view_873, permute_282);  view_873 = permute_282 = None
        view_874 = torch.ops.aten.view.default(mm_178, [2, 8192, 4096]);  mm_178 = None
        add_101 = torch.ops.aten.add.Tensor(add_99, view_874);  view_874 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16)
        all_gather_into_tensor_231 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_845, 64, '0');  convert_element_type_845 = None
        wait_tensor_231 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_231);  all_gather_into_tensor_231 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_846, 2)
        mean_51 = torch.ops.aten.mean.dim(pow_52, [2], True);  pow_52 = None
        add_102 = torch.ops.aten.add.Scalar(mean_51, 1e-05);  mean_51 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_204 = torch.ops.aten.mul.Tensor(convert_element_type_846, rsqrt_51);  convert_element_type_846 = rsqrt_51 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, wait_tensor_231);  mul_204 = wait_tensor_231 = None
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16);  mul_205 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16)
        all_gather_into_tensor_232 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_848, 64, '0');  convert_element_type_848 = None
        wait_tensor_232 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_232);  all_gather_into_tensor_232 = None
        permute_283 = torch.ops.aten.permute.default(wait_tensor_232, [1, 0]);  wait_tensor_232 = None
        view_877 = torch.ops.aten.view.default(convert_element_type_847, [16384, 4096]);  convert_element_type_847 = None
        mm_179 = torch.ops.aten.mm.default(view_877, permute_283);  permute_283 = None
        view_878 = torch.ops.aten.view.default(mm_179, [2, 8192, 14336])
        convert_element_type_851 = torch.ops.prims.convert_element_type.default(view_878, torch.float32);  view_878 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(convert_element_type_851)
        mul_206 = torch.ops.aten.mul.Tensor(convert_element_type_851, sigmoid_25);  convert_element_type_851 = sigmoid_25 = None
        convert_element_type_852 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16)
        all_gather_into_tensor_233 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_853, 64, '0');  convert_element_type_853 = None
        wait_tensor_233 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_233);  all_gather_into_tensor_233 = None
        permute_284 = torch.ops.aten.permute.default(wait_tensor_233, [1, 0]);  wait_tensor_233 = None
        mm_180 = torch.ops.aten.mm.default(view_877, permute_284);  view_877 = permute_284 = None
        view_881 = torch.ops.aten.view.default(mm_180, [2, 8192, 14336]);  mm_180 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_852, view_881);  convert_element_type_852 = view_881 = None
        convert_element_type_856 = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16)
        all_gather_into_tensor_234 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_856, 64, '0');  convert_element_type_856 = None
        wait_tensor_234 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_234);  all_gather_into_tensor_234 = None
        permute_285 = torch.ops.aten.permute.default(wait_tensor_234, [1, 0]);  wait_tensor_234 = None
        view_883 = torch.ops.aten.view.default(mul_207, [16384, 14336]);  mul_207 = None
        mm_181 = torch.ops.aten.mm.default(view_883, permute_285);  view_883 = permute_285 = None
        view_884 = torch.ops.aten.view.default(mm_181, [2, 8192, 4096]);  mm_181 = None
        add_103 = torch.ops.aten.add.Tensor(add_101, view_884);  add_101 = view_884 = None
        convert_element_type_859 = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16)
        all_gather_into_tensor_235 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_859, 64, '0');  convert_element_type_859 = None
        wait_tensor_235 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_235);  all_gather_into_tensor_235 = None
        convert_element_type_860 = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_860, 2)
        mean_52 = torch.ops.aten.mean.dim(pow_53, [2], True);  pow_53 = None
        add_104 = torch.ops.aten.add.Scalar(mean_52, 1e-05);  mean_52 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_860, rsqrt_52);  convert_element_type_860 = rsqrt_52 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, wait_tensor_235);  mul_208 = wait_tensor_235 = None
        convert_element_type_861 = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        convert_element_type_862 = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16)
        all_gather_into_tensor_236 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_862, 64, '0');  convert_element_type_862 = None
        wait_tensor_236 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_236);  all_gather_into_tensor_236 = None
        permute_286 = torch.ops.aten.permute.default(wait_tensor_236, [1, 0]);  wait_tensor_236 = None
        view_887 = torch.ops.aten.view.default(convert_element_type_861, [16384, 4096]);  convert_element_type_861 = None
        mm_182 = torch.ops.aten.mm.default(view_887, permute_286);  permute_286 = None
        view_888 = torch.ops.aten.view.default(mm_182, [2, 8192, 4096])
        convert_element_type_865 = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16)
        all_gather_into_tensor_237 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_865, 64, '0');  convert_element_type_865 = None
        wait_tensor_237 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_237);  all_gather_into_tensor_237 = None
        permute_287 = torch.ops.aten.permute.default(wait_tensor_237, [1, 0]);  wait_tensor_237 = None
        mm_183 = torch.ops.aten.mm.default(view_887, permute_287);  permute_287 = None
        view_891 = torch.ops.aten.view.default(mm_183, [2, 8192, 1024]);  mm_183 = None
        convert_element_type_868 = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16)
        all_gather_into_tensor_238 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_868, 64, '0');  convert_element_type_868 = None
        wait_tensor_238 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_238);  all_gather_into_tensor_238 = None
        permute_288 = torch.ops.aten.permute.default(wait_tensor_238, [1, 0]);  wait_tensor_238 = None
        mm_184 = torch.ops.aten.mm.default(view_887, permute_288);  view_887 = permute_288 = None
        view_894 = torch.ops.aten.view.default(mm_184, [2, 8192, 1024])
        view_895 = torch.ops.aten.view.default(view_888, [2, 8192, -1, 128]);  view_888 = None
        view_896 = torch.ops.aten.view.default(view_891, [2, 8192, -1, 128]);  view_891 = None
        view_897 = torch.ops.aten.view.default(view_894, [2, 8192, -1, 128]);  view_894 = None
        convert_element_type_871 = torch.ops.prims.convert_element_type.default(view_895, torch.float32);  view_895 = None
        view_898 = torch.ops.aten.view.default(convert_element_type_871, [2, 8192, 32, -1, 2]);  convert_element_type_871 = None
        view_as_complex_52 = torch.ops.aten.view_as_complex.default(view_898);  view_898 = None
        convert_element_type_872 = torch.ops.prims.convert_element_type.default(view_896, torch.float32);  view_896 = None
        view_899 = torch.ops.aten.view.default(convert_element_type_872, [2, 8192, 8, -1, 2]);  convert_element_type_872 = None
        view_as_complex_53 = torch.ops.aten.view_as_complex.default(view_899);  view_899 = None
        mul_210 = torch.ops.aten.mul.Tensor(view_as_complex_52, view_16);  view_as_complex_52 = None
        view_as_real_52 = torch.ops.aten.view_as_real.default(mul_210);  mul_210 = None
        view_901 = torch.ops.aten.view.default(view_as_real_52, [2, 8192, 32, 128]);  view_as_real_52 = None
        mul_211 = torch.ops.aten.mul.Tensor(view_as_complex_53, view_16);  view_as_complex_53 = None
        view_as_real_53 = torch.ops.aten.view_as_real.default(mul_211);  mul_211 = None
        view_902 = torch.ops.aten.view.default(view_as_real_53, [2, 8192, 8, 128]);  view_as_real_53 = None
        convert_element_type_873 = torch.ops.prims.convert_element_type.default(view_901, torch.bfloat16);  view_901 = None
        convert_element_type_874 = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(convert_element_type_874, 3);  convert_element_type_874 = None
        expand_52 = torch.ops.aten.expand.default(unsqueeze_52, [2, 8192, 8, 4, 128]);  unsqueeze_52 = None
        clone_52 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_903 = torch.ops.aten.view.default(clone_52, [2, 8192, 32, 128]);  clone_52 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(view_897, 3);  view_897 = None
        expand_53 = torch.ops.aten.expand.default(unsqueeze_53, [2, 8192, 8, 4, 128]);  unsqueeze_53 = None
        clone_53 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_904 = torch.ops.aten.view.default(clone_53, [2, 8192, 32, 128]);  clone_53 = None
        permute_289 = torch.ops.aten.permute.default(convert_element_type_873, [0, 2, 1, 3]);  convert_element_type_873 = None
        permute_290 = torch.ops.aten.permute.default(view_903, [0, 2, 1, 3]);  view_903 = None
        permute_291 = torch.ops.aten.permute.default(view_904, [0, 2, 1, 3]);  view_904 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_289, permute_290, permute_291, None, True, 0.0, True);  permute_289 = permute_290 = permute_291 = None
        getitem_234 = _scaled_dot_product_cudnn_attention_26[0]
        getitem_235 = _scaled_dot_product_cudnn_attention_26[1]
        getitem_240 = _scaled_dot_product_cudnn_attention_26[6]
        getitem_241 = _scaled_dot_product_cudnn_attention_26[7];  _scaled_dot_product_cudnn_attention_26 = None
        permute_292 = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3])
        view_905 = torch.ops.aten.view.default(permute_292, [2, 8192, -1]);  permute_292 = None
        convert_element_type_875 = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16)
        all_gather_into_tensor_239 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_875, 64, '0');  convert_element_type_875 = None
        wait_tensor_239 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_239);  all_gather_into_tensor_239 = None
        permute_293 = torch.ops.aten.permute.default(wait_tensor_239, [1, 0]);  wait_tensor_239 = None
        view_907 = torch.ops.aten.view.default(view_905, [16384, 4096]);  view_905 = None
        mm_185 = torch.ops.aten.mm.default(view_907, permute_293);  view_907 = permute_293 = None
        view_908 = torch.ops.aten.view.default(mm_185, [2, 8192, 4096]);  mm_185 = None
        add_105 = torch.ops.aten.add.Tensor(add_103, view_908);  view_908 = None
        convert_element_type_878 = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16)
        all_gather_into_tensor_240 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_878, 64, '0');  convert_element_type_878 = None
        wait_tensor_240 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_240);  all_gather_into_tensor_240 = None
        convert_element_type_879 = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_879, 2)
        mean_53 = torch.ops.aten.mean.dim(pow_54, [2], True);  pow_54 = None
        add_106 = torch.ops.aten.add.Scalar(mean_53, 1e-05);  mean_53 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_212 = torch.ops.aten.mul.Tensor(convert_element_type_879, rsqrt_53);  convert_element_type_879 = rsqrt_53 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, wait_tensor_240);  mul_212 = wait_tensor_240 = None
        convert_element_type_880 = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        convert_element_type_881 = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16)
        all_gather_into_tensor_241 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_881, 64, '0');  convert_element_type_881 = None
        wait_tensor_241 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_241);  all_gather_into_tensor_241 = None
        permute_294 = torch.ops.aten.permute.default(wait_tensor_241, [1, 0]);  wait_tensor_241 = None
        view_911 = torch.ops.aten.view.default(convert_element_type_880, [16384, 4096]);  convert_element_type_880 = None
        mm_186 = torch.ops.aten.mm.default(view_911, permute_294);  permute_294 = None
        view_912 = torch.ops.aten.view.default(mm_186, [2, 8192, 14336])
        convert_element_type_884 = torch.ops.prims.convert_element_type.default(view_912, torch.float32);  view_912 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(convert_element_type_884)
        mul_214 = torch.ops.aten.mul.Tensor(convert_element_type_884, sigmoid_26);  convert_element_type_884 = sigmoid_26 = None
        convert_element_type_885 = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        convert_element_type_886 = torch.ops.prims.convert_element_type.default(primals_245, torch.bfloat16)
        all_gather_into_tensor_242 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_886, 64, '0');  convert_element_type_886 = None
        wait_tensor_242 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_242);  all_gather_into_tensor_242 = None
        permute_295 = torch.ops.aten.permute.default(wait_tensor_242, [1, 0]);  wait_tensor_242 = None
        mm_187 = torch.ops.aten.mm.default(view_911, permute_295);  view_911 = permute_295 = None
        view_915 = torch.ops.aten.view.default(mm_187, [2, 8192, 14336]);  mm_187 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_885, view_915);  convert_element_type_885 = view_915 = None
        convert_element_type_889 = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16)
        all_gather_into_tensor_243 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_889, 64, '0');  convert_element_type_889 = None
        wait_tensor_243 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_243);  all_gather_into_tensor_243 = None
        permute_296 = torch.ops.aten.permute.default(wait_tensor_243, [1, 0]);  wait_tensor_243 = None
        view_917 = torch.ops.aten.view.default(mul_215, [16384, 14336]);  mul_215 = None
        mm_188 = torch.ops.aten.mm.default(view_917, permute_296);  view_917 = permute_296 = None
        view_918 = torch.ops.aten.view.default(mm_188, [2, 8192, 4096]);  mm_188 = None
        add_107 = torch.ops.aten.add.Tensor(add_105, view_918);  add_105 = view_918 = None
        convert_element_type_892 = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16)
        all_gather_into_tensor_244 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_892, 64, '0');  convert_element_type_892 = None
        wait_tensor_244 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_244);  all_gather_into_tensor_244 = None
        convert_element_type_893 = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_893, 2)
        mean_54 = torch.ops.aten.mean.dim(pow_55, [2], True);  pow_55 = None
        add_108 = torch.ops.aten.add.Scalar(mean_54, 1e-05);  mean_54 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_216 = torch.ops.aten.mul.Tensor(convert_element_type_893, rsqrt_54);  convert_element_type_893 = rsqrt_54 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, wait_tensor_244);  mul_216 = wait_tensor_244 = None
        convert_element_type_894 = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16);  mul_217 = None
        convert_element_type_895 = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16)
        all_gather_into_tensor_245 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_895, 64, '0');  convert_element_type_895 = None
        wait_tensor_245 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_245);  all_gather_into_tensor_245 = None
        permute_297 = torch.ops.aten.permute.default(wait_tensor_245, [1, 0]);  wait_tensor_245 = None
        view_921 = torch.ops.aten.view.default(convert_element_type_894, [16384, 4096]);  convert_element_type_894 = None
        mm_189 = torch.ops.aten.mm.default(view_921, permute_297);  permute_297 = None
        view_922 = torch.ops.aten.view.default(mm_189, [2, 8192, 4096])
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16)
        all_gather_into_tensor_246 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_898, 64, '0');  convert_element_type_898 = None
        wait_tensor_246 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_246);  all_gather_into_tensor_246 = None
        permute_298 = torch.ops.aten.permute.default(wait_tensor_246, [1, 0]);  wait_tensor_246 = None
        mm_190 = torch.ops.aten.mm.default(view_921, permute_298);  permute_298 = None
        view_925 = torch.ops.aten.view.default(mm_190, [2, 8192, 1024]);  mm_190 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16)
        all_gather_into_tensor_247 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_901, 64, '0');  convert_element_type_901 = None
        wait_tensor_247 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_247);  all_gather_into_tensor_247 = None
        permute_299 = torch.ops.aten.permute.default(wait_tensor_247, [1, 0]);  wait_tensor_247 = None
        mm_191 = torch.ops.aten.mm.default(view_921, permute_299);  view_921 = permute_299 = None
        view_928 = torch.ops.aten.view.default(mm_191, [2, 8192, 1024])
        view_929 = torch.ops.aten.view.default(view_922, [2, 8192, -1, 128]);  view_922 = None
        view_930 = torch.ops.aten.view.default(view_925, [2, 8192, -1, 128]);  view_925 = None
        view_931 = torch.ops.aten.view.default(view_928, [2, 8192, -1, 128]);  view_928 = None
        convert_element_type_904 = torch.ops.prims.convert_element_type.default(view_929, torch.float32);  view_929 = None
        view_932 = torch.ops.aten.view.default(convert_element_type_904, [2, 8192, 32, -1, 2]);  convert_element_type_904 = None
        view_as_complex_54 = torch.ops.aten.view_as_complex.default(view_932);  view_932 = None
        convert_element_type_905 = torch.ops.prims.convert_element_type.default(view_930, torch.float32);  view_930 = None
        view_933 = torch.ops.aten.view.default(convert_element_type_905, [2, 8192, 8, -1, 2]);  convert_element_type_905 = None
        view_as_complex_55 = torch.ops.aten.view_as_complex.default(view_933);  view_933 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_as_complex_54, view_16);  view_as_complex_54 = None
        view_as_real_54 = torch.ops.aten.view_as_real.default(mul_218);  mul_218 = None
        view_935 = torch.ops.aten.view.default(view_as_real_54, [2, 8192, 32, 128]);  view_as_real_54 = None
        mul_219 = torch.ops.aten.mul.Tensor(view_as_complex_55, view_16);  view_as_complex_55 = None
        view_as_real_55 = torch.ops.aten.view_as_real.default(mul_219);  mul_219 = None
        view_936 = torch.ops.aten.view.default(view_as_real_55, [2, 8192, 8, 128]);  view_as_real_55 = None
        convert_element_type_906 = torch.ops.prims.convert_element_type.default(view_935, torch.bfloat16);  view_935 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(view_936, torch.bfloat16);  view_936 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(convert_element_type_907, 3);  convert_element_type_907 = None
        expand_54 = torch.ops.aten.expand.default(unsqueeze_54, [2, 8192, 8, 4, 128]);  unsqueeze_54 = None
        clone_54 = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_937 = torch.ops.aten.view.default(clone_54, [2, 8192, 32, 128]);  clone_54 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(view_931, 3);  view_931 = None
        expand_55 = torch.ops.aten.expand.default(unsqueeze_55, [2, 8192, 8, 4, 128]);  unsqueeze_55 = None
        clone_55 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_938 = torch.ops.aten.view.default(clone_55, [2, 8192, 32, 128]);  clone_55 = None
        permute_300 = torch.ops.aten.permute.default(convert_element_type_906, [0, 2, 1, 3]);  convert_element_type_906 = None
        permute_301 = torch.ops.aten.permute.default(view_937, [0, 2, 1, 3]);  view_937 = None
        permute_302 = torch.ops.aten.permute.default(view_938, [0, 2, 1, 3]);  view_938 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_300, permute_301, permute_302, None, True, 0.0, True);  permute_300 = permute_301 = permute_302 = None
        getitem_243 = _scaled_dot_product_cudnn_attention_27[0]
        getitem_244 = _scaled_dot_product_cudnn_attention_27[1]
        getitem_249 = _scaled_dot_product_cudnn_attention_27[6]
        getitem_250 = _scaled_dot_product_cudnn_attention_27[7];  _scaled_dot_product_cudnn_attention_27 = None
        permute_303 = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3])
        view_939 = torch.ops.aten.view.default(permute_303, [2, 8192, -1]);  permute_303 = None
        convert_element_type_908 = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16)
        all_gather_into_tensor_248 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_908, 64, '0');  convert_element_type_908 = None
        wait_tensor_248 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_248);  all_gather_into_tensor_248 = None
        permute_304 = torch.ops.aten.permute.default(wait_tensor_248, [1, 0]);  wait_tensor_248 = None
        view_941 = torch.ops.aten.view.default(view_939, [16384, 4096]);  view_939 = None
        mm_192 = torch.ops.aten.mm.default(view_941, permute_304);  view_941 = permute_304 = None
        view_942 = torch.ops.aten.view.default(mm_192, [2, 8192, 4096]);  mm_192 = None
        add_109 = torch.ops.aten.add.Tensor(add_107, view_942);  view_942 = None
        convert_element_type_911 = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16)
        all_gather_into_tensor_249 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_911, 64, '0');  convert_element_type_911 = None
        wait_tensor_249 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_249);  all_gather_into_tensor_249 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_912, 2)
        mean_55 = torch.ops.aten.mean.dim(pow_56, [2], True);  pow_56 = None
        add_110 = torch.ops.aten.add.Scalar(mean_55, 1e-05);  mean_55 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_912, rsqrt_55);  convert_element_type_912 = rsqrt_55 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, wait_tensor_249);  mul_220 = wait_tensor_249 = None
        convert_element_type_913 = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        convert_element_type_914 = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16)
        all_gather_into_tensor_250 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_914, 64, '0');  convert_element_type_914 = None
        wait_tensor_250 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_250);  all_gather_into_tensor_250 = None
        permute_305 = torch.ops.aten.permute.default(wait_tensor_250, [1, 0]);  wait_tensor_250 = None
        view_945 = torch.ops.aten.view.default(convert_element_type_913, [16384, 4096]);  convert_element_type_913 = None
        mm_193 = torch.ops.aten.mm.default(view_945, permute_305);  permute_305 = None
        view_946 = torch.ops.aten.view.default(mm_193, [2, 8192, 14336])
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(view_946, torch.float32);  view_946 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(convert_element_type_917)
        mul_222 = torch.ops.aten.mul.Tensor(convert_element_type_917, sigmoid_27);  convert_element_type_917 = sigmoid_27 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None
        convert_element_type_919 = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16)
        all_gather_into_tensor_251 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_919, 64, '0');  convert_element_type_919 = None
        wait_tensor_251 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_251);  all_gather_into_tensor_251 = None
        permute_306 = torch.ops.aten.permute.default(wait_tensor_251, [1, 0]);  wait_tensor_251 = None
        mm_194 = torch.ops.aten.mm.default(view_945, permute_306);  view_945 = permute_306 = None
        view_949 = torch.ops.aten.view.default(mm_194, [2, 8192, 14336]);  mm_194 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_918, view_949);  convert_element_type_918 = view_949 = None
        convert_element_type_922 = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16)
        all_gather_into_tensor_252 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_922, 64, '0');  convert_element_type_922 = None
        wait_tensor_252 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_252);  all_gather_into_tensor_252 = None
        permute_307 = torch.ops.aten.permute.default(wait_tensor_252, [1, 0]);  wait_tensor_252 = None
        view_951 = torch.ops.aten.view.default(mul_223, [16384, 14336]);  mul_223 = None
        mm_195 = torch.ops.aten.mm.default(view_951, permute_307);  view_951 = permute_307 = None
        view_952 = torch.ops.aten.view.default(mm_195, [2, 8192, 4096]);  mm_195 = None
        add_111 = torch.ops.aten.add.Tensor(add_109, view_952);  add_109 = view_952 = None
        convert_element_type_925 = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16)
        all_gather_into_tensor_253 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_925, 64, '0');  convert_element_type_925 = None
        wait_tensor_253 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_253);  all_gather_into_tensor_253 = None
        convert_element_type_926 = torch.ops.prims.convert_element_type.default(add_111, torch.float32)
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_926, 2)
        mean_56 = torch.ops.aten.mean.dim(pow_57, [2], True);  pow_57 = None
        add_112 = torch.ops.aten.add.Scalar(mean_56, 1e-05);  mean_56 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_224 = torch.ops.aten.mul.Tensor(convert_element_type_926, rsqrt_56);  convert_element_type_926 = rsqrt_56 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, wait_tensor_253);  mul_224 = wait_tensor_253 = None
        convert_element_type_927 = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        convert_element_type_928 = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16)
        all_gather_into_tensor_254 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_928, 64, '0');  convert_element_type_928 = None
        wait_tensor_254 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_254);  all_gather_into_tensor_254 = None
        permute_308 = torch.ops.aten.permute.default(wait_tensor_254, [1, 0]);  wait_tensor_254 = None
        view_955 = torch.ops.aten.view.default(convert_element_type_927, [16384, 4096]);  convert_element_type_927 = None
        mm_196 = torch.ops.aten.mm.default(view_955, permute_308);  permute_308 = None
        view_956 = torch.ops.aten.view.default(mm_196, [2, 8192, 4096])
        convert_element_type_931 = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16)
        all_gather_into_tensor_255 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_931, 64, '0');  convert_element_type_931 = None
        wait_tensor_255 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_255);  all_gather_into_tensor_255 = None
        permute_309 = torch.ops.aten.permute.default(wait_tensor_255, [1, 0]);  wait_tensor_255 = None
        mm_197 = torch.ops.aten.mm.default(view_955, permute_309);  permute_309 = None
        view_959 = torch.ops.aten.view.default(mm_197, [2, 8192, 1024]);  mm_197 = None
        convert_element_type_934 = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16)
        all_gather_into_tensor_256 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_934, 64, '0');  convert_element_type_934 = None
        wait_tensor_256 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_256);  all_gather_into_tensor_256 = None
        permute_310 = torch.ops.aten.permute.default(wait_tensor_256, [1, 0]);  wait_tensor_256 = None
        mm_198 = torch.ops.aten.mm.default(view_955, permute_310);  view_955 = permute_310 = None
        view_962 = torch.ops.aten.view.default(mm_198, [2, 8192, 1024])
        view_963 = torch.ops.aten.view.default(view_956, [2, 8192, -1, 128]);  view_956 = None
        view_964 = torch.ops.aten.view.default(view_959, [2, 8192, -1, 128]);  view_959 = None
        view_965 = torch.ops.aten.view.default(view_962, [2, 8192, -1, 128]);  view_962 = None
        convert_element_type_937 = torch.ops.prims.convert_element_type.default(view_963, torch.float32);  view_963 = None
        view_966 = torch.ops.aten.view.default(convert_element_type_937, [2, 8192, 32, -1, 2]);  convert_element_type_937 = None
        view_as_complex_56 = torch.ops.aten.view_as_complex.default(view_966);  view_966 = None
        convert_element_type_938 = torch.ops.prims.convert_element_type.default(view_964, torch.float32);  view_964 = None
        view_967 = torch.ops.aten.view.default(convert_element_type_938, [2, 8192, 8, -1, 2]);  convert_element_type_938 = None
        view_as_complex_57 = torch.ops.aten.view_as_complex.default(view_967);  view_967 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_as_complex_56, view_16);  view_as_complex_56 = None
        view_as_real_56 = torch.ops.aten.view_as_real.default(mul_226);  mul_226 = None
        view_969 = torch.ops.aten.view.default(view_as_real_56, [2, 8192, 32, 128]);  view_as_real_56 = None
        mul_227 = torch.ops.aten.mul.Tensor(view_as_complex_57, view_16);  view_as_complex_57 = None
        view_as_real_57 = torch.ops.aten.view_as_real.default(mul_227);  mul_227 = None
        view_970 = torch.ops.aten.view.default(view_as_real_57, [2, 8192, 8, 128]);  view_as_real_57 = None
        convert_element_type_939 = torch.ops.prims.convert_element_type.default(view_969, torch.bfloat16);  view_969 = None
        convert_element_type_940 = torch.ops.prims.convert_element_type.default(view_970, torch.bfloat16);  view_970 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(convert_element_type_940, 3);  convert_element_type_940 = None
        expand_56 = torch.ops.aten.expand.default(unsqueeze_56, [2, 8192, 8, 4, 128]);  unsqueeze_56 = None
        clone_56 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_971 = torch.ops.aten.view.default(clone_56, [2, 8192, 32, 128]);  clone_56 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(view_965, 3);  view_965 = None
        expand_57 = torch.ops.aten.expand.default(unsqueeze_57, [2, 8192, 8, 4, 128]);  unsqueeze_57 = None
        clone_57 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_972 = torch.ops.aten.view.default(clone_57, [2, 8192, 32, 128]);  clone_57 = None
        permute_311 = torch.ops.aten.permute.default(convert_element_type_939, [0, 2, 1, 3]);  convert_element_type_939 = None
        permute_312 = torch.ops.aten.permute.default(view_971, [0, 2, 1, 3]);  view_971 = None
        permute_313 = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_311, permute_312, permute_313, None, True, 0.0, True);  permute_311 = permute_312 = permute_313 = None
        getitem_252 = _scaled_dot_product_cudnn_attention_28[0]
        getitem_253 = _scaled_dot_product_cudnn_attention_28[1]
        getitem_258 = _scaled_dot_product_cudnn_attention_28[6]
        getitem_259 = _scaled_dot_product_cudnn_attention_28[7];  _scaled_dot_product_cudnn_attention_28 = None
        permute_314 = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3])
        view_973 = torch.ops.aten.view.default(permute_314, [2, 8192, -1]);  permute_314 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16)
        all_gather_into_tensor_257 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_941, 64, '0');  convert_element_type_941 = None
        wait_tensor_257 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_257);  all_gather_into_tensor_257 = None
        permute_315 = torch.ops.aten.permute.default(wait_tensor_257, [1, 0]);  wait_tensor_257 = None
        view_975 = torch.ops.aten.view.default(view_973, [16384, 4096]);  view_973 = None
        mm_199 = torch.ops.aten.mm.default(view_975, permute_315);  view_975 = permute_315 = None
        view_976 = torch.ops.aten.view.default(mm_199, [2, 8192, 4096]);  mm_199 = None
        add_113 = torch.ops.aten.add.Tensor(add_111, view_976);  view_976 = None
        convert_element_type_944 = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16)
        all_gather_into_tensor_258 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_944, 64, '0');  convert_element_type_944 = None
        wait_tensor_258 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_258);  all_gather_into_tensor_258 = None
        convert_element_type_945 = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_945, 2)
        mean_57 = torch.ops.aten.mean.dim(pow_58, [2], True);  pow_58 = None
        add_114 = torch.ops.aten.add.Scalar(mean_57, 1e-05);  mean_57 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_228 = torch.ops.aten.mul.Tensor(convert_element_type_945, rsqrt_57);  convert_element_type_945 = rsqrt_57 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, wait_tensor_258);  mul_228 = wait_tensor_258 = None
        convert_element_type_946 = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16)
        all_gather_into_tensor_259 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_947, 64, '0');  convert_element_type_947 = None
        wait_tensor_259 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_259);  all_gather_into_tensor_259 = None
        permute_316 = torch.ops.aten.permute.default(wait_tensor_259, [1, 0]);  wait_tensor_259 = None
        view_979 = torch.ops.aten.view.default(convert_element_type_946, [16384, 4096]);  convert_element_type_946 = None
        mm_200 = torch.ops.aten.mm.default(view_979, permute_316);  permute_316 = None
        view_980 = torch.ops.aten.view.default(mm_200, [2, 8192, 14336])
        convert_element_type_950 = torch.ops.prims.convert_element_type.default(view_980, torch.float32);  view_980 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(convert_element_type_950)
        mul_230 = torch.ops.aten.mul.Tensor(convert_element_type_950, sigmoid_28);  convert_element_type_950 = sigmoid_28 = None
        convert_element_type_951 = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16);  mul_230 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16)
        all_gather_into_tensor_260 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_952, 64, '0');  convert_element_type_952 = None
        wait_tensor_260 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_260);  all_gather_into_tensor_260 = None
        permute_317 = torch.ops.aten.permute.default(wait_tensor_260, [1, 0]);  wait_tensor_260 = None
        mm_201 = torch.ops.aten.mm.default(view_979, permute_317);  view_979 = permute_317 = None
        view_983 = torch.ops.aten.view.default(mm_201, [2, 8192, 14336]);  mm_201 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_951, view_983);  convert_element_type_951 = view_983 = None
        convert_element_type_955 = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16)
        all_gather_into_tensor_261 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_955, 64, '0');  convert_element_type_955 = None
        wait_tensor_261 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_261);  all_gather_into_tensor_261 = None
        permute_318 = torch.ops.aten.permute.default(wait_tensor_261, [1, 0]);  wait_tensor_261 = None
        view_985 = torch.ops.aten.view.default(mul_231, [16384, 14336]);  mul_231 = None
        mm_202 = torch.ops.aten.mm.default(view_985, permute_318);  view_985 = permute_318 = None
        view_986 = torch.ops.aten.view.default(mm_202, [2, 8192, 4096]);  mm_202 = None
        add_115 = torch.ops.aten.add.Tensor(add_113, view_986);  add_113 = view_986 = None
        convert_element_type_958 = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16)
        all_gather_into_tensor_262 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_958, 64, '0');  convert_element_type_958 = None
        wait_tensor_262 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_262);  all_gather_into_tensor_262 = None
        convert_element_type_959 = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        pow_59 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_959, 2)
        mean_58 = torch.ops.aten.mean.dim(pow_59, [2], True);  pow_59 = None
        add_116 = torch.ops.aten.add.Scalar(mean_58, 1e-05);  mean_58 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_959, rsqrt_58);  convert_element_type_959 = rsqrt_58 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, wait_tensor_262);  mul_232 = wait_tensor_262 = None
        convert_element_type_960 = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None
        convert_element_type_961 = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16)
        all_gather_into_tensor_263 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_961, 64, '0');  convert_element_type_961 = None
        wait_tensor_263 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_263);  all_gather_into_tensor_263 = None
        permute_319 = torch.ops.aten.permute.default(wait_tensor_263, [1, 0]);  wait_tensor_263 = None
        view_989 = torch.ops.aten.view.default(convert_element_type_960, [16384, 4096]);  convert_element_type_960 = None
        mm_203 = torch.ops.aten.mm.default(view_989, permute_319);  permute_319 = None
        view_990 = torch.ops.aten.view.default(mm_203, [2, 8192, 4096])
        convert_element_type_964 = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16)
        all_gather_into_tensor_264 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_964, 64, '0');  convert_element_type_964 = None
        wait_tensor_264 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_264);  all_gather_into_tensor_264 = None
        permute_320 = torch.ops.aten.permute.default(wait_tensor_264, [1, 0]);  wait_tensor_264 = None
        mm_204 = torch.ops.aten.mm.default(view_989, permute_320);  permute_320 = None
        view_993 = torch.ops.aten.view.default(mm_204, [2, 8192, 1024]);  mm_204 = None
        convert_element_type_967 = torch.ops.prims.convert_element_type.default(primals_268, torch.bfloat16)
        all_gather_into_tensor_265 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_967, 64, '0');  convert_element_type_967 = None
        wait_tensor_265 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_265);  all_gather_into_tensor_265 = None
        permute_321 = torch.ops.aten.permute.default(wait_tensor_265, [1, 0]);  wait_tensor_265 = None
        mm_205 = torch.ops.aten.mm.default(view_989, permute_321);  view_989 = permute_321 = None
        view_996 = torch.ops.aten.view.default(mm_205, [2, 8192, 1024])
        view_997 = torch.ops.aten.view.default(view_990, [2, 8192, -1, 128]);  view_990 = None
        view_998 = torch.ops.aten.view.default(view_993, [2, 8192, -1, 128]);  view_993 = None
        view_999 = torch.ops.aten.view.default(view_996, [2, 8192, -1, 128]);  view_996 = None
        convert_element_type_970 = torch.ops.prims.convert_element_type.default(view_997, torch.float32);  view_997 = None
        view_1000 = torch.ops.aten.view.default(convert_element_type_970, [2, 8192, 32, -1, 2]);  convert_element_type_970 = None
        view_as_complex_58 = torch.ops.aten.view_as_complex.default(view_1000);  view_1000 = None
        convert_element_type_971 = torch.ops.prims.convert_element_type.default(view_998, torch.float32);  view_998 = None
        view_1001 = torch.ops.aten.view.default(convert_element_type_971, [2, 8192, 8, -1, 2]);  convert_element_type_971 = None
        view_as_complex_59 = torch.ops.aten.view_as_complex.default(view_1001);  view_1001 = None
        mul_234 = torch.ops.aten.mul.Tensor(view_as_complex_58, view_16);  view_as_complex_58 = None
        view_as_real_58 = torch.ops.aten.view_as_real.default(mul_234);  mul_234 = None
        view_1003 = torch.ops.aten.view.default(view_as_real_58, [2, 8192, 32, 128]);  view_as_real_58 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_as_complex_59, view_16);  view_as_complex_59 = None
        view_as_real_59 = torch.ops.aten.view_as_real.default(mul_235);  mul_235 = None
        view_1004 = torch.ops.aten.view.default(view_as_real_59, [2, 8192, 8, 128]);  view_as_real_59 = None
        convert_element_type_972 = torch.ops.prims.convert_element_type.default(view_1003, torch.bfloat16);  view_1003 = None
        convert_element_type_973 = torch.ops.prims.convert_element_type.default(view_1004, torch.bfloat16);  view_1004 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(convert_element_type_973, 3);  convert_element_type_973 = None
        expand_58 = torch.ops.aten.expand.default(unsqueeze_58, [2, 8192, 8, 4, 128]);  unsqueeze_58 = None
        clone_58 = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_1005 = torch.ops.aten.view.default(clone_58, [2, 8192, 32, 128]);  clone_58 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(view_999, 3);  view_999 = None
        expand_59 = torch.ops.aten.expand.default(unsqueeze_59, [2, 8192, 8, 4, 128]);  unsqueeze_59 = None
        clone_59 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_1006 = torch.ops.aten.view.default(clone_59, [2, 8192, 32, 128]);  clone_59 = None
        permute_322 = torch.ops.aten.permute.default(convert_element_type_972, [0, 2, 1, 3]);  convert_element_type_972 = None
        permute_323 = torch.ops.aten.permute.default(view_1005, [0, 2, 1, 3]);  view_1005 = None
        permute_324 = torch.ops.aten.permute.default(view_1006, [0, 2, 1, 3]);  view_1006 = None
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_322, permute_323, permute_324, None, True, 0.0, True);  permute_322 = permute_323 = permute_324 = None
        getitem_261 = _scaled_dot_product_cudnn_attention_29[0]
        getitem_262 = _scaled_dot_product_cudnn_attention_29[1]
        getitem_267 = _scaled_dot_product_cudnn_attention_29[6]
        getitem_268 = _scaled_dot_product_cudnn_attention_29[7];  _scaled_dot_product_cudnn_attention_29 = None
        permute_325 = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3])
        view_1007 = torch.ops.aten.view.default(permute_325, [2, 8192, -1]);  permute_325 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16)
        all_gather_into_tensor_266 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_974, 64, '0');  convert_element_type_974 = None
        wait_tensor_266 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_266);  all_gather_into_tensor_266 = None
        permute_326 = torch.ops.aten.permute.default(wait_tensor_266, [1, 0]);  wait_tensor_266 = None
        view_1009 = torch.ops.aten.view.default(view_1007, [16384, 4096]);  view_1007 = None
        mm_206 = torch.ops.aten.mm.default(view_1009, permute_326);  view_1009 = permute_326 = None
        view_1010 = torch.ops.aten.view.default(mm_206, [2, 8192, 4096]);  mm_206 = None
        add_117 = torch.ops.aten.add.Tensor(add_115, view_1010);  view_1010 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16)
        all_gather_into_tensor_267 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_977, 64, '0');  convert_element_type_977 = None
        wait_tensor_267 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_267);  all_gather_into_tensor_267 = None
        convert_element_type_978 = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        pow_60 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_978, 2)
        mean_59 = torch.ops.aten.mean.dim(pow_60, [2], True);  pow_60 = None
        add_118 = torch.ops.aten.add.Scalar(mean_59, 1e-05);  mean_59 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_236 = torch.ops.aten.mul.Tensor(convert_element_type_978, rsqrt_59);  convert_element_type_978 = rsqrt_59 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, wait_tensor_267);  mul_236 = wait_tensor_267 = None
        convert_element_type_979 = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        convert_element_type_980 = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16)
        all_gather_into_tensor_268 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_980, 64, '0');  convert_element_type_980 = None
        wait_tensor_268 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_268);  all_gather_into_tensor_268 = None
        permute_327 = torch.ops.aten.permute.default(wait_tensor_268, [1, 0]);  wait_tensor_268 = None
        view_1013 = torch.ops.aten.view.default(convert_element_type_979, [16384, 4096]);  convert_element_type_979 = None
        mm_207 = torch.ops.aten.mm.default(view_1013, permute_327);  permute_327 = None
        view_1014 = torch.ops.aten.view.default(mm_207, [2, 8192, 14336])
        convert_element_type_983 = torch.ops.prims.convert_element_type.default(view_1014, torch.float32);  view_1014 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(convert_element_type_983)
        mul_238 = torch.ops.aten.mul.Tensor(convert_element_type_983, sigmoid_29);  convert_element_type_983 = sigmoid_29 = None
        convert_element_type_984 = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None
        convert_element_type_985 = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16)
        all_gather_into_tensor_269 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_985, 64, '0');  convert_element_type_985 = None
        wait_tensor_269 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_269);  all_gather_into_tensor_269 = None
        permute_328 = torch.ops.aten.permute.default(wait_tensor_269, [1, 0]);  wait_tensor_269 = None
        mm_208 = torch.ops.aten.mm.default(view_1013, permute_328);  view_1013 = permute_328 = None
        view_1017 = torch.ops.aten.view.default(mm_208, [2, 8192, 14336]);  mm_208 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_984, view_1017);  convert_element_type_984 = view_1017 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16)
        all_gather_into_tensor_270 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_988, 64, '0');  convert_element_type_988 = None
        wait_tensor_270 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_270);  all_gather_into_tensor_270 = None
        permute_329 = torch.ops.aten.permute.default(wait_tensor_270, [1, 0]);  wait_tensor_270 = None
        view_1019 = torch.ops.aten.view.default(mul_239, [16384, 14336]);  mul_239 = None
        mm_209 = torch.ops.aten.mm.default(view_1019, permute_329);  view_1019 = permute_329 = None
        view_1020 = torch.ops.aten.view.default(mm_209, [2, 8192, 4096]);  mm_209 = None
        add_119 = torch.ops.aten.add.Tensor(add_117, view_1020);  add_117 = view_1020 = None
        convert_element_type_991 = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16)
        all_gather_into_tensor_271 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_991, 64, '0');  convert_element_type_991 = None
        wait_tensor_271 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_271);  all_gather_into_tensor_271 = None
        convert_element_type_992 = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        pow_61 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_992, 2)
        mean_60 = torch.ops.aten.mean.dim(pow_61, [2], True);  pow_61 = None
        add_120 = torch.ops.aten.add.Scalar(mean_60, 1e-05);  mean_60 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_992, rsqrt_60);  convert_element_type_992 = rsqrt_60 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, wait_tensor_271);  mul_240 = wait_tensor_271 = None
        convert_element_type_993 = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        convert_element_type_994 = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16)
        all_gather_into_tensor_272 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_994, 64, '0');  convert_element_type_994 = None
        wait_tensor_272 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_272);  all_gather_into_tensor_272 = None
        permute_330 = torch.ops.aten.permute.default(wait_tensor_272, [1, 0]);  wait_tensor_272 = None
        view_1023 = torch.ops.aten.view.default(convert_element_type_993, [16384, 4096]);  convert_element_type_993 = None
        mm_210 = torch.ops.aten.mm.default(view_1023, permute_330);  permute_330 = None
        view_1024 = torch.ops.aten.view.default(mm_210, [2, 8192, 4096])
        convert_element_type_997 = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16)
        all_gather_into_tensor_273 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_997, 64, '0');  convert_element_type_997 = None
        wait_tensor_273 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_273);  all_gather_into_tensor_273 = None
        permute_331 = torch.ops.aten.permute.default(wait_tensor_273, [1, 0]);  wait_tensor_273 = None
        mm_211 = torch.ops.aten.mm.default(view_1023, permute_331);  permute_331 = None
        view_1027 = torch.ops.aten.view.default(mm_211, [2, 8192, 1024]);  mm_211 = None
        convert_element_type_1000 = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16)
        all_gather_into_tensor_274 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1000, 64, '0');  convert_element_type_1000 = None
        wait_tensor_274 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_274);  all_gather_into_tensor_274 = None
        permute_332 = torch.ops.aten.permute.default(wait_tensor_274, [1, 0]);  wait_tensor_274 = None
        mm_212 = torch.ops.aten.mm.default(view_1023, permute_332);  view_1023 = permute_332 = None
        view_1030 = torch.ops.aten.view.default(mm_212, [2, 8192, 1024])
        view_1031 = torch.ops.aten.view.default(view_1024, [2, 8192, -1, 128]);  view_1024 = None
        view_1032 = torch.ops.aten.view.default(view_1027, [2, 8192, -1, 128]);  view_1027 = None
        view_1033 = torch.ops.aten.view.default(view_1030, [2, 8192, -1, 128]);  view_1030 = None
        convert_element_type_1003 = torch.ops.prims.convert_element_type.default(view_1031, torch.float32);  view_1031 = None
        view_1034 = torch.ops.aten.view.default(convert_element_type_1003, [2, 8192, 32, -1, 2]);  convert_element_type_1003 = None
        view_as_complex_60 = torch.ops.aten.view_as_complex.default(view_1034);  view_1034 = None
        convert_element_type_1004 = torch.ops.prims.convert_element_type.default(view_1032, torch.float32);  view_1032 = None
        view_1035 = torch.ops.aten.view.default(convert_element_type_1004, [2, 8192, 8, -1, 2]);  convert_element_type_1004 = None
        view_as_complex_61 = torch.ops.aten.view_as_complex.default(view_1035);  view_1035 = None
        mul_242 = torch.ops.aten.mul.Tensor(view_as_complex_60, view_16);  view_as_complex_60 = None
        view_as_real_60 = torch.ops.aten.view_as_real.default(mul_242);  mul_242 = None
        view_1037 = torch.ops.aten.view.default(view_as_real_60, [2, 8192, 32, 128]);  view_as_real_60 = None
        mul_243 = torch.ops.aten.mul.Tensor(view_as_complex_61, view_16);  view_as_complex_61 = None
        view_as_real_61 = torch.ops.aten.view_as_real.default(mul_243);  mul_243 = None
        view_1038 = torch.ops.aten.view.default(view_as_real_61, [2, 8192, 8, 128]);  view_as_real_61 = None
        convert_element_type_1005 = torch.ops.prims.convert_element_type.default(view_1037, torch.bfloat16);  view_1037 = None
        convert_element_type_1006 = torch.ops.prims.convert_element_type.default(view_1038, torch.bfloat16);  view_1038 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(convert_element_type_1006, 3);  convert_element_type_1006 = None
        expand_60 = torch.ops.aten.expand.default(unsqueeze_60, [2, 8192, 8, 4, 128]);  unsqueeze_60 = None
        clone_60 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_1039 = torch.ops.aten.view.default(clone_60, [2, 8192, 32, 128]);  clone_60 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(view_1033, 3);  view_1033 = None
        expand_61 = torch.ops.aten.expand.default(unsqueeze_61, [2, 8192, 8, 4, 128]);  unsqueeze_61 = None
        clone_61 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_1040 = torch.ops.aten.view.default(clone_61, [2, 8192, 32, 128]);  clone_61 = None
        permute_333 = torch.ops.aten.permute.default(convert_element_type_1005, [0, 2, 1, 3]);  convert_element_type_1005 = None
        permute_334 = torch.ops.aten.permute.default(view_1039, [0, 2, 1, 3]);  view_1039 = None
        permute_335 = torch.ops.aten.permute.default(view_1040, [0, 2, 1, 3]);  view_1040 = None
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_333, permute_334, permute_335, None, True, 0.0, True);  permute_333 = permute_334 = permute_335 = None
        getitem_270 = _scaled_dot_product_cudnn_attention_30[0]
        getitem_271 = _scaled_dot_product_cudnn_attention_30[1]
        getitem_276 = _scaled_dot_product_cudnn_attention_30[6]
        getitem_277 = _scaled_dot_product_cudnn_attention_30[7];  _scaled_dot_product_cudnn_attention_30 = None
        permute_336 = torch.ops.aten.permute.default(getitem_270, [0, 2, 1, 3])
        view_1041 = torch.ops.aten.view.default(permute_336, [2, 8192, -1]);  permute_336 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16)
        all_gather_into_tensor_275 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1007, 64, '0');  convert_element_type_1007 = None
        wait_tensor_275 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_275);  all_gather_into_tensor_275 = None
        permute_337 = torch.ops.aten.permute.default(wait_tensor_275, [1, 0]);  wait_tensor_275 = None
        view_1043 = torch.ops.aten.view.default(view_1041, [16384, 4096]);  view_1041 = None
        mm_213 = torch.ops.aten.mm.default(view_1043, permute_337);  view_1043 = permute_337 = None
        view_1044 = torch.ops.aten.view.default(mm_213, [2, 8192, 4096]);  mm_213 = None
        add_121 = torch.ops.aten.add.Tensor(add_119, view_1044);  view_1044 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16)
        all_gather_into_tensor_276 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1010, 64, '0');  convert_element_type_1010 = None
        wait_tensor_276 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_276);  all_gather_into_tensor_276 = None
        convert_element_type_1011 = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        pow_62 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1011, 2)
        mean_61 = torch.ops.aten.mean.dim(pow_62, [2], True);  pow_62 = None
        add_122 = torch.ops.aten.add.Scalar(mean_61, 1e-05);  mean_61 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_244 = torch.ops.aten.mul.Tensor(convert_element_type_1011, rsqrt_61);  convert_element_type_1011 = rsqrt_61 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, wait_tensor_276);  mul_244 = wait_tensor_276 = None
        convert_element_type_1012 = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16);  mul_245 = None
        convert_element_type_1013 = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16)
        all_gather_into_tensor_277 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1013, 64, '0');  convert_element_type_1013 = None
        wait_tensor_277 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_277);  all_gather_into_tensor_277 = None
        permute_338 = torch.ops.aten.permute.default(wait_tensor_277, [1, 0]);  wait_tensor_277 = None
        view_1047 = torch.ops.aten.view.default(convert_element_type_1012, [16384, 4096]);  convert_element_type_1012 = None
        mm_214 = torch.ops.aten.mm.default(view_1047, permute_338);  permute_338 = None
        view_1048 = torch.ops.aten.view.default(mm_214, [2, 8192, 14336])
        convert_element_type_1016 = torch.ops.prims.convert_element_type.default(view_1048, torch.float32);  view_1048 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(convert_element_type_1016)
        mul_246 = torch.ops.aten.mul.Tensor(convert_element_type_1016, sigmoid_30);  convert_element_type_1016 = sigmoid_30 = None
        convert_element_type_1017 = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        convert_element_type_1018 = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16)
        all_gather_into_tensor_278 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1018, 64, '0');  convert_element_type_1018 = None
        wait_tensor_278 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_278);  all_gather_into_tensor_278 = None
        permute_339 = torch.ops.aten.permute.default(wait_tensor_278, [1, 0]);  wait_tensor_278 = None
        mm_215 = torch.ops.aten.mm.default(view_1047, permute_339);  view_1047 = permute_339 = None
        view_1051 = torch.ops.aten.view.default(mm_215, [2, 8192, 14336]);  mm_215 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_1017, view_1051);  convert_element_type_1017 = view_1051 = None
        convert_element_type_1021 = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16)
        all_gather_into_tensor_279 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1021, 64, '0');  convert_element_type_1021 = None
        wait_tensor_279 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_279);  all_gather_into_tensor_279 = None
        permute_340 = torch.ops.aten.permute.default(wait_tensor_279, [1, 0]);  wait_tensor_279 = None
        view_1053 = torch.ops.aten.view.default(mul_247, [16384, 14336]);  mul_247 = None
        mm_216 = torch.ops.aten.mm.default(view_1053, permute_340);  view_1053 = permute_340 = None
        view_1054 = torch.ops.aten.view.default(mm_216, [2, 8192, 4096]);  mm_216 = None
        add_123 = torch.ops.aten.add.Tensor(add_121, view_1054);  add_121 = view_1054 = None
        convert_element_type_1024 = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16)
        all_gather_into_tensor_280 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1024, 64, '0');  convert_element_type_1024 = None
        wait_tensor_280 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_280);  all_gather_into_tensor_280 = None
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        pow_63 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1025, 2)
        mean_62 = torch.ops.aten.mean.dim(pow_63, [2], True);  pow_63 = None
        add_124 = torch.ops.aten.add.Scalar(mean_62, 1e-05);  mean_62 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_1025, rsqrt_62);  convert_element_type_1025 = rsqrt_62 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, wait_tensor_280);  mul_248 = wait_tensor_280 = None
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        convert_element_type_1027 = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16)
        all_gather_into_tensor_281 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1027, 64, '0');  convert_element_type_1027 = None
        wait_tensor_281 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_281);  all_gather_into_tensor_281 = None
        permute_341 = torch.ops.aten.permute.default(wait_tensor_281, [1, 0]);  wait_tensor_281 = None
        view_1057 = torch.ops.aten.view.default(convert_element_type_1026, [16384, 4096]);  convert_element_type_1026 = None
        mm_217 = torch.ops.aten.mm.default(view_1057, permute_341);  permute_341 = None
        view_1058 = torch.ops.aten.view.default(mm_217, [2, 8192, 4096])
        convert_element_type_1030 = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16)
        all_gather_into_tensor_282 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1030, 64, '0');  convert_element_type_1030 = None
        wait_tensor_282 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_282);  all_gather_into_tensor_282 = None
        permute_342 = torch.ops.aten.permute.default(wait_tensor_282, [1, 0]);  wait_tensor_282 = None
        mm_218 = torch.ops.aten.mm.default(view_1057, permute_342);  permute_342 = None
        view_1061 = torch.ops.aten.view.default(mm_218, [2, 8192, 1024]);  mm_218 = None
        convert_element_type_1033 = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16)
        all_gather_into_tensor_283 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1033, 64, '0');  convert_element_type_1033 = None
        wait_tensor_283 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_283);  all_gather_into_tensor_283 = None
        permute_343 = torch.ops.aten.permute.default(wait_tensor_283, [1, 0]);  wait_tensor_283 = None
        mm_219 = torch.ops.aten.mm.default(view_1057, permute_343);  view_1057 = permute_343 = None
        view_1064 = torch.ops.aten.view.default(mm_219, [2, 8192, 1024])
        view_1065 = torch.ops.aten.view.default(view_1058, [2, 8192, -1, 128]);  view_1058 = None
        view_1066 = torch.ops.aten.view.default(view_1061, [2, 8192, -1, 128]);  view_1061 = None
        view_1067 = torch.ops.aten.view.default(view_1064, [2, 8192, -1, 128]);  view_1064 = None
        convert_element_type_1036 = torch.ops.prims.convert_element_type.default(view_1065, torch.float32);  view_1065 = None
        view_1068 = torch.ops.aten.view.default(convert_element_type_1036, [2, 8192, 32, -1, 2]);  convert_element_type_1036 = None
        view_as_complex_62 = torch.ops.aten.view_as_complex.default(view_1068);  view_1068 = None
        convert_element_type_1037 = torch.ops.prims.convert_element_type.default(view_1066, torch.float32);  view_1066 = None
        view_1069 = torch.ops.aten.view.default(convert_element_type_1037, [2, 8192, 8, -1, 2]);  convert_element_type_1037 = None
        view_as_complex_63 = torch.ops.aten.view_as_complex.default(view_1069);  view_1069 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_as_complex_62, view_16);  view_as_complex_62 = None
        view_as_real_62 = torch.ops.aten.view_as_real.default(mul_250);  mul_250 = None
        view_1071 = torch.ops.aten.view.default(view_as_real_62, [2, 8192, 32, 128]);  view_as_real_62 = None
        mul_251 = torch.ops.aten.mul.Tensor(view_as_complex_63, view_16);  view_as_complex_63 = view_16 = None
        view_as_real_63 = torch.ops.aten.view_as_real.default(mul_251);  mul_251 = None
        view_1072 = torch.ops.aten.view.default(view_as_real_63, [2, 8192, 8, 128]);  view_as_real_63 = None
        convert_element_type_1038 = torch.ops.prims.convert_element_type.default(view_1071, torch.bfloat16);  view_1071 = None
        convert_element_type_1039 = torch.ops.prims.convert_element_type.default(view_1072, torch.bfloat16);  view_1072 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(convert_element_type_1039, 3);  convert_element_type_1039 = None
        expand_62 = torch.ops.aten.expand.default(unsqueeze_62, [2, 8192, 8, 4, 128]);  unsqueeze_62 = None
        clone_62 = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_1073 = torch.ops.aten.view.default(clone_62, [2, 8192, 32, 128]);  clone_62 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(view_1067, 3);  view_1067 = None
        expand_63 = torch.ops.aten.expand.default(unsqueeze_63, [2, 8192, 8, 4, 128]);  unsqueeze_63 = None
        clone_63 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_1074 = torch.ops.aten.view.default(clone_63, [2, 8192, 32, 128]);  clone_63 = None
        permute_344 = torch.ops.aten.permute.default(convert_element_type_1038, [0, 2, 1, 3]);  convert_element_type_1038 = None
        permute_345 = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None
        permute_346 = torch.ops.aten.permute.default(view_1074, [0, 2, 1, 3]);  view_1074 = None
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_344, permute_345, permute_346, None, True, 0.0, True);  permute_344 = permute_345 = permute_346 = None
        getitem_279 = _scaled_dot_product_cudnn_attention_31[0]
        getitem_280 = _scaled_dot_product_cudnn_attention_31[1]
        getitem_285 = _scaled_dot_product_cudnn_attention_31[6]
        getitem_286 = _scaled_dot_product_cudnn_attention_31[7];  _scaled_dot_product_cudnn_attention_31 = None
        permute_347 = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3])
        view_1075 = torch.ops.aten.view.default(permute_347, [2, 8192, -1]);  permute_347 = None
        convert_element_type_1040 = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16)
        all_gather_into_tensor_284 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1040, 64, '0');  convert_element_type_1040 = None
        wait_tensor_284 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_284);  all_gather_into_tensor_284 = None
        permute_348 = torch.ops.aten.permute.default(wait_tensor_284, [1, 0]);  wait_tensor_284 = None
        view_1077 = torch.ops.aten.view.default(view_1075, [16384, 4096]);  view_1075 = None
        mm_220 = torch.ops.aten.mm.default(view_1077, permute_348);  view_1077 = permute_348 = None
        view_1078 = torch.ops.aten.view.default(mm_220, [2, 8192, 4096]);  mm_220 = None
        add_125 = torch.ops.aten.add.Tensor(add_123, view_1078);  view_1078 = None
        convert_element_type_1043 = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16)
        all_gather_into_tensor_285 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1043, 64, '0');  convert_element_type_1043 = None
        wait_tensor_285 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_285);  all_gather_into_tensor_285 = None
        convert_element_type_1044 = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_64 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1044, 2)
        mean_63 = torch.ops.aten.mean.dim(pow_64, [2], True);  pow_64 = None
        add_126 = torch.ops.aten.add.Scalar(mean_63, 1e-05);  mean_63 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_252 = torch.ops.aten.mul.Tensor(convert_element_type_1044, rsqrt_63);  convert_element_type_1044 = rsqrt_63 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, wait_tensor_285);  mul_252 = wait_tensor_285 = None
        convert_element_type_1045 = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None
        convert_element_type_1046 = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16)
        all_gather_into_tensor_286 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1046, 64, '0');  convert_element_type_1046 = None
        wait_tensor_286 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_286);  all_gather_into_tensor_286 = None
        permute_349 = torch.ops.aten.permute.default(wait_tensor_286, [1, 0]);  wait_tensor_286 = None
        view_1081 = torch.ops.aten.view.default(convert_element_type_1045, [16384, 4096]);  convert_element_type_1045 = None
        mm_221 = torch.ops.aten.mm.default(view_1081, permute_349);  permute_349 = None
        view_1082 = torch.ops.aten.view.default(mm_221, [2, 8192, 14336])
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(convert_element_type_1049)
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_1049, sigmoid_31);  convert_element_type_1049 = sigmoid_31 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        convert_element_type_1051 = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16)
        all_gather_into_tensor_287 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1051, 64, '0');  convert_element_type_1051 = None
        wait_tensor_287 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_287);  all_gather_into_tensor_287 = None
        permute_350 = torch.ops.aten.permute.default(wait_tensor_287, [1, 0]);  wait_tensor_287 = None
        mm_222 = torch.ops.aten.mm.default(view_1081, permute_350);  view_1081 = permute_350 = None
        view_1085 = torch.ops.aten.view.default(mm_222, [2, 8192, 14336]);  mm_222 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_1050, view_1085);  convert_element_type_1050 = view_1085 = None
        convert_element_type_1054 = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16)
        all_gather_into_tensor_288 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1054, 64, '0');  convert_element_type_1054 = None
        wait_tensor_288 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_288);  all_gather_into_tensor_288 = None
        permute_351 = torch.ops.aten.permute.default(wait_tensor_288, [1, 0]);  wait_tensor_288 = None
        view_1087 = torch.ops.aten.view.default(mul_255, [16384, 14336]);  mul_255 = None
        mm_223 = torch.ops.aten.mm.default(view_1087, permute_351);  view_1087 = permute_351 = None
        view_1088 = torch.ops.aten.view.default(mm_223, [2, 8192, 4096])
        add_127 = torch.ops.aten.add.Tensor(add_125, view_1088);  add_125 = view_1088 = None
        convert_element_type_1057 = torch.ops.prims.convert_element_type.default(primals_292, torch.bfloat16)
        all_gather_into_tensor_289 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1057, 64, '0');  convert_element_type_1057 = None
        wait_tensor_289 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_289);  all_gather_into_tensor_289 = None
        convert_element_type_1058 = torch.ops.prims.convert_element_type.default(add_127, torch.float32);  add_127 = None
        pow_65 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1058, 2)
        mean_64 = torch.ops.aten.mean.dim(pow_65, [2], True);  pow_65 = None
        add_128 = torch.ops.aten.add.Scalar(mean_64, 1e-05);  mean_64 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_256 = torch.ops.aten.mul.Tensor(convert_element_type_1058, rsqrt_64);  convert_element_type_1058 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, wait_tensor_289);  mul_256 = wait_tensor_289 = None
        convert_element_type_1059 = torch.ops.prims.convert_element_type.default(mul_257, torch.bfloat16);  mul_257 = None
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16)
        all_gather_into_tensor_290 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1060, 64, '0');  convert_element_type_1060 = None
        wait_tensor_290 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_290);  all_gather_into_tensor_290 = None
        permute_352 = torch.ops.aten.permute.default(wait_tensor_290, [1, 0]);  wait_tensor_290 = None
        view_1091 = torch.ops.aten.view.default(convert_element_type_1059, [16384, 4096]);  convert_element_type_1059 = None
        mm_224 = torch.ops.aten.mm.default(view_1091, permute_352);  permute_352 = None
        view_1092 = torch.ops.aten.view.default(mm_224, [2, 8192, 128256]);  mm_224 = None
        return (view_1092, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, embedding, mm, mm_2, getitem, getitem_1, getitem_6, getitem_7, mm_4, add_3, mm_7, mm_9, getitem_9, getitem_10, getitem_15, getitem_16, mm_11, add_7, mm_14, mm_16, getitem_18, getitem_19, getitem_24, getitem_25, mm_18, add_11, mm_21, mm_23, getitem_27, getitem_28, getitem_33, getitem_34, mm_25, add_15, mm_28, mm_30, getitem_36, getitem_37, getitem_42, getitem_43, mm_32, add_19, mm_35, mm_37, getitem_45, getitem_46, getitem_51, getitem_52, mm_39, add_23, mm_42, mm_44, getitem_54, getitem_55, getitem_60, getitem_61, mm_46, add_27, mm_49, mm_51, getitem_63, getitem_64, getitem_69, getitem_70, mm_53, add_31, mm_56, mm_58, getitem_72, getitem_73, getitem_78, getitem_79, mm_60, add_35, mm_63, mm_65, getitem_81, getitem_82, getitem_87, getitem_88, mm_67, add_39, mm_70, mm_72, getitem_90, getitem_91, getitem_96, getitem_97, mm_74, add_43, mm_77, mm_79, getitem_99, getitem_100, getitem_105, getitem_106, mm_81, add_47, mm_84, mm_86, getitem_108, getitem_109, getitem_114, getitem_115, mm_88, add_51, mm_91, mm_93, getitem_117, getitem_118, getitem_123, getitem_124, mm_95, add_55, mm_98, mm_100, getitem_126, getitem_127, getitem_132, getitem_133, mm_102, add_59, mm_105, mm_107, getitem_135, getitem_136, getitem_141, getitem_142, mm_109, add_63, mm_112, mm_114, getitem_144, getitem_145, getitem_150, getitem_151, mm_116, add_67, mm_119, mm_121, getitem_153, getitem_154, getitem_159, getitem_160, mm_123, add_71, mm_126, mm_128, getitem_162, getitem_163, getitem_168, getitem_169, mm_130, add_75, mm_133, mm_135, getitem_171, getitem_172, getitem_177, getitem_178, mm_137, add_79, mm_140, mm_142, getitem_180, getitem_181, getitem_186, getitem_187, mm_144, add_83, mm_147, mm_149, getitem_189, getitem_190, getitem_195, getitem_196, mm_151, add_87, mm_154, mm_156, getitem_198, getitem_199, getitem_204, getitem_205, mm_158, add_91, mm_161, mm_163, getitem_207, getitem_208, getitem_213, getitem_214, mm_165, add_95, mm_168, mm_170, getitem_216, getitem_217, getitem_222, getitem_223, mm_172, add_99, mm_175, mm_177, getitem_225, getitem_226, getitem_231, getitem_232, mm_179, add_103, mm_182, mm_184, getitem_234, getitem_235, getitem_240, getitem_241, mm_186, add_107, mm_189, mm_191, getitem_243, getitem_244, getitem_249, getitem_250, mm_193, add_111, mm_196, mm_198, getitem_252, getitem_253, getitem_258, getitem_259, mm_200, add_115, mm_203, mm_205, getitem_261, getitem_262, getitem_267, getitem_268, mm_207, add_119, mm_210, mm_212, getitem_270, getitem_271, getitem_276, getitem_277, mm_214, add_123, mm_217, mm_219, getitem_279, getitem_280, getitem_285, getitem_286, mm_221, mm_223, rsqrt_64, view_1091)
        
def load_args(reader):
    buf0 = reader.storage(None, 32833536, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2004, 4096), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf2, (8192, 64), dtype=torch.complex64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64, 4096), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16, 4096), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf6, (16, 4096), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 4096), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf9, (224, 4096), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf10, (224, 4096), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 14336), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64, 4096), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16, 4096), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16, 4096), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64, 4096), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf18, (224, 4096), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf19, (224, 4096), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64, 14336), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64, 4096), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16, 4096), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (16, 4096), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64, 4096), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf27, (224, 4096), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf28, (224, 4096), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64, 14336), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 4096), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf32, (16, 4096), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf33, (16, 4096), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64, 4096), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf36, (224, 4096), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf37, (224, 4096), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64, 14336), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64, 4096), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (16, 4096), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf42, (16, 4096), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64, 4096), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf45, (224, 4096), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf46, (224, 4096), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64, 14336), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64, 4096), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf50, (16, 4096), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf51, (16, 4096), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64, 4096), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf54, (224, 4096), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf55, (224, 4096), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64, 14336), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf57, (64,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf58, (64, 4096), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf59, (16, 4096), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (16, 4096), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (64, 4096), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf62, (64,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf63, (224, 4096), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf64, (224, 4096), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf65, (64, 14336), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf66, (64,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (64, 4096), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf68, (16, 4096), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (16, 4096), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (64, 4096), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf71, (64,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf72, (224, 4096), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf73, (224, 4096), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf74, (64, 14336), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf75, (64,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf76, (64, 4096), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf77, (16, 4096), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf78, (16, 4096), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf79, (64, 4096), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf80, (64,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf81, (224, 4096), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf82, (224, 4096), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf83, (64, 14336), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf84, (64,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (64, 4096), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf86, (16, 4096), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (16, 4096), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf88, (64, 4096), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf89, (64,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf90, (224, 4096), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf91, (224, 4096), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf92, (64, 14336), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf93, (64,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf94, (64, 4096), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf95, (16, 4096), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (16, 4096), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf97, (64, 4096), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf98, (64,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf99, (224, 4096), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf100, (224, 4096), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf101, (64, 14336), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf102, (64,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64, 4096), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf104, (16, 4096), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf105, (16, 4096), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf106, (64, 4096), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf107, (64,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf108, (224, 4096), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf109, (224, 4096), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf110, (64, 14336), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf111, (64,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf112, (64, 4096), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf113, (16, 4096), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf114, (16, 4096), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf115, (64, 4096), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf116, (64,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf117, (224, 4096), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf118, (224, 4096), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf119, (64, 14336), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf120, (64,), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf121, (64, 4096), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf122, (16, 4096), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf123, (16, 4096), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf124, (64, 4096), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf125, (64,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf126, (224, 4096), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf127, (224, 4096), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf128, (64, 14336), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf129, (64,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf130, (64, 4096), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf131, (16, 4096), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (16, 4096), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64, 4096), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf134, (64,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf135, (224, 4096), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf136, (224, 4096), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf137, (64, 14336), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf138, (64,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf139, (64, 4096), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (16, 4096), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (16, 4096), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf142, (64, 4096), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf143, (64,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf144, (224, 4096), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf145, (224, 4096), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf146, (64, 14336), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf147, (64,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf148, (64, 4096), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf149, (16, 4096), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf150, (16, 4096), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf151, (64, 4096), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf152, (64,), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf153, (224, 4096), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf154, (224, 4096), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf155, (64, 14336), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf156, (64,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf157, (64, 4096), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf158, (16, 4096), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf159, (16, 4096), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf160, (64, 4096), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf161, (64,), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf162, (224, 4096), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf163, (224, 4096), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf164, (64, 14336), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf165, (64,), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf166, (64, 4096), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf167, (16, 4096), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf168, (16, 4096), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf169, (64, 4096), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf170, (64,), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf171, (224, 4096), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf172, (224, 4096), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf173, (64, 14336), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf174, (64,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf175, (64, 4096), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf176, (16, 4096), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf177, (16, 4096), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf178, (64, 4096), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf179, (64,), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf180, (224, 4096), is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf181, (224, 4096), is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf182, (64, 14336), is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf183, (64,), is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf184, (64, 4096), is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf185, (16, 4096), is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf186, (16, 4096), is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf187, (64, 4096), is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf188, (64,), is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf189, (224, 4096), is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf190, (224, 4096), is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf191, (64, 14336), is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf192, (64,), is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf193, (64, 4096), is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf194, (16, 4096), is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf195, (16, 4096), is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf196, (64, 4096), is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf197, (64,), is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf198, (224, 4096), is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf199, (224, 4096), is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf200, (64, 14336), is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf201, (64,), is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf202, (64, 4096), is_leaf=True)  # primals_203
    buf203 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf203, (16, 4096), is_leaf=True)  # primals_204
    buf204 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf204, (16, 4096), is_leaf=True)  # primals_205
    buf205 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf205, (64, 4096), is_leaf=True)  # primals_206
    buf206 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf206, (64,), is_leaf=True)  # primals_207
    buf207 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf207, (224, 4096), is_leaf=True)  # primals_208
    buf208 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf208, (224, 4096), is_leaf=True)  # primals_209
    buf209 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf209, (64, 14336), is_leaf=True)  # primals_210
    buf210 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf210, (64,), is_leaf=True)  # primals_211
    buf211 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf211, (64, 4096), is_leaf=True)  # primals_212
    buf212 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf212, (16, 4096), is_leaf=True)  # primals_213
    buf213 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf213, (16, 4096), is_leaf=True)  # primals_214
    buf214 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf214, (64, 4096), is_leaf=True)  # primals_215
    buf215 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf215, (64,), is_leaf=True)  # primals_216
    buf216 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf216, (224, 4096), is_leaf=True)  # primals_217
    buf217 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf217, (224, 4096), is_leaf=True)  # primals_218
    buf218 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf218, (64, 14336), is_leaf=True)  # primals_219
    buf219 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf219, (64,), is_leaf=True)  # primals_220
    buf220 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf220, (64, 4096), is_leaf=True)  # primals_221
    buf221 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf221, (16, 4096), is_leaf=True)  # primals_222
    buf222 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf222, (16, 4096), is_leaf=True)  # primals_223
    buf223 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf223, (64, 4096), is_leaf=True)  # primals_224
    buf224 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf224, (64,), is_leaf=True)  # primals_225
    buf225 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf225, (224, 4096), is_leaf=True)  # primals_226
    buf226 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf226, (224, 4096), is_leaf=True)  # primals_227
    buf227 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf227, (64, 14336), is_leaf=True)  # primals_228
    buf228 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf228, (64,), is_leaf=True)  # primals_229
    buf229 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf229, (64, 4096), is_leaf=True)  # primals_230
    buf230 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf230, (16, 4096), is_leaf=True)  # primals_231
    buf231 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf231, (16, 4096), is_leaf=True)  # primals_232
    buf232 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf232, (64, 4096), is_leaf=True)  # primals_233
    buf233 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf233, (64,), is_leaf=True)  # primals_234
    buf234 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf234, (224, 4096), is_leaf=True)  # primals_235
    buf235 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf235, (224, 4096), is_leaf=True)  # primals_236
    buf236 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf236, (64, 14336), is_leaf=True)  # primals_237
    buf237 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf237, (64,), is_leaf=True)  # primals_238
    buf238 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf238, (64, 4096), is_leaf=True)  # primals_239
    buf239 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf239, (16, 4096), is_leaf=True)  # primals_240
    buf240 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf240, (16, 4096), is_leaf=True)  # primals_241
    buf241 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf241, (64, 4096), is_leaf=True)  # primals_242
    buf242 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf242, (64,), is_leaf=True)  # primals_243
    buf243 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf243, (224, 4096), is_leaf=True)  # primals_244
    buf244 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf244, (224, 4096), is_leaf=True)  # primals_245
    buf245 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf245, (64, 14336), is_leaf=True)  # primals_246
    buf246 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf246, (64,), is_leaf=True)  # primals_247
    buf247 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf247, (64, 4096), is_leaf=True)  # primals_248
    buf248 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf248, (16, 4096), is_leaf=True)  # primals_249
    buf249 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf249, (16, 4096), is_leaf=True)  # primals_250
    buf250 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf250, (64, 4096), is_leaf=True)  # primals_251
    buf251 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf251, (64,), is_leaf=True)  # primals_252
    buf252 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf252, (224, 4096), is_leaf=True)  # primals_253
    buf253 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf253, (224, 4096), is_leaf=True)  # primals_254
    buf254 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf254, (64, 14336), is_leaf=True)  # primals_255
    buf255 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf255, (64,), is_leaf=True)  # primals_256
    buf256 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf256, (64, 4096), is_leaf=True)  # primals_257
    buf257 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf257, (16, 4096), is_leaf=True)  # primals_258
    buf258 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf258, (16, 4096), is_leaf=True)  # primals_259
    buf259 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf259, (64, 4096), is_leaf=True)  # primals_260
    buf260 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf260, (64,), is_leaf=True)  # primals_261
    buf261 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf261, (224, 4096), is_leaf=True)  # primals_262
    buf262 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf262, (224, 4096), is_leaf=True)  # primals_263
    buf263 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf263, (64, 14336), is_leaf=True)  # primals_264
    buf264 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf264, (64,), is_leaf=True)  # primals_265
    buf265 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf265, (64, 4096), is_leaf=True)  # primals_266
    buf266 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf266, (16, 4096), is_leaf=True)  # primals_267
    buf267 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf267, (16, 4096), is_leaf=True)  # primals_268
    buf268 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf268, (64, 4096), is_leaf=True)  # primals_269
    buf269 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf269, (64,), is_leaf=True)  # primals_270
    buf270 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf270, (224, 4096), is_leaf=True)  # primals_271
    buf271 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf271, (224, 4096), is_leaf=True)  # primals_272
    buf272 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf272, (64, 14336), is_leaf=True)  # primals_273
    buf273 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf273, (64,), is_leaf=True)  # primals_274
    buf274 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf274, (64, 4096), is_leaf=True)  # primals_275
    buf275 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf275, (16, 4096), is_leaf=True)  # primals_276
    buf276 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf276, (16, 4096), is_leaf=True)  # primals_277
    buf277 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf277, (64, 4096), is_leaf=True)  # primals_278
    buf278 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf278, (64,), is_leaf=True)  # primals_279
    buf279 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf279, (224, 4096), is_leaf=True)  # primals_280
    buf280 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf280, (224, 4096), is_leaf=True)  # primals_281
    buf281 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf281, (64, 14336), is_leaf=True)  # primals_282
    buf282 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf282, (64,), is_leaf=True)  # primals_283
    buf283 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf283, (64, 4096), is_leaf=True)  # primals_284
    buf284 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf284, (16, 4096), is_leaf=True)  # primals_285
    buf285 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf285, (16, 4096), is_leaf=True)  # primals_286
    buf286 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf286, (64, 4096), is_leaf=True)  # primals_287
    buf287 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf287, (64,), is_leaf=True)  # primals_288
    buf288 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf288, (224, 4096), is_leaf=True)  # primals_289
    buf289 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf289, (224, 4096), is_leaf=True)  # primals_290
    buf290 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf290, (64, 14336), is_leaf=True)  # primals_291
    buf291 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf291, (64,), is_leaf=True)  # primals_292
    buf292 = reader.storage(None, 32833536, device=device(type='cuda', index=0))
    reader.tensor(buf292, (2004, 4096), is_leaf=True)  # primals_293

load_args._version = 0

def get_pg_config():
    return {'0': {'size': 64, 'rank': 0}}

def get_colls_estimations_file():
    return "colls8_8.table"
