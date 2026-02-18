import torch
from torch.nn import *
from torch import tensor, device


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16)
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type, 32, '0');  convert_element_type = None
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        lt = torch.ops.aten.lt.Scalar(primals_1, 0)
        ge = torch.ops.aten.ge.Scalar(primals_1, 16032)
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        sub = torch.ops.aten.sub.Tensor(primals_1, 0)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(sub, [bitwise_or], full_default);  sub = full_default = None
        embedding = torch.ops.aten.embedding.default(wait_tensor, index_put);  wait_tensor = index_put = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(embedding, [bitwise_or], full_default_1);  embedding = bitwise_or = full_default_1 = None
        split_1 = torch.ops.aten.split.Tensor(index_put_1, 1024, 1);  index_put_1 = None
        getitem_8 = split_1[0]
        getitem_17 = split_1[1]
        getitem_26 = split_1[2]
        getitem_35 = split_1[3]
        getitem_44 = split_1[4]
        getitem_53 = split_1[5]
        getitem_62 = split_1[6]
        getitem_71 = split_1[7];  split_1 = None
        cat = torch.ops.aten.cat.default([getitem_8, getitem_17, getitem_26, getitem_35, getitem_44, getitem_53, getitem_62, getitem_71]);  getitem_8 = getitem_17 = getitem_26 = getitem_35 = getitem_44 = getitem_53 = getitem_62 = getitem_71 = None
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat, 'sum', 8, '1');  cat = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16)
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 32, '0');  convert_element_type_1 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(wait_tensor_1, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, wait_tensor_2);  mul = wait_tensor_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_3, 8, '1');  convert_element_type_3 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        split_9 = torch.ops.aten.split.Tensor(wait_tensor_3, 2);  wait_tensor_3 = None
        getitem_72 = split_9[0]
        getitem_73 = split_9[1]
        getitem_74 = split_9[2]
        getitem_75 = split_9[3]
        getitem_76 = split_9[4]
        getitem_77 = split_9[5]
        getitem_78 = split_9[6]
        getitem_79 = split_9[7];  split_9 = None
        cat_1 = torch.ops.aten.cat.default([getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79], 1);  getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16)
        all_gather_into_tensor_3 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_4, 32, '0');  convert_element_type_4 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        permute = torch.ops.aten.permute.default(wait_tensor_4, [1, 0]);  wait_tensor_4 = None
        view_15 = torch.ops.aten.view.default(cat_1, [16384, 4096]);  cat_1 = None
        mm = torch.ops.aten.mm.default(view_15, permute);  permute = None
        view_16 = torch.ops.aten.view.default(mm, [2, 8192, 512])
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16)
        all_gather_into_tensor_4 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_7, 32, '0');  convert_element_type_7 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        permute_1 = torch.ops.aten.permute.default(wait_tensor_5, [1, 0]);  wait_tensor_5 = None
        mm_1 = torch.ops.aten.mm.default(view_15, permute_1);  permute_1 = None
        view_23 = torch.ops.aten.view.default(mm_1, [2, 8192, 128]);  mm_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16)
        all_gather_into_tensor_5 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_10, 32, '0');  convert_element_type_10 = None
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        permute_2 = torch.ops.aten.permute.default(wait_tensor_6, [1, 0]);  wait_tensor_6 = None
        mm_2 = torch.ops.aten.mm.default(view_15, permute_2);  view_15 = permute_2 = None
        view_30 = torch.ops.aten.view.default(mm_2, [2, 8192, 128])
        view_32 = torch.ops.aten.view.default(view_16, [2, 8192, -1, 128]);  view_16 = None
        view_33 = torch.ops.aten.view.default(view_23, [2, 8192, -1, 128]);  view_23 = None
        view_34 = torch.ops.aten.view.default(view_30, [2, 8192, -1, 128]);  view_30 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        view_35 = torch.ops.aten.view.default(convert_element_type_13, [2, 8192, 4, -1, 2]);  convert_element_type_13 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_35);  view_35 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_14, [2, 8192, 1, -1, 2]);  convert_element_type_14 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_36);  view_36 = None
        view_37 = torch.ops.aten.view.default(primals_3, [1, 8192, 1, 64])
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_37);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_38 = torch.ops.aten.view.default(view_as_real, [2, 8192, 4, 128]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_37);  view_as_complex_1 = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
        view_39 = torch.ops.aten.view.default(view_as_real_1, [2, 8192, 1, 128]);  view_as_real_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(view_39, torch.bfloat16);  view_39 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_16, 3);  convert_element_type_16 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [2, 8192, 1, 4, 128]);  unsqueeze = None
        view_40 = torch.ops.aten.view.default(expand, [2, 8192, 4, 128]);  expand = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_34, 3);  view_34 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [2, 8192, 1, 4, 128]);  unsqueeze_1 = None
        view_41 = torch.ops.aten.view.default(expand_1, [2, 8192, 4, 128]);  expand_1 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1, 3]);  convert_element_type_15 = None
        permute_4 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        permute_5 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_3, permute_4, permute_5, None, True, 0.0, True);  permute_3 = permute_4 = permute_5 = None
        getitem_80 = _scaled_dot_product_cudnn_attention[0]
        getitem_81 = _scaled_dot_product_cudnn_attention[1]
        getitem_86 = _scaled_dot_product_cudnn_attention[6]
        getitem_87 = _scaled_dot_product_cudnn_attention[7];  _scaled_dot_product_cudnn_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3])
        view_42 = torch.ops.aten.view.default(permute_6, [2, 8192, -1]);  permute_6 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16)
        all_gather_into_tensor_6 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_17, 32, '0');  convert_element_type_17 = None
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        permute_7 = torch.ops.aten.permute.default(wait_tensor_7, [1, 0]);  wait_tensor_7 = None
        view_48 = torch.ops.aten.view.default(view_42, [16384, 512]);  view_42 = None
        mm_3 = torch.ops.aten.mm.default(view_48, permute_7);  view_48 = permute_7 = None
        view_49 = torch.ops.aten.view.default(mm_3, [2, 8192, 4096]);  mm_3 = None
        split_10 = torch.ops.aten.split.Tensor(view_49, 1024, 1);  view_49 = None
        getitem_89 = split_10[0]
        getitem_90 = split_10[1]
        getitem_91 = split_10[2]
        getitem_92 = split_10[3]
        getitem_93 = split_10[4]
        getitem_94 = split_10[5]
        getitem_95 = split_10[6]
        getitem_96 = split_10[7];  split_10 = None
        cat_2 = torch.ops.aten.cat.default([getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96]);  getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        reduce_scatter_tensor_1 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_2, 'sum', 8, '1');  cat_2 = None
        wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1)
        add_1 = torch.ops.aten.add.Tensor(wait_tensor_1, wait_tensor_8);  wait_tensor_8 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16)
        all_gather_into_tensor_7 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_20, 32, '0');  convert_element_type_20 = None
        wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Scalar(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_1);  convert_element_type_21 = rsqrt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, wait_tensor_9);  mul_4 = wait_tensor_9 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        all_gather_into_tensor_8 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_22, 8, '1');  convert_element_type_22 = None
        wait_tensor_10 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        split_11 = torch.ops.aten.split.Tensor(wait_tensor_10, 2);  wait_tensor_10 = None
        getitem_97 = split_11[0]
        getitem_98 = split_11[1]
        getitem_99 = split_11[2]
        getitem_100 = split_11[3]
        getitem_101 = split_11[4]
        getitem_102 = split_11[5]
        getitem_103 = split_11[6]
        getitem_104 = split_11[7];  split_11 = None
        cat_3 = torch.ops.aten.cat.default([getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104], 1);  getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16)
        all_gather_into_tensor_9 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_23, 32, '0');  convert_element_type_23 = None
        wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        permute_8 = torch.ops.aten.permute.default(wait_tensor_11, [1, 0]);  wait_tensor_11 = None
        view_60 = torch.ops.aten.view.default(cat_3, [16384, 4096]);  cat_3 = None
        mm_4 = torch.ops.aten.mm.default(view_60, permute_8);  permute_8 = None
        view_61 = torch.ops.aten.view.default(mm_4, [2, 8192, 1792])
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_26)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_26, sigmoid);  convert_element_type_26 = sigmoid = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16)
        all_gather_into_tensor_10 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_28, 32, '0');  convert_element_type_28 = None
        wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        permute_9 = torch.ops.aten.permute.default(wait_tensor_12, [1, 0]);  wait_tensor_12 = None
        mm_5 = torch.ops.aten.mm.default(view_60, permute_9);  view_60 = permute_9 = None
        view_68 = torch.ops.aten.view.default(mm_5, [2, 8192, 1792]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_27, view_68);  convert_element_type_27 = view_68 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16)
        all_gather_into_tensor_11 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_31, 32, '0');  convert_element_type_31 = None
        wait_tensor_13 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        permute_10 = torch.ops.aten.permute.default(wait_tensor_13, [1, 0]);  wait_tensor_13 = None
        view_75 = torch.ops.aten.view.default(mul_7, [16384, 1792]);  mul_7 = None
        mm_6 = torch.ops.aten.mm.default(view_75, permute_10);  view_75 = permute_10 = None
        view_76 = torch.ops.aten.view.default(mm_6, [2, 8192, 4096]);  mm_6 = None
        split_12 = torch.ops.aten.split.Tensor(view_76, 1024, 1);  view_76 = None
        getitem_105 = split_12[0]
        getitem_106 = split_12[1]
        getitem_107 = split_12[2]
        getitem_108 = split_12[3]
        getitem_109 = split_12[4]
        getitem_110 = split_12[5]
        getitem_111 = split_12[6]
        getitem_112 = split_12[7];  split_12 = None
        cat_4 = torch.ops.aten.cat.default([getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112]);  getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = None
        reduce_scatter_tensor_2 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_4, 'sum', 8, '1');  cat_4 = None
        wait_tensor_14 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        add_3 = torch.ops.aten.add.Tensor(add_1, wait_tensor_14);  add_1 = wait_tensor_14 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16)
        all_gather_into_tensor_12 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_34, 32, '0');  convert_element_type_34 = None
        wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_4 = torch.ops.aten.add.Scalar(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_2);  convert_element_type_35 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, wait_tensor_15);  mul_8 = wait_tensor_15 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        all_gather_into_tensor_13 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_36, 8, '1');  convert_element_type_36 = None
        wait_tensor_16 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_13);  all_gather_into_tensor_13 = None
        split_13 = torch.ops.aten.split.Tensor(wait_tensor_16, 2);  wait_tensor_16 = None
        getitem_113 = split_13[0]
        getitem_114 = split_13[1]
        getitem_115 = split_13[2]
        getitem_116 = split_13[3]
        getitem_117 = split_13[4]
        getitem_118 = split_13[5]
        getitem_119 = split_13[6]
        getitem_120 = split_13[7];  split_13 = None
        cat_5 = torch.ops.aten.cat.default([getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120], 1);  getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16)
        all_gather_into_tensor_14 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_37, 32, '0');  convert_element_type_37 = None
        wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        permute_11 = torch.ops.aten.permute.default(wait_tensor_17, [1, 0]);  wait_tensor_17 = None
        view_87 = torch.ops.aten.view.default(cat_5, [16384, 4096]);  cat_5 = None
        mm_7 = torch.ops.aten.mm.default(view_87, permute_11);  permute_11 = None
        view_88 = torch.ops.aten.view.default(mm_7, [2, 8192, 512])
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        all_gather_into_tensor_15 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_40, 32, '0');  convert_element_type_40 = None
        wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        permute_12 = torch.ops.aten.permute.default(wait_tensor_18, [1, 0]);  wait_tensor_18 = None
        mm_8 = torch.ops.aten.mm.default(view_87, permute_12);  permute_12 = None
        view_95 = torch.ops.aten.view.default(mm_8, [2, 8192, 128]);  mm_8 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16)
        all_gather_into_tensor_16 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_43, 32, '0');  convert_element_type_43 = None
        wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        permute_13 = torch.ops.aten.permute.default(wait_tensor_19, [1, 0]);  wait_tensor_19 = None
        mm_9 = torch.ops.aten.mm.default(view_87, permute_13);  view_87 = permute_13 = None
        view_102 = torch.ops.aten.view.default(mm_9, [2, 8192, 128])
        view_104 = torch.ops.aten.view.default(view_88, [2, 8192, -1, 128]);  view_88 = None
        view_105 = torch.ops.aten.view.default(view_95, [2, 8192, -1, 128]);  view_95 = None
        view_106 = torch.ops.aten.view.default(view_102, [2, 8192, -1, 128]);  view_102 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        view_107 = torch.ops.aten.view.default(convert_element_type_46, [2, 8192, 4, -1, 2]);  convert_element_type_46 = None
        view_as_complex_2 = torch.ops.aten.view_as_complex.default(view_107);  view_107 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_47, [2, 8192, 1, -1, 2]);  convert_element_type_47 = None
        view_as_complex_3 = torch.ops.aten.view_as_complex.default(view_108);  view_108 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_as_complex_2, view_37);  view_as_complex_2 = None
        view_as_real_2 = torch.ops.aten.view_as_real.default(mul_10);  mul_10 = None
        view_110 = torch.ops.aten.view.default(view_as_real_2, [2, 8192, 4, 128]);  view_as_real_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_as_complex_3, view_37);  view_as_complex_3 = None
        view_as_real_3 = torch.ops.aten.view_as_real.default(mul_11);  mul_11 = None
        view_111 = torch.ops.aten.view.default(view_as_real_3, [2, 8192, 1, 128]);  view_as_real_3 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(view_110, torch.bfloat16);  view_110 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(view_111, torch.bfloat16);  view_111 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_49, 3);  convert_element_type_49 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_2, [2, 8192, 1, 4, 128]);  unsqueeze_2 = None
        view_112 = torch.ops.aten.view.default(expand_2, [2, 8192, 4, 128]);  expand_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_106, 3);  view_106 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_3, [2, 8192, 1, 4, 128]);  unsqueeze_3 = None
        view_113 = torch.ops.aten.view.default(expand_3, [2, 8192, 4, 128]);  expand_3 = None
        permute_14 = torch.ops.aten.permute.default(convert_element_type_48, [0, 2, 1, 3]);  convert_element_type_48 = None
        permute_15 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        permute_16 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_14, permute_15, permute_16, None, True, 0.0, True);  permute_14 = permute_15 = permute_16 = None
        getitem_121 = _scaled_dot_product_cudnn_attention_1[0]
        getitem_122 = _scaled_dot_product_cudnn_attention_1[1]
        getitem_127 = _scaled_dot_product_cudnn_attention_1[6]
        getitem_128 = _scaled_dot_product_cudnn_attention_1[7];  _scaled_dot_product_cudnn_attention_1 = None
        permute_17 = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3])
        view_114 = torch.ops.aten.view.default(permute_17, [2, 8192, -1]);  permute_17 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16)
        all_gather_into_tensor_17 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_50, 32, '0');  convert_element_type_50 = None
        wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_17);  all_gather_into_tensor_17 = None
        permute_18 = torch.ops.aten.permute.default(wait_tensor_20, [1, 0]);  wait_tensor_20 = None
        view_120 = torch.ops.aten.view.default(view_114, [16384, 512]);  view_114 = None
        mm_10 = torch.ops.aten.mm.default(view_120, permute_18);  view_120 = permute_18 = None
        view_121 = torch.ops.aten.view.default(mm_10, [2, 8192, 4096]);  mm_10 = None
        split_14 = torch.ops.aten.split.Tensor(view_121, 1024, 1);  view_121 = None
        getitem_130 = split_14[0]
        getitem_131 = split_14[1]
        getitem_132 = split_14[2]
        getitem_133 = split_14[3]
        getitem_134 = split_14[4]
        getitem_135 = split_14[5]
        getitem_136 = split_14[6]
        getitem_137 = split_14[7];  split_14 = None
        cat_6 = torch.ops.aten.cat.default([getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137]);  getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = None
        reduce_scatter_tensor_3 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_6, 'sum', 8, '1');  cat_6 = None
        wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3)
        add_5 = torch.ops.aten.add.Tensor(add_3, wait_tensor_21);  wait_tensor_21 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16)
        all_gather_into_tensor_18 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_53, 32, '0');  convert_element_type_53 = None
        wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_18);  all_gather_into_tensor_18 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_5, torch.float32)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_6 = torch.ops.aten.add.Scalar(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_3);  convert_element_type_54 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, wait_tensor_22);  mul_12 = wait_tensor_22 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        all_gather_into_tensor_19 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_55, 8, '1');  convert_element_type_55 = None
        wait_tensor_23 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_19);  all_gather_into_tensor_19 = None
        split_15 = torch.ops.aten.split.Tensor(wait_tensor_23, 2);  wait_tensor_23 = None
        getitem_138 = split_15[0]
        getitem_139 = split_15[1]
        getitem_140 = split_15[2]
        getitem_141 = split_15[3]
        getitem_142 = split_15[4]
        getitem_143 = split_15[5]
        getitem_144 = split_15[6]
        getitem_145 = split_15[7];  split_15 = None
        cat_7 = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145], 1);  getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16)
        all_gather_into_tensor_20 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_56, 32, '0');  convert_element_type_56 = None
        wait_tensor_24 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_20);  all_gather_into_tensor_20 = None
        permute_19 = torch.ops.aten.permute.default(wait_tensor_24, [1, 0]);  wait_tensor_24 = None
        view_132 = torch.ops.aten.view.default(cat_7, [16384, 4096]);  cat_7 = None
        mm_11 = torch.ops.aten.mm.default(view_132, permute_19);  permute_19 = None
        view_133 = torch.ops.aten.view.default(mm_11, [2, 8192, 1792])
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_59, sigmoid_1);  convert_element_type_59 = sigmoid_1 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16)
        all_gather_into_tensor_21 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_61, 32, '0');  convert_element_type_61 = None
        wait_tensor_25 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_21);  all_gather_into_tensor_21 = None
        permute_20 = torch.ops.aten.permute.default(wait_tensor_25, [1, 0]);  wait_tensor_25 = None
        mm_12 = torch.ops.aten.mm.default(view_132, permute_20);  view_132 = permute_20 = None
        view_140 = torch.ops.aten.view.default(mm_12, [2, 8192, 1792]);  mm_12 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_60, view_140);  convert_element_type_60 = view_140 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16)
        all_gather_into_tensor_22 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_64, 32, '0');  convert_element_type_64 = None
        wait_tensor_26 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_22);  all_gather_into_tensor_22 = None
        permute_21 = torch.ops.aten.permute.default(wait_tensor_26, [1, 0]);  wait_tensor_26 = None
        view_147 = torch.ops.aten.view.default(mul_15, [16384, 1792]);  mul_15 = None
        mm_13 = torch.ops.aten.mm.default(view_147, permute_21);  view_147 = permute_21 = None
        view_148 = torch.ops.aten.view.default(mm_13, [2, 8192, 4096]);  mm_13 = None
        split_16 = torch.ops.aten.split.Tensor(view_148, 1024, 1);  view_148 = None
        getitem_146 = split_16[0]
        getitem_147 = split_16[1]
        getitem_148 = split_16[2]
        getitem_149 = split_16[3]
        getitem_150 = split_16[4]
        getitem_151 = split_16[5]
        getitem_152 = split_16[6]
        getitem_153 = split_16[7];  split_16 = None
        cat_8 = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153]);  getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = None
        reduce_scatter_tensor_4 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_8, 'sum', 8, '1');  cat_8 = None
        wait_tensor_27 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, wait_tensor_27);  add_5 = wait_tensor_27 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16)
        all_gather_into_tensor_23 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_67, 32, '0');  convert_element_type_67 = None
        wait_tensor_28 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_23);  all_gather_into_tensor_23 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_68, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_8 = torch.ops.aten.add.Scalar(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_68, rsqrt_4);  convert_element_type_68 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, wait_tensor_28);  mul_16 = wait_tensor_28 = None
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        all_gather_into_tensor_24 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_69, 8, '1');  convert_element_type_69 = None
        wait_tensor_29 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_24);  all_gather_into_tensor_24 = None
        split_17 = torch.ops.aten.split.Tensor(wait_tensor_29, 2);  wait_tensor_29 = None
        getitem_154 = split_17[0]
        getitem_155 = split_17[1]
        getitem_156 = split_17[2]
        getitem_157 = split_17[3]
        getitem_158 = split_17[4]
        getitem_159 = split_17[5]
        getitem_160 = split_17[6]
        getitem_161 = split_17[7];  split_17 = None
        cat_9 = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161], 1);  getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16)
        all_gather_into_tensor_25 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_70, 32, '0');  convert_element_type_70 = None
        wait_tensor_30 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_25);  all_gather_into_tensor_25 = None
        permute_22 = torch.ops.aten.permute.default(wait_tensor_30, [1, 0]);  wait_tensor_30 = None
        view_159 = torch.ops.aten.view.default(cat_9, [16384, 4096]);  cat_9 = None
        mm_14 = torch.ops.aten.mm.default(view_159, permute_22);  permute_22 = None
        view_160 = torch.ops.aten.view.default(mm_14, [2, 8192, 512])
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16)
        all_gather_into_tensor_26 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_73, 32, '0');  convert_element_type_73 = None
        wait_tensor_31 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_26);  all_gather_into_tensor_26 = None
        permute_23 = torch.ops.aten.permute.default(wait_tensor_31, [1, 0]);  wait_tensor_31 = None
        mm_15 = torch.ops.aten.mm.default(view_159, permute_23);  permute_23 = None
        view_167 = torch.ops.aten.view.default(mm_15, [2, 8192, 128]);  mm_15 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16)
        all_gather_into_tensor_27 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_76, 32, '0');  convert_element_type_76 = None
        wait_tensor_32 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_27);  all_gather_into_tensor_27 = None
        permute_24 = torch.ops.aten.permute.default(wait_tensor_32, [1, 0]);  wait_tensor_32 = None
        mm_16 = torch.ops.aten.mm.default(view_159, permute_24);  view_159 = permute_24 = None
        view_174 = torch.ops.aten.view.default(mm_16, [2, 8192, 128])
        view_176 = torch.ops.aten.view.default(view_160, [2, 8192, -1, 128]);  view_160 = None
        view_177 = torch.ops.aten.view.default(view_167, [2, 8192, -1, 128]);  view_167 = None
        view_178 = torch.ops.aten.view.default(view_174, [2, 8192, -1, 128]);  view_174 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(view_176, torch.float32);  view_176 = None
        view_179 = torch.ops.aten.view.default(convert_element_type_79, [2, 8192, 4, -1, 2]);  convert_element_type_79 = None
        view_as_complex_4 = torch.ops.aten.view_as_complex.default(view_179);  view_179 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        view_180 = torch.ops.aten.view.default(convert_element_type_80, [2, 8192, 1, -1, 2]);  convert_element_type_80 = None
        view_as_complex_5 = torch.ops.aten.view_as_complex.default(view_180);  view_180 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_as_complex_4, view_37);  view_as_complex_4 = None
        view_as_real_4 = torch.ops.aten.view_as_real.default(mul_18);  mul_18 = None
        view_182 = torch.ops.aten.view.default(view_as_real_4, [2, 8192, 4, 128]);  view_as_real_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_as_complex_5, view_37);  view_as_complex_5 = None
        view_as_real_5 = torch.ops.aten.view_as_real.default(mul_19);  mul_19 = None
        view_183 = torch.ops.aten.view.default(view_as_real_5, [2, 8192, 1, 128]);  view_as_real_5 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(view_182, torch.bfloat16);  view_182 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(view_183, torch.bfloat16);  view_183 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(convert_element_type_82, 3);  convert_element_type_82 = None
        expand_4 = torch.ops.aten.expand.default(unsqueeze_4, [2, 8192, 1, 4, 128]);  unsqueeze_4 = None
        view_184 = torch.ops.aten.view.default(expand_4, [2, 8192, 4, 128]);  expand_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(view_178, 3);  view_178 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_5, [2, 8192, 1, 4, 128]);  unsqueeze_5 = None
        view_185 = torch.ops.aten.view.default(expand_5, [2, 8192, 4, 128]);  expand_5 = None
        permute_25 = torch.ops.aten.permute.default(convert_element_type_81, [0, 2, 1, 3]);  convert_element_type_81 = None
        permute_26 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        permute_27 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_25, permute_26, permute_27, None, True, 0.0, True);  permute_25 = permute_26 = permute_27 = None
        getitem_162 = _scaled_dot_product_cudnn_attention_2[0]
        getitem_163 = _scaled_dot_product_cudnn_attention_2[1]
        getitem_168 = _scaled_dot_product_cudnn_attention_2[6]
        getitem_169 = _scaled_dot_product_cudnn_attention_2[7];  _scaled_dot_product_cudnn_attention_2 = None
        permute_28 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3])
        view_186 = torch.ops.aten.view.default(permute_28, [2, 8192, -1]);  permute_28 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16)
        all_gather_into_tensor_28 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_83, 32, '0');  convert_element_type_83 = None
        wait_tensor_33 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_28);  all_gather_into_tensor_28 = None
        permute_29 = torch.ops.aten.permute.default(wait_tensor_33, [1, 0]);  wait_tensor_33 = None
        view_192 = torch.ops.aten.view.default(view_186, [16384, 512]);  view_186 = None
        mm_17 = torch.ops.aten.mm.default(view_192, permute_29);  view_192 = permute_29 = None
        view_193 = torch.ops.aten.view.default(mm_17, [2, 8192, 4096]);  mm_17 = None
        split_18 = torch.ops.aten.split.Tensor(view_193, 1024, 1);  view_193 = None
        getitem_171 = split_18[0]
        getitem_172 = split_18[1]
        getitem_173 = split_18[2]
        getitem_174 = split_18[3]
        getitem_175 = split_18[4]
        getitem_176 = split_18[5]
        getitem_177 = split_18[6]
        getitem_178 = split_18[7];  split_18 = None
        cat_10 = torch.ops.aten.cat.default([getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178]);  getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = None
        reduce_scatter_tensor_5 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_10, 'sum', 8, '1');  cat_10 = None
        wait_tensor_34 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_5)
        add_9 = torch.ops.aten.add.Tensor(add_7, wait_tensor_34);  wait_tensor_34 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16)
        all_gather_into_tensor_29 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_86, 32, '0');  convert_element_type_86 = None
        wait_tensor_35 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_29);  all_gather_into_tensor_29 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_9, torch.float32)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_10 = torch.ops.aten.add.Scalar(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_5);  convert_element_type_87 = rsqrt_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, wait_tensor_35);  mul_20 = wait_tensor_35 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        all_gather_into_tensor_30 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_88, 8, '1');  convert_element_type_88 = None
        wait_tensor_36 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_30);  all_gather_into_tensor_30 = None
        split_19 = torch.ops.aten.split.Tensor(wait_tensor_36, 2);  wait_tensor_36 = None
        getitem_179 = split_19[0]
        getitem_180 = split_19[1]
        getitem_181 = split_19[2]
        getitem_182 = split_19[3]
        getitem_183 = split_19[4]
        getitem_184 = split_19[5]
        getitem_185 = split_19[6]
        getitem_186 = split_19[7];  split_19 = None
        cat_11 = torch.ops.aten.cat.default([getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186], 1);  getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16)
        all_gather_into_tensor_31 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_89, 32, '0');  convert_element_type_89 = None
        wait_tensor_37 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_31);  all_gather_into_tensor_31 = None
        permute_30 = torch.ops.aten.permute.default(wait_tensor_37, [1, 0]);  wait_tensor_37 = None
        view_204 = torch.ops.aten.view.default(cat_11, [16384, 4096]);  cat_11 = None
        mm_18 = torch.ops.aten.mm.default(view_204, permute_30);  permute_30 = None
        view_205 = torch.ops.aten.view.default(mm_18, [2, 8192, 1792])
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_92)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_92, sigmoid_2);  convert_element_type_92 = sigmoid_2 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16)
        all_gather_into_tensor_32 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_94, 32, '0');  convert_element_type_94 = None
        wait_tensor_38 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_32);  all_gather_into_tensor_32 = None
        permute_31 = torch.ops.aten.permute.default(wait_tensor_38, [1, 0]);  wait_tensor_38 = None
        mm_19 = torch.ops.aten.mm.default(view_204, permute_31);  view_204 = permute_31 = None
        view_212 = torch.ops.aten.view.default(mm_19, [2, 8192, 1792]);  mm_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_93, view_212);  convert_element_type_93 = view_212 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16)
        all_gather_into_tensor_33 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_97, 32, '0');  convert_element_type_97 = None
        wait_tensor_39 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_33);  all_gather_into_tensor_33 = None
        permute_32 = torch.ops.aten.permute.default(wait_tensor_39, [1, 0]);  wait_tensor_39 = None
        view_219 = torch.ops.aten.view.default(mul_23, [16384, 1792]);  mul_23 = None
        mm_20 = torch.ops.aten.mm.default(view_219, permute_32);  view_219 = permute_32 = None
        view_220 = torch.ops.aten.view.default(mm_20, [2, 8192, 4096]);  mm_20 = None
        split_20 = torch.ops.aten.split.Tensor(view_220, 1024, 1);  view_220 = None
        getitem_187 = split_20[0]
        getitem_188 = split_20[1]
        getitem_189 = split_20[2]
        getitem_190 = split_20[3]
        getitem_191 = split_20[4]
        getitem_192 = split_20[5]
        getitem_193 = split_20[6]
        getitem_194 = split_20[7];  split_20 = None
        cat_12 = torch.ops.aten.cat.default([getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194]);  getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = None
        reduce_scatter_tensor_6 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_12, 'sum', 8, '1');  cat_12 = None
        wait_tensor_40 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_6);  reduce_scatter_tensor_6 = None
        add_11 = torch.ops.aten.add.Tensor(add_9, wait_tensor_40);  add_9 = wait_tensor_40 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16)
        all_gather_into_tensor_34 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_100, 32, '0');  convert_element_type_100 = None
        wait_tensor_41 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_34);  all_gather_into_tensor_34 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_12 = torch.ops.aten.add.Scalar(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_6);  convert_element_type_101 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, wait_tensor_41);  mul_24 = wait_tensor_41 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        all_gather_into_tensor_35 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_102, 8, '1');  convert_element_type_102 = None
        wait_tensor_42 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_35);  all_gather_into_tensor_35 = None
        split_21 = torch.ops.aten.split.Tensor(wait_tensor_42, 2);  wait_tensor_42 = None
        getitem_195 = split_21[0]
        getitem_196 = split_21[1]
        getitem_197 = split_21[2]
        getitem_198 = split_21[3]
        getitem_199 = split_21[4]
        getitem_200 = split_21[5]
        getitem_201 = split_21[6]
        getitem_202 = split_21[7];  split_21 = None
        cat_13 = torch.ops.aten.cat.default([getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202], 1);  getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16)
        all_gather_into_tensor_36 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_103, 32, '0');  convert_element_type_103 = None
        wait_tensor_43 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_36);  all_gather_into_tensor_36 = None
        permute_33 = torch.ops.aten.permute.default(wait_tensor_43, [1, 0]);  wait_tensor_43 = None
        view_231 = torch.ops.aten.view.default(cat_13, [16384, 4096]);  cat_13 = None
        mm_21 = torch.ops.aten.mm.default(view_231, permute_33);  permute_33 = None
        view_232 = torch.ops.aten.view.default(mm_21, [2, 8192, 512])
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16)
        all_gather_into_tensor_37 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_106, 32, '0');  convert_element_type_106 = None
        wait_tensor_44 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_37);  all_gather_into_tensor_37 = None
        permute_34 = torch.ops.aten.permute.default(wait_tensor_44, [1, 0]);  wait_tensor_44 = None
        mm_22 = torch.ops.aten.mm.default(view_231, permute_34);  permute_34 = None
        view_239 = torch.ops.aten.view.default(mm_22, [2, 8192, 128]);  mm_22 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16)
        all_gather_into_tensor_38 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_109, 32, '0');  convert_element_type_109 = None
        wait_tensor_45 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_38);  all_gather_into_tensor_38 = None
        permute_35 = torch.ops.aten.permute.default(wait_tensor_45, [1, 0]);  wait_tensor_45 = None
        mm_23 = torch.ops.aten.mm.default(view_231, permute_35);  view_231 = permute_35 = None
        view_246 = torch.ops.aten.view.default(mm_23, [2, 8192, 128])
        view_248 = torch.ops.aten.view.default(view_232, [2, 8192, -1, 128]);  view_232 = None
        view_249 = torch.ops.aten.view.default(view_239, [2, 8192, -1, 128]);  view_239 = None
        view_250 = torch.ops.aten.view.default(view_246, [2, 8192, -1, 128]);  view_246 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None
        view_251 = torch.ops.aten.view.default(convert_element_type_112, [2, 8192, 4, -1, 2]);  convert_element_type_112 = None
        view_as_complex_6 = torch.ops.aten.view_as_complex.default(view_251);  view_251 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        view_252 = torch.ops.aten.view.default(convert_element_type_113, [2, 8192, 1, -1, 2]);  convert_element_type_113 = None
        view_as_complex_7 = torch.ops.aten.view_as_complex.default(view_252);  view_252 = None
        mul_26 = torch.ops.aten.mul.Tensor(view_as_complex_6, view_37);  view_as_complex_6 = None
        view_as_real_6 = torch.ops.aten.view_as_real.default(mul_26);  mul_26 = None
        view_254 = torch.ops.aten.view.default(view_as_real_6, [2, 8192, 4, 128]);  view_as_real_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_as_complex_7, view_37);  view_as_complex_7 = None
        view_as_real_7 = torch.ops.aten.view_as_real.default(mul_27);  mul_27 = None
        view_255 = torch.ops.aten.view.default(view_as_real_7, [2, 8192, 1, 128]);  view_as_real_7 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(view_254, torch.bfloat16);  view_254 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(view_255, torch.bfloat16);  view_255 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(convert_element_type_115, 3);  convert_element_type_115 = None
        expand_6 = torch.ops.aten.expand.default(unsqueeze_6, [2, 8192, 1, 4, 128]);  unsqueeze_6 = None
        view_256 = torch.ops.aten.view.default(expand_6, [2, 8192, 4, 128]);  expand_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_250, 3);  view_250 = None
        expand_7 = torch.ops.aten.expand.default(unsqueeze_7, [2, 8192, 1, 4, 128]);  unsqueeze_7 = None
        view_257 = torch.ops.aten.view.default(expand_7, [2, 8192, 4, 128]);  expand_7 = None
        permute_36 = torch.ops.aten.permute.default(convert_element_type_114, [0, 2, 1, 3]);  convert_element_type_114 = None
        permute_37 = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        permute_38 = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_36, permute_37, permute_38, None, True, 0.0, True);  permute_36 = permute_37 = permute_38 = None
        getitem_203 = _scaled_dot_product_cudnn_attention_3[0]
        getitem_204 = _scaled_dot_product_cudnn_attention_3[1]
        getitem_209 = _scaled_dot_product_cudnn_attention_3[6]
        getitem_210 = _scaled_dot_product_cudnn_attention_3[7];  _scaled_dot_product_cudnn_attention_3 = None
        permute_39 = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3])
        view_258 = torch.ops.aten.view.default(permute_39, [2, 8192, -1]);  permute_39 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16)
        all_gather_into_tensor_39 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_116, 32, '0');  convert_element_type_116 = None
        wait_tensor_46 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_39);  all_gather_into_tensor_39 = None
        permute_40 = torch.ops.aten.permute.default(wait_tensor_46, [1, 0]);  wait_tensor_46 = None
        view_264 = torch.ops.aten.view.default(view_258, [16384, 512]);  view_258 = None
        mm_24 = torch.ops.aten.mm.default(view_264, permute_40);  view_264 = permute_40 = None
        view_265 = torch.ops.aten.view.default(mm_24, [2, 8192, 4096]);  mm_24 = None
        split_22 = torch.ops.aten.split.Tensor(view_265, 1024, 1);  view_265 = None
        getitem_212 = split_22[0]
        getitem_213 = split_22[1]
        getitem_214 = split_22[2]
        getitem_215 = split_22[3]
        getitem_216 = split_22[4]
        getitem_217 = split_22[5]
        getitem_218 = split_22[6]
        getitem_219 = split_22[7];  split_22 = None
        cat_14 = torch.ops.aten.cat.default([getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219]);  getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = None
        reduce_scatter_tensor_7 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_14, 'sum', 8, '1');  cat_14 = None
        wait_tensor_47 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7)
        add_13 = torch.ops.aten.add.Tensor(add_11, wait_tensor_47);  wait_tensor_47 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16)
        all_gather_into_tensor_40 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_119, 32, '0');  convert_element_type_119 = None
        wait_tensor_48 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_40);  all_gather_into_tensor_40 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_13, torch.float32)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_14 = torch.ops.aten.add.Scalar(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_7);  convert_element_type_120 = rsqrt_7 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, wait_tensor_48);  mul_28 = wait_tensor_48 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        all_gather_into_tensor_41 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_121, 8, '1');  convert_element_type_121 = None
        wait_tensor_49 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_41);  all_gather_into_tensor_41 = None
        split_23 = torch.ops.aten.split.Tensor(wait_tensor_49, 2);  wait_tensor_49 = None
        getitem_220 = split_23[0]
        getitem_221 = split_23[1]
        getitem_222 = split_23[2]
        getitem_223 = split_23[3]
        getitem_224 = split_23[4]
        getitem_225 = split_23[5]
        getitem_226 = split_23[6]
        getitem_227 = split_23[7];  split_23 = None
        cat_15 = torch.ops.aten.cat.default([getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227], 1);  getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16)
        all_gather_into_tensor_42 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_122, 32, '0');  convert_element_type_122 = None
        wait_tensor_50 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_42);  all_gather_into_tensor_42 = None
        permute_41 = torch.ops.aten.permute.default(wait_tensor_50, [1, 0]);  wait_tensor_50 = None
        view_276 = torch.ops.aten.view.default(cat_15, [16384, 4096]);  cat_15 = None
        mm_25 = torch.ops.aten.mm.default(view_276, permute_41);  permute_41 = None
        view_277 = torch.ops.aten.view.default(mm_25, [2, 8192, 1792])
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_125, sigmoid_3);  convert_element_type_125 = sigmoid_3 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16)
        all_gather_into_tensor_43 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_127, 32, '0');  convert_element_type_127 = None
        wait_tensor_51 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_43);  all_gather_into_tensor_43 = None
        permute_42 = torch.ops.aten.permute.default(wait_tensor_51, [1, 0]);  wait_tensor_51 = None
        mm_26 = torch.ops.aten.mm.default(view_276, permute_42);  view_276 = permute_42 = None
        view_284 = torch.ops.aten.view.default(mm_26, [2, 8192, 1792]);  mm_26 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_126, view_284);  convert_element_type_126 = view_284 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16)
        all_gather_into_tensor_44 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_130, 32, '0');  convert_element_type_130 = None
        wait_tensor_52 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_44);  all_gather_into_tensor_44 = None
        permute_43 = torch.ops.aten.permute.default(wait_tensor_52, [1, 0]);  wait_tensor_52 = None
        view_291 = torch.ops.aten.view.default(mul_31, [16384, 1792]);  mul_31 = None
        mm_27 = torch.ops.aten.mm.default(view_291, permute_43);  view_291 = permute_43 = None
        view_292 = torch.ops.aten.view.default(mm_27, [2, 8192, 4096]);  mm_27 = None
        split_24 = torch.ops.aten.split.Tensor(view_292, 1024, 1);  view_292 = None
        getitem_228 = split_24[0]
        getitem_229 = split_24[1]
        getitem_230 = split_24[2]
        getitem_231 = split_24[3]
        getitem_232 = split_24[4]
        getitem_233 = split_24[5]
        getitem_234 = split_24[6]
        getitem_235 = split_24[7];  split_24 = None
        cat_16 = torch.ops.aten.cat.default([getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235]);  getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = None
        reduce_scatter_tensor_8 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_16, 'sum', 8, '1');  cat_16 = None
        wait_tensor_53 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_8);  reduce_scatter_tensor_8 = None
        add_15 = torch.ops.aten.add.Tensor(add_13, wait_tensor_53);  add_13 = wait_tensor_53 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16)
        all_gather_into_tensor_45 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_133, 32, '0');  convert_element_type_133 = None
        wait_tensor_54 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_45);  all_gather_into_tensor_45 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_16 = torch.ops.aten.add.Scalar(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_8);  convert_element_type_134 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, wait_tensor_54);  mul_32 = wait_tensor_54 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        all_gather_into_tensor_46 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_135, 8, '1');  convert_element_type_135 = None
        wait_tensor_55 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_46);  all_gather_into_tensor_46 = None
        split_25 = torch.ops.aten.split.Tensor(wait_tensor_55, 2);  wait_tensor_55 = None
        getitem_236 = split_25[0]
        getitem_237 = split_25[1]
        getitem_238 = split_25[2]
        getitem_239 = split_25[3]
        getitem_240 = split_25[4]
        getitem_241 = split_25[5]
        getitem_242 = split_25[6]
        getitem_243 = split_25[7];  split_25 = None
        cat_17 = torch.ops.aten.cat.default([getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243], 1);  getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16)
        all_gather_into_tensor_47 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_136, 32, '0');  convert_element_type_136 = None
        wait_tensor_56 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_47);  all_gather_into_tensor_47 = None
        permute_44 = torch.ops.aten.permute.default(wait_tensor_56, [1, 0]);  wait_tensor_56 = None
        view_303 = torch.ops.aten.view.default(cat_17, [16384, 4096]);  cat_17 = None
        mm_28 = torch.ops.aten.mm.default(view_303, permute_44);  permute_44 = None
        view_304 = torch.ops.aten.view.default(mm_28, [2, 8192, 512])
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16)
        all_gather_into_tensor_48 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_139, 32, '0');  convert_element_type_139 = None
        wait_tensor_57 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_48);  all_gather_into_tensor_48 = None
        permute_45 = torch.ops.aten.permute.default(wait_tensor_57, [1, 0]);  wait_tensor_57 = None
        mm_29 = torch.ops.aten.mm.default(view_303, permute_45);  permute_45 = None
        view_311 = torch.ops.aten.view.default(mm_29, [2, 8192, 128]);  mm_29 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16)
        all_gather_into_tensor_49 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_142, 32, '0');  convert_element_type_142 = None
        wait_tensor_58 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_49);  all_gather_into_tensor_49 = None
        permute_46 = torch.ops.aten.permute.default(wait_tensor_58, [1, 0]);  wait_tensor_58 = None
        mm_30 = torch.ops.aten.mm.default(view_303, permute_46);  view_303 = permute_46 = None
        view_318 = torch.ops.aten.view.default(mm_30, [2, 8192, 128])
        view_320 = torch.ops.aten.view.default(view_304, [2, 8192, -1, 128]);  view_304 = None
        view_321 = torch.ops.aten.view.default(view_311, [2, 8192, -1, 128]);  view_311 = None
        view_322 = torch.ops.aten.view.default(view_318, [2, 8192, -1, 128]);  view_318 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        view_323 = torch.ops.aten.view.default(convert_element_type_145, [2, 8192, 4, -1, 2]);  convert_element_type_145 = None
        view_as_complex_8 = torch.ops.aten.view_as_complex.default(view_323);  view_323 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        view_324 = torch.ops.aten.view.default(convert_element_type_146, [2, 8192, 1, -1, 2]);  convert_element_type_146 = None
        view_as_complex_9 = torch.ops.aten.view_as_complex.default(view_324);  view_324 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_as_complex_8, view_37);  view_as_complex_8 = None
        view_as_real_8 = torch.ops.aten.view_as_real.default(mul_34);  mul_34 = None
        view_326 = torch.ops.aten.view.default(view_as_real_8, [2, 8192, 4, 128]);  view_as_real_8 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_as_complex_9, view_37);  view_as_complex_9 = None
        view_as_real_9 = torch.ops.aten.view_as_real.default(mul_35);  mul_35 = None
        view_327 = torch.ops.aten.view.default(view_as_real_9, [2, 8192, 1, 128]);  view_as_real_9 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(view_327, torch.bfloat16);  view_327 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(convert_element_type_148, 3);  convert_element_type_148 = None
        expand_8 = torch.ops.aten.expand.default(unsqueeze_8, [2, 8192, 1, 4, 128]);  unsqueeze_8 = None
        view_328 = torch.ops.aten.view.default(expand_8, [2, 8192, 4, 128]);  expand_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(view_322, 3);  view_322 = None
        expand_9 = torch.ops.aten.expand.default(unsqueeze_9, [2, 8192, 1, 4, 128]);  unsqueeze_9 = None
        view_329 = torch.ops.aten.view.default(expand_9, [2, 8192, 4, 128]);  expand_9 = None
        permute_47 = torch.ops.aten.permute.default(convert_element_type_147, [0, 2, 1, 3]);  convert_element_type_147 = None
        permute_48 = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        permute_49 = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_47, permute_48, permute_49, None, True, 0.0, True);  permute_47 = permute_48 = permute_49 = None
        getitem_244 = _scaled_dot_product_cudnn_attention_4[0]
        getitem_245 = _scaled_dot_product_cudnn_attention_4[1]
        getitem_250 = _scaled_dot_product_cudnn_attention_4[6]
        getitem_251 = _scaled_dot_product_cudnn_attention_4[7];  _scaled_dot_product_cudnn_attention_4 = None
        permute_50 = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3])
        view_330 = torch.ops.aten.view.default(permute_50, [2, 8192, -1]);  permute_50 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16)
        all_gather_into_tensor_50 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_149, 32, '0');  convert_element_type_149 = None
        wait_tensor_59 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_50);  all_gather_into_tensor_50 = None
        permute_51 = torch.ops.aten.permute.default(wait_tensor_59, [1, 0]);  wait_tensor_59 = None
        view_336 = torch.ops.aten.view.default(view_330, [16384, 512]);  view_330 = None
        mm_31 = torch.ops.aten.mm.default(view_336, permute_51);  view_336 = permute_51 = None
        view_337 = torch.ops.aten.view.default(mm_31, [2, 8192, 4096]);  mm_31 = None
        split_26 = torch.ops.aten.split.Tensor(view_337, 1024, 1);  view_337 = None
        getitem_253 = split_26[0]
        getitem_254 = split_26[1]
        getitem_255 = split_26[2]
        getitem_256 = split_26[3]
        getitem_257 = split_26[4]
        getitem_258 = split_26[5]
        getitem_259 = split_26[6]
        getitem_260 = split_26[7];  split_26 = None
        cat_18 = torch.ops.aten.cat.default([getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260]);  getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = None
        reduce_scatter_tensor_9 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_18, 'sum', 8, '1');  cat_18 = None
        wait_tensor_60 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_9)
        add_17 = torch.ops.aten.add.Tensor(add_15, wait_tensor_60);  wait_tensor_60 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16)
        all_gather_into_tensor_51 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_152, 32, '0');  convert_element_type_152 = None
        wait_tensor_61 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_51);  all_gather_into_tensor_51 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_18 = torch.ops.aten.add.Scalar(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_9);  convert_element_type_153 = rsqrt_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, wait_tensor_61);  mul_36 = wait_tensor_61 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        all_gather_into_tensor_52 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_154, 8, '1');  convert_element_type_154 = None
        wait_tensor_62 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_52);  all_gather_into_tensor_52 = None
        split_27 = torch.ops.aten.split.Tensor(wait_tensor_62, 2);  wait_tensor_62 = None
        getitem_261 = split_27[0]
        getitem_262 = split_27[1]
        getitem_263 = split_27[2]
        getitem_264 = split_27[3]
        getitem_265 = split_27[4]
        getitem_266 = split_27[5]
        getitem_267 = split_27[6]
        getitem_268 = split_27[7];  split_27 = None
        cat_19 = torch.ops.aten.cat.default([getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268], 1);  getitem_261 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16)
        all_gather_into_tensor_53 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_155, 32, '0');  convert_element_type_155 = None
        wait_tensor_63 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_53);  all_gather_into_tensor_53 = None
        permute_52 = torch.ops.aten.permute.default(wait_tensor_63, [1, 0]);  wait_tensor_63 = None
        view_348 = torch.ops.aten.view.default(cat_19, [16384, 4096]);  cat_19 = None
        mm_32 = torch.ops.aten.mm.default(view_348, permute_52);  permute_52 = None
        view_349 = torch.ops.aten.view.default(mm_32, [2, 8192, 1792])
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_158)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_158, sigmoid_4);  convert_element_type_158 = sigmoid_4 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16)
        all_gather_into_tensor_54 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_160, 32, '0');  convert_element_type_160 = None
        wait_tensor_64 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_54);  all_gather_into_tensor_54 = None
        permute_53 = torch.ops.aten.permute.default(wait_tensor_64, [1, 0]);  wait_tensor_64 = None
        mm_33 = torch.ops.aten.mm.default(view_348, permute_53);  view_348 = permute_53 = None
        view_356 = torch.ops.aten.view.default(mm_33, [2, 8192, 1792]);  mm_33 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_159, view_356);  convert_element_type_159 = view_356 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16)
        all_gather_into_tensor_55 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_163, 32, '0');  convert_element_type_163 = None
        wait_tensor_65 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_55);  all_gather_into_tensor_55 = None
        permute_54 = torch.ops.aten.permute.default(wait_tensor_65, [1, 0]);  wait_tensor_65 = None
        view_363 = torch.ops.aten.view.default(mul_39, [16384, 1792]);  mul_39 = None
        mm_34 = torch.ops.aten.mm.default(view_363, permute_54);  view_363 = permute_54 = None
        view_364 = torch.ops.aten.view.default(mm_34, [2, 8192, 4096]);  mm_34 = None
        split_28 = torch.ops.aten.split.Tensor(view_364, 1024, 1);  view_364 = None
        getitem_269 = split_28[0]
        getitem_270 = split_28[1]
        getitem_271 = split_28[2]
        getitem_272 = split_28[3]
        getitem_273 = split_28[4]
        getitem_274 = split_28[5]
        getitem_275 = split_28[6]
        getitem_276 = split_28[7];  split_28 = None
        cat_20 = torch.ops.aten.cat.default([getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276]);  getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = None
        reduce_scatter_tensor_10 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_20, 'sum', 8, '1');  cat_20 = None
        wait_tensor_66 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_10);  reduce_scatter_tensor_10 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, wait_tensor_66);  add_17 = wait_tensor_66 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16)
        all_gather_into_tensor_56 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_166, 32, '0');  convert_element_type_166 = None
        wait_tensor_67 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_56);  all_gather_into_tensor_56 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
        add_20 = torch.ops.aten.add.Scalar(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_167, rsqrt_10);  convert_element_type_167 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, wait_tensor_67);  mul_40 = wait_tensor_67 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        all_gather_into_tensor_57 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_168, 8, '1');  convert_element_type_168 = None
        wait_tensor_68 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_57);  all_gather_into_tensor_57 = None
        split_29 = torch.ops.aten.split.Tensor(wait_tensor_68, 2);  wait_tensor_68 = None
        getitem_277 = split_29[0]
        getitem_278 = split_29[1]
        getitem_279 = split_29[2]
        getitem_280 = split_29[3]
        getitem_281 = split_29[4]
        getitem_282 = split_29[5]
        getitem_283 = split_29[6]
        getitem_284 = split_29[7];  split_29 = None
        cat_21 = torch.ops.aten.cat.default([getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284], 1);  getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16)
        all_gather_into_tensor_58 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_169, 32, '0');  convert_element_type_169 = None
        wait_tensor_69 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_58);  all_gather_into_tensor_58 = None
        permute_55 = torch.ops.aten.permute.default(wait_tensor_69, [1, 0]);  wait_tensor_69 = None
        view_375 = torch.ops.aten.view.default(cat_21, [16384, 4096]);  cat_21 = None
        mm_35 = torch.ops.aten.mm.default(view_375, permute_55);  permute_55 = None
        view_376 = torch.ops.aten.view.default(mm_35, [2, 8192, 512])
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16)
        all_gather_into_tensor_59 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_172, 32, '0');  convert_element_type_172 = None
        wait_tensor_70 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_59);  all_gather_into_tensor_59 = None
        permute_56 = torch.ops.aten.permute.default(wait_tensor_70, [1, 0]);  wait_tensor_70 = None
        mm_36 = torch.ops.aten.mm.default(view_375, permute_56);  permute_56 = None
        view_383 = torch.ops.aten.view.default(mm_36, [2, 8192, 128]);  mm_36 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16)
        all_gather_into_tensor_60 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_175, 32, '0');  convert_element_type_175 = None
        wait_tensor_71 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_60);  all_gather_into_tensor_60 = None
        permute_57 = torch.ops.aten.permute.default(wait_tensor_71, [1, 0]);  wait_tensor_71 = None
        mm_37 = torch.ops.aten.mm.default(view_375, permute_57);  view_375 = permute_57 = None
        view_390 = torch.ops.aten.view.default(mm_37, [2, 8192, 128])
        view_392 = torch.ops.aten.view.default(view_376, [2, 8192, -1, 128]);  view_376 = None
        view_393 = torch.ops.aten.view.default(view_383, [2, 8192, -1, 128]);  view_383 = None
        view_394 = torch.ops.aten.view.default(view_390, [2, 8192, -1, 128]);  view_390 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(view_392, torch.float32);  view_392 = None
        view_395 = torch.ops.aten.view.default(convert_element_type_178, [2, 8192, 4, -1, 2]);  convert_element_type_178 = None
        view_as_complex_10 = torch.ops.aten.view_as_complex.default(view_395);  view_395 = None
        convert_element_type_179 = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        view_396 = torch.ops.aten.view.default(convert_element_type_179, [2, 8192, 1, -1, 2]);  convert_element_type_179 = None
        view_as_complex_11 = torch.ops.aten.view_as_complex.default(view_396);  view_396 = None
        mul_42 = torch.ops.aten.mul.Tensor(view_as_complex_10, view_37);  view_as_complex_10 = None
        view_as_real_10 = torch.ops.aten.view_as_real.default(mul_42);  mul_42 = None
        view_398 = torch.ops.aten.view.default(view_as_real_10, [2, 8192, 4, 128]);  view_as_real_10 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_as_complex_11, view_37);  view_as_complex_11 = None
        view_as_real_11 = torch.ops.aten.view_as_real.default(mul_43);  mul_43 = None
        view_399 = torch.ops.aten.view.default(view_as_real_11, [2, 8192, 1, 128]);  view_as_real_11 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(view_398, torch.bfloat16);  view_398 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(view_399, torch.bfloat16);  view_399 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3);  convert_element_type_181 = None
        expand_10 = torch.ops.aten.expand.default(unsqueeze_10, [2, 8192, 1, 4, 128]);  unsqueeze_10 = None
        view_400 = torch.ops.aten.view.default(expand_10, [2, 8192, 4, 128]);  expand_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(view_394, 3);  view_394 = None
        expand_11 = torch.ops.aten.expand.default(unsqueeze_11, [2, 8192, 1, 4, 128]);  unsqueeze_11 = None
        view_401 = torch.ops.aten.view.default(expand_11, [2, 8192, 4, 128]);  expand_11 = None
        permute_58 = torch.ops.aten.permute.default(convert_element_type_180, [0, 2, 1, 3]);  convert_element_type_180 = None
        permute_59 = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        permute_60 = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_58, permute_59, permute_60, None, True, 0.0, True);  permute_58 = permute_59 = permute_60 = None
        getitem_285 = _scaled_dot_product_cudnn_attention_5[0]
        getitem_286 = _scaled_dot_product_cudnn_attention_5[1]
        getitem_291 = _scaled_dot_product_cudnn_attention_5[6]
        getitem_292 = _scaled_dot_product_cudnn_attention_5[7];  _scaled_dot_product_cudnn_attention_5 = None
        permute_61 = torch.ops.aten.permute.default(getitem_285, [0, 2, 1, 3])
        view_402 = torch.ops.aten.view.default(permute_61, [2, 8192, -1]);  permute_61 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16)
        all_gather_into_tensor_61 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_182, 32, '0');  convert_element_type_182 = None
        wait_tensor_72 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_61);  all_gather_into_tensor_61 = None
        permute_62 = torch.ops.aten.permute.default(wait_tensor_72, [1, 0]);  wait_tensor_72 = None
        view_408 = torch.ops.aten.view.default(view_402, [16384, 512]);  view_402 = None
        mm_38 = torch.ops.aten.mm.default(view_408, permute_62);  view_408 = permute_62 = None
        view_409 = torch.ops.aten.view.default(mm_38, [2, 8192, 4096]);  mm_38 = None
        split_30 = torch.ops.aten.split.Tensor(view_409, 1024, 1);  view_409 = None
        getitem_294 = split_30[0]
        getitem_295 = split_30[1]
        getitem_296 = split_30[2]
        getitem_297 = split_30[3]
        getitem_298 = split_30[4]
        getitem_299 = split_30[5]
        getitem_300 = split_30[6]
        getitem_301 = split_30[7];  split_30 = None
        cat_22 = torch.ops.aten.cat.default([getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301]);  getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = getitem_301 = None
        reduce_scatter_tensor_11 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_22, 'sum', 8, '1');  cat_22 = None
        wait_tensor_73 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_11)
        add_21 = torch.ops.aten.add.Tensor(add_19, wait_tensor_73);  wait_tensor_73 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16)
        all_gather_into_tensor_62 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_185, 32, '0');  convert_element_type_185 = None
        wait_tensor_74 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_62);  all_gather_into_tensor_62 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_22 = torch.ops.aten.add.Scalar(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_186, rsqrt_11);  convert_element_type_186 = rsqrt_11 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, wait_tensor_74);  mul_44 = wait_tensor_74 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        all_gather_into_tensor_63 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_187, 8, '1');  convert_element_type_187 = None
        wait_tensor_75 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_63);  all_gather_into_tensor_63 = None
        split_31 = torch.ops.aten.split.Tensor(wait_tensor_75, 2);  wait_tensor_75 = None
        getitem_302 = split_31[0]
        getitem_303 = split_31[1]
        getitem_304 = split_31[2]
        getitem_305 = split_31[3]
        getitem_306 = split_31[4]
        getitem_307 = split_31[5]
        getitem_308 = split_31[6]
        getitem_309 = split_31[7];  split_31 = None
        cat_23 = torch.ops.aten.cat.default([getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309], 1);  getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16)
        all_gather_into_tensor_64 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_188, 32, '0');  convert_element_type_188 = None
        wait_tensor_76 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_64);  all_gather_into_tensor_64 = None
        permute_63 = torch.ops.aten.permute.default(wait_tensor_76, [1, 0]);  wait_tensor_76 = None
        view_420 = torch.ops.aten.view.default(cat_23, [16384, 4096]);  cat_23 = None
        mm_39 = torch.ops.aten.mm.default(view_420, permute_63);  permute_63 = None
        view_421 = torch.ops.aten.view.default(mm_39, [2, 8192, 1792])
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_191)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_191, sigmoid_5);  convert_element_type_191 = sigmoid_5 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16)
        all_gather_into_tensor_65 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_193, 32, '0');  convert_element_type_193 = None
        wait_tensor_77 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_65);  all_gather_into_tensor_65 = None
        permute_64 = torch.ops.aten.permute.default(wait_tensor_77, [1, 0]);  wait_tensor_77 = None
        mm_40 = torch.ops.aten.mm.default(view_420, permute_64);  view_420 = permute_64 = None
        view_428 = torch.ops.aten.view.default(mm_40, [2, 8192, 1792]);  mm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_192, view_428);  convert_element_type_192 = view_428 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16)
        all_gather_into_tensor_66 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_196, 32, '0');  convert_element_type_196 = None
        wait_tensor_78 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_66);  all_gather_into_tensor_66 = None
        permute_65 = torch.ops.aten.permute.default(wait_tensor_78, [1, 0]);  wait_tensor_78 = None
        view_435 = torch.ops.aten.view.default(mul_47, [16384, 1792]);  mul_47 = None
        mm_41 = torch.ops.aten.mm.default(view_435, permute_65);  view_435 = permute_65 = None
        view_436 = torch.ops.aten.view.default(mm_41, [2, 8192, 4096]);  mm_41 = None
        split_32 = torch.ops.aten.split.Tensor(view_436, 1024, 1);  view_436 = None
        getitem_310 = split_32[0]
        getitem_311 = split_32[1]
        getitem_312 = split_32[2]
        getitem_313 = split_32[3]
        getitem_314 = split_32[4]
        getitem_315 = split_32[5]
        getitem_316 = split_32[6]
        getitem_317 = split_32[7];  split_32 = None
        cat_24 = torch.ops.aten.cat.default([getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317]);  getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = None
        reduce_scatter_tensor_12 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_24, 'sum', 8, '1');  cat_24 = None
        wait_tensor_79 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_12);  reduce_scatter_tensor_12 = None
        add_23 = torch.ops.aten.add.Tensor(add_21, wait_tensor_79);  add_21 = wait_tensor_79 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16)
        all_gather_into_tensor_67 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_199, 32, '0');  convert_element_type_199 = None
        wait_tensor_80 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_67);  all_gather_into_tensor_67 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [2], True);  pow_13 = None
        add_24 = torch.ops.aten.add.Scalar(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_12);  convert_element_type_200 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, wait_tensor_80);  mul_48 = wait_tensor_80 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        all_gather_into_tensor_68 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_201, 8, '1');  convert_element_type_201 = None
        wait_tensor_81 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_68);  all_gather_into_tensor_68 = None
        split_33 = torch.ops.aten.split.Tensor(wait_tensor_81, 2);  wait_tensor_81 = None
        getitem_318 = split_33[0]
        getitem_319 = split_33[1]
        getitem_320 = split_33[2]
        getitem_321 = split_33[3]
        getitem_322 = split_33[4]
        getitem_323 = split_33[5]
        getitem_324 = split_33[6]
        getitem_325 = split_33[7];  split_33 = None
        cat_25 = torch.ops.aten.cat.default([getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325], 1);  getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16)
        all_gather_into_tensor_69 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_202, 32, '0');  convert_element_type_202 = None
        wait_tensor_82 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_69);  all_gather_into_tensor_69 = None
        permute_66 = torch.ops.aten.permute.default(wait_tensor_82, [1, 0]);  wait_tensor_82 = None
        view_447 = torch.ops.aten.view.default(cat_25, [16384, 4096]);  cat_25 = None
        mm_42 = torch.ops.aten.mm.default(view_447, permute_66);  permute_66 = None
        view_448 = torch.ops.aten.view.default(mm_42, [2, 8192, 512])
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16)
        all_gather_into_tensor_70 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_205, 32, '0');  convert_element_type_205 = None
        wait_tensor_83 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_70);  all_gather_into_tensor_70 = None
        permute_67 = torch.ops.aten.permute.default(wait_tensor_83, [1, 0]);  wait_tensor_83 = None
        mm_43 = torch.ops.aten.mm.default(view_447, permute_67);  permute_67 = None
        view_455 = torch.ops.aten.view.default(mm_43, [2, 8192, 128]);  mm_43 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16)
        all_gather_into_tensor_71 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_208, 32, '0');  convert_element_type_208 = None
        wait_tensor_84 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_71);  all_gather_into_tensor_71 = None
        permute_68 = torch.ops.aten.permute.default(wait_tensor_84, [1, 0]);  wait_tensor_84 = None
        mm_44 = torch.ops.aten.mm.default(view_447, permute_68);  view_447 = permute_68 = None
        view_462 = torch.ops.aten.view.default(mm_44, [2, 8192, 128])
        view_464 = torch.ops.aten.view.default(view_448, [2, 8192, -1, 128]);  view_448 = None
        view_465 = torch.ops.aten.view.default(view_455, [2, 8192, -1, 128]);  view_455 = None
        view_466 = torch.ops.aten.view.default(view_462, [2, 8192, -1, 128]);  view_462 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(view_464, torch.float32);  view_464 = None
        view_467 = torch.ops.aten.view.default(convert_element_type_211, [2, 8192, 4, -1, 2]);  convert_element_type_211 = None
        view_as_complex_12 = torch.ops.aten.view_as_complex.default(view_467);  view_467 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(view_465, torch.float32);  view_465 = None
        view_468 = torch.ops.aten.view.default(convert_element_type_212, [2, 8192, 1, -1, 2]);  convert_element_type_212 = None
        view_as_complex_13 = torch.ops.aten.view_as_complex.default(view_468);  view_468 = None
        mul_50 = torch.ops.aten.mul.Tensor(view_as_complex_12, view_37);  view_as_complex_12 = None
        view_as_real_12 = torch.ops.aten.view_as_real.default(mul_50);  mul_50 = None
        view_470 = torch.ops.aten.view.default(view_as_real_12, [2, 8192, 4, 128]);  view_as_real_12 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_as_complex_13, view_37);  view_as_complex_13 = None
        view_as_real_13 = torch.ops.aten.view_as_real.default(mul_51);  mul_51 = None
        view_471 = torch.ops.aten.view.default(view_as_real_13, [2, 8192, 1, 128]);  view_as_real_13 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(view_470, torch.bfloat16);  view_470 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(convert_element_type_214, 3);  convert_element_type_214 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_12, [2, 8192, 1, 4, 128]);  unsqueeze_12 = None
        view_472 = torch.ops.aten.view.default(expand_12, [2, 8192, 4, 128]);  expand_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(view_466, 3);  view_466 = None
        expand_13 = torch.ops.aten.expand.default(unsqueeze_13, [2, 8192, 1, 4, 128]);  unsqueeze_13 = None
        view_473 = torch.ops.aten.view.default(expand_13, [2, 8192, 4, 128]);  expand_13 = None
        permute_69 = torch.ops.aten.permute.default(convert_element_type_213, [0, 2, 1, 3]);  convert_element_type_213 = None
        permute_70 = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        permute_71 = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_69, permute_70, permute_71, None, True, 0.0, True);  permute_69 = permute_70 = permute_71 = None
        getitem_326 = _scaled_dot_product_cudnn_attention_6[0]
        getitem_327 = _scaled_dot_product_cudnn_attention_6[1]
        getitem_332 = _scaled_dot_product_cudnn_attention_6[6]
        getitem_333 = _scaled_dot_product_cudnn_attention_6[7];  _scaled_dot_product_cudnn_attention_6 = None
        permute_72 = torch.ops.aten.permute.default(getitem_326, [0, 2, 1, 3])
        view_474 = torch.ops.aten.view.default(permute_72, [2, 8192, -1]);  permute_72 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16)
        all_gather_into_tensor_72 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_215, 32, '0');  convert_element_type_215 = None
        wait_tensor_85 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_72);  all_gather_into_tensor_72 = None
        permute_73 = torch.ops.aten.permute.default(wait_tensor_85, [1, 0]);  wait_tensor_85 = None
        view_480 = torch.ops.aten.view.default(view_474, [16384, 512]);  view_474 = None
        mm_45 = torch.ops.aten.mm.default(view_480, permute_73);  view_480 = permute_73 = None
        view_481 = torch.ops.aten.view.default(mm_45, [2, 8192, 4096]);  mm_45 = None
        split_34 = torch.ops.aten.split.Tensor(view_481, 1024, 1);  view_481 = None
        getitem_335 = split_34[0]
        getitem_336 = split_34[1]
        getitem_337 = split_34[2]
        getitem_338 = split_34[3]
        getitem_339 = split_34[4]
        getitem_340 = split_34[5]
        getitem_341 = split_34[6]
        getitem_342 = split_34[7];  split_34 = None
        cat_26 = torch.ops.aten.cat.default([getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342]);  getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = None
        reduce_scatter_tensor_13 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_26, 'sum', 8, '1');  cat_26 = None
        wait_tensor_86 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_13)
        add_25 = torch.ops.aten.add.Tensor(add_23, wait_tensor_86);  wait_tensor_86 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16)
        all_gather_into_tensor_73 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_218, 32, '0');  convert_element_type_218 = None
        wait_tensor_87 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_73);  all_gather_into_tensor_73 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [2], True);  pow_14 = None
        add_26 = torch.ops.aten.add.Scalar(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_13);  convert_element_type_219 = rsqrt_13 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, wait_tensor_87);  mul_52 = wait_tensor_87 = None
        convert_element_type_220 = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        all_gather_into_tensor_74 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_220, 8, '1');  convert_element_type_220 = None
        wait_tensor_88 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_74);  all_gather_into_tensor_74 = None
        split_35 = torch.ops.aten.split.Tensor(wait_tensor_88, 2);  wait_tensor_88 = None
        getitem_343 = split_35[0]
        getitem_344 = split_35[1]
        getitem_345 = split_35[2]
        getitem_346 = split_35[3]
        getitem_347 = split_35[4]
        getitem_348 = split_35[5]
        getitem_349 = split_35[6]
        getitem_350 = split_35[7];  split_35 = None
        cat_27 = torch.ops.aten.cat.default([getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350], 1);  getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16)
        all_gather_into_tensor_75 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_221, 32, '0');  convert_element_type_221 = None
        wait_tensor_89 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_75);  all_gather_into_tensor_75 = None
        permute_74 = torch.ops.aten.permute.default(wait_tensor_89, [1, 0]);  wait_tensor_89 = None
        view_492 = torch.ops.aten.view.default(cat_27, [16384, 4096]);  cat_27 = None
        mm_46 = torch.ops.aten.mm.default(view_492, permute_74);  permute_74 = None
        view_493 = torch.ops.aten.view.default(mm_46, [2, 8192, 1792])
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_224)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_224, sigmoid_6);  convert_element_type_224 = sigmoid_6 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16)
        all_gather_into_tensor_76 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_226, 32, '0');  convert_element_type_226 = None
        wait_tensor_90 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_76);  all_gather_into_tensor_76 = None
        permute_75 = torch.ops.aten.permute.default(wait_tensor_90, [1, 0]);  wait_tensor_90 = None
        mm_47 = torch.ops.aten.mm.default(view_492, permute_75);  view_492 = permute_75 = None
        view_500 = torch.ops.aten.view.default(mm_47, [2, 8192, 1792]);  mm_47 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_225, view_500);  convert_element_type_225 = view_500 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16)
        all_gather_into_tensor_77 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_229, 32, '0');  convert_element_type_229 = None
        wait_tensor_91 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_77);  all_gather_into_tensor_77 = None
        permute_76 = torch.ops.aten.permute.default(wait_tensor_91, [1, 0]);  wait_tensor_91 = None
        view_507 = torch.ops.aten.view.default(mul_55, [16384, 1792]);  mul_55 = None
        mm_48 = torch.ops.aten.mm.default(view_507, permute_76);  view_507 = permute_76 = None
        view_508 = torch.ops.aten.view.default(mm_48, [2, 8192, 4096]);  mm_48 = None
        split_36 = torch.ops.aten.split.Tensor(view_508, 1024, 1);  view_508 = None
        getitem_351 = split_36[0]
        getitem_352 = split_36[1]
        getitem_353 = split_36[2]
        getitem_354 = split_36[3]
        getitem_355 = split_36[4]
        getitem_356 = split_36[5]
        getitem_357 = split_36[6]
        getitem_358 = split_36[7];  split_36 = None
        cat_28 = torch.ops.aten.cat.default([getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358]);  getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = None
        reduce_scatter_tensor_14 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_28, 'sum', 8, '1');  cat_28 = None
        wait_tensor_92 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_14);  reduce_scatter_tensor_14 = None
        add_27 = torch.ops.aten.add.Tensor(add_25, wait_tensor_92);  add_25 = wait_tensor_92 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16)
        all_gather_into_tensor_78 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_232, 32, '0');  convert_element_type_232 = None
        wait_tensor_93 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_78);  all_gather_into_tensor_78 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_28 = torch.ops.aten.add.Scalar(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_14);  convert_element_type_233 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, wait_tensor_93);  mul_56 = wait_tensor_93 = None
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        all_gather_into_tensor_79 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_234, 8, '1');  convert_element_type_234 = None
        wait_tensor_94 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_79);  all_gather_into_tensor_79 = None
        split_37 = torch.ops.aten.split.Tensor(wait_tensor_94, 2);  wait_tensor_94 = None
        getitem_359 = split_37[0]
        getitem_360 = split_37[1]
        getitem_361 = split_37[2]
        getitem_362 = split_37[3]
        getitem_363 = split_37[4]
        getitem_364 = split_37[5]
        getitem_365 = split_37[6]
        getitem_366 = split_37[7];  split_37 = None
        cat_29 = torch.ops.aten.cat.default([getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366], 1);  getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16)
        all_gather_into_tensor_80 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_235, 32, '0');  convert_element_type_235 = None
        wait_tensor_95 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_80);  all_gather_into_tensor_80 = None
        permute_77 = torch.ops.aten.permute.default(wait_tensor_95, [1, 0]);  wait_tensor_95 = None
        view_519 = torch.ops.aten.view.default(cat_29, [16384, 4096]);  cat_29 = None
        mm_49 = torch.ops.aten.mm.default(view_519, permute_77);  permute_77 = None
        view_520 = torch.ops.aten.view.default(mm_49, [2, 8192, 512])
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16)
        all_gather_into_tensor_81 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_238, 32, '0');  convert_element_type_238 = None
        wait_tensor_96 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_81);  all_gather_into_tensor_81 = None
        permute_78 = torch.ops.aten.permute.default(wait_tensor_96, [1, 0]);  wait_tensor_96 = None
        mm_50 = torch.ops.aten.mm.default(view_519, permute_78);  permute_78 = None
        view_527 = torch.ops.aten.view.default(mm_50, [2, 8192, 128]);  mm_50 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16)
        all_gather_into_tensor_82 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_241, 32, '0');  convert_element_type_241 = None
        wait_tensor_97 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_82);  all_gather_into_tensor_82 = None
        permute_79 = torch.ops.aten.permute.default(wait_tensor_97, [1, 0]);  wait_tensor_97 = None
        mm_51 = torch.ops.aten.mm.default(view_519, permute_79);  view_519 = permute_79 = None
        view_534 = torch.ops.aten.view.default(mm_51, [2, 8192, 128])
        view_536 = torch.ops.aten.view.default(view_520, [2, 8192, -1, 128]);  view_520 = None
        view_537 = torch.ops.aten.view.default(view_527, [2, 8192, -1, 128]);  view_527 = None
        view_538 = torch.ops.aten.view.default(view_534, [2, 8192, -1, 128]);  view_534 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        view_539 = torch.ops.aten.view.default(convert_element_type_244, [2, 8192, 4, -1, 2]);  convert_element_type_244 = None
        view_as_complex_14 = torch.ops.aten.view_as_complex.default(view_539);  view_539 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        view_540 = torch.ops.aten.view.default(convert_element_type_245, [2, 8192, 1, -1, 2]);  convert_element_type_245 = None
        view_as_complex_15 = torch.ops.aten.view_as_complex.default(view_540);  view_540 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_as_complex_14, view_37);  view_as_complex_14 = None
        view_as_real_14 = torch.ops.aten.view_as_real.default(mul_58);  mul_58 = None
        view_542 = torch.ops.aten.view.default(view_as_real_14, [2, 8192, 4, 128]);  view_as_real_14 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_as_complex_15, view_37);  view_as_complex_15 = None
        view_as_real_15 = torch.ops.aten.view_as_real.default(mul_59);  mul_59 = None
        view_543 = torch.ops.aten.view.default(view_as_real_15, [2, 8192, 1, 128]);  view_as_real_15 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(view_543, torch.bfloat16);  view_543 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(convert_element_type_247, 3);  convert_element_type_247 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_14, [2, 8192, 1, 4, 128]);  unsqueeze_14 = None
        view_544 = torch.ops.aten.view.default(expand_14, [2, 8192, 4, 128]);  expand_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(view_538, 3);  view_538 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_15, [2, 8192, 1, 4, 128]);  unsqueeze_15 = None
        view_545 = torch.ops.aten.view.default(expand_15, [2, 8192, 4, 128]);  expand_15 = None
        permute_80 = torch.ops.aten.permute.default(convert_element_type_246, [0, 2, 1, 3]);  convert_element_type_246 = None
        permute_81 = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        permute_82 = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_80, permute_81, permute_82, None, True, 0.0, True);  permute_80 = permute_81 = permute_82 = None
        getitem_367 = _scaled_dot_product_cudnn_attention_7[0]
        getitem_368 = _scaled_dot_product_cudnn_attention_7[1]
        getitem_373 = _scaled_dot_product_cudnn_attention_7[6]
        getitem_374 = _scaled_dot_product_cudnn_attention_7[7];  _scaled_dot_product_cudnn_attention_7 = None
        permute_83 = torch.ops.aten.permute.default(getitem_367, [0, 2, 1, 3])
        view_546 = torch.ops.aten.view.default(permute_83, [2, 8192, -1]);  permute_83 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16)
        all_gather_into_tensor_83 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_248, 32, '0');  convert_element_type_248 = None
        wait_tensor_98 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_83);  all_gather_into_tensor_83 = None
        permute_84 = torch.ops.aten.permute.default(wait_tensor_98, [1, 0]);  wait_tensor_98 = None
        view_552 = torch.ops.aten.view.default(view_546, [16384, 512]);  view_546 = None
        mm_52 = torch.ops.aten.mm.default(view_552, permute_84);  view_552 = permute_84 = None
        view_553 = torch.ops.aten.view.default(mm_52, [2, 8192, 4096]);  mm_52 = None
        split_38 = torch.ops.aten.split.Tensor(view_553, 1024, 1);  view_553 = None
        getitem_376 = split_38[0]
        getitem_377 = split_38[1]
        getitem_378 = split_38[2]
        getitem_379 = split_38[3]
        getitem_380 = split_38[4]
        getitem_381 = split_38[5]
        getitem_382 = split_38[6]
        getitem_383 = split_38[7];  split_38 = None
        cat_30 = torch.ops.aten.cat.default([getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383]);  getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = None
        reduce_scatter_tensor_15 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_30, 'sum', 8, '1');  cat_30 = None
        wait_tensor_99 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_15)
        add_29 = torch.ops.aten.add.Tensor(add_27, wait_tensor_99);  wait_tensor_99 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16)
        all_gather_into_tensor_84 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_251, 32, '0');  convert_element_type_251 = None
        wait_tensor_100 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_84);  all_gather_into_tensor_84 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [2], True);  pow_16 = None
        add_30 = torch.ops.aten.add.Scalar(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_15);  convert_element_type_252 = rsqrt_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, wait_tensor_100);  mul_60 = wait_tensor_100 = None
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        all_gather_into_tensor_85 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_253, 8, '1');  convert_element_type_253 = None
        wait_tensor_101 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_85);  all_gather_into_tensor_85 = None
        split_39 = torch.ops.aten.split.Tensor(wait_tensor_101, 2);  wait_tensor_101 = None
        getitem_384 = split_39[0]
        getitem_385 = split_39[1]
        getitem_386 = split_39[2]
        getitem_387 = split_39[3]
        getitem_388 = split_39[4]
        getitem_389 = split_39[5]
        getitem_390 = split_39[6]
        getitem_391 = split_39[7];  split_39 = None
        cat_31 = torch.ops.aten.cat.default([getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391], 1);  getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16)
        all_gather_into_tensor_86 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_254, 32, '0');  convert_element_type_254 = None
        wait_tensor_102 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_86);  all_gather_into_tensor_86 = None
        permute_85 = torch.ops.aten.permute.default(wait_tensor_102, [1, 0]);  wait_tensor_102 = None
        view_564 = torch.ops.aten.view.default(cat_31, [16384, 4096]);  cat_31 = None
        mm_53 = torch.ops.aten.mm.default(view_564, permute_85);  permute_85 = None
        view_565 = torch.ops.aten.view.default(mm_53, [2, 8192, 1792])
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(view_565, torch.float32);  view_565 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_7);  convert_element_type_257 = sigmoid_7 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16)
        all_gather_into_tensor_87 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_259, 32, '0');  convert_element_type_259 = None
        wait_tensor_103 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_87);  all_gather_into_tensor_87 = None
        permute_86 = torch.ops.aten.permute.default(wait_tensor_103, [1, 0]);  wait_tensor_103 = None
        mm_54 = torch.ops.aten.mm.default(view_564, permute_86);  view_564 = permute_86 = None
        view_572 = torch.ops.aten.view.default(mm_54, [2, 8192, 1792]);  mm_54 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_258, view_572);  convert_element_type_258 = view_572 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16)
        all_gather_into_tensor_88 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_262, 32, '0');  convert_element_type_262 = None
        wait_tensor_104 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_88);  all_gather_into_tensor_88 = None
        permute_87 = torch.ops.aten.permute.default(wait_tensor_104, [1, 0]);  wait_tensor_104 = None
        view_579 = torch.ops.aten.view.default(mul_63, [16384, 1792]);  mul_63 = None
        mm_55 = torch.ops.aten.mm.default(view_579, permute_87);  view_579 = permute_87 = None
        view_580 = torch.ops.aten.view.default(mm_55, [2, 8192, 4096]);  mm_55 = None
        split_40 = torch.ops.aten.split.Tensor(view_580, 1024, 1);  view_580 = None
        getitem_392 = split_40[0]
        getitem_393 = split_40[1]
        getitem_394 = split_40[2]
        getitem_395 = split_40[3]
        getitem_396 = split_40[4]
        getitem_397 = split_40[5]
        getitem_398 = split_40[6]
        getitem_399 = split_40[7];  split_40 = None
        cat_32 = torch.ops.aten.cat.default([getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399]);  getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = None
        reduce_scatter_tensor_16 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_32, 'sum', 8, '1');  cat_32 = None
        wait_tensor_105 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_16);  reduce_scatter_tensor_16 = None
        add_31 = torch.ops.aten.add.Tensor(add_29, wait_tensor_105);  add_29 = wait_tensor_105 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16)
        all_gather_into_tensor_89 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_265, 32, '0');  convert_element_type_265 = None
        wait_tensor_106 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_89);  all_gather_into_tensor_89 = None
        convert_element_type_266 = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_32 = torch.ops.aten.add.Scalar(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_16);  convert_element_type_266 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, wait_tensor_106);  mul_64 = wait_tensor_106 = None
        convert_element_type_267 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        all_gather_into_tensor_90 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_267, 8, '1');  convert_element_type_267 = None
        wait_tensor_107 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_90);  all_gather_into_tensor_90 = None
        split_41 = torch.ops.aten.split.Tensor(wait_tensor_107, 2);  wait_tensor_107 = None
        getitem_400 = split_41[0]
        getitem_401 = split_41[1]
        getitem_402 = split_41[2]
        getitem_403 = split_41[3]
        getitem_404 = split_41[4]
        getitem_405 = split_41[5]
        getitem_406 = split_41[6]
        getitem_407 = split_41[7];  split_41 = None
        cat_33 = torch.ops.aten.cat.default([getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407], 1);  getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16)
        all_gather_into_tensor_91 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_268, 32, '0');  convert_element_type_268 = None
        wait_tensor_108 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_91);  all_gather_into_tensor_91 = None
        permute_88 = torch.ops.aten.permute.default(wait_tensor_108, [1, 0]);  wait_tensor_108 = None
        view_591 = torch.ops.aten.view.default(cat_33, [16384, 4096]);  cat_33 = None
        mm_56 = torch.ops.aten.mm.default(view_591, permute_88);  permute_88 = None
        view_592 = torch.ops.aten.view.default(mm_56, [2, 8192, 512])
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16)
        all_gather_into_tensor_92 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_271, 32, '0');  convert_element_type_271 = None
        wait_tensor_109 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_92);  all_gather_into_tensor_92 = None
        permute_89 = torch.ops.aten.permute.default(wait_tensor_109, [1, 0]);  wait_tensor_109 = None
        mm_57 = torch.ops.aten.mm.default(view_591, permute_89);  permute_89 = None
        view_599 = torch.ops.aten.view.default(mm_57, [2, 8192, 128]);  mm_57 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16)
        all_gather_into_tensor_93 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_274, 32, '0');  convert_element_type_274 = None
        wait_tensor_110 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_93);  all_gather_into_tensor_93 = None
        permute_90 = torch.ops.aten.permute.default(wait_tensor_110, [1, 0]);  wait_tensor_110 = None
        mm_58 = torch.ops.aten.mm.default(view_591, permute_90);  view_591 = permute_90 = None
        view_606 = torch.ops.aten.view.default(mm_58, [2, 8192, 128])
        view_608 = torch.ops.aten.view.default(view_592, [2, 8192, -1, 128]);  view_592 = None
        view_609 = torch.ops.aten.view.default(view_599, [2, 8192, -1, 128]);  view_599 = None
        view_610 = torch.ops.aten.view.default(view_606, [2, 8192, -1, 128]);  view_606 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(view_608, torch.float32);  view_608 = None
        view_611 = torch.ops.aten.view.default(convert_element_type_277, [2, 8192, 4, -1, 2]);  convert_element_type_277 = None
        view_as_complex_16 = torch.ops.aten.view_as_complex.default(view_611);  view_611 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(view_609, torch.float32);  view_609 = None
        view_612 = torch.ops.aten.view.default(convert_element_type_278, [2, 8192, 1, -1, 2]);  convert_element_type_278 = None
        view_as_complex_17 = torch.ops.aten.view_as_complex.default(view_612);  view_612 = None
        mul_66 = torch.ops.aten.mul.Tensor(view_as_complex_16, view_37);  view_as_complex_16 = None
        view_as_real_16 = torch.ops.aten.view_as_real.default(mul_66);  mul_66 = None
        view_614 = torch.ops.aten.view.default(view_as_real_16, [2, 8192, 4, 128]);  view_as_real_16 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_as_complex_17, view_37);  view_as_complex_17 = None
        view_as_real_17 = torch.ops.aten.view_as_real.default(mul_67);  mul_67 = None
        view_615 = torch.ops.aten.view.default(view_as_real_17, [2, 8192, 1, 128]);  view_as_real_17 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(view_614, torch.bfloat16);  view_614 = None
        convert_element_type_280 = torch.ops.prims.convert_element_type.default(view_615, torch.bfloat16);  view_615 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3);  convert_element_type_280 = None
        expand_16 = torch.ops.aten.expand.default(unsqueeze_16, [2, 8192, 1, 4, 128]);  unsqueeze_16 = None
        view_616 = torch.ops.aten.view.default(expand_16, [2, 8192, 4, 128]);  expand_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_610, 3);  view_610 = None
        expand_17 = torch.ops.aten.expand.default(unsqueeze_17, [2, 8192, 1, 4, 128]);  unsqueeze_17 = None
        view_617 = torch.ops.aten.view.default(expand_17, [2, 8192, 4, 128]);  expand_17 = None
        permute_91 = torch.ops.aten.permute.default(convert_element_type_279, [0, 2, 1, 3]);  convert_element_type_279 = None
        permute_92 = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        permute_93 = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_91, permute_92, permute_93, None, True, 0.0, True);  permute_91 = permute_92 = permute_93 = None
        getitem_408 = _scaled_dot_product_cudnn_attention_8[0]
        getitem_409 = _scaled_dot_product_cudnn_attention_8[1]
        getitem_414 = _scaled_dot_product_cudnn_attention_8[6]
        getitem_415 = _scaled_dot_product_cudnn_attention_8[7];  _scaled_dot_product_cudnn_attention_8 = None
        permute_94 = torch.ops.aten.permute.default(getitem_408, [0, 2, 1, 3])
        view_618 = torch.ops.aten.view.default(permute_94, [2, 8192, -1]);  permute_94 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16)
        all_gather_into_tensor_94 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_281, 32, '0');  convert_element_type_281 = None
        wait_tensor_111 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_94);  all_gather_into_tensor_94 = None
        permute_95 = torch.ops.aten.permute.default(wait_tensor_111, [1, 0]);  wait_tensor_111 = None
        view_624 = torch.ops.aten.view.default(view_618, [16384, 512]);  view_618 = None
        mm_59 = torch.ops.aten.mm.default(view_624, permute_95);  view_624 = permute_95 = None
        view_625 = torch.ops.aten.view.default(mm_59, [2, 8192, 4096]);  mm_59 = None
        split_42 = torch.ops.aten.split.Tensor(view_625, 1024, 1);  view_625 = None
        getitem_417 = split_42[0]
        getitem_418 = split_42[1]
        getitem_419 = split_42[2]
        getitem_420 = split_42[3]
        getitem_421 = split_42[4]
        getitem_422 = split_42[5]
        getitem_423 = split_42[6]
        getitem_424 = split_42[7];  split_42 = None
        cat_34 = torch.ops.aten.cat.default([getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424]);  getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = None
        reduce_scatter_tensor_17 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_34, 'sum', 8, '1');  cat_34 = None
        wait_tensor_112 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_17)
        add_33 = torch.ops.aten.add.Tensor(add_31, wait_tensor_112);  wait_tensor_112 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16)
        all_gather_into_tensor_95 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_284, 32, '0');  convert_element_type_284 = None
        wait_tensor_113 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_95);  all_gather_into_tensor_95 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [2], True);  pow_18 = None
        add_34 = torch.ops.aten.add.Scalar(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_285, rsqrt_17);  convert_element_type_285 = rsqrt_17 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, wait_tensor_113);  mul_68 = wait_tensor_113 = None
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        all_gather_into_tensor_96 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_286, 8, '1');  convert_element_type_286 = None
        wait_tensor_114 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_96);  all_gather_into_tensor_96 = None
        split_43 = torch.ops.aten.split.Tensor(wait_tensor_114, 2);  wait_tensor_114 = None
        getitem_425 = split_43[0]
        getitem_426 = split_43[1]
        getitem_427 = split_43[2]
        getitem_428 = split_43[3]
        getitem_429 = split_43[4]
        getitem_430 = split_43[5]
        getitem_431 = split_43[6]
        getitem_432 = split_43[7];  split_43 = None
        cat_35 = torch.ops.aten.cat.default([getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432], 1);  getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16)
        all_gather_into_tensor_97 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_287, 32, '0');  convert_element_type_287 = None
        wait_tensor_115 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_97);  all_gather_into_tensor_97 = None
        permute_96 = torch.ops.aten.permute.default(wait_tensor_115, [1, 0]);  wait_tensor_115 = None
        view_636 = torch.ops.aten.view.default(cat_35, [16384, 4096]);  cat_35 = None
        mm_60 = torch.ops.aten.mm.default(view_636, permute_96);  permute_96 = None
        view_637 = torch.ops.aten.view.default(mm_60, [2, 8192, 1792])
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(view_637, torch.float32);  view_637 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_290)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_290, sigmoid_8);  convert_element_type_290 = sigmoid_8 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16)
        all_gather_into_tensor_98 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_292, 32, '0');  convert_element_type_292 = None
        wait_tensor_116 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_98);  all_gather_into_tensor_98 = None
        permute_97 = torch.ops.aten.permute.default(wait_tensor_116, [1, 0]);  wait_tensor_116 = None
        mm_61 = torch.ops.aten.mm.default(view_636, permute_97);  view_636 = permute_97 = None
        view_644 = torch.ops.aten.view.default(mm_61, [2, 8192, 1792]);  mm_61 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_291, view_644);  convert_element_type_291 = view_644 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16)
        all_gather_into_tensor_99 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_295, 32, '0');  convert_element_type_295 = None
        wait_tensor_117 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_99);  all_gather_into_tensor_99 = None
        permute_98 = torch.ops.aten.permute.default(wait_tensor_117, [1, 0]);  wait_tensor_117 = None
        view_651 = torch.ops.aten.view.default(mul_71, [16384, 1792]);  mul_71 = None
        mm_62 = torch.ops.aten.mm.default(view_651, permute_98);  view_651 = permute_98 = None
        view_652 = torch.ops.aten.view.default(mm_62, [2, 8192, 4096]);  mm_62 = None
        split_44 = torch.ops.aten.split.Tensor(view_652, 1024, 1);  view_652 = None
        getitem_433 = split_44[0]
        getitem_434 = split_44[1]
        getitem_435 = split_44[2]
        getitem_436 = split_44[3]
        getitem_437 = split_44[4]
        getitem_438 = split_44[5]
        getitem_439 = split_44[6]
        getitem_440 = split_44[7];  split_44 = None
        cat_36 = torch.ops.aten.cat.default([getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440]);  getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = None
        reduce_scatter_tensor_18 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_36, 'sum', 8, '1');  cat_36 = None
        wait_tensor_118 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_18);  reduce_scatter_tensor_18 = None
        add_35 = torch.ops.aten.add.Tensor(add_33, wait_tensor_118);  add_33 = wait_tensor_118 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16)
        all_gather_into_tensor_100 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_298, 32, '0');  convert_element_type_298 = None
        wait_tensor_119 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_100);  all_gather_into_tensor_100 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [2], True);  pow_19 = None
        add_36 = torch.ops.aten.add.Scalar(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_18);  convert_element_type_299 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, wait_tensor_119);  mul_72 = wait_tensor_119 = None
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        all_gather_into_tensor_101 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_300, 8, '1');  convert_element_type_300 = None
        wait_tensor_120 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_101);  all_gather_into_tensor_101 = None
        split_45 = torch.ops.aten.split.Tensor(wait_tensor_120, 2);  wait_tensor_120 = None
        getitem_441 = split_45[0]
        getitem_442 = split_45[1]
        getitem_443 = split_45[2]
        getitem_444 = split_45[3]
        getitem_445 = split_45[4]
        getitem_446 = split_45[5]
        getitem_447 = split_45[6]
        getitem_448 = split_45[7];  split_45 = None
        cat_37 = torch.ops.aten.cat.default([getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448], 1);  getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16)
        all_gather_into_tensor_102 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_301, 32, '0');  convert_element_type_301 = None
        wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_102);  all_gather_into_tensor_102 = None
        permute_99 = torch.ops.aten.permute.default(wait_tensor_121, [1, 0]);  wait_tensor_121 = None
        view_663 = torch.ops.aten.view.default(cat_37, [16384, 4096]);  cat_37 = None
        mm_63 = torch.ops.aten.mm.default(view_663, permute_99);  permute_99 = None
        view_664 = torch.ops.aten.view.default(mm_63, [2, 8192, 512])
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16)
        all_gather_into_tensor_103 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_304, 32, '0');  convert_element_type_304 = None
        wait_tensor_122 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_103);  all_gather_into_tensor_103 = None
        permute_100 = torch.ops.aten.permute.default(wait_tensor_122, [1, 0]);  wait_tensor_122 = None
        mm_64 = torch.ops.aten.mm.default(view_663, permute_100);  permute_100 = None
        view_671 = torch.ops.aten.view.default(mm_64, [2, 8192, 128]);  mm_64 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16)
        all_gather_into_tensor_104 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_307, 32, '0');  convert_element_type_307 = None
        wait_tensor_123 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_104);  all_gather_into_tensor_104 = None
        permute_101 = torch.ops.aten.permute.default(wait_tensor_123, [1, 0]);  wait_tensor_123 = None
        mm_65 = torch.ops.aten.mm.default(view_663, permute_101);  view_663 = permute_101 = None
        view_678 = torch.ops.aten.view.default(mm_65, [2, 8192, 128])
        view_680 = torch.ops.aten.view.default(view_664, [2, 8192, -1, 128]);  view_664 = None
        view_681 = torch.ops.aten.view.default(view_671, [2, 8192, -1, 128]);  view_671 = None
        view_682 = torch.ops.aten.view.default(view_678, [2, 8192, -1, 128]);  view_678 = None
        convert_element_type_310 = torch.ops.prims.convert_element_type.default(view_680, torch.float32);  view_680 = None
        view_683 = torch.ops.aten.view.default(convert_element_type_310, [2, 8192, 4, -1, 2]);  convert_element_type_310 = None
        view_as_complex_18 = torch.ops.aten.view_as_complex.default(view_683);  view_683 = None
        convert_element_type_311 = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        view_684 = torch.ops.aten.view.default(convert_element_type_311, [2, 8192, 1, -1, 2]);  convert_element_type_311 = None
        view_as_complex_19 = torch.ops.aten.view_as_complex.default(view_684);  view_684 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_as_complex_18, view_37);  view_as_complex_18 = None
        view_as_real_18 = torch.ops.aten.view_as_real.default(mul_74);  mul_74 = None
        view_686 = torch.ops.aten.view.default(view_as_real_18, [2, 8192, 4, 128]);  view_as_real_18 = None
        mul_75 = torch.ops.aten.mul.Tensor(view_as_complex_19, view_37);  view_as_complex_19 = None
        view_as_real_19 = torch.ops.aten.view_as_real.default(mul_75);  mul_75 = None
        view_687 = torch.ops.aten.view.default(view_as_real_19, [2, 8192, 1, 128]);  view_as_real_19 = None
        convert_element_type_312 = torch.ops.prims.convert_element_type.default(view_686, torch.bfloat16);  view_686 = None
        convert_element_type_313 = torch.ops.prims.convert_element_type.default(view_687, torch.bfloat16);  view_687 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(convert_element_type_313, 3);  convert_element_type_313 = None
        expand_18 = torch.ops.aten.expand.default(unsqueeze_18, [2, 8192, 1, 4, 128]);  unsqueeze_18 = None
        view_688 = torch.ops.aten.view.default(expand_18, [2, 8192, 4, 128]);  expand_18 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(view_682, 3);  view_682 = None
        expand_19 = torch.ops.aten.expand.default(unsqueeze_19, [2, 8192, 1, 4, 128]);  unsqueeze_19 = None
        view_689 = torch.ops.aten.view.default(expand_19, [2, 8192, 4, 128]);  expand_19 = None
        permute_102 = torch.ops.aten.permute.default(convert_element_type_312, [0, 2, 1, 3]);  convert_element_type_312 = None
        permute_103 = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        permute_104 = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_102, permute_103, permute_104, None, True, 0.0, True);  permute_102 = permute_103 = permute_104 = None
        getitem_449 = _scaled_dot_product_cudnn_attention_9[0]
        getitem_450 = _scaled_dot_product_cudnn_attention_9[1]
        getitem_455 = _scaled_dot_product_cudnn_attention_9[6]
        getitem_456 = _scaled_dot_product_cudnn_attention_9[7];  _scaled_dot_product_cudnn_attention_9 = None
        permute_105 = torch.ops.aten.permute.default(getitem_449, [0, 2, 1, 3])
        view_690 = torch.ops.aten.view.default(permute_105, [2, 8192, -1]);  permute_105 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16)
        all_gather_into_tensor_105 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_314, 32, '0');  convert_element_type_314 = None
        wait_tensor_124 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_105);  all_gather_into_tensor_105 = None
        permute_106 = torch.ops.aten.permute.default(wait_tensor_124, [1, 0]);  wait_tensor_124 = None
        view_696 = torch.ops.aten.view.default(view_690, [16384, 512]);  view_690 = None
        mm_66 = torch.ops.aten.mm.default(view_696, permute_106);  view_696 = permute_106 = None
        view_697 = torch.ops.aten.view.default(mm_66, [2, 8192, 4096]);  mm_66 = None
        split_46 = torch.ops.aten.split.Tensor(view_697, 1024, 1);  view_697 = None
        getitem_458 = split_46[0]
        getitem_459 = split_46[1]
        getitem_460 = split_46[2]
        getitem_461 = split_46[3]
        getitem_462 = split_46[4]
        getitem_463 = split_46[5]
        getitem_464 = split_46[6]
        getitem_465 = split_46[7];  split_46 = None
        cat_38 = torch.ops.aten.cat.default([getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465]);  getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = None
        reduce_scatter_tensor_19 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_38, 'sum', 8, '1');  cat_38 = None
        wait_tensor_125 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_19)
        add_37 = torch.ops.aten.add.Tensor(add_35, wait_tensor_125);  wait_tensor_125 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16)
        all_gather_into_tensor_106 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_317, 32, '0');  convert_element_type_317 = None
        wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_106);  all_gather_into_tensor_106 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_38 = torch.ops.aten.add.Scalar(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_19);  convert_element_type_318 = rsqrt_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, wait_tensor_126);  mul_76 = wait_tensor_126 = None
        convert_element_type_319 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        all_gather_into_tensor_107 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_319, 8, '1');  convert_element_type_319 = None
        wait_tensor_127 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_107);  all_gather_into_tensor_107 = None
        split_47 = torch.ops.aten.split.Tensor(wait_tensor_127, 2);  wait_tensor_127 = None
        getitem_466 = split_47[0]
        getitem_467 = split_47[1]
        getitem_468 = split_47[2]
        getitem_469 = split_47[3]
        getitem_470 = split_47[4]
        getitem_471 = split_47[5]
        getitem_472 = split_47[6]
        getitem_473 = split_47[7];  split_47 = None
        cat_39 = torch.ops.aten.cat.default([getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473], 1);  getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16)
        all_gather_into_tensor_108 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_320, 32, '0');  convert_element_type_320 = None
        wait_tensor_128 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_108);  all_gather_into_tensor_108 = None
        permute_107 = torch.ops.aten.permute.default(wait_tensor_128, [1, 0]);  wait_tensor_128 = None
        view_708 = torch.ops.aten.view.default(cat_39, [16384, 4096]);  cat_39 = None
        mm_67 = torch.ops.aten.mm.default(view_708, permute_107);  permute_107 = None
        view_709 = torch.ops.aten.view.default(mm_67, [2, 8192, 1792])
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_323)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_323, sigmoid_9);  convert_element_type_323 = sigmoid_9 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16)
        all_gather_into_tensor_109 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_325, 32, '0');  convert_element_type_325 = None
        wait_tensor_129 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_109);  all_gather_into_tensor_109 = None
        permute_108 = torch.ops.aten.permute.default(wait_tensor_129, [1, 0]);  wait_tensor_129 = None
        mm_68 = torch.ops.aten.mm.default(view_708, permute_108);  view_708 = permute_108 = None
        view_716 = torch.ops.aten.view.default(mm_68, [2, 8192, 1792]);  mm_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_324, view_716);  convert_element_type_324 = view_716 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16)
        all_gather_into_tensor_110 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_328, 32, '0');  convert_element_type_328 = None
        wait_tensor_130 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_110);  all_gather_into_tensor_110 = None
        permute_109 = torch.ops.aten.permute.default(wait_tensor_130, [1, 0]);  wait_tensor_130 = None
        view_723 = torch.ops.aten.view.default(mul_79, [16384, 1792]);  mul_79 = None
        mm_69 = torch.ops.aten.mm.default(view_723, permute_109);  view_723 = permute_109 = None
        view_724 = torch.ops.aten.view.default(mm_69, [2, 8192, 4096]);  mm_69 = None
        split_48 = torch.ops.aten.split.Tensor(view_724, 1024, 1);  view_724 = None
        getitem_474 = split_48[0]
        getitem_475 = split_48[1]
        getitem_476 = split_48[2]
        getitem_477 = split_48[3]
        getitem_478 = split_48[4]
        getitem_479 = split_48[5]
        getitem_480 = split_48[6]
        getitem_481 = split_48[7];  split_48 = None
        cat_40 = torch.ops.aten.cat.default([getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481]);  getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = None
        reduce_scatter_tensor_20 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_40, 'sum', 8, '1');  cat_40 = None
        wait_tensor_131 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_20);  reduce_scatter_tensor_20 = None
        add_39 = torch.ops.aten.add.Tensor(add_37, wait_tensor_131);  add_37 = wait_tensor_131 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16)
        all_gather_into_tensor_111 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_331, 32, '0');  convert_element_type_331 = None
        wait_tensor_132 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_111);  all_gather_into_tensor_111 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [2], True);  pow_21 = None
        add_40 = torch.ops.aten.add.Scalar(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_20);  convert_element_type_332 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, wait_tensor_132);  mul_80 = wait_tensor_132 = None
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        all_gather_into_tensor_112 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_333, 8, '1');  convert_element_type_333 = None
        wait_tensor_133 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_112);  all_gather_into_tensor_112 = None
        split_49 = torch.ops.aten.split.Tensor(wait_tensor_133, 2);  wait_tensor_133 = None
        getitem_482 = split_49[0]
        getitem_483 = split_49[1]
        getitem_484 = split_49[2]
        getitem_485 = split_49[3]
        getitem_486 = split_49[4]
        getitem_487 = split_49[5]
        getitem_488 = split_49[6]
        getitem_489 = split_49[7];  split_49 = None
        cat_41 = torch.ops.aten.cat.default([getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489], 1);  getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16)
        all_gather_into_tensor_113 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_334, 32, '0');  convert_element_type_334 = None
        wait_tensor_134 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_113);  all_gather_into_tensor_113 = None
        permute_110 = torch.ops.aten.permute.default(wait_tensor_134, [1, 0]);  wait_tensor_134 = None
        view_735 = torch.ops.aten.view.default(cat_41, [16384, 4096]);  cat_41 = None
        mm_70 = torch.ops.aten.mm.default(view_735, permute_110);  permute_110 = None
        view_736 = torch.ops.aten.view.default(mm_70, [2, 8192, 512])
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16)
        all_gather_into_tensor_114 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_337, 32, '0');  convert_element_type_337 = None
        wait_tensor_135 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_114);  all_gather_into_tensor_114 = None
        permute_111 = torch.ops.aten.permute.default(wait_tensor_135, [1, 0]);  wait_tensor_135 = None
        mm_71 = torch.ops.aten.mm.default(view_735, permute_111);  permute_111 = None
        view_743 = torch.ops.aten.view.default(mm_71, [2, 8192, 128]);  mm_71 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16)
        all_gather_into_tensor_115 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_340, 32, '0');  convert_element_type_340 = None
        wait_tensor_136 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_115);  all_gather_into_tensor_115 = None
        permute_112 = torch.ops.aten.permute.default(wait_tensor_136, [1, 0]);  wait_tensor_136 = None
        mm_72 = torch.ops.aten.mm.default(view_735, permute_112);  view_735 = permute_112 = None
        view_750 = torch.ops.aten.view.default(mm_72, [2, 8192, 128])
        view_752 = torch.ops.aten.view.default(view_736, [2, 8192, -1, 128]);  view_736 = None
        view_753 = torch.ops.aten.view.default(view_743, [2, 8192, -1, 128]);  view_743 = None
        view_754 = torch.ops.aten.view.default(view_750, [2, 8192, -1, 128]);  view_750 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(view_752, torch.float32);  view_752 = None
        view_755 = torch.ops.aten.view.default(convert_element_type_343, [2, 8192, 4, -1, 2]);  convert_element_type_343 = None
        view_as_complex_20 = torch.ops.aten.view_as_complex.default(view_755);  view_755 = None
        convert_element_type_344 = torch.ops.prims.convert_element_type.default(view_753, torch.float32);  view_753 = None
        view_756 = torch.ops.aten.view.default(convert_element_type_344, [2, 8192, 1, -1, 2]);  convert_element_type_344 = None
        view_as_complex_21 = torch.ops.aten.view_as_complex.default(view_756);  view_756 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_as_complex_20, view_37);  view_as_complex_20 = None
        view_as_real_20 = torch.ops.aten.view_as_real.default(mul_82);  mul_82 = None
        view_758 = torch.ops.aten.view.default(view_as_real_20, [2, 8192, 4, 128]);  view_as_real_20 = None
        mul_83 = torch.ops.aten.mul.Tensor(view_as_complex_21, view_37);  view_as_complex_21 = None
        view_as_real_21 = torch.ops.aten.view_as_real.default(mul_83);  mul_83 = None
        view_759 = torch.ops.aten.view.default(view_as_real_21, [2, 8192, 1, 128]);  view_as_real_21 = None
        convert_element_type_345 = torch.ops.prims.convert_element_type.default(view_758, torch.bfloat16);  view_758 = None
        convert_element_type_346 = torch.ops.prims.convert_element_type.default(view_759, torch.bfloat16);  view_759 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(convert_element_type_346, 3);  convert_element_type_346 = None
        expand_20 = torch.ops.aten.expand.default(unsqueeze_20, [2, 8192, 1, 4, 128]);  unsqueeze_20 = None
        view_760 = torch.ops.aten.view.default(expand_20, [2, 8192, 4, 128]);  expand_20 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(view_754, 3);  view_754 = None
        expand_21 = torch.ops.aten.expand.default(unsqueeze_21, [2, 8192, 1, 4, 128]);  unsqueeze_21 = None
        view_761 = torch.ops.aten.view.default(expand_21, [2, 8192, 4, 128]);  expand_21 = None
        permute_113 = torch.ops.aten.permute.default(convert_element_type_345, [0, 2, 1, 3]);  convert_element_type_345 = None
        permute_114 = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
        permute_115 = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_113, permute_114, permute_115, None, True, 0.0, True);  permute_113 = permute_114 = permute_115 = None
        getitem_490 = _scaled_dot_product_cudnn_attention_10[0]
        getitem_491 = _scaled_dot_product_cudnn_attention_10[1]
        getitem_496 = _scaled_dot_product_cudnn_attention_10[6]
        getitem_497 = _scaled_dot_product_cudnn_attention_10[7];  _scaled_dot_product_cudnn_attention_10 = None
        permute_116 = torch.ops.aten.permute.default(getitem_490, [0, 2, 1, 3])
        view_762 = torch.ops.aten.view.default(permute_116, [2, 8192, -1]);  permute_116 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16)
        all_gather_into_tensor_116 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_347, 32, '0');  convert_element_type_347 = None
        wait_tensor_137 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_116);  all_gather_into_tensor_116 = None
        permute_117 = torch.ops.aten.permute.default(wait_tensor_137, [1, 0]);  wait_tensor_137 = None
        view_768 = torch.ops.aten.view.default(view_762, [16384, 512]);  view_762 = None
        mm_73 = torch.ops.aten.mm.default(view_768, permute_117);  view_768 = permute_117 = None
        view_769 = torch.ops.aten.view.default(mm_73, [2, 8192, 4096]);  mm_73 = None
        split_50 = torch.ops.aten.split.Tensor(view_769, 1024, 1);  view_769 = None
        getitem_499 = split_50[0]
        getitem_500 = split_50[1]
        getitem_501 = split_50[2]
        getitem_502 = split_50[3]
        getitem_503 = split_50[4]
        getitem_504 = split_50[5]
        getitem_505 = split_50[6]
        getitem_506 = split_50[7];  split_50 = None
        cat_42 = torch.ops.aten.cat.default([getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506]);  getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_504 = getitem_505 = getitem_506 = None
        reduce_scatter_tensor_21 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_42, 'sum', 8, '1');  cat_42 = None
        wait_tensor_138 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_21)
        add_41 = torch.ops.aten.add.Tensor(add_39, wait_tensor_138);  wait_tensor_138 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16)
        all_gather_into_tensor_117 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_350, 32, '0');  convert_element_type_350 = None
        wait_tensor_139 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_117);  all_gather_into_tensor_117 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(add_41, torch.float32)
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_42 = torch.ops.aten.add.Scalar(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_21);  convert_element_type_351 = rsqrt_21 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, wait_tensor_139);  mul_84 = wait_tensor_139 = None
        convert_element_type_352 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        all_gather_into_tensor_118 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_352, 8, '1');  convert_element_type_352 = None
        wait_tensor_140 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_118);  all_gather_into_tensor_118 = None
        split_51 = torch.ops.aten.split.Tensor(wait_tensor_140, 2);  wait_tensor_140 = None
        getitem_507 = split_51[0]
        getitem_508 = split_51[1]
        getitem_509 = split_51[2]
        getitem_510 = split_51[3]
        getitem_511 = split_51[4]
        getitem_512 = split_51[5]
        getitem_513 = split_51[6]
        getitem_514 = split_51[7];  split_51 = None
        cat_43 = torch.ops.aten.cat.default([getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514], 1);  getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16)
        all_gather_into_tensor_119 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_353, 32, '0');  convert_element_type_353 = None
        wait_tensor_141 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_119);  all_gather_into_tensor_119 = None
        permute_118 = torch.ops.aten.permute.default(wait_tensor_141, [1, 0]);  wait_tensor_141 = None
        view_780 = torch.ops.aten.view.default(cat_43, [16384, 4096]);  cat_43 = None
        mm_74 = torch.ops.aten.mm.default(view_780, permute_118);  permute_118 = None
        view_781 = torch.ops.aten.view.default(mm_74, [2, 8192, 1792])
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_356)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_356, sigmoid_10);  convert_element_type_356 = sigmoid_10 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16)
        all_gather_into_tensor_120 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_358, 32, '0');  convert_element_type_358 = None
        wait_tensor_142 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_120);  all_gather_into_tensor_120 = None
        permute_119 = torch.ops.aten.permute.default(wait_tensor_142, [1, 0]);  wait_tensor_142 = None
        mm_75 = torch.ops.aten.mm.default(view_780, permute_119);  view_780 = permute_119 = None
        view_788 = torch.ops.aten.view.default(mm_75, [2, 8192, 1792]);  mm_75 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_357, view_788);  convert_element_type_357 = view_788 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16)
        all_gather_into_tensor_121 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_361, 32, '0');  convert_element_type_361 = None
        wait_tensor_143 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_121);  all_gather_into_tensor_121 = None
        permute_120 = torch.ops.aten.permute.default(wait_tensor_143, [1, 0]);  wait_tensor_143 = None
        view_795 = torch.ops.aten.view.default(mul_87, [16384, 1792]);  mul_87 = None
        mm_76 = torch.ops.aten.mm.default(view_795, permute_120);  view_795 = permute_120 = None
        view_796 = torch.ops.aten.view.default(mm_76, [2, 8192, 4096]);  mm_76 = None
        split_52 = torch.ops.aten.split.Tensor(view_796, 1024, 1);  view_796 = None
        getitem_515 = split_52[0]
        getitem_516 = split_52[1]
        getitem_517 = split_52[2]
        getitem_518 = split_52[3]
        getitem_519 = split_52[4]
        getitem_520 = split_52[5]
        getitem_521 = split_52[6]
        getitem_522 = split_52[7];  split_52 = None
        cat_44 = torch.ops.aten.cat.default([getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522]);  getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = None
        reduce_scatter_tensor_22 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_44, 'sum', 8, '1');  cat_44 = None
        wait_tensor_144 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_22);  reduce_scatter_tensor_22 = None
        add_43 = torch.ops.aten.add.Tensor(add_41, wait_tensor_144);  add_41 = wait_tensor_144 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16)
        all_gather_into_tensor_122 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_364, 32, '0');  convert_element_type_364 = None
        wait_tensor_145 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_122);  all_gather_into_tensor_122 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_365, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [2], True);  pow_23 = None
        add_44 = torch.ops.aten.add.Scalar(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_365, rsqrt_22);  convert_element_type_365 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, wait_tensor_145);  mul_88 = wait_tensor_145 = None
        convert_element_type_366 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        all_gather_into_tensor_123 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_366, 8, '1');  convert_element_type_366 = None
        wait_tensor_146 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_123);  all_gather_into_tensor_123 = None
        split_53 = torch.ops.aten.split.Tensor(wait_tensor_146, 2);  wait_tensor_146 = None
        getitem_523 = split_53[0]
        getitem_524 = split_53[1]
        getitem_525 = split_53[2]
        getitem_526 = split_53[3]
        getitem_527 = split_53[4]
        getitem_528 = split_53[5]
        getitem_529 = split_53[6]
        getitem_530 = split_53[7];  split_53 = None
        cat_45 = torch.ops.aten.cat.default([getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530], 1);  getitem_523 = getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16)
        all_gather_into_tensor_124 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_367, 32, '0');  convert_element_type_367 = None
        wait_tensor_147 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_124);  all_gather_into_tensor_124 = None
        permute_121 = torch.ops.aten.permute.default(wait_tensor_147, [1, 0]);  wait_tensor_147 = None
        view_807 = torch.ops.aten.view.default(cat_45, [16384, 4096]);  cat_45 = None
        mm_77 = torch.ops.aten.mm.default(view_807, permute_121);  permute_121 = None
        view_808 = torch.ops.aten.view.default(mm_77, [2, 8192, 512])
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16)
        all_gather_into_tensor_125 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_370, 32, '0');  convert_element_type_370 = None
        wait_tensor_148 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_125);  all_gather_into_tensor_125 = None
        permute_122 = torch.ops.aten.permute.default(wait_tensor_148, [1, 0]);  wait_tensor_148 = None
        mm_78 = torch.ops.aten.mm.default(view_807, permute_122);  permute_122 = None
        view_815 = torch.ops.aten.view.default(mm_78, [2, 8192, 128]);  mm_78 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16)
        all_gather_into_tensor_126 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_373, 32, '0');  convert_element_type_373 = None
        wait_tensor_149 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_126);  all_gather_into_tensor_126 = None
        permute_123 = torch.ops.aten.permute.default(wait_tensor_149, [1, 0]);  wait_tensor_149 = None
        mm_79 = torch.ops.aten.mm.default(view_807, permute_123);  view_807 = permute_123 = None
        view_822 = torch.ops.aten.view.default(mm_79, [2, 8192, 128])
        view_824 = torch.ops.aten.view.default(view_808, [2, 8192, -1, 128]);  view_808 = None
        view_825 = torch.ops.aten.view.default(view_815, [2, 8192, -1, 128]);  view_815 = None
        view_826 = torch.ops.aten.view.default(view_822, [2, 8192, -1, 128]);  view_822 = None
        convert_element_type_376 = torch.ops.prims.convert_element_type.default(view_824, torch.float32);  view_824 = None
        view_827 = torch.ops.aten.view.default(convert_element_type_376, [2, 8192, 4, -1, 2]);  convert_element_type_376 = None
        view_as_complex_22 = torch.ops.aten.view_as_complex.default(view_827);  view_827 = None
        convert_element_type_377 = torch.ops.prims.convert_element_type.default(view_825, torch.float32);  view_825 = None
        view_828 = torch.ops.aten.view.default(convert_element_type_377, [2, 8192, 1, -1, 2]);  convert_element_type_377 = None
        view_as_complex_23 = torch.ops.aten.view_as_complex.default(view_828);  view_828 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_as_complex_22, view_37);  view_as_complex_22 = None
        view_as_real_22 = torch.ops.aten.view_as_real.default(mul_90);  mul_90 = None
        view_830 = torch.ops.aten.view.default(view_as_real_22, [2, 8192, 4, 128]);  view_as_real_22 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_as_complex_23, view_37);  view_as_complex_23 = None
        view_as_real_23 = torch.ops.aten.view_as_real.default(mul_91);  mul_91 = None
        view_831 = torch.ops.aten.view.default(view_as_real_23, [2, 8192, 1, 128]);  view_as_real_23 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(view_830, torch.bfloat16);  view_830 = None
        convert_element_type_379 = torch.ops.prims.convert_element_type.default(view_831, torch.bfloat16);  view_831 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(convert_element_type_379, 3);  convert_element_type_379 = None
        expand_22 = torch.ops.aten.expand.default(unsqueeze_22, [2, 8192, 1, 4, 128]);  unsqueeze_22 = None
        view_832 = torch.ops.aten.view.default(expand_22, [2, 8192, 4, 128]);  expand_22 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(view_826, 3);  view_826 = None
        expand_23 = torch.ops.aten.expand.default(unsqueeze_23, [2, 8192, 1, 4, 128]);  unsqueeze_23 = None
        view_833 = torch.ops.aten.view.default(expand_23, [2, 8192, 4, 128]);  expand_23 = None
        permute_124 = torch.ops.aten.permute.default(convert_element_type_378, [0, 2, 1, 3]);  convert_element_type_378 = None
        permute_125 = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
        permute_126 = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_124, permute_125, permute_126, None, True, 0.0, True);  permute_124 = permute_125 = permute_126 = None
        getitem_531 = _scaled_dot_product_cudnn_attention_11[0]
        getitem_532 = _scaled_dot_product_cudnn_attention_11[1]
        getitem_537 = _scaled_dot_product_cudnn_attention_11[6]
        getitem_538 = _scaled_dot_product_cudnn_attention_11[7];  _scaled_dot_product_cudnn_attention_11 = None
        permute_127 = torch.ops.aten.permute.default(getitem_531, [0, 2, 1, 3])
        view_834 = torch.ops.aten.view.default(permute_127, [2, 8192, -1]);  permute_127 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16)
        all_gather_into_tensor_127 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_380, 32, '0');  convert_element_type_380 = None
        wait_tensor_150 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_127);  all_gather_into_tensor_127 = None
        permute_128 = torch.ops.aten.permute.default(wait_tensor_150, [1, 0]);  wait_tensor_150 = None
        view_840 = torch.ops.aten.view.default(view_834, [16384, 512]);  view_834 = None
        mm_80 = torch.ops.aten.mm.default(view_840, permute_128);  view_840 = permute_128 = None
        view_841 = torch.ops.aten.view.default(mm_80, [2, 8192, 4096]);  mm_80 = None
        split_54 = torch.ops.aten.split.Tensor(view_841, 1024, 1);  view_841 = None
        getitem_540 = split_54[0]
        getitem_541 = split_54[1]
        getitem_542 = split_54[2]
        getitem_543 = split_54[3]
        getitem_544 = split_54[4]
        getitem_545 = split_54[5]
        getitem_546 = split_54[6]
        getitem_547 = split_54[7];  split_54 = None
        cat_46 = torch.ops.aten.cat.default([getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547]);  getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = None
        reduce_scatter_tensor_23 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_46, 'sum', 8, '1');  cat_46 = None
        wait_tensor_151 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_23)
        add_45 = torch.ops.aten.add.Tensor(add_43, wait_tensor_151);  wait_tensor_151 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16)
        all_gather_into_tensor_128 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_383, 32, '0');  convert_element_type_383 = None
        wait_tensor_152 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_128);  all_gather_into_tensor_128 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [2], True);  pow_24 = None
        add_46 = torch.ops.aten.add.Scalar(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_23);  convert_element_type_384 = rsqrt_23 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, wait_tensor_152);  mul_92 = wait_tensor_152 = None
        convert_element_type_385 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        all_gather_into_tensor_129 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_385, 8, '1');  convert_element_type_385 = None
        wait_tensor_153 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_129);  all_gather_into_tensor_129 = None
        split_55 = torch.ops.aten.split.Tensor(wait_tensor_153, 2);  wait_tensor_153 = None
        getitem_548 = split_55[0]
        getitem_549 = split_55[1]
        getitem_550 = split_55[2]
        getitem_551 = split_55[3]
        getitem_552 = split_55[4]
        getitem_553 = split_55[5]
        getitem_554 = split_55[6]
        getitem_555 = split_55[7];  split_55 = None
        cat_47 = torch.ops.aten.cat.default([getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555], 1);  getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16)
        all_gather_into_tensor_130 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_386, 32, '0');  convert_element_type_386 = None
        wait_tensor_154 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_130);  all_gather_into_tensor_130 = None
        permute_129 = torch.ops.aten.permute.default(wait_tensor_154, [1, 0]);  wait_tensor_154 = None
        view_852 = torch.ops.aten.view.default(cat_47, [16384, 4096]);  cat_47 = None
        mm_81 = torch.ops.aten.mm.default(view_852, permute_129);  permute_129 = None
        view_853 = torch.ops.aten.view.default(mm_81, [2, 8192, 1792])
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(view_853, torch.float32);  view_853 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_389)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_389, sigmoid_11);  convert_element_type_389 = sigmoid_11 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16)
        all_gather_into_tensor_131 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_391, 32, '0');  convert_element_type_391 = None
        wait_tensor_155 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_131);  all_gather_into_tensor_131 = None
        permute_130 = torch.ops.aten.permute.default(wait_tensor_155, [1, 0]);  wait_tensor_155 = None
        mm_82 = torch.ops.aten.mm.default(view_852, permute_130);  view_852 = permute_130 = None
        view_860 = torch.ops.aten.view.default(mm_82, [2, 8192, 1792]);  mm_82 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_390, view_860);  convert_element_type_390 = view_860 = None
        convert_element_type_394 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16)
        all_gather_into_tensor_132 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_394, 32, '0');  convert_element_type_394 = None
        wait_tensor_156 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_132);  all_gather_into_tensor_132 = None
        permute_131 = torch.ops.aten.permute.default(wait_tensor_156, [1, 0]);  wait_tensor_156 = None
        view_867 = torch.ops.aten.view.default(mul_95, [16384, 1792]);  mul_95 = None
        mm_83 = torch.ops.aten.mm.default(view_867, permute_131);  view_867 = permute_131 = None
        view_868 = torch.ops.aten.view.default(mm_83, [2, 8192, 4096]);  mm_83 = None
        split_56 = torch.ops.aten.split.Tensor(view_868, 1024, 1);  view_868 = None
        getitem_556 = split_56[0]
        getitem_557 = split_56[1]
        getitem_558 = split_56[2]
        getitem_559 = split_56[3]
        getitem_560 = split_56[4]
        getitem_561 = split_56[5]
        getitem_562 = split_56[6]
        getitem_563 = split_56[7];  split_56 = None
        cat_48 = torch.ops.aten.cat.default([getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563]);  getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = None
        reduce_scatter_tensor_24 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_48, 'sum', 8, '1');  cat_48 = None
        wait_tensor_157 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_24);  reduce_scatter_tensor_24 = None
        add_47 = torch.ops.aten.add.Tensor(add_45, wait_tensor_157);  add_45 = wait_tensor_157 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16)
        all_gather_into_tensor_133 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_397, 32, '0');  convert_element_type_397 = None
        wait_tensor_158 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_133);  all_gather_into_tensor_133 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_48 = torch.ops.aten.add.Scalar(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_24);  convert_element_type_398 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, wait_tensor_158);  mul_96 = wait_tensor_158 = None
        convert_element_type_399 = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None
        all_gather_into_tensor_134 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_399, 8, '1');  convert_element_type_399 = None
        wait_tensor_159 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_134);  all_gather_into_tensor_134 = None
        split_57 = torch.ops.aten.split.Tensor(wait_tensor_159, 2);  wait_tensor_159 = None
        getitem_564 = split_57[0]
        getitem_565 = split_57[1]
        getitem_566 = split_57[2]
        getitem_567 = split_57[3]
        getitem_568 = split_57[4]
        getitem_569 = split_57[5]
        getitem_570 = split_57[6]
        getitem_571 = split_57[7];  split_57 = None
        cat_49 = torch.ops.aten.cat.default([getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571], 1);  getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16)
        all_gather_into_tensor_135 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_400, 32, '0');  convert_element_type_400 = None
        wait_tensor_160 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_135);  all_gather_into_tensor_135 = None
        permute_132 = torch.ops.aten.permute.default(wait_tensor_160, [1, 0]);  wait_tensor_160 = None
        view_879 = torch.ops.aten.view.default(cat_49, [16384, 4096]);  cat_49 = None
        mm_84 = torch.ops.aten.mm.default(view_879, permute_132);  permute_132 = None
        view_880 = torch.ops.aten.view.default(mm_84, [2, 8192, 512])
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16)
        all_gather_into_tensor_136 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_403, 32, '0');  convert_element_type_403 = None
        wait_tensor_161 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_136);  all_gather_into_tensor_136 = None
        permute_133 = torch.ops.aten.permute.default(wait_tensor_161, [1, 0]);  wait_tensor_161 = None
        mm_85 = torch.ops.aten.mm.default(view_879, permute_133);  permute_133 = None
        view_887 = torch.ops.aten.view.default(mm_85, [2, 8192, 128]);  mm_85 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16)
        all_gather_into_tensor_137 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_406, 32, '0');  convert_element_type_406 = None
        wait_tensor_162 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_137);  all_gather_into_tensor_137 = None
        permute_134 = torch.ops.aten.permute.default(wait_tensor_162, [1, 0]);  wait_tensor_162 = None
        mm_86 = torch.ops.aten.mm.default(view_879, permute_134);  view_879 = permute_134 = None
        view_894 = torch.ops.aten.view.default(mm_86, [2, 8192, 128])
        view_896 = torch.ops.aten.view.default(view_880, [2, 8192, -1, 128]);  view_880 = None
        view_897 = torch.ops.aten.view.default(view_887, [2, 8192, -1, 128]);  view_887 = None
        view_898 = torch.ops.aten.view.default(view_894, [2, 8192, -1, 128]);  view_894 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(view_896, torch.float32);  view_896 = None
        view_899 = torch.ops.aten.view.default(convert_element_type_409, [2, 8192, 4, -1, 2]);  convert_element_type_409 = None
        view_as_complex_24 = torch.ops.aten.view_as_complex.default(view_899);  view_899 = None
        convert_element_type_410 = torch.ops.prims.convert_element_type.default(view_897, torch.float32);  view_897 = None
        view_900 = torch.ops.aten.view.default(convert_element_type_410, [2, 8192, 1, -1, 2]);  convert_element_type_410 = None
        view_as_complex_25 = torch.ops.aten.view_as_complex.default(view_900);  view_900 = None
        mul_98 = torch.ops.aten.mul.Tensor(view_as_complex_24, view_37);  view_as_complex_24 = None
        view_as_real_24 = torch.ops.aten.view_as_real.default(mul_98);  mul_98 = None
        view_902 = torch.ops.aten.view.default(view_as_real_24, [2, 8192, 4, 128]);  view_as_real_24 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_as_complex_25, view_37);  view_as_complex_25 = None
        view_as_real_25 = torch.ops.aten.view_as_real.default(mul_99);  mul_99 = None
        view_903 = torch.ops.aten.view.default(view_as_real_25, [2, 8192, 1, 128]);  view_as_real_25 = None
        convert_element_type_411 = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(view_903, torch.bfloat16);  view_903 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(convert_element_type_412, 3);  convert_element_type_412 = None
        expand_24 = torch.ops.aten.expand.default(unsqueeze_24, [2, 8192, 1, 4, 128]);  unsqueeze_24 = None
        view_904 = torch.ops.aten.view.default(expand_24, [2, 8192, 4, 128]);  expand_24 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(view_898, 3);  view_898 = None
        expand_25 = torch.ops.aten.expand.default(unsqueeze_25, [2, 8192, 1, 4, 128]);  unsqueeze_25 = None
        view_905 = torch.ops.aten.view.default(expand_25, [2, 8192, 4, 128]);  expand_25 = None
        permute_135 = torch.ops.aten.permute.default(convert_element_type_411, [0, 2, 1, 3]);  convert_element_type_411 = None
        permute_136 = torch.ops.aten.permute.default(view_904, [0, 2, 1, 3]);  view_904 = None
        permute_137 = torch.ops.aten.permute.default(view_905, [0, 2, 1, 3]);  view_905 = None
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_135, permute_136, permute_137, None, True, 0.0, True);  permute_135 = permute_136 = permute_137 = None
        getitem_572 = _scaled_dot_product_cudnn_attention_12[0]
        getitem_573 = _scaled_dot_product_cudnn_attention_12[1]
        getitem_578 = _scaled_dot_product_cudnn_attention_12[6]
        getitem_579 = _scaled_dot_product_cudnn_attention_12[7];  _scaled_dot_product_cudnn_attention_12 = None
        permute_138 = torch.ops.aten.permute.default(getitem_572, [0, 2, 1, 3])
        view_906 = torch.ops.aten.view.default(permute_138, [2, 8192, -1]);  permute_138 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16)
        all_gather_into_tensor_138 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_413, 32, '0');  convert_element_type_413 = None
        wait_tensor_163 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_138);  all_gather_into_tensor_138 = None
        permute_139 = torch.ops.aten.permute.default(wait_tensor_163, [1, 0]);  wait_tensor_163 = None
        view_912 = torch.ops.aten.view.default(view_906, [16384, 512]);  view_906 = None
        mm_87 = torch.ops.aten.mm.default(view_912, permute_139);  view_912 = permute_139 = None
        view_913 = torch.ops.aten.view.default(mm_87, [2, 8192, 4096]);  mm_87 = None
        split_58 = torch.ops.aten.split.Tensor(view_913, 1024, 1);  view_913 = None
        getitem_581 = split_58[0]
        getitem_582 = split_58[1]
        getitem_583 = split_58[2]
        getitem_584 = split_58[3]
        getitem_585 = split_58[4]
        getitem_586 = split_58[5]
        getitem_587 = split_58[6]
        getitem_588 = split_58[7];  split_58 = None
        cat_50 = torch.ops.aten.cat.default([getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588]);  getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = None
        reduce_scatter_tensor_25 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_50, 'sum', 8, '1');  cat_50 = None
        wait_tensor_164 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_25)
        add_49 = torch.ops.aten.add.Tensor(add_47, wait_tensor_164);  wait_tensor_164 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16)
        all_gather_into_tensor_139 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_416, 32, '0');  convert_element_type_416 = None
        wait_tensor_165 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_139);  all_gather_into_tensor_139 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_417, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [2], True);  pow_26 = None
        add_50 = torch.ops.aten.add.Scalar(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_25);  convert_element_type_417 = rsqrt_25 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, wait_tensor_165);  mul_100 = wait_tensor_165 = None
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        all_gather_into_tensor_140 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_418, 8, '1');  convert_element_type_418 = None
        wait_tensor_166 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_140);  all_gather_into_tensor_140 = None
        split_59 = torch.ops.aten.split.Tensor(wait_tensor_166, 2);  wait_tensor_166 = None
        getitem_589 = split_59[0]
        getitem_590 = split_59[1]
        getitem_591 = split_59[2]
        getitem_592 = split_59[3]
        getitem_593 = split_59[4]
        getitem_594 = split_59[5]
        getitem_595 = split_59[6]
        getitem_596 = split_59[7];  split_59 = None
        cat_51 = torch.ops.aten.cat.default([getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596], 1);  getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16)
        all_gather_into_tensor_141 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_419, 32, '0');  convert_element_type_419 = None
        wait_tensor_167 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_141);  all_gather_into_tensor_141 = None
        permute_140 = torch.ops.aten.permute.default(wait_tensor_167, [1, 0]);  wait_tensor_167 = None
        view_924 = torch.ops.aten.view.default(cat_51, [16384, 4096]);  cat_51 = None
        mm_88 = torch.ops.aten.mm.default(view_924, permute_140);  permute_140 = None
        view_925 = torch.ops.aten.view.default(mm_88, [2, 8192, 1792])
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_925, torch.float32);  view_925 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_422)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_422, sigmoid_12);  convert_element_type_422 = sigmoid_12 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16)
        all_gather_into_tensor_142 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_424, 32, '0');  convert_element_type_424 = None
        wait_tensor_168 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_142);  all_gather_into_tensor_142 = None
        permute_141 = torch.ops.aten.permute.default(wait_tensor_168, [1, 0]);  wait_tensor_168 = None
        mm_89 = torch.ops.aten.mm.default(view_924, permute_141);  view_924 = permute_141 = None
        view_932 = torch.ops.aten.view.default(mm_89, [2, 8192, 1792]);  mm_89 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_423, view_932);  convert_element_type_423 = view_932 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16)
        all_gather_into_tensor_143 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_427, 32, '0');  convert_element_type_427 = None
        wait_tensor_169 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_143);  all_gather_into_tensor_143 = None
        permute_142 = torch.ops.aten.permute.default(wait_tensor_169, [1, 0]);  wait_tensor_169 = None
        view_939 = torch.ops.aten.view.default(mul_103, [16384, 1792]);  mul_103 = None
        mm_90 = torch.ops.aten.mm.default(view_939, permute_142);  view_939 = permute_142 = None
        view_940 = torch.ops.aten.view.default(mm_90, [2, 8192, 4096]);  mm_90 = None
        split_60 = torch.ops.aten.split.Tensor(view_940, 1024, 1);  view_940 = None
        getitem_597 = split_60[0]
        getitem_598 = split_60[1]
        getitem_599 = split_60[2]
        getitem_600 = split_60[3]
        getitem_601 = split_60[4]
        getitem_602 = split_60[5]
        getitem_603 = split_60[6]
        getitem_604 = split_60[7];  split_60 = None
        cat_52 = torch.ops.aten.cat.default([getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604]);  getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = getitem_602 = getitem_603 = getitem_604 = None
        reduce_scatter_tensor_26 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_52, 'sum', 8, '1');  cat_52 = None
        wait_tensor_170 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_26);  reduce_scatter_tensor_26 = None
        add_51 = torch.ops.aten.add.Tensor(add_49, wait_tensor_170);  add_49 = wait_tensor_170 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16)
        all_gather_into_tensor_144 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_430, 32, '0');  convert_element_type_430 = None
        wait_tensor_171 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_144);  all_gather_into_tensor_144 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_52 = torch.ops.aten.add.Scalar(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_431, rsqrt_26);  convert_element_type_431 = rsqrt_26 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, wait_tensor_171);  mul_104 = wait_tensor_171 = None
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        all_gather_into_tensor_145 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_432, 8, '1');  convert_element_type_432 = None
        wait_tensor_172 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_145);  all_gather_into_tensor_145 = None
        split_61 = torch.ops.aten.split.Tensor(wait_tensor_172, 2);  wait_tensor_172 = None
        getitem_605 = split_61[0]
        getitem_606 = split_61[1]
        getitem_607 = split_61[2]
        getitem_608 = split_61[3]
        getitem_609 = split_61[4]
        getitem_610 = split_61[5]
        getitem_611 = split_61[6]
        getitem_612 = split_61[7];  split_61 = None
        cat_53 = torch.ops.aten.cat.default([getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612], 1);  getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = None
        convert_element_type_433 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16)
        all_gather_into_tensor_146 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_433, 32, '0');  convert_element_type_433 = None
        wait_tensor_173 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_146);  all_gather_into_tensor_146 = None
        permute_143 = torch.ops.aten.permute.default(wait_tensor_173, [1, 0]);  wait_tensor_173 = None
        view_951 = torch.ops.aten.view.default(cat_53, [16384, 4096]);  cat_53 = None
        mm_91 = torch.ops.aten.mm.default(view_951, permute_143);  permute_143 = None
        view_952 = torch.ops.aten.view.default(mm_91, [2, 8192, 512])
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16)
        all_gather_into_tensor_147 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_436, 32, '0');  convert_element_type_436 = None
        wait_tensor_174 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_147);  all_gather_into_tensor_147 = None
        permute_144 = torch.ops.aten.permute.default(wait_tensor_174, [1, 0]);  wait_tensor_174 = None
        mm_92 = torch.ops.aten.mm.default(view_951, permute_144);  permute_144 = None
        view_959 = torch.ops.aten.view.default(mm_92, [2, 8192, 128]);  mm_92 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16)
        all_gather_into_tensor_148 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_439, 32, '0');  convert_element_type_439 = None
        wait_tensor_175 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_148);  all_gather_into_tensor_148 = None
        permute_145 = torch.ops.aten.permute.default(wait_tensor_175, [1, 0]);  wait_tensor_175 = None
        mm_93 = torch.ops.aten.mm.default(view_951, permute_145);  view_951 = permute_145 = None
        view_966 = torch.ops.aten.view.default(mm_93, [2, 8192, 128])
        view_968 = torch.ops.aten.view.default(view_952, [2, 8192, -1, 128]);  view_952 = None
        view_969 = torch.ops.aten.view.default(view_959, [2, 8192, -1, 128]);  view_959 = None
        view_970 = torch.ops.aten.view.default(view_966, [2, 8192, -1, 128]);  view_966 = None
        convert_element_type_442 = torch.ops.prims.convert_element_type.default(view_968, torch.float32);  view_968 = None
        view_971 = torch.ops.aten.view.default(convert_element_type_442, [2, 8192, 4, -1, 2]);  convert_element_type_442 = None
        view_as_complex_26 = torch.ops.aten.view_as_complex.default(view_971);  view_971 = None
        convert_element_type_443 = torch.ops.prims.convert_element_type.default(view_969, torch.float32);  view_969 = None
        view_972 = torch.ops.aten.view.default(convert_element_type_443, [2, 8192, 1, -1, 2]);  convert_element_type_443 = None
        view_as_complex_27 = torch.ops.aten.view_as_complex.default(view_972);  view_972 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_as_complex_26, view_37);  view_as_complex_26 = None
        view_as_real_26 = torch.ops.aten.view_as_real.default(mul_106);  mul_106 = None
        view_974 = torch.ops.aten.view.default(view_as_real_26, [2, 8192, 4, 128]);  view_as_real_26 = None
        mul_107 = torch.ops.aten.mul.Tensor(view_as_complex_27, view_37);  view_as_complex_27 = None
        view_as_real_27 = torch.ops.aten.view_as_real.default(mul_107);  mul_107 = None
        view_975 = torch.ops.aten.view.default(view_as_real_27, [2, 8192, 1, 128]);  view_as_real_27 = None
        convert_element_type_444 = torch.ops.prims.convert_element_type.default(view_974, torch.bfloat16);  view_974 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(view_975, torch.bfloat16);  view_975 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(convert_element_type_445, 3);  convert_element_type_445 = None
        expand_26 = torch.ops.aten.expand.default(unsqueeze_26, [2, 8192, 1, 4, 128]);  unsqueeze_26 = None
        view_976 = torch.ops.aten.view.default(expand_26, [2, 8192, 4, 128]);  expand_26 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(view_970, 3);  view_970 = None
        expand_27 = torch.ops.aten.expand.default(unsqueeze_27, [2, 8192, 1, 4, 128]);  unsqueeze_27 = None
        view_977 = torch.ops.aten.view.default(expand_27, [2, 8192, 4, 128]);  expand_27 = None
        permute_146 = torch.ops.aten.permute.default(convert_element_type_444, [0, 2, 1, 3]);  convert_element_type_444 = None
        permute_147 = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
        permute_148 = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_146, permute_147, permute_148, None, True, 0.0, True);  permute_146 = permute_147 = permute_148 = None
        getitem_613 = _scaled_dot_product_cudnn_attention_13[0]
        getitem_614 = _scaled_dot_product_cudnn_attention_13[1]
        getitem_619 = _scaled_dot_product_cudnn_attention_13[6]
        getitem_620 = _scaled_dot_product_cudnn_attention_13[7];  _scaled_dot_product_cudnn_attention_13 = None
        permute_149 = torch.ops.aten.permute.default(getitem_613, [0, 2, 1, 3])
        view_978 = torch.ops.aten.view.default(permute_149, [2, 8192, -1]);  permute_149 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16)
        all_gather_into_tensor_149 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_446, 32, '0');  convert_element_type_446 = None
        wait_tensor_176 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_149);  all_gather_into_tensor_149 = None
        permute_150 = torch.ops.aten.permute.default(wait_tensor_176, [1, 0]);  wait_tensor_176 = None
        view_984 = torch.ops.aten.view.default(view_978, [16384, 512]);  view_978 = None
        mm_94 = torch.ops.aten.mm.default(view_984, permute_150);  view_984 = permute_150 = None
        view_985 = torch.ops.aten.view.default(mm_94, [2, 8192, 4096]);  mm_94 = None
        split_62 = torch.ops.aten.split.Tensor(view_985, 1024, 1);  view_985 = None
        getitem_622 = split_62[0]
        getitem_623 = split_62[1]
        getitem_624 = split_62[2]
        getitem_625 = split_62[3]
        getitem_626 = split_62[4]
        getitem_627 = split_62[5]
        getitem_628 = split_62[6]
        getitem_629 = split_62[7];  split_62 = None
        cat_54 = torch.ops.aten.cat.default([getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629]);  getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = None
        reduce_scatter_tensor_27 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_54, 'sum', 8, '1');  cat_54 = None
        wait_tensor_177 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_27)
        add_53 = torch.ops.aten.add.Tensor(add_51, wait_tensor_177);  wait_tensor_177 = None
        convert_element_type_449 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16)
        all_gather_into_tensor_150 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_449, 32, '0');  convert_element_type_449 = None
        wait_tensor_178 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_150);  all_gather_into_tensor_150 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_450, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [2], True);  pow_28 = None
        add_54 = torch.ops.aten.add.Scalar(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_27);  convert_element_type_450 = rsqrt_27 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, wait_tensor_178);  mul_108 = wait_tensor_178 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        all_gather_into_tensor_151 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_451, 8, '1');  convert_element_type_451 = None
        wait_tensor_179 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_151);  all_gather_into_tensor_151 = None
        split_63 = torch.ops.aten.split.Tensor(wait_tensor_179, 2);  wait_tensor_179 = None
        getitem_630 = split_63[0]
        getitem_631 = split_63[1]
        getitem_632 = split_63[2]
        getitem_633 = split_63[3]
        getitem_634 = split_63[4]
        getitem_635 = split_63[5]
        getitem_636 = split_63[6]
        getitem_637 = split_63[7];  split_63 = None
        cat_55 = torch.ops.aten.cat.default([getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637], 1);  getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16)
        all_gather_into_tensor_152 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_452, 32, '0');  convert_element_type_452 = None
        wait_tensor_180 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_152);  all_gather_into_tensor_152 = None
        permute_151 = torch.ops.aten.permute.default(wait_tensor_180, [1, 0]);  wait_tensor_180 = None
        view_996 = torch.ops.aten.view.default(cat_55, [16384, 4096]);  cat_55 = None
        mm_95 = torch.ops.aten.mm.default(view_996, permute_151);  permute_151 = None
        view_997 = torch.ops.aten.view.default(mm_95, [2, 8192, 1792])
        convert_element_type_455 = torch.ops.prims.convert_element_type.default(view_997, torch.float32);  view_997 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_455)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_13);  convert_element_type_455 = sigmoid_13 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16)
        all_gather_into_tensor_153 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_457, 32, '0');  convert_element_type_457 = None
        wait_tensor_181 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_153);  all_gather_into_tensor_153 = None
        permute_152 = torch.ops.aten.permute.default(wait_tensor_181, [1, 0]);  wait_tensor_181 = None
        mm_96 = torch.ops.aten.mm.default(view_996, permute_152);  view_996 = permute_152 = None
        view_1004 = torch.ops.aten.view.default(mm_96, [2, 8192, 1792]);  mm_96 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_456, view_1004);  convert_element_type_456 = view_1004 = None
        convert_element_type_460 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16)
        all_gather_into_tensor_154 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_460, 32, '0');  convert_element_type_460 = None
        wait_tensor_182 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_154);  all_gather_into_tensor_154 = None
        permute_153 = torch.ops.aten.permute.default(wait_tensor_182, [1, 0]);  wait_tensor_182 = None
        view_1011 = torch.ops.aten.view.default(mul_111, [16384, 1792]);  mul_111 = None
        mm_97 = torch.ops.aten.mm.default(view_1011, permute_153);  view_1011 = permute_153 = None
        view_1012 = torch.ops.aten.view.default(mm_97, [2, 8192, 4096]);  mm_97 = None
        split_64 = torch.ops.aten.split.Tensor(view_1012, 1024, 1);  view_1012 = None
        getitem_638 = split_64[0]
        getitem_639 = split_64[1]
        getitem_640 = split_64[2]
        getitem_641 = split_64[3]
        getitem_642 = split_64[4]
        getitem_643 = split_64[5]
        getitem_644 = split_64[6]
        getitem_645 = split_64[7];  split_64 = None
        cat_56 = torch.ops.aten.cat.default([getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645]);  getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = None
        reduce_scatter_tensor_28 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_56, 'sum', 8, '1');  cat_56 = None
        wait_tensor_183 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_28);  reduce_scatter_tensor_28 = None
        add_55 = torch.ops.aten.add.Tensor(add_53, wait_tensor_183);  add_53 = wait_tensor_183 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16)
        all_gather_into_tensor_155 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_463, 32, '0');  convert_element_type_463 = None
        wait_tensor_184 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_155);  all_gather_into_tensor_155 = None
        convert_element_type_464 = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [2], True);  pow_29 = None
        add_56 = torch.ops.aten.add.Scalar(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_28);  convert_element_type_464 = rsqrt_28 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, wait_tensor_184);  mul_112 = wait_tensor_184 = None
        convert_element_type_465 = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        all_gather_into_tensor_156 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_465, 8, '1');  convert_element_type_465 = None
        wait_tensor_185 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_156);  all_gather_into_tensor_156 = None
        split_65 = torch.ops.aten.split.Tensor(wait_tensor_185, 2);  wait_tensor_185 = None
        getitem_646 = split_65[0]
        getitem_647 = split_65[1]
        getitem_648 = split_65[2]
        getitem_649 = split_65[3]
        getitem_650 = split_65[4]
        getitem_651 = split_65[5]
        getitem_652 = split_65[6]
        getitem_653 = split_65[7];  split_65 = None
        cat_57 = torch.ops.aten.cat.default([getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653], 1);  getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16)
        all_gather_into_tensor_157 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_466, 32, '0');  convert_element_type_466 = None
        wait_tensor_186 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_157);  all_gather_into_tensor_157 = None
        permute_154 = torch.ops.aten.permute.default(wait_tensor_186, [1, 0]);  wait_tensor_186 = None
        view_1023 = torch.ops.aten.view.default(cat_57, [16384, 4096]);  cat_57 = None
        mm_98 = torch.ops.aten.mm.default(view_1023, permute_154);  permute_154 = None
        view_1024 = torch.ops.aten.view.default(mm_98, [2, 8192, 512])
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16)
        all_gather_into_tensor_158 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_469, 32, '0');  convert_element_type_469 = None
        wait_tensor_187 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_158);  all_gather_into_tensor_158 = None
        permute_155 = torch.ops.aten.permute.default(wait_tensor_187, [1, 0]);  wait_tensor_187 = None
        mm_99 = torch.ops.aten.mm.default(view_1023, permute_155);  permute_155 = None
        view_1031 = torch.ops.aten.view.default(mm_99, [2, 8192, 128]);  mm_99 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16)
        all_gather_into_tensor_159 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_472, 32, '0');  convert_element_type_472 = None
        wait_tensor_188 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_159);  all_gather_into_tensor_159 = None
        permute_156 = torch.ops.aten.permute.default(wait_tensor_188, [1, 0]);  wait_tensor_188 = None
        mm_100 = torch.ops.aten.mm.default(view_1023, permute_156);  view_1023 = permute_156 = None
        view_1038 = torch.ops.aten.view.default(mm_100, [2, 8192, 128])
        view_1040 = torch.ops.aten.view.default(view_1024, [2, 8192, -1, 128]);  view_1024 = None
        view_1041 = torch.ops.aten.view.default(view_1031, [2, 8192, -1, 128]);  view_1031 = None
        view_1042 = torch.ops.aten.view.default(view_1038, [2, 8192, -1, 128]);  view_1038 = None
        convert_element_type_475 = torch.ops.prims.convert_element_type.default(view_1040, torch.float32);  view_1040 = None
        view_1043 = torch.ops.aten.view.default(convert_element_type_475, [2, 8192, 4, -1, 2]);  convert_element_type_475 = None
        view_as_complex_28 = torch.ops.aten.view_as_complex.default(view_1043);  view_1043 = None
        convert_element_type_476 = torch.ops.prims.convert_element_type.default(view_1041, torch.float32);  view_1041 = None
        view_1044 = torch.ops.aten.view.default(convert_element_type_476, [2, 8192, 1, -1, 2]);  convert_element_type_476 = None
        view_as_complex_29 = torch.ops.aten.view_as_complex.default(view_1044);  view_1044 = None
        mul_114 = torch.ops.aten.mul.Tensor(view_as_complex_28, view_37);  view_as_complex_28 = None
        view_as_real_28 = torch.ops.aten.view_as_real.default(mul_114);  mul_114 = None
        view_1046 = torch.ops.aten.view.default(view_as_real_28, [2, 8192, 4, 128]);  view_as_real_28 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_as_complex_29, view_37);  view_as_complex_29 = None
        view_as_real_29 = torch.ops.aten.view_as_real.default(mul_115);  mul_115 = None
        view_1047 = torch.ops.aten.view.default(view_as_real_29, [2, 8192, 1, 128]);  view_as_real_29 = None
        convert_element_type_477 = torch.ops.prims.convert_element_type.default(view_1046, torch.bfloat16);  view_1046 = None
        convert_element_type_478 = torch.ops.prims.convert_element_type.default(view_1047, torch.bfloat16);  view_1047 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(convert_element_type_478, 3);  convert_element_type_478 = None
        expand_28 = torch.ops.aten.expand.default(unsqueeze_28, [2, 8192, 1, 4, 128]);  unsqueeze_28 = None
        view_1048 = torch.ops.aten.view.default(expand_28, [2, 8192, 4, 128]);  expand_28 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(view_1042, 3);  view_1042 = None
        expand_29 = torch.ops.aten.expand.default(unsqueeze_29, [2, 8192, 1, 4, 128]);  unsqueeze_29 = None
        view_1049 = torch.ops.aten.view.default(expand_29, [2, 8192, 4, 128]);  expand_29 = None
        permute_157 = torch.ops.aten.permute.default(convert_element_type_477, [0, 2, 1, 3]);  convert_element_type_477 = None
        permute_158 = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None
        permute_159 = torch.ops.aten.permute.default(view_1049, [0, 2, 1, 3]);  view_1049 = None
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_157, permute_158, permute_159, None, True, 0.0, True);  permute_157 = permute_158 = permute_159 = None
        getitem_654 = _scaled_dot_product_cudnn_attention_14[0]
        getitem_655 = _scaled_dot_product_cudnn_attention_14[1]
        getitem_660 = _scaled_dot_product_cudnn_attention_14[6]
        getitem_661 = _scaled_dot_product_cudnn_attention_14[7];  _scaled_dot_product_cudnn_attention_14 = None
        permute_160 = torch.ops.aten.permute.default(getitem_654, [0, 2, 1, 3])
        view_1050 = torch.ops.aten.view.default(permute_160, [2, 8192, -1]);  permute_160 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16)
        all_gather_into_tensor_160 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_479, 32, '0');  convert_element_type_479 = None
        wait_tensor_189 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_160);  all_gather_into_tensor_160 = None
        permute_161 = torch.ops.aten.permute.default(wait_tensor_189, [1, 0]);  wait_tensor_189 = None
        view_1056 = torch.ops.aten.view.default(view_1050, [16384, 512]);  view_1050 = None
        mm_101 = torch.ops.aten.mm.default(view_1056, permute_161);  view_1056 = permute_161 = None
        view_1057 = torch.ops.aten.view.default(mm_101, [2, 8192, 4096]);  mm_101 = None
        split_66 = torch.ops.aten.split.Tensor(view_1057, 1024, 1);  view_1057 = None
        getitem_663 = split_66[0]
        getitem_664 = split_66[1]
        getitem_665 = split_66[2]
        getitem_666 = split_66[3]
        getitem_667 = split_66[4]
        getitem_668 = split_66[5]
        getitem_669 = split_66[6]
        getitem_670 = split_66[7];  split_66 = None
        cat_58 = torch.ops.aten.cat.default([getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670]);  getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = None
        reduce_scatter_tensor_29 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_58, 'sum', 8, '1');  cat_58 = None
        wait_tensor_190 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_29)
        add_57 = torch.ops.aten.add.Tensor(add_55, wait_tensor_190);  wait_tensor_190 = None
        convert_element_type_482 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16)
        all_gather_into_tensor_161 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_482, 32, '0');  convert_element_type_482 = None
        wait_tensor_191 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_161);  all_gather_into_tensor_161 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_58 = torch.ops.aten.add.Scalar(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_29);  convert_element_type_483 = rsqrt_29 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, wait_tensor_191);  mul_116 = wait_tensor_191 = None
        convert_element_type_484 = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        all_gather_into_tensor_162 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_484, 8, '1');  convert_element_type_484 = None
        wait_tensor_192 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_162);  all_gather_into_tensor_162 = None
        split_67 = torch.ops.aten.split.Tensor(wait_tensor_192, 2);  wait_tensor_192 = None
        getitem_671 = split_67[0]
        getitem_672 = split_67[1]
        getitem_673 = split_67[2]
        getitem_674 = split_67[3]
        getitem_675 = split_67[4]
        getitem_676 = split_67[5]
        getitem_677 = split_67[6]
        getitem_678 = split_67[7];  split_67 = None
        cat_59 = torch.ops.aten.cat.default([getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678], 1);  getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16)
        all_gather_into_tensor_163 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_485, 32, '0');  convert_element_type_485 = None
        wait_tensor_193 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_163);  all_gather_into_tensor_163 = None
        permute_162 = torch.ops.aten.permute.default(wait_tensor_193, [1, 0]);  wait_tensor_193 = None
        view_1068 = torch.ops.aten.view.default(cat_59, [16384, 4096]);  cat_59 = None
        mm_102 = torch.ops.aten.mm.default(view_1068, permute_162);  permute_162 = None
        view_1069 = torch.ops.aten.view.default(mm_102, [2, 8192, 1792])
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(view_1069, torch.float32);  view_1069 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_488)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_488, sigmoid_14);  convert_element_type_488 = sigmoid_14 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16)
        all_gather_into_tensor_164 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_490, 32, '0');  convert_element_type_490 = None
        wait_tensor_194 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_164);  all_gather_into_tensor_164 = None
        permute_163 = torch.ops.aten.permute.default(wait_tensor_194, [1, 0]);  wait_tensor_194 = None
        mm_103 = torch.ops.aten.mm.default(view_1068, permute_163);  view_1068 = permute_163 = None
        view_1076 = torch.ops.aten.view.default(mm_103, [2, 8192, 1792]);  mm_103 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_489, view_1076);  convert_element_type_489 = view_1076 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16)
        all_gather_into_tensor_165 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_493, 32, '0');  convert_element_type_493 = None
        wait_tensor_195 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_165);  all_gather_into_tensor_165 = None
        permute_164 = torch.ops.aten.permute.default(wait_tensor_195, [1, 0]);  wait_tensor_195 = None
        view_1083 = torch.ops.aten.view.default(mul_119, [16384, 1792]);  mul_119 = None
        mm_104 = torch.ops.aten.mm.default(view_1083, permute_164);  view_1083 = permute_164 = None
        view_1084 = torch.ops.aten.view.default(mm_104, [2, 8192, 4096]);  mm_104 = None
        split_68 = torch.ops.aten.split.Tensor(view_1084, 1024, 1);  view_1084 = None
        getitem_679 = split_68[0]
        getitem_680 = split_68[1]
        getitem_681 = split_68[2]
        getitem_682 = split_68[3]
        getitem_683 = split_68[4]
        getitem_684 = split_68[5]
        getitem_685 = split_68[6]
        getitem_686 = split_68[7];  split_68 = None
        cat_60 = torch.ops.aten.cat.default([getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686]);  getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = None
        reduce_scatter_tensor_30 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_60, 'sum', 8, '1');  cat_60 = None
        wait_tensor_196 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_30);  reduce_scatter_tensor_30 = None
        add_59 = torch.ops.aten.add.Tensor(add_57, wait_tensor_196);  add_57 = wait_tensor_196 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16)
        all_gather_into_tensor_166 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_496, 32, '0');  convert_element_type_496 = None
        wait_tensor_197 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_166);  all_gather_into_tensor_166 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [2], True);  pow_31 = None
        add_60 = torch.ops.aten.add.Scalar(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_30);  convert_element_type_497 = rsqrt_30 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, wait_tensor_197);  mul_120 = wait_tensor_197 = None
        convert_element_type_498 = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        all_gather_into_tensor_167 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_498, 8, '1');  convert_element_type_498 = None
        wait_tensor_198 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_167);  all_gather_into_tensor_167 = None
        split_69 = torch.ops.aten.split.Tensor(wait_tensor_198, 2);  wait_tensor_198 = None
        getitem_687 = split_69[0]
        getitem_688 = split_69[1]
        getitem_689 = split_69[2]
        getitem_690 = split_69[3]
        getitem_691 = split_69[4]
        getitem_692 = split_69[5]
        getitem_693 = split_69[6]
        getitem_694 = split_69[7];  split_69 = None
        cat_61 = torch.ops.aten.cat.default([getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694], 1);  getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = None
        convert_element_type_499 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16)
        all_gather_into_tensor_168 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_499, 32, '0');  convert_element_type_499 = None
        wait_tensor_199 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_168);  all_gather_into_tensor_168 = None
        permute_165 = torch.ops.aten.permute.default(wait_tensor_199, [1, 0]);  wait_tensor_199 = None
        view_1095 = torch.ops.aten.view.default(cat_61, [16384, 4096]);  cat_61 = None
        mm_105 = torch.ops.aten.mm.default(view_1095, permute_165);  permute_165 = None
        view_1096 = torch.ops.aten.view.default(mm_105, [2, 8192, 512])
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16)
        all_gather_into_tensor_169 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_502, 32, '0');  convert_element_type_502 = None
        wait_tensor_200 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_169);  all_gather_into_tensor_169 = None
        permute_166 = torch.ops.aten.permute.default(wait_tensor_200, [1, 0]);  wait_tensor_200 = None
        mm_106 = torch.ops.aten.mm.default(view_1095, permute_166);  permute_166 = None
        view_1103 = torch.ops.aten.view.default(mm_106, [2, 8192, 128]);  mm_106 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16)
        all_gather_into_tensor_170 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_505, 32, '0');  convert_element_type_505 = None
        wait_tensor_201 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_170);  all_gather_into_tensor_170 = None
        permute_167 = torch.ops.aten.permute.default(wait_tensor_201, [1, 0]);  wait_tensor_201 = None
        mm_107 = torch.ops.aten.mm.default(view_1095, permute_167);  view_1095 = permute_167 = None
        view_1110 = torch.ops.aten.view.default(mm_107, [2, 8192, 128])
        view_1112 = torch.ops.aten.view.default(view_1096, [2, 8192, -1, 128]);  view_1096 = None
        view_1113 = torch.ops.aten.view.default(view_1103, [2, 8192, -1, 128]);  view_1103 = None
        view_1114 = torch.ops.aten.view.default(view_1110, [2, 8192, -1, 128]);  view_1110 = None
        convert_element_type_508 = torch.ops.prims.convert_element_type.default(view_1112, torch.float32);  view_1112 = None
        view_1115 = torch.ops.aten.view.default(convert_element_type_508, [2, 8192, 4, -1, 2]);  convert_element_type_508 = None
        view_as_complex_30 = torch.ops.aten.view_as_complex.default(view_1115);  view_1115 = None
        convert_element_type_509 = torch.ops.prims.convert_element_type.default(view_1113, torch.float32);  view_1113 = None
        view_1116 = torch.ops.aten.view.default(convert_element_type_509, [2, 8192, 1, -1, 2]);  convert_element_type_509 = None
        view_as_complex_31 = torch.ops.aten.view_as_complex.default(view_1116);  view_1116 = None
        mul_122 = torch.ops.aten.mul.Tensor(view_as_complex_30, view_37);  view_as_complex_30 = None
        view_as_real_30 = torch.ops.aten.view_as_real.default(mul_122);  mul_122 = None
        view_1118 = torch.ops.aten.view.default(view_as_real_30, [2, 8192, 4, 128]);  view_as_real_30 = None
        mul_123 = torch.ops.aten.mul.Tensor(view_as_complex_31, view_37);  view_as_complex_31 = None
        view_as_real_31 = torch.ops.aten.view_as_real.default(mul_123);  mul_123 = None
        view_1119 = torch.ops.aten.view.default(view_as_real_31, [2, 8192, 1, 128]);  view_as_real_31 = None
        convert_element_type_510 = torch.ops.prims.convert_element_type.default(view_1118, torch.bfloat16);  view_1118 = None
        convert_element_type_511 = torch.ops.prims.convert_element_type.default(view_1119, torch.bfloat16);  view_1119 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(convert_element_type_511, 3);  convert_element_type_511 = None
        expand_30 = torch.ops.aten.expand.default(unsqueeze_30, [2, 8192, 1, 4, 128]);  unsqueeze_30 = None
        view_1120 = torch.ops.aten.view.default(expand_30, [2, 8192, 4, 128]);  expand_30 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(view_1114, 3);  view_1114 = None
        expand_31 = torch.ops.aten.expand.default(unsqueeze_31, [2, 8192, 1, 4, 128]);  unsqueeze_31 = None
        view_1121 = torch.ops.aten.view.default(expand_31, [2, 8192, 4, 128]);  expand_31 = None
        permute_168 = torch.ops.aten.permute.default(convert_element_type_510, [0, 2, 1, 3]);  convert_element_type_510 = None
        permute_169 = torch.ops.aten.permute.default(view_1120, [0, 2, 1, 3]);  view_1120 = None
        permute_170 = torch.ops.aten.permute.default(view_1121, [0, 2, 1, 3]);  view_1121 = None
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_168, permute_169, permute_170, None, True, 0.0, True);  permute_168 = permute_169 = permute_170 = None
        getitem_695 = _scaled_dot_product_cudnn_attention_15[0]
        getitem_696 = _scaled_dot_product_cudnn_attention_15[1]
        getitem_701 = _scaled_dot_product_cudnn_attention_15[6]
        getitem_702 = _scaled_dot_product_cudnn_attention_15[7];  _scaled_dot_product_cudnn_attention_15 = None
        permute_171 = torch.ops.aten.permute.default(getitem_695, [0, 2, 1, 3])
        view_1122 = torch.ops.aten.view.default(permute_171, [2, 8192, -1]);  permute_171 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16)
        all_gather_into_tensor_171 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_512, 32, '0');  convert_element_type_512 = None
        wait_tensor_202 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_171);  all_gather_into_tensor_171 = None
        permute_172 = torch.ops.aten.permute.default(wait_tensor_202, [1, 0]);  wait_tensor_202 = None
        view_1128 = torch.ops.aten.view.default(view_1122, [16384, 512]);  view_1122 = None
        mm_108 = torch.ops.aten.mm.default(view_1128, permute_172);  view_1128 = permute_172 = None
        view_1129 = torch.ops.aten.view.default(mm_108, [2, 8192, 4096]);  mm_108 = None
        split_70 = torch.ops.aten.split.Tensor(view_1129, 1024, 1);  view_1129 = None
        getitem_704 = split_70[0]
        getitem_705 = split_70[1]
        getitem_706 = split_70[2]
        getitem_707 = split_70[3]
        getitem_708 = split_70[4]
        getitem_709 = split_70[5]
        getitem_710 = split_70[6]
        getitem_711 = split_70[7];  split_70 = None
        cat_62 = torch.ops.aten.cat.default([getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711]);  getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = None
        reduce_scatter_tensor_31 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_62, 'sum', 8, '1');  cat_62 = None
        wait_tensor_203 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_31)
        add_61 = torch.ops.aten.add.Tensor(add_59, wait_tensor_203);  wait_tensor_203 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16)
        all_gather_into_tensor_172 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_515, 32, '0');  convert_element_type_515 = None
        wait_tensor_204 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_172);  all_gather_into_tensor_172 = None
        convert_element_type_516 = torch.ops.prims.convert_element_type.default(add_61, torch.float32)
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_516, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_62 = torch.ops.aten.add.Scalar(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_516, rsqrt_31);  convert_element_type_516 = rsqrt_31 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, wait_tensor_204);  mul_124 = wait_tensor_204 = None
        convert_element_type_517 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        all_gather_into_tensor_173 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_517, 8, '1');  convert_element_type_517 = None
        wait_tensor_205 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_173);  all_gather_into_tensor_173 = None
        split_71 = torch.ops.aten.split.Tensor(wait_tensor_205, 2);  wait_tensor_205 = None
        getitem_712 = split_71[0]
        getitem_713 = split_71[1]
        getitem_714 = split_71[2]
        getitem_715 = split_71[3]
        getitem_716 = split_71[4]
        getitem_717 = split_71[5]
        getitem_718 = split_71[6]
        getitem_719 = split_71[7];  split_71 = None
        cat_63 = torch.ops.aten.cat.default([getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719], 1);  getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16)
        all_gather_into_tensor_174 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_518, 32, '0');  convert_element_type_518 = None
        wait_tensor_206 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_174);  all_gather_into_tensor_174 = None
        permute_173 = torch.ops.aten.permute.default(wait_tensor_206, [1, 0]);  wait_tensor_206 = None
        view_1140 = torch.ops.aten.view.default(cat_63, [16384, 4096]);  cat_63 = None
        mm_109 = torch.ops.aten.mm.default(view_1140, permute_173);  permute_173 = None
        view_1141 = torch.ops.aten.view.default(mm_109, [2, 8192, 1792])
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(view_1141, torch.float32);  view_1141 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_521)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_521, sigmoid_15);  convert_element_type_521 = sigmoid_15 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16)
        all_gather_into_tensor_175 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_523, 32, '0');  convert_element_type_523 = None
        wait_tensor_207 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_175);  all_gather_into_tensor_175 = None
        permute_174 = torch.ops.aten.permute.default(wait_tensor_207, [1, 0]);  wait_tensor_207 = None
        mm_110 = torch.ops.aten.mm.default(view_1140, permute_174);  view_1140 = permute_174 = None
        view_1148 = torch.ops.aten.view.default(mm_110, [2, 8192, 1792]);  mm_110 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_522, view_1148);  convert_element_type_522 = view_1148 = None
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16)
        all_gather_into_tensor_176 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_526, 32, '0');  convert_element_type_526 = None
        wait_tensor_208 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_176);  all_gather_into_tensor_176 = None
        permute_175 = torch.ops.aten.permute.default(wait_tensor_208, [1, 0]);  wait_tensor_208 = None
        view_1155 = torch.ops.aten.view.default(mul_127, [16384, 1792]);  mul_127 = None
        mm_111 = torch.ops.aten.mm.default(view_1155, permute_175);  view_1155 = permute_175 = None
        view_1156 = torch.ops.aten.view.default(mm_111, [2, 8192, 4096]);  mm_111 = None
        split_72 = torch.ops.aten.split.Tensor(view_1156, 1024, 1);  view_1156 = None
        getitem_720 = split_72[0]
        getitem_721 = split_72[1]
        getitem_722 = split_72[2]
        getitem_723 = split_72[3]
        getitem_724 = split_72[4]
        getitem_725 = split_72[5]
        getitem_726 = split_72[6]
        getitem_727 = split_72[7];  split_72 = None
        cat_64 = torch.ops.aten.cat.default([getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727]);  getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = None
        reduce_scatter_tensor_32 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_64, 'sum', 8, '1');  cat_64 = None
        wait_tensor_209 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_32);  reduce_scatter_tensor_32 = None
        add_63 = torch.ops.aten.add.Tensor(add_61, wait_tensor_209);  add_61 = wait_tensor_209 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16)
        all_gather_into_tensor_177 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_529, 32, '0');  convert_element_type_529 = None
        wait_tensor_210 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_177);  all_gather_into_tensor_177 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_530, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_33, [2], True);  pow_33 = None
        add_64 = torch.ops.aten.add.Scalar(mean_32, 1e-05);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_32);  convert_element_type_530 = rsqrt_32 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, wait_tensor_210);  mul_128 = wait_tensor_210 = None
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        all_gather_into_tensor_178 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_531, 8, '1');  convert_element_type_531 = None
        wait_tensor_211 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_178);  all_gather_into_tensor_178 = None
        split_73 = torch.ops.aten.split.Tensor(wait_tensor_211, 2);  wait_tensor_211 = None
        getitem_728 = split_73[0]
        getitem_729 = split_73[1]
        getitem_730 = split_73[2]
        getitem_731 = split_73[3]
        getitem_732 = split_73[4]
        getitem_733 = split_73[5]
        getitem_734 = split_73[6]
        getitem_735 = split_73[7];  split_73 = None
        cat_65 = torch.ops.aten.cat.default([getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735], 1);  getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16)
        all_gather_into_tensor_179 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_532, 32, '0');  convert_element_type_532 = None
        wait_tensor_212 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_179);  all_gather_into_tensor_179 = None
        permute_176 = torch.ops.aten.permute.default(wait_tensor_212, [1, 0]);  wait_tensor_212 = None
        view_1167 = torch.ops.aten.view.default(cat_65, [16384, 4096]);  cat_65 = None
        mm_112 = torch.ops.aten.mm.default(view_1167, permute_176);  permute_176 = None
        view_1168 = torch.ops.aten.view.default(mm_112, [2, 8192, 512])
        convert_element_type_535 = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16)
        all_gather_into_tensor_180 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_535, 32, '0');  convert_element_type_535 = None
        wait_tensor_213 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_180);  all_gather_into_tensor_180 = None
        permute_177 = torch.ops.aten.permute.default(wait_tensor_213, [1, 0]);  wait_tensor_213 = None
        mm_113 = torch.ops.aten.mm.default(view_1167, permute_177);  permute_177 = None
        view_1175 = torch.ops.aten.view.default(mm_113, [2, 8192, 128]);  mm_113 = None
        convert_element_type_538 = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16)
        all_gather_into_tensor_181 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_538, 32, '0');  convert_element_type_538 = None
        wait_tensor_214 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_181);  all_gather_into_tensor_181 = None
        permute_178 = torch.ops.aten.permute.default(wait_tensor_214, [1, 0]);  wait_tensor_214 = None
        mm_114 = torch.ops.aten.mm.default(view_1167, permute_178);  view_1167 = permute_178 = None
        view_1182 = torch.ops.aten.view.default(mm_114, [2, 8192, 128])
        view_1184 = torch.ops.aten.view.default(view_1168, [2, 8192, -1, 128]);  view_1168 = None
        view_1185 = torch.ops.aten.view.default(view_1175, [2, 8192, -1, 128]);  view_1175 = None
        view_1186 = torch.ops.aten.view.default(view_1182, [2, 8192, -1, 128]);  view_1182 = None
        convert_element_type_541 = torch.ops.prims.convert_element_type.default(view_1184, torch.float32);  view_1184 = None
        view_1187 = torch.ops.aten.view.default(convert_element_type_541, [2, 8192, 4, -1, 2]);  convert_element_type_541 = None
        view_as_complex_32 = torch.ops.aten.view_as_complex.default(view_1187);  view_1187 = None
        convert_element_type_542 = torch.ops.prims.convert_element_type.default(view_1185, torch.float32);  view_1185 = None
        view_1188 = torch.ops.aten.view.default(convert_element_type_542, [2, 8192, 1, -1, 2]);  convert_element_type_542 = None
        view_as_complex_33 = torch.ops.aten.view_as_complex.default(view_1188);  view_1188 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_as_complex_32, view_37);  view_as_complex_32 = None
        view_as_real_32 = torch.ops.aten.view_as_real.default(mul_130);  mul_130 = None
        view_1190 = torch.ops.aten.view.default(view_as_real_32, [2, 8192, 4, 128]);  view_as_real_32 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_as_complex_33, view_37);  view_as_complex_33 = None
        view_as_real_33 = torch.ops.aten.view_as_real.default(mul_131);  mul_131 = None
        view_1191 = torch.ops.aten.view.default(view_as_real_33, [2, 8192, 1, 128]);  view_as_real_33 = None
        convert_element_type_543 = torch.ops.prims.convert_element_type.default(view_1190, torch.bfloat16);  view_1190 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(view_1191, torch.bfloat16);  view_1191 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(convert_element_type_544, 3);  convert_element_type_544 = None
        expand_32 = torch.ops.aten.expand.default(unsqueeze_32, [2, 8192, 1, 4, 128]);  unsqueeze_32 = None
        view_1192 = torch.ops.aten.view.default(expand_32, [2, 8192, 4, 128]);  expand_32 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(view_1186, 3);  view_1186 = None
        expand_33 = torch.ops.aten.expand.default(unsqueeze_33, [2, 8192, 1, 4, 128]);  unsqueeze_33 = None
        view_1193 = torch.ops.aten.view.default(expand_33, [2, 8192, 4, 128]);  expand_33 = None
        permute_179 = torch.ops.aten.permute.default(convert_element_type_543, [0, 2, 1, 3]);  convert_element_type_543 = None
        permute_180 = torch.ops.aten.permute.default(view_1192, [0, 2, 1, 3]);  view_1192 = None
        permute_181 = torch.ops.aten.permute.default(view_1193, [0, 2, 1, 3]);  view_1193 = None
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_179, permute_180, permute_181, None, True, 0.0, True);  permute_179 = permute_180 = permute_181 = None
        getitem_736 = _scaled_dot_product_cudnn_attention_16[0]
        getitem_737 = _scaled_dot_product_cudnn_attention_16[1]
        getitem_742 = _scaled_dot_product_cudnn_attention_16[6]
        getitem_743 = _scaled_dot_product_cudnn_attention_16[7];  _scaled_dot_product_cudnn_attention_16 = None
        permute_182 = torch.ops.aten.permute.default(getitem_736, [0, 2, 1, 3])
        view_1194 = torch.ops.aten.view.default(permute_182, [2, 8192, -1]);  permute_182 = None
        convert_element_type_545 = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16)
        all_gather_into_tensor_182 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_545, 32, '0');  convert_element_type_545 = None
        wait_tensor_215 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_182);  all_gather_into_tensor_182 = None
        permute_183 = torch.ops.aten.permute.default(wait_tensor_215, [1, 0]);  wait_tensor_215 = None
        view_1200 = torch.ops.aten.view.default(view_1194, [16384, 512]);  view_1194 = None
        mm_115 = torch.ops.aten.mm.default(view_1200, permute_183);  view_1200 = permute_183 = None
        view_1201 = torch.ops.aten.view.default(mm_115, [2, 8192, 4096]);  mm_115 = None
        split_74 = torch.ops.aten.split.Tensor(view_1201, 1024, 1);  view_1201 = None
        getitem_745 = split_74[0]
        getitem_746 = split_74[1]
        getitem_747 = split_74[2]
        getitem_748 = split_74[3]
        getitem_749 = split_74[4]
        getitem_750 = split_74[5]
        getitem_751 = split_74[6]
        getitem_752 = split_74[7];  split_74 = None
        cat_66 = torch.ops.aten.cat.default([getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752]);  getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = None
        reduce_scatter_tensor_33 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_66, 'sum', 8, '1');  cat_66 = None
        wait_tensor_216 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_33)
        add_65 = torch.ops.aten.add.Tensor(add_63, wait_tensor_216);  wait_tensor_216 = None
        convert_element_type_548 = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16)
        all_gather_into_tensor_183 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_548, 32, '0');  convert_element_type_548 = None
        wait_tensor_217 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_183);  all_gather_into_tensor_183 = None
        convert_element_type_549 = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_549, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_34, [2], True);  pow_34 = None
        add_66 = torch.ops.aten.add.Scalar(mean_33, 1e-05);  mean_33 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_132 = torch.ops.aten.mul.Tensor(convert_element_type_549, rsqrt_33);  convert_element_type_549 = rsqrt_33 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, wait_tensor_217);  mul_132 = wait_tensor_217 = None
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        all_gather_into_tensor_184 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_550, 8, '1');  convert_element_type_550 = None
        wait_tensor_218 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_184);  all_gather_into_tensor_184 = None
        split_75 = torch.ops.aten.split.Tensor(wait_tensor_218, 2);  wait_tensor_218 = None
        getitem_753 = split_75[0]
        getitem_754 = split_75[1]
        getitem_755 = split_75[2]
        getitem_756 = split_75[3]
        getitem_757 = split_75[4]
        getitem_758 = split_75[5]
        getitem_759 = split_75[6]
        getitem_760 = split_75[7];  split_75 = None
        cat_67 = torch.ops.aten.cat.default([getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760], 1);  getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = None
        convert_element_type_551 = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16)
        all_gather_into_tensor_185 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_551, 32, '0');  convert_element_type_551 = None
        wait_tensor_219 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_185);  all_gather_into_tensor_185 = None
        permute_184 = torch.ops.aten.permute.default(wait_tensor_219, [1, 0]);  wait_tensor_219 = None
        view_1212 = torch.ops.aten.view.default(cat_67, [16384, 4096]);  cat_67 = None
        mm_116 = torch.ops.aten.mm.default(view_1212, permute_184);  permute_184 = None
        view_1213 = torch.ops.aten.view.default(mm_116, [2, 8192, 1792])
        convert_element_type_554 = torch.ops.prims.convert_element_type.default(view_1213, torch.float32);  view_1213 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(convert_element_type_554)
        mul_134 = torch.ops.aten.mul.Tensor(convert_element_type_554, sigmoid_16);  convert_element_type_554 = sigmoid_16 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16)
        all_gather_into_tensor_186 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_556, 32, '0');  convert_element_type_556 = None
        wait_tensor_220 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_186);  all_gather_into_tensor_186 = None
        permute_185 = torch.ops.aten.permute.default(wait_tensor_220, [1, 0]);  wait_tensor_220 = None
        mm_117 = torch.ops.aten.mm.default(view_1212, permute_185);  view_1212 = permute_185 = None
        view_1220 = torch.ops.aten.view.default(mm_117, [2, 8192, 1792]);  mm_117 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_555, view_1220);  convert_element_type_555 = view_1220 = None
        convert_element_type_559 = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16)
        all_gather_into_tensor_187 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_559, 32, '0');  convert_element_type_559 = None
        wait_tensor_221 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_187);  all_gather_into_tensor_187 = None
        permute_186 = torch.ops.aten.permute.default(wait_tensor_221, [1, 0]);  wait_tensor_221 = None
        view_1227 = torch.ops.aten.view.default(mul_135, [16384, 1792]);  mul_135 = None
        mm_118 = torch.ops.aten.mm.default(view_1227, permute_186);  view_1227 = permute_186 = None
        view_1228 = torch.ops.aten.view.default(mm_118, [2, 8192, 4096]);  mm_118 = None
        split_76 = torch.ops.aten.split.Tensor(view_1228, 1024, 1);  view_1228 = None
        getitem_761 = split_76[0]
        getitem_762 = split_76[1]
        getitem_763 = split_76[2]
        getitem_764 = split_76[3]
        getitem_765 = split_76[4]
        getitem_766 = split_76[5]
        getitem_767 = split_76[6]
        getitem_768 = split_76[7];  split_76 = None
        cat_68 = torch.ops.aten.cat.default([getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768]);  getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = None
        reduce_scatter_tensor_34 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_68, 'sum', 8, '1');  cat_68 = None
        wait_tensor_222 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_34);  reduce_scatter_tensor_34 = None
        add_67 = torch.ops.aten.add.Tensor(add_65, wait_tensor_222);  add_65 = wait_tensor_222 = None
        convert_element_type_562 = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16)
        all_gather_into_tensor_188 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_562, 32, '0');  convert_element_type_562 = None
        wait_tensor_223 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_188);  all_gather_into_tensor_188 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_35, [2], True);  pow_35 = None
        add_68 = torch.ops.aten.add.Scalar(mean_34, 1e-05);  mean_34 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_563, rsqrt_34);  convert_element_type_563 = rsqrt_34 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, wait_tensor_223);  mul_136 = wait_tensor_223 = None
        convert_element_type_564 = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        all_gather_into_tensor_189 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_564, 8, '1');  convert_element_type_564 = None
        wait_tensor_224 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_189);  all_gather_into_tensor_189 = None
        split_77 = torch.ops.aten.split.Tensor(wait_tensor_224, 2);  wait_tensor_224 = None
        getitem_769 = split_77[0]
        getitem_770 = split_77[1]
        getitem_771 = split_77[2]
        getitem_772 = split_77[3]
        getitem_773 = split_77[4]
        getitem_774 = split_77[5]
        getitem_775 = split_77[6]
        getitem_776 = split_77[7];  split_77 = None
        cat_69 = torch.ops.aten.cat.default([getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776], 1);  getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = None
        convert_element_type_565 = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16)
        all_gather_into_tensor_190 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_565, 32, '0');  convert_element_type_565 = None
        wait_tensor_225 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_190);  all_gather_into_tensor_190 = None
        permute_187 = torch.ops.aten.permute.default(wait_tensor_225, [1, 0]);  wait_tensor_225 = None
        view_1239 = torch.ops.aten.view.default(cat_69, [16384, 4096]);  cat_69 = None
        mm_119 = torch.ops.aten.mm.default(view_1239, permute_187);  permute_187 = None
        view_1240 = torch.ops.aten.view.default(mm_119, [2, 8192, 512])
        convert_element_type_568 = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16)
        all_gather_into_tensor_191 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_568, 32, '0');  convert_element_type_568 = None
        wait_tensor_226 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_191);  all_gather_into_tensor_191 = None
        permute_188 = torch.ops.aten.permute.default(wait_tensor_226, [1, 0]);  wait_tensor_226 = None
        mm_120 = torch.ops.aten.mm.default(view_1239, permute_188);  permute_188 = None
        view_1247 = torch.ops.aten.view.default(mm_120, [2, 8192, 128]);  mm_120 = None
        convert_element_type_571 = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16)
        all_gather_into_tensor_192 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_571, 32, '0');  convert_element_type_571 = None
        wait_tensor_227 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_192);  all_gather_into_tensor_192 = None
        permute_189 = torch.ops.aten.permute.default(wait_tensor_227, [1, 0]);  wait_tensor_227 = None
        mm_121 = torch.ops.aten.mm.default(view_1239, permute_189);  view_1239 = permute_189 = None
        view_1254 = torch.ops.aten.view.default(mm_121, [2, 8192, 128])
        view_1256 = torch.ops.aten.view.default(view_1240, [2, 8192, -1, 128]);  view_1240 = None
        view_1257 = torch.ops.aten.view.default(view_1247, [2, 8192, -1, 128]);  view_1247 = None
        view_1258 = torch.ops.aten.view.default(view_1254, [2, 8192, -1, 128]);  view_1254 = None
        convert_element_type_574 = torch.ops.prims.convert_element_type.default(view_1256, torch.float32);  view_1256 = None
        view_1259 = torch.ops.aten.view.default(convert_element_type_574, [2, 8192, 4, -1, 2]);  convert_element_type_574 = None
        view_as_complex_34 = torch.ops.aten.view_as_complex.default(view_1259);  view_1259 = None
        convert_element_type_575 = torch.ops.prims.convert_element_type.default(view_1257, torch.float32);  view_1257 = None
        view_1260 = torch.ops.aten.view.default(convert_element_type_575, [2, 8192, 1, -1, 2]);  convert_element_type_575 = None
        view_as_complex_35 = torch.ops.aten.view_as_complex.default(view_1260);  view_1260 = None
        mul_138 = torch.ops.aten.mul.Tensor(view_as_complex_34, view_37);  view_as_complex_34 = None
        view_as_real_34 = torch.ops.aten.view_as_real.default(mul_138);  mul_138 = None
        view_1262 = torch.ops.aten.view.default(view_as_real_34, [2, 8192, 4, 128]);  view_as_real_34 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_as_complex_35, view_37);  view_as_complex_35 = None
        view_as_real_35 = torch.ops.aten.view_as_real.default(mul_139);  mul_139 = None
        view_1263 = torch.ops.aten.view.default(view_as_real_35, [2, 8192, 1, 128]);  view_as_real_35 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(view_1262, torch.bfloat16);  view_1262 = None
        convert_element_type_577 = torch.ops.prims.convert_element_type.default(view_1263, torch.bfloat16);  view_1263 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(convert_element_type_577, 3);  convert_element_type_577 = None
        expand_34 = torch.ops.aten.expand.default(unsqueeze_34, [2, 8192, 1, 4, 128]);  unsqueeze_34 = None
        view_1264 = torch.ops.aten.view.default(expand_34, [2, 8192, 4, 128]);  expand_34 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(view_1258, 3);  view_1258 = None
        expand_35 = torch.ops.aten.expand.default(unsqueeze_35, [2, 8192, 1, 4, 128]);  unsqueeze_35 = None
        view_1265 = torch.ops.aten.view.default(expand_35, [2, 8192, 4, 128]);  expand_35 = None
        permute_190 = torch.ops.aten.permute.default(convert_element_type_576, [0, 2, 1, 3]);  convert_element_type_576 = None
        permute_191 = torch.ops.aten.permute.default(view_1264, [0, 2, 1, 3]);  view_1264 = None
        permute_192 = torch.ops.aten.permute.default(view_1265, [0, 2, 1, 3]);  view_1265 = None
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_190, permute_191, permute_192, None, True, 0.0, True);  permute_190 = permute_191 = permute_192 = None
        getitem_777 = _scaled_dot_product_cudnn_attention_17[0]
        getitem_778 = _scaled_dot_product_cudnn_attention_17[1]
        getitem_783 = _scaled_dot_product_cudnn_attention_17[6]
        getitem_784 = _scaled_dot_product_cudnn_attention_17[7];  _scaled_dot_product_cudnn_attention_17 = None
        permute_193 = torch.ops.aten.permute.default(getitem_777, [0, 2, 1, 3])
        view_1266 = torch.ops.aten.view.default(permute_193, [2, 8192, -1]);  permute_193 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16)
        all_gather_into_tensor_193 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_578, 32, '0');  convert_element_type_578 = None
        wait_tensor_228 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_193);  all_gather_into_tensor_193 = None
        permute_194 = torch.ops.aten.permute.default(wait_tensor_228, [1, 0]);  wait_tensor_228 = None
        view_1272 = torch.ops.aten.view.default(view_1266, [16384, 512]);  view_1266 = None
        mm_122 = torch.ops.aten.mm.default(view_1272, permute_194);  view_1272 = permute_194 = None
        view_1273 = torch.ops.aten.view.default(mm_122, [2, 8192, 4096]);  mm_122 = None
        split_78 = torch.ops.aten.split.Tensor(view_1273, 1024, 1);  view_1273 = None
        getitem_786 = split_78[0]
        getitem_787 = split_78[1]
        getitem_788 = split_78[2]
        getitem_789 = split_78[3]
        getitem_790 = split_78[4]
        getitem_791 = split_78[5]
        getitem_792 = split_78[6]
        getitem_793 = split_78[7];  split_78 = None
        cat_70 = torch.ops.aten.cat.default([getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793]);  getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = None
        reduce_scatter_tensor_35 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_70, 'sum', 8, '1');  cat_70 = None
        wait_tensor_229 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_35)
        add_69 = torch.ops.aten.add.Tensor(add_67, wait_tensor_229);  wait_tensor_229 = None
        convert_element_type_581 = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16)
        all_gather_into_tensor_194 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_581, 32, '0');  convert_element_type_581 = None
        wait_tensor_230 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_194);  all_gather_into_tensor_194 = None
        convert_element_type_582 = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_582, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_36, [2], True);  pow_36 = None
        add_70 = torch.ops.aten.add.Scalar(mean_35, 1e-05);  mean_35 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_582, rsqrt_35);  convert_element_type_582 = rsqrt_35 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, wait_tensor_230);  mul_140 = wait_tensor_230 = None
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        all_gather_into_tensor_195 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_583, 8, '1');  convert_element_type_583 = None
        wait_tensor_231 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_195);  all_gather_into_tensor_195 = None
        split_79 = torch.ops.aten.split.Tensor(wait_tensor_231, 2);  wait_tensor_231 = None
        getitem_794 = split_79[0]
        getitem_795 = split_79[1]
        getitem_796 = split_79[2]
        getitem_797 = split_79[3]
        getitem_798 = split_79[4]
        getitem_799 = split_79[5]
        getitem_800 = split_79[6]
        getitem_801 = split_79[7];  split_79 = None
        cat_71 = torch.ops.aten.cat.default([getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801], 1);  getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = None
        convert_element_type_584 = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16)
        all_gather_into_tensor_196 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_584, 32, '0');  convert_element_type_584 = None
        wait_tensor_232 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_196);  all_gather_into_tensor_196 = None
        permute_195 = torch.ops.aten.permute.default(wait_tensor_232, [1, 0]);  wait_tensor_232 = None
        view_1284 = torch.ops.aten.view.default(cat_71, [16384, 4096]);  cat_71 = None
        mm_123 = torch.ops.aten.mm.default(view_1284, permute_195);  permute_195 = None
        view_1285 = torch.ops.aten.view.default(mm_123, [2, 8192, 1792])
        convert_element_type_587 = torch.ops.prims.convert_element_type.default(view_1285, torch.float32);  view_1285 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(convert_element_type_587)
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_587, sigmoid_17);  convert_element_type_587 = sigmoid_17 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        convert_element_type_589 = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16)
        all_gather_into_tensor_197 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_589, 32, '0');  convert_element_type_589 = None
        wait_tensor_233 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_197);  all_gather_into_tensor_197 = None
        permute_196 = torch.ops.aten.permute.default(wait_tensor_233, [1, 0]);  wait_tensor_233 = None
        mm_124 = torch.ops.aten.mm.default(view_1284, permute_196);  view_1284 = permute_196 = None
        view_1292 = torch.ops.aten.view.default(mm_124, [2, 8192, 1792]);  mm_124 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_588, view_1292);  convert_element_type_588 = view_1292 = None
        convert_element_type_592 = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16)
        all_gather_into_tensor_198 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_592, 32, '0');  convert_element_type_592 = None
        wait_tensor_234 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_198);  all_gather_into_tensor_198 = None
        permute_197 = torch.ops.aten.permute.default(wait_tensor_234, [1, 0]);  wait_tensor_234 = None
        view_1299 = torch.ops.aten.view.default(mul_143, [16384, 1792]);  mul_143 = None
        mm_125 = torch.ops.aten.mm.default(view_1299, permute_197);  view_1299 = permute_197 = None
        view_1300 = torch.ops.aten.view.default(mm_125, [2, 8192, 4096]);  mm_125 = None
        split_80 = torch.ops.aten.split.Tensor(view_1300, 1024, 1);  view_1300 = None
        getitem_802 = split_80[0]
        getitem_803 = split_80[1]
        getitem_804 = split_80[2]
        getitem_805 = split_80[3]
        getitem_806 = split_80[4]
        getitem_807 = split_80[5]
        getitem_808 = split_80[6]
        getitem_809 = split_80[7];  split_80 = None
        cat_72 = torch.ops.aten.cat.default([getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809]);  getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = None
        reduce_scatter_tensor_36 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_72, 'sum', 8, '1');  cat_72 = None
        wait_tensor_235 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_36);  reduce_scatter_tensor_36 = None
        add_71 = torch.ops.aten.add.Tensor(add_69, wait_tensor_235);  add_69 = wait_tensor_235 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16)
        all_gather_into_tensor_199 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_595, 32, '0');  convert_element_type_595 = None
        wait_tensor_236 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_199);  all_gather_into_tensor_199 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_596, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_37, [2], True);  pow_37 = None
        add_72 = torch.ops.aten.add.Scalar(mean_36, 1e-05);  mean_36 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_596, rsqrt_36);  convert_element_type_596 = rsqrt_36 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, wait_tensor_236);  mul_144 = wait_tensor_236 = None
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        all_gather_into_tensor_200 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_597, 8, '1');  convert_element_type_597 = None
        wait_tensor_237 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_200);  all_gather_into_tensor_200 = None
        split_81 = torch.ops.aten.split.Tensor(wait_tensor_237, 2);  wait_tensor_237 = None
        getitem_810 = split_81[0]
        getitem_811 = split_81[1]
        getitem_812 = split_81[2]
        getitem_813 = split_81[3]
        getitem_814 = split_81[4]
        getitem_815 = split_81[5]
        getitem_816 = split_81[6]
        getitem_817 = split_81[7];  split_81 = None
        cat_73 = torch.ops.aten.cat.default([getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817], 1);  getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = None
        convert_element_type_598 = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16)
        all_gather_into_tensor_201 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_598, 32, '0');  convert_element_type_598 = None
        wait_tensor_238 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_201);  all_gather_into_tensor_201 = None
        permute_198 = torch.ops.aten.permute.default(wait_tensor_238, [1, 0]);  wait_tensor_238 = None
        view_1311 = torch.ops.aten.view.default(cat_73, [16384, 4096]);  cat_73 = None
        mm_126 = torch.ops.aten.mm.default(view_1311, permute_198);  permute_198 = None
        view_1312 = torch.ops.aten.view.default(mm_126, [2, 8192, 512])
        convert_element_type_601 = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16)
        all_gather_into_tensor_202 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_601, 32, '0');  convert_element_type_601 = None
        wait_tensor_239 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_202);  all_gather_into_tensor_202 = None
        permute_199 = torch.ops.aten.permute.default(wait_tensor_239, [1, 0]);  wait_tensor_239 = None
        mm_127 = torch.ops.aten.mm.default(view_1311, permute_199);  permute_199 = None
        view_1319 = torch.ops.aten.view.default(mm_127, [2, 8192, 128]);  mm_127 = None
        convert_element_type_604 = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16)
        all_gather_into_tensor_203 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_604, 32, '0');  convert_element_type_604 = None
        wait_tensor_240 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_203);  all_gather_into_tensor_203 = None
        permute_200 = torch.ops.aten.permute.default(wait_tensor_240, [1, 0]);  wait_tensor_240 = None
        mm_128 = torch.ops.aten.mm.default(view_1311, permute_200);  view_1311 = permute_200 = None
        view_1326 = torch.ops.aten.view.default(mm_128, [2, 8192, 128])
        view_1328 = torch.ops.aten.view.default(view_1312, [2, 8192, -1, 128]);  view_1312 = None
        view_1329 = torch.ops.aten.view.default(view_1319, [2, 8192, -1, 128]);  view_1319 = None
        view_1330 = torch.ops.aten.view.default(view_1326, [2, 8192, -1, 128]);  view_1326 = None
        convert_element_type_607 = torch.ops.prims.convert_element_type.default(view_1328, torch.float32);  view_1328 = None
        view_1331 = torch.ops.aten.view.default(convert_element_type_607, [2, 8192, 4, -1, 2]);  convert_element_type_607 = None
        view_as_complex_36 = torch.ops.aten.view_as_complex.default(view_1331);  view_1331 = None
        convert_element_type_608 = torch.ops.prims.convert_element_type.default(view_1329, torch.float32);  view_1329 = None
        view_1332 = torch.ops.aten.view.default(convert_element_type_608, [2, 8192, 1, -1, 2]);  convert_element_type_608 = None
        view_as_complex_37 = torch.ops.aten.view_as_complex.default(view_1332);  view_1332 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_as_complex_36, view_37);  view_as_complex_36 = None
        view_as_real_36 = torch.ops.aten.view_as_real.default(mul_146);  mul_146 = None
        view_1334 = torch.ops.aten.view.default(view_as_real_36, [2, 8192, 4, 128]);  view_as_real_36 = None
        mul_147 = torch.ops.aten.mul.Tensor(view_as_complex_37, view_37);  view_as_complex_37 = None
        view_as_real_37 = torch.ops.aten.view_as_real.default(mul_147);  mul_147 = None
        view_1335 = torch.ops.aten.view.default(view_as_real_37, [2, 8192, 1, 128]);  view_as_real_37 = None
        convert_element_type_609 = torch.ops.prims.convert_element_type.default(view_1334, torch.bfloat16);  view_1334 = None
        convert_element_type_610 = torch.ops.prims.convert_element_type.default(view_1335, torch.bfloat16);  view_1335 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(convert_element_type_610, 3);  convert_element_type_610 = None
        expand_36 = torch.ops.aten.expand.default(unsqueeze_36, [2, 8192, 1, 4, 128]);  unsqueeze_36 = None
        view_1336 = torch.ops.aten.view.default(expand_36, [2, 8192, 4, 128]);  expand_36 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(view_1330, 3);  view_1330 = None
        expand_37 = torch.ops.aten.expand.default(unsqueeze_37, [2, 8192, 1, 4, 128]);  unsqueeze_37 = None
        view_1337 = torch.ops.aten.view.default(expand_37, [2, 8192, 4, 128]);  expand_37 = None
        permute_201 = torch.ops.aten.permute.default(convert_element_type_609, [0, 2, 1, 3]);  convert_element_type_609 = None
        permute_202 = torch.ops.aten.permute.default(view_1336, [0, 2, 1, 3]);  view_1336 = None
        permute_203 = torch.ops.aten.permute.default(view_1337, [0, 2, 1, 3]);  view_1337 = None
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_201, permute_202, permute_203, None, True, 0.0, True);  permute_201 = permute_202 = permute_203 = None
        getitem_818 = _scaled_dot_product_cudnn_attention_18[0]
        getitem_819 = _scaled_dot_product_cudnn_attention_18[1]
        getitem_824 = _scaled_dot_product_cudnn_attention_18[6]
        getitem_825 = _scaled_dot_product_cudnn_attention_18[7];  _scaled_dot_product_cudnn_attention_18 = None
        permute_204 = torch.ops.aten.permute.default(getitem_818, [0, 2, 1, 3])
        view_1338 = torch.ops.aten.view.default(permute_204, [2, 8192, -1]);  permute_204 = None
        convert_element_type_611 = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16)
        all_gather_into_tensor_204 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_611, 32, '0');  convert_element_type_611 = None
        wait_tensor_241 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_204);  all_gather_into_tensor_204 = None
        permute_205 = torch.ops.aten.permute.default(wait_tensor_241, [1, 0]);  wait_tensor_241 = None
        view_1344 = torch.ops.aten.view.default(view_1338, [16384, 512]);  view_1338 = None
        mm_129 = torch.ops.aten.mm.default(view_1344, permute_205);  view_1344 = permute_205 = None
        view_1345 = torch.ops.aten.view.default(mm_129, [2, 8192, 4096]);  mm_129 = None
        split_82 = torch.ops.aten.split.Tensor(view_1345, 1024, 1);  view_1345 = None
        getitem_827 = split_82[0]
        getitem_828 = split_82[1]
        getitem_829 = split_82[2]
        getitem_830 = split_82[3]
        getitem_831 = split_82[4]
        getitem_832 = split_82[5]
        getitem_833 = split_82[6]
        getitem_834 = split_82[7];  split_82 = None
        cat_74 = torch.ops.aten.cat.default([getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834]);  getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = None
        reduce_scatter_tensor_37 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_74, 'sum', 8, '1');  cat_74 = None
        wait_tensor_242 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_37)
        add_73 = torch.ops.aten.add.Tensor(add_71, wait_tensor_242);  wait_tensor_242 = None
        convert_element_type_614 = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16)
        all_gather_into_tensor_205 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_614, 32, '0');  convert_element_type_614 = None
        wait_tensor_243 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_205);  all_gather_into_tensor_205 = None
        convert_element_type_615 = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_615, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_38, [2], True);  pow_38 = None
        add_74 = torch.ops.aten.add.Scalar(mean_37, 1e-05);  mean_37 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_148 = torch.ops.aten.mul.Tensor(convert_element_type_615, rsqrt_37);  convert_element_type_615 = rsqrt_37 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, wait_tensor_243);  mul_148 = wait_tensor_243 = None
        convert_element_type_616 = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        all_gather_into_tensor_206 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_616, 8, '1');  convert_element_type_616 = None
        wait_tensor_244 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_206);  all_gather_into_tensor_206 = None
        split_83 = torch.ops.aten.split.Tensor(wait_tensor_244, 2);  wait_tensor_244 = None
        getitem_835 = split_83[0]
        getitem_836 = split_83[1]
        getitem_837 = split_83[2]
        getitem_838 = split_83[3]
        getitem_839 = split_83[4]
        getitem_840 = split_83[5]
        getitem_841 = split_83[6]
        getitem_842 = split_83[7];  split_83 = None
        cat_75 = torch.ops.aten.cat.default([getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842], 1);  getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16)
        all_gather_into_tensor_207 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_617, 32, '0');  convert_element_type_617 = None
        wait_tensor_245 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_207);  all_gather_into_tensor_207 = None
        permute_206 = torch.ops.aten.permute.default(wait_tensor_245, [1, 0]);  wait_tensor_245 = None
        view_1356 = torch.ops.aten.view.default(cat_75, [16384, 4096]);  cat_75 = None
        mm_130 = torch.ops.aten.mm.default(view_1356, permute_206);  permute_206 = None
        view_1357 = torch.ops.aten.view.default(mm_130, [2, 8192, 1792])
        convert_element_type_620 = torch.ops.prims.convert_element_type.default(view_1357, torch.float32);  view_1357 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(convert_element_type_620)
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_620, sigmoid_18);  convert_element_type_620 = sigmoid_18 = None
        convert_element_type_621 = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16)
        all_gather_into_tensor_208 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_622, 32, '0');  convert_element_type_622 = None
        wait_tensor_246 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_208);  all_gather_into_tensor_208 = None
        permute_207 = torch.ops.aten.permute.default(wait_tensor_246, [1, 0]);  wait_tensor_246 = None
        mm_131 = torch.ops.aten.mm.default(view_1356, permute_207);  view_1356 = permute_207 = None
        view_1364 = torch.ops.aten.view.default(mm_131, [2, 8192, 1792]);  mm_131 = None
        mul_151 = torch.ops.aten.mul.Tensor(convert_element_type_621, view_1364);  convert_element_type_621 = view_1364 = None
        convert_element_type_625 = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16)
        all_gather_into_tensor_209 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_625, 32, '0');  convert_element_type_625 = None
        wait_tensor_247 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_209);  all_gather_into_tensor_209 = None
        permute_208 = torch.ops.aten.permute.default(wait_tensor_247, [1, 0]);  wait_tensor_247 = None
        view_1371 = torch.ops.aten.view.default(mul_151, [16384, 1792]);  mul_151 = None
        mm_132 = torch.ops.aten.mm.default(view_1371, permute_208);  view_1371 = permute_208 = None
        view_1372 = torch.ops.aten.view.default(mm_132, [2, 8192, 4096]);  mm_132 = None
        split_84 = torch.ops.aten.split.Tensor(view_1372, 1024, 1);  view_1372 = None
        getitem_843 = split_84[0]
        getitem_844 = split_84[1]
        getitem_845 = split_84[2]
        getitem_846 = split_84[3]
        getitem_847 = split_84[4]
        getitem_848 = split_84[5]
        getitem_849 = split_84[6]
        getitem_850 = split_84[7];  split_84 = None
        cat_76 = torch.ops.aten.cat.default([getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850]);  getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = None
        reduce_scatter_tensor_38 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_76, 'sum', 8, '1');  cat_76 = None
        wait_tensor_248 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_38);  reduce_scatter_tensor_38 = None
        add_75 = torch.ops.aten.add.Tensor(add_73, wait_tensor_248);  add_73 = wait_tensor_248 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16)
        all_gather_into_tensor_210 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_628, 32, '0');  convert_element_type_628 = None
        wait_tensor_249 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_210);  all_gather_into_tensor_210 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_629, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_39, [2], True);  pow_39 = None
        add_76 = torch.ops.aten.add.Scalar(mean_38, 1e-05);  mean_38 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_629, rsqrt_38);  convert_element_type_629 = rsqrt_38 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, wait_tensor_249);  mul_152 = wait_tensor_249 = None
        convert_element_type_630 = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None
        all_gather_into_tensor_211 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_630, 8, '1');  convert_element_type_630 = None
        wait_tensor_250 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_211);  all_gather_into_tensor_211 = None
        split_85 = torch.ops.aten.split.Tensor(wait_tensor_250, 2);  wait_tensor_250 = None
        getitem_851 = split_85[0]
        getitem_852 = split_85[1]
        getitem_853 = split_85[2]
        getitem_854 = split_85[3]
        getitem_855 = split_85[4]
        getitem_856 = split_85[5]
        getitem_857 = split_85[6]
        getitem_858 = split_85[7];  split_85 = None
        cat_77 = torch.ops.aten.cat.default([getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858], 1);  getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16)
        all_gather_into_tensor_212 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_631, 32, '0');  convert_element_type_631 = None
        wait_tensor_251 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_212);  all_gather_into_tensor_212 = None
        permute_209 = torch.ops.aten.permute.default(wait_tensor_251, [1, 0]);  wait_tensor_251 = None
        view_1383 = torch.ops.aten.view.default(cat_77, [16384, 4096]);  cat_77 = None
        mm_133 = torch.ops.aten.mm.default(view_1383, permute_209);  permute_209 = None
        view_1384 = torch.ops.aten.view.default(mm_133, [2, 8192, 512])
        convert_element_type_634 = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16)
        all_gather_into_tensor_213 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_634, 32, '0');  convert_element_type_634 = None
        wait_tensor_252 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_213);  all_gather_into_tensor_213 = None
        permute_210 = torch.ops.aten.permute.default(wait_tensor_252, [1, 0]);  wait_tensor_252 = None
        mm_134 = torch.ops.aten.mm.default(view_1383, permute_210);  permute_210 = None
        view_1391 = torch.ops.aten.view.default(mm_134, [2, 8192, 128]);  mm_134 = None
        convert_element_type_637 = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16)
        all_gather_into_tensor_214 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_637, 32, '0');  convert_element_type_637 = None
        wait_tensor_253 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_214);  all_gather_into_tensor_214 = None
        permute_211 = torch.ops.aten.permute.default(wait_tensor_253, [1, 0]);  wait_tensor_253 = None
        mm_135 = torch.ops.aten.mm.default(view_1383, permute_211);  view_1383 = permute_211 = None
        view_1398 = torch.ops.aten.view.default(mm_135, [2, 8192, 128])
        view_1400 = torch.ops.aten.view.default(view_1384, [2, 8192, -1, 128]);  view_1384 = None
        view_1401 = torch.ops.aten.view.default(view_1391, [2, 8192, -1, 128]);  view_1391 = None
        view_1402 = torch.ops.aten.view.default(view_1398, [2, 8192, -1, 128]);  view_1398 = None
        convert_element_type_640 = torch.ops.prims.convert_element_type.default(view_1400, torch.float32);  view_1400 = None
        view_1403 = torch.ops.aten.view.default(convert_element_type_640, [2, 8192, 4, -1, 2]);  convert_element_type_640 = None
        view_as_complex_38 = torch.ops.aten.view_as_complex.default(view_1403);  view_1403 = None
        convert_element_type_641 = torch.ops.prims.convert_element_type.default(view_1401, torch.float32);  view_1401 = None
        view_1404 = torch.ops.aten.view.default(convert_element_type_641, [2, 8192, 1, -1, 2]);  convert_element_type_641 = None
        view_as_complex_39 = torch.ops.aten.view_as_complex.default(view_1404);  view_1404 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_as_complex_38, view_37);  view_as_complex_38 = None
        view_as_real_38 = torch.ops.aten.view_as_real.default(mul_154);  mul_154 = None
        view_1406 = torch.ops.aten.view.default(view_as_real_38, [2, 8192, 4, 128]);  view_as_real_38 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_as_complex_39, view_37);  view_as_complex_39 = None
        view_as_real_39 = torch.ops.aten.view_as_real.default(mul_155);  mul_155 = None
        view_1407 = torch.ops.aten.view.default(view_as_real_39, [2, 8192, 1, 128]);  view_as_real_39 = None
        convert_element_type_642 = torch.ops.prims.convert_element_type.default(view_1406, torch.bfloat16);  view_1406 = None
        convert_element_type_643 = torch.ops.prims.convert_element_type.default(view_1407, torch.bfloat16);  view_1407 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(convert_element_type_643, 3);  convert_element_type_643 = None
        expand_38 = torch.ops.aten.expand.default(unsqueeze_38, [2, 8192, 1, 4, 128]);  unsqueeze_38 = None
        view_1408 = torch.ops.aten.view.default(expand_38, [2, 8192, 4, 128]);  expand_38 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(view_1402, 3);  view_1402 = None
        expand_39 = torch.ops.aten.expand.default(unsqueeze_39, [2, 8192, 1, 4, 128]);  unsqueeze_39 = None
        view_1409 = torch.ops.aten.view.default(expand_39, [2, 8192, 4, 128]);  expand_39 = None
        permute_212 = torch.ops.aten.permute.default(convert_element_type_642, [0, 2, 1, 3]);  convert_element_type_642 = None
        permute_213 = torch.ops.aten.permute.default(view_1408, [0, 2, 1, 3]);  view_1408 = None
        permute_214 = torch.ops.aten.permute.default(view_1409, [0, 2, 1, 3]);  view_1409 = None
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_212, permute_213, permute_214, None, True, 0.0, True);  permute_212 = permute_213 = permute_214 = None
        getitem_859 = _scaled_dot_product_cudnn_attention_19[0]
        getitem_860 = _scaled_dot_product_cudnn_attention_19[1]
        getitem_865 = _scaled_dot_product_cudnn_attention_19[6]
        getitem_866 = _scaled_dot_product_cudnn_attention_19[7];  _scaled_dot_product_cudnn_attention_19 = None
        permute_215 = torch.ops.aten.permute.default(getitem_859, [0, 2, 1, 3])
        view_1410 = torch.ops.aten.view.default(permute_215, [2, 8192, -1]);  permute_215 = None
        convert_element_type_644 = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16)
        all_gather_into_tensor_215 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_644, 32, '0');  convert_element_type_644 = None
        wait_tensor_254 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_215);  all_gather_into_tensor_215 = None
        permute_216 = torch.ops.aten.permute.default(wait_tensor_254, [1, 0]);  wait_tensor_254 = None
        view_1416 = torch.ops.aten.view.default(view_1410, [16384, 512]);  view_1410 = None
        mm_136 = torch.ops.aten.mm.default(view_1416, permute_216);  view_1416 = permute_216 = None
        view_1417 = torch.ops.aten.view.default(mm_136, [2, 8192, 4096]);  mm_136 = None
        split_86 = torch.ops.aten.split.Tensor(view_1417, 1024, 1);  view_1417 = None
        getitem_868 = split_86[0]
        getitem_869 = split_86[1]
        getitem_870 = split_86[2]
        getitem_871 = split_86[3]
        getitem_872 = split_86[4]
        getitem_873 = split_86[5]
        getitem_874 = split_86[6]
        getitem_875 = split_86[7];  split_86 = None
        cat_78 = torch.ops.aten.cat.default([getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875]);  getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = None
        reduce_scatter_tensor_39 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_78, 'sum', 8, '1');  cat_78 = None
        wait_tensor_255 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_39)
        add_77 = torch.ops.aten.add.Tensor(add_75, wait_tensor_255);  wait_tensor_255 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16)
        all_gather_into_tensor_216 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_647, 32, '0');  convert_element_type_647 = None
        wait_tensor_256 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_216);  all_gather_into_tensor_216 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_648, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_40, [2], True);  pow_40 = None
        add_78 = torch.ops.aten.add.Scalar(mean_39, 1e-05);  mean_39 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_156 = torch.ops.aten.mul.Tensor(convert_element_type_648, rsqrt_39);  convert_element_type_648 = rsqrt_39 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, wait_tensor_256);  mul_156 = wait_tensor_256 = None
        convert_element_type_649 = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        all_gather_into_tensor_217 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_649, 8, '1');  convert_element_type_649 = None
        wait_tensor_257 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_217);  all_gather_into_tensor_217 = None
        split_87 = torch.ops.aten.split.Tensor(wait_tensor_257, 2);  wait_tensor_257 = None
        getitem_876 = split_87[0]
        getitem_877 = split_87[1]
        getitem_878 = split_87[2]
        getitem_879 = split_87[3]
        getitem_880 = split_87[4]
        getitem_881 = split_87[5]
        getitem_882 = split_87[6]
        getitem_883 = split_87[7];  split_87 = None
        cat_79 = torch.ops.aten.cat.default([getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883], 1);  getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16)
        all_gather_into_tensor_218 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_650, 32, '0');  convert_element_type_650 = None
        wait_tensor_258 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_218);  all_gather_into_tensor_218 = None
        permute_217 = torch.ops.aten.permute.default(wait_tensor_258, [1, 0]);  wait_tensor_258 = None
        view_1428 = torch.ops.aten.view.default(cat_79, [16384, 4096]);  cat_79 = None
        mm_137 = torch.ops.aten.mm.default(view_1428, permute_217);  permute_217 = None
        view_1429 = torch.ops.aten.view.default(mm_137, [2, 8192, 1792])
        convert_element_type_653 = torch.ops.prims.convert_element_type.default(view_1429, torch.float32);  view_1429 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(convert_element_type_653)
        mul_158 = torch.ops.aten.mul.Tensor(convert_element_type_653, sigmoid_19);  convert_element_type_653 = sigmoid_19 = None
        convert_element_type_654 = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        convert_element_type_655 = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16)
        all_gather_into_tensor_219 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_655, 32, '0');  convert_element_type_655 = None
        wait_tensor_259 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_219);  all_gather_into_tensor_219 = None
        permute_218 = torch.ops.aten.permute.default(wait_tensor_259, [1, 0]);  wait_tensor_259 = None
        mm_138 = torch.ops.aten.mm.default(view_1428, permute_218);  view_1428 = permute_218 = None
        view_1436 = torch.ops.aten.view.default(mm_138, [2, 8192, 1792]);  mm_138 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_654, view_1436);  convert_element_type_654 = view_1436 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16)
        all_gather_into_tensor_220 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_658, 32, '0');  convert_element_type_658 = None
        wait_tensor_260 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_220);  all_gather_into_tensor_220 = None
        permute_219 = torch.ops.aten.permute.default(wait_tensor_260, [1, 0]);  wait_tensor_260 = None
        view_1443 = torch.ops.aten.view.default(mul_159, [16384, 1792]);  mul_159 = None
        mm_139 = torch.ops.aten.mm.default(view_1443, permute_219);  view_1443 = permute_219 = None
        view_1444 = torch.ops.aten.view.default(mm_139, [2, 8192, 4096]);  mm_139 = None
        split_88 = torch.ops.aten.split.Tensor(view_1444, 1024, 1);  view_1444 = None
        getitem_884 = split_88[0]
        getitem_885 = split_88[1]
        getitem_886 = split_88[2]
        getitem_887 = split_88[3]
        getitem_888 = split_88[4]
        getitem_889 = split_88[5]
        getitem_890 = split_88[6]
        getitem_891 = split_88[7];  split_88 = None
        cat_80 = torch.ops.aten.cat.default([getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891]);  getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = None
        reduce_scatter_tensor_40 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_80, 'sum', 8, '1');  cat_80 = None
        wait_tensor_261 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_40);  reduce_scatter_tensor_40 = None
        add_79 = torch.ops.aten.add.Tensor(add_77, wait_tensor_261);  add_77 = wait_tensor_261 = None
        convert_element_type_661 = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16)
        all_gather_into_tensor_221 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_661, 32, '0');  convert_element_type_661 = None
        wait_tensor_262 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_221);  all_gather_into_tensor_221 = None
        convert_element_type_662 = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_662, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_41, [2], True);  pow_41 = None
        add_80 = torch.ops.aten.add.Scalar(mean_40, 1e-05);  mean_40 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_662, rsqrt_40);  convert_element_type_662 = rsqrt_40 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, wait_tensor_262);  mul_160 = wait_tensor_262 = None
        convert_element_type_663 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        all_gather_into_tensor_222 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_663, 8, '1');  convert_element_type_663 = None
        wait_tensor_263 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_222);  all_gather_into_tensor_222 = None
        split_89 = torch.ops.aten.split.Tensor(wait_tensor_263, 2);  wait_tensor_263 = None
        getitem_892 = split_89[0]
        getitem_893 = split_89[1]
        getitem_894 = split_89[2]
        getitem_895 = split_89[3]
        getitem_896 = split_89[4]
        getitem_897 = split_89[5]
        getitem_898 = split_89[6]
        getitem_899 = split_89[7];  split_89 = None
        cat_81 = torch.ops.aten.cat.default([getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899], 1);  getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = None
        convert_element_type_664 = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16)
        all_gather_into_tensor_223 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_664, 32, '0');  convert_element_type_664 = None
        wait_tensor_264 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_223);  all_gather_into_tensor_223 = None
        permute_220 = torch.ops.aten.permute.default(wait_tensor_264, [1, 0]);  wait_tensor_264 = None
        view_1455 = torch.ops.aten.view.default(cat_81, [16384, 4096]);  cat_81 = None
        mm_140 = torch.ops.aten.mm.default(view_1455, permute_220);  permute_220 = None
        view_1456 = torch.ops.aten.view.default(mm_140, [2, 8192, 512])
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16)
        all_gather_into_tensor_224 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_667, 32, '0');  convert_element_type_667 = None
        wait_tensor_265 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_224);  all_gather_into_tensor_224 = None
        permute_221 = torch.ops.aten.permute.default(wait_tensor_265, [1, 0]);  wait_tensor_265 = None
        mm_141 = torch.ops.aten.mm.default(view_1455, permute_221);  permute_221 = None
        view_1463 = torch.ops.aten.view.default(mm_141, [2, 8192, 128]);  mm_141 = None
        convert_element_type_670 = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16)
        all_gather_into_tensor_225 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_670, 32, '0');  convert_element_type_670 = None
        wait_tensor_266 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_225);  all_gather_into_tensor_225 = None
        permute_222 = torch.ops.aten.permute.default(wait_tensor_266, [1, 0]);  wait_tensor_266 = None
        mm_142 = torch.ops.aten.mm.default(view_1455, permute_222);  view_1455 = permute_222 = None
        view_1470 = torch.ops.aten.view.default(mm_142, [2, 8192, 128])
        view_1472 = torch.ops.aten.view.default(view_1456, [2, 8192, -1, 128]);  view_1456 = None
        view_1473 = torch.ops.aten.view.default(view_1463, [2, 8192, -1, 128]);  view_1463 = None
        view_1474 = torch.ops.aten.view.default(view_1470, [2, 8192, -1, 128]);  view_1470 = None
        convert_element_type_673 = torch.ops.prims.convert_element_type.default(view_1472, torch.float32);  view_1472 = None
        view_1475 = torch.ops.aten.view.default(convert_element_type_673, [2, 8192, 4, -1, 2]);  convert_element_type_673 = None
        view_as_complex_40 = torch.ops.aten.view_as_complex.default(view_1475);  view_1475 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(view_1473, torch.float32);  view_1473 = None
        view_1476 = torch.ops.aten.view.default(convert_element_type_674, [2, 8192, 1, -1, 2]);  convert_element_type_674 = None
        view_as_complex_41 = torch.ops.aten.view_as_complex.default(view_1476);  view_1476 = None
        mul_162 = torch.ops.aten.mul.Tensor(view_as_complex_40, view_37);  view_as_complex_40 = None
        view_as_real_40 = torch.ops.aten.view_as_real.default(mul_162);  mul_162 = None
        view_1478 = torch.ops.aten.view.default(view_as_real_40, [2, 8192, 4, 128]);  view_as_real_40 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_as_complex_41, view_37);  view_as_complex_41 = None
        view_as_real_41 = torch.ops.aten.view_as_real.default(mul_163);  mul_163 = None
        view_1479 = torch.ops.aten.view.default(view_as_real_41, [2, 8192, 1, 128]);  view_as_real_41 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(view_1478, torch.bfloat16);  view_1478 = None
        convert_element_type_676 = torch.ops.prims.convert_element_type.default(view_1479, torch.bfloat16);  view_1479 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(convert_element_type_676, 3);  convert_element_type_676 = None
        expand_40 = torch.ops.aten.expand.default(unsqueeze_40, [2, 8192, 1, 4, 128]);  unsqueeze_40 = None
        view_1480 = torch.ops.aten.view.default(expand_40, [2, 8192, 4, 128]);  expand_40 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(view_1474, 3);  view_1474 = None
        expand_41 = torch.ops.aten.expand.default(unsqueeze_41, [2, 8192, 1, 4, 128]);  unsqueeze_41 = None
        view_1481 = torch.ops.aten.view.default(expand_41, [2, 8192, 4, 128]);  expand_41 = None
        permute_223 = torch.ops.aten.permute.default(convert_element_type_675, [0, 2, 1, 3]);  convert_element_type_675 = None
        permute_224 = torch.ops.aten.permute.default(view_1480, [0, 2, 1, 3]);  view_1480 = None
        permute_225 = torch.ops.aten.permute.default(view_1481, [0, 2, 1, 3]);  view_1481 = None
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_223, permute_224, permute_225, None, True, 0.0, True);  permute_223 = permute_224 = permute_225 = None
        getitem_900 = _scaled_dot_product_cudnn_attention_20[0]
        getitem_901 = _scaled_dot_product_cudnn_attention_20[1]
        getitem_906 = _scaled_dot_product_cudnn_attention_20[6]
        getitem_907 = _scaled_dot_product_cudnn_attention_20[7];  _scaled_dot_product_cudnn_attention_20 = None
        permute_226 = torch.ops.aten.permute.default(getitem_900, [0, 2, 1, 3])
        view_1482 = torch.ops.aten.view.default(permute_226, [2, 8192, -1]);  permute_226 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16)
        all_gather_into_tensor_226 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_677, 32, '0');  convert_element_type_677 = None
        wait_tensor_267 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_226);  all_gather_into_tensor_226 = None
        permute_227 = torch.ops.aten.permute.default(wait_tensor_267, [1, 0]);  wait_tensor_267 = None
        view_1488 = torch.ops.aten.view.default(view_1482, [16384, 512]);  view_1482 = None
        mm_143 = torch.ops.aten.mm.default(view_1488, permute_227);  view_1488 = permute_227 = None
        view_1489 = torch.ops.aten.view.default(mm_143, [2, 8192, 4096]);  mm_143 = None
        split_90 = torch.ops.aten.split.Tensor(view_1489, 1024, 1);  view_1489 = None
        getitem_909 = split_90[0]
        getitem_910 = split_90[1]
        getitem_911 = split_90[2]
        getitem_912 = split_90[3]
        getitem_913 = split_90[4]
        getitem_914 = split_90[5]
        getitem_915 = split_90[6]
        getitem_916 = split_90[7];  split_90 = None
        cat_82 = torch.ops.aten.cat.default([getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916]);  getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = None
        reduce_scatter_tensor_41 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_82, 'sum', 8, '1');  cat_82 = None
        wait_tensor_268 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_41)
        add_81 = torch.ops.aten.add.Tensor(add_79, wait_tensor_268);  wait_tensor_268 = None
        convert_element_type_680 = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16)
        all_gather_into_tensor_227 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_680, 32, '0');  convert_element_type_680 = None
        wait_tensor_269 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_227);  all_gather_into_tensor_227 = None
        convert_element_type_681 = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_681, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_42, [2], True);  pow_42 = None
        add_82 = torch.ops.aten.add.Scalar(mean_41, 1e-05);  mean_41 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_164 = torch.ops.aten.mul.Tensor(convert_element_type_681, rsqrt_41);  convert_element_type_681 = rsqrt_41 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, wait_tensor_269);  mul_164 = wait_tensor_269 = None
        convert_element_type_682 = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        all_gather_into_tensor_228 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_682, 8, '1');  convert_element_type_682 = None
        wait_tensor_270 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_228);  all_gather_into_tensor_228 = None
        split_91 = torch.ops.aten.split.Tensor(wait_tensor_270, 2);  wait_tensor_270 = None
        getitem_917 = split_91[0]
        getitem_918 = split_91[1]
        getitem_919 = split_91[2]
        getitem_920 = split_91[3]
        getitem_921 = split_91[4]
        getitem_922 = split_91[5]
        getitem_923 = split_91[6]
        getitem_924 = split_91[7];  split_91 = None
        cat_83 = torch.ops.aten.cat.default([getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924], 1);  getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16)
        all_gather_into_tensor_229 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_683, 32, '0');  convert_element_type_683 = None
        wait_tensor_271 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_229);  all_gather_into_tensor_229 = None
        permute_228 = torch.ops.aten.permute.default(wait_tensor_271, [1, 0]);  wait_tensor_271 = None
        view_1500 = torch.ops.aten.view.default(cat_83, [16384, 4096]);  cat_83 = None
        mm_144 = torch.ops.aten.mm.default(view_1500, permute_228);  permute_228 = None
        view_1501 = torch.ops.aten.view.default(mm_144, [2, 8192, 1792])
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_1501, torch.float32);  view_1501 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(convert_element_type_686)
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_686, sigmoid_20);  convert_element_type_686 = sigmoid_20 = None
        convert_element_type_687 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        convert_element_type_688 = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16)
        all_gather_into_tensor_230 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_688, 32, '0');  convert_element_type_688 = None
        wait_tensor_272 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_230);  all_gather_into_tensor_230 = None
        permute_229 = torch.ops.aten.permute.default(wait_tensor_272, [1, 0]);  wait_tensor_272 = None
        mm_145 = torch.ops.aten.mm.default(view_1500, permute_229);  view_1500 = permute_229 = None
        view_1508 = torch.ops.aten.view.default(mm_145, [2, 8192, 1792]);  mm_145 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_687, view_1508);  convert_element_type_687 = view_1508 = None
        convert_element_type_691 = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16)
        all_gather_into_tensor_231 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_691, 32, '0');  convert_element_type_691 = None
        wait_tensor_273 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_231);  all_gather_into_tensor_231 = None
        permute_230 = torch.ops.aten.permute.default(wait_tensor_273, [1, 0]);  wait_tensor_273 = None
        view_1515 = torch.ops.aten.view.default(mul_167, [16384, 1792]);  mul_167 = None
        mm_146 = torch.ops.aten.mm.default(view_1515, permute_230);  view_1515 = permute_230 = None
        view_1516 = torch.ops.aten.view.default(mm_146, [2, 8192, 4096]);  mm_146 = None
        split_92 = torch.ops.aten.split.Tensor(view_1516, 1024, 1);  view_1516 = None
        getitem_925 = split_92[0]
        getitem_926 = split_92[1]
        getitem_927 = split_92[2]
        getitem_928 = split_92[3]
        getitem_929 = split_92[4]
        getitem_930 = split_92[5]
        getitem_931 = split_92[6]
        getitem_932 = split_92[7];  split_92 = None
        cat_84 = torch.ops.aten.cat.default([getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932]);  getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = None
        reduce_scatter_tensor_42 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_84, 'sum', 8, '1');  cat_84 = None
        wait_tensor_274 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_42);  reduce_scatter_tensor_42 = None
        add_83 = torch.ops.aten.add.Tensor(add_81, wait_tensor_274);  add_81 = wait_tensor_274 = None
        convert_element_type_694 = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16)
        all_gather_into_tensor_232 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_694, 32, '0');  convert_element_type_694 = None
        wait_tensor_275 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_232);  all_gather_into_tensor_232 = None
        convert_element_type_695 = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 2)
        mean_42 = torch.ops.aten.mean.dim(pow_43, [2], True);  pow_43 = None
        add_84 = torch.ops.aten.add.Scalar(mean_42, 1e-05);  mean_42 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_168 = torch.ops.aten.mul.Tensor(convert_element_type_695, rsqrt_42);  convert_element_type_695 = rsqrt_42 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, wait_tensor_275);  mul_168 = wait_tensor_275 = None
        convert_element_type_696 = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        all_gather_into_tensor_233 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_696, 8, '1');  convert_element_type_696 = None
        wait_tensor_276 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_233);  all_gather_into_tensor_233 = None
        split_93 = torch.ops.aten.split.Tensor(wait_tensor_276, 2);  wait_tensor_276 = None
        getitem_933 = split_93[0]
        getitem_934 = split_93[1]
        getitem_935 = split_93[2]
        getitem_936 = split_93[3]
        getitem_937 = split_93[4]
        getitem_938 = split_93[5]
        getitem_939 = split_93[6]
        getitem_940 = split_93[7];  split_93 = None
        cat_85 = torch.ops.aten.cat.default([getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940], 1);  getitem_933 = getitem_934 = getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = None
        convert_element_type_697 = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16)
        all_gather_into_tensor_234 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_697, 32, '0');  convert_element_type_697 = None
        wait_tensor_277 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_234);  all_gather_into_tensor_234 = None
        permute_231 = torch.ops.aten.permute.default(wait_tensor_277, [1, 0]);  wait_tensor_277 = None
        view_1527 = torch.ops.aten.view.default(cat_85, [16384, 4096]);  cat_85 = None
        mm_147 = torch.ops.aten.mm.default(view_1527, permute_231);  permute_231 = None
        view_1528 = torch.ops.aten.view.default(mm_147, [2, 8192, 512])
        convert_element_type_700 = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16)
        all_gather_into_tensor_235 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_700, 32, '0');  convert_element_type_700 = None
        wait_tensor_278 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_235);  all_gather_into_tensor_235 = None
        permute_232 = torch.ops.aten.permute.default(wait_tensor_278, [1, 0]);  wait_tensor_278 = None
        mm_148 = torch.ops.aten.mm.default(view_1527, permute_232);  permute_232 = None
        view_1535 = torch.ops.aten.view.default(mm_148, [2, 8192, 128]);  mm_148 = None
        convert_element_type_703 = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16)
        all_gather_into_tensor_236 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_703, 32, '0');  convert_element_type_703 = None
        wait_tensor_279 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_236);  all_gather_into_tensor_236 = None
        permute_233 = torch.ops.aten.permute.default(wait_tensor_279, [1, 0]);  wait_tensor_279 = None
        mm_149 = torch.ops.aten.mm.default(view_1527, permute_233);  view_1527 = permute_233 = None
        view_1542 = torch.ops.aten.view.default(mm_149, [2, 8192, 128])
        view_1544 = torch.ops.aten.view.default(view_1528, [2, 8192, -1, 128]);  view_1528 = None
        view_1545 = torch.ops.aten.view.default(view_1535, [2, 8192, -1, 128]);  view_1535 = None
        view_1546 = torch.ops.aten.view.default(view_1542, [2, 8192, -1, 128]);  view_1542 = None
        convert_element_type_706 = torch.ops.prims.convert_element_type.default(view_1544, torch.float32);  view_1544 = None
        view_1547 = torch.ops.aten.view.default(convert_element_type_706, [2, 8192, 4, -1, 2]);  convert_element_type_706 = None
        view_as_complex_42 = torch.ops.aten.view_as_complex.default(view_1547);  view_1547 = None
        convert_element_type_707 = torch.ops.prims.convert_element_type.default(view_1545, torch.float32);  view_1545 = None
        view_1548 = torch.ops.aten.view.default(convert_element_type_707, [2, 8192, 1, -1, 2]);  convert_element_type_707 = None
        view_as_complex_43 = torch.ops.aten.view_as_complex.default(view_1548);  view_1548 = None
        mul_170 = torch.ops.aten.mul.Tensor(view_as_complex_42, view_37);  view_as_complex_42 = None
        view_as_real_42 = torch.ops.aten.view_as_real.default(mul_170);  mul_170 = None
        view_1550 = torch.ops.aten.view.default(view_as_real_42, [2, 8192, 4, 128]);  view_as_real_42 = None
        mul_171 = torch.ops.aten.mul.Tensor(view_as_complex_43, view_37);  view_as_complex_43 = None
        view_as_real_43 = torch.ops.aten.view_as_real.default(mul_171);  mul_171 = None
        view_1551 = torch.ops.aten.view.default(view_as_real_43, [2, 8192, 1, 128]);  view_as_real_43 = None
        convert_element_type_708 = torch.ops.prims.convert_element_type.default(view_1550, torch.bfloat16);  view_1550 = None
        convert_element_type_709 = torch.ops.prims.convert_element_type.default(view_1551, torch.bfloat16);  view_1551 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(convert_element_type_709, 3);  convert_element_type_709 = None
        expand_42 = torch.ops.aten.expand.default(unsqueeze_42, [2, 8192, 1, 4, 128]);  unsqueeze_42 = None
        view_1552 = torch.ops.aten.view.default(expand_42, [2, 8192, 4, 128]);  expand_42 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(view_1546, 3);  view_1546 = None
        expand_43 = torch.ops.aten.expand.default(unsqueeze_43, [2, 8192, 1, 4, 128]);  unsqueeze_43 = None
        view_1553 = torch.ops.aten.view.default(expand_43, [2, 8192, 4, 128]);  expand_43 = None
        permute_234 = torch.ops.aten.permute.default(convert_element_type_708, [0, 2, 1, 3]);  convert_element_type_708 = None
        permute_235 = torch.ops.aten.permute.default(view_1552, [0, 2, 1, 3]);  view_1552 = None
        permute_236 = torch.ops.aten.permute.default(view_1553, [0, 2, 1, 3]);  view_1553 = None
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_234, permute_235, permute_236, None, True, 0.0, True);  permute_234 = permute_235 = permute_236 = None
        getitem_941 = _scaled_dot_product_cudnn_attention_21[0]
        getitem_942 = _scaled_dot_product_cudnn_attention_21[1]
        getitem_947 = _scaled_dot_product_cudnn_attention_21[6]
        getitem_948 = _scaled_dot_product_cudnn_attention_21[7];  _scaled_dot_product_cudnn_attention_21 = None
        permute_237 = torch.ops.aten.permute.default(getitem_941, [0, 2, 1, 3])
        view_1554 = torch.ops.aten.view.default(permute_237, [2, 8192, -1]);  permute_237 = None
        convert_element_type_710 = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16)
        all_gather_into_tensor_237 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_710, 32, '0');  convert_element_type_710 = None
        wait_tensor_280 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_237);  all_gather_into_tensor_237 = None
        permute_238 = torch.ops.aten.permute.default(wait_tensor_280, [1, 0]);  wait_tensor_280 = None
        view_1560 = torch.ops.aten.view.default(view_1554, [16384, 512]);  view_1554 = None
        mm_150 = torch.ops.aten.mm.default(view_1560, permute_238);  view_1560 = permute_238 = None
        view_1561 = torch.ops.aten.view.default(mm_150, [2, 8192, 4096]);  mm_150 = None
        split_94 = torch.ops.aten.split.Tensor(view_1561, 1024, 1);  view_1561 = None
        getitem_950 = split_94[0]
        getitem_951 = split_94[1]
        getitem_952 = split_94[2]
        getitem_953 = split_94[3]
        getitem_954 = split_94[4]
        getitem_955 = split_94[5]
        getitem_956 = split_94[6]
        getitem_957 = split_94[7];  split_94 = None
        cat_86 = torch.ops.aten.cat.default([getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957]);  getitem_950 = getitem_951 = getitem_952 = getitem_953 = getitem_954 = getitem_955 = getitem_956 = getitem_957 = None
        reduce_scatter_tensor_43 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_86, 'sum', 8, '1');  cat_86 = None
        wait_tensor_281 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_43)
        add_85 = torch.ops.aten.add.Tensor(add_83, wait_tensor_281);  wait_tensor_281 = None
        convert_element_type_713 = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16)
        all_gather_into_tensor_238 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_713, 32, '0');  convert_element_type_713 = None
        wait_tensor_282 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_238);  all_gather_into_tensor_238 = None
        convert_element_type_714 = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_714, 2)
        mean_43 = torch.ops.aten.mean.dim(pow_44, [2], True);  pow_44 = None
        add_86 = torch.ops.aten.add.Scalar(mean_43, 1e-05);  mean_43 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_172 = torch.ops.aten.mul.Tensor(convert_element_type_714, rsqrt_43);  convert_element_type_714 = rsqrt_43 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, wait_tensor_282);  mul_172 = wait_tensor_282 = None
        convert_element_type_715 = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        all_gather_into_tensor_239 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_715, 8, '1');  convert_element_type_715 = None
        wait_tensor_283 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_239);  all_gather_into_tensor_239 = None
        split_95 = torch.ops.aten.split.Tensor(wait_tensor_283, 2);  wait_tensor_283 = None
        getitem_958 = split_95[0]
        getitem_959 = split_95[1]
        getitem_960 = split_95[2]
        getitem_961 = split_95[3]
        getitem_962 = split_95[4]
        getitem_963 = split_95[5]
        getitem_964 = split_95[6]
        getitem_965 = split_95[7];  split_95 = None
        cat_87 = torch.ops.aten.cat.default([getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965], 1);  getitem_958 = getitem_959 = getitem_960 = getitem_961 = getitem_962 = getitem_963 = getitem_964 = getitem_965 = None
        convert_element_type_716 = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16)
        all_gather_into_tensor_240 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_716, 32, '0');  convert_element_type_716 = None
        wait_tensor_284 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_240);  all_gather_into_tensor_240 = None
        permute_239 = torch.ops.aten.permute.default(wait_tensor_284, [1, 0]);  wait_tensor_284 = None
        view_1572 = torch.ops.aten.view.default(cat_87, [16384, 4096]);  cat_87 = None
        mm_151 = torch.ops.aten.mm.default(view_1572, permute_239);  permute_239 = None
        view_1573 = torch.ops.aten.view.default(mm_151, [2, 8192, 1792])
        convert_element_type_719 = torch.ops.prims.convert_element_type.default(view_1573, torch.float32);  view_1573 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(convert_element_type_719)
        mul_174 = torch.ops.aten.mul.Tensor(convert_element_type_719, sigmoid_21);  convert_element_type_719 = sigmoid_21 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        convert_element_type_721 = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16)
        all_gather_into_tensor_241 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_721, 32, '0');  convert_element_type_721 = None
        wait_tensor_285 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_241);  all_gather_into_tensor_241 = None
        permute_240 = torch.ops.aten.permute.default(wait_tensor_285, [1, 0]);  wait_tensor_285 = None
        mm_152 = torch.ops.aten.mm.default(view_1572, permute_240);  view_1572 = permute_240 = None
        view_1580 = torch.ops.aten.view.default(mm_152, [2, 8192, 1792]);  mm_152 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_720, view_1580);  convert_element_type_720 = view_1580 = None
        convert_element_type_724 = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16)
        all_gather_into_tensor_242 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_724, 32, '0');  convert_element_type_724 = None
        wait_tensor_286 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_242);  all_gather_into_tensor_242 = None
        permute_241 = torch.ops.aten.permute.default(wait_tensor_286, [1, 0]);  wait_tensor_286 = None
        view_1587 = torch.ops.aten.view.default(mul_175, [16384, 1792]);  mul_175 = None
        mm_153 = torch.ops.aten.mm.default(view_1587, permute_241);  view_1587 = permute_241 = None
        view_1588 = torch.ops.aten.view.default(mm_153, [2, 8192, 4096]);  mm_153 = None
        split_96 = torch.ops.aten.split.Tensor(view_1588, 1024, 1);  view_1588 = None
        getitem_966 = split_96[0]
        getitem_967 = split_96[1]
        getitem_968 = split_96[2]
        getitem_969 = split_96[3]
        getitem_970 = split_96[4]
        getitem_971 = split_96[5]
        getitem_972 = split_96[6]
        getitem_973 = split_96[7];  split_96 = None
        cat_88 = torch.ops.aten.cat.default([getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973]);  getitem_966 = getitem_967 = getitem_968 = getitem_969 = getitem_970 = getitem_971 = getitem_972 = getitem_973 = None
        reduce_scatter_tensor_44 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_88, 'sum', 8, '1');  cat_88 = None
        wait_tensor_287 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_44);  reduce_scatter_tensor_44 = None
        add_87 = torch.ops.aten.add.Tensor(add_85, wait_tensor_287);  add_85 = wait_tensor_287 = None
        convert_element_type_727 = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16)
        all_gather_into_tensor_243 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_727, 32, '0');  convert_element_type_727 = None
        wait_tensor_288 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_243);  all_gather_into_tensor_243 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_728, 2)
        mean_44 = torch.ops.aten.mean.dim(pow_45, [2], True);  pow_45 = None
        add_88 = torch.ops.aten.add.Scalar(mean_44, 1e-05);  mean_44 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_176 = torch.ops.aten.mul.Tensor(convert_element_type_728, rsqrt_44);  convert_element_type_728 = rsqrt_44 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, wait_tensor_288);  mul_176 = wait_tensor_288 = None
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        all_gather_into_tensor_244 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_729, 8, '1');  convert_element_type_729 = None
        wait_tensor_289 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_244);  all_gather_into_tensor_244 = None
        split_97 = torch.ops.aten.split.Tensor(wait_tensor_289, 2);  wait_tensor_289 = None
        getitem_974 = split_97[0]
        getitem_975 = split_97[1]
        getitem_976 = split_97[2]
        getitem_977 = split_97[3]
        getitem_978 = split_97[4]
        getitem_979 = split_97[5]
        getitem_980 = split_97[6]
        getitem_981 = split_97[7];  split_97 = None
        cat_89 = torch.ops.aten.cat.default([getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981], 1);  getitem_974 = getitem_975 = getitem_976 = getitem_977 = getitem_978 = getitem_979 = getitem_980 = getitem_981 = None
        convert_element_type_730 = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16)
        all_gather_into_tensor_245 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_730, 32, '0');  convert_element_type_730 = None
        wait_tensor_290 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_245);  all_gather_into_tensor_245 = None
        permute_242 = torch.ops.aten.permute.default(wait_tensor_290, [1, 0]);  wait_tensor_290 = None
        view_1599 = torch.ops.aten.view.default(cat_89, [16384, 4096]);  cat_89 = None
        mm_154 = torch.ops.aten.mm.default(view_1599, permute_242);  permute_242 = None
        view_1600 = torch.ops.aten.view.default(mm_154, [2, 8192, 512])
        convert_element_type_733 = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16)
        all_gather_into_tensor_246 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_733, 32, '0');  convert_element_type_733 = None
        wait_tensor_291 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_246);  all_gather_into_tensor_246 = None
        permute_243 = torch.ops.aten.permute.default(wait_tensor_291, [1, 0]);  wait_tensor_291 = None
        mm_155 = torch.ops.aten.mm.default(view_1599, permute_243);  permute_243 = None
        view_1607 = torch.ops.aten.view.default(mm_155, [2, 8192, 128]);  mm_155 = None
        convert_element_type_736 = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16)
        all_gather_into_tensor_247 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_736, 32, '0');  convert_element_type_736 = None
        wait_tensor_292 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_247);  all_gather_into_tensor_247 = None
        permute_244 = torch.ops.aten.permute.default(wait_tensor_292, [1, 0]);  wait_tensor_292 = None
        mm_156 = torch.ops.aten.mm.default(view_1599, permute_244);  view_1599 = permute_244 = None
        view_1614 = torch.ops.aten.view.default(mm_156, [2, 8192, 128])
        view_1616 = torch.ops.aten.view.default(view_1600, [2, 8192, -1, 128]);  view_1600 = None
        view_1617 = torch.ops.aten.view.default(view_1607, [2, 8192, -1, 128]);  view_1607 = None
        view_1618 = torch.ops.aten.view.default(view_1614, [2, 8192, -1, 128]);  view_1614 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(view_1616, torch.float32);  view_1616 = None
        view_1619 = torch.ops.aten.view.default(convert_element_type_739, [2, 8192, 4, -1, 2]);  convert_element_type_739 = None
        view_as_complex_44 = torch.ops.aten.view_as_complex.default(view_1619);  view_1619 = None
        convert_element_type_740 = torch.ops.prims.convert_element_type.default(view_1617, torch.float32);  view_1617 = None
        view_1620 = torch.ops.aten.view.default(convert_element_type_740, [2, 8192, 1, -1, 2]);  convert_element_type_740 = None
        view_as_complex_45 = torch.ops.aten.view_as_complex.default(view_1620);  view_1620 = None
        mul_178 = torch.ops.aten.mul.Tensor(view_as_complex_44, view_37);  view_as_complex_44 = None
        view_as_real_44 = torch.ops.aten.view_as_real.default(mul_178);  mul_178 = None
        view_1622 = torch.ops.aten.view.default(view_as_real_44, [2, 8192, 4, 128]);  view_as_real_44 = None
        mul_179 = torch.ops.aten.mul.Tensor(view_as_complex_45, view_37);  view_as_complex_45 = None
        view_as_real_45 = torch.ops.aten.view_as_real.default(mul_179);  mul_179 = None
        view_1623 = torch.ops.aten.view.default(view_as_real_45, [2, 8192, 1, 128]);  view_as_real_45 = None
        convert_element_type_741 = torch.ops.prims.convert_element_type.default(view_1622, torch.bfloat16);  view_1622 = None
        convert_element_type_742 = torch.ops.prims.convert_element_type.default(view_1623, torch.bfloat16);  view_1623 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(convert_element_type_742, 3);  convert_element_type_742 = None
        expand_44 = torch.ops.aten.expand.default(unsqueeze_44, [2, 8192, 1, 4, 128]);  unsqueeze_44 = None
        view_1624 = torch.ops.aten.view.default(expand_44, [2, 8192, 4, 128]);  expand_44 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_1618, 3);  view_1618 = None
        expand_45 = torch.ops.aten.expand.default(unsqueeze_45, [2, 8192, 1, 4, 128]);  unsqueeze_45 = None
        view_1625 = torch.ops.aten.view.default(expand_45, [2, 8192, 4, 128]);  expand_45 = None
        permute_245 = torch.ops.aten.permute.default(convert_element_type_741, [0, 2, 1, 3]);  convert_element_type_741 = None
        permute_246 = torch.ops.aten.permute.default(view_1624, [0, 2, 1, 3]);  view_1624 = None
        permute_247 = torch.ops.aten.permute.default(view_1625, [0, 2, 1, 3]);  view_1625 = None
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_245, permute_246, permute_247, None, True, 0.0, True);  permute_245 = permute_246 = permute_247 = None
        getitem_982 = _scaled_dot_product_cudnn_attention_22[0]
        getitem_983 = _scaled_dot_product_cudnn_attention_22[1]
        getitem_988 = _scaled_dot_product_cudnn_attention_22[6]
        getitem_989 = _scaled_dot_product_cudnn_attention_22[7];  _scaled_dot_product_cudnn_attention_22 = None
        permute_248 = torch.ops.aten.permute.default(getitem_982, [0, 2, 1, 3])
        view_1626 = torch.ops.aten.view.default(permute_248, [2, 8192, -1]);  permute_248 = None
        convert_element_type_743 = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16)
        all_gather_into_tensor_248 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_743, 32, '0');  convert_element_type_743 = None
        wait_tensor_293 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_248);  all_gather_into_tensor_248 = None
        permute_249 = torch.ops.aten.permute.default(wait_tensor_293, [1, 0]);  wait_tensor_293 = None
        view_1632 = torch.ops.aten.view.default(view_1626, [16384, 512]);  view_1626 = None
        mm_157 = torch.ops.aten.mm.default(view_1632, permute_249);  view_1632 = permute_249 = None
        view_1633 = torch.ops.aten.view.default(mm_157, [2, 8192, 4096]);  mm_157 = None
        split_98 = torch.ops.aten.split.Tensor(view_1633, 1024, 1);  view_1633 = None
        getitem_991 = split_98[0]
        getitem_992 = split_98[1]
        getitem_993 = split_98[2]
        getitem_994 = split_98[3]
        getitem_995 = split_98[4]
        getitem_996 = split_98[5]
        getitem_997 = split_98[6]
        getitem_998 = split_98[7];  split_98 = None
        cat_90 = torch.ops.aten.cat.default([getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998]);  getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = getitem_997 = getitem_998 = None
        reduce_scatter_tensor_45 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_90, 'sum', 8, '1');  cat_90 = None
        wait_tensor_294 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_45)
        add_89 = torch.ops.aten.add.Tensor(add_87, wait_tensor_294);  wait_tensor_294 = None
        convert_element_type_746 = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16)
        all_gather_into_tensor_249 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_746, 32, '0');  convert_element_type_746 = None
        wait_tensor_295 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_249);  all_gather_into_tensor_249 = None
        convert_element_type_747 = torch.ops.prims.convert_element_type.default(add_89, torch.float32)
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_747, 2)
        mean_45 = torch.ops.aten.mean.dim(pow_46, [2], True);  pow_46 = None
        add_90 = torch.ops.aten.add.Scalar(mean_45, 1e-05);  mean_45 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_180 = torch.ops.aten.mul.Tensor(convert_element_type_747, rsqrt_45);  convert_element_type_747 = rsqrt_45 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, wait_tensor_295);  mul_180 = wait_tensor_295 = None
        convert_element_type_748 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        all_gather_into_tensor_250 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_748, 8, '1');  convert_element_type_748 = None
        wait_tensor_296 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_250);  all_gather_into_tensor_250 = None
        split_99 = torch.ops.aten.split.Tensor(wait_tensor_296, 2);  wait_tensor_296 = None
        getitem_999 = split_99[0]
        getitem_1000 = split_99[1]
        getitem_1001 = split_99[2]
        getitem_1002 = split_99[3]
        getitem_1003 = split_99[4]
        getitem_1004 = split_99[5]
        getitem_1005 = split_99[6]
        getitem_1006 = split_99[7];  split_99 = None
        cat_91 = torch.ops.aten.cat.default([getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006], 1);  getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = getitem_1005 = getitem_1006 = None
        convert_element_type_749 = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16)
        all_gather_into_tensor_251 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_749, 32, '0');  convert_element_type_749 = None
        wait_tensor_297 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_251);  all_gather_into_tensor_251 = None
        permute_250 = torch.ops.aten.permute.default(wait_tensor_297, [1, 0]);  wait_tensor_297 = None
        view_1644 = torch.ops.aten.view.default(cat_91, [16384, 4096]);  cat_91 = None
        mm_158 = torch.ops.aten.mm.default(view_1644, permute_250);  permute_250 = None
        view_1645 = torch.ops.aten.view.default(mm_158, [2, 8192, 1792])
        convert_element_type_752 = torch.ops.prims.convert_element_type.default(view_1645, torch.float32);  view_1645 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(convert_element_type_752)
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_752, sigmoid_22);  convert_element_type_752 = sigmoid_22 = None
        convert_element_type_753 = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        convert_element_type_754 = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16)
        all_gather_into_tensor_252 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_754, 32, '0');  convert_element_type_754 = None
        wait_tensor_298 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_252);  all_gather_into_tensor_252 = None
        permute_251 = torch.ops.aten.permute.default(wait_tensor_298, [1, 0]);  wait_tensor_298 = None
        mm_159 = torch.ops.aten.mm.default(view_1644, permute_251);  view_1644 = permute_251 = None
        view_1652 = torch.ops.aten.view.default(mm_159, [2, 8192, 1792]);  mm_159 = None
        mul_183 = torch.ops.aten.mul.Tensor(convert_element_type_753, view_1652);  convert_element_type_753 = view_1652 = None
        convert_element_type_757 = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16)
        all_gather_into_tensor_253 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_757, 32, '0');  convert_element_type_757 = None
        wait_tensor_299 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_253);  all_gather_into_tensor_253 = None
        permute_252 = torch.ops.aten.permute.default(wait_tensor_299, [1, 0]);  wait_tensor_299 = None
        view_1659 = torch.ops.aten.view.default(mul_183, [16384, 1792]);  mul_183 = None
        mm_160 = torch.ops.aten.mm.default(view_1659, permute_252);  view_1659 = permute_252 = None
        view_1660 = torch.ops.aten.view.default(mm_160, [2, 8192, 4096]);  mm_160 = None
        split_100 = torch.ops.aten.split.Tensor(view_1660, 1024, 1);  view_1660 = None
        getitem_1007 = split_100[0]
        getitem_1008 = split_100[1]
        getitem_1009 = split_100[2]
        getitem_1010 = split_100[3]
        getitem_1011 = split_100[4]
        getitem_1012 = split_100[5]
        getitem_1013 = split_100[6]
        getitem_1014 = split_100[7];  split_100 = None
        cat_92 = torch.ops.aten.cat.default([getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014]);  getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = None
        reduce_scatter_tensor_46 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_92, 'sum', 8, '1');  cat_92 = None
        wait_tensor_300 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_46);  reduce_scatter_tensor_46 = None
        add_91 = torch.ops.aten.add.Tensor(add_89, wait_tensor_300);  add_89 = wait_tensor_300 = None
        convert_element_type_760 = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16)
        all_gather_into_tensor_254 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_760, 32, '0');  convert_element_type_760 = None
        wait_tensor_301 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_254);  all_gather_into_tensor_254 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_761, 2)
        mean_46 = torch.ops.aten.mean.dim(pow_47, [2], True);  pow_47 = None
        add_92 = torch.ops.aten.add.Scalar(mean_46, 1e-05);  mean_46 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_184 = torch.ops.aten.mul.Tensor(convert_element_type_761, rsqrt_46);  convert_element_type_761 = rsqrt_46 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, wait_tensor_301);  mul_184 = wait_tensor_301 = None
        convert_element_type_762 = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        all_gather_into_tensor_255 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_762, 8, '1');  convert_element_type_762 = None
        wait_tensor_302 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_255);  all_gather_into_tensor_255 = None
        split_101 = torch.ops.aten.split.Tensor(wait_tensor_302, 2);  wait_tensor_302 = None
        getitem_1015 = split_101[0]
        getitem_1016 = split_101[1]
        getitem_1017 = split_101[2]
        getitem_1018 = split_101[3]
        getitem_1019 = split_101[4]
        getitem_1020 = split_101[5]
        getitem_1021 = split_101[6]
        getitem_1022 = split_101[7];  split_101 = None
        cat_93 = torch.ops.aten.cat.default([getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022], 1);  getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = None
        convert_element_type_763 = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16)
        all_gather_into_tensor_256 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_763, 32, '0');  convert_element_type_763 = None
        wait_tensor_303 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_256);  all_gather_into_tensor_256 = None
        permute_253 = torch.ops.aten.permute.default(wait_tensor_303, [1, 0]);  wait_tensor_303 = None
        view_1671 = torch.ops.aten.view.default(cat_93, [16384, 4096]);  cat_93 = None
        mm_161 = torch.ops.aten.mm.default(view_1671, permute_253);  permute_253 = None
        view_1672 = torch.ops.aten.view.default(mm_161, [2, 8192, 512])
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16)
        all_gather_into_tensor_257 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_766, 32, '0');  convert_element_type_766 = None
        wait_tensor_304 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_257);  all_gather_into_tensor_257 = None
        permute_254 = torch.ops.aten.permute.default(wait_tensor_304, [1, 0]);  wait_tensor_304 = None
        mm_162 = torch.ops.aten.mm.default(view_1671, permute_254);  permute_254 = None
        view_1679 = torch.ops.aten.view.default(mm_162, [2, 8192, 128]);  mm_162 = None
        convert_element_type_769 = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16)
        all_gather_into_tensor_258 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_769, 32, '0');  convert_element_type_769 = None
        wait_tensor_305 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_258);  all_gather_into_tensor_258 = None
        permute_255 = torch.ops.aten.permute.default(wait_tensor_305, [1, 0]);  wait_tensor_305 = None
        mm_163 = torch.ops.aten.mm.default(view_1671, permute_255);  view_1671 = permute_255 = None
        view_1686 = torch.ops.aten.view.default(mm_163, [2, 8192, 128])
        view_1688 = torch.ops.aten.view.default(view_1672, [2, 8192, -1, 128]);  view_1672 = None
        view_1689 = torch.ops.aten.view.default(view_1679, [2, 8192, -1, 128]);  view_1679 = None
        view_1690 = torch.ops.aten.view.default(view_1686, [2, 8192, -1, 128]);  view_1686 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(view_1688, torch.float32);  view_1688 = None
        view_1691 = torch.ops.aten.view.default(convert_element_type_772, [2, 8192, 4, -1, 2]);  convert_element_type_772 = None
        view_as_complex_46 = torch.ops.aten.view_as_complex.default(view_1691);  view_1691 = None
        convert_element_type_773 = torch.ops.prims.convert_element_type.default(view_1689, torch.float32);  view_1689 = None
        view_1692 = torch.ops.aten.view.default(convert_element_type_773, [2, 8192, 1, -1, 2]);  convert_element_type_773 = None
        view_as_complex_47 = torch.ops.aten.view_as_complex.default(view_1692);  view_1692 = None
        mul_186 = torch.ops.aten.mul.Tensor(view_as_complex_46, view_37);  view_as_complex_46 = None
        view_as_real_46 = torch.ops.aten.view_as_real.default(mul_186);  mul_186 = None
        view_1694 = torch.ops.aten.view.default(view_as_real_46, [2, 8192, 4, 128]);  view_as_real_46 = None
        mul_187 = torch.ops.aten.mul.Tensor(view_as_complex_47, view_37);  view_as_complex_47 = None
        view_as_real_47 = torch.ops.aten.view_as_real.default(mul_187);  mul_187 = None
        view_1695 = torch.ops.aten.view.default(view_as_real_47, [2, 8192, 1, 128]);  view_as_real_47 = None
        convert_element_type_774 = torch.ops.prims.convert_element_type.default(view_1694, torch.bfloat16);  view_1694 = None
        convert_element_type_775 = torch.ops.prims.convert_element_type.default(view_1695, torch.bfloat16);  view_1695 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(convert_element_type_775, 3);  convert_element_type_775 = None
        expand_46 = torch.ops.aten.expand.default(unsqueeze_46, [2, 8192, 1, 4, 128]);  unsqueeze_46 = None
        view_1696 = torch.ops.aten.view.default(expand_46, [2, 8192, 4, 128]);  expand_46 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(view_1690, 3);  view_1690 = None
        expand_47 = torch.ops.aten.expand.default(unsqueeze_47, [2, 8192, 1, 4, 128]);  unsqueeze_47 = None
        view_1697 = torch.ops.aten.view.default(expand_47, [2, 8192, 4, 128]);  expand_47 = None
        permute_256 = torch.ops.aten.permute.default(convert_element_type_774, [0, 2, 1, 3]);  convert_element_type_774 = None
        permute_257 = torch.ops.aten.permute.default(view_1696, [0, 2, 1, 3]);  view_1696 = None
        permute_258 = torch.ops.aten.permute.default(view_1697, [0, 2, 1, 3]);  view_1697 = None
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_256, permute_257, permute_258, None, True, 0.0, True);  permute_256 = permute_257 = permute_258 = None
        getitem_1023 = _scaled_dot_product_cudnn_attention_23[0]
        getitem_1024 = _scaled_dot_product_cudnn_attention_23[1]
        getitem_1029 = _scaled_dot_product_cudnn_attention_23[6]
        getitem_1030 = _scaled_dot_product_cudnn_attention_23[7];  _scaled_dot_product_cudnn_attention_23 = None
        permute_259 = torch.ops.aten.permute.default(getitem_1023, [0, 2, 1, 3])
        view_1698 = torch.ops.aten.view.default(permute_259, [2, 8192, -1]);  permute_259 = None
        convert_element_type_776 = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16)
        all_gather_into_tensor_259 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_776, 32, '0');  convert_element_type_776 = None
        wait_tensor_306 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_259);  all_gather_into_tensor_259 = None
        permute_260 = torch.ops.aten.permute.default(wait_tensor_306, [1, 0]);  wait_tensor_306 = None
        view_1704 = torch.ops.aten.view.default(view_1698, [16384, 512]);  view_1698 = None
        mm_164 = torch.ops.aten.mm.default(view_1704, permute_260);  view_1704 = permute_260 = None
        view_1705 = torch.ops.aten.view.default(mm_164, [2, 8192, 4096]);  mm_164 = None
        split_102 = torch.ops.aten.split.Tensor(view_1705, 1024, 1);  view_1705 = None
        getitem_1032 = split_102[0]
        getitem_1033 = split_102[1]
        getitem_1034 = split_102[2]
        getitem_1035 = split_102[3]
        getitem_1036 = split_102[4]
        getitem_1037 = split_102[5]
        getitem_1038 = split_102[6]
        getitem_1039 = split_102[7];  split_102 = None
        cat_94 = torch.ops.aten.cat.default([getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039]);  getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = None
        reduce_scatter_tensor_47 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_94, 'sum', 8, '1');  cat_94 = None
        wait_tensor_307 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_47)
        add_93 = torch.ops.aten.add.Tensor(add_91, wait_tensor_307);  wait_tensor_307 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16)
        all_gather_into_tensor_260 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_779, 32, '0');  convert_element_type_779 = None
        wait_tensor_308 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_260);  all_gather_into_tensor_260 = None
        convert_element_type_780 = torch.ops.prims.convert_element_type.default(add_93, torch.float32)
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_780, 2)
        mean_47 = torch.ops.aten.mean.dim(pow_48, [2], True);  pow_48 = None
        add_94 = torch.ops.aten.add.Scalar(mean_47, 1e-05);  mean_47 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_188 = torch.ops.aten.mul.Tensor(convert_element_type_780, rsqrt_47);  convert_element_type_780 = rsqrt_47 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, wait_tensor_308);  mul_188 = wait_tensor_308 = None
        convert_element_type_781 = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        all_gather_into_tensor_261 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_781, 8, '1');  convert_element_type_781 = None
        wait_tensor_309 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_261);  all_gather_into_tensor_261 = None
        split_103 = torch.ops.aten.split.Tensor(wait_tensor_309, 2);  wait_tensor_309 = None
        getitem_1040 = split_103[0]
        getitem_1041 = split_103[1]
        getitem_1042 = split_103[2]
        getitem_1043 = split_103[3]
        getitem_1044 = split_103[4]
        getitem_1045 = split_103[5]
        getitem_1046 = split_103[6]
        getitem_1047 = split_103[7];  split_103 = None
        cat_95 = torch.ops.aten.cat.default([getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047], 1);  getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16)
        all_gather_into_tensor_262 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_782, 32, '0');  convert_element_type_782 = None
        wait_tensor_310 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_262);  all_gather_into_tensor_262 = None
        permute_261 = torch.ops.aten.permute.default(wait_tensor_310, [1, 0]);  wait_tensor_310 = None
        view_1716 = torch.ops.aten.view.default(cat_95, [16384, 4096]);  cat_95 = None
        mm_165 = torch.ops.aten.mm.default(view_1716, permute_261);  permute_261 = None
        view_1717 = torch.ops.aten.view.default(mm_165, [2, 8192, 1792])
        convert_element_type_785 = torch.ops.prims.convert_element_type.default(view_1717, torch.float32);  view_1717 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(convert_element_type_785)
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_785, sigmoid_23);  convert_element_type_785 = sigmoid_23 = None
        convert_element_type_786 = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        convert_element_type_787 = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16)
        all_gather_into_tensor_263 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_787, 32, '0');  convert_element_type_787 = None
        wait_tensor_311 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_263);  all_gather_into_tensor_263 = None
        permute_262 = torch.ops.aten.permute.default(wait_tensor_311, [1, 0]);  wait_tensor_311 = None
        mm_166 = torch.ops.aten.mm.default(view_1716, permute_262);  view_1716 = permute_262 = None
        view_1724 = torch.ops.aten.view.default(mm_166, [2, 8192, 1792]);  mm_166 = None
        mul_191 = torch.ops.aten.mul.Tensor(convert_element_type_786, view_1724);  convert_element_type_786 = view_1724 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16)
        all_gather_into_tensor_264 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_790, 32, '0');  convert_element_type_790 = None
        wait_tensor_312 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_264);  all_gather_into_tensor_264 = None
        permute_263 = torch.ops.aten.permute.default(wait_tensor_312, [1, 0]);  wait_tensor_312 = None
        view_1731 = torch.ops.aten.view.default(mul_191, [16384, 1792]);  mul_191 = None
        mm_167 = torch.ops.aten.mm.default(view_1731, permute_263);  view_1731 = permute_263 = None
        view_1732 = torch.ops.aten.view.default(mm_167, [2, 8192, 4096]);  mm_167 = None
        split_104 = torch.ops.aten.split.Tensor(view_1732, 1024, 1);  view_1732 = None
        getitem_1048 = split_104[0]
        getitem_1049 = split_104[1]
        getitem_1050 = split_104[2]
        getitem_1051 = split_104[3]
        getitem_1052 = split_104[4]
        getitem_1053 = split_104[5]
        getitem_1054 = split_104[6]
        getitem_1055 = split_104[7];  split_104 = None
        cat_96 = torch.ops.aten.cat.default([getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055]);  getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = None
        reduce_scatter_tensor_48 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_96, 'sum', 8, '1');  cat_96 = None
        wait_tensor_313 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_48);  reduce_scatter_tensor_48 = None
        add_95 = torch.ops.aten.add.Tensor(add_93, wait_tensor_313);  add_93 = wait_tensor_313 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16)
        all_gather_into_tensor_265 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_793, 32, '0');  convert_element_type_793 = None
        wait_tensor_314 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_265);  all_gather_into_tensor_265 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_794, 2)
        mean_48 = torch.ops.aten.mean.dim(pow_49, [2], True);  pow_49 = None
        add_96 = torch.ops.aten.add.Scalar(mean_48, 1e-05);  mean_48 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_794, rsqrt_48);  convert_element_type_794 = rsqrt_48 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, wait_tensor_314);  mul_192 = wait_tensor_314 = None
        convert_element_type_795 = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        all_gather_into_tensor_266 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_795, 8, '1');  convert_element_type_795 = None
        wait_tensor_315 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_266);  all_gather_into_tensor_266 = None
        split_105 = torch.ops.aten.split.Tensor(wait_tensor_315, 2);  wait_tensor_315 = None
        getitem_1056 = split_105[0]
        getitem_1057 = split_105[1]
        getitem_1058 = split_105[2]
        getitem_1059 = split_105[3]
        getitem_1060 = split_105[4]
        getitem_1061 = split_105[5]
        getitem_1062 = split_105[6]
        getitem_1063 = split_105[7];  split_105 = None
        cat_97 = torch.ops.aten.cat.default([getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063], 1);  getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = None
        convert_element_type_796 = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16)
        all_gather_into_tensor_267 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_796, 32, '0');  convert_element_type_796 = None
        wait_tensor_316 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_267);  all_gather_into_tensor_267 = None
        permute_264 = torch.ops.aten.permute.default(wait_tensor_316, [1, 0]);  wait_tensor_316 = None
        view_1743 = torch.ops.aten.view.default(cat_97, [16384, 4096]);  cat_97 = None
        mm_168 = torch.ops.aten.mm.default(view_1743, permute_264);  permute_264 = None
        view_1744 = torch.ops.aten.view.default(mm_168, [2, 8192, 512])
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16)
        all_gather_into_tensor_268 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_799, 32, '0');  convert_element_type_799 = None
        wait_tensor_317 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_268);  all_gather_into_tensor_268 = None
        permute_265 = torch.ops.aten.permute.default(wait_tensor_317, [1, 0]);  wait_tensor_317 = None
        mm_169 = torch.ops.aten.mm.default(view_1743, permute_265);  permute_265 = None
        view_1751 = torch.ops.aten.view.default(mm_169, [2, 8192, 128]);  mm_169 = None
        convert_element_type_802 = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16)
        all_gather_into_tensor_269 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_802, 32, '0');  convert_element_type_802 = None
        wait_tensor_318 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_269);  all_gather_into_tensor_269 = None
        permute_266 = torch.ops.aten.permute.default(wait_tensor_318, [1, 0]);  wait_tensor_318 = None
        mm_170 = torch.ops.aten.mm.default(view_1743, permute_266);  view_1743 = permute_266 = None
        view_1758 = torch.ops.aten.view.default(mm_170, [2, 8192, 128])
        view_1760 = torch.ops.aten.view.default(view_1744, [2, 8192, -1, 128]);  view_1744 = None
        view_1761 = torch.ops.aten.view.default(view_1751, [2, 8192, -1, 128]);  view_1751 = None
        view_1762 = torch.ops.aten.view.default(view_1758, [2, 8192, -1, 128]);  view_1758 = None
        convert_element_type_805 = torch.ops.prims.convert_element_type.default(view_1760, torch.float32);  view_1760 = None
        view_1763 = torch.ops.aten.view.default(convert_element_type_805, [2, 8192, 4, -1, 2]);  convert_element_type_805 = None
        view_as_complex_48 = torch.ops.aten.view_as_complex.default(view_1763);  view_1763 = None
        convert_element_type_806 = torch.ops.prims.convert_element_type.default(view_1761, torch.float32);  view_1761 = None
        view_1764 = torch.ops.aten.view.default(convert_element_type_806, [2, 8192, 1, -1, 2]);  convert_element_type_806 = None
        view_as_complex_49 = torch.ops.aten.view_as_complex.default(view_1764);  view_1764 = None
        mul_194 = torch.ops.aten.mul.Tensor(view_as_complex_48, view_37);  view_as_complex_48 = None
        view_as_real_48 = torch.ops.aten.view_as_real.default(mul_194);  mul_194 = None
        view_1766 = torch.ops.aten.view.default(view_as_real_48, [2, 8192, 4, 128]);  view_as_real_48 = None
        mul_195 = torch.ops.aten.mul.Tensor(view_as_complex_49, view_37);  view_as_complex_49 = None
        view_as_real_49 = torch.ops.aten.view_as_real.default(mul_195);  mul_195 = None
        view_1767 = torch.ops.aten.view.default(view_as_real_49, [2, 8192, 1, 128]);  view_as_real_49 = None
        convert_element_type_807 = torch.ops.prims.convert_element_type.default(view_1766, torch.bfloat16);  view_1766 = None
        convert_element_type_808 = torch.ops.prims.convert_element_type.default(view_1767, torch.bfloat16);  view_1767 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(convert_element_type_808, 3);  convert_element_type_808 = None
        expand_48 = torch.ops.aten.expand.default(unsqueeze_48, [2, 8192, 1, 4, 128]);  unsqueeze_48 = None
        view_1768 = torch.ops.aten.view.default(expand_48, [2, 8192, 4, 128]);  expand_48 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(view_1762, 3);  view_1762 = None
        expand_49 = torch.ops.aten.expand.default(unsqueeze_49, [2, 8192, 1, 4, 128]);  unsqueeze_49 = None
        view_1769 = torch.ops.aten.view.default(expand_49, [2, 8192, 4, 128]);  expand_49 = None
        permute_267 = torch.ops.aten.permute.default(convert_element_type_807, [0, 2, 1, 3]);  convert_element_type_807 = None
        permute_268 = torch.ops.aten.permute.default(view_1768, [0, 2, 1, 3]);  view_1768 = None
        permute_269 = torch.ops.aten.permute.default(view_1769, [0, 2, 1, 3]);  view_1769 = None
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_267, permute_268, permute_269, None, True, 0.0, True);  permute_267 = permute_268 = permute_269 = None
        getitem_1064 = _scaled_dot_product_cudnn_attention_24[0]
        getitem_1065 = _scaled_dot_product_cudnn_attention_24[1]
        getitem_1070 = _scaled_dot_product_cudnn_attention_24[6]
        getitem_1071 = _scaled_dot_product_cudnn_attention_24[7];  _scaled_dot_product_cudnn_attention_24 = None
        permute_270 = torch.ops.aten.permute.default(getitem_1064, [0, 2, 1, 3])
        view_1770 = torch.ops.aten.view.default(permute_270, [2, 8192, -1]);  permute_270 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16)
        all_gather_into_tensor_270 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_809, 32, '0');  convert_element_type_809 = None
        wait_tensor_319 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_270);  all_gather_into_tensor_270 = None
        permute_271 = torch.ops.aten.permute.default(wait_tensor_319, [1, 0]);  wait_tensor_319 = None
        view_1776 = torch.ops.aten.view.default(view_1770, [16384, 512]);  view_1770 = None
        mm_171 = torch.ops.aten.mm.default(view_1776, permute_271);  view_1776 = permute_271 = None
        view_1777 = torch.ops.aten.view.default(mm_171, [2, 8192, 4096]);  mm_171 = None
        split_106 = torch.ops.aten.split.Tensor(view_1777, 1024, 1);  view_1777 = None
        getitem_1073 = split_106[0]
        getitem_1074 = split_106[1]
        getitem_1075 = split_106[2]
        getitem_1076 = split_106[3]
        getitem_1077 = split_106[4]
        getitem_1078 = split_106[5]
        getitem_1079 = split_106[6]
        getitem_1080 = split_106[7];  split_106 = None
        cat_98 = torch.ops.aten.cat.default([getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080]);  getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = None
        reduce_scatter_tensor_49 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_98, 'sum', 8, '1');  cat_98 = None
        wait_tensor_320 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_49)
        add_97 = torch.ops.aten.add.Tensor(add_95, wait_tensor_320);  wait_tensor_320 = None
        convert_element_type_812 = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16)
        all_gather_into_tensor_271 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_812, 32, '0');  convert_element_type_812 = None
        wait_tensor_321 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_271);  all_gather_into_tensor_271 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_813, 2)
        mean_49 = torch.ops.aten.mean.dim(pow_50, [2], True);  pow_50 = None
        add_98 = torch.ops.aten.add.Scalar(mean_49, 1e-05);  mean_49 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_196 = torch.ops.aten.mul.Tensor(convert_element_type_813, rsqrt_49);  convert_element_type_813 = rsqrt_49 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, wait_tensor_321);  mul_196 = wait_tensor_321 = None
        convert_element_type_814 = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16);  mul_197 = None
        all_gather_into_tensor_272 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_814, 8, '1');  convert_element_type_814 = None
        wait_tensor_322 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_272);  all_gather_into_tensor_272 = None
        split_107 = torch.ops.aten.split.Tensor(wait_tensor_322, 2);  wait_tensor_322 = None
        getitem_1081 = split_107[0]
        getitem_1082 = split_107[1]
        getitem_1083 = split_107[2]
        getitem_1084 = split_107[3]
        getitem_1085 = split_107[4]
        getitem_1086 = split_107[5]
        getitem_1087 = split_107[6]
        getitem_1088 = split_107[7];  split_107 = None
        cat_99 = torch.ops.aten.cat.default([getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088], 1);  getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = None
        convert_element_type_815 = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16)
        all_gather_into_tensor_273 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_815, 32, '0');  convert_element_type_815 = None
        wait_tensor_323 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_273);  all_gather_into_tensor_273 = None
        permute_272 = torch.ops.aten.permute.default(wait_tensor_323, [1, 0]);  wait_tensor_323 = None
        view_1788 = torch.ops.aten.view.default(cat_99, [16384, 4096]);  cat_99 = None
        mm_172 = torch.ops.aten.mm.default(view_1788, permute_272);  permute_272 = None
        view_1789 = torch.ops.aten.view.default(mm_172, [2, 8192, 1792])
        convert_element_type_818 = torch.ops.prims.convert_element_type.default(view_1789, torch.float32);  view_1789 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(convert_element_type_818)
        mul_198 = torch.ops.aten.mul.Tensor(convert_element_type_818, sigmoid_24);  convert_element_type_818 = sigmoid_24 = None
        convert_element_type_819 = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16)
        all_gather_into_tensor_274 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_820, 32, '0');  convert_element_type_820 = None
        wait_tensor_324 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_274);  all_gather_into_tensor_274 = None
        permute_273 = torch.ops.aten.permute.default(wait_tensor_324, [1, 0]);  wait_tensor_324 = None
        mm_173 = torch.ops.aten.mm.default(view_1788, permute_273);  view_1788 = permute_273 = None
        view_1796 = torch.ops.aten.view.default(mm_173, [2, 8192, 1792]);  mm_173 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_819, view_1796);  convert_element_type_819 = view_1796 = None
        convert_element_type_823 = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16)
        all_gather_into_tensor_275 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_823, 32, '0');  convert_element_type_823 = None
        wait_tensor_325 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_275);  all_gather_into_tensor_275 = None
        permute_274 = torch.ops.aten.permute.default(wait_tensor_325, [1, 0]);  wait_tensor_325 = None
        view_1803 = torch.ops.aten.view.default(mul_199, [16384, 1792]);  mul_199 = None
        mm_174 = torch.ops.aten.mm.default(view_1803, permute_274);  view_1803 = permute_274 = None
        view_1804 = torch.ops.aten.view.default(mm_174, [2, 8192, 4096]);  mm_174 = None
        split_108 = torch.ops.aten.split.Tensor(view_1804, 1024, 1);  view_1804 = None
        getitem_1089 = split_108[0]
        getitem_1090 = split_108[1]
        getitem_1091 = split_108[2]
        getitem_1092 = split_108[3]
        getitem_1093 = split_108[4]
        getitem_1094 = split_108[5]
        getitem_1095 = split_108[6]
        getitem_1096 = split_108[7];  split_108 = None
        cat_100 = torch.ops.aten.cat.default([getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096]);  getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = None
        reduce_scatter_tensor_50 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_100, 'sum', 8, '1');  cat_100 = None
        wait_tensor_326 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_50);  reduce_scatter_tensor_50 = None
        add_99 = torch.ops.aten.add.Tensor(add_97, wait_tensor_326);  add_97 = wait_tensor_326 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16)
        all_gather_into_tensor_276 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_826, 32, '0');  convert_element_type_826 = None
        wait_tensor_327 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_276);  all_gather_into_tensor_276 = None
        convert_element_type_827 = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_827, 2)
        mean_50 = torch.ops.aten.mean.dim(pow_51, [2], True);  pow_51 = None
        add_100 = torch.ops.aten.add.Scalar(mean_50, 1e-05);  mean_50 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_827, rsqrt_50);  convert_element_type_827 = rsqrt_50 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, wait_tensor_327);  mul_200 = wait_tensor_327 = None
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        all_gather_into_tensor_277 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_828, 8, '1');  convert_element_type_828 = None
        wait_tensor_328 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_277);  all_gather_into_tensor_277 = None
        split_109 = torch.ops.aten.split.Tensor(wait_tensor_328, 2);  wait_tensor_328 = None
        getitem_1097 = split_109[0]
        getitem_1098 = split_109[1]
        getitem_1099 = split_109[2]
        getitem_1100 = split_109[3]
        getitem_1101 = split_109[4]
        getitem_1102 = split_109[5]
        getitem_1103 = split_109[6]
        getitem_1104 = split_109[7];  split_109 = None
        cat_101 = torch.ops.aten.cat.default([getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104], 1);  getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = None
        convert_element_type_829 = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16)
        all_gather_into_tensor_278 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_829, 32, '0');  convert_element_type_829 = None
        wait_tensor_329 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_278);  all_gather_into_tensor_278 = None
        permute_275 = torch.ops.aten.permute.default(wait_tensor_329, [1, 0]);  wait_tensor_329 = None
        view_1815 = torch.ops.aten.view.default(cat_101, [16384, 4096]);  cat_101 = None
        mm_175 = torch.ops.aten.mm.default(view_1815, permute_275);  permute_275 = None
        view_1816 = torch.ops.aten.view.default(mm_175, [2, 8192, 512])
        convert_element_type_832 = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16)
        all_gather_into_tensor_279 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_832, 32, '0');  convert_element_type_832 = None
        wait_tensor_330 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_279);  all_gather_into_tensor_279 = None
        permute_276 = torch.ops.aten.permute.default(wait_tensor_330, [1, 0]);  wait_tensor_330 = None
        mm_176 = torch.ops.aten.mm.default(view_1815, permute_276);  permute_276 = None
        view_1823 = torch.ops.aten.view.default(mm_176, [2, 8192, 128]);  mm_176 = None
        convert_element_type_835 = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16)
        all_gather_into_tensor_280 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_835, 32, '0');  convert_element_type_835 = None
        wait_tensor_331 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_280);  all_gather_into_tensor_280 = None
        permute_277 = torch.ops.aten.permute.default(wait_tensor_331, [1, 0]);  wait_tensor_331 = None
        mm_177 = torch.ops.aten.mm.default(view_1815, permute_277);  view_1815 = permute_277 = None
        view_1830 = torch.ops.aten.view.default(mm_177, [2, 8192, 128])
        view_1832 = torch.ops.aten.view.default(view_1816, [2, 8192, -1, 128]);  view_1816 = None
        view_1833 = torch.ops.aten.view.default(view_1823, [2, 8192, -1, 128]);  view_1823 = None
        view_1834 = torch.ops.aten.view.default(view_1830, [2, 8192, -1, 128]);  view_1830 = None
        convert_element_type_838 = torch.ops.prims.convert_element_type.default(view_1832, torch.float32);  view_1832 = None
        view_1835 = torch.ops.aten.view.default(convert_element_type_838, [2, 8192, 4, -1, 2]);  convert_element_type_838 = None
        view_as_complex_50 = torch.ops.aten.view_as_complex.default(view_1835);  view_1835 = None
        convert_element_type_839 = torch.ops.prims.convert_element_type.default(view_1833, torch.float32);  view_1833 = None
        view_1836 = torch.ops.aten.view.default(convert_element_type_839, [2, 8192, 1, -1, 2]);  convert_element_type_839 = None
        view_as_complex_51 = torch.ops.aten.view_as_complex.default(view_1836);  view_1836 = None
        mul_202 = torch.ops.aten.mul.Tensor(view_as_complex_50, view_37);  view_as_complex_50 = None
        view_as_real_50 = torch.ops.aten.view_as_real.default(mul_202);  mul_202 = None
        view_1838 = torch.ops.aten.view.default(view_as_real_50, [2, 8192, 4, 128]);  view_as_real_50 = None
        mul_203 = torch.ops.aten.mul.Tensor(view_as_complex_51, view_37);  view_as_complex_51 = None
        view_as_real_51 = torch.ops.aten.view_as_real.default(mul_203);  mul_203 = None
        view_1839 = torch.ops.aten.view.default(view_as_real_51, [2, 8192, 1, 128]);  view_as_real_51 = None
        convert_element_type_840 = torch.ops.prims.convert_element_type.default(view_1838, torch.bfloat16);  view_1838 = None
        convert_element_type_841 = torch.ops.prims.convert_element_type.default(view_1839, torch.bfloat16);  view_1839 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(convert_element_type_841, 3);  convert_element_type_841 = None
        expand_50 = torch.ops.aten.expand.default(unsqueeze_50, [2, 8192, 1, 4, 128]);  unsqueeze_50 = None
        view_1840 = torch.ops.aten.view.default(expand_50, [2, 8192, 4, 128]);  expand_50 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(view_1834, 3);  view_1834 = None
        expand_51 = torch.ops.aten.expand.default(unsqueeze_51, [2, 8192, 1, 4, 128]);  unsqueeze_51 = None
        view_1841 = torch.ops.aten.view.default(expand_51, [2, 8192, 4, 128]);  expand_51 = None
        permute_278 = torch.ops.aten.permute.default(convert_element_type_840, [0, 2, 1, 3]);  convert_element_type_840 = None
        permute_279 = torch.ops.aten.permute.default(view_1840, [0, 2, 1, 3]);  view_1840 = None
        permute_280 = torch.ops.aten.permute.default(view_1841, [0, 2, 1, 3]);  view_1841 = None
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_278, permute_279, permute_280, None, True, 0.0, True);  permute_278 = permute_279 = permute_280 = None
        getitem_1105 = _scaled_dot_product_cudnn_attention_25[0]
        getitem_1106 = _scaled_dot_product_cudnn_attention_25[1]
        getitem_1111 = _scaled_dot_product_cudnn_attention_25[6]
        getitem_1112 = _scaled_dot_product_cudnn_attention_25[7];  _scaled_dot_product_cudnn_attention_25 = None
        permute_281 = torch.ops.aten.permute.default(getitem_1105, [0, 2, 1, 3])
        view_1842 = torch.ops.aten.view.default(permute_281, [2, 8192, -1]);  permute_281 = None
        convert_element_type_842 = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16)
        all_gather_into_tensor_281 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_842, 32, '0');  convert_element_type_842 = None
        wait_tensor_332 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_281);  all_gather_into_tensor_281 = None
        permute_282 = torch.ops.aten.permute.default(wait_tensor_332, [1, 0]);  wait_tensor_332 = None
        view_1848 = torch.ops.aten.view.default(view_1842, [16384, 512]);  view_1842 = None
        mm_178 = torch.ops.aten.mm.default(view_1848, permute_282);  view_1848 = permute_282 = None
        view_1849 = torch.ops.aten.view.default(mm_178, [2, 8192, 4096]);  mm_178 = None
        split_110 = torch.ops.aten.split.Tensor(view_1849, 1024, 1);  view_1849 = None
        getitem_1114 = split_110[0]
        getitem_1115 = split_110[1]
        getitem_1116 = split_110[2]
        getitem_1117 = split_110[3]
        getitem_1118 = split_110[4]
        getitem_1119 = split_110[5]
        getitem_1120 = split_110[6]
        getitem_1121 = split_110[7];  split_110 = None
        cat_102 = torch.ops.aten.cat.default([getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121]);  getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = None
        reduce_scatter_tensor_51 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_102, 'sum', 8, '1');  cat_102 = None
        wait_tensor_333 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_51)
        add_101 = torch.ops.aten.add.Tensor(add_99, wait_tensor_333);  wait_tensor_333 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16)
        all_gather_into_tensor_282 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_845, 32, '0');  convert_element_type_845 = None
        wait_tensor_334 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_282);  all_gather_into_tensor_282 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_846, 2)
        mean_51 = torch.ops.aten.mean.dim(pow_52, [2], True);  pow_52 = None
        add_102 = torch.ops.aten.add.Scalar(mean_51, 1e-05);  mean_51 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_204 = torch.ops.aten.mul.Tensor(convert_element_type_846, rsqrt_51);  convert_element_type_846 = rsqrt_51 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, wait_tensor_334);  mul_204 = wait_tensor_334 = None
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16);  mul_205 = None
        all_gather_into_tensor_283 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_847, 8, '1');  convert_element_type_847 = None
        wait_tensor_335 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_283);  all_gather_into_tensor_283 = None
        split_111 = torch.ops.aten.split.Tensor(wait_tensor_335, 2);  wait_tensor_335 = None
        getitem_1122 = split_111[0]
        getitem_1123 = split_111[1]
        getitem_1124 = split_111[2]
        getitem_1125 = split_111[3]
        getitem_1126 = split_111[4]
        getitem_1127 = split_111[5]
        getitem_1128 = split_111[6]
        getitem_1129 = split_111[7];  split_111 = None
        cat_103 = torch.ops.aten.cat.default([getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129], 1);  getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16)
        all_gather_into_tensor_284 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_848, 32, '0');  convert_element_type_848 = None
        wait_tensor_336 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_284);  all_gather_into_tensor_284 = None
        permute_283 = torch.ops.aten.permute.default(wait_tensor_336, [1, 0]);  wait_tensor_336 = None
        view_1860 = torch.ops.aten.view.default(cat_103, [16384, 4096]);  cat_103 = None
        mm_179 = torch.ops.aten.mm.default(view_1860, permute_283);  permute_283 = None
        view_1861 = torch.ops.aten.view.default(mm_179, [2, 8192, 1792])
        convert_element_type_851 = torch.ops.prims.convert_element_type.default(view_1861, torch.float32);  view_1861 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(convert_element_type_851)
        mul_206 = torch.ops.aten.mul.Tensor(convert_element_type_851, sigmoid_25);  convert_element_type_851 = sigmoid_25 = None
        convert_element_type_852 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16)
        all_gather_into_tensor_285 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_853, 32, '0');  convert_element_type_853 = None
        wait_tensor_337 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_285);  all_gather_into_tensor_285 = None
        permute_284 = torch.ops.aten.permute.default(wait_tensor_337, [1, 0]);  wait_tensor_337 = None
        mm_180 = torch.ops.aten.mm.default(view_1860, permute_284);  view_1860 = permute_284 = None
        view_1868 = torch.ops.aten.view.default(mm_180, [2, 8192, 1792]);  mm_180 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_852, view_1868);  convert_element_type_852 = view_1868 = None
        convert_element_type_856 = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16)
        all_gather_into_tensor_286 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_856, 32, '0');  convert_element_type_856 = None
        wait_tensor_338 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_286);  all_gather_into_tensor_286 = None
        permute_285 = torch.ops.aten.permute.default(wait_tensor_338, [1, 0]);  wait_tensor_338 = None
        view_1875 = torch.ops.aten.view.default(mul_207, [16384, 1792]);  mul_207 = None
        mm_181 = torch.ops.aten.mm.default(view_1875, permute_285);  view_1875 = permute_285 = None
        view_1876 = torch.ops.aten.view.default(mm_181, [2, 8192, 4096]);  mm_181 = None
        split_112 = torch.ops.aten.split.Tensor(view_1876, 1024, 1);  view_1876 = None
        getitem_1130 = split_112[0]
        getitem_1131 = split_112[1]
        getitem_1132 = split_112[2]
        getitem_1133 = split_112[3]
        getitem_1134 = split_112[4]
        getitem_1135 = split_112[5]
        getitem_1136 = split_112[6]
        getitem_1137 = split_112[7];  split_112 = None
        cat_104 = torch.ops.aten.cat.default([getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137]);  getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = None
        reduce_scatter_tensor_52 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_104, 'sum', 8, '1');  cat_104 = None
        wait_tensor_339 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_52);  reduce_scatter_tensor_52 = None
        add_103 = torch.ops.aten.add.Tensor(add_101, wait_tensor_339);  add_101 = wait_tensor_339 = None
        convert_element_type_859 = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16)
        all_gather_into_tensor_287 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_859, 32, '0');  convert_element_type_859 = None
        wait_tensor_340 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_287);  all_gather_into_tensor_287 = None
        convert_element_type_860 = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_860, 2)
        mean_52 = torch.ops.aten.mean.dim(pow_53, [2], True);  pow_53 = None
        add_104 = torch.ops.aten.add.Scalar(mean_52, 1e-05);  mean_52 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_860, rsqrt_52);  convert_element_type_860 = rsqrt_52 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, wait_tensor_340);  mul_208 = wait_tensor_340 = None
        convert_element_type_861 = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        all_gather_into_tensor_288 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_861, 8, '1');  convert_element_type_861 = None
        wait_tensor_341 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_288);  all_gather_into_tensor_288 = None
        split_113 = torch.ops.aten.split.Tensor(wait_tensor_341, 2);  wait_tensor_341 = None
        getitem_1138 = split_113[0]
        getitem_1139 = split_113[1]
        getitem_1140 = split_113[2]
        getitem_1141 = split_113[3]
        getitem_1142 = split_113[4]
        getitem_1143 = split_113[5]
        getitem_1144 = split_113[6]
        getitem_1145 = split_113[7];  split_113 = None
        cat_105 = torch.ops.aten.cat.default([getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145], 1);  getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = None
        convert_element_type_862 = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16)
        all_gather_into_tensor_289 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_862, 32, '0');  convert_element_type_862 = None
        wait_tensor_342 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_289);  all_gather_into_tensor_289 = None
        permute_286 = torch.ops.aten.permute.default(wait_tensor_342, [1, 0]);  wait_tensor_342 = None
        view_1887 = torch.ops.aten.view.default(cat_105, [16384, 4096]);  cat_105 = None
        mm_182 = torch.ops.aten.mm.default(view_1887, permute_286);  permute_286 = None
        view_1888 = torch.ops.aten.view.default(mm_182, [2, 8192, 512])
        convert_element_type_865 = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16)
        all_gather_into_tensor_290 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_865, 32, '0');  convert_element_type_865 = None
        wait_tensor_343 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_290);  all_gather_into_tensor_290 = None
        permute_287 = torch.ops.aten.permute.default(wait_tensor_343, [1, 0]);  wait_tensor_343 = None
        mm_183 = torch.ops.aten.mm.default(view_1887, permute_287);  permute_287 = None
        view_1895 = torch.ops.aten.view.default(mm_183, [2, 8192, 128]);  mm_183 = None
        convert_element_type_868 = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16)
        all_gather_into_tensor_291 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_868, 32, '0');  convert_element_type_868 = None
        wait_tensor_344 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_291);  all_gather_into_tensor_291 = None
        permute_288 = torch.ops.aten.permute.default(wait_tensor_344, [1, 0]);  wait_tensor_344 = None
        mm_184 = torch.ops.aten.mm.default(view_1887, permute_288);  view_1887 = permute_288 = None
        view_1902 = torch.ops.aten.view.default(mm_184, [2, 8192, 128])
        view_1904 = torch.ops.aten.view.default(view_1888, [2, 8192, -1, 128]);  view_1888 = None
        view_1905 = torch.ops.aten.view.default(view_1895, [2, 8192, -1, 128]);  view_1895 = None
        view_1906 = torch.ops.aten.view.default(view_1902, [2, 8192, -1, 128]);  view_1902 = None
        convert_element_type_871 = torch.ops.prims.convert_element_type.default(view_1904, torch.float32);  view_1904 = None
        view_1907 = torch.ops.aten.view.default(convert_element_type_871, [2, 8192, 4, -1, 2]);  convert_element_type_871 = None
        view_as_complex_52 = torch.ops.aten.view_as_complex.default(view_1907);  view_1907 = None
        convert_element_type_872 = torch.ops.prims.convert_element_type.default(view_1905, torch.float32);  view_1905 = None
        view_1908 = torch.ops.aten.view.default(convert_element_type_872, [2, 8192, 1, -1, 2]);  convert_element_type_872 = None
        view_as_complex_53 = torch.ops.aten.view_as_complex.default(view_1908);  view_1908 = None
        mul_210 = torch.ops.aten.mul.Tensor(view_as_complex_52, view_37);  view_as_complex_52 = None
        view_as_real_52 = torch.ops.aten.view_as_real.default(mul_210);  mul_210 = None
        view_1910 = torch.ops.aten.view.default(view_as_real_52, [2, 8192, 4, 128]);  view_as_real_52 = None
        mul_211 = torch.ops.aten.mul.Tensor(view_as_complex_53, view_37);  view_as_complex_53 = None
        view_as_real_53 = torch.ops.aten.view_as_real.default(mul_211);  mul_211 = None
        view_1911 = torch.ops.aten.view.default(view_as_real_53, [2, 8192, 1, 128]);  view_as_real_53 = None
        convert_element_type_873 = torch.ops.prims.convert_element_type.default(view_1910, torch.bfloat16);  view_1910 = None
        convert_element_type_874 = torch.ops.prims.convert_element_type.default(view_1911, torch.bfloat16);  view_1911 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(convert_element_type_874, 3);  convert_element_type_874 = None
        expand_52 = torch.ops.aten.expand.default(unsqueeze_52, [2, 8192, 1, 4, 128]);  unsqueeze_52 = None
        view_1912 = torch.ops.aten.view.default(expand_52, [2, 8192, 4, 128]);  expand_52 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(view_1906, 3);  view_1906 = None
        expand_53 = torch.ops.aten.expand.default(unsqueeze_53, [2, 8192, 1, 4, 128]);  unsqueeze_53 = None
        view_1913 = torch.ops.aten.view.default(expand_53, [2, 8192, 4, 128]);  expand_53 = None
        permute_289 = torch.ops.aten.permute.default(convert_element_type_873, [0, 2, 1, 3]);  convert_element_type_873 = None
        permute_290 = torch.ops.aten.permute.default(view_1912, [0, 2, 1, 3]);  view_1912 = None
        permute_291 = torch.ops.aten.permute.default(view_1913, [0, 2, 1, 3]);  view_1913 = None
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_289, permute_290, permute_291, None, True, 0.0, True);  permute_289 = permute_290 = permute_291 = None
        getitem_1146 = _scaled_dot_product_cudnn_attention_26[0]
        getitem_1147 = _scaled_dot_product_cudnn_attention_26[1]
        getitem_1152 = _scaled_dot_product_cudnn_attention_26[6]
        getitem_1153 = _scaled_dot_product_cudnn_attention_26[7];  _scaled_dot_product_cudnn_attention_26 = None
        permute_292 = torch.ops.aten.permute.default(getitem_1146, [0, 2, 1, 3])
        view_1914 = torch.ops.aten.view.default(permute_292, [2, 8192, -1]);  permute_292 = None
        convert_element_type_875 = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16)
        all_gather_into_tensor_292 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_875, 32, '0');  convert_element_type_875 = None
        wait_tensor_345 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_292);  all_gather_into_tensor_292 = None
        permute_293 = torch.ops.aten.permute.default(wait_tensor_345, [1, 0]);  wait_tensor_345 = None
        view_1920 = torch.ops.aten.view.default(view_1914, [16384, 512]);  view_1914 = None
        mm_185 = torch.ops.aten.mm.default(view_1920, permute_293);  view_1920 = permute_293 = None
        view_1921 = torch.ops.aten.view.default(mm_185, [2, 8192, 4096]);  mm_185 = None
        split_114 = torch.ops.aten.split.Tensor(view_1921, 1024, 1);  view_1921 = None
        getitem_1155 = split_114[0]
        getitem_1156 = split_114[1]
        getitem_1157 = split_114[2]
        getitem_1158 = split_114[3]
        getitem_1159 = split_114[4]
        getitem_1160 = split_114[5]
        getitem_1161 = split_114[6]
        getitem_1162 = split_114[7];  split_114 = None
        cat_106 = torch.ops.aten.cat.default([getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162]);  getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = None
        reduce_scatter_tensor_53 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_106, 'sum', 8, '1');  cat_106 = None
        wait_tensor_346 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_53)
        add_105 = torch.ops.aten.add.Tensor(add_103, wait_tensor_346);  wait_tensor_346 = None
        convert_element_type_878 = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16)
        all_gather_into_tensor_293 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_878, 32, '0');  convert_element_type_878 = None
        wait_tensor_347 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_293);  all_gather_into_tensor_293 = None
        convert_element_type_879 = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_879, 2)
        mean_53 = torch.ops.aten.mean.dim(pow_54, [2], True);  pow_54 = None
        add_106 = torch.ops.aten.add.Scalar(mean_53, 1e-05);  mean_53 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_212 = torch.ops.aten.mul.Tensor(convert_element_type_879, rsqrt_53);  convert_element_type_879 = rsqrt_53 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, wait_tensor_347);  mul_212 = wait_tensor_347 = None
        convert_element_type_880 = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        all_gather_into_tensor_294 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_880, 8, '1');  convert_element_type_880 = None
        wait_tensor_348 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_294);  all_gather_into_tensor_294 = None
        split_115 = torch.ops.aten.split.Tensor(wait_tensor_348, 2);  wait_tensor_348 = None
        getitem_1163 = split_115[0]
        getitem_1164 = split_115[1]
        getitem_1165 = split_115[2]
        getitem_1166 = split_115[3]
        getitem_1167 = split_115[4]
        getitem_1168 = split_115[5]
        getitem_1169 = split_115[6]
        getitem_1170 = split_115[7];  split_115 = None
        cat_107 = torch.ops.aten.cat.default([getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170], 1);  getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = None
        convert_element_type_881 = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16)
        all_gather_into_tensor_295 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_881, 32, '0');  convert_element_type_881 = None
        wait_tensor_349 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_295);  all_gather_into_tensor_295 = None
        permute_294 = torch.ops.aten.permute.default(wait_tensor_349, [1, 0]);  wait_tensor_349 = None
        view_1932 = torch.ops.aten.view.default(cat_107, [16384, 4096]);  cat_107 = None
        mm_186 = torch.ops.aten.mm.default(view_1932, permute_294);  permute_294 = None
        view_1933 = torch.ops.aten.view.default(mm_186, [2, 8192, 1792])
        convert_element_type_884 = torch.ops.prims.convert_element_type.default(view_1933, torch.float32);  view_1933 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(convert_element_type_884)
        mul_214 = torch.ops.aten.mul.Tensor(convert_element_type_884, sigmoid_26);  convert_element_type_884 = sigmoid_26 = None
        convert_element_type_885 = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        convert_element_type_886 = torch.ops.prims.convert_element_type.default(primals_245, torch.bfloat16)
        all_gather_into_tensor_296 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_886, 32, '0');  convert_element_type_886 = None
        wait_tensor_350 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_296);  all_gather_into_tensor_296 = None
        permute_295 = torch.ops.aten.permute.default(wait_tensor_350, [1, 0]);  wait_tensor_350 = None
        mm_187 = torch.ops.aten.mm.default(view_1932, permute_295);  view_1932 = permute_295 = None
        view_1940 = torch.ops.aten.view.default(mm_187, [2, 8192, 1792]);  mm_187 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_885, view_1940);  convert_element_type_885 = view_1940 = None
        convert_element_type_889 = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16)
        all_gather_into_tensor_297 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_889, 32, '0');  convert_element_type_889 = None
        wait_tensor_351 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_297);  all_gather_into_tensor_297 = None
        permute_296 = torch.ops.aten.permute.default(wait_tensor_351, [1, 0]);  wait_tensor_351 = None
        view_1947 = torch.ops.aten.view.default(mul_215, [16384, 1792]);  mul_215 = None
        mm_188 = torch.ops.aten.mm.default(view_1947, permute_296);  view_1947 = permute_296 = None
        view_1948 = torch.ops.aten.view.default(mm_188, [2, 8192, 4096]);  mm_188 = None
        split_116 = torch.ops.aten.split.Tensor(view_1948, 1024, 1);  view_1948 = None
        getitem_1171 = split_116[0]
        getitem_1172 = split_116[1]
        getitem_1173 = split_116[2]
        getitem_1174 = split_116[3]
        getitem_1175 = split_116[4]
        getitem_1176 = split_116[5]
        getitem_1177 = split_116[6]
        getitem_1178 = split_116[7];  split_116 = None
        cat_108 = torch.ops.aten.cat.default([getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178]);  getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = None
        reduce_scatter_tensor_54 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_108, 'sum', 8, '1');  cat_108 = None
        wait_tensor_352 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_54);  reduce_scatter_tensor_54 = None
        add_107 = torch.ops.aten.add.Tensor(add_105, wait_tensor_352);  add_105 = wait_tensor_352 = None
        convert_element_type_892 = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16)
        all_gather_into_tensor_298 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_892, 32, '0');  convert_element_type_892 = None
        wait_tensor_353 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_298);  all_gather_into_tensor_298 = None
        convert_element_type_893 = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_893, 2)
        mean_54 = torch.ops.aten.mean.dim(pow_55, [2], True);  pow_55 = None
        add_108 = torch.ops.aten.add.Scalar(mean_54, 1e-05);  mean_54 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_216 = torch.ops.aten.mul.Tensor(convert_element_type_893, rsqrt_54);  convert_element_type_893 = rsqrt_54 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, wait_tensor_353);  mul_216 = wait_tensor_353 = None
        convert_element_type_894 = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16);  mul_217 = None
        all_gather_into_tensor_299 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_894, 8, '1');  convert_element_type_894 = None
        wait_tensor_354 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_299);  all_gather_into_tensor_299 = None
        split_117 = torch.ops.aten.split.Tensor(wait_tensor_354, 2);  wait_tensor_354 = None
        getitem_1179 = split_117[0]
        getitem_1180 = split_117[1]
        getitem_1181 = split_117[2]
        getitem_1182 = split_117[3]
        getitem_1183 = split_117[4]
        getitem_1184 = split_117[5]
        getitem_1185 = split_117[6]
        getitem_1186 = split_117[7];  split_117 = None
        cat_109 = torch.ops.aten.cat.default([getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186], 1);  getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = None
        convert_element_type_895 = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16)
        all_gather_into_tensor_300 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_895, 32, '0');  convert_element_type_895 = None
        wait_tensor_355 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_300);  all_gather_into_tensor_300 = None
        permute_297 = torch.ops.aten.permute.default(wait_tensor_355, [1, 0]);  wait_tensor_355 = None
        view_1959 = torch.ops.aten.view.default(cat_109, [16384, 4096]);  cat_109 = None
        mm_189 = torch.ops.aten.mm.default(view_1959, permute_297);  permute_297 = None
        view_1960 = torch.ops.aten.view.default(mm_189, [2, 8192, 512])
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16)
        all_gather_into_tensor_301 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_898, 32, '0');  convert_element_type_898 = None
        wait_tensor_356 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_301);  all_gather_into_tensor_301 = None
        permute_298 = torch.ops.aten.permute.default(wait_tensor_356, [1, 0]);  wait_tensor_356 = None
        mm_190 = torch.ops.aten.mm.default(view_1959, permute_298);  permute_298 = None
        view_1967 = torch.ops.aten.view.default(mm_190, [2, 8192, 128]);  mm_190 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16)
        all_gather_into_tensor_302 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_901, 32, '0');  convert_element_type_901 = None
        wait_tensor_357 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_302);  all_gather_into_tensor_302 = None
        permute_299 = torch.ops.aten.permute.default(wait_tensor_357, [1, 0]);  wait_tensor_357 = None
        mm_191 = torch.ops.aten.mm.default(view_1959, permute_299);  view_1959 = permute_299 = None
        view_1974 = torch.ops.aten.view.default(mm_191, [2, 8192, 128])
        view_1976 = torch.ops.aten.view.default(view_1960, [2, 8192, -1, 128]);  view_1960 = None
        view_1977 = torch.ops.aten.view.default(view_1967, [2, 8192, -1, 128]);  view_1967 = None
        view_1978 = torch.ops.aten.view.default(view_1974, [2, 8192, -1, 128]);  view_1974 = None
        convert_element_type_904 = torch.ops.prims.convert_element_type.default(view_1976, torch.float32);  view_1976 = None
        view_1979 = torch.ops.aten.view.default(convert_element_type_904, [2, 8192, 4, -1, 2]);  convert_element_type_904 = None
        view_as_complex_54 = torch.ops.aten.view_as_complex.default(view_1979);  view_1979 = None
        convert_element_type_905 = torch.ops.prims.convert_element_type.default(view_1977, torch.float32);  view_1977 = None
        view_1980 = torch.ops.aten.view.default(convert_element_type_905, [2, 8192, 1, -1, 2]);  convert_element_type_905 = None
        view_as_complex_55 = torch.ops.aten.view_as_complex.default(view_1980);  view_1980 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_as_complex_54, view_37);  view_as_complex_54 = None
        view_as_real_54 = torch.ops.aten.view_as_real.default(mul_218);  mul_218 = None
        view_1982 = torch.ops.aten.view.default(view_as_real_54, [2, 8192, 4, 128]);  view_as_real_54 = None
        mul_219 = torch.ops.aten.mul.Tensor(view_as_complex_55, view_37);  view_as_complex_55 = None
        view_as_real_55 = torch.ops.aten.view_as_real.default(mul_219);  mul_219 = None
        view_1983 = torch.ops.aten.view.default(view_as_real_55, [2, 8192, 1, 128]);  view_as_real_55 = None
        convert_element_type_906 = torch.ops.prims.convert_element_type.default(view_1982, torch.bfloat16);  view_1982 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(view_1983, torch.bfloat16);  view_1983 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(convert_element_type_907, 3);  convert_element_type_907 = None
        expand_54 = torch.ops.aten.expand.default(unsqueeze_54, [2, 8192, 1, 4, 128]);  unsqueeze_54 = None
        view_1984 = torch.ops.aten.view.default(expand_54, [2, 8192, 4, 128]);  expand_54 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(view_1978, 3);  view_1978 = None
        expand_55 = torch.ops.aten.expand.default(unsqueeze_55, [2, 8192, 1, 4, 128]);  unsqueeze_55 = None
        view_1985 = torch.ops.aten.view.default(expand_55, [2, 8192, 4, 128]);  expand_55 = None
        permute_300 = torch.ops.aten.permute.default(convert_element_type_906, [0, 2, 1, 3]);  convert_element_type_906 = None
        permute_301 = torch.ops.aten.permute.default(view_1984, [0, 2, 1, 3]);  view_1984 = None
        permute_302 = torch.ops.aten.permute.default(view_1985, [0, 2, 1, 3]);  view_1985 = None
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_300, permute_301, permute_302, None, True, 0.0, True);  permute_300 = permute_301 = permute_302 = None
        getitem_1187 = _scaled_dot_product_cudnn_attention_27[0]
        getitem_1188 = _scaled_dot_product_cudnn_attention_27[1]
        getitem_1193 = _scaled_dot_product_cudnn_attention_27[6]
        getitem_1194 = _scaled_dot_product_cudnn_attention_27[7];  _scaled_dot_product_cudnn_attention_27 = None
        permute_303 = torch.ops.aten.permute.default(getitem_1187, [0, 2, 1, 3])
        view_1986 = torch.ops.aten.view.default(permute_303, [2, 8192, -1]);  permute_303 = None
        convert_element_type_908 = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16)
        all_gather_into_tensor_303 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_908, 32, '0');  convert_element_type_908 = None
        wait_tensor_358 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_303);  all_gather_into_tensor_303 = None
        permute_304 = torch.ops.aten.permute.default(wait_tensor_358, [1, 0]);  wait_tensor_358 = None
        view_1992 = torch.ops.aten.view.default(view_1986, [16384, 512]);  view_1986 = None
        mm_192 = torch.ops.aten.mm.default(view_1992, permute_304);  view_1992 = permute_304 = None
        view_1993 = torch.ops.aten.view.default(mm_192, [2, 8192, 4096]);  mm_192 = None
        split_118 = torch.ops.aten.split.Tensor(view_1993, 1024, 1);  view_1993 = None
        getitem_1196 = split_118[0]
        getitem_1197 = split_118[1]
        getitem_1198 = split_118[2]
        getitem_1199 = split_118[3]
        getitem_1200 = split_118[4]
        getitem_1201 = split_118[5]
        getitem_1202 = split_118[6]
        getitem_1203 = split_118[7];  split_118 = None
        cat_110 = torch.ops.aten.cat.default([getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203]);  getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = None
        reduce_scatter_tensor_55 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_110, 'sum', 8, '1');  cat_110 = None
        wait_tensor_359 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_55)
        add_109 = torch.ops.aten.add.Tensor(add_107, wait_tensor_359);  wait_tensor_359 = None
        convert_element_type_911 = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16)
        all_gather_into_tensor_304 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_911, 32, '0');  convert_element_type_911 = None
        wait_tensor_360 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_304);  all_gather_into_tensor_304 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_912, 2)
        mean_55 = torch.ops.aten.mean.dim(pow_56, [2], True);  pow_56 = None
        add_110 = torch.ops.aten.add.Scalar(mean_55, 1e-05);  mean_55 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_912, rsqrt_55);  convert_element_type_912 = rsqrt_55 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, wait_tensor_360);  mul_220 = wait_tensor_360 = None
        convert_element_type_913 = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        all_gather_into_tensor_305 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_913, 8, '1');  convert_element_type_913 = None
        wait_tensor_361 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_305);  all_gather_into_tensor_305 = None
        split_119 = torch.ops.aten.split.Tensor(wait_tensor_361, 2);  wait_tensor_361 = None
        getitem_1204 = split_119[0]
        getitem_1205 = split_119[1]
        getitem_1206 = split_119[2]
        getitem_1207 = split_119[3]
        getitem_1208 = split_119[4]
        getitem_1209 = split_119[5]
        getitem_1210 = split_119[6]
        getitem_1211 = split_119[7];  split_119 = None
        cat_111 = torch.ops.aten.cat.default([getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211], 1);  getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = None
        convert_element_type_914 = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16)
        all_gather_into_tensor_306 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_914, 32, '0');  convert_element_type_914 = None
        wait_tensor_362 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_306);  all_gather_into_tensor_306 = None
        permute_305 = torch.ops.aten.permute.default(wait_tensor_362, [1, 0]);  wait_tensor_362 = None
        view_2004 = torch.ops.aten.view.default(cat_111, [16384, 4096]);  cat_111 = None
        mm_193 = torch.ops.aten.mm.default(view_2004, permute_305);  permute_305 = None
        view_2005 = torch.ops.aten.view.default(mm_193, [2, 8192, 1792])
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(view_2005, torch.float32);  view_2005 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(convert_element_type_917)
        mul_222 = torch.ops.aten.mul.Tensor(convert_element_type_917, sigmoid_27);  convert_element_type_917 = sigmoid_27 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None
        convert_element_type_919 = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16)
        all_gather_into_tensor_307 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_919, 32, '0');  convert_element_type_919 = None
        wait_tensor_363 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_307);  all_gather_into_tensor_307 = None
        permute_306 = torch.ops.aten.permute.default(wait_tensor_363, [1, 0]);  wait_tensor_363 = None
        mm_194 = torch.ops.aten.mm.default(view_2004, permute_306);  view_2004 = permute_306 = None
        view_2012 = torch.ops.aten.view.default(mm_194, [2, 8192, 1792]);  mm_194 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_918, view_2012);  convert_element_type_918 = view_2012 = None
        convert_element_type_922 = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16)
        all_gather_into_tensor_308 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_922, 32, '0');  convert_element_type_922 = None
        wait_tensor_364 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_308);  all_gather_into_tensor_308 = None
        permute_307 = torch.ops.aten.permute.default(wait_tensor_364, [1, 0]);  wait_tensor_364 = None
        view_2019 = torch.ops.aten.view.default(mul_223, [16384, 1792]);  mul_223 = None
        mm_195 = torch.ops.aten.mm.default(view_2019, permute_307);  view_2019 = permute_307 = None
        view_2020 = torch.ops.aten.view.default(mm_195, [2, 8192, 4096]);  mm_195 = None
        split_120 = torch.ops.aten.split.Tensor(view_2020, 1024, 1);  view_2020 = None
        getitem_1212 = split_120[0]
        getitem_1213 = split_120[1]
        getitem_1214 = split_120[2]
        getitem_1215 = split_120[3]
        getitem_1216 = split_120[4]
        getitem_1217 = split_120[5]
        getitem_1218 = split_120[6]
        getitem_1219 = split_120[7];  split_120 = None
        cat_112 = torch.ops.aten.cat.default([getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219]);  getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = None
        reduce_scatter_tensor_56 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_112, 'sum', 8, '1');  cat_112 = None
        wait_tensor_365 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_56);  reduce_scatter_tensor_56 = None
        add_111 = torch.ops.aten.add.Tensor(add_109, wait_tensor_365);  add_109 = wait_tensor_365 = None
        convert_element_type_925 = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16)
        all_gather_into_tensor_309 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_925, 32, '0');  convert_element_type_925 = None
        wait_tensor_366 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_309);  all_gather_into_tensor_309 = None
        convert_element_type_926 = torch.ops.prims.convert_element_type.default(add_111, torch.float32)
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_926, 2)
        mean_56 = torch.ops.aten.mean.dim(pow_57, [2], True);  pow_57 = None
        add_112 = torch.ops.aten.add.Scalar(mean_56, 1e-05);  mean_56 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_224 = torch.ops.aten.mul.Tensor(convert_element_type_926, rsqrt_56);  convert_element_type_926 = rsqrt_56 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, wait_tensor_366);  mul_224 = wait_tensor_366 = None
        convert_element_type_927 = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        all_gather_into_tensor_310 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_927, 8, '1');  convert_element_type_927 = None
        wait_tensor_367 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_310);  all_gather_into_tensor_310 = None
        split_121 = torch.ops.aten.split.Tensor(wait_tensor_367, 2);  wait_tensor_367 = None
        getitem_1220 = split_121[0]
        getitem_1221 = split_121[1]
        getitem_1222 = split_121[2]
        getitem_1223 = split_121[3]
        getitem_1224 = split_121[4]
        getitem_1225 = split_121[5]
        getitem_1226 = split_121[6]
        getitem_1227 = split_121[7];  split_121 = None
        cat_113 = torch.ops.aten.cat.default([getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227], 1);  getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = None
        convert_element_type_928 = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16)
        all_gather_into_tensor_311 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_928, 32, '0');  convert_element_type_928 = None
        wait_tensor_368 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_311);  all_gather_into_tensor_311 = None
        permute_308 = torch.ops.aten.permute.default(wait_tensor_368, [1, 0]);  wait_tensor_368 = None
        view_2031 = torch.ops.aten.view.default(cat_113, [16384, 4096]);  cat_113 = None
        mm_196 = torch.ops.aten.mm.default(view_2031, permute_308);  permute_308 = None
        view_2032 = torch.ops.aten.view.default(mm_196, [2, 8192, 512])
        convert_element_type_931 = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16)
        all_gather_into_tensor_312 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_931, 32, '0');  convert_element_type_931 = None
        wait_tensor_369 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_312);  all_gather_into_tensor_312 = None
        permute_309 = torch.ops.aten.permute.default(wait_tensor_369, [1, 0]);  wait_tensor_369 = None
        mm_197 = torch.ops.aten.mm.default(view_2031, permute_309);  permute_309 = None
        view_2039 = torch.ops.aten.view.default(mm_197, [2, 8192, 128]);  mm_197 = None
        convert_element_type_934 = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16)
        all_gather_into_tensor_313 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_934, 32, '0');  convert_element_type_934 = None
        wait_tensor_370 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_313);  all_gather_into_tensor_313 = None
        permute_310 = torch.ops.aten.permute.default(wait_tensor_370, [1, 0]);  wait_tensor_370 = None
        mm_198 = torch.ops.aten.mm.default(view_2031, permute_310);  view_2031 = permute_310 = None
        view_2046 = torch.ops.aten.view.default(mm_198, [2, 8192, 128])
        view_2048 = torch.ops.aten.view.default(view_2032, [2, 8192, -1, 128]);  view_2032 = None
        view_2049 = torch.ops.aten.view.default(view_2039, [2, 8192, -1, 128]);  view_2039 = None
        view_2050 = torch.ops.aten.view.default(view_2046, [2, 8192, -1, 128]);  view_2046 = None
        convert_element_type_937 = torch.ops.prims.convert_element_type.default(view_2048, torch.float32);  view_2048 = None
        view_2051 = torch.ops.aten.view.default(convert_element_type_937, [2, 8192, 4, -1, 2]);  convert_element_type_937 = None
        view_as_complex_56 = torch.ops.aten.view_as_complex.default(view_2051);  view_2051 = None
        convert_element_type_938 = torch.ops.prims.convert_element_type.default(view_2049, torch.float32);  view_2049 = None
        view_2052 = torch.ops.aten.view.default(convert_element_type_938, [2, 8192, 1, -1, 2]);  convert_element_type_938 = None
        view_as_complex_57 = torch.ops.aten.view_as_complex.default(view_2052);  view_2052 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_as_complex_56, view_37);  view_as_complex_56 = None
        view_as_real_56 = torch.ops.aten.view_as_real.default(mul_226);  mul_226 = None
        view_2054 = torch.ops.aten.view.default(view_as_real_56, [2, 8192, 4, 128]);  view_as_real_56 = None
        mul_227 = torch.ops.aten.mul.Tensor(view_as_complex_57, view_37);  view_as_complex_57 = None
        view_as_real_57 = torch.ops.aten.view_as_real.default(mul_227);  mul_227 = None
        view_2055 = torch.ops.aten.view.default(view_as_real_57, [2, 8192, 1, 128]);  view_as_real_57 = None
        convert_element_type_939 = torch.ops.prims.convert_element_type.default(view_2054, torch.bfloat16);  view_2054 = None
        convert_element_type_940 = torch.ops.prims.convert_element_type.default(view_2055, torch.bfloat16);  view_2055 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(convert_element_type_940, 3);  convert_element_type_940 = None
        expand_56 = torch.ops.aten.expand.default(unsqueeze_56, [2, 8192, 1, 4, 128]);  unsqueeze_56 = None
        view_2056 = torch.ops.aten.view.default(expand_56, [2, 8192, 4, 128]);  expand_56 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(view_2050, 3);  view_2050 = None
        expand_57 = torch.ops.aten.expand.default(unsqueeze_57, [2, 8192, 1, 4, 128]);  unsqueeze_57 = None
        view_2057 = torch.ops.aten.view.default(expand_57, [2, 8192, 4, 128]);  expand_57 = None
        permute_311 = torch.ops.aten.permute.default(convert_element_type_939, [0, 2, 1, 3]);  convert_element_type_939 = None
        permute_312 = torch.ops.aten.permute.default(view_2056, [0, 2, 1, 3]);  view_2056 = None
        permute_313 = torch.ops.aten.permute.default(view_2057, [0, 2, 1, 3]);  view_2057 = None
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_311, permute_312, permute_313, None, True, 0.0, True);  permute_311 = permute_312 = permute_313 = None
        getitem_1228 = _scaled_dot_product_cudnn_attention_28[0]
        getitem_1229 = _scaled_dot_product_cudnn_attention_28[1]
        getitem_1234 = _scaled_dot_product_cudnn_attention_28[6]
        getitem_1235 = _scaled_dot_product_cudnn_attention_28[7];  _scaled_dot_product_cudnn_attention_28 = None
        permute_314 = torch.ops.aten.permute.default(getitem_1228, [0, 2, 1, 3])
        view_2058 = torch.ops.aten.view.default(permute_314, [2, 8192, -1]);  permute_314 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16)
        all_gather_into_tensor_314 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_941, 32, '0');  convert_element_type_941 = None
        wait_tensor_371 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_314);  all_gather_into_tensor_314 = None
        permute_315 = torch.ops.aten.permute.default(wait_tensor_371, [1, 0]);  wait_tensor_371 = None
        view_2064 = torch.ops.aten.view.default(view_2058, [16384, 512]);  view_2058 = None
        mm_199 = torch.ops.aten.mm.default(view_2064, permute_315);  view_2064 = permute_315 = None
        view_2065 = torch.ops.aten.view.default(mm_199, [2, 8192, 4096]);  mm_199 = None
        split_122 = torch.ops.aten.split.Tensor(view_2065, 1024, 1);  view_2065 = None
        getitem_1237 = split_122[0]
        getitem_1238 = split_122[1]
        getitem_1239 = split_122[2]
        getitem_1240 = split_122[3]
        getitem_1241 = split_122[4]
        getitem_1242 = split_122[5]
        getitem_1243 = split_122[6]
        getitem_1244 = split_122[7];  split_122 = None
        cat_114 = torch.ops.aten.cat.default([getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244]);  getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = None
        reduce_scatter_tensor_57 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_114, 'sum', 8, '1');  cat_114 = None
        wait_tensor_372 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_57)
        add_113 = torch.ops.aten.add.Tensor(add_111, wait_tensor_372);  wait_tensor_372 = None
        convert_element_type_944 = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16)
        all_gather_into_tensor_315 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_944, 32, '0');  convert_element_type_944 = None
        wait_tensor_373 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_315);  all_gather_into_tensor_315 = None
        convert_element_type_945 = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_945, 2)
        mean_57 = torch.ops.aten.mean.dim(pow_58, [2], True);  pow_58 = None
        add_114 = torch.ops.aten.add.Scalar(mean_57, 1e-05);  mean_57 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_228 = torch.ops.aten.mul.Tensor(convert_element_type_945, rsqrt_57);  convert_element_type_945 = rsqrt_57 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, wait_tensor_373);  mul_228 = wait_tensor_373 = None
        convert_element_type_946 = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        all_gather_into_tensor_316 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_946, 8, '1');  convert_element_type_946 = None
        wait_tensor_374 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_316);  all_gather_into_tensor_316 = None
        split_123 = torch.ops.aten.split.Tensor(wait_tensor_374, 2);  wait_tensor_374 = None
        getitem_1245 = split_123[0]
        getitem_1246 = split_123[1]
        getitem_1247 = split_123[2]
        getitem_1248 = split_123[3]
        getitem_1249 = split_123[4]
        getitem_1250 = split_123[5]
        getitem_1251 = split_123[6]
        getitem_1252 = split_123[7];  split_123 = None
        cat_115 = torch.ops.aten.cat.default([getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252], 1);  getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = None
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16)
        all_gather_into_tensor_317 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_947, 32, '0');  convert_element_type_947 = None
        wait_tensor_375 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_317);  all_gather_into_tensor_317 = None
        permute_316 = torch.ops.aten.permute.default(wait_tensor_375, [1, 0]);  wait_tensor_375 = None
        view_2076 = torch.ops.aten.view.default(cat_115, [16384, 4096]);  cat_115 = None
        mm_200 = torch.ops.aten.mm.default(view_2076, permute_316);  permute_316 = None
        view_2077 = torch.ops.aten.view.default(mm_200, [2, 8192, 1792])
        convert_element_type_950 = torch.ops.prims.convert_element_type.default(view_2077, torch.float32);  view_2077 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(convert_element_type_950)
        mul_230 = torch.ops.aten.mul.Tensor(convert_element_type_950, sigmoid_28);  convert_element_type_950 = sigmoid_28 = None
        convert_element_type_951 = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16);  mul_230 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16)
        all_gather_into_tensor_318 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_952, 32, '0');  convert_element_type_952 = None
        wait_tensor_376 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_318);  all_gather_into_tensor_318 = None
        permute_317 = torch.ops.aten.permute.default(wait_tensor_376, [1, 0]);  wait_tensor_376 = None
        mm_201 = torch.ops.aten.mm.default(view_2076, permute_317);  view_2076 = permute_317 = None
        view_2084 = torch.ops.aten.view.default(mm_201, [2, 8192, 1792]);  mm_201 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_951, view_2084);  convert_element_type_951 = view_2084 = None
        convert_element_type_955 = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16)
        all_gather_into_tensor_319 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_955, 32, '0');  convert_element_type_955 = None
        wait_tensor_377 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_319);  all_gather_into_tensor_319 = None
        permute_318 = torch.ops.aten.permute.default(wait_tensor_377, [1, 0]);  wait_tensor_377 = None
        view_2091 = torch.ops.aten.view.default(mul_231, [16384, 1792]);  mul_231 = None
        mm_202 = torch.ops.aten.mm.default(view_2091, permute_318);  view_2091 = permute_318 = None
        view_2092 = torch.ops.aten.view.default(mm_202, [2, 8192, 4096]);  mm_202 = None
        split_124 = torch.ops.aten.split.Tensor(view_2092, 1024, 1);  view_2092 = None
        getitem_1253 = split_124[0]
        getitem_1254 = split_124[1]
        getitem_1255 = split_124[2]
        getitem_1256 = split_124[3]
        getitem_1257 = split_124[4]
        getitem_1258 = split_124[5]
        getitem_1259 = split_124[6]
        getitem_1260 = split_124[7];  split_124 = None
        cat_116 = torch.ops.aten.cat.default([getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260]);  getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = None
        reduce_scatter_tensor_58 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_116, 'sum', 8, '1');  cat_116 = None
        wait_tensor_378 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_58);  reduce_scatter_tensor_58 = None
        add_115 = torch.ops.aten.add.Tensor(add_113, wait_tensor_378);  add_113 = wait_tensor_378 = None
        convert_element_type_958 = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16)
        all_gather_into_tensor_320 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_958, 32, '0');  convert_element_type_958 = None
        wait_tensor_379 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_320);  all_gather_into_tensor_320 = None
        convert_element_type_959 = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        pow_59 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_959, 2)
        mean_58 = torch.ops.aten.mean.dim(pow_59, [2], True);  pow_59 = None
        add_116 = torch.ops.aten.add.Scalar(mean_58, 1e-05);  mean_58 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_959, rsqrt_58);  convert_element_type_959 = rsqrt_58 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, wait_tensor_379);  mul_232 = wait_tensor_379 = None
        convert_element_type_960 = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None
        all_gather_into_tensor_321 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_960, 8, '1');  convert_element_type_960 = None
        wait_tensor_380 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_321);  all_gather_into_tensor_321 = None
        split_125 = torch.ops.aten.split.Tensor(wait_tensor_380, 2);  wait_tensor_380 = None
        getitem_1261 = split_125[0]
        getitem_1262 = split_125[1]
        getitem_1263 = split_125[2]
        getitem_1264 = split_125[3]
        getitem_1265 = split_125[4]
        getitem_1266 = split_125[5]
        getitem_1267 = split_125[6]
        getitem_1268 = split_125[7];  split_125 = None
        cat_117 = torch.ops.aten.cat.default([getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268], 1);  getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = None
        convert_element_type_961 = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16)
        all_gather_into_tensor_322 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_961, 32, '0');  convert_element_type_961 = None
        wait_tensor_381 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_322);  all_gather_into_tensor_322 = None
        permute_319 = torch.ops.aten.permute.default(wait_tensor_381, [1, 0]);  wait_tensor_381 = None
        view_2103 = torch.ops.aten.view.default(cat_117, [16384, 4096]);  cat_117 = None
        mm_203 = torch.ops.aten.mm.default(view_2103, permute_319);  permute_319 = None
        view_2104 = torch.ops.aten.view.default(mm_203, [2, 8192, 512])
        convert_element_type_964 = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16)
        all_gather_into_tensor_323 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_964, 32, '0');  convert_element_type_964 = None
        wait_tensor_382 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_323);  all_gather_into_tensor_323 = None
        permute_320 = torch.ops.aten.permute.default(wait_tensor_382, [1, 0]);  wait_tensor_382 = None
        mm_204 = torch.ops.aten.mm.default(view_2103, permute_320);  permute_320 = None
        view_2111 = torch.ops.aten.view.default(mm_204, [2, 8192, 128]);  mm_204 = None
        convert_element_type_967 = torch.ops.prims.convert_element_type.default(primals_268, torch.bfloat16)
        all_gather_into_tensor_324 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_967, 32, '0');  convert_element_type_967 = None
        wait_tensor_383 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_324);  all_gather_into_tensor_324 = None
        permute_321 = torch.ops.aten.permute.default(wait_tensor_383, [1, 0]);  wait_tensor_383 = None
        mm_205 = torch.ops.aten.mm.default(view_2103, permute_321);  view_2103 = permute_321 = None
        view_2118 = torch.ops.aten.view.default(mm_205, [2, 8192, 128])
        view_2120 = torch.ops.aten.view.default(view_2104, [2, 8192, -1, 128]);  view_2104 = None
        view_2121 = torch.ops.aten.view.default(view_2111, [2, 8192, -1, 128]);  view_2111 = None
        view_2122 = torch.ops.aten.view.default(view_2118, [2, 8192, -1, 128]);  view_2118 = None
        convert_element_type_970 = torch.ops.prims.convert_element_type.default(view_2120, torch.float32);  view_2120 = None
        view_2123 = torch.ops.aten.view.default(convert_element_type_970, [2, 8192, 4, -1, 2]);  convert_element_type_970 = None
        view_as_complex_58 = torch.ops.aten.view_as_complex.default(view_2123);  view_2123 = None
        convert_element_type_971 = torch.ops.prims.convert_element_type.default(view_2121, torch.float32);  view_2121 = None
        view_2124 = torch.ops.aten.view.default(convert_element_type_971, [2, 8192, 1, -1, 2]);  convert_element_type_971 = None
        view_as_complex_59 = torch.ops.aten.view_as_complex.default(view_2124);  view_2124 = None
        mul_234 = torch.ops.aten.mul.Tensor(view_as_complex_58, view_37);  view_as_complex_58 = None
        view_as_real_58 = torch.ops.aten.view_as_real.default(mul_234);  mul_234 = None
        view_2126 = torch.ops.aten.view.default(view_as_real_58, [2, 8192, 4, 128]);  view_as_real_58 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_as_complex_59, view_37);  view_as_complex_59 = None
        view_as_real_59 = torch.ops.aten.view_as_real.default(mul_235);  mul_235 = None
        view_2127 = torch.ops.aten.view.default(view_as_real_59, [2, 8192, 1, 128]);  view_as_real_59 = None
        convert_element_type_972 = torch.ops.prims.convert_element_type.default(view_2126, torch.bfloat16);  view_2126 = None
        convert_element_type_973 = torch.ops.prims.convert_element_type.default(view_2127, torch.bfloat16);  view_2127 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(convert_element_type_973, 3);  convert_element_type_973 = None
        expand_58 = torch.ops.aten.expand.default(unsqueeze_58, [2, 8192, 1, 4, 128]);  unsqueeze_58 = None
        view_2128 = torch.ops.aten.view.default(expand_58, [2, 8192, 4, 128]);  expand_58 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(view_2122, 3);  view_2122 = None
        expand_59 = torch.ops.aten.expand.default(unsqueeze_59, [2, 8192, 1, 4, 128]);  unsqueeze_59 = None
        view_2129 = torch.ops.aten.view.default(expand_59, [2, 8192, 4, 128]);  expand_59 = None
        permute_322 = torch.ops.aten.permute.default(convert_element_type_972, [0, 2, 1, 3]);  convert_element_type_972 = None
        permute_323 = torch.ops.aten.permute.default(view_2128, [0, 2, 1, 3]);  view_2128 = None
        permute_324 = torch.ops.aten.permute.default(view_2129, [0, 2, 1, 3]);  view_2129 = None
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_322, permute_323, permute_324, None, True, 0.0, True);  permute_322 = permute_323 = permute_324 = None
        getitem_1269 = _scaled_dot_product_cudnn_attention_29[0]
        getitem_1270 = _scaled_dot_product_cudnn_attention_29[1]
        getitem_1275 = _scaled_dot_product_cudnn_attention_29[6]
        getitem_1276 = _scaled_dot_product_cudnn_attention_29[7];  _scaled_dot_product_cudnn_attention_29 = None
        permute_325 = torch.ops.aten.permute.default(getitem_1269, [0, 2, 1, 3])
        view_2130 = torch.ops.aten.view.default(permute_325, [2, 8192, -1]);  permute_325 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16)
        all_gather_into_tensor_325 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_974, 32, '0');  convert_element_type_974 = None
        wait_tensor_384 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_325);  all_gather_into_tensor_325 = None
        permute_326 = torch.ops.aten.permute.default(wait_tensor_384, [1, 0]);  wait_tensor_384 = None
        view_2136 = torch.ops.aten.view.default(view_2130, [16384, 512]);  view_2130 = None
        mm_206 = torch.ops.aten.mm.default(view_2136, permute_326);  view_2136 = permute_326 = None
        view_2137 = torch.ops.aten.view.default(mm_206, [2, 8192, 4096]);  mm_206 = None
        split_126 = torch.ops.aten.split.Tensor(view_2137, 1024, 1);  view_2137 = None
        getitem_1278 = split_126[0]
        getitem_1279 = split_126[1]
        getitem_1280 = split_126[2]
        getitem_1281 = split_126[3]
        getitem_1282 = split_126[4]
        getitem_1283 = split_126[5]
        getitem_1284 = split_126[6]
        getitem_1285 = split_126[7];  split_126 = None
        cat_118 = torch.ops.aten.cat.default([getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285]);  getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = None
        reduce_scatter_tensor_59 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_118, 'sum', 8, '1');  cat_118 = None
        wait_tensor_385 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_59)
        add_117 = torch.ops.aten.add.Tensor(add_115, wait_tensor_385);  wait_tensor_385 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16)
        all_gather_into_tensor_326 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_977, 32, '0');  convert_element_type_977 = None
        wait_tensor_386 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_326);  all_gather_into_tensor_326 = None
        convert_element_type_978 = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        pow_60 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_978, 2)
        mean_59 = torch.ops.aten.mean.dim(pow_60, [2], True);  pow_60 = None
        add_118 = torch.ops.aten.add.Scalar(mean_59, 1e-05);  mean_59 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_236 = torch.ops.aten.mul.Tensor(convert_element_type_978, rsqrt_59);  convert_element_type_978 = rsqrt_59 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, wait_tensor_386);  mul_236 = wait_tensor_386 = None
        convert_element_type_979 = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        all_gather_into_tensor_327 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_979, 8, '1');  convert_element_type_979 = None
        wait_tensor_387 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_327);  all_gather_into_tensor_327 = None
        split_127 = torch.ops.aten.split.Tensor(wait_tensor_387, 2);  wait_tensor_387 = None
        getitem_1286 = split_127[0]
        getitem_1287 = split_127[1]
        getitem_1288 = split_127[2]
        getitem_1289 = split_127[3]
        getitem_1290 = split_127[4]
        getitem_1291 = split_127[5]
        getitem_1292 = split_127[6]
        getitem_1293 = split_127[7];  split_127 = None
        cat_119 = torch.ops.aten.cat.default([getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293], 1);  getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = None
        convert_element_type_980 = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16)
        all_gather_into_tensor_328 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_980, 32, '0');  convert_element_type_980 = None
        wait_tensor_388 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_328);  all_gather_into_tensor_328 = None
        permute_327 = torch.ops.aten.permute.default(wait_tensor_388, [1, 0]);  wait_tensor_388 = None
        view_2148 = torch.ops.aten.view.default(cat_119, [16384, 4096]);  cat_119 = None
        mm_207 = torch.ops.aten.mm.default(view_2148, permute_327);  permute_327 = None
        view_2149 = torch.ops.aten.view.default(mm_207, [2, 8192, 1792])
        convert_element_type_983 = torch.ops.prims.convert_element_type.default(view_2149, torch.float32);  view_2149 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(convert_element_type_983)
        mul_238 = torch.ops.aten.mul.Tensor(convert_element_type_983, sigmoid_29);  convert_element_type_983 = sigmoid_29 = None
        convert_element_type_984 = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None
        convert_element_type_985 = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16)
        all_gather_into_tensor_329 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_985, 32, '0');  convert_element_type_985 = None
        wait_tensor_389 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_329);  all_gather_into_tensor_329 = None
        permute_328 = torch.ops.aten.permute.default(wait_tensor_389, [1, 0]);  wait_tensor_389 = None
        mm_208 = torch.ops.aten.mm.default(view_2148, permute_328);  view_2148 = permute_328 = None
        view_2156 = torch.ops.aten.view.default(mm_208, [2, 8192, 1792]);  mm_208 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_984, view_2156);  convert_element_type_984 = view_2156 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16)
        all_gather_into_tensor_330 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_988, 32, '0');  convert_element_type_988 = None
        wait_tensor_390 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_330);  all_gather_into_tensor_330 = None
        permute_329 = torch.ops.aten.permute.default(wait_tensor_390, [1, 0]);  wait_tensor_390 = None
        view_2163 = torch.ops.aten.view.default(mul_239, [16384, 1792]);  mul_239 = None
        mm_209 = torch.ops.aten.mm.default(view_2163, permute_329);  view_2163 = permute_329 = None
        view_2164 = torch.ops.aten.view.default(mm_209, [2, 8192, 4096]);  mm_209 = None
        split_128 = torch.ops.aten.split.Tensor(view_2164, 1024, 1);  view_2164 = None
        getitem_1294 = split_128[0]
        getitem_1295 = split_128[1]
        getitem_1296 = split_128[2]
        getitem_1297 = split_128[3]
        getitem_1298 = split_128[4]
        getitem_1299 = split_128[5]
        getitem_1300 = split_128[6]
        getitem_1301 = split_128[7];  split_128 = None
        cat_120 = torch.ops.aten.cat.default([getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301]);  getitem_1294 = getitem_1295 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = None
        reduce_scatter_tensor_60 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_120, 'sum', 8, '1');  cat_120 = None
        wait_tensor_391 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_60);  reduce_scatter_tensor_60 = None
        add_119 = torch.ops.aten.add.Tensor(add_117, wait_tensor_391);  add_117 = wait_tensor_391 = None
        convert_element_type_991 = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16)
        all_gather_into_tensor_331 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_991, 32, '0');  convert_element_type_991 = None
        wait_tensor_392 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_331);  all_gather_into_tensor_331 = None
        convert_element_type_992 = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        pow_61 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_992, 2)
        mean_60 = torch.ops.aten.mean.dim(pow_61, [2], True);  pow_61 = None
        add_120 = torch.ops.aten.add.Scalar(mean_60, 1e-05);  mean_60 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_992, rsqrt_60);  convert_element_type_992 = rsqrt_60 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, wait_tensor_392);  mul_240 = wait_tensor_392 = None
        convert_element_type_993 = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        all_gather_into_tensor_332 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_993, 8, '1');  convert_element_type_993 = None
        wait_tensor_393 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_332);  all_gather_into_tensor_332 = None
        split_129 = torch.ops.aten.split.Tensor(wait_tensor_393, 2);  wait_tensor_393 = None
        getitem_1302 = split_129[0]
        getitem_1303 = split_129[1]
        getitem_1304 = split_129[2]
        getitem_1305 = split_129[3]
        getitem_1306 = split_129[4]
        getitem_1307 = split_129[5]
        getitem_1308 = split_129[6]
        getitem_1309 = split_129[7];  split_129 = None
        cat_121 = torch.ops.aten.cat.default([getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309], 1);  getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = None
        convert_element_type_994 = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16)
        all_gather_into_tensor_333 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_994, 32, '0');  convert_element_type_994 = None
        wait_tensor_394 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_333);  all_gather_into_tensor_333 = None
        permute_330 = torch.ops.aten.permute.default(wait_tensor_394, [1, 0]);  wait_tensor_394 = None
        view_2175 = torch.ops.aten.view.default(cat_121, [16384, 4096]);  cat_121 = None
        mm_210 = torch.ops.aten.mm.default(view_2175, permute_330);  permute_330 = None
        view_2176 = torch.ops.aten.view.default(mm_210, [2, 8192, 512])
        convert_element_type_997 = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16)
        all_gather_into_tensor_334 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_997, 32, '0');  convert_element_type_997 = None
        wait_tensor_395 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_334);  all_gather_into_tensor_334 = None
        permute_331 = torch.ops.aten.permute.default(wait_tensor_395, [1, 0]);  wait_tensor_395 = None
        mm_211 = torch.ops.aten.mm.default(view_2175, permute_331);  permute_331 = None
        view_2183 = torch.ops.aten.view.default(mm_211, [2, 8192, 128]);  mm_211 = None
        convert_element_type_1000 = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16)
        all_gather_into_tensor_335 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1000, 32, '0');  convert_element_type_1000 = None
        wait_tensor_396 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_335);  all_gather_into_tensor_335 = None
        permute_332 = torch.ops.aten.permute.default(wait_tensor_396, [1, 0]);  wait_tensor_396 = None
        mm_212 = torch.ops.aten.mm.default(view_2175, permute_332);  view_2175 = permute_332 = None
        view_2190 = torch.ops.aten.view.default(mm_212, [2, 8192, 128])
        view_2192 = torch.ops.aten.view.default(view_2176, [2, 8192, -1, 128]);  view_2176 = None
        view_2193 = torch.ops.aten.view.default(view_2183, [2, 8192, -1, 128]);  view_2183 = None
        view_2194 = torch.ops.aten.view.default(view_2190, [2, 8192, -1, 128]);  view_2190 = None
        convert_element_type_1003 = torch.ops.prims.convert_element_type.default(view_2192, torch.float32);  view_2192 = None
        view_2195 = torch.ops.aten.view.default(convert_element_type_1003, [2, 8192, 4, -1, 2]);  convert_element_type_1003 = None
        view_as_complex_60 = torch.ops.aten.view_as_complex.default(view_2195);  view_2195 = None
        convert_element_type_1004 = torch.ops.prims.convert_element_type.default(view_2193, torch.float32);  view_2193 = None
        view_2196 = torch.ops.aten.view.default(convert_element_type_1004, [2, 8192, 1, -1, 2]);  convert_element_type_1004 = None
        view_as_complex_61 = torch.ops.aten.view_as_complex.default(view_2196);  view_2196 = None
        mul_242 = torch.ops.aten.mul.Tensor(view_as_complex_60, view_37);  view_as_complex_60 = None
        view_as_real_60 = torch.ops.aten.view_as_real.default(mul_242);  mul_242 = None
        view_2198 = torch.ops.aten.view.default(view_as_real_60, [2, 8192, 4, 128]);  view_as_real_60 = None
        mul_243 = torch.ops.aten.mul.Tensor(view_as_complex_61, view_37);  view_as_complex_61 = None
        view_as_real_61 = torch.ops.aten.view_as_real.default(mul_243);  mul_243 = None
        view_2199 = torch.ops.aten.view.default(view_as_real_61, [2, 8192, 1, 128]);  view_as_real_61 = None
        convert_element_type_1005 = torch.ops.prims.convert_element_type.default(view_2198, torch.bfloat16);  view_2198 = None
        convert_element_type_1006 = torch.ops.prims.convert_element_type.default(view_2199, torch.bfloat16);  view_2199 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(convert_element_type_1006, 3);  convert_element_type_1006 = None
        expand_60 = torch.ops.aten.expand.default(unsqueeze_60, [2, 8192, 1, 4, 128]);  unsqueeze_60 = None
        view_2200 = torch.ops.aten.view.default(expand_60, [2, 8192, 4, 128]);  expand_60 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(view_2194, 3);  view_2194 = None
        expand_61 = torch.ops.aten.expand.default(unsqueeze_61, [2, 8192, 1, 4, 128]);  unsqueeze_61 = None
        view_2201 = torch.ops.aten.view.default(expand_61, [2, 8192, 4, 128]);  expand_61 = None
        permute_333 = torch.ops.aten.permute.default(convert_element_type_1005, [0, 2, 1, 3]);  convert_element_type_1005 = None
        permute_334 = torch.ops.aten.permute.default(view_2200, [0, 2, 1, 3]);  view_2200 = None
        permute_335 = torch.ops.aten.permute.default(view_2201, [0, 2, 1, 3]);  view_2201 = None
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_333, permute_334, permute_335, None, True, 0.0, True);  permute_333 = permute_334 = permute_335 = None
        getitem_1310 = _scaled_dot_product_cudnn_attention_30[0]
        getitem_1311 = _scaled_dot_product_cudnn_attention_30[1]
        getitem_1316 = _scaled_dot_product_cudnn_attention_30[6]
        getitem_1317 = _scaled_dot_product_cudnn_attention_30[7];  _scaled_dot_product_cudnn_attention_30 = None
        permute_336 = torch.ops.aten.permute.default(getitem_1310, [0, 2, 1, 3])
        view_2202 = torch.ops.aten.view.default(permute_336, [2, 8192, -1]);  permute_336 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16)
        all_gather_into_tensor_336 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1007, 32, '0');  convert_element_type_1007 = None
        wait_tensor_397 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_336);  all_gather_into_tensor_336 = None
        permute_337 = torch.ops.aten.permute.default(wait_tensor_397, [1, 0]);  wait_tensor_397 = None
        view_2208 = torch.ops.aten.view.default(view_2202, [16384, 512]);  view_2202 = None
        mm_213 = torch.ops.aten.mm.default(view_2208, permute_337);  view_2208 = permute_337 = None
        view_2209 = torch.ops.aten.view.default(mm_213, [2, 8192, 4096]);  mm_213 = None
        split_130 = torch.ops.aten.split.Tensor(view_2209, 1024, 1);  view_2209 = None
        getitem_1319 = split_130[0]
        getitem_1320 = split_130[1]
        getitem_1321 = split_130[2]
        getitem_1322 = split_130[3]
        getitem_1323 = split_130[4]
        getitem_1324 = split_130[5]
        getitem_1325 = split_130[6]
        getitem_1326 = split_130[7];  split_130 = None
        cat_122 = torch.ops.aten.cat.default([getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326]);  getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = None
        reduce_scatter_tensor_61 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_122, 'sum', 8, '1');  cat_122 = None
        wait_tensor_398 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_61)
        add_121 = torch.ops.aten.add.Tensor(add_119, wait_tensor_398);  wait_tensor_398 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16)
        all_gather_into_tensor_337 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1010, 32, '0');  convert_element_type_1010 = None
        wait_tensor_399 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_337);  all_gather_into_tensor_337 = None
        convert_element_type_1011 = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        pow_62 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1011, 2)
        mean_61 = torch.ops.aten.mean.dim(pow_62, [2], True);  pow_62 = None
        add_122 = torch.ops.aten.add.Scalar(mean_61, 1e-05);  mean_61 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_244 = torch.ops.aten.mul.Tensor(convert_element_type_1011, rsqrt_61);  convert_element_type_1011 = rsqrt_61 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, wait_tensor_399);  mul_244 = wait_tensor_399 = None
        convert_element_type_1012 = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16);  mul_245 = None
        all_gather_into_tensor_338 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1012, 8, '1');  convert_element_type_1012 = None
        wait_tensor_400 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_338);  all_gather_into_tensor_338 = None
        split_131 = torch.ops.aten.split.Tensor(wait_tensor_400, 2);  wait_tensor_400 = None
        getitem_1327 = split_131[0]
        getitem_1328 = split_131[1]
        getitem_1329 = split_131[2]
        getitem_1330 = split_131[3]
        getitem_1331 = split_131[4]
        getitem_1332 = split_131[5]
        getitem_1333 = split_131[6]
        getitem_1334 = split_131[7];  split_131 = None
        cat_123 = torch.ops.aten.cat.default([getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334], 1);  getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = None
        convert_element_type_1013 = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16)
        all_gather_into_tensor_339 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1013, 32, '0');  convert_element_type_1013 = None
        wait_tensor_401 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_339);  all_gather_into_tensor_339 = None
        permute_338 = torch.ops.aten.permute.default(wait_tensor_401, [1, 0]);  wait_tensor_401 = None
        view_2220 = torch.ops.aten.view.default(cat_123, [16384, 4096]);  cat_123 = None
        mm_214 = torch.ops.aten.mm.default(view_2220, permute_338);  permute_338 = None
        view_2221 = torch.ops.aten.view.default(mm_214, [2, 8192, 1792])
        convert_element_type_1016 = torch.ops.prims.convert_element_type.default(view_2221, torch.float32);  view_2221 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(convert_element_type_1016)
        mul_246 = torch.ops.aten.mul.Tensor(convert_element_type_1016, sigmoid_30);  convert_element_type_1016 = sigmoid_30 = None
        convert_element_type_1017 = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        convert_element_type_1018 = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16)
        all_gather_into_tensor_340 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1018, 32, '0');  convert_element_type_1018 = None
        wait_tensor_402 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_340);  all_gather_into_tensor_340 = None
        permute_339 = torch.ops.aten.permute.default(wait_tensor_402, [1, 0]);  wait_tensor_402 = None
        mm_215 = torch.ops.aten.mm.default(view_2220, permute_339);  view_2220 = permute_339 = None
        view_2228 = torch.ops.aten.view.default(mm_215, [2, 8192, 1792]);  mm_215 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_1017, view_2228);  convert_element_type_1017 = view_2228 = None
        convert_element_type_1021 = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16)
        all_gather_into_tensor_341 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1021, 32, '0');  convert_element_type_1021 = None
        wait_tensor_403 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_341);  all_gather_into_tensor_341 = None
        permute_340 = torch.ops.aten.permute.default(wait_tensor_403, [1, 0]);  wait_tensor_403 = None
        view_2235 = torch.ops.aten.view.default(mul_247, [16384, 1792]);  mul_247 = None
        mm_216 = torch.ops.aten.mm.default(view_2235, permute_340);  view_2235 = permute_340 = None
        view_2236 = torch.ops.aten.view.default(mm_216, [2, 8192, 4096]);  mm_216 = None
        split_132 = torch.ops.aten.split.Tensor(view_2236, 1024, 1);  view_2236 = None
        getitem_1335 = split_132[0]
        getitem_1336 = split_132[1]
        getitem_1337 = split_132[2]
        getitem_1338 = split_132[3]
        getitem_1339 = split_132[4]
        getitem_1340 = split_132[5]
        getitem_1341 = split_132[6]
        getitem_1342 = split_132[7];  split_132 = None
        cat_124 = torch.ops.aten.cat.default([getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342]);  getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = getitem_1341 = getitem_1342 = None
        reduce_scatter_tensor_62 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_124, 'sum', 8, '1');  cat_124 = None
        wait_tensor_404 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_62);  reduce_scatter_tensor_62 = None
        add_123 = torch.ops.aten.add.Tensor(add_121, wait_tensor_404);  add_121 = wait_tensor_404 = None
        convert_element_type_1024 = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16)
        all_gather_into_tensor_342 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1024, 32, '0');  convert_element_type_1024 = None
        wait_tensor_405 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_342);  all_gather_into_tensor_342 = None
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        pow_63 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1025, 2)
        mean_62 = torch.ops.aten.mean.dim(pow_63, [2], True);  pow_63 = None
        add_124 = torch.ops.aten.add.Scalar(mean_62, 1e-05);  mean_62 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_1025, rsqrt_62);  convert_element_type_1025 = rsqrt_62 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, wait_tensor_405);  mul_248 = wait_tensor_405 = None
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        all_gather_into_tensor_343 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1026, 8, '1');  convert_element_type_1026 = None
        wait_tensor_406 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_343);  all_gather_into_tensor_343 = None
        split_133 = torch.ops.aten.split.Tensor(wait_tensor_406, 2);  wait_tensor_406 = None
        getitem_1343 = split_133[0]
        getitem_1344 = split_133[1]
        getitem_1345 = split_133[2]
        getitem_1346 = split_133[3]
        getitem_1347 = split_133[4]
        getitem_1348 = split_133[5]
        getitem_1349 = split_133[6]
        getitem_1350 = split_133[7];  split_133 = None
        cat_125 = torch.ops.aten.cat.default([getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350], 1);  getitem_1343 = getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = None
        convert_element_type_1027 = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16)
        all_gather_into_tensor_344 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1027, 32, '0');  convert_element_type_1027 = None
        wait_tensor_407 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_344);  all_gather_into_tensor_344 = None
        permute_341 = torch.ops.aten.permute.default(wait_tensor_407, [1, 0]);  wait_tensor_407 = None
        view_2247 = torch.ops.aten.view.default(cat_125, [16384, 4096]);  cat_125 = None
        mm_217 = torch.ops.aten.mm.default(view_2247, permute_341);  permute_341 = None
        view_2248 = torch.ops.aten.view.default(mm_217, [2, 8192, 512])
        convert_element_type_1030 = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16)
        all_gather_into_tensor_345 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1030, 32, '0');  convert_element_type_1030 = None
        wait_tensor_408 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_345);  all_gather_into_tensor_345 = None
        permute_342 = torch.ops.aten.permute.default(wait_tensor_408, [1, 0]);  wait_tensor_408 = None
        mm_218 = torch.ops.aten.mm.default(view_2247, permute_342);  permute_342 = None
        view_2255 = torch.ops.aten.view.default(mm_218, [2, 8192, 128]);  mm_218 = None
        convert_element_type_1033 = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16)
        all_gather_into_tensor_346 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1033, 32, '0');  convert_element_type_1033 = None
        wait_tensor_409 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_346);  all_gather_into_tensor_346 = None
        permute_343 = torch.ops.aten.permute.default(wait_tensor_409, [1, 0]);  wait_tensor_409 = None
        mm_219 = torch.ops.aten.mm.default(view_2247, permute_343);  view_2247 = permute_343 = None
        view_2262 = torch.ops.aten.view.default(mm_219, [2, 8192, 128])
        view_2264 = torch.ops.aten.view.default(view_2248, [2, 8192, -1, 128]);  view_2248 = None
        view_2265 = torch.ops.aten.view.default(view_2255, [2, 8192, -1, 128]);  view_2255 = None
        view_2266 = torch.ops.aten.view.default(view_2262, [2, 8192, -1, 128]);  view_2262 = None
        convert_element_type_1036 = torch.ops.prims.convert_element_type.default(view_2264, torch.float32);  view_2264 = None
        view_2267 = torch.ops.aten.view.default(convert_element_type_1036, [2, 8192, 4, -1, 2]);  convert_element_type_1036 = None
        view_as_complex_62 = torch.ops.aten.view_as_complex.default(view_2267);  view_2267 = None
        convert_element_type_1037 = torch.ops.prims.convert_element_type.default(view_2265, torch.float32);  view_2265 = None
        view_2268 = torch.ops.aten.view.default(convert_element_type_1037, [2, 8192, 1, -1, 2]);  convert_element_type_1037 = None
        view_as_complex_63 = torch.ops.aten.view_as_complex.default(view_2268);  view_2268 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_as_complex_62, view_37);  view_as_complex_62 = None
        view_as_real_62 = torch.ops.aten.view_as_real.default(mul_250);  mul_250 = None
        view_2270 = torch.ops.aten.view.default(view_as_real_62, [2, 8192, 4, 128]);  view_as_real_62 = None
        mul_251 = torch.ops.aten.mul.Tensor(view_as_complex_63, view_37);  view_as_complex_63 = view_37 = None
        view_as_real_63 = torch.ops.aten.view_as_real.default(mul_251);  mul_251 = None
        view_2271 = torch.ops.aten.view.default(view_as_real_63, [2, 8192, 1, 128]);  view_as_real_63 = None
        convert_element_type_1038 = torch.ops.prims.convert_element_type.default(view_2270, torch.bfloat16);  view_2270 = None
        convert_element_type_1039 = torch.ops.prims.convert_element_type.default(view_2271, torch.bfloat16);  view_2271 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(convert_element_type_1039, 3);  convert_element_type_1039 = None
        expand_62 = torch.ops.aten.expand.default(unsqueeze_62, [2, 8192, 1, 4, 128]);  unsqueeze_62 = None
        view_2272 = torch.ops.aten.view.default(expand_62, [2, 8192, 4, 128]);  expand_62 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(view_2266, 3);  view_2266 = None
        expand_63 = torch.ops.aten.expand.default(unsqueeze_63, [2, 8192, 1, 4, 128]);  unsqueeze_63 = None
        view_2273 = torch.ops.aten.view.default(expand_63, [2, 8192, 4, 128]);  expand_63 = None
        permute_344 = torch.ops.aten.permute.default(convert_element_type_1038, [0, 2, 1, 3]);  convert_element_type_1038 = None
        permute_345 = torch.ops.aten.permute.default(view_2272, [0, 2, 1, 3]);  view_2272 = None
        permute_346 = torch.ops.aten.permute.default(view_2273, [0, 2, 1, 3]);  view_2273 = None
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_344, permute_345, permute_346, None, True, 0.0, True);  permute_344 = permute_345 = permute_346 = None
        getitem_1351 = _scaled_dot_product_cudnn_attention_31[0]
        getitem_1352 = _scaled_dot_product_cudnn_attention_31[1]
        getitem_1357 = _scaled_dot_product_cudnn_attention_31[6]
        getitem_1358 = _scaled_dot_product_cudnn_attention_31[7];  _scaled_dot_product_cudnn_attention_31 = None
        permute_347 = torch.ops.aten.permute.default(getitem_1351, [0, 2, 1, 3])
        view_2274 = torch.ops.aten.view.default(permute_347, [2, 8192, -1]);  permute_347 = None
        convert_element_type_1040 = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16)
        all_gather_into_tensor_347 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1040, 32, '0');  convert_element_type_1040 = None
        wait_tensor_410 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_347);  all_gather_into_tensor_347 = None
        permute_348 = torch.ops.aten.permute.default(wait_tensor_410, [1, 0]);  wait_tensor_410 = None
        view_2280 = torch.ops.aten.view.default(view_2274, [16384, 512]);  view_2274 = None
        mm_220 = torch.ops.aten.mm.default(view_2280, permute_348);  view_2280 = permute_348 = None
        view_2281 = torch.ops.aten.view.default(mm_220, [2, 8192, 4096]);  mm_220 = None
        split_134 = torch.ops.aten.split.Tensor(view_2281, 1024, 1);  view_2281 = None
        getitem_1360 = split_134[0]
        getitem_1361 = split_134[1]
        getitem_1362 = split_134[2]
        getitem_1363 = split_134[3]
        getitem_1364 = split_134[4]
        getitem_1365 = split_134[5]
        getitem_1366 = split_134[6]
        getitem_1367 = split_134[7];  split_134 = None
        cat_126 = torch.ops.aten.cat.default([getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367]);  getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = None
        reduce_scatter_tensor_63 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_126, 'sum', 8, '1');  cat_126 = None
        wait_tensor_411 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_63)
        add_125 = torch.ops.aten.add.Tensor(add_123, wait_tensor_411);  wait_tensor_411 = None
        convert_element_type_1043 = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16)
        all_gather_into_tensor_348 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1043, 32, '0');  convert_element_type_1043 = None
        wait_tensor_412 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_348);  all_gather_into_tensor_348 = None
        convert_element_type_1044 = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_64 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1044, 2)
        mean_63 = torch.ops.aten.mean.dim(pow_64, [2], True);  pow_64 = None
        add_126 = torch.ops.aten.add.Scalar(mean_63, 1e-05);  mean_63 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_252 = torch.ops.aten.mul.Tensor(convert_element_type_1044, rsqrt_63);  convert_element_type_1044 = rsqrt_63 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, wait_tensor_412);  mul_252 = wait_tensor_412 = None
        convert_element_type_1045 = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None
        all_gather_into_tensor_349 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1045, 8, '1');  convert_element_type_1045 = None
        wait_tensor_413 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_349);  all_gather_into_tensor_349 = None
        split_135 = torch.ops.aten.split.Tensor(wait_tensor_413, 2);  wait_tensor_413 = None
        getitem_1368 = split_135[0]
        getitem_1369 = split_135[1]
        getitem_1370 = split_135[2]
        getitem_1371 = split_135[3]
        getitem_1372 = split_135[4]
        getitem_1373 = split_135[5]
        getitem_1374 = split_135[6]
        getitem_1375 = split_135[7];  split_135 = None
        cat_127 = torch.ops.aten.cat.default([getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375], 1);  getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = None
        convert_element_type_1046 = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16)
        all_gather_into_tensor_350 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1046, 32, '0');  convert_element_type_1046 = None
        wait_tensor_414 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_350);  all_gather_into_tensor_350 = None
        permute_349 = torch.ops.aten.permute.default(wait_tensor_414, [1, 0]);  wait_tensor_414 = None
        view_2292 = torch.ops.aten.view.default(cat_127, [16384, 4096]);  cat_127 = None
        mm_221 = torch.ops.aten.mm.default(view_2292, permute_349);  permute_349 = None
        view_2293 = torch.ops.aten.view.default(mm_221, [2, 8192, 1792])
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(view_2293, torch.float32);  view_2293 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(convert_element_type_1049)
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_1049, sigmoid_31);  convert_element_type_1049 = sigmoid_31 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        convert_element_type_1051 = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16)
        all_gather_into_tensor_351 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1051, 32, '0');  convert_element_type_1051 = None
        wait_tensor_415 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_351);  all_gather_into_tensor_351 = None
        permute_350 = torch.ops.aten.permute.default(wait_tensor_415, [1, 0]);  wait_tensor_415 = None
        mm_222 = torch.ops.aten.mm.default(view_2292, permute_350);  view_2292 = permute_350 = None
        view_2300 = torch.ops.aten.view.default(mm_222, [2, 8192, 1792]);  mm_222 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_1050, view_2300);  convert_element_type_1050 = view_2300 = None
        convert_element_type_1054 = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16)
        all_gather_into_tensor_352 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1054, 32, '0');  convert_element_type_1054 = None
        wait_tensor_416 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_352);  all_gather_into_tensor_352 = None
        permute_351 = torch.ops.aten.permute.default(wait_tensor_416, [1, 0]);  wait_tensor_416 = None
        view_2307 = torch.ops.aten.view.default(mul_255, [16384, 1792]);  mul_255 = None
        mm_223 = torch.ops.aten.mm.default(view_2307, permute_351);  view_2307 = permute_351 = None
        view_2308 = torch.ops.aten.view.default(mm_223, [2, 8192, 4096]);  mm_223 = None
        split_136 = torch.ops.aten.split.Tensor(view_2308, 1024, 1);  view_2308 = None
        getitem_1376 = split_136[0]
        getitem_1377 = split_136[1]
        getitem_1378 = split_136[2]
        getitem_1379 = split_136[3]
        getitem_1380 = split_136[4]
        getitem_1381 = split_136[5]
        getitem_1382 = split_136[6]
        getitem_1383 = split_136[7];  split_136 = None
        cat_128 = torch.ops.aten.cat.default([getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383]);  getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = None
        reduce_scatter_tensor_64 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_128, 'sum', 8, '1');  cat_128 = None
        wait_tensor_417 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_64)
        add_127 = torch.ops.aten.add.Tensor(add_125, wait_tensor_417);  add_125 = wait_tensor_417 = None
        convert_element_type_1057 = torch.ops.prims.convert_element_type.default(primals_292, torch.bfloat16)
        all_gather_into_tensor_353 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1057, 32, '0');  convert_element_type_1057 = None
        wait_tensor_418 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_353);  all_gather_into_tensor_353 = None
        convert_element_type_1058 = torch.ops.prims.convert_element_type.default(add_127, torch.float32);  add_127 = None
        pow_65 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1058, 2)
        mean_64 = torch.ops.aten.mean.dim(pow_65, [2], True);  pow_65 = None
        add_128 = torch.ops.aten.add.Scalar(mean_64, 1e-05);  mean_64 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_256 = torch.ops.aten.mul.Tensor(convert_element_type_1058, rsqrt_64);  convert_element_type_1058 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, wait_tensor_418);  mul_256 = wait_tensor_418 = None
        convert_element_type_1059 = torch.ops.prims.convert_element_type.default(mul_257, torch.bfloat16);  mul_257 = None
        all_gather_into_tensor_354 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1059, 8, '1');  convert_element_type_1059 = None
        wait_tensor_419 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_354);  all_gather_into_tensor_354 = None
        split_137 = torch.ops.aten.split.Tensor(wait_tensor_419, 2);  wait_tensor_419 = None
        getitem_1384 = split_137[0]
        getitem_1385 = split_137[1]
        getitem_1386 = split_137[2]
        getitem_1387 = split_137[3]
        getitem_1388 = split_137[4]
        getitem_1389 = split_137[5]
        getitem_1390 = split_137[6]
        getitem_1391 = split_137[7];  split_137 = None
        cat_129 = torch.ops.aten.cat.default([getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391], 1);  getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = None
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16)
        all_gather_into_tensor_355 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1060, 32, '0');  convert_element_type_1060 = None
        wait_tensor_420 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_355);  all_gather_into_tensor_355 = None
        permute_352 = torch.ops.aten.permute.default(wait_tensor_420, [1, 0]);  wait_tensor_420 = None
        view_2319 = torch.ops.aten.view.default(cat_129, [16384, 4096]);  cat_129 = None
        mm_224 = torch.ops.aten.mm.default(view_2319, permute_352);  permute_352 = None
        view_2320 = torch.ops.aten.view.default(mm_224, [2, 8192, 16032]);  mm_224 = None
        return (view_2320, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, wait_tensor_1, mm, mm_2, getitem_80, getitem_81, getitem_86, getitem_87, reduce_scatter_tensor_1, mm_4, add_3, mm_7, mm_9, getitem_121, getitem_122, getitem_127, getitem_128, reduce_scatter_tensor_3, mm_11, add_7, mm_14, mm_16, getitem_162, getitem_163, getitem_168, getitem_169, reduce_scatter_tensor_5, mm_18, add_11, mm_21, mm_23, getitem_203, getitem_204, getitem_209, getitem_210, reduce_scatter_tensor_7, mm_25, add_15, mm_28, mm_30, getitem_244, getitem_245, getitem_250, getitem_251, reduce_scatter_tensor_9, mm_32, add_19, mm_35, mm_37, getitem_285, getitem_286, getitem_291, getitem_292, reduce_scatter_tensor_11, mm_39, add_23, mm_42, mm_44, getitem_326, getitem_327, getitem_332, getitem_333, reduce_scatter_tensor_13, mm_46, add_27, mm_49, mm_51, getitem_367, getitem_368, getitem_373, getitem_374, reduce_scatter_tensor_15, mm_53, add_31, mm_56, mm_58, getitem_408, getitem_409, getitem_414, getitem_415, reduce_scatter_tensor_17, mm_60, add_35, mm_63, mm_65, getitem_449, getitem_450, getitem_455, getitem_456, reduce_scatter_tensor_19, mm_67, add_39, mm_70, mm_72, getitem_490, getitem_491, getitem_496, getitem_497, reduce_scatter_tensor_21, mm_74, add_43, mm_77, mm_79, getitem_531, getitem_532, getitem_537, getitem_538, reduce_scatter_tensor_23, mm_81, add_47, mm_84, mm_86, getitem_572, getitem_573, getitem_578, getitem_579, reduce_scatter_tensor_25, mm_88, add_51, mm_91, mm_93, getitem_613, getitem_614, getitem_619, getitem_620, reduce_scatter_tensor_27, mm_95, add_55, mm_98, mm_100, getitem_654, getitem_655, getitem_660, getitem_661, reduce_scatter_tensor_29, mm_102, add_59, mm_105, mm_107, getitem_695, getitem_696, getitem_701, getitem_702, reduce_scatter_tensor_31, mm_109, add_63, mm_112, mm_114, getitem_736, getitem_737, getitem_742, getitem_743, reduce_scatter_tensor_33, mm_116, add_67, mm_119, mm_121, getitem_777, getitem_778, getitem_783, getitem_784, reduce_scatter_tensor_35, mm_123, add_71, mm_126, mm_128, getitem_818, getitem_819, getitem_824, getitem_825, reduce_scatter_tensor_37, mm_130, add_75, mm_133, mm_135, getitem_859, getitem_860, getitem_865, getitem_866, reduce_scatter_tensor_39, mm_137, add_79, mm_140, mm_142, getitem_900, getitem_901, getitem_906, getitem_907, reduce_scatter_tensor_41, mm_144, add_83, mm_147, mm_149, getitem_941, getitem_942, getitem_947, getitem_948, reduce_scatter_tensor_43, mm_151, add_87, mm_154, mm_156, getitem_982, getitem_983, getitem_988, getitem_989, reduce_scatter_tensor_45, mm_158, add_91, mm_161, mm_163, getitem_1023, getitem_1024, getitem_1029, getitem_1030, reduce_scatter_tensor_47, mm_165, add_95, mm_168, mm_170, getitem_1064, getitem_1065, getitem_1070, getitem_1071, reduce_scatter_tensor_49, mm_172, add_99, mm_175, mm_177, getitem_1105, getitem_1106, getitem_1111, getitem_1112, reduce_scatter_tensor_51, mm_179, add_103, mm_182, mm_184, getitem_1146, getitem_1147, getitem_1152, getitem_1153, reduce_scatter_tensor_53, mm_186, add_107, mm_189, mm_191, getitem_1187, getitem_1188, getitem_1193, getitem_1194, reduce_scatter_tensor_55, mm_193, add_111, mm_196, mm_198, getitem_1228, getitem_1229, getitem_1234, getitem_1235, reduce_scatter_tensor_57, mm_200, add_115, mm_203, mm_205, getitem_1269, getitem_1270, getitem_1275, getitem_1276, reduce_scatter_tensor_59, mm_207, add_119, mm_210, mm_212, getitem_1310, getitem_1311, getitem_1316, getitem_1317, reduce_scatter_tensor_61, mm_214, add_123, mm_217, mm_219, getitem_1351, getitem_1352, getitem_1357, getitem_1358, reduce_scatter_tensor_63, mm_221, reduce_scatter_tensor_64, rsqrt_64, view_2319)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 8208384, device=device(type='cuda', index=0))
    reader.tensor(buf1, (501, 4096), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf2, (8192, 64), dtype=torch.complex64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16, 4096), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4, 4096), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (4, 4096), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128, 512), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf9, (56, 4096), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf10, (56, 4096), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 1792), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16, 4096), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4, 4096), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (4, 4096), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128, 512), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf18, (56, 4096), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf19, (56, 4096), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 1792), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf22, (16, 4096), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf23, (4, 4096), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf24, (4, 4096), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128, 512), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf27, (56, 4096), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf28, (56, 4096), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128, 1792), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf31, (16, 4096), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4, 4096), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4, 4096), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf34, (128, 512), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf36, (56, 4096), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf37, (56, 4096), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128, 1792), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf40, (16, 4096), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (4, 4096), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf42, (4, 4096), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf43, (128, 512), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf45, (56, 4096), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf46, (56, 4096), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf47, (128, 1792), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf48, (128,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf49, (16, 4096), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf50, (4, 4096), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (4, 4096), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128, 512), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf54, (56, 4096), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf55, (56, 4096), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf56, (128, 1792), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf58, (16, 4096), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf59, (4, 4096), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf60, (4, 4096), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 512), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf63, (56, 4096), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf64, (56, 4096), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128, 1792), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (16, 4096), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf68, (4, 4096), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf69, (4, 4096), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128, 512), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf72, (56, 4096), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf73, (56, 4096), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128, 1792), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf76, (16, 4096), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (4, 4096), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (4, 4096), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128, 512), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf81, (56, 4096), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf82, (56, 4096), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 1792), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (16, 4096), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf86, (4, 4096), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (4, 4096), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128, 512), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf90, (56, 4096), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf91, (56, 4096), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128, 1792), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf94, (16, 4096), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (4, 4096), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf96, (4, 4096), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128, 512), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf99, (56, 4096), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf100, (56, 4096), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128, 1792), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf103, (16, 4096), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf104, (4, 4096), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (4, 4096), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128, 512), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf108, (56, 4096), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf109, (56, 4096), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128, 1792), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf111, (128,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf112, (16, 4096), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (4, 4096), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (4, 4096), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128, 512), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf117, (56, 4096), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf118, (56, 4096), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128, 1792), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (16, 4096), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (4, 4096), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (4, 4096), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128, 512), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf125, (128,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf126, (56, 4096), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf127, (56, 4096), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf128, (128, 1792), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf129, (128,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (16, 4096), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (4, 4096), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf132, (4, 4096), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (128, 512), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf134, (128,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf135, (56, 4096), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf136, (56, 4096), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128, 1792), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf138, (128,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (16, 4096), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (4, 4096), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (4, 4096), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128, 512), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf143, (128,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf144, (56, 4096), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf145, (56, 4096), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf146, (128, 1792), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf147, (128,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf148, (16, 4096), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf149, (4, 4096), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (4, 4096), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf151, (128, 512), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf152, (128,), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf153, (56, 4096), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf154, (56, 4096), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf155, (128, 1792), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf156, (128,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (16, 4096), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf158, (4, 4096), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (4, 4096), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf160, (128, 512), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf161, (128,), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf162, (56, 4096), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf163, (56, 4096), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf164, (128, 1792), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf165, (128,), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf166, (16, 4096), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (4, 4096), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf168, (4, 4096), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf169, (128, 512), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf170, (128,), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf171, (56, 4096), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf172, (56, 4096), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf173, (128, 1792), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf174, (128,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf175, (16, 4096), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4, 4096), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (4, 4096), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf178, (128, 512), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf179, (128,), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf180, (56, 4096), is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf181, (56, 4096), is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf182, (128, 1792), is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf183, (128,), is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf184, (16, 4096), is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf185, (4, 4096), is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf186, (4, 4096), is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf187, (128, 512), is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf188, (128,), is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf189, (56, 4096), is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf190, (56, 4096), is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf191, (128, 1792), is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf192, (128,), is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (16, 4096), is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4, 4096), is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (4, 4096), is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf196, (128, 512), is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf197, (128,), is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf198, (56, 4096), is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf199, (56, 4096), is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128, 1792), is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf201, (128,), is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf202, (16, 4096), is_leaf=True)  # primals_203
    buf203 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf203, (4, 4096), is_leaf=True)  # primals_204
    buf204 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf204, (4, 4096), is_leaf=True)  # primals_205
    buf205 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (128, 512), is_leaf=True)  # primals_206
    buf206 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf206, (128,), is_leaf=True)  # primals_207
    buf207 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf207, (56, 4096), is_leaf=True)  # primals_208
    buf208 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf208, (56, 4096), is_leaf=True)  # primals_209
    buf209 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf209, (128, 1792), is_leaf=True)  # primals_210
    buf210 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf210, (128,), is_leaf=True)  # primals_211
    buf211 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf211, (16, 4096), is_leaf=True)  # primals_212
    buf212 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf212, (4, 4096), is_leaf=True)  # primals_213
    buf213 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf213, (4, 4096), is_leaf=True)  # primals_214
    buf214 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf214, (128, 512), is_leaf=True)  # primals_215
    buf215 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf215, (128,), is_leaf=True)  # primals_216
    buf216 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf216, (56, 4096), is_leaf=True)  # primals_217
    buf217 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf217, (56, 4096), is_leaf=True)  # primals_218
    buf218 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf218, (128, 1792), is_leaf=True)  # primals_219
    buf219 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf219, (128,), is_leaf=True)  # primals_220
    buf220 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf220, (16, 4096), is_leaf=True)  # primals_221
    buf221 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf221, (4, 4096), is_leaf=True)  # primals_222
    buf222 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf222, (4, 4096), is_leaf=True)  # primals_223
    buf223 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (128, 512), is_leaf=True)  # primals_224
    buf224 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf224, (128,), is_leaf=True)  # primals_225
    buf225 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf225, (56, 4096), is_leaf=True)  # primals_226
    buf226 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf226, (56, 4096), is_leaf=True)  # primals_227
    buf227 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf227, (128, 1792), is_leaf=True)  # primals_228
    buf228 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf228, (128,), is_leaf=True)  # primals_229
    buf229 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf229, (16, 4096), is_leaf=True)  # primals_230
    buf230 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf230, (4, 4096), is_leaf=True)  # primals_231
    buf231 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf231, (4, 4096), is_leaf=True)  # primals_232
    buf232 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf232, (128, 512), is_leaf=True)  # primals_233
    buf233 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf233, (128,), is_leaf=True)  # primals_234
    buf234 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf234, (56, 4096), is_leaf=True)  # primals_235
    buf235 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf235, (56, 4096), is_leaf=True)  # primals_236
    buf236 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf236, (128, 1792), is_leaf=True)  # primals_237
    buf237 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf237, (128,), is_leaf=True)  # primals_238
    buf238 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf238, (16, 4096), is_leaf=True)  # primals_239
    buf239 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf239, (4, 4096), is_leaf=True)  # primals_240
    buf240 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (4, 4096), is_leaf=True)  # primals_241
    buf241 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf241, (128, 512), is_leaf=True)  # primals_242
    buf242 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf242, (128,), is_leaf=True)  # primals_243
    buf243 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf243, (56, 4096), is_leaf=True)  # primals_244
    buf244 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf244, (56, 4096), is_leaf=True)  # primals_245
    buf245 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf245, (128, 1792), is_leaf=True)  # primals_246
    buf246 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf246, (128,), is_leaf=True)  # primals_247
    buf247 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf247, (16, 4096), is_leaf=True)  # primals_248
    buf248 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf248, (4, 4096), is_leaf=True)  # primals_249
    buf249 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (4, 4096), is_leaf=True)  # primals_250
    buf250 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf250, (128, 512), is_leaf=True)  # primals_251
    buf251 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf251, (128,), is_leaf=True)  # primals_252
    buf252 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf252, (56, 4096), is_leaf=True)  # primals_253
    buf253 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf253, (56, 4096), is_leaf=True)  # primals_254
    buf254 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf254, (128, 1792), is_leaf=True)  # primals_255
    buf255 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf255, (128,), is_leaf=True)  # primals_256
    buf256 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf256, (16, 4096), is_leaf=True)  # primals_257
    buf257 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf257, (4, 4096), is_leaf=True)  # primals_258
    buf258 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf258, (4, 4096), is_leaf=True)  # primals_259
    buf259 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf259, (128, 512), is_leaf=True)  # primals_260
    buf260 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf260, (128,), is_leaf=True)  # primals_261
    buf261 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf261, (56, 4096), is_leaf=True)  # primals_262
    buf262 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf262, (56, 4096), is_leaf=True)  # primals_263
    buf263 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf263, (128, 1792), is_leaf=True)  # primals_264
    buf264 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf264, (128,), is_leaf=True)  # primals_265
    buf265 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf265, (16, 4096), is_leaf=True)  # primals_266
    buf266 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf266, (4, 4096), is_leaf=True)  # primals_267
    buf267 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf267, (4, 4096), is_leaf=True)  # primals_268
    buf268 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf268, (128, 512), is_leaf=True)  # primals_269
    buf269 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf269, (128,), is_leaf=True)  # primals_270
    buf270 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf270, (56, 4096), is_leaf=True)  # primals_271
    buf271 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf271, (56, 4096), is_leaf=True)  # primals_272
    buf272 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf272, (128, 1792), is_leaf=True)  # primals_273
    buf273 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf273, (128,), is_leaf=True)  # primals_274
    buf274 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf274, (16, 4096), is_leaf=True)  # primals_275
    buf275 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf275, (4, 4096), is_leaf=True)  # primals_276
    buf276 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf276, (4, 4096), is_leaf=True)  # primals_277
    buf277 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf277, (128, 512), is_leaf=True)  # primals_278
    buf278 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf278, (128,), is_leaf=True)  # primals_279
    buf279 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf279, (56, 4096), is_leaf=True)  # primals_280
    buf280 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf280, (56, 4096), is_leaf=True)  # primals_281
    buf281 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf281, (128, 1792), is_leaf=True)  # primals_282
    buf282 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf282, (128,), is_leaf=True)  # primals_283
    buf283 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf283, (16, 4096), is_leaf=True)  # primals_284
    buf284 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf284, (4, 4096), is_leaf=True)  # primals_285
    buf285 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf285, (4, 4096), is_leaf=True)  # primals_286
    buf286 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf286, (128, 512), is_leaf=True)  # primals_287
    buf287 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf287, (128,), is_leaf=True)  # primals_288
    buf288 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf288, (56, 4096), is_leaf=True)  # primals_289
    buf289 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf289, (56, 4096), is_leaf=True)  # primals_290
    buf290 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf290, (128, 1792), is_leaf=True)  # primals_291
    buf291 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf291, (128,), is_leaf=True)  # primals_292
    buf292 = reader.storage(None, 8208384, device=device(type='cuda', index=0))
    reader.tensor(buf292, (501, 4096), is_leaf=True)  # primals_293

load_args._version = 0

def get_mesh_sizes():
    return 32, 8

def get_colls_estimations_file():
    return "colls32_8.table"

def get_pg_names():
    return "0", "1"

