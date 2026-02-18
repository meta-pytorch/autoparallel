import torch
from torch.nn import *
from torch import tensor, device


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, embedding, mm, mm_2, getitem, getitem_1, getitem_6, getitem_7, mm_4, add_3, mm_7, mm_9, getitem_9, getitem_10, getitem_15, getitem_16, mm_11, add_7, mm_14, mm_16, getitem_18, getitem_19, getitem_24, getitem_25, mm_18, add_11, mm_21, mm_23, getitem_27, getitem_28, getitem_33, getitem_34, mm_25, add_15, mm_28, mm_30, getitem_36, getitem_37, getitem_42, getitem_43, mm_32, add_19, mm_35, mm_37, getitem_45, getitem_46, getitem_51, getitem_52, mm_39, add_23, mm_42, mm_44, getitem_54, getitem_55, getitem_60, getitem_61, mm_46, add_27, mm_49, mm_51, getitem_63, getitem_64, getitem_69, getitem_70, mm_53, add_31, mm_56, mm_58, getitem_72, getitem_73, getitem_78, getitem_79, mm_60, add_35, mm_63, mm_65, getitem_81, getitem_82, getitem_87, getitem_88, mm_67, add_39, mm_70, mm_72, getitem_90, getitem_91, getitem_96, getitem_97, mm_74, add_43, mm_77, mm_79, getitem_99, getitem_100, getitem_105, getitem_106, mm_81, add_47, mm_84, mm_86, getitem_108, getitem_109, getitem_114, getitem_115, mm_88, add_51, mm_91, mm_93, getitem_117, getitem_118, getitem_123, getitem_124, mm_95, add_55, mm_98, mm_100, getitem_126, getitem_127, getitem_132, getitem_133, mm_102, add_59, mm_105, mm_107, getitem_135, getitem_136, getitem_141, getitem_142, mm_109, add_63, mm_112, mm_114, getitem_144, getitem_145, getitem_150, getitem_151, mm_116, add_67, mm_119, mm_121, getitem_153, getitem_154, getitem_159, getitem_160, mm_123, add_71, mm_126, mm_128, getitem_162, getitem_163, getitem_168, getitem_169, mm_130, add_75, mm_133, mm_135, getitem_171, getitem_172, getitem_177, getitem_178, mm_137, add_79, mm_140, mm_142, getitem_180, getitem_181, getitem_186, getitem_187, mm_144, add_83, mm_147, mm_149, getitem_189, getitem_190, getitem_195, getitem_196, mm_151, add_87, mm_154, mm_156, getitem_198, getitem_199, getitem_204, getitem_205, mm_158, add_91, mm_161, mm_163, getitem_207, getitem_208, getitem_213, getitem_214, mm_165, add_95, mm_168, mm_170, getitem_216, getitem_217, getitem_222, getitem_223, mm_172, add_99, mm_175, mm_177, getitem_225, getitem_226, getitem_231, getitem_232, mm_179, add_103, mm_182, mm_184, getitem_234, getitem_235, getitem_240, getitem_241, mm_186, add_107, mm_189, mm_191, getitem_243, getitem_244, getitem_249, getitem_250, mm_193, add_111, mm_196, mm_198, getitem_252, getitem_253, getitem_258, getitem_259, mm_200, add_115, mm_203, mm_205, getitem_261, getitem_262, getitem_267, getitem_268, mm_207, add_119, mm_210, mm_212, getitem_270, getitem_271, getitem_276, getitem_277, mm_214, add_123, mm_217, mm_219, getitem_279, getitem_280, getitem_285, getitem_286, mm_221, mm_223, rsqrt_64, view_1091, tangents_1):
        view_1093 = torch.ops.aten.view.default(tangents_1, [16384, 128256]);  tangents_1 = None
        permute_353 = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_225 = torch.ops.aten.mm.default(permute_353, view_1091);  permute_353 = view_1091 = None
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16);  primals_293 = None
        all_gather_into_tensor_290 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1060, 256, '0');  convert_element_type_1060 = None
        wait_tensor_290 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_290);  all_gather_into_tensor_290 = None
        permute_352 = torch.ops.aten.permute.default(wait_tensor_290, [1, 0]);  wait_tensor_290 = None
        permute_355 = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
        mm_226 = torch.ops.aten.mm.default(view_1093, permute_355);  view_1093 = permute_355 = None
        view_1094 = torch.ops.aten.view.default(mm_226, [2, 8192, 4096]);  mm_226 = None
        convert_element_type_1067 = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1067, 'avg', 256, '0');  convert_element_type_1067 = None
        wait_tensor_291 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        convert_element_type_1068 = torch.ops.prims.convert_element_type.default(view_1094, torch.float32);  view_1094 = None
        convert_element_type_1057 = torch.ops.prims.convert_element_type.default(primals_292, torch.bfloat16);  primals_292 = None
        all_gather_into_tensor_289 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1057, 256, '0');  convert_element_type_1057 = None
        wait_tensor_289 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_289);  all_gather_into_tensor_289 = None
        convert_element_type_1070 = torch.ops.prims.convert_element_type.default(wait_tensor_289, torch.float32);  wait_tensor_289 = None
        mul_258 = torch.ops.aten.mul.Tensor(convert_element_type_1068, convert_element_type_1070);  convert_element_type_1070 = None
        permute_347 = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3])
        view_1075 = torch.ops.aten.view.default(permute_347, [2, 8192, -1]);  permute_347 = None
        convert_element_type_1040 = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16);  primals_287 = None
        all_gather_into_tensor_284 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1040, 256, '0');  convert_element_type_1040 = None
        wait_tensor_284 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_284);  all_gather_into_tensor_284 = None
        permute_348 = torch.ops.aten.permute.default(wait_tensor_284, [1, 0]);  wait_tensor_284 = None
        view_1077 = torch.ops.aten.view.default(view_1075, [16384, 4096]);  view_1075 = None
        mm_220 = torch.ops.aten.mm.default(view_1077, permute_348)
        view_1078 = torch.ops.aten.view.default(mm_220, [2, 8192, 4096]);  mm_220 = None
        add_125 = torch.ops.aten.add.Tensor(add_123, view_1078);  view_1078 = None
        view_1088 = torch.ops.aten.view.default(mm_223, [2, 8192, 4096]);  mm_223 = None
        add_127 = torch.ops.aten.add.Tensor(add_125, view_1088);  view_1088 = None
        convert_element_type_1058 = torch.ops.prims.convert_element_type.default(add_127, torch.float32);  add_127 = None
        mul_256 = torch.ops.aten.mul.Tensor(convert_element_type_1058, rsqrt_64);  convert_element_type_1058 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_256, mul_258)
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_260, [2], True);  mul_260 = None
        div = torch.ops.aten.div.Tensor(mul_256, 4096)
        mul_261 = torch.ops.aten.mul.Tensor(div, sum_1);  div = sum_1 = None
        sub = torch.ops.aten.sub.Tensor(mul_258, mul_261);  mul_258 = mul_261 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub, rsqrt_64);  sub = rsqrt_64 = None
        mul_263 = torch.ops.aten.mul.Tensor(convert_element_type_1068, mul_256);  convert_element_type_1068 = mul_256 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1]);  mul_263 = None
        convert_element_type_1071 = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None
        convert_element_type_default_65 = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        reduce_scatter_tensor_1 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_65, 'avg', 256, '0');  convert_element_type_default_65 = None
        wait_tensor_292 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        view_1095 = torch.ops.aten.view.default(convert_element_type_1071, [16384, 4096])
        permute_357 = torch.ops.aten.permute.default(view_1095, [1, 0])
        convert_element_type_1043 = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        all_gather_into_tensor_285 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1043, 256, '0');  convert_element_type_1043 = None
        wait_tensor_285 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_285);  all_gather_into_tensor_285 = None
        convert_element_type_1044 = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        pow_64 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1044, 2)
        mean_63 = torch.ops.aten.mean.dim(pow_64, [2], True);  pow_64 = None
        add_126 = torch.ops.aten.add.Scalar(mean_63, 1e-05);  mean_63 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_252 = torch.ops.aten.mul.Tensor(convert_element_type_1044, rsqrt_63);  convert_element_type_1044 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, wait_tensor_285)
        convert_element_type_1045 = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None
        view_1081 = torch.ops.aten.view.default(convert_element_type_1045, [16384, 4096]);  convert_element_type_1045 = None
        view_1082 = torch.ops.aten.view.default(mm_221, [2, 8192, 14336]);  mm_221 = None
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(convert_element_type_1049)
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_1049, sigmoid_31);  sigmoid_31 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        convert_element_type_1051 = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        all_gather_into_tensor_287 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1051, 256, '0');  convert_element_type_1051 = None
        wait_tensor_287 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_287);  all_gather_into_tensor_287 = None
        permute_350 = torch.ops.aten.permute.default(wait_tensor_287, [1, 0]);  wait_tensor_287 = None
        mm_222 = torch.ops.aten.mm.default(view_1081, permute_350)
        view_1085 = torch.ops.aten.view.default(mm_222, [2, 8192, 14336]);  mm_222 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_1050, view_1085)
        view_1087 = torch.ops.aten.view.default(mul_255, [16384, 14336]);  mul_255 = None
        mm_227 = torch.ops.aten.mm.default(permute_357, view_1087);  permute_357 = view_1087 = None
        convert_element_type_1054 = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16);  primals_291 = None
        all_gather_into_tensor_288 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1054, 256, '0');  convert_element_type_1054 = None
        wait_tensor_288 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_288);  all_gather_into_tensor_288 = None
        permute_351 = torch.ops.aten.permute.default(wait_tensor_288, [1, 0]);  wait_tensor_288 = None
        permute_359 = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
        mm_228 = torch.ops.aten.mm.default(view_1095, permute_359);  view_1095 = permute_359 = None
        view_1096 = torch.ops.aten.view.default(mm_228, [2, 8192, 14336]);  mm_228 = None
        convert_element_type_1078 = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None
        reduce_scatter_tensor_2 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1078, 'avg', 256, '0');  convert_element_type_1078 = None
        wait_tensor_293 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        mul_264 = torch.ops.aten.mul.Tensor(view_1096, convert_element_type_1050);  convert_element_type_1050 = None
        mul_265 = torch.ops.aten.mul.Tensor(view_1096, view_1085);  view_1096 = view_1085 = None
        view_1097 = torch.ops.aten.view.default(mul_264, [16384, 14336]);  mul_264 = None
        permute_361 = torch.ops.aten.permute.default(view_1097, [1, 0])
        mm_229 = torch.ops.aten.mm.default(permute_361, view_1081);  permute_361 = None
        permute_363 = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
        mm_230 = torch.ops.aten.mm.default(view_1097, permute_363);  view_1097 = permute_363 = None
        view_1098 = torch.ops.aten.view.default(mm_230, [2, 8192, 4096]);  mm_230 = None
        convert_element_type_1083 = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None
        reduce_scatter_tensor_3 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1083, 'avg', 256, '0');  convert_element_type_1083 = None
        wait_tensor_294 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3);  reduce_scatter_tensor_3 = None
        convert_element_type_1084 = torch.ops.prims.convert_element_type.default(mul_265, torch.float32);  mul_265 = None
        neg = torch.ops.aten.neg.default(convert_element_type_1049)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add_129 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal = torch.ops.aten.reciprocal.default(add_129);  add_129 = None
        mul_266 = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_267 = torch.ops.aten.mul.Tensor(convert_element_type_1084, mul_266);  convert_element_type_1084 = None
        sub_1 = torch.ops.aten.sub.Tensor(1, mul_266);  mul_266 = None
        mul_268 = torch.ops.aten.mul.Tensor(convert_element_type_1049, sub_1);  convert_element_type_1049 = sub_1 = None
        add_130 = torch.ops.aten.add.Tensor(mul_268, 1);  mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_267, add_130);  mul_267 = add_130 = None
        convert_element_type_1086 = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        view_1099 = torch.ops.aten.view.default(convert_element_type_1086, [16384, 14336]);  convert_element_type_1086 = None
        permute_365 = torch.ops.aten.permute.default(view_1099, [1, 0])
        mm_231 = torch.ops.aten.mm.default(permute_365, view_1081);  permute_365 = view_1081 = None
        convert_element_type_1046 = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16);  primals_289 = None
        all_gather_into_tensor_286 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1046, 256, '0');  convert_element_type_1046 = None
        wait_tensor_286 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_286);  all_gather_into_tensor_286 = None
        permute_349 = torch.ops.aten.permute.default(wait_tensor_286, [1, 0]);  wait_tensor_286 = None
        permute_367 = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
        mm_232 = torch.ops.aten.mm.default(view_1099, permute_367);  view_1099 = permute_367 = None
        view_1100 = torch.ops.aten.view.default(mm_232, [2, 8192, 4096]);  mm_232 = None
        add_131 = torch.ops.aten.add.Tensor(view_1098, view_1100);  view_1098 = view_1100 = None
        convert_element_type_1091 = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        reduce_scatter_tensor_4 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1091, 'avg', 256, '0');  convert_element_type_1091 = None
        wait_tensor_295 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        convert_element_type_1092 = torch.ops.prims.convert_element_type.default(add_131, torch.float32);  add_131 = None
        convert_element_type_1094 = torch.ops.prims.convert_element_type.default(wait_tensor_285, torch.float32);  wait_tensor_285 = None
        mul_270 = torch.ops.aten.mul.Tensor(convert_element_type_1092, convert_element_type_1094);  convert_element_type_1094 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_252, mul_270)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        div_1 = torch.ops.aten.div.Tensor(mul_252, 4096)
        mul_273 = torch.ops.aten.mul.Tensor(div_1, sum_3);  div_1 = sum_3 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_270, mul_273);  mul_270 = mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_63);  sub_2 = rsqrt_63 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_1092, mul_252);  convert_element_type_1092 = mul_252 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        convert_element_type_1095 = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None
        add_132 = torch.ops.aten.add.Tensor(convert_element_type_1071, convert_element_type_1095);  convert_element_type_1071 = convert_element_type_1095 = None
        convert_element_type_default_64 = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        reduce_scatter_tensor_5 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_64, 'avg', 256, '0');  convert_element_type_default_64 = None
        wait_tensor_296 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_5);  reduce_scatter_tensor_5 = None
        view_1101 = torch.ops.aten.view.default(add_132, [16384, 4096])
        permute_369 = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_233 = torch.ops.aten.mm.default(permute_369, view_1077);  permute_369 = view_1077 = None
        permute_371 = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
        mm_234 = torch.ops.aten.mm.default(view_1101, permute_371);  view_1101 = permute_371 = None
        view_1102 = torch.ops.aten.view.default(mm_234, [2, 8192, 4096]);  mm_234 = None
        convert_element_type_1102 = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        reduce_scatter_tensor_6 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1102, 'avg', 256, '0');  convert_element_type_1102 = None
        wait_tensor_297 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_6);  reduce_scatter_tensor_6 = None
        view_1103 = torch.ops.aten.view.default(view_1102, [2, 8192, 32, 128]);  view_1102 = None
        permute_373 = torch.ops.aten.permute.default(view_1103, [0, 2, 1, 3]);  view_1103 = None
        view_16 = torch.ops.aten.view.default(primals_3, [1, 8192, 1, 64]);  primals_3 = None
        convert_element_type_1024 = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        all_gather_into_tensor_280 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1024, 256, '0');  convert_element_type_1024 = None
        wait_tensor_280 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_280);  all_gather_into_tensor_280 = None
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(add_123, torch.float32);  add_123 = None
        pow_63 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1025, 2)
        mean_62 = torch.ops.aten.mean.dim(pow_63, [2], True);  pow_63 = None
        add_124 = torch.ops.aten.add.Scalar(mean_62, 1e-05);  mean_62 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_1025, rsqrt_62);  convert_element_type_1025 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, wait_tensor_280)
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        view_1057 = torch.ops.aten.view.default(convert_element_type_1026, [16384, 4096]);  convert_element_type_1026 = None
        view_1058 = torch.ops.aten.view.default(mm_217, [2, 8192, 4096]);  mm_217 = None
        convert_element_type_1030 = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        all_gather_into_tensor_282 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1030, 256, '0');  convert_element_type_1030 = None
        wait_tensor_282 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_282);  all_gather_into_tensor_282 = None
        permute_342 = torch.ops.aten.permute.default(wait_tensor_282, [1, 0]);  wait_tensor_282 = None
        mm_218 = torch.ops.aten.mm.default(view_1057, permute_342)
        view_1061 = torch.ops.aten.view.default(mm_218, [2, 8192, 1024]);  mm_218 = None
        view_1064 = torch.ops.aten.view.default(mm_219, [2, 8192, 1024]);  mm_219 = None
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
        mul_251 = torch.ops.aten.mul.Tensor(view_as_complex_63, view_16);  view_as_complex_63 = None
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
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_373, permute_344, permute_345, permute_346, getitem_279, getitem_280, getitem_285, getitem_286, None, None, None, 8192, 8192, 0.0, True);  permute_373 = permute_344 = permute_345 = permute_346 = getitem_279 = getitem_280 = getitem_285 = getitem_286 = None
        getitem_288 = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_289 = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_290 = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None
        permute_374 = torch.ops.aten.permute.default(getitem_290, [0, 2, 1, 3]);  getitem_290 = None
        permute_375 = torch.ops.aten.permute.default(getitem_289, [0, 2, 1, 3]);  getitem_289 = None
        permute_376 = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None
        view_1104 = torch.ops.aten.view.default(permute_374, [2, 8192, 8, 4, 128]);  permute_374 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_1104, [3], True);  view_1104 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_5, 3);  sum_5 = None
        view_1105 = torch.ops.aten.view.default(permute_375, [2, 8192, 8, 4, 128]);  permute_375 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_1105, [3], True);  view_1105 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_6, 3);  sum_6 = None
        convert_element_type_1103 = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_1104 = torch.ops.prims.convert_element_type.default(permute_376, torch.float32);  permute_376 = None
        view_1106 = torch.ops.aten.view.default(convert_element_type_1103, [2, 8192, 8, 64, 2]);  convert_element_type_1103 = None
        view_as_complex_64 = torch.ops.aten.view_as_complex.default(view_1106);  view_1106 = None
        _conj = torch.ops.aten._conj.default(view_16)
        mul_276 = torch.ops.aten.mul.Tensor(view_as_complex_64, _conj);  view_as_complex_64 = None
        view_1107 = torch.ops.aten.view.default(convert_element_type_1104, [2, 8192, 32, 64, 2]);  convert_element_type_1104 = None
        view_as_complex_65 = torch.ops.aten.view_as_complex.default(view_1107);  view_1107 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_as_complex_65, _conj);  view_as_complex_65 = None
        view_as_real_64 = torch.ops.aten.view_as_real.default(mul_276);  mul_276 = None
        view_1108 = torch.ops.aten.view.default(view_as_real_64, [2, 8192, 8, 128]);  view_as_real_64 = None
        convert_element_type_1105 = torch.ops.prims.convert_element_type.default(view_1108, torch.bfloat16);  view_1108 = None
        view_as_real_65 = torch.ops.aten.view_as_real.default(mul_277);  mul_277 = None
        view_1109 = torch.ops.aten.view.default(view_as_real_65, [2, 8192, 32, 128]);  view_as_real_65 = None
        convert_element_type_1106 = torch.ops.prims.convert_element_type.default(view_1109, torch.bfloat16);  view_1109 = None
        view_1110 = torch.ops.aten.view.default(squeeze, [2, 8192, 1024]);  squeeze = None
        view_1111 = torch.ops.aten.view.default(convert_element_type_1105, [2, 8192, 1024]);  convert_element_type_1105 = None
        view_1112 = torch.ops.aten.view.default(convert_element_type_1106, [2, 8192, 4096]);  convert_element_type_1106 = None
        view_1113 = torch.ops.aten.view.default(view_1110, [16384, 1024]);  view_1110 = None
        permute_377 = torch.ops.aten.permute.default(view_1113, [1, 0])
        mm_235 = torch.ops.aten.mm.default(permute_377, view_1057);  permute_377 = None
        convert_element_type_1033 = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16);  primals_286 = None
        all_gather_into_tensor_283 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1033, 256, '0');  convert_element_type_1033 = None
        wait_tensor_283 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_283);  all_gather_into_tensor_283 = None
        permute_343 = torch.ops.aten.permute.default(wait_tensor_283, [1, 0]);  wait_tensor_283 = None
        permute_379 = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
        mm_236 = torch.ops.aten.mm.default(view_1113, permute_379);  view_1113 = permute_379 = None
        view_1114 = torch.ops.aten.view.default(mm_236, [2, 8192, 4096]);  mm_236 = None
        convert_element_type_1111 = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        reduce_scatter_tensor_7 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1111, 'avg', 256, '0');  convert_element_type_1111 = None
        wait_tensor_298 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7);  reduce_scatter_tensor_7 = None
        view_1115 = torch.ops.aten.view.default(view_1111, [16384, 1024]);  view_1111 = None
        permute_381 = torch.ops.aten.permute.default(view_1115, [1, 0])
        mm_237 = torch.ops.aten.mm.default(permute_381, view_1057);  permute_381 = None
        permute_383 = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        mm_238 = torch.ops.aten.mm.default(view_1115, permute_383);  view_1115 = permute_383 = None
        view_1116 = torch.ops.aten.view.default(mm_238, [2, 8192, 4096]);  mm_238 = None
        add_133 = torch.ops.aten.add.Tensor(view_1114, view_1116);  view_1114 = view_1116 = None
        convert_element_type_1116 = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        reduce_scatter_tensor_8 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1116, 'avg', 256, '0');  convert_element_type_1116 = None
        wait_tensor_299 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_8);  reduce_scatter_tensor_8 = None
        view_1117 = torch.ops.aten.view.default(view_1112, [16384, 4096]);  view_1112 = None
        permute_385 = torch.ops.aten.permute.default(view_1117, [1, 0])
        mm_239 = torch.ops.aten.mm.default(permute_385, view_1057);  permute_385 = view_1057 = None
        convert_element_type_1027 = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        all_gather_into_tensor_281 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1027, 256, '0');  convert_element_type_1027 = None
        wait_tensor_281 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_281);  all_gather_into_tensor_281 = None
        permute_341 = torch.ops.aten.permute.default(wait_tensor_281, [1, 0]);  wait_tensor_281 = None
        permute_387 = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        mm_240 = torch.ops.aten.mm.default(view_1117, permute_387);  view_1117 = permute_387 = None
        view_1118 = torch.ops.aten.view.default(mm_240, [2, 8192, 4096]);  mm_240 = None
        add_134 = torch.ops.aten.add.Tensor(add_133, view_1118);  add_133 = view_1118 = None
        convert_element_type_1121 = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None
        reduce_scatter_tensor_9 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1121, 'avg', 256, '0');  convert_element_type_1121 = None
        wait_tensor_300 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_9);  reduce_scatter_tensor_9 = None
        convert_element_type_1122 = torch.ops.prims.convert_element_type.default(add_134, torch.float32);  add_134 = None
        convert_element_type_1124 = torch.ops.prims.convert_element_type.default(wait_tensor_280, torch.float32);  wait_tensor_280 = None
        mul_278 = torch.ops.aten.mul.Tensor(convert_element_type_1122, convert_element_type_1124);  convert_element_type_1124 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_248, mul_278)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        div_2 = torch.ops.aten.div.Tensor(mul_248, 4096)
        mul_281 = torch.ops.aten.mul.Tensor(div_2, sum_7);  div_2 = sum_7 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_278, mul_281);  mul_278 = mul_281 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_62);  sub_3 = rsqrt_62 = None
        mul_283 = torch.ops.aten.mul.Tensor(convert_element_type_1122, mul_248);  convert_element_type_1122 = mul_248 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        convert_element_type_1125 = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        add_135 = torch.ops.aten.add.Tensor(add_132, convert_element_type_1125);  add_132 = convert_element_type_1125 = None
        convert_element_type_default_63 = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        reduce_scatter_tensor_10 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_63, 'avg', 256, '0');  convert_element_type_default_63 = None
        wait_tensor_301 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_10);  reduce_scatter_tensor_10 = None
        view_1119 = torch.ops.aten.view.default(add_135, [16384, 4096])
        permute_389 = torch.ops.aten.permute.default(view_1119, [1, 0])
        permute_336 = torch.ops.aten.permute.default(getitem_270, [0, 2, 1, 3])
        view_1041 = torch.ops.aten.view.default(permute_336, [2, 8192, -1]);  permute_336 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        all_gather_into_tensor_275 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1007, 256, '0');  convert_element_type_1007 = None
        wait_tensor_275 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_275);  all_gather_into_tensor_275 = None
        permute_337 = torch.ops.aten.permute.default(wait_tensor_275, [1, 0]);  wait_tensor_275 = None
        view_1043 = torch.ops.aten.view.default(view_1041, [16384, 4096]);  view_1041 = None
        mm_213 = torch.ops.aten.mm.default(view_1043, permute_337)
        view_1044 = torch.ops.aten.view.default(mm_213, [2, 8192, 4096]);  mm_213 = None
        add_121 = torch.ops.aten.add.Tensor(add_119, view_1044);  view_1044 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16);  primals_279 = None
        all_gather_into_tensor_276 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1010, 256, '0');  convert_element_type_1010 = None
        wait_tensor_276 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_276);  all_gather_into_tensor_276 = None
        convert_element_type_1011 = torch.ops.prims.convert_element_type.default(add_121, torch.float32);  add_121 = None
        pow_62 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1011, 2)
        mean_61 = torch.ops.aten.mean.dim(pow_62, [2], True);  pow_62 = None
        add_122 = torch.ops.aten.add.Scalar(mean_61, 1e-05);  mean_61 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_244 = torch.ops.aten.mul.Tensor(convert_element_type_1011, rsqrt_61);  convert_element_type_1011 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, wait_tensor_276)
        convert_element_type_1012 = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16);  mul_245 = None
        view_1047 = torch.ops.aten.view.default(convert_element_type_1012, [16384, 4096]);  convert_element_type_1012 = None
        view_1048 = torch.ops.aten.view.default(mm_214, [2, 8192, 14336]);  mm_214 = None
        convert_element_type_1016 = torch.ops.prims.convert_element_type.default(view_1048, torch.float32);  view_1048 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(convert_element_type_1016)
        mul_246 = torch.ops.aten.mul.Tensor(convert_element_type_1016, sigmoid_30);  sigmoid_30 = None
        convert_element_type_1017 = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        convert_element_type_1018 = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        all_gather_into_tensor_278 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1018, 256, '0');  convert_element_type_1018 = None
        wait_tensor_278 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_278);  all_gather_into_tensor_278 = None
        permute_339 = torch.ops.aten.permute.default(wait_tensor_278, [1, 0]);  wait_tensor_278 = None
        mm_215 = torch.ops.aten.mm.default(view_1047, permute_339)
        view_1051 = torch.ops.aten.view.default(mm_215, [2, 8192, 14336]);  mm_215 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_1017, view_1051)
        view_1053 = torch.ops.aten.view.default(mul_247, [16384, 14336]);  mul_247 = None
        mm_241 = torch.ops.aten.mm.default(permute_389, view_1053);  permute_389 = view_1053 = None
        convert_element_type_1021 = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        all_gather_into_tensor_279 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1021, 256, '0');  convert_element_type_1021 = None
        wait_tensor_279 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_279);  all_gather_into_tensor_279 = None
        permute_340 = torch.ops.aten.permute.default(wait_tensor_279, [1, 0]);  wait_tensor_279 = None
        permute_391 = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
        mm_242 = torch.ops.aten.mm.default(view_1119, permute_391);  view_1119 = permute_391 = None
        view_1120 = torch.ops.aten.view.default(mm_242, [2, 8192, 14336]);  mm_242 = None
        convert_element_type_1132 = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None
        reduce_scatter_tensor_11 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1132, 'avg', 256, '0');  convert_element_type_1132 = None
        wait_tensor_302 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_11);  reduce_scatter_tensor_11 = None
        mul_284 = torch.ops.aten.mul.Tensor(view_1120, convert_element_type_1017);  convert_element_type_1017 = None
        mul_285 = torch.ops.aten.mul.Tensor(view_1120, view_1051);  view_1120 = view_1051 = None
        view_1121 = torch.ops.aten.view.default(mul_284, [16384, 14336]);  mul_284 = None
        permute_393 = torch.ops.aten.permute.default(view_1121, [1, 0])
        mm_243 = torch.ops.aten.mm.default(permute_393, view_1047);  permute_393 = None
        permute_395 = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
        mm_244 = torch.ops.aten.mm.default(view_1121, permute_395);  view_1121 = permute_395 = None
        view_1122 = torch.ops.aten.view.default(mm_244, [2, 8192, 4096]);  mm_244 = None
        convert_element_type_1137 = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        reduce_scatter_tensor_12 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1137, 'avg', 256, '0');  convert_element_type_1137 = None
        wait_tensor_303 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_12);  reduce_scatter_tensor_12 = None
        convert_element_type_1138 = torch.ops.prims.convert_element_type.default(mul_285, torch.float32);  mul_285 = None
        neg_1 = torch.ops.aten.neg.default(convert_element_type_1016)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_136 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(add_136);  add_136 = None
        mul_286 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_287 = torch.ops.aten.mul.Tensor(convert_element_type_1138, mul_286);  convert_element_type_1138 = None
        sub_4 = torch.ops.aten.sub.Tensor(1, mul_286);  mul_286 = None
        mul_288 = torch.ops.aten.mul.Tensor(convert_element_type_1016, sub_4);  convert_element_type_1016 = sub_4 = None
        add_137 = torch.ops.aten.add.Tensor(mul_288, 1);  mul_288 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_287, add_137);  mul_287 = add_137 = None
        convert_element_type_1140 = torch.ops.prims.convert_element_type.default(mul_289, torch.bfloat16);  mul_289 = None
        view_1123 = torch.ops.aten.view.default(convert_element_type_1140, [16384, 14336]);  convert_element_type_1140 = None
        permute_397 = torch.ops.aten.permute.default(view_1123, [1, 0])
        mm_245 = torch.ops.aten.mm.default(permute_397, view_1047);  permute_397 = view_1047 = None
        convert_element_type_1013 = torch.ops.prims.convert_element_type.default(primals_280, torch.bfloat16);  primals_280 = None
        all_gather_into_tensor_277 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1013, 256, '0');  convert_element_type_1013 = None
        wait_tensor_277 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_277);  all_gather_into_tensor_277 = None
        permute_338 = torch.ops.aten.permute.default(wait_tensor_277, [1, 0]);  wait_tensor_277 = None
        permute_399 = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        mm_246 = torch.ops.aten.mm.default(view_1123, permute_399);  view_1123 = permute_399 = None
        view_1124 = torch.ops.aten.view.default(mm_246, [2, 8192, 4096]);  mm_246 = None
        add_138 = torch.ops.aten.add.Tensor(view_1122, view_1124);  view_1122 = view_1124 = None
        convert_element_type_1145 = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        reduce_scatter_tensor_13 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1145, 'avg', 256, '0');  convert_element_type_1145 = None
        wait_tensor_304 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_13);  reduce_scatter_tensor_13 = None
        convert_element_type_1146 = torch.ops.prims.convert_element_type.default(add_138, torch.float32);  add_138 = None
        convert_element_type_1148 = torch.ops.prims.convert_element_type.default(wait_tensor_276, torch.float32);  wait_tensor_276 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_1146, convert_element_type_1148);  convert_element_type_1148 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_244, mul_290)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        div_3 = torch.ops.aten.div.Tensor(mul_244, 4096)
        mul_293 = torch.ops.aten.mul.Tensor(div_3, sum_9);  div_3 = sum_9 = None
        sub_5 = torch.ops.aten.sub.Tensor(mul_290, mul_293);  mul_290 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_61);  sub_5 = rsqrt_61 = None
        mul_295 = torch.ops.aten.mul.Tensor(convert_element_type_1146, mul_244);  convert_element_type_1146 = mul_244 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        convert_element_type_1149 = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None
        add_139 = torch.ops.aten.add.Tensor(add_135, convert_element_type_1149);  add_135 = convert_element_type_1149 = None
        convert_element_type_default_62 = torch.ops.prims.convert_element_type.default(sum_10, torch.float32);  sum_10 = None
        reduce_scatter_tensor_14 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_62, 'avg', 256, '0');  convert_element_type_default_62 = None
        wait_tensor_305 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_14);  reduce_scatter_tensor_14 = None
        view_1125 = torch.ops.aten.view.default(add_139, [16384, 4096])
        permute_401 = torch.ops.aten.permute.default(view_1125, [1, 0])
        mm_247 = torch.ops.aten.mm.default(permute_401, view_1043);  permute_401 = view_1043 = None
        permute_403 = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        mm_248 = torch.ops.aten.mm.default(view_1125, permute_403);  view_1125 = permute_403 = None
        view_1126 = torch.ops.aten.view.default(mm_248, [2, 8192, 4096]);  mm_248 = None
        convert_element_type_1156 = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        reduce_scatter_tensor_15 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1156, 'avg', 256, '0');  convert_element_type_1156 = None
        wait_tensor_306 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_15);  reduce_scatter_tensor_15 = None
        view_1127 = torch.ops.aten.view.default(view_1126, [2, 8192, 32, 128]);  view_1126 = None
        permute_405 = torch.ops.aten.permute.default(view_1127, [0, 2, 1, 3]);  view_1127 = None
        convert_element_type_991 = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16);  primals_274 = None
        all_gather_into_tensor_271 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_991, 256, '0');  convert_element_type_991 = None
        wait_tensor_271 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_271);  all_gather_into_tensor_271 = None
        convert_element_type_992 = torch.ops.prims.convert_element_type.default(add_119, torch.float32);  add_119 = None
        pow_61 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_992, 2)
        mean_60 = torch.ops.aten.mean.dim(pow_61, [2], True);  pow_61 = None
        add_120 = torch.ops.aten.add.Scalar(mean_60, 1e-05);  mean_60 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_992, rsqrt_60);  convert_element_type_992 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, wait_tensor_271)
        convert_element_type_993 = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        view_1023 = torch.ops.aten.view.default(convert_element_type_993, [16384, 4096]);  convert_element_type_993 = None
        view_1024 = torch.ops.aten.view.default(mm_210, [2, 8192, 4096]);  mm_210 = None
        convert_element_type_997 = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        all_gather_into_tensor_273 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_997, 256, '0');  convert_element_type_997 = None
        wait_tensor_273 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_273);  all_gather_into_tensor_273 = None
        permute_331 = torch.ops.aten.permute.default(wait_tensor_273, [1, 0]);  wait_tensor_273 = None
        mm_211 = torch.ops.aten.mm.default(view_1023, permute_331)
        view_1027 = torch.ops.aten.view.default(mm_211, [2, 8192, 1024]);  mm_211 = None
        view_1030 = torch.ops.aten.view.default(mm_212, [2, 8192, 1024]);  mm_212 = None
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
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_405, permute_333, permute_334, permute_335, getitem_270, getitem_271, getitem_276, getitem_277, None, None, None, 8192, 8192, 0.0, True);  permute_405 = permute_333 = permute_334 = permute_335 = getitem_270 = getitem_271 = getitem_276 = getitem_277 = None
        getitem_291 = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_292 = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_293 = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None
        permute_406 = torch.ops.aten.permute.default(getitem_293, [0, 2, 1, 3]);  getitem_293 = None
        permute_407 = torch.ops.aten.permute.default(getitem_292, [0, 2, 1, 3]);  getitem_292 = None
        permute_408 = torch.ops.aten.permute.default(getitem_291, [0, 2, 1, 3]);  getitem_291 = None
        view_1128 = torch.ops.aten.view.default(permute_406, [2, 8192, 8, 4, 128]);  permute_406 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_1128, [3], True);  view_1128 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(sum_11, 3);  sum_11 = None
        view_1129 = torch.ops.aten.view.default(permute_407, [2, 8192, 8, 4, 128]);  permute_407 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_1129, [3], True);  view_1129 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(sum_12, 3);  sum_12 = None
        convert_element_type_1157 = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        convert_element_type_1158 = torch.ops.prims.convert_element_type.default(permute_408, torch.float32);  permute_408 = None
        view_1130 = torch.ops.aten.view.default(convert_element_type_1157, [2, 8192, 8, 64, 2]);  convert_element_type_1157 = None
        view_as_complex_66 = torch.ops.aten.view_as_complex.default(view_1130);  view_1130 = None
        mul_296 = torch.ops.aten.mul.Tensor(view_as_complex_66, _conj);  view_as_complex_66 = None
        view_1131 = torch.ops.aten.view.default(convert_element_type_1158, [2, 8192, 32, 64, 2]);  convert_element_type_1158 = None
        view_as_complex_67 = torch.ops.aten.view_as_complex.default(view_1131);  view_1131 = None
        mul_297 = torch.ops.aten.mul.Tensor(view_as_complex_67, _conj);  view_as_complex_67 = None
        view_as_real_66 = torch.ops.aten.view_as_real.default(mul_296);  mul_296 = None
        view_1132 = torch.ops.aten.view.default(view_as_real_66, [2, 8192, 8, 128]);  view_as_real_66 = None
        convert_element_type_1159 = torch.ops.prims.convert_element_type.default(view_1132, torch.bfloat16);  view_1132 = None
        view_as_real_67 = torch.ops.aten.view_as_real.default(mul_297);  mul_297 = None
        view_1133 = torch.ops.aten.view.default(view_as_real_67, [2, 8192, 32, 128]);  view_as_real_67 = None
        convert_element_type_1160 = torch.ops.prims.convert_element_type.default(view_1133, torch.bfloat16);  view_1133 = None
        view_1134 = torch.ops.aten.view.default(squeeze_2, [2, 8192, 1024]);  squeeze_2 = None
        view_1135 = torch.ops.aten.view.default(convert_element_type_1159, [2, 8192, 1024]);  convert_element_type_1159 = None
        view_1136 = torch.ops.aten.view.default(convert_element_type_1160, [2, 8192, 4096]);  convert_element_type_1160 = None
        view_1137 = torch.ops.aten.view.default(view_1134, [16384, 1024]);  view_1134 = None
        permute_409 = torch.ops.aten.permute.default(view_1137, [1, 0])
        mm_249 = torch.ops.aten.mm.default(permute_409, view_1023);  permute_409 = None
        convert_element_type_1000 = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16);  primals_277 = None
        all_gather_into_tensor_274 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1000, 256, '0');  convert_element_type_1000 = None
        wait_tensor_274 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_274);  all_gather_into_tensor_274 = None
        permute_332 = torch.ops.aten.permute.default(wait_tensor_274, [1, 0]);  wait_tensor_274 = None
        permute_411 = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
        mm_250 = torch.ops.aten.mm.default(view_1137, permute_411);  view_1137 = permute_411 = None
        view_1138 = torch.ops.aten.view.default(mm_250, [2, 8192, 4096]);  mm_250 = None
        convert_element_type_1165 = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        reduce_scatter_tensor_16 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1165, 'avg', 256, '0');  convert_element_type_1165 = None
        wait_tensor_307 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_16);  reduce_scatter_tensor_16 = None
        view_1139 = torch.ops.aten.view.default(view_1135, [16384, 1024]);  view_1135 = None
        permute_413 = torch.ops.aten.permute.default(view_1139, [1, 0])
        mm_251 = torch.ops.aten.mm.default(permute_413, view_1023);  permute_413 = None
        permute_415 = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
        mm_252 = torch.ops.aten.mm.default(view_1139, permute_415);  view_1139 = permute_415 = None
        view_1140 = torch.ops.aten.view.default(mm_252, [2, 8192, 4096]);  mm_252 = None
        add_140 = torch.ops.aten.add.Tensor(view_1138, view_1140);  view_1138 = view_1140 = None
        convert_element_type_1170 = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        reduce_scatter_tensor_17 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1170, 'avg', 256, '0');  convert_element_type_1170 = None
        wait_tensor_308 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_17);  reduce_scatter_tensor_17 = None
        view_1141 = torch.ops.aten.view.default(view_1136, [16384, 4096]);  view_1136 = None
        permute_417 = torch.ops.aten.permute.default(view_1141, [1, 0])
        mm_253 = torch.ops.aten.mm.default(permute_417, view_1023);  permute_417 = view_1023 = None
        convert_element_type_994 = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        all_gather_into_tensor_272 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_994, 256, '0');  convert_element_type_994 = None
        wait_tensor_272 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_272);  all_gather_into_tensor_272 = None
        permute_330 = torch.ops.aten.permute.default(wait_tensor_272, [1, 0]);  wait_tensor_272 = None
        permute_419 = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
        mm_254 = torch.ops.aten.mm.default(view_1141, permute_419);  view_1141 = permute_419 = None
        view_1142 = torch.ops.aten.view.default(mm_254, [2, 8192, 4096]);  mm_254 = None
        add_141 = torch.ops.aten.add.Tensor(add_140, view_1142);  add_140 = view_1142 = None
        convert_element_type_1175 = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None
        reduce_scatter_tensor_18 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1175, 'avg', 256, '0');  convert_element_type_1175 = None
        wait_tensor_309 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_18);  reduce_scatter_tensor_18 = None
        convert_element_type_1176 = torch.ops.prims.convert_element_type.default(add_141, torch.float32);  add_141 = None
        convert_element_type_1178 = torch.ops.prims.convert_element_type.default(wait_tensor_271, torch.float32);  wait_tensor_271 = None
        mul_298 = torch.ops.aten.mul.Tensor(convert_element_type_1176, convert_element_type_1178);  convert_element_type_1178 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_240, mul_298)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        div_4 = torch.ops.aten.div.Tensor(mul_240, 4096)
        mul_301 = torch.ops.aten.mul.Tensor(div_4, sum_13);  div_4 = sum_13 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_298, mul_301);  mul_298 = mul_301 = None
        mul_302 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_60);  sub_6 = rsqrt_60 = None
        mul_303 = torch.ops.aten.mul.Tensor(convert_element_type_1176, mul_240);  convert_element_type_1176 = mul_240 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        convert_element_type_1179 = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        add_142 = torch.ops.aten.add.Tensor(add_139, convert_element_type_1179);  add_139 = convert_element_type_1179 = None
        convert_element_type_default_61 = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        reduce_scatter_tensor_19 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_61, 'avg', 256, '0');  convert_element_type_default_61 = None
        wait_tensor_310 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_19);  reduce_scatter_tensor_19 = None
        view_1143 = torch.ops.aten.view.default(add_142, [16384, 4096])
        permute_421 = torch.ops.aten.permute.default(view_1143, [1, 0])
        permute_325 = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3])
        view_1007 = torch.ops.aten.view.default(permute_325, [2, 8192, -1]);  permute_325 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        all_gather_into_tensor_266 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_974, 256, '0');  convert_element_type_974 = None
        wait_tensor_266 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_266);  all_gather_into_tensor_266 = None
        permute_326 = torch.ops.aten.permute.default(wait_tensor_266, [1, 0]);  wait_tensor_266 = None
        view_1009 = torch.ops.aten.view.default(view_1007, [16384, 4096]);  view_1007 = None
        mm_206 = torch.ops.aten.mm.default(view_1009, permute_326)
        view_1010 = torch.ops.aten.view.default(mm_206, [2, 8192, 4096]);  mm_206 = None
        add_117 = torch.ops.aten.add.Tensor(add_115, view_1010);  view_1010 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        all_gather_into_tensor_267 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_977, 256, '0');  convert_element_type_977 = None
        wait_tensor_267 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_267);  all_gather_into_tensor_267 = None
        convert_element_type_978 = torch.ops.prims.convert_element_type.default(add_117, torch.float32);  add_117 = None
        pow_60 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_978, 2)
        mean_59 = torch.ops.aten.mean.dim(pow_60, [2], True);  pow_60 = None
        add_118 = torch.ops.aten.add.Scalar(mean_59, 1e-05);  mean_59 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_236 = torch.ops.aten.mul.Tensor(convert_element_type_978, rsqrt_59);  convert_element_type_978 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, wait_tensor_267)
        convert_element_type_979 = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        view_1013 = torch.ops.aten.view.default(convert_element_type_979, [16384, 4096]);  convert_element_type_979 = None
        view_1014 = torch.ops.aten.view.default(mm_207, [2, 8192, 14336]);  mm_207 = None
        convert_element_type_983 = torch.ops.prims.convert_element_type.default(view_1014, torch.float32);  view_1014 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(convert_element_type_983)
        mul_238 = torch.ops.aten.mul.Tensor(convert_element_type_983, sigmoid_29);  sigmoid_29 = None
        convert_element_type_984 = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None
        convert_element_type_985 = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        all_gather_into_tensor_269 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_985, 256, '0');  convert_element_type_985 = None
        wait_tensor_269 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_269);  all_gather_into_tensor_269 = None
        permute_328 = torch.ops.aten.permute.default(wait_tensor_269, [1, 0]);  wait_tensor_269 = None
        mm_208 = torch.ops.aten.mm.default(view_1013, permute_328)
        view_1017 = torch.ops.aten.view.default(mm_208, [2, 8192, 14336]);  mm_208 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_984, view_1017)
        view_1019 = torch.ops.aten.view.default(mul_239, [16384, 14336]);  mul_239 = None
        mm_255 = torch.ops.aten.mm.default(permute_421, view_1019);  permute_421 = view_1019 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16);  primals_273 = None
        all_gather_into_tensor_270 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_988, 256, '0');  convert_element_type_988 = None
        wait_tensor_270 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_270);  all_gather_into_tensor_270 = None
        permute_329 = torch.ops.aten.permute.default(wait_tensor_270, [1, 0]);  wait_tensor_270 = None
        permute_423 = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
        mm_256 = torch.ops.aten.mm.default(view_1143, permute_423);  view_1143 = permute_423 = None
        view_1144 = torch.ops.aten.view.default(mm_256, [2, 8192, 14336]);  mm_256 = None
        convert_element_type_1186 = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        reduce_scatter_tensor_20 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1186, 'avg', 256, '0');  convert_element_type_1186 = None
        wait_tensor_311 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_20);  reduce_scatter_tensor_20 = None
        mul_304 = torch.ops.aten.mul.Tensor(view_1144, convert_element_type_984);  convert_element_type_984 = None
        mul_305 = torch.ops.aten.mul.Tensor(view_1144, view_1017);  view_1144 = view_1017 = None
        view_1145 = torch.ops.aten.view.default(mul_304, [16384, 14336]);  mul_304 = None
        permute_425 = torch.ops.aten.permute.default(view_1145, [1, 0])
        mm_257 = torch.ops.aten.mm.default(permute_425, view_1013);  permute_425 = None
        permute_427 = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
        mm_258 = torch.ops.aten.mm.default(view_1145, permute_427);  view_1145 = permute_427 = None
        view_1146 = torch.ops.aten.view.default(mm_258, [2, 8192, 4096]);  mm_258 = None
        convert_element_type_1191 = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        reduce_scatter_tensor_21 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1191, 'avg', 256, '0');  convert_element_type_1191 = None
        wait_tensor_312 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_21);  reduce_scatter_tensor_21 = None
        convert_element_type_1192 = torch.ops.prims.convert_element_type.default(mul_305, torch.float32);  mul_305 = None
        neg_2 = torch.ops.aten.neg.default(convert_element_type_983)
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_143 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(add_143);  add_143 = None
        mul_306 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_1192, mul_306);  convert_element_type_1192 = None
        sub_7 = torch.ops.aten.sub.Tensor(1, mul_306);  mul_306 = None
        mul_308 = torch.ops.aten.mul.Tensor(convert_element_type_983, sub_7);  convert_element_type_983 = sub_7 = None
        add_144 = torch.ops.aten.add.Tensor(mul_308, 1);  mul_308 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_307, add_144);  mul_307 = add_144 = None
        convert_element_type_1194 = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None
        view_1147 = torch.ops.aten.view.default(convert_element_type_1194, [16384, 14336]);  convert_element_type_1194 = None
        permute_429 = torch.ops.aten.permute.default(view_1147, [1, 0])
        mm_259 = torch.ops.aten.mm.default(permute_429, view_1013);  permute_429 = view_1013 = None
        convert_element_type_980 = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        all_gather_into_tensor_268 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_980, 256, '0');  convert_element_type_980 = None
        wait_tensor_268 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_268);  all_gather_into_tensor_268 = None
        permute_327 = torch.ops.aten.permute.default(wait_tensor_268, [1, 0]);  wait_tensor_268 = None
        permute_431 = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
        mm_260 = torch.ops.aten.mm.default(view_1147, permute_431);  view_1147 = permute_431 = None
        view_1148 = torch.ops.aten.view.default(mm_260, [2, 8192, 4096]);  mm_260 = None
        add_145 = torch.ops.aten.add.Tensor(view_1146, view_1148);  view_1146 = view_1148 = None
        convert_element_type_1199 = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        reduce_scatter_tensor_22 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1199, 'avg', 256, '0');  convert_element_type_1199 = None
        wait_tensor_313 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_22);  reduce_scatter_tensor_22 = None
        convert_element_type_1200 = torch.ops.prims.convert_element_type.default(add_145, torch.float32);  add_145 = None
        convert_element_type_1202 = torch.ops.prims.convert_element_type.default(wait_tensor_267, torch.float32);  wait_tensor_267 = None
        mul_310 = torch.ops.aten.mul.Tensor(convert_element_type_1200, convert_element_type_1202);  convert_element_type_1202 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_236, mul_310)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        div_5 = torch.ops.aten.div.Tensor(mul_236, 4096)
        mul_313 = torch.ops.aten.mul.Tensor(div_5, sum_15);  div_5 = sum_15 = None
        sub_8 = torch.ops.aten.sub.Tensor(mul_310, mul_313);  mul_310 = mul_313 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_59);  sub_8 = rsqrt_59 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_1200, mul_236);  convert_element_type_1200 = mul_236 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        convert_element_type_1203 = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16);  mul_314 = None
        add_146 = torch.ops.aten.add.Tensor(add_142, convert_element_type_1203);  add_142 = convert_element_type_1203 = None
        convert_element_type_default_60 = torch.ops.prims.convert_element_type.default(sum_16, torch.float32);  sum_16 = None
        reduce_scatter_tensor_23 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_60, 'avg', 256, '0');  convert_element_type_default_60 = None
        wait_tensor_314 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_23);  reduce_scatter_tensor_23 = None
        view_1149 = torch.ops.aten.view.default(add_146, [16384, 4096])
        permute_433 = torch.ops.aten.permute.default(view_1149, [1, 0])
        mm_261 = torch.ops.aten.mm.default(permute_433, view_1009);  permute_433 = view_1009 = None
        permute_435 = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
        mm_262 = torch.ops.aten.mm.default(view_1149, permute_435);  view_1149 = permute_435 = None
        view_1150 = torch.ops.aten.view.default(mm_262, [2, 8192, 4096]);  mm_262 = None
        convert_element_type_1210 = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        reduce_scatter_tensor_24 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1210, 'avg', 256, '0');  convert_element_type_1210 = None
        wait_tensor_315 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_24);  reduce_scatter_tensor_24 = None
        view_1151 = torch.ops.aten.view.default(view_1150, [2, 8192, 32, 128]);  view_1150 = None
        permute_437 = torch.ops.aten.permute.default(view_1151, [0, 2, 1, 3]);  view_1151 = None
        convert_element_type_958 = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        all_gather_into_tensor_262 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_958, 256, '0');  convert_element_type_958 = None
        wait_tensor_262 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_262);  all_gather_into_tensor_262 = None
        convert_element_type_959 = torch.ops.prims.convert_element_type.default(add_115, torch.float32);  add_115 = None
        pow_59 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_959, 2)
        mean_58 = torch.ops.aten.mean.dim(pow_59, [2], True);  pow_59 = None
        add_116 = torch.ops.aten.add.Scalar(mean_58, 1e-05);  mean_58 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_959, rsqrt_58);  convert_element_type_959 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, wait_tensor_262)
        convert_element_type_960 = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None
        view_989 = torch.ops.aten.view.default(convert_element_type_960, [16384, 4096]);  convert_element_type_960 = None
        view_990 = torch.ops.aten.view.default(mm_203, [2, 8192, 4096]);  mm_203 = None
        convert_element_type_964 = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16);  primals_267 = None
        all_gather_into_tensor_264 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_964, 256, '0');  convert_element_type_964 = None
        wait_tensor_264 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_264);  all_gather_into_tensor_264 = None
        permute_320 = torch.ops.aten.permute.default(wait_tensor_264, [1, 0]);  wait_tensor_264 = None
        mm_204 = torch.ops.aten.mm.default(view_989, permute_320)
        view_993 = torch.ops.aten.view.default(mm_204, [2, 8192, 1024]);  mm_204 = None
        view_996 = torch.ops.aten.view.default(mm_205, [2, 8192, 1024]);  mm_205 = None
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
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_437, permute_322, permute_323, permute_324, getitem_261, getitem_262, getitem_267, getitem_268, None, None, None, 8192, 8192, 0.0, True);  permute_437 = permute_322 = permute_323 = permute_324 = getitem_261 = getitem_262 = getitem_267 = getitem_268 = None
        getitem_294 = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_295 = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_296 = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None
        permute_438 = torch.ops.aten.permute.default(getitem_296, [0, 2, 1, 3]);  getitem_296 = None
        permute_439 = torch.ops.aten.permute.default(getitem_295, [0, 2, 1, 3]);  getitem_295 = None
        permute_440 = torch.ops.aten.permute.default(getitem_294, [0, 2, 1, 3]);  getitem_294 = None
        view_1152 = torch.ops.aten.view.default(permute_438, [2, 8192, 8, 4, 128]);  permute_438 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_1152, [3], True);  view_1152 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(sum_17, 3);  sum_17 = None
        view_1153 = torch.ops.aten.view.default(permute_439, [2, 8192, 8, 4, 128]);  permute_439 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_1153, [3], True);  view_1153 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(sum_18, 3);  sum_18 = None
        convert_element_type_1211 = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        convert_element_type_1212 = torch.ops.prims.convert_element_type.default(permute_440, torch.float32);  permute_440 = None
        view_1154 = torch.ops.aten.view.default(convert_element_type_1211, [2, 8192, 8, 64, 2]);  convert_element_type_1211 = None
        view_as_complex_68 = torch.ops.aten.view_as_complex.default(view_1154);  view_1154 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_as_complex_68, _conj);  view_as_complex_68 = None
        view_1155 = torch.ops.aten.view.default(convert_element_type_1212, [2, 8192, 32, 64, 2]);  convert_element_type_1212 = None
        view_as_complex_69 = torch.ops.aten.view_as_complex.default(view_1155);  view_1155 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_as_complex_69, _conj);  view_as_complex_69 = None
        view_as_real_68 = torch.ops.aten.view_as_real.default(mul_316);  mul_316 = None
        view_1156 = torch.ops.aten.view.default(view_as_real_68, [2, 8192, 8, 128]);  view_as_real_68 = None
        convert_element_type_1213 = torch.ops.prims.convert_element_type.default(view_1156, torch.bfloat16);  view_1156 = None
        view_as_real_69 = torch.ops.aten.view_as_real.default(mul_317);  mul_317 = None
        view_1157 = torch.ops.aten.view.default(view_as_real_69, [2, 8192, 32, 128]);  view_as_real_69 = None
        convert_element_type_1214 = torch.ops.prims.convert_element_type.default(view_1157, torch.bfloat16);  view_1157 = None
        view_1158 = torch.ops.aten.view.default(squeeze_4, [2, 8192, 1024]);  squeeze_4 = None
        view_1159 = torch.ops.aten.view.default(convert_element_type_1213, [2, 8192, 1024]);  convert_element_type_1213 = None
        view_1160 = torch.ops.aten.view.default(convert_element_type_1214, [2, 8192, 4096]);  convert_element_type_1214 = None
        view_1161 = torch.ops.aten.view.default(view_1158, [16384, 1024]);  view_1158 = None
        permute_441 = torch.ops.aten.permute.default(view_1161, [1, 0])
        mm_263 = torch.ops.aten.mm.default(permute_441, view_989);  permute_441 = None
        convert_element_type_967 = torch.ops.prims.convert_element_type.default(primals_268, torch.bfloat16);  primals_268 = None
        all_gather_into_tensor_265 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_967, 256, '0');  convert_element_type_967 = None
        wait_tensor_265 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_265);  all_gather_into_tensor_265 = None
        permute_321 = torch.ops.aten.permute.default(wait_tensor_265, [1, 0]);  wait_tensor_265 = None
        permute_443 = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
        mm_264 = torch.ops.aten.mm.default(view_1161, permute_443);  view_1161 = permute_443 = None
        view_1162 = torch.ops.aten.view.default(mm_264, [2, 8192, 4096]);  mm_264 = None
        convert_element_type_1219 = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        reduce_scatter_tensor_25 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1219, 'avg', 256, '0');  convert_element_type_1219 = None
        wait_tensor_316 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_25);  reduce_scatter_tensor_25 = None
        view_1163 = torch.ops.aten.view.default(view_1159, [16384, 1024]);  view_1159 = None
        permute_445 = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_265 = torch.ops.aten.mm.default(permute_445, view_989);  permute_445 = None
        permute_447 = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
        mm_266 = torch.ops.aten.mm.default(view_1163, permute_447);  view_1163 = permute_447 = None
        view_1164 = torch.ops.aten.view.default(mm_266, [2, 8192, 4096]);  mm_266 = None
        add_147 = torch.ops.aten.add.Tensor(view_1162, view_1164);  view_1162 = view_1164 = None
        convert_element_type_1224 = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        reduce_scatter_tensor_26 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1224, 'avg', 256, '0');  convert_element_type_1224 = None
        wait_tensor_317 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_26);  reduce_scatter_tensor_26 = None
        view_1165 = torch.ops.aten.view.default(view_1160, [16384, 4096]);  view_1160 = None
        permute_449 = torch.ops.aten.permute.default(view_1165, [1, 0])
        mm_267 = torch.ops.aten.mm.default(permute_449, view_989);  permute_449 = view_989 = None
        convert_element_type_961 = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        all_gather_into_tensor_263 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_961, 256, '0');  convert_element_type_961 = None
        wait_tensor_263 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_263);  all_gather_into_tensor_263 = None
        permute_319 = torch.ops.aten.permute.default(wait_tensor_263, [1, 0]);  wait_tensor_263 = None
        permute_451 = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
        mm_268 = torch.ops.aten.mm.default(view_1165, permute_451);  view_1165 = permute_451 = None
        view_1166 = torch.ops.aten.view.default(mm_268, [2, 8192, 4096]);  mm_268 = None
        add_148 = torch.ops.aten.add.Tensor(add_147, view_1166);  add_147 = view_1166 = None
        convert_element_type_1229 = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        reduce_scatter_tensor_27 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1229, 'avg', 256, '0');  convert_element_type_1229 = None
        wait_tensor_318 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_27);  reduce_scatter_tensor_27 = None
        convert_element_type_1230 = torch.ops.prims.convert_element_type.default(add_148, torch.float32);  add_148 = None
        convert_element_type_1232 = torch.ops.prims.convert_element_type.default(wait_tensor_262, torch.float32);  wait_tensor_262 = None
        mul_318 = torch.ops.aten.mul.Tensor(convert_element_type_1230, convert_element_type_1232);  convert_element_type_1232 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_232, mul_318)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        div_6 = torch.ops.aten.div.Tensor(mul_232, 4096)
        mul_321 = torch.ops.aten.mul.Tensor(div_6, sum_19);  div_6 = sum_19 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_318, mul_321);  mul_318 = mul_321 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_58);  sub_9 = rsqrt_58 = None
        mul_323 = torch.ops.aten.mul.Tensor(convert_element_type_1230, mul_232);  convert_element_type_1230 = mul_232 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        convert_element_type_1233 = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, convert_element_type_1233);  add_146 = convert_element_type_1233 = None
        convert_element_type_default_59 = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        reduce_scatter_tensor_28 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_59, 'avg', 256, '0');  convert_element_type_default_59 = None
        wait_tensor_319 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_28);  reduce_scatter_tensor_28 = None
        view_1167 = torch.ops.aten.view.default(add_149, [16384, 4096])
        permute_453 = torch.ops.aten.permute.default(view_1167, [1, 0])
        permute_314 = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3])
        view_973 = torch.ops.aten.view.default(permute_314, [2, 8192, -1]);  permute_314 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        all_gather_into_tensor_257 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_941, 256, '0');  convert_element_type_941 = None
        wait_tensor_257 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_257);  all_gather_into_tensor_257 = None
        permute_315 = torch.ops.aten.permute.default(wait_tensor_257, [1, 0]);  wait_tensor_257 = None
        view_975 = torch.ops.aten.view.default(view_973, [16384, 4096]);  view_973 = None
        mm_199 = torch.ops.aten.mm.default(view_975, permute_315)
        view_976 = torch.ops.aten.view.default(mm_199, [2, 8192, 4096]);  mm_199 = None
        add_113 = torch.ops.aten.add.Tensor(add_111, view_976);  view_976 = None
        convert_element_type_944 = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16);  primals_261 = None
        all_gather_into_tensor_258 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_944, 256, '0');  convert_element_type_944 = None
        wait_tensor_258 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_258);  all_gather_into_tensor_258 = None
        convert_element_type_945 = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_945, 2)
        mean_57 = torch.ops.aten.mean.dim(pow_58, [2], True);  pow_58 = None
        add_114 = torch.ops.aten.add.Scalar(mean_57, 1e-05);  mean_57 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_228 = torch.ops.aten.mul.Tensor(convert_element_type_945, rsqrt_57);  convert_element_type_945 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, wait_tensor_258)
        convert_element_type_946 = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        view_979 = torch.ops.aten.view.default(convert_element_type_946, [16384, 4096]);  convert_element_type_946 = None
        view_980 = torch.ops.aten.view.default(mm_200, [2, 8192, 14336]);  mm_200 = None
        convert_element_type_950 = torch.ops.prims.convert_element_type.default(view_980, torch.float32);  view_980 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(convert_element_type_950)
        mul_230 = torch.ops.aten.mul.Tensor(convert_element_type_950, sigmoid_28);  sigmoid_28 = None
        convert_element_type_951 = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16);  mul_230 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(primals_263, torch.bfloat16);  primals_263 = None
        all_gather_into_tensor_260 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_952, 256, '0');  convert_element_type_952 = None
        wait_tensor_260 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_260);  all_gather_into_tensor_260 = None
        permute_317 = torch.ops.aten.permute.default(wait_tensor_260, [1, 0]);  wait_tensor_260 = None
        mm_201 = torch.ops.aten.mm.default(view_979, permute_317)
        view_983 = torch.ops.aten.view.default(mm_201, [2, 8192, 14336]);  mm_201 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_951, view_983)
        view_985 = torch.ops.aten.view.default(mul_231, [16384, 14336]);  mul_231 = None
        mm_269 = torch.ops.aten.mm.default(permute_453, view_985);  permute_453 = view_985 = None
        convert_element_type_955 = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16);  primals_264 = None
        all_gather_into_tensor_261 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_955, 256, '0');  convert_element_type_955 = None
        wait_tensor_261 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_261);  all_gather_into_tensor_261 = None
        permute_318 = torch.ops.aten.permute.default(wait_tensor_261, [1, 0]);  wait_tensor_261 = None
        permute_455 = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
        mm_270 = torch.ops.aten.mm.default(view_1167, permute_455);  view_1167 = permute_455 = None
        view_1168 = torch.ops.aten.view.default(mm_270, [2, 8192, 14336]);  mm_270 = None
        convert_element_type_1240 = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        reduce_scatter_tensor_29 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1240, 'avg', 256, '0');  convert_element_type_1240 = None
        wait_tensor_320 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_29);  reduce_scatter_tensor_29 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_1168, convert_element_type_951);  convert_element_type_951 = None
        mul_325 = torch.ops.aten.mul.Tensor(view_1168, view_983);  view_1168 = view_983 = None
        view_1169 = torch.ops.aten.view.default(mul_324, [16384, 14336]);  mul_324 = None
        permute_457 = torch.ops.aten.permute.default(view_1169, [1, 0])
        mm_271 = torch.ops.aten.mm.default(permute_457, view_979);  permute_457 = None
        permute_459 = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
        mm_272 = torch.ops.aten.mm.default(view_1169, permute_459);  view_1169 = permute_459 = None
        view_1170 = torch.ops.aten.view.default(mm_272, [2, 8192, 4096]);  mm_272 = None
        convert_element_type_1245 = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        reduce_scatter_tensor_30 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1245, 'avg', 256, '0');  convert_element_type_1245 = None
        wait_tensor_321 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_30);  reduce_scatter_tensor_30 = None
        convert_element_type_1246 = torch.ops.prims.convert_element_type.default(mul_325, torch.float32);  mul_325 = None
        neg_3 = torch.ops.aten.neg.default(convert_element_type_950)
        exp_3 = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_150 = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(add_150);  add_150 = None
        mul_326 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_327 = torch.ops.aten.mul.Tensor(convert_element_type_1246, mul_326);  convert_element_type_1246 = None
        sub_10 = torch.ops.aten.sub.Tensor(1, mul_326);  mul_326 = None
        mul_328 = torch.ops.aten.mul.Tensor(convert_element_type_950, sub_10);  convert_element_type_950 = sub_10 = None
        add_151 = torch.ops.aten.add.Tensor(mul_328, 1);  mul_328 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_327, add_151);  mul_327 = add_151 = None
        convert_element_type_1248 = torch.ops.prims.convert_element_type.default(mul_329, torch.bfloat16);  mul_329 = None
        view_1171 = torch.ops.aten.view.default(convert_element_type_1248, [16384, 14336]);  convert_element_type_1248 = None
        permute_461 = torch.ops.aten.permute.default(view_1171, [1, 0])
        mm_273 = torch.ops.aten.mm.default(permute_461, view_979);  permute_461 = view_979 = None
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16);  primals_262 = None
        all_gather_into_tensor_259 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_947, 256, '0');  convert_element_type_947 = None
        wait_tensor_259 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_259);  all_gather_into_tensor_259 = None
        permute_316 = torch.ops.aten.permute.default(wait_tensor_259, [1, 0]);  wait_tensor_259 = None
        permute_463 = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
        mm_274 = torch.ops.aten.mm.default(view_1171, permute_463);  view_1171 = permute_463 = None
        view_1172 = torch.ops.aten.view.default(mm_274, [2, 8192, 4096]);  mm_274 = None
        add_152 = torch.ops.aten.add.Tensor(view_1170, view_1172);  view_1170 = view_1172 = None
        convert_element_type_1253 = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        reduce_scatter_tensor_31 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1253, 'avg', 256, '0');  convert_element_type_1253 = None
        wait_tensor_322 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_31);  reduce_scatter_tensor_31 = None
        convert_element_type_1254 = torch.ops.prims.convert_element_type.default(add_152, torch.float32);  add_152 = None
        convert_element_type_1256 = torch.ops.prims.convert_element_type.default(wait_tensor_258, torch.float32);  wait_tensor_258 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_1254, convert_element_type_1256);  convert_element_type_1256 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_228, mul_330)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        div_7 = torch.ops.aten.div.Tensor(mul_228, 4096)
        mul_333 = torch.ops.aten.mul.Tensor(div_7, sum_21);  div_7 = sum_21 = None
        sub_11 = torch.ops.aten.sub.Tensor(mul_330, mul_333);  mul_330 = mul_333 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_57);  sub_11 = rsqrt_57 = None
        mul_335 = torch.ops.aten.mul.Tensor(convert_element_type_1254, mul_228);  convert_element_type_1254 = mul_228 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        convert_element_type_1257 = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, convert_element_type_1257);  add_149 = convert_element_type_1257 = None
        convert_element_type_default_58 = torch.ops.prims.convert_element_type.default(sum_22, torch.float32);  sum_22 = None
        reduce_scatter_tensor_32 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_58, 'avg', 256, '0');  convert_element_type_default_58 = None
        wait_tensor_323 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_32);  reduce_scatter_tensor_32 = None
        view_1173 = torch.ops.aten.view.default(add_153, [16384, 4096])
        permute_465 = torch.ops.aten.permute.default(view_1173, [1, 0])
        mm_275 = torch.ops.aten.mm.default(permute_465, view_975);  permute_465 = view_975 = None
        permute_467 = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
        mm_276 = torch.ops.aten.mm.default(view_1173, permute_467);  view_1173 = permute_467 = None
        view_1174 = torch.ops.aten.view.default(mm_276, [2, 8192, 4096]);  mm_276 = None
        convert_element_type_1264 = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None
        reduce_scatter_tensor_33 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1264, 'avg', 256, '0');  convert_element_type_1264 = None
        wait_tensor_324 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_33);  reduce_scatter_tensor_33 = None
        view_1175 = torch.ops.aten.view.default(view_1174, [2, 8192, 32, 128]);  view_1174 = None
        permute_469 = torch.ops.aten.permute.default(view_1175, [0, 2, 1, 3]);  view_1175 = None
        convert_element_type_925 = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16);  primals_256 = None
        all_gather_into_tensor_253 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_925, 256, '0');  convert_element_type_925 = None
        wait_tensor_253 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_253);  all_gather_into_tensor_253 = None
        convert_element_type_926 = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_926, 2)
        mean_56 = torch.ops.aten.mean.dim(pow_57, [2], True);  pow_57 = None
        add_112 = torch.ops.aten.add.Scalar(mean_56, 1e-05);  mean_56 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_224 = torch.ops.aten.mul.Tensor(convert_element_type_926, rsqrt_56);  convert_element_type_926 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, wait_tensor_253)
        convert_element_type_927 = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        view_955 = torch.ops.aten.view.default(convert_element_type_927, [16384, 4096]);  convert_element_type_927 = None
        view_956 = torch.ops.aten.view.default(mm_196, [2, 8192, 4096]);  mm_196 = None
        convert_element_type_931 = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        all_gather_into_tensor_255 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_931, 256, '0');  convert_element_type_931 = None
        wait_tensor_255 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_255);  all_gather_into_tensor_255 = None
        permute_309 = torch.ops.aten.permute.default(wait_tensor_255, [1, 0]);  wait_tensor_255 = None
        mm_197 = torch.ops.aten.mm.default(view_955, permute_309)
        view_959 = torch.ops.aten.view.default(mm_197, [2, 8192, 1024]);  mm_197 = None
        view_962 = torch.ops.aten.view.default(mm_198, [2, 8192, 1024]);  mm_198 = None
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
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_469, permute_311, permute_312, permute_313, getitem_252, getitem_253, getitem_258, getitem_259, None, None, None, 8192, 8192, 0.0, True);  permute_469 = permute_311 = permute_312 = permute_313 = getitem_252 = getitem_253 = getitem_258 = getitem_259 = None
        getitem_297 = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_298 = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_299 = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None
        permute_470 = torch.ops.aten.permute.default(getitem_299, [0, 2, 1, 3]);  getitem_299 = None
        permute_471 = torch.ops.aten.permute.default(getitem_298, [0, 2, 1, 3]);  getitem_298 = None
        permute_472 = torch.ops.aten.permute.default(getitem_297, [0, 2, 1, 3]);  getitem_297 = None
        view_1176 = torch.ops.aten.view.default(permute_470, [2, 8192, 8, 4, 128]);  permute_470 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_1176, [3], True);  view_1176 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(sum_23, 3);  sum_23 = None
        view_1177 = torch.ops.aten.view.default(permute_471, [2, 8192, 8, 4, 128]);  permute_471 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_1177, [3], True);  view_1177 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(sum_24, 3);  sum_24 = None
        convert_element_type_1265 = torch.ops.prims.convert_element_type.default(squeeze_7, torch.float32);  squeeze_7 = None
        convert_element_type_1266 = torch.ops.prims.convert_element_type.default(permute_472, torch.float32);  permute_472 = None
        view_1178 = torch.ops.aten.view.default(convert_element_type_1265, [2, 8192, 8, 64, 2]);  convert_element_type_1265 = None
        view_as_complex_70 = torch.ops.aten.view_as_complex.default(view_1178);  view_1178 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_as_complex_70, _conj);  view_as_complex_70 = None
        view_1179 = torch.ops.aten.view.default(convert_element_type_1266, [2, 8192, 32, 64, 2]);  convert_element_type_1266 = None
        view_as_complex_71 = torch.ops.aten.view_as_complex.default(view_1179);  view_1179 = None
        mul_337 = torch.ops.aten.mul.Tensor(view_as_complex_71, _conj);  view_as_complex_71 = None
        view_as_real_70 = torch.ops.aten.view_as_real.default(mul_336);  mul_336 = None
        view_1180 = torch.ops.aten.view.default(view_as_real_70, [2, 8192, 8, 128]);  view_as_real_70 = None
        convert_element_type_1267 = torch.ops.prims.convert_element_type.default(view_1180, torch.bfloat16);  view_1180 = None
        view_as_real_71 = torch.ops.aten.view_as_real.default(mul_337);  mul_337 = None
        view_1181 = torch.ops.aten.view.default(view_as_real_71, [2, 8192, 32, 128]);  view_as_real_71 = None
        convert_element_type_1268 = torch.ops.prims.convert_element_type.default(view_1181, torch.bfloat16);  view_1181 = None
        view_1182 = torch.ops.aten.view.default(squeeze_6, [2, 8192, 1024]);  squeeze_6 = None
        view_1183 = torch.ops.aten.view.default(convert_element_type_1267, [2, 8192, 1024]);  convert_element_type_1267 = None
        view_1184 = torch.ops.aten.view.default(convert_element_type_1268, [2, 8192, 4096]);  convert_element_type_1268 = None
        view_1185 = torch.ops.aten.view.default(view_1182, [16384, 1024]);  view_1182 = None
        permute_473 = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_277 = torch.ops.aten.mm.default(permute_473, view_955);  permute_473 = None
        convert_element_type_934 = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        all_gather_into_tensor_256 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_934, 256, '0');  convert_element_type_934 = None
        wait_tensor_256 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_256);  all_gather_into_tensor_256 = None
        permute_310 = torch.ops.aten.permute.default(wait_tensor_256, [1, 0]);  wait_tensor_256 = None
        permute_475 = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
        mm_278 = torch.ops.aten.mm.default(view_1185, permute_475);  view_1185 = permute_475 = None
        view_1186 = torch.ops.aten.view.default(mm_278, [2, 8192, 4096]);  mm_278 = None
        convert_element_type_1273 = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        reduce_scatter_tensor_34 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1273, 'avg', 256, '0');  convert_element_type_1273 = None
        wait_tensor_325 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_34);  reduce_scatter_tensor_34 = None
        view_1187 = torch.ops.aten.view.default(view_1183, [16384, 1024]);  view_1183 = None
        permute_477 = torch.ops.aten.permute.default(view_1187, [1, 0])
        mm_279 = torch.ops.aten.mm.default(permute_477, view_955);  permute_477 = None
        permute_479 = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
        mm_280 = torch.ops.aten.mm.default(view_1187, permute_479);  view_1187 = permute_479 = None
        view_1188 = torch.ops.aten.view.default(mm_280, [2, 8192, 4096]);  mm_280 = None
        add_154 = torch.ops.aten.add.Tensor(view_1186, view_1188);  view_1186 = view_1188 = None
        convert_element_type_1278 = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        reduce_scatter_tensor_35 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1278, 'avg', 256, '0');  convert_element_type_1278 = None
        wait_tensor_326 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_35);  reduce_scatter_tensor_35 = None
        view_1189 = torch.ops.aten.view.default(view_1184, [16384, 4096]);  view_1184 = None
        permute_481 = torch.ops.aten.permute.default(view_1189, [1, 0])
        mm_281 = torch.ops.aten.mm.default(permute_481, view_955);  permute_481 = view_955 = None
        convert_element_type_928 = torch.ops.prims.convert_element_type.default(primals_257, torch.bfloat16);  primals_257 = None
        all_gather_into_tensor_254 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_928, 256, '0');  convert_element_type_928 = None
        wait_tensor_254 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_254);  all_gather_into_tensor_254 = None
        permute_308 = torch.ops.aten.permute.default(wait_tensor_254, [1, 0]);  wait_tensor_254 = None
        permute_483 = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        mm_282 = torch.ops.aten.mm.default(view_1189, permute_483);  view_1189 = permute_483 = None
        view_1190 = torch.ops.aten.view.default(mm_282, [2, 8192, 4096]);  mm_282 = None
        add_155 = torch.ops.aten.add.Tensor(add_154, view_1190);  add_154 = view_1190 = None
        convert_element_type_1283 = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        reduce_scatter_tensor_36 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1283, 'avg', 256, '0');  convert_element_type_1283 = None
        wait_tensor_327 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_36);  reduce_scatter_tensor_36 = None
        convert_element_type_1284 = torch.ops.prims.convert_element_type.default(add_155, torch.float32);  add_155 = None
        convert_element_type_1286 = torch.ops.prims.convert_element_type.default(wait_tensor_253, torch.float32);  wait_tensor_253 = None
        mul_338 = torch.ops.aten.mul.Tensor(convert_element_type_1284, convert_element_type_1286);  convert_element_type_1286 = None
        mul_340 = torch.ops.aten.mul.Tensor(mul_224, mul_338)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
        div_8 = torch.ops.aten.div.Tensor(mul_224, 4096)
        mul_341 = torch.ops.aten.mul.Tensor(div_8, sum_25);  div_8 = sum_25 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_338, mul_341);  mul_338 = mul_341 = None
        mul_342 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_56);  sub_12 = rsqrt_56 = None
        mul_343 = torch.ops.aten.mul.Tensor(convert_element_type_1284, mul_224);  convert_element_type_1284 = mul_224 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
        convert_element_type_1287 = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        add_156 = torch.ops.aten.add.Tensor(add_153, convert_element_type_1287);  add_153 = convert_element_type_1287 = None
        convert_element_type_default_57 = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        reduce_scatter_tensor_37 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_57, 'avg', 256, '0');  convert_element_type_default_57 = None
        wait_tensor_328 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_37);  reduce_scatter_tensor_37 = None
        view_1191 = torch.ops.aten.view.default(add_156, [16384, 4096])
        permute_485 = torch.ops.aten.permute.default(view_1191, [1, 0])
        permute_303 = torch.ops.aten.permute.default(getitem_243, [0, 2, 1, 3])
        view_939 = torch.ops.aten.view.default(permute_303, [2, 8192, -1]);  permute_303 = None
        convert_element_type_908 = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        all_gather_into_tensor_248 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_908, 256, '0');  convert_element_type_908 = None
        wait_tensor_248 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_248);  all_gather_into_tensor_248 = None
        permute_304 = torch.ops.aten.permute.default(wait_tensor_248, [1, 0]);  wait_tensor_248 = None
        view_941 = torch.ops.aten.view.default(view_939, [16384, 4096]);  view_939 = None
        mm_192 = torch.ops.aten.mm.default(view_941, permute_304)
        view_942 = torch.ops.aten.view.default(mm_192, [2, 8192, 4096]);  mm_192 = None
        add_109 = torch.ops.aten.add.Tensor(add_107, view_942);  view_942 = None
        convert_element_type_911 = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        all_gather_into_tensor_249 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_911, 256, '0');  convert_element_type_911 = None
        wait_tensor_249 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_249);  all_gather_into_tensor_249 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(add_109, torch.float32);  add_109 = None
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_912, 2)
        mean_55 = torch.ops.aten.mean.dim(pow_56, [2], True);  pow_56 = None
        add_110 = torch.ops.aten.add.Scalar(mean_55, 1e-05);  mean_55 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_912, rsqrt_55);  convert_element_type_912 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, wait_tensor_249)
        convert_element_type_913 = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        view_945 = torch.ops.aten.view.default(convert_element_type_913, [16384, 4096]);  convert_element_type_913 = None
        view_946 = torch.ops.aten.view.default(mm_193, [2, 8192, 14336]);  mm_193 = None
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(view_946, torch.float32);  view_946 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(convert_element_type_917)
        mul_222 = torch.ops.aten.mul.Tensor(convert_element_type_917, sigmoid_27);  sigmoid_27 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None
        convert_element_type_919 = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        all_gather_into_tensor_251 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_919, 256, '0');  convert_element_type_919 = None
        wait_tensor_251 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_251);  all_gather_into_tensor_251 = None
        permute_306 = torch.ops.aten.permute.default(wait_tensor_251, [1, 0]);  wait_tensor_251 = None
        mm_194 = torch.ops.aten.mm.default(view_945, permute_306)
        view_949 = torch.ops.aten.view.default(mm_194, [2, 8192, 14336]);  mm_194 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_918, view_949)
        view_951 = torch.ops.aten.view.default(mul_223, [16384, 14336]);  mul_223 = None
        mm_283 = torch.ops.aten.mm.default(permute_485, view_951);  permute_485 = view_951 = None
        convert_element_type_922 = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16);  primals_255 = None
        all_gather_into_tensor_252 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_922, 256, '0');  convert_element_type_922 = None
        wait_tensor_252 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_252);  all_gather_into_tensor_252 = None
        permute_307 = torch.ops.aten.permute.default(wait_tensor_252, [1, 0]);  wait_tensor_252 = None
        permute_487 = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        mm_284 = torch.ops.aten.mm.default(view_1191, permute_487);  view_1191 = permute_487 = None
        view_1192 = torch.ops.aten.view.default(mm_284, [2, 8192, 14336]);  mm_284 = None
        convert_element_type_1294 = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        reduce_scatter_tensor_38 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1294, 'avg', 256, '0');  convert_element_type_1294 = None
        wait_tensor_329 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_38);  reduce_scatter_tensor_38 = None
        mul_344 = torch.ops.aten.mul.Tensor(view_1192, convert_element_type_918);  convert_element_type_918 = None
        mul_345 = torch.ops.aten.mul.Tensor(view_1192, view_949);  view_1192 = view_949 = None
        view_1193 = torch.ops.aten.view.default(mul_344, [16384, 14336]);  mul_344 = None
        permute_489 = torch.ops.aten.permute.default(view_1193, [1, 0])
        mm_285 = torch.ops.aten.mm.default(permute_489, view_945);  permute_489 = None
        permute_491 = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
        mm_286 = torch.ops.aten.mm.default(view_1193, permute_491);  view_1193 = permute_491 = None
        view_1194 = torch.ops.aten.view.default(mm_286, [2, 8192, 4096]);  mm_286 = None
        convert_element_type_1299 = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        reduce_scatter_tensor_39 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1299, 'avg', 256, '0');  convert_element_type_1299 = None
        wait_tensor_330 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_39);  reduce_scatter_tensor_39 = None
        convert_element_type_1300 = torch.ops.prims.convert_element_type.default(mul_345, torch.float32);  mul_345 = None
        neg_4 = torch.ops.aten.neg.default(convert_element_type_917)
        exp_4 = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_157 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(add_157);  add_157 = None
        mul_346 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_347 = torch.ops.aten.mul.Tensor(convert_element_type_1300, mul_346);  convert_element_type_1300 = None
        sub_13 = torch.ops.aten.sub.Tensor(1, mul_346);  mul_346 = None
        mul_348 = torch.ops.aten.mul.Tensor(convert_element_type_917, sub_13);  convert_element_type_917 = sub_13 = None
        add_158 = torch.ops.aten.add.Tensor(mul_348, 1);  mul_348 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_347, add_158);  mul_347 = add_158 = None
        convert_element_type_1302 = torch.ops.prims.convert_element_type.default(mul_349, torch.bfloat16);  mul_349 = None
        view_1195 = torch.ops.aten.view.default(convert_element_type_1302, [16384, 14336]);  convert_element_type_1302 = None
        permute_493 = torch.ops.aten.permute.default(view_1195, [1, 0])
        mm_287 = torch.ops.aten.mm.default(permute_493, view_945);  permute_493 = view_945 = None
        convert_element_type_914 = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        all_gather_into_tensor_250 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_914, 256, '0');  convert_element_type_914 = None
        wait_tensor_250 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_250);  all_gather_into_tensor_250 = None
        permute_305 = torch.ops.aten.permute.default(wait_tensor_250, [1, 0]);  wait_tensor_250 = None
        permute_495 = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        mm_288 = torch.ops.aten.mm.default(view_1195, permute_495);  view_1195 = permute_495 = None
        view_1196 = torch.ops.aten.view.default(mm_288, [2, 8192, 4096]);  mm_288 = None
        add_159 = torch.ops.aten.add.Tensor(view_1194, view_1196);  view_1194 = view_1196 = None
        convert_element_type_1307 = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None
        reduce_scatter_tensor_40 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1307, 'avg', 256, '0');  convert_element_type_1307 = None
        wait_tensor_331 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_40);  reduce_scatter_tensor_40 = None
        convert_element_type_1308 = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        convert_element_type_1310 = torch.ops.prims.convert_element_type.default(wait_tensor_249, torch.float32);  wait_tensor_249 = None
        mul_350 = torch.ops.aten.mul.Tensor(convert_element_type_1308, convert_element_type_1310);  convert_element_type_1310 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_220, mul_350)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        div_9 = torch.ops.aten.div.Tensor(mul_220, 4096)
        mul_353 = torch.ops.aten.mul.Tensor(div_9, sum_27);  div_9 = sum_27 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_350, mul_353);  mul_350 = mul_353 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_55);  sub_14 = rsqrt_55 = None
        mul_355 = torch.ops.aten.mul.Tensor(convert_element_type_1308, mul_220);  convert_element_type_1308 = mul_220 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        convert_element_type_1311 = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        add_160 = torch.ops.aten.add.Tensor(add_156, convert_element_type_1311);  add_156 = convert_element_type_1311 = None
        convert_element_type_default_56 = torch.ops.prims.convert_element_type.default(sum_28, torch.float32);  sum_28 = None
        reduce_scatter_tensor_41 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_56, 'avg', 256, '0');  convert_element_type_default_56 = None
        wait_tensor_332 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_41);  reduce_scatter_tensor_41 = None
        view_1197 = torch.ops.aten.view.default(add_160, [16384, 4096])
        permute_497 = torch.ops.aten.permute.default(view_1197, [1, 0])
        mm_289 = torch.ops.aten.mm.default(permute_497, view_941);  permute_497 = view_941 = None
        permute_499 = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
        mm_290 = torch.ops.aten.mm.default(view_1197, permute_499);  view_1197 = permute_499 = None
        view_1198 = torch.ops.aten.view.default(mm_290, [2, 8192, 4096]);  mm_290 = None
        convert_element_type_1318 = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None
        reduce_scatter_tensor_42 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1318, 'avg', 256, '0');  convert_element_type_1318 = None
        wait_tensor_333 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_42);  reduce_scatter_tensor_42 = None
        view_1199 = torch.ops.aten.view.default(view_1198, [2, 8192, 32, 128]);  view_1198 = None
        permute_501 = torch.ops.aten.permute.default(view_1199, [0, 2, 1, 3]);  view_1199 = None
        convert_element_type_892 = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16);  primals_247 = None
        all_gather_into_tensor_244 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_892, 256, '0');  convert_element_type_892 = None
        wait_tensor_244 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_244);  all_gather_into_tensor_244 = None
        convert_element_type_893 = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_893, 2)
        mean_54 = torch.ops.aten.mean.dim(pow_55, [2], True);  pow_55 = None
        add_108 = torch.ops.aten.add.Scalar(mean_54, 1e-05);  mean_54 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_216 = torch.ops.aten.mul.Tensor(convert_element_type_893, rsqrt_54);  convert_element_type_893 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, wait_tensor_244)
        convert_element_type_894 = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16);  mul_217 = None
        view_921 = torch.ops.aten.view.default(convert_element_type_894, [16384, 4096]);  convert_element_type_894 = None
        view_922 = torch.ops.aten.view.default(mm_189, [2, 8192, 4096]);  mm_189 = None
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16);  primals_249 = None
        all_gather_into_tensor_246 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_898, 256, '0');  convert_element_type_898 = None
        wait_tensor_246 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_246);  all_gather_into_tensor_246 = None
        permute_298 = torch.ops.aten.permute.default(wait_tensor_246, [1, 0]);  wait_tensor_246 = None
        mm_190 = torch.ops.aten.mm.default(view_921, permute_298)
        view_925 = torch.ops.aten.view.default(mm_190, [2, 8192, 1024]);  mm_190 = None
        view_928 = torch.ops.aten.view.default(mm_191, [2, 8192, 1024]);  mm_191 = None
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
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_501, permute_300, permute_301, permute_302, getitem_243, getitem_244, getitem_249, getitem_250, None, None, None, 8192, 8192, 0.0, True);  permute_501 = permute_300 = permute_301 = permute_302 = getitem_243 = getitem_244 = getitem_249 = getitem_250 = None
        getitem_300 = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_301 = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_302 = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None
        permute_502 = torch.ops.aten.permute.default(getitem_302, [0, 2, 1, 3]);  getitem_302 = None
        permute_503 = torch.ops.aten.permute.default(getitem_301, [0, 2, 1, 3]);  getitem_301 = None
        permute_504 = torch.ops.aten.permute.default(getitem_300, [0, 2, 1, 3]);  getitem_300 = None
        view_1200 = torch.ops.aten.view.default(permute_502, [2, 8192, 8, 4, 128]);  permute_502 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(view_1200, [3], True);  view_1200 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(sum_29, 3);  sum_29 = None
        view_1201 = torch.ops.aten.view.default(permute_503, [2, 8192, 8, 4, 128]);  permute_503 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(view_1201, [3], True);  view_1201 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(sum_30, 3);  sum_30 = None
        convert_element_type_1319 = torch.ops.prims.convert_element_type.default(squeeze_9, torch.float32);  squeeze_9 = None
        convert_element_type_1320 = torch.ops.prims.convert_element_type.default(permute_504, torch.float32);  permute_504 = None
        view_1202 = torch.ops.aten.view.default(convert_element_type_1319, [2, 8192, 8, 64, 2]);  convert_element_type_1319 = None
        view_as_complex_72 = torch.ops.aten.view_as_complex.default(view_1202);  view_1202 = None
        mul_356 = torch.ops.aten.mul.Tensor(view_as_complex_72, _conj);  view_as_complex_72 = None
        view_1203 = torch.ops.aten.view.default(convert_element_type_1320, [2, 8192, 32, 64, 2]);  convert_element_type_1320 = None
        view_as_complex_73 = torch.ops.aten.view_as_complex.default(view_1203);  view_1203 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_as_complex_73, _conj);  view_as_complex_73 = None
        view_as_real_72 = torch.ops.aten.view_as_real.default(mul_356);  mul_356 = None
        view_1204 = torch.ops.aten.view.default(view_as_real_72, [2, 8192, 8, 128]);  view_as_real_72 = None
        convert_element_type_1321 = torch.ops.prims.convert_element_type.default(view_1204, torch.bfloat16);  view_1204 = None
        view_as_real_73 = torch.ops.aten.view_as_real.default(mul_357);  mul_357 = None
        view_1205 = torch.ops.aten.view.default(view_as_real_73, [2, 8192, 32, 128]);  view_as_real_73 = None
        convert_element_type_1322 = torch.ops.prims.convert_element_type.default(view_1205, torch.bfloat16);  view_1205 = None
        view_1206 = torch.ops.aten.view.default(squeeze_8, [2, 8192, 1024]);  squeeze_8 = None
        view_1207 = torch.ops.aten.view.default(convert_element_type_1321, [2, 8192, 1024]);  convert_element_type_1321 = None
        view_1208 = torch.ops.aten.view.default(convert_element_type_1322, [2, 8192, 4096]);  convert_element_type_1322 = None
        view_1209 = torch.ops.aten.view.default(view_1206, [16384, 1024]);  view_1206 = None
        permute_505 = torch.ops.aten.permute.default(view_1209, [1, 0])
        mm_291 = torch.ops.aten.mm.default(permute_505, view_921);  permute_505 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        all_gather_into_tensor_247 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_901, 256, '0');  convert_element_type_901 = None
        wait_tensor_247 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_247);  all_gather_into_tensor_247 = None
        permute_299 = torch.ops.aten.permute.default(wait_tensor_247, [1, 0]);  wait_tensor_247 = None
        permute_507 = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        mm_292 = torch.ops.aten.mm.default(view_1209, permute_507);  view_1209 = permute_507 = None
        view_1210 = torch.ops.aten.view.default(mm_292, [2, 8192, 4096]);  mm_292 = None
        convert_element_type_1327 = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        reduce_scatter_tensor_43 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1327, 'avg', 256, '0');  convert_element_type_1327 = None
        wait_tensor_334 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_43);  reduce_scatter_tensor_43 = None
        view_1211 = torch.ops.aten.view.default(view_1207, [16384, 1024]);  view_1207 = None
        permute_509 = torch.ops.aten.permute.default(view_1211, [1, 0])
        mm_293 = torch.ops.aten.mm.default(permute_509, view_921);  permute_509 = None
        permute_511 = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        mm_294 = torch.ops.aten.mm.default(view_1211, permute_511);  view_1211 = permute_511 = None
        view_1212 = torch.ops.aten.view.default(mm_294, [2, 8192, 4096]);  mm_294 = None
        add_161 = torch.ops.aten.add.Tensor(view_1210, view_1212);  view_1210 = view_1212 = None
        convert_element_type_1332 = torch.ops.prims.convert_element_type.default(mm_293, torch.float32);  mm_293 = None
        reduce_scatter_tensor_44 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1332, 'avg', 256, '0');  convert_element_type_1332 = None
        wait_tensor_335 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_44);  reduce_scatter_tensor_44 = None
        view_1213 = torch.ops.aten.view.default(view_1208, [16384, 4096]);  view_1208 = None
        permute_513 = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_295 = torch.ops.aten.mm.default(permute_513, view_921);  permute_513 = view_921 = None
        convert_element_type_895 = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        all_gather_into_tensor_245 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_895, 256, '0');  convert_element_type_895 = None
        wait_tensor_245 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_245);  all_gather_into_tensor_245 = None
        permute_297 = torch.ops.aten.permute.default(wait_tensor_245, [1, 0]);  wait_tensor_245 = None
        permute_515 = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        mm_296 = torch.ops.aten.mm.default(view_1213, permute_515);  view_1213 = permute_515 = None
        view_1214 = torch.ops.aten.view.default(mm_296, [2, 8192, 4096]);  mm_296 = None
        add_162 = torch.ops.aten.add.Tensor(add_161, view_1214);  add_161 = view_1214 = None
        convert_element_type_1337 = torch.ops.prims.convert_element_type.default(mm_295, torch.float32);  mm_295 = None
        reduce_scatter_tensor_45 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1337, 'avg', 256, '0');  convert_element_type_1337 = None
        wait_tensor_336 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_45);  reduce_scatter_tensor_45 = None
        convert_element_type_1338 = torch.ops.prims.convert_element_type.default(add_162, torch.float32);  add_162 = None
        convert_element_type_1340 = torch.ops.prims.convert_element_type.default(wait_tensor_244, torch.float32);  wait_tensor_244 = None
        mul_358 = torch.ops.aten.mul.Tensor(convert_element_type_1338, convert_element_type_1340);  convert_element_type_1340 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_216, mul_358)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
        div_10 = torch.ops.aten.div.Tensor(mul_216, 4096)
        mul_361 = torch.ops.aten.mul.Tensor(div_10, sum_31);  div_10 = sum_31 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_358, mul_361);  mul_358 = mul_361 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_54);  sub_15 = rsqrt_54 = None
        mul_363 = torch.ops.aten.mul.Tensor(convert_element_type_1338, mul_216);  convert_element_type_1338 = mul_216 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
        convert_element_type_1341 = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        add_163 = torch.ops.aten.add.Tensor(add_160, convert_element_type_1341);  add_160 = convert_element_type_1341 = None
        convert_element_type_default_55 = torch.ops.prims.convert_element_type.default(sum_32, torch.float32);  sum_32 = None
        reduce_scatter_tensor_46 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_55, 'avg', 256, '0');  convert_element_type_default_55 = None
        wait_tensor_337 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_46);  reduce_scatter_tensor_46 = None
        view_1215 = torch.ops.aten.view.default(add_163, [16384, 4096])
        permute_517 = torch.ops.aten.permute.default(view_1215, [1, 0])
        permute_292 = torch.ops.aten.permute.default(getitem_234, [0, 2, 1, 3])
        view_905 = torch.ops.aten.view.default(permute_292, [2, 8192, -1]);  permute_292 = None
        convert_element_type_875 = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        all_gather_into_tensor_239 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_875, 256, '0');  convert_element_type_875 = None
        wait_tensor_239 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_239);  all_gather_into_tensor_239 = None
        permute_293 = torch.ops.aten.permute.default(wait_tensor_239, [1, 0]);  wait_tensor_239 = None
        view_907 = torch.ops.aten.view.default(view_905, [16384, 4096]);  view_905 = None
        mm_185 = torch.ops.aten.mm.default(view_907, permute_293)
        view_908 = torch.ops.aten.view.default(mm_185, [2, 8192, 4096]);  mm_185 = None
        add_105 = torch.ops.aten.add.Tensor(add_103, view_908);  view_908 = None
        convert_element_type_878 = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        all_gather_into_tensor_240 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_878, 256, '0');  convert_element_type_878 = None
        wait_tensor_240 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_240);  all_gather_into_tensor_240 = None
        convert_element_type_879 = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_879, 2)
        mean_53 = torch.ops.aten.mean.dim(pow_54, [2], True);  pow_54 = None
        add_106 = torch.ops.aten.add.Scalar(mean_53, 1e-05);  mean_53 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_212 = torch.ops.aten.mul.Tensor(convert_element_type_879, rsqrt_53);  convert_element_type_879 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, wait_tensor_240)
        convert_element_type_880 = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        view_911 = torch.ops.aten.view.default(convert_element_type_880, [16384, 4096]);  convert_element_type_880 = None
        view_912 = torch.ops.aten.view.default(mm_186, [2, 8192, 14336]);  mm_186 = None
        convert_element_type_884 = torch.ops.prims.convert_element_type.default(view_912, torch.float32);  view_912 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(convert_element_type_884)
        mul_214 = torch.ops.aten.mul.Tensor(convert_element_type_884, sigmoid_26);  sigmoid_26 = None
        convert_element_type_885 = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        convert_element_type_886 = torch.ops.prims.convert_element_type.default(primals_245, torch.bfloat16);  primals_245 = None
        all_gather_into_tensor_242 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_886, 256, '0');  convert_element_type_886 = None
        wait_tensor_242 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_242);  all_gather_into_tensor_242 = None
        permute_295 = torch.ops.aten.permute.default(wait_tensor_242, [1, 0]);  wait_tensor_242 = None
        mm_187 = torch.ops.aten.mm.default(view_911, permute_295)
        view_915 = torch.ops.aten.view.default(mm_187, [2, 8192, 14336]);  mm_187 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_885, view_915)
        view_917 = torch.ops.aten.view.default(mul_215, [16384, 14336]);  mul_215 = None
        mm_297 = torch.ops.aten.mm.default(permute_517, view_917);  permute_517 = view_917 = None
        convert_element_type_889 = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16);  primals_246 = None
        all_gather_into_tensor_243 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_889, 256, '0');  convert_element_type_889 = None
        wait_tensor_243 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_243);  all_gather_into_tensor_243 = None
        permute_296 = torch.ops.aten.permute.default(wait_tensor_243, [1, 0]);  wait_tensor_243 = None
        permute_519 = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        mm_298 = torch.ops.aten.mm.default(view_1215, permute_519);  view_1215 = permute_519 = None
        view_1216 = torch.ops.aten.view.default(mm_298, [2, 8192, 14336]);  mm_298 = None
        convert_element_type_1348 = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None
        reduce_scatter_tensor_47 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1348, 'avg', 256, '0');  convert_element_type_1348 = None
        wait_tensor_338 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_47);  reduce_scatter_tensor_47 = None
        mul_364 = torch.ops.aten.mul.Tensor(view_1216, convert_element_type_885);  convert_element_type_885 = None
        mul_365 = torch.ops.aten.mul.Tensor(view_1216, view_915);  view_1216 = view_915 = None
        view_1217 = torch.ops.aten.view.default(mul_364, [16384, 14336]);  mul_364 = None
        permute_521 = torch.ops.aten.permute.default(view_1217, [1, 0])
        mm_299 = torch.ops.aten.mm.default(permute_521, view_911);  permute_521 = None
        permute_523 = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        mm_300 = torch.ops.aten.mm.default(view_1217, permute_523);  view_1217 = permute_523 = None
        view_1218 = torch.ops.aten.view.default(mm_300, [2, 8192, 4096]);  mm_300 = None
        convert_element_type_1353 = torch.ops.prims.convert_element_type.default(mm_299, torch.float32);  mm_299 = None
        reduce_scatter_tensor_48 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1353, 'avg', 256, '0');  convert_element_type_1353 = None
        wait_tensor_339 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_48);  reduce_scatter_tensor_48 = None
        convert_element_type_1354 = torch.ops.prims.convert_element_type.default(mul_365, torch.float32);  mul_365 = None
        neg_5 = torch.ops.aten.neg.default(convert_element_type_884)
        exp_5 = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_164 = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(add_164);  add_164 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_367 = torch.ops.aten.mul.Tensor(convert_element_type_1354, mul_366);  convert_element_type_1354 = None
        sub_16 = torch.ops.aten.sub.Tensor(1, mul_366);  mul_366 = None
        mul_368 = torch.ops.aten.mul.Tensor(convert_element_type_884, sub_16);  convert_element_type_884 = sub_16 = None
        add_165 = torch.ops.aten.add.Tensor(mul_368, 1);  mul_368 = None
        mul_369 = torch.ops.aten.mul.Tensor(mul_367, add_165);  mul_367 = add_165 = None
        convert_element_type_1356 = torch.ops.prims.convert_element_type.default(mul_369, torch.bfloat16);  mul_369 = None
        view_1219 = torch.ops.aten.view.default(convert_element_type_1356, [16384, 14336]);  convert_element_type_1356 = None
        permute_525 = torch.ops.aten.permute.default(view_1219, [1, 0])
        mm_301 = torch.ops.aten.mm.default(permute_525, view_911);  permute_525 = view_911 = None
        convert_element_type_881 = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        all_gather_into_tensor_241 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_881, 256, '0');  convert_element_type_881 = None
        wait_tensor_241 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_241);  all_gather_into_tensor_241 = None
        permute_294 = torch.ops.aten.permute.default(wait_tensor_241, [1, 0]);  wait_tensor_241 = None
        permute_527 = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
        mm_302 = torch.ops.aten.mm.default(view_1219, permute_527);  view_1219 = permute_527 = None
        view_1220 = torch.ops.aten.view.default(mm_302, [2, 8192, 4096]);  mm_302 = None
        add_166 = torch.ops.aten.add.Tensor(view_1218, view_1220);  view_1218 = view_1220 = None
        convert_element_type_1361 = torch.ops.prims.convert_element_type.default(mm_301, torch.float32);  mm_301 = None
        reduce_scatter_tensor_49 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1361, 'avg', 256, '0');  convert_element_type_1361 = None
        wait_tensor_340 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_49);  reduce_scatter_tensor_49 = None
        convert_element_type_1362 = torch.ops.prims.convert_element_type.default(add_166, torch.float32);  add_166 = None
        convert_element_type_1364 = torch.ops.prims.convert_element_type.default(wait_tensor_240, torch.float32);  wait_tensor_240 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_1362, convert_element_type_1364);  convert_element_type_1364 = None
        mul_372 = torch.ops.aten.mul.Tensor(mul_212, mul_370)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        div_11 = torch.ops.aten.div.Tensor(mul_212, 4096)
        mul_373 = torch.ops.aten.mul.Tensor(div_11, sum_33);  div_11 = sum_33 = None
        sub_17 = torch.ops.aten.sub.Tensor(mul_370, mul_373);  mul_370 = mul_373 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_53);  sub_17 = rsqrt_53 = None
        mul_375 = torch.ops.aten.mul.Tensor(convert_element_type_1362, mul_212);  convert_element_type_1362 = mul_212 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        convert_element_type_1365 = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None
        add_167 = torch.ops.aten.add.Tensor(add_163, convert_element_type_1365);  add_163 = convert_element_type_1365 = None
        convert_element_type_default_54 = torch.ops.prims.convert_element_type.default(sum_34, torch.float32);  sum_34 = None
        reduce_scatter_tensor_50 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_54, 'avg', 256, '0');  convert_element_type_default_54 = None
        wait_tensor_341 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_50);  reduce_scatter_tensor_50 = None
        view_1221 = torch.ops.aten.view.default(add_167, [16384, 4096])
        permute_529 = torch.ops.aten.permute.default(view_1221, [1, 0])
        mm_303 = torch.ops.aten.mm.default(permute_529, view_907);  permute_529 = view_907 = None
        permute_531 = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
        mm_304 = torch.ops.aten.mm.default(view_1221, permute_531);  view_1221 = permute_531 = None
        view_1222 = torch.ops.aten.view.default(mm_304, [2, 8192, 4096]);  mm_304 = None
        convert_element_type_1372 = torch.ops.prims.convert_element_type.default(mm_303, torch.float32);  mm_303 = None
        reduce_scatter_tensor_51 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1372, 'avg', 256, '0');  convert_element_type_1372 = None
        wait_tensor_342 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_51);  reduce_scatter_tensor_51 = None
        view_1223 = torch.ops.aten.view.default(view_1222, [2, 8192, 32, 128]);  view_1222 = None
        permute_533 = torch.ops.aten.permute.default(view_1223, [0, 2, 1, 3]);  view_1223 = None
        convert_element_type_859 = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        all_gather_into_tensor_235 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_859, 256, '0');  convert_element_type_859 = None
        wait_tensor_235 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_235);  all_gather_into_tensor_235 = None
        convert_element_type_860 = torch.ops.prims.convert_element_type.default(add_103, torch.float32);  add_103 = None
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_860, 2)
        mean_52 = torch.ops.aten.mean.dim(pow_53, [2], True);  pow_53 = None
        add_104 = torch.ops.aten.add.Scalar(mean_52, 1e-05);  mean_52 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_860, rsqrt_52);  convert_element_type_860 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, wait_tensor_235)
        convert_element_type_861 = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        view_887 = torch.ops.aten.view.default(convert_element_type_861, [16384, 4096]);  convert_element_type_861 = None
        view_888 = torch.ops.aten.view.default(mm_182, [2, 8192, 4096]);  mm_182 = None
        convert_element_type_865 = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        all_gather_into_tensor_237 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_865, 256, '0');  convert_element_type_865 = None
        wait_tensor_237 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_237);  all_gather_into_tensor_237 = None
        permute_287 = torch.ops.aten.permute.default(wait_tensor_237, [1, 0]);  wait_tensor_237 = None
        mm_183 = torch.ops.aten.mm.default(view_887, permute_287)
        view_891 = torch.ops.aten.view.default(mm_183, [2, 8192, 1024]);  mm_183 = None
        view_894 = torch.ops.aten.view.default(mm_184, [2, 8192, 1024]);  mm_184 = None
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
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_533, permute_289, permute_290, permute_291, getitem_234, getitem_235, getitem_240, getitem_241, None, None, None, 8192, 8192, 0.0, True);  permute_533 = permute_289 = permute_290 = permute_291 = getitem_234 = getitem_235 = getitem_240 = getitem_241 = None
        getitem_303 = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_304 = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_305 = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None
        permute_534 = torch.ops.aten.permute.default(getitem_305, [0, 2, 1, 3]);  getitem_305 = None
        permute_535 = torch.ops.aten.permute.default(getitem_304, [0, 2, 1, 3]);  getitem_304 = None
        permute_536 = torch.ops.aten.permute.default(getitem_303, [0, 2, 1, 3]);  getitem_303 = None
        view_1224 = torch.ops.aten.view.default(permute_534, [2, 8192, 8, 4, 128]);  permute_534 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(view_1224, [3], True);  view_1224 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(sum_35, 3);  sum_35 = None
        view_1225 = torch.ops.aten.view.default(permute_535, [2, 8192, 8, 4, 128]);  permute_535 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(view_1225, [3], True);  view_1225 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(sum_36, 3);  sum_36 = None
        convert_element_type_1373 = torch.ops.prims.convert_element_type.default(squeeze_11, torch.float32);  squeeze_11 = None
        convert_element_type_1374 = torch.ops.prims.convert_element_type.default(permute_536, torch.float32);  permute_536 = None
        view_1226 = torch.ops.aten.view.default(convert_element_type_1373, [2, 8192, 8, 64, 2]);  convert_element_type_1373 = None
        view_as_complex_74 = torch.ops.aten.view_as_complex.default(view_1226);  view_1226 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_as_complex_74, _conj);  view_as_complex_74 = None
        view_1227 = torch.ops.aten.view.default(convert_element_type_1374, [2, 8192, 32, 64, 2]);  convert_element_type_1374 = None
        view_as_complex_75 = torch.ops.aten.view_as_complex.default(view_1227);  view_1227 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_as_complex_75, _conj);  view_as_complex_75 = None
        view_as_real_74 = torch.ops.aten.view_as_real.default(mul_376);  mul_376 = None
        view_1228 = torch.ops.aten.view.default(view_as_real_74, [2, 8192, 8, 128]);  view_as_real_74 = None
        convert_element_type_1375 = torch.ops.prims.convert_element_type.default(view_1228, torch.bfloat16);  view_1228 = None
        view_as_real_75 = torch.ops.aten.view_as_real.default(mul_377);  mul_377 = None
        view_1229 = torch.ops.aten.view.default(view_as_real_75, [2, 8192, 32, 128]);  view_as_real_75 = None
        convert_element_type_1376 = torch.ops.prims.convert_element_type.default(view_1229, torch.bfloat16);  view_1229 = None
        view_1230 = torch.ops.aten.view.default(squeeze_10, [2, 8192, 1024]);  squeeze_10 = None
        view_1231 = torch.ops.aten.view.default(convert_element_type_1375, [2, 8192, 1024]);  convert_element_type_1375 = None
        view_1232 = torch.ops.aten.view.default(convert_element_type_1376, [2, 8192, 4096]);  convert_element_type_1376 = None
        view_1233 = torch.ops.aten.view.default(view_1230, [16384, 1024]);  view_1230 = None
        permute_537 = torch.ops.aten.permute.default(view_1233, [1, 0])
        mm_305 = torch.ops.aten.mm.default(permute_537, view_887);  permute_537 = None
        convert_element_type_868 = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16);  primals_241 = None
        all_gather_into_tensor_238 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_868, 256, '0');  convert_element_type_868 = None
        wait_tensor_238 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_238);  all_gather_into_tensor_238 = None
        permute_288 = torch.ops.aten.permute.default(wait_tensor_238, [1, 0]);  wait_tensor_238 = None
        permute_539 = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        mm_306 = torch.ops.aten.mm.default(view_1233, permute_539);  view_1233 = permute_539 = None
        view_1234 = torch.ops.aten.view.default(mm_306, [2, 8192, 4096]);  mm_306 = None
        convert_element_type_1381 = torch.ops.prims.convert_element_type.default(mm_305, torch.float32);  mm_305 = None
        reduce_scatter_tensor_52 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1381, 'avg', 256, '0');  convert_element_type_1381 = None
        wait_tensor_343 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_52);  reduce_scatter_tensor_52 = None
        view_1235 = torch.ops.aten.view.default(view_1231, [16384, 1024]);  view_1231 = None
        permute_541 = torch.ops.aten.permute.default(view_1235, [1, 0])
        mm_307 = torch.ops.aten.mm.default(permute_541, view_887);  permute_541 = None
        permute_543 = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        mm_308 = torch.ops.aten.mm.default(view_1235, permute_543);  view_1235 = permute_543 = None
        view_1236 = torch.ops.aten.view.default(mm_308, [2, 8192, 4096]);  mm_308 = None
        add_168 = torch.ops.aten.add.Tensor(view_1234, view_1236);  view_1234 = view_1236 = None
        convert_element_type_1386 = torch.ops.prims.convert_element_type.default(mm_307, torch.float32);  mm_307 = None
        reduce_scatter_tensor_53 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1386, 'avg', 256, '0');  convert_element_type_1386 = None
        wait_tensor_344 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_53);  reduce_scatter_tensor_53 = None
        view_1237 = torch.ops.aten.view.default(view_1232, [16384, 4096]);  view_1232 = None
        permute_545 = torch.ops.aten.permute.default(view_1237, [1, 0])
        mm_309 = torch.ops.aten.mm.default(permute_545, view_887);  permute_545 = view_887 = None
        convert_element_type_862 = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16);  primals_239 = None
        all_gather_into_tensor_236 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_862, 256, '0');  convert_element_type_862 = None
        wait_tensor_236 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_236);  all_gather_into_tensor_236 = None
        permute_286 = torch.ops.aten.permute.default(wait_tensor_236, [1, 0]);  wait_tensor_236 = None
        permute_547 = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
        mm_310 = torch.ops.aten.mm.default(view_1237, permute_547);  view_1237 = permute_547 = None
        view_1238 = torch.ops.aten.view.default(mm_310, [2, 8192, 4096]);  mm_310 = None
        add_169 = torch.ops.aten.add.Tensor(add_168, view_1238);  add_168 = view_1238 = None
        convert_element_type_1391 = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None
        reduce_scatter_tensor_54 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1391, 'avg', 256, '0');  convert_element_type_1391 = None
        wait_tensor_345 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_54);  reduce_scatter_tensor_54 = None
        convert_element_type_1392 = torch.ops.prims.convert_element_type.default(add_169, torch.float32);  add_169 = None
        convert_element_type_1394 = torch.ops.prims.convert_element_type.default(wait_tensor_235, torch.float32);  wait_tensor_235 = None
        mul_378 = torch.ops.aten.mul.Tensor(convert_element_type_1392, convert_element_type_1394);  convert_element_type_1394 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_208, mul_378)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
        div_12 = torch.ops.aten.div.Tensor(mul_208, 4096)
        mul_381 = torch.ops.aten.mul.Tensor(div_12, sum_37);  div_12 = sum_37 = None
        sub_18 = torch.ops.aten.sub.Tensor(mul_378, mul_381);  mul_378 = mul_381 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_52);  sub_18 = rsqrt_52 = None
        mul_383 = torch.ops.aten.mul.Tensor(convert_element_type_1392, mul_208);  convert_element_type_1392 = mul_208 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1]);  mul_383 = None
        convert_element_type_1395 = torch.ops.prims.convert_element_type.default(mul_382, torch.bfloat16);  mul_382 = None
        add_170 = torch.ops.aten.add.Tensor(add_167, convert_element_type_1395);  add_167 = convert_element_type_1395 = None
        convert_element_type_default_53 = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None
        reduce_scatter_tensor_55 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_53, 'avg', 256, '0');  convert_element_type_default_53 = None
        wait_tensor_346 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_55);  reduce_scatter_tensor_55 = None
        view_1239 = torch.ops.aten.view.default(add_170, [16384, 4096])
        permute_549 = torch.ops.aten.permute.default(view_1239, [1, 0])
        permute_281 = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3])
        view_871 = torch.ops.aten.view.default(permute_281, [2, 8192, -1]);  permute_281 = None
        convert_element_type_842 = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16);  primals_233 = None
        all_gather_into_tensor_230 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_842, 256, '0');  convert_element_type_842 = None
        wait_tensor_230 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_230);  all_gather_into_tensor_230 = None
        permute_282 = torch.ops.aten.permute.default(wait_tensor_230, [1, 0]);  wait_tensor_230 = None
        view_873 = torch.ops.aten.view.default(view_871, [16384, 4096]);  view_871 = None
        mm_178 = torch.ops.aten.mm.default(view_873, permute_282)
        view_874 = torch.ops.aten.view.default(mm_178, [2, 8192, 4096]);  mm_178 = None
        add_101 = torch.ops.aten.add.Tensor(add_99, view_874);  view_874 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16);  primals_234 = None
        all_gather_into_tensor_231 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_845, 256, '0');  convert_element_type_845 = None
        wait_tensor_231 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_231);  all_gather_into_tensor_231 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_846, 2)
        mean_51 = torch.ops.aten.mean.dim(pow_52, [2], True);  pow_52 = None
        add_102 = torch.ops.aten.add.Scalar(mean_51, 1e-05);  mean_51 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_204 = torch.ops.aten.mul.Tensor(convert_element_type_846, rsqrt_51);  convert_element_type_846 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, wait_tensor_231)
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16);  mul_205 = None
        view_877 = torch.ops.aten.view.default(convert_element_type_847, [16384, 4096]);  convert_element_type_847 = None
        view_878 = torch.ops.aten.view.default(mm_179, [2, 8192, 14336]);  mm_179 = None
        convert_element_type_851 = torch.ops.prims.convert_element_type.default(view_878, torch.float32);  view_878 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(convert_element_type_851)
        mul_206 = torch.ops.aten.mul.Tensor(convert_element_type_851, sigmoid_25);  sigmoid_25 = None
        convert_element_type_852 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        all_gather_into_tensor_233 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_853, 256, '0');  convert_element_type_853 = None
        wait_tensor_233 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_233);  all_gather_into_tensor_233 = None
        permute_284 = torch.ops.aten.permute.default(wait_tensor_233, [1, 0]);  wait_tensor_233 = None
        mm_180 = torch.ops.aten.mm.default(view_877, permute_284)
        view_881 = torch.ops.aten.view.default(mm_180, [2, 8192, 14336]);  mm_180 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_852, view_881)
        view_883 = torch.ops.aten.view.default(mul_207, [16384, 14336]);  mul_207 = None
        mm_311 = torch.ops.aten.mm.default(permute_549, view_883);  permute_549 = view_883 = None
        convert_element_type_856 = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        all_gather_into_tensor_234 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_856, 256, '0');  convert_element_type_856 = None
        wait_tensor_234 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_234);  all_gather_into_tensor_234 = None
        permute_285 = torch.ops.aten.permute.default(wait_tensor_234, [1, 0]);  wait_tensor_234 = None
        permute_551 = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        mm_312 = torch.ops.aten.mm.default(view_1239, permute_551);  view_1239 = permute_551 = None
        view_1240 = torch.ops.aten.view.default(mm_312, [2, 8192, 14336]);  mm_312 = None
        convert_element_type_1402 = torch.ops.prims.convert_element_type.default(mm_311, torch.float32);  mm_311 = None
        reduce_scatter_tensor_56 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1402, 'avg', 256, '0');  convert_element_type_1402 = None
        wait_tensor_347 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_56);  reduce_scatter_tensor_56 = None
        mul_384 = torch.ops.aten.mul.Tensor(view_1240, convert_element_type_852);  convert_element_type_852 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_1240, view_881);  view_1240 = view_881 = None
        view_1241 = torch.ops.aten.view.default(mul_384, [16384, 14336]);  mul_384 = None
        permute_553 = torch.ops.aten.permute.default(view_1241, [1, 0])
        mm_313 = torch.ops.aten.mm.default(permute_553, view_877);  permute_553 = None
        permute_555 = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        mm_314 = torch.ops.aten.mm.default(view_1241, permute_555);  view_1241 = permute_555 = None
        view_1242 = torch.ops.aten.view.default(mm_314, [2, 8192, 4096]);  mm_314 = None
        convert_element_type_1407 = torch.ops.prims.convert_element_type.default(mm_313, torch.float32);  mm_313 = None
        reduce_scatter_tensor_57 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1407, 'avg', 256, '0');  convert_element_type_1407 = None
        wait_tensor_348 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_57);  reduce_scatter_tensor_57 = None
        convert_element_type_1408 = torch.ops.prims.convert_element_type.default(mul_385, torch.float32);  mul_385 = None
        neg_6 = torch.ops.aten.neg.default(convert_element_type_851)
        exp_6 = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_171 = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(add_171);  add_171 = None
        mul_386 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_387 = torch.ops.aten.mul.Tensor(convert_element_type_1408, mul_386);  convert_element_type_1408 = None
        sub_19 = torch.ops.aten.sub.Tensor(1, mul_386);  mul_386 = None
        mul_388 = torch.ops.aten.mul.Tensor(convert_element_type_851, sub_19);  convert_element_type_851 = sub_19 = None
        add_172 = torch.ops.aten.add.Tensor(mul_388, 1);  mul_388 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_387, add_172);  mul_387 = add_172 = None
        convert_element_type_1410 = torch.ops.prims.convert_element_type.default(mul_389, torch.bfloat16);  mul_389 = None
        view_1243 = torch.ops.aten.view.default(convert_element_type_1410, [16384, 14336]);  convert_element_type_1410 = None
        permute_557 = torch.ops.aten.permute.default(view_1243, [1, 0])
        mm_315 = torch.ops.aten.mm.default(permute_557, view_877);  permute_557 = view_877 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16);  primals_235 = None
        all_gather_into_tensor_232 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_848, 256, '0');  convert_element_type_848 = None
        wait_tensor_232 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_232);  all_gather_into_tensor_232 = None
        permute_283 = torch.ops.aten.permute.default(wait_tensor_232, [1, 0]);  wait_tensor_232 = None
        permute_559 = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        mm_316 = torch.ops.aten.mm.default(view_1243, permute_559);  view_1243 = permute_559 = None
        view_1244 = torch.ops.aten.view.default(mm_316, [2, 8192, 4096]);  mm_316 = None
        add_173 = torch.ops.aten.add.Tensor(view_1242, view_1244);  view_1242 = view_1244 = None
        convert_element_type_1415 = torch.ops.prims.convert_element_type.default(mm_315, torch.float32);  mm_315 = None
        reduce_scatter_tensor_58 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1415, 'avg', 256, '0');  convert_element_type_1415 = None
        wait_tensor_349 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_58);  reduce_scatter_tensor_58 = None
        convert_element_type_1416 = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        convert_element_type_1418 = torch.ops.prims.convert_element_type.default(wait_tensor_231, torch.float32);  wait_tensor_231 = None
        mul_390 = torch.ops.aten.mul.Tensor(convert_element_type_1416, convert_element_type_1418);  convert_element_type_1418 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_204, mul_390)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        div_13 = torch.ops.aten.div.Tensor(mul_204, 4096)
        mul_393 = torch.ops.aten.mul.Tensor(div_13, sum_39);  div_13 = sum_39 = None
        sub_20 = torch.ops.aten.sub.Tensor(mul_390, mul_393);  mul_390 = mul_393 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_51);  sub_20 = rsqrt_51 = None
        mul_395 = torch.ops.aten.mul.Tensor(convert_element_type_1416, mul_204);  convert_element_type_1416 = mul_204 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        convert_element_type_1419 = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None
        add_174 = torch.ops.aten.add.Tensor(add_170, convert_element_type_1419);  add_170 = convert_element_type_1419 = None
        convert_element_type_default_52 = torch.ops.prims.convert_element_type.default(sum_40, torch.float32);  sum_40 = None
        reduce_scatter_tensor_59 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_52, 'avg', 256, '0');  convert_element_type_default_52 = None
        wait_tensor_350 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_59);  reduce_scatter_tensor_59 = None
        view_1245 = torch.ops.aten.view.default(add_174, [16384, 4096])
        permute_561 = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_317 = torch.ops.aten.mm.default(permute_561, view_873);  permute_561 = view_873 = None
        permute_563 = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
        mm_318 = torch.ops.aten.mm.default(view_1245, permute_563);  view_1245 = permute_563 = None
        view_1246 = torch.ops.aten.view.default(mm_318, [2, 8192, 4096]);  mm_318 = None
        convert_element_type_1426 = torch.ops.prims.convert_element_type.default(mm_317, torch.float32);  mm_317 = None
        reduce_scatter_tensor_60 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1426, 'avg', 256, '0');  convert_element_type_1426 = None
        wait_tensor_351 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_60);  reduce_scatter_tensor_60 = None
        view_1247 = torch.ops.aten.view.default(view_1246, [2, 8192, 32, 128]);  view_1246 = None
        permute_565 = torch.ops.aten.permute.default(view_1247, [0, 2, 1, 3]);  view_1247 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        all_gather_into_tensor_226 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_826, 256, '0');  convert_element_type_826 = None
        wait_tensor_226 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_226);  all_gather_into_tensor_226 = None
        convert_element_type_827 = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_827, 2)
        mean_50 = torch.ops.aten.mean.dim(pow_51, [2], True);  pow_51 = None
        add_100 = torch.ops.aten.add.Scalar(mean_50, 1e-05);  mean_50 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_827, rsqrt_50);  convert_element_type_827 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, wait_tensor_226)
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        view_853 = torch.ops.aten.view.default(convert_element_type_828, [16384, 4096]);  convert_element_type_828 = None
        view_854 = torch.ops.aten.view.default(mm_175, [2, 8192, 4096]);  mm_175 = None
        convert_element_type_832 = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16);  primals_231 = None
        all_gather_into_tensor_228 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_832, 256, '0');  convert_element_type_832 = None
        wait_tensor_228 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_228);  all_gather_into_tensor_228 = None
        permute_276 = torch.ops.aten.permute.default(wait_tensor_228, [1, 0]);  wait_tensor_228 = None
        mm_176 = torch.ops.aten.mm.default(view_853, permute_276)
        view_857 = torch.ops.aten.view.default(mm_176, [2, 8192, 1024]);  mm_176 = None
        view_860 = torch.ops.aten.view.default(mm_177, [2, 8192, 1024]);  mm_177 = None
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
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_565, permute_278, permute_279, permute_280, getitem_225, getitem_226, getitem_231, getitem_232, None, None, None, 8192, 8192, 0.0, True);  permute_565 = permute_278 = permute_279 = permute_280 = getitem_225 = getitem_226 = getitem_231 = getitem_232 = None
        getitem_306 = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_307 = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_308 = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None
        permute_566 = torch.ops.aten.permute.default(getitem_308, [0, 2, 1, 3]);  getitem_308 = None
        permute_567 = torch.ops.aten.permute.default(getitem_307, [0, 2, 1, 3]);  getitem_307 = None
        permute_568 = torch.ops.aten.permute.default(getitem_306, [0, 2, 1, 3]);  getitem_306 = None
        view_1248 = torch.ops.aten.view.default(permute_566, [2, 8192, 8, 4, 128]);  permute_566 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(view_1248, [3], True);  view_1248 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(sum_41, 3);  sum_41 = None
        view_1249 = torch.ops.aten.view.default(permute_567, [2, 8192, 8, 4, 128]);  permute_567 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_1249, [3], True);  view_1249 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(sum_42, 3);  sum_42 = None
        convert_element_type_1427 = torch.ops.prims.convert_element_type.default(squeeze_13, torch.float32);  squeeze_13 = None
        convert_element_type_1428 = torch.ops.prims.convert_element_type.default(permute_568, torch.float32);  permute_568 = None
        view_1250 = torch.ops.aten.view.default(convert_element_type_1427, [2, 8192, 8, 64, 2]);  convert_element_type_1427 = None
        view_as_complex_76 = torch.ops.aten.view_as_complex.default(view_1250);  view_1250 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_as_complex_76, _conj);  view_as_complex_76 = None
        view_1251 = torch.ops.aten.view.default(convert_element_type_1428, [2, 8192, 32, 64, 2]);  convert_element_type_1428 = None
        view_as_complex_77 = torch.ops.aten.view_as_complex.default(view_1251);  view_1251 = None
        mul_397 = torch.ops.aten.mul.Tensor(view_as_complex_77, _conj);  view_as_complex_77 = None
        view_as_real_76 = torch.ops.aten.view_as_real.default(mul_396);  mul_396 = None
        view_1252 = torch.ops.aten.view.default(view_as_real_76, [2, 8192, 8, 128]);  view_as_real_76 = None
        convert_element_type_1429 = torch.ops.prims.convert_element_type.default(view_1252, torch.bfloat16);  view_1252 = None
        view_as_real_77 = torch.ops.aten.view_as_real.default(mul_397);  mul_397 = None
        view_1253 = torch.ops.aten.view.default(view_as_real_77, [2, 8192, 32, 128]);  view_as_real_77 = None
        convert_element_type_1430 = torch.ops.prims.convert_element_type.default(view_1253, torch.bfloat16);  view_1253 = None
        view_1254 = torch.ops.aten.view.default(squeeze_12, [2, 8192, 1024]);  squeeze_12 = None
        view_1255 = torch.ops.aten.view.default(convert_element_type_1429, [2, 8192, 1024]);  convert_element_type_1429 = None
        view_1256 = torch.ops.aten.view.default(convert_element_type_1430, [2, 8192, 4096]);  convert_element_type_1430 = None
        view_1257 = torch.ops.aten.view.default(view_1254, [16384, 1024]);  view_1254 = None
        permute_569 = torch.ops.aten.permute.default(view_1257, [1, 0])
        mm_319 = torch.ops.aten.mm.default(permute_569, view_853);  permute_569 = None
        convert_element_type_835 = torch.ops.prims.convert_element_type.default(primals_232, torch.bfloat16);  primals_232 = None
        all_gather_into_tensor_229 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_835, 256, '0');  convert_element_type_835 = None
        wait_tensor_229 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_229);  all_gather_into_tensor_229 = None
        permute_277 = torch.ops.aten.permute.default(wait_tensor_229, [1, 0]);  wait_tensor_229 = None
        permute_571 = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
        mm_320 = torch.ops.aten.mm.default(view_1257, permute_571);  view_1257 = permute_571 = None
        view_1258 = torch.ops.aten.view.default(mm_320, [2, 8192, 4096]);  mm_320 = None
        convert_element_type_1435 = torch.ops.prims.convert_element_type.default(mm_319, torch.float32);  mm_319 = None
        reduce_scatter_tensor_61 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1435, 'avg', 256, '0');  convert_element_type_1435 = None
        wait_tensor_352 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_61);  reduce_scatter_tensor_61 = None
        view_1259 = torch.ops.aten.view.default(view_1255, [16384, 1024]);  view_1255 = None
        permute_573 = torch.ops.aten.permute.default(view_1259, [1, 0])
        mm_321 = torch.ops.aten.mm.default(permute_573, view_853);  permute_573 = None
        permute_575 = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        mm_322 = torch.ops.aten.mm.default(view_1259, permute_575);  view_1259 = permute_575 = None
        view_1260 = torch.ops.aten.view.default(mm_322, [2, 8192, 4096]);  mm_322 = None
        add_175 = torch.ops.aten.add.Tensor(view_1258, view_1260);  view_1258 = view_1260 = None
        convert_element_type_1440 = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None
        reduce_scatter_tensor_62 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1440, 'avg', 256, '0');  convert_element_type_1440 = None
        wait_tensor_353 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_62);  reduce_scatter_tensor_62 = None
        view_1261 = torch.ops.aten.view.default(view_1256, [16384, 4096]);  view_1256 = None
        permute_577 = torch.ops.aten.permute.default(view_1261, [1, 0])
        mm_323 = torch.ops.aten.mm.default(permute_577, view_853);  permute_577 = view_853 = None
        convert_element_type_829 = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        all_gather_into_tensor_227 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_829, 256, '0');  convert_element_type_829 = None
        wait_tensor_227 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_227);  all_gather_into_tensor_227 = None
        permute_275 = torch.ops.aten.permute.default(wait_tensor_227, [1, 0]);  wait_tensor_227 = None
        permute_579 = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        mm_324 = torch.ops.aten.mm.default(view_1261, permute_579);  view_1261 = permute_579 = None
        view_1262 = torch.ops.aten.view.default(mm_324, [2, 8192, 4096]);  mm_324 = None
        add_176 = torch.ops.aten.add.Tensor(add_175, view_1262);  add_175 = view_1262 = None
        convert_element_type_1445 = torch.ops.prims.convert_element_type.default(mm_323, torch.float32);  mm_323 = None
        reduce_scatter_tensor_63 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1445, 'avg', 256, '0');  convert_element_type_1445 = None
        wait_tensor_354 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_63);  reduce_scatter_tensor_63 = None
        convert_element_type_1446 = torch.ops.prims.convert_element_type.default(add_176, torch.float32);  add_176 = None
        convert_element_type_1448 = torch.ops.prims.convert_element_type.default(wait_tensor_226, torch.float32);  wait_tensor_226 = None
        mul_398 = torch.ops.aten.mul.Tensor(convert_element_type_1446, convert_element_type_1448);  convert_element_type_1448 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_200, mul_398)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
        div_14 = torch.ops.aten.div.Tensor(mul_200, 4096)
        mul_401 = torch.ops.aten.mul.Tensor(div_14, sum_43);  div_14 = sum_43 = None
        sub_21 = torch.ops.aten.sub.Tensor(mul_398, mul_401);  mul_398 = mul_401 = None
        mul_402 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_50);  sub_21 = rsqrt_50 = None
        mul_403 = torch.ops.aten.mul.Tensor(convert_element_type_1446, mul_200);  convert_element_type_1446 = mul_200 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
        convert_element_type_1449 = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None
        add_177 = torch.ops.aten.add.Tensor(add_174, convert_element_type_1449);  add_174 = convert_element_type_1449 = None
        convert_element_type_default_51 = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None
        reduce_scatter_tensor_64 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_51, 'avg', 256, '0');  convert_element_type_default_51 = None
        wait_tensor_355 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_64);  reduce_scatter_tensor_64 = None
        view_1263 = torch.ops.aten.view.default(add_177, [16384, 4096])
        permute_581 = torch.ops.aten.permute.default(view_1263, [1, 0])
        permute_270 = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3])
        view_837 = torch.ops.aten.view.default(permute_270, [2, 8192, -1]);  permute_270 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        all_gather_into_tensor_221 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_809, 256, '0');  convert_element_type_809 = None
        wait_tensor_221 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_221);  all_gather_into_tensor_221 = None
        permute_271 = torch.ops.aten.permute.default(wait_tensor_221, [1, 0]);  wait_tensor_221 = None
        view_839 = torch.ops.aten.view.default(view_837, [16384, 4096]);  view_837 = None
        mm_171 = torch.ops.aten.mm.default(view_839, permute_271)
        view_840 = torch.ops.aten.view.default(mm_171, [2, 8192, 4096]);  mm_171 = None
        add_97 = torch.ops.aten.add.Tensor(add_95, view_840);  view_840 = None
        convert_element_type_812 = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16);  primals_225 = None
        all_gather_into_tensor_222 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_812, 256, '0');  convert_element_type_812 = None
        wait_tensor_222 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_222);  all_gather_into_tensor_222 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(add_97, torch.float32);  add_97 = None
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_813, 2)
        mean_49 = torch.ops.aten.mean.dim(pow_50, [2], True);  pow_50 = None
        add_98 = torch.ops.aten.add.Scalar(mean_49, 1e-05);  mean_49 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_196 = torch.ops.aten.mul.Tensor(convert_element_type_813, rsqrt_49);  convert_element_type_813 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, wait_tensor_222)
        convert_element_type_814 = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16);  mul_197 = None
        view_843 = torch.ops.aten.view.default(convert_element_type_814, [16384, 4096]);  convert_element_type_814 = None
        view_844 = torch.ops.aten.view.default(mm_172, [2, 8192, 14336]);  mm_172 = None
        convert_element_type_818 = torch.ops.prims.convert_element_type.default(view_844, torch.float32);  view_844 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(convert_element_type_818)
        mul_198 = torch.ops.aten.mul.Tensor(convert_element_type_818, sigmoid_24);  sigmoid_24 = None
        convert_element_type_819 = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16);  primals_227 = None
        all_gather_into_tensor_224 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_820, 256, '0');  convert_element_type_820 = None
        wait_tensor_224 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_224);  all_gather_into_tensor_224 = None
        permute_273 = torch.ops.aten.permute.default(wait_tensor_224, [1, 0]);  wait_tensor_224 = None
        mm_173 = torch.ops.aten.mm.default(view_843, permute_273)
        view_847 = torch.ops.aten.view.default(mm_173, [2, 8192, 14336]);  mm_173 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_819, view_847)
        view_849 = torch.ops.aten.view.default(mul_199, [16384, 14336]);  mul_199 = None
        mm_325 = torch.ops.aten.mm.default(permute_581, view_849);  permute_581 = view_849 = None
        convert_element_type_823 = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        all_gather_into_tensor_225 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_823, 256, '0');  convert_element_type_823 = None
        wait_tensor_225 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_225);  all_gather_into_tensor_225 = None
        permute_274 = torch.ops.aten.permute.default(wait_tensor_225, [1, 0]);  wait_tensor_225 = None
        permute_583 = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
        mm_326 = torch.ops.aten.mm.default(view_1263, permute_583);  view_1263 = permute_583 = None
        view_1264 = torch.ops.aten.view.default(mm_326, [2, 8192, 14336]);  mm_326 = None
        convert_element_type_1456 = torch.ops.prims.convert_element_type.default(mm_325, torch.float32);  mm_325 = None
        reduce_scatter_tensor_65 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1456, 'avg', 256, '0');  convert_element_type_1456 = None
        wait_tensor_356 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_65);  reduce_scatter_tensor_65 = None
        mul_404 = torch.ops.aten.mul.Tensor(view_1264, convert_element_type_819);  convert_element_type_819 = None
        mul_405 = torch.ops.aten.mul.Tensor(view_1264, view_847);  view_1264 = view_847 = None
        view_1265 = torch.ops.aten.view.default(mul_404, [16384, 14336]);  mul_404 = None
        permute_585 = torch.ops.aten.permute.default(view_1265, [1, 0])
        mm_327 = torch.ops.aten.mm.default(permute_585, view_843);  permute_585 = None
        permute_587 = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        mm_328 = torch.ops.aten.mm.default(view_1265, permute_587);  view_1265 = permute_587 = None
        view_1266 = torch.ops.aten.view.default(mm_328, [2, 8192, 4096]);  mm_328 = None
        convert_element_type_1461 = torch.ops.prims.convert_element_type.default(mm_327, torch.float32);  mm_327 = None
        reduce_scatter_tensor_66 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1461, 'avg', 256, '0');  convert_element_type_1461 = None
        wait_tensor_357 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_66);  reduce_scatter_tensor_66 = None
        convert_element_type_1462 = torch.ops.prims.convert_element_type.default(mul_405, torch.float32);  mul_405 = None
        neg_7 = torch.ops.aten.neg.default(convert_element_type_818)
        exp_7 = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_178 = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(add_178);  add_178 = None
        mul_406 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_407 = torch.ops.aten.mul.Tensor(convert_element_type_1462, mul_406);  convert_element_type_1462 = None
        sub_22 = torch.ops.aten.sub.Tensor(1, mul_406);  mul_406 = None
        mul_408 = torch.ops.aten.mul.Tensor(convert_element_type_818, sub_22);  convert_element_type_818 = sub_22 = None
        add_179 = torch.ops.aten.add.Tensor(mul_408, 1);  mul_408 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_407, add_179);  mul_407 = add_179 = None
        convert_element_type_1464 = torch.ops.prims.convert_element_type.default(mul_409, torch.bfloat16);  mul_409 = None
        view_1267 = torch.ops.aten.view.default(convert_element_type_1464, [16384, 14336]);  convert_element_type_1464 = None
        permute_589 = torch.ops.aten.permute.default(view_1267, [1, 0])
        mm_329 = torch.ops.aten.mm.default(permute_589, view_843);  permute_589 = view_843 = None
        convert_element_type_815 = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16);  primals_226 = None
        all_gather_into_tensor_223 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_815, 256, '0');  convert_element_type_815 = None
        wait_tensor_223 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_223);  all_gather_into_tensor_223 = None
        permute_272 = torch.ops.aten.permute.default(wait_tensor_223, [1, 0]);  wait_tensor_223 = None
        permute_591 = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        mm_330 = torch.ops.aten.mm.default(view_1267, permute_591);  view_1267 = permute_591 = None
        view_1268 = torch.ops.aten.view.default(mm_330, [2, 8192, 4096]);  mm_330 = None
        add_180 = torch.ops.aten.add.Tensor(view_1266, view_1268);  view_1266 = view_1268 = None
        convert_element_type_1469 = torch.ops.prims.convert_element_type.default(mm_329, torch.float32);  mm_329 = None
        reduce_scatter_tensor_67 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1469, 'avg', 256, '0');  convert_element_type_1469 = None
        wait_tensor_358 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_67);  reduce_scatter_tensor_67 = None
        convert_element_type_1470 = torch.ops.prims.convert_element_type.default(add_180, torch.float32);  add_180 = None
        convert_element_type_1472 = torch.ops.prims.convert_element_type.default(wait_tensor_222, torch.float32);  wait_tensor_222 = None
        mul_410 = torch.ops.aten.mul.Tensor(convert_element_type_1470, convert_element_type_1472);  convert_element_type_1472 = None
        mul_412 = torch.ops.aten.mul.Tensor(mul_196, mul_410)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_412, [2], True);  mul_412 = None
        div_15 = torch.ops.aten.div.Tensor(mul_196, 4096)
        mul_413 = torch.ops.aten.mul.Tensor(div_15, sum_45);  div_15 = sum_45 = None
        sub_23 = torch.ops.aten.sub.Tensor(mul_410, mul_413);  mul_410 = mul_413 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_49);  sub_23 = rsqrt_49 = None
        mul_415 = torch.ops.aten.mul.Tensor(convert_element_type_1470, mul_196);  convert_element_type_1470 = mul_196 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1]);  mul_415 = None
        convert_element_type_1473 = torch.ops.prims.convert_element_type.default(mul_414, torch.bfloat16);  mul_414 = None
        add_181 = torch.ops.aten.add.Tensor(add_177, convert_element_type_1473);  add_177 = convert_element_type_1473 = None
        convert_element_type_default_50 = torch.ops.prims.convert_element_type.default(sum_46, torch.float32);  sum_46 = None
        reduce_scatter_tensor_68 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_50, 'avg', 256, '0');  convert_element_type_default_50 = None
        wait_tensor_359 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_68);  reduce_scatter_tensor_68 = None
        view_1269 = torch.ops.aten.view.default(add_181, [16384, 4096])
        permute_593 = torch.ops.aten.permute.default(view_1269, [1, 0])
        mm_331 = torch.ops.aten.mm.default(permute_593, view_839);  permute_593 = view_839 = None
        permute_595 = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
        mm_332 = torch.ops.aten.mm.default(view_1269, permute_595);  view_1269 = permute_595 = None
        view_1270 = torch.ops.aten.view.default(mm_332, [2, 8192, 4096]);  mm_332 = None
        convert_element_type_1480 = torch.ops.prims.convert_element_type.default(mm_331, torch.float32);  mm_331 = None
        reduce_scatter_tensor_69 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1480, 'avg', 256, '0');  convert_element_type_1480 = None
        wait_tensor_360 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_69);  reduce_scatter_tensor_69 = None
        view_1271 = torch.ops.aten.view.default(view_1270, [2, 8192, 32, 128]);  view_1270 = None
        permute_597 = torch.ops.aten.permute.default(view_1271, [0, 2, 1, 3]);  view_1271 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16);  primals_220 = None
        all_gather_into_tensor_217 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_793, 256, '0');  convert_element_type_793 = None
        wait_tensor_217 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_217);  all_gather_into_tensor_217 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_794, 2)
        mean_48 = torch.ops.aten.mean.dim(pow_49, [2], True);  pow_49 = None
        add_96 = torch.ops.aten.add.Scalar(mean_48, 1e-05);  mean_48 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_794, rsqrt_48);  convert_element_type_794 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, wait_tensor_217)
        convert_element_type_795 = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        view_819 = torch.ops.aten.view.default(convert_element_type_795, [16384, 4096]);  convert_element_type_795 = None
        view_820 = torch.ops.aten.view.default(mm_168, [2, 8192, 4096]);  mm_168 = None
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        all_gather_into_tensor_219 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_799, 256, '0');  convert_element_type_799 = None
        wait_tensor_219 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_219);  all_gather_into_tensor_219 = None
        permute_265 = torch.ops.aten.permute.default(wait_tensor_219, [1, 0]);  wait_tensor_219 = None
        mm_169 = torch.ops.aten.mm.default(view_819, permute_265)
        view_823 = torch.ops.aten.view.default(mm_169, [2, 8192, 1024]);  mm_169 = None
        view_826 = torch.ops.aten.view.default(mm_170, [2, 8192, 1024]);  mm_170 = None
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
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_597, permute_267, permute_268, permute_269, getitem_216, getitem_217, getitem_222, getitem_223, None, None, None, 8192, 8192, 0.0, True);  permute_597 = permute_267 = permute_268 = permute_269 = getitem_216 = getitem_217 = getitem_222 = getitem_223 = None
        getitem_309 = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_310 = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_311 = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None
        permute_598 = torch.ops.aten.permute.default(getitem_311, [0, 2, 1, 3]);  getitem_311 = None
        permute_599 = torch.ops.aten.permute.default(getitem_310, [0, 2, 1, 3]);  getitem_310 = None
        permute_600 = torch.ops.aten.permute.default(getitem_309, [0, 2, 1, 3]);  getitem_309 = None
        view_1272 = torch.ops.aten.view.default(permute_598, [2, 8192, 8, 4, 128]);  permute_598 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(view_1272, [3], True);  view_1272 = None
        squeeze_14 = torch.ops.aten.squeeze.dim(sum_47, 3);  sum_47 = None
        view_1273 = torch.ops.aten.view.default(permute_599, [2, 8192, 8, 4, 128]);  permute_599 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(view_1273, [3], True);  view_1273 = None
        squeeze_15 = torch.ops.aten.squeeze.dim(sum_48, 3);  sum_48 = None
        convert_element_type_1481 = torch.ops.prims.convert_element_type.default(squeeze_15, torch.float32);  squeeze_15 = None
        convert_element_type_1482 = torch.ops.prims.convert_element_type.default(permute_600, torch.float32);  permute_600 = None
        view_1274 = torch.ops.aten.view.default(convert_element_type_1481, [2, 8192, 8, 64, 2]);  convert_element_type_1481 = None
        view_as_complex_78 = torch.ops.aten.view_as_complex.default(view_1274);  view_1274 = None
        mul_416 = torch.ops.aten.mul.Tensor(view_as_complex_78, _conj);  view_as_complex_78 = None
        view_1275 = torch.ops.aten.view.default(convert_element_type_1482, [2, 8192, 32, 64, 2]);  convert_element_type_1482 = None
        view_as_complex_79 = torch.ops.aten.view_as_complex.default(view_1275);  view_1275 = None
        mul_417 = torch.ops.aten.mul.Tensor(view_as_complex_79, _conj);  view_as_complex_79 = None
        view_as_real_78 = torch.ops.aten.view_as_real.default(mul_416);  mul_416 = None
        view_1276 = torch.ops.aten.view.default(view_as_real_78, [2, 8192, 8, 128]);  view_as_real_78 = None
        convert_element_type_1483 = torch.ops.prims.convert_element_type.default(view_1276, torch.bfloat16);  view_1276 = None
        view_as_real_79 = torch.ops.aten.view_as_real.default(mul_417);  mul_417 = None
        view_1277 = torch.ops.aten.view.default(view_as_real_79, [2, 8192, 32, 128]);  view_as_real_79 = None
        convert_element_type_1484 = torch.ops.prims.convert_element_type.default(view_1277, torch.bfloat16);  view_1277 = None
        view_1278 = torch.ops.aten.view.default(squeeze_14, [2, 8192, 1024]);  squeeze_14 = None
        view_1279 = torch.ops.aten.view.default(convert_element_type_1483, [2, 8192, 1024]);  convert_element_type_1483 = None
        view_1280 = torch.ops.aten.view.default(convert_element_type_1484, [2, 8192, 4096]);  convert_element_type_1484 = None
        view_1281 = torch.ops.aten.view.default(view_1278, [16384, 1024]);  view_1278 = None
        permute_601 = torch.ops.aten.permute.default(view_1281, [1, 0])
        mm_333 = torch.ops.aten.mm.default(permute_601, view_819);  permute_601 = None
        convert_element_type_802 = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16);  primals_223 = None
        all_gather_into_tensor_220 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_802, 256, '0');  convert_element_type_802 = None
        wait_tensor_220 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_220);  all_gather_into_tensor_220 = None
        permute_266 = torch.ops.aten.permute.default(wait_tensor_220, [1, 0]);  wait_tensor_220 = None
        permute_603 = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        mm_334 = torch.ops.aten.mm.default(view_1281, permute_603);  view_1281 = permute_603 = None
        view_1282 = torch.ops.aten.view.default(mm_334, [2, 8192, 4096]);  mm_334 = None
        convert_element_type_1489 = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None
        reduce_scatter_tensor_70 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1489, 'avg', 256, '0');  convert_element_type_1489 = None
        wait_tensor_361 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_70);  reduce_scatter_tensor_70 = None
        view_1283 = torch.ops.aten.view.default(view_1279, [16384, 1024]);  view_1279 = None
        permute_605 = torch.ops.aten.permute.default(view_1283, [1, 0])
        mm_335 = torch.ops.aten.mm.default(permute_605, view_819);  permute_605 = None
        permute_607 = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm_336 = torch.ops.aten.mm.default(view_1283, permute_607);  view_1283 = permute_607 = None
        view_1284 = torch.ops.aten.view.default(mm_336, [2, 8192, 4096]);  mm_336 = None
        add_182 = torch.ops.aten.add.Tensor(view_1282, view_1284);  view_1282 = view_1284 = None
        convert_element_type_1494 = torch.ops.prims.convert_element_type.default(mm_335, torch.float32);  mm_335 = None
        reduce_scatter_tensor_71 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1494, 'avg', 256, '0');  convert_element_type_1494 = None
        wait_tensor_362 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_71);  reduce_scatter_tensor_71 = None
        view_1285 = torch.ops.aten.view.default(view_1280, [16384, 4096]);  view_1280 = None
        permute_609 = torch.ops.aten.permute.default(view_1285, [1, 0])
        mm_337 = torch.ops.aten.mm.default(permute_609, view_819);  permute_609 = view_819 = None
        convert_element_type_796 = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        all_gather_into_tensor_218 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_796, 256, '0');  convert_element_type_796 = None
        wait_tensor_218 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_218);  all_gather_into_tensor_218 = None
        permute_264 = torch.ops.aten.permute.default(wait_tensor_218, [1, 0]);  wait_tensor_218 = None
        permute_611 = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_338 = torch.ops.aten.mm.default(view_1285, permute_611);  view_1285 = permute_611 = None
        view_1286 = torch.ops.aten.view.default(mm_338, [2, 8192, 4096]);  mm_338 = None
        add_183 = torch.ops.aten.add.Tensor(add_182, view_1286);  add_182 = view_1286 = None
        convert_element_type_1499 = torch.ops.prims.convert_element_type.default(mm_337, torch.float32);  mm_337 = None
        reduce_scatter_tensor_72 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1499, 'avg', 256, '0');  convert_element_type_1499 = None
        wait_tensor_363 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_72);  reduce_scatter_tensor_72 = None
        convert_element_type_1500 = torch.ops.prims.convert_element_type.default(add_183, torch.float32);  add_183 = None
        convert_element_type_1502 = torch.ops.prims.convert_element_type.default(wait_tensor_217, torch.float32);  wait_tensor_217 = None
        mul_418 = torch.ops.aten.mul.Tensor(convert_element_type_1500, convert_element_type_1502);  convert_element_type_1502 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_192, mul_418)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_420, [2], True);  mul_420 = None
        div_16 = torch.ops.aten.div.Tensor(mul_192, 4096)
        mul_421 = torch.ops.aten.mul.Tensor(div_16, sum_49);  div_16 = sum_49 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_418, mul_421);  mul_418 = mul_421 = None
        mul_422 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_48);  sub_24 = rsqrt_48 = None
        mul_423 = torch.ops.aten.mul.Tensor(convert_element_type_1500, mul_192);  convert_element_type_1500 = mul_192 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_423, [0, 1]);  mul_423 = None
        convert_element_type_1503 = torch.ops.prims.convert_element_type.default(mul_422, torch.bfloat16);  mul_422 = None
        add_184 = torch.ops.aten.add.Tensor(add_181, convert_element_type_1503);  add_181 = convert_element_type_1503 = None
        convert_element_type_default_49 = torch.ops.prims.convert_element_type.default(sum_50, torch.float32);  sum_50 = None
        reduce_scatter_tensor_73 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_49, 'avg', 256, '0');  convert_element_type_default_49 = None
        wait_tensor_364 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_73);  reduce_scatter_tensor_73 = None
        view_1287 = torch.ops.aten.view.default(add_184, [16384, 4096])
        permute_613 = torch.ops.aten.permute.default(view_1287, [1, 0])
        permute_259 = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3])
        view_803 = torch.ops.aten.view.default(permute_259, [2, 8192, -1]);  permute_259 = None
        convert_element_type_776 = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        all_gather_into_tensor_212 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_776, 256, '0');  convert_element_type_776 = None
        wait_tensor_212 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_212);  all_gather_into_tensor_212 = None
        permute_260 = torch.ops.aten.permute.default(wait_tensor_212, [1, 0]);  wait_tensor_212 = None
        view_805 = torch.ops.aten.view.default(view_803, [16384, 4096]);  view_803 = None
        mm_164 = torch.ops.aten.mm.default(view_805, permute_260)
        view_806 = torch.ops.aten.view.default(mm_164, [2, 8192, 4096]);  mm_164 = None
        add_93 = torch.ops.aten.add.Tensor(add_91, view_806);  view_806 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        all_gather_into_tensor_213 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_779, 256, '0');  convert_element_type_779 = None
        wait_tensor_213 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_213);  all_gather_into_tensor_213 = None
        convert_element_type_780 = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_780, 2)
        mean_47 = torch.ops.aten.mean.dim(pow_48, [2], True);  pow_48 = None
        add_94 = torch.ops.aten.add.Scalar(mean_47, 1e-05);  mean_47 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_188 = torch.ops.aten.mul.Tensor(convert_element_type_780, rsqrt_47);  convert_element_type_780 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, wait_tensor_213)
        convert_element_type_781 = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        view_809 = torch.ops.aten.view.default(convert_element_type_781, [16384, 4096]);  convert_element_type_781 = None
        view_810 = torch.ops.aten.view.default(mm_165, [2, 8192, 14336]);  mm_165 = None
        convert_element_type_785 = torch.ops.prims.convert_element_type.default(view_810, torch.float32);  view_810 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(convert_element_type_785)
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_785, sigmoid_23);  sigmoid_23 = None
        convert_element_type_786 = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        convert_element_type_787 = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        all_gather_into_tensor_215 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_787, 256, '0');  convert_element_type_787 = None
        wait_tensor_215 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_215);  all_gather_into_tensor_215 = None
        permute_262 = torch.ops.aten.permute.default(wait_tensor_215, [1, 0]);  wait_tensor_215 = None
        mm_166 = torch.ops.aten.mm.default(view_809, permute_262)
        view_813 = torch.ops.aten.view.default(mm_166, [2, 8192, 14336]);  mm_166 = None
        mul_191 = torch.ops.aten.mul.Tensor(convert_element_type_786, view_813)
        view_815 = torch.ops.aten.view.default(mul_191, [16384, 14336]);  mul_191 = None
        mm_339 = torch.ops.aten.mm.default(permute_613, view_815);  permute_613 = view_815 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16);  primals_219 = None
        all_gather_into_tensor_216 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_790, 256, '0');  convert_element_type_790 = None
        wait_tensor_216 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_216);  all_gather_into_tensor_216 = None
        permute_263 = torch.ops.aten.permute.default(wait_tensor_216, [1, 0]);  wait_tensor_216 = None
        permute_615 = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_340 = torch.ops.aten.mm.default(view_1287, permute_615);  view_1287 = permute_615 = None
        view_1288 = torch.ops.aten.view.default(mm_340, [2, 8192, 14336]);  mm_340 = None
        convert_element_type_1510 = torch.ops.prims.convert_element_type.default(mm_339, torch.float32);  mm_339 = None
        reduce_scatter_tensor_74 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1510, 'avg', 256, '0');  convert_element_type_1510 = None
        wait_tensor_365 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_74);  reduce_scatter_tensor_74 = None
        mul_424 = torch.ops.aten.mul.Tensor(view_1288, convert_element_type_786);  convert_element_type_786 = None
        mul_425 = torch.ops.aten.mul.Tensor(view_1288, view_813);  view_1288 = view_813 = None
        view_1289 = torch.ops.aten.view.default(mul_424, [16384, 14336]);  mul_424 = None
        permute_617 = torch.ops.aten.permute.default(view_1289, [1, 0])
        mm_341 = torch.ops.aten.mm.default(permute_617, view_809);  permute_617 = None
        permute_619 = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_342 = torch.ops.aten.mm.default(view_1289, permute_619);  view_1289 = permute_619 = None
        view_1290 = torch.ops.aten.view.default(mm_342, [2, 8192, 4096]);  mm_342 = None
        convert_element_type_1515 = torch.ops.prims.convert_element_type.default(mm_341, torch.float32);  mm_341 = None
        reduce_scatter_tensor_75 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1515, 'avg', 256, '0');  convert_element_type_1515 = None
        wait_tensor_366 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_75);  reduce_scatter_tensor_75 = None
        convert_element_type_1516 = torch.ops.prims.convert_element_type.default(mul_425, torch.float32);  mul_425 = None
        neg_8 = torch.ops.aten.neg.default(convert_element_type_785)
        exp_8 = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_185 = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(add_185);  add_185 = None
        mul_426 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_427 = torch.ops.aten.mul.Tensor(convert_element_type_1516, mul_426);  convert_element_type_1516 = None
        sub_25 = torch.ops.aten.sub.Tensor(1, mul_426);  mul_426 = None
        mul_428 = torch.ops.aten.mul.Tensor(convert_element_type_785, sub_25);  convert_element_type_785 = sub_25 = None
        add_186 = torch.ops.aten.add.Tensor(mul_428, 1);  mul_428 = None
        mul_429 = torch.ops.aten.mul.Tensor(mul_427, add_186);  mul_427 = add_186 = None
        convert_element_type_1518 = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None
        view_1291 = torch.ops.aten.view.default(convert_element_type_1518, [16384, 14336]);  convert_element_type_1518 = None
        permute_621 = torch.ops.aten.permute.default(view_1291, [1, 0])
        mm_343 = torch.ops.aten.mm.default(permute_621, view_809);  permute_621 = view_809 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        all_gather_into_tensor_214 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_782, 256, '0');  convert_element_type_782 = None
        wait_tensor_214 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_214);  all_gather_into_tensor_214 = None
        permute_261 = torch.ops.aten.permute.default(wait_tensor_214, [1, 0]);  wait_tensor_214 = None
        permute_623 = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_344 = torch.ops.aten.mm.default(view_1291, permute_623);  view_1291 = permute_623 = None
        view_1292 = torch.ops.aten.view.default(mm_344, [2, 8192, 4096]);  mm_344 = None
        add_187 = torch.ops.aten.add.Tensor(view_1290, view_1292);  view_1290 = view_1292 = None
        convert_element_type_1523 = torch.ops.prims.convert_element_type.default(mm_343, torch.float32);  mm_343 = None
        reduce_scatter_tensor_76 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1523, 'avg', 256, '0');  convert_element_type_1523 = None
        wait_tensor_367 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_76);  reduce_scatter_tensor_76 = None
        convert_element_type_1524 = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        convert_element_type_1526 = torch.ops.prims.convert_element_type.default(wait_tensor_213, torch.float32);  wait_tensor_213 = None
        mul_430 = torch.ops.aten.mul.Tensor(convert_element_type_1524, convert_element_type_1526);  convert_element_type_1526 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_188, mul_430)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
        div_17 = torch.ops.aten.div.Tensor(mul_188, 4096)
        mul_433 = torch.ops.aten.mul.Tensor(div_17, sum_51);  div_17 = sum_51 = None
        sub_26 = torch.ops.aten.sub.Tensor(mul_430, mul_433);  mul_430 = mul_433 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_47);  sub_26 = rsqrt_47 = None
        mul_435 = torch.ops.aten.mul.Tensor(convert_element_type_1524, mul_188);  convert_element_type_1524 = mul_188 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
        convert_element_type_1527 = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        add_188 = torch.ops.aten.add.Tensor(add_184, convert_element_type_1527);  add_184 = convert_element_type_1527 = None
        convert_element_type_default_48 = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None
        reduce_scatter_tensor_77 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_48, 'avg', 256, '0');  convert_element_type_default_48 = None
        wait_tensor_368 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_77);  reduce_scatter_tensor_77 = None
        view_1293 = torch.ops.aten.view.default(add_188, [16384, 4096])
        permute_625 = torch.ops.aten.permute.default(view_1293, [1, 0])
        mm_345 = torch.ops.aten.mm.default(permute_625, view_805);  permute_625 = view_805 = None
        permute_627 = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
        mm_346 = torch.ops.aten.mm.default(view_1293, permute_627);  view_1293 = permute_627 = None
        view_1294 = torch.ops.aten.view.default(mm_346, [2, 8192, 4096]);  mm_346 = None
        convert_element_type_1534 = torch.ops.prims.convert_element_type.default(mm_345, torch.float32);  mm_345 = None
        reduce_scatter_tensor_78 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1534, 'avg', 256, '0');  convert_element_type_1534 = None
        wait_tensor_369 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_78);  reduce_scatter_tensor_78 = None
        view_1295 = torch.ops.aten.view.default(view_1294, [2, 8192, 32, 128]);  view_1294 = None
        permute_629 = torch.ops.aten.permute.default(view_1295, [0, 2, 1, 3]);  view_1295 = None
        convert_element_type_760 = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        all_gather_into_tensor_208 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_760, 256, '0');  convert_element_type_760 = None
        wait_tensor_208 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_208);  all_gather_into_tensor_208 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_761, 2)
        mean_46 = torch.ops.aten.mean.dim(pow_47, [2], True);  pow_47 = None
        add_92 = torch.ops.aten.add.Scalar(mean_46, 1e-05);  mean_46 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_184 = torch.ops.aten.mul.Tensor(convert_element_type_761, rsqrt_46);  convert_element_type_761 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, wait_tensor_208)
        convert_element_type_762 = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None
        view_785 = torch.ops.aten.view.default(convert_element_type_762, [16384, 4096]);  convert_element_type_762 = None
        view_786 = torch.ops.aten.view.default(mm_161, [2, 8192, 4096]);  mm_161 = None
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        all_gather_into_tensor_210 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_766, 256, '0');  convert_element_type_766 = None
        wait_tensor_210 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_210);  all_gather_into_tensor_210 = None
        permute_254 = torch.ops.aten.permute.default(wait_tensor_210, [1, 0]);  wait_tensor_210 = None
        mm_162 = torch.ops.aten.mm.default(view_785, permute_254)
        view_789 = torch.ops.aten.view.default(mm_162, [2, 8192, 1024]);  mm_162 = None
        view_792 = torch.ops.aten.view.default(mm_163, [2, 8192, 1024]);  mm_163 = None
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
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_629, permute_256, permute_257, permute_258, getitem_207, getitem_208, getitem_213, getitem_214, None, None, None, 8192, 8192, 0.0, True);  permute_629 = permute_256 = permute_257 = permute_258 = getitem_207 = getitem_208 = getitem_213 = getitem_214 = None
        getitem_312 = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_313 = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_314 = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None
        permute_630 = torch.ops.aten.permute.default(getitem_314, [0, 2, 1, 3]);  getitem_314 = None
        permute_631 = torch.ops.aten.permute.default(getitem_313, [0, 2, 1, 3]);  getitem_313 = None
        permute_632 = torch.ops.aten.permute.default(getitem_312, [0, 2, 1, 3]);  getitem_312 = None
        view_1296 = torch.ops.aten.view.default(permute_630, [2, 8192, 8, 4, 128]);  permute_630 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(view_1296, [3], True);  view_1296 = None
        squeeze_16 = torch.ops.aten.squeeze.dim(sum_53, 3);  sum_53 = None
        view_1297 = torch.ops.aten.view.default(permute_631, [2, 8192, 8, 4, 128]);  permute_631 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(view_1297, [3], True);  view_1297 = None
        squeeze_17 = torch.ops.aten.squeeze.dim(sum_54, 3);  sum_54 = None
        convert_element_type_1535 = torch.ops.prims.convert_element_type.default(squeeze_17, torch.float32);  squeeze_17 = None
        convert_element_type_1536 = torch.ops.prims.convert_element_type.default(permute_632, torch.float32);  permute_632 = None
        view_1298 = torch.ops.aten.view.default(convert_element_type_1535, [2, 8192, 8, 64, 2]);  convert_element_type_1535 = None
        view_as_complex_80 = torch.ops.aten.view_as_complex.default(view_1298);  view_1298 = None
        mul_436 = torch.ops.aten.mul.Tensor(view_as_complex_80, _conj);  view_as_complex_80 = None
        view_1299 = torch.ops.aten.view.default(convert_element_type_1536, [2, 8192, 32, 64, 2]);  convert_element_type_1536 = None
        view_as_complex_81 = torch.ops.aten.view_as_complex.default(view_1299);  view_1299 = None
        mul_437 = torch.ops.aten.mul.Tensor(view_as_complex_81, _conj);  view_as_complex_81 = None
        view_as_real_80 = torch.ops.aten.view_as_real.default(mul_436);  mul_436 = None
        view_1300 = torch.ops.aten.view.default(view_as_real_80, [2, 8192, 8, 128]);  view_as_real_80 = None
        convert_element_type_1537 = torch.ops.prims.convert_element_type.default(view_1300, torch.bfloat16);  view_1300 = None
        view_as_real_81 = torch.ops.aten.view_as_real.default(mul_437);  mul_437 = None
        view_1301 = torch.ops.aten.view.default(view_as_real_81, [2, 8192, 32, 128]);  view_as_real_81 = None
        convert_element_type_1538 = torch.ops.prims.convert_element_type.default(view_1301, torch.bfloat16);  view_1301 = None
        view_1302 = torch.ops.aten.view.default(squeeze_16, [2, 8192, 1024]);  squeeze_16 = None
        view_1303 = torch.ops.aten.view.default(convert_element_type_1537, [2, 8192, 1024]);  convert_element_type_1537 = None
        view_1304 = torch.ops.aten.view.default(convert_element_type_1538, [2, 8192, 4096]);  convert_element_type_1538 = None
        view_1305 = torch.ops.aten.view.default(view_1302, [16384, 1024]);  view_1302 = None
        permute_633 = torch.ops.aten.permute.default(view_1305, [1, 0])
        mm_347 = torch.ops.aten.mm.default(permute_633, view_785);  permute_633 = None
        convert_element_type_769 = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        all_gather_into_tensor_211 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_769, 256, '0');  convert_element_type_769 = None
        wait_tensor_211 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_211);  all_gather_into_tensor_211 = None
        permute_255 = torch.ops.aten.permute.default(wait_tensor_211, [1, 0]);  wait_tensor_211 = None
        permute_635 = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_348 = torch.ops.aten.mm.default(view_1305, permute_635);  view_1305 = permute_635 = None
        view_1306 = torch.ops.aten.view.default(mm_348, [2, 8192, 4096]);  mm_348 = None
        convert_element_type_1543 = torch.ops.prims.convert_element_type.default(mm_347, torch.float32);  mm_347 = None
        reduce_scatter_tensor_79 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1543, 'avg', 256, '0');  convert_element_type_1543 = None
        wait_tensor_370 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_79);  reduce_scatter_tensor_79 = None
        view_1307 = torch.ops.aten.view.default(view_1303, [16384, 1024]);  view_1303 = None
        permute_637 = torch.ops.aten.permute.default(view_1307, [1, 0])
        mm_349 = torch.ops.aten.mm.default(permute_637, view_785);  permute_637 = None
        permute_639 = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        mm_350 = torch.ops.aten.mm.default(view_1307, permute_639);  view_1307 = permute_639 = None
        view_1308 = torch.ops.aten.view.default(mm_350, [2, 8192, 4096]);  mm_350 = None
        add_189 = torch.ops.aten.add.Tensor(view_1306, view_1308);  view_1306 = view_1308 = None
        convert_element_type_1548 = torch.ops.prims.convert_element_type.default(mm_349, torch.float32);  mm_349 = None
        reduce_scatter_tensor_80 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1548, 'avg', 256, '0');  convert_element_type_1548 = None
        wait_tensor_371 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_80);  reduce_scatter_tensor_80 = None
        view_1309 = torch.ops.aten.view.default(view_1304, [16384, 4096]);  view_1304 = None
        permute_641 = torch.ops.aten.permute.default(view_1309, [1, 0])
        mm_351 = torch.ops.aten.mm.default(permute_641, view_785);  permute_641 = view_785 = None
        convert_element_type_763 = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        all_gather_into_tensor_209 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_763, 256, '0');  convert_element_type_763 = None
        wait_tensor_209 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_209);  all_gather_into_tensor_209 = None
        permute_253 = torch.ops.aten.permute.default(wait_tensor_209, [1, 0]);  wait_tensor_209 = None
        permute_643 = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_352 = torch.ops.aten.mm.default(view_1309, permute_643);  view_1309 = permute_643 = None
        view_1310 = torch.ops.aten.view.default(mm_352, [2, 8192, 4096]);  mm_352 = None
        add_190 = torch.ops.aten.add.Tensor(add_189, view_1310);  add_189 = view_1310 = None
        convert_element_type_1553 = torch.ops.prims.convert_element_type.default(mm_351, torch.float32);  mm_351 = None
        reduce_scatter_tensor_81 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1553, 'avg', 256, '0');  convert_element_type_1553 = None
        wait_tensor_372 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_81);  reduce_scatter_tensor_81 = None
        convert_element_type_1554 = torch.ops.prims.convert_element_type.default(add_190, torch.float32);  add_190 = None
        convert_element_type_1556 = torch.ops.prims.convert_element_type.default(wait_tensor_208, torch.float32);  wait_tensor_208 = None
        mul_438 = torch.ops.aten.mul.Tensor(convert_element_type_1554, convert_element_type_1556);  convert_element_type_1556 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_184, mul_438)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_440, [2], True);  mul_440 = None
        div_18 = torch.ops.aten.div.Tensor(mul_184, 4096)
        mul_441 = torch.ops.aten.mul.Tensor(div_18, sum_55);  div_18 = sum_55 = None
        sub_27 = torch.ops.aten.sub.Tensor(mul_438, mul_441);  mul_438 = mul_441 = None
        mul_442 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_46);  sub_27 = rsqrt_46 = None
        mul_443 = torch.ops.aten.mul.Tensor(convert_element_type_1554, mul_184);  convert_element_type_1554 = mul_184 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_443, [0, 1]);  mul_443 = None
        convert_element_type_1557 = torch.ops.prims.convert_element_type.default(mul_442, torch.bfloat16);  mul_442 = None
        add_191 = torch.ops.aten.add.Tensor(add_188, convert_element_type_1557);  add_188 = convert_element_type_1557 = None
        convert_element_type_default_47 = torch.ops.prims.convert_element_type.default(sum_56, torch.float32);  sum_56 = None
        reduce_scatter_tensor_82 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_47, 'avg', 256, '0');  convert_element_type_default_47 = None
        wait_tensor_373 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_82);  reduce_scatter_tensor_82 = None
        view_1311 = torch.ops.aten.view.default(add_191, [16384, 4096])
        permute_645 = torch.ops.aten.permute.default(view_1311, [1, 0])
        permute_248 = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3])
        view_769 = torch.ops.aten.view.default(permute_248, [2, 8192, -1]);  permute_248 = None
        convert_element_type_743 = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        all_gather_into_tensor_203 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_743, 256, '0');  convert_element_type_743 = None
        wait_tensor_203 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_203);  all_gather_into_tensor_203 = None
        permute_249 = torch.ops.aten.permute.default(wait_tensor_203, [1, 0]);  wait_tensor_203 = None
        view_771 = torch.ops.aten.view.default(view_769, [16384, 4096]);  view_769 = None
        mm_157 = torch.ops.aten.mm.default(view_771, permute_249)
        view_772 = torch.ops.aten.view.default(mm_157, [2, 8192, 4096]);  mm_157 = None
        add_89 = torch.ops.aten.add.Tensor(add_87, view_772);  view_772 = None
        convert_element_type_746 = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        all_gather_into_tensor_204 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_746, 256, '0');  convert_element_type_746 = None
        wait_tensor_204 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_204);  all_gather_into_tensor_204 = None
        convert_element_type_747 = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_747, 2)
        mean_45 = torch.ops.aten.mean.dim(pow_46, [2], True);  pow_46 = None
        add_90 = torch.ops.aten.add.Scalar(mean_45, 1e-05);  mean_45 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_180 = torch.ops.aten.mul.Tensor(convert_element_type_747, rsqrt_45);  convert_element_type_747 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, wait_tensor_204)
        convert_element_type_748 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_775 = torch.ops.aten.view.default(convert_element_type_748, [16384, 4096]);  convert_element_type_748 = None
        view_776 = torch.ops.aten.view.default(mm_158, [2, 8192, 14336]);  mm_158 = None
        convert_element_type_752 = torch.ops.prims.convert_element_type.default(view_776, torch.float32);  view_776 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(convert_element_type_752)
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_752, sigmoid_22);  sigmoid_22 = None
        convert_element_type_753 = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        convert_element_type_754 = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16);  primals_209 = None
        all_gather_into_tensor_206 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_754, 256, '0');  convert_element_type_754 = None
        wait_tensor_206 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_206);  all_gather_into_tensor_206 = None
        permute_251 = torch.ops.aten.permute.default(wait_tensor_206, [1, 0]);  wait_tensor_206 = None
        mm_159 = torch.ops.aten.mm.default(view_775, permute_251)
        view_779 = torch.ops.aten.view.default(mm_159, [2, 8192, 14336]);  mm_159 = None
        mul_183 = torch.ops.aten.mul.Tensor(convert_element_type_753, view_779)
        view_781 = torch.ops.aten.view.default(mul_183, [16384, 14336]);  mul_183 = None
        mm_353 = torch.ops.aten.mm.default(permute_645, view_781);  permute_645 = view_781 = None
        convert_element_type_757 = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        all_gather_into_tensor_207 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_757, 256, '0');  convert_element_type_757 = None
        wait_tensor_207 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_207);  all_gather_into_tensor_207 = None
        permute_252 = torch.ops.aten.permute.default(wait_tensor_207, [1, 0]);  wait_tensor_207 = None
        permute_647 = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_354 = torch.ops.aten.mm.default(view_1311, permute_647);  view_1311 = permute_647 = None
        view_1312 = torch.ops.aten.view.default(mm_354, [2, 8192, 14336]);  mm_354 = None
        convert_element_type_1564 = torch.ops.prims.convert_element_type.default(mm_353, torch.float32);  mm_353 = None
        reduce_scatter_tensor_83 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1564, 'avg', 256, '0');  convert_element_type_1564 = None
        wait_tensor_374 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_83);  reduce_scatter_tensor_83 = None
        mul_444 = torch.ops.aten.mul.Tensor(view_1312, convert_element_type_753);  convert_element_type_753 = None
        mul_445 = torch.ops.aten.mul.Tensor(view_1312, view_779);  view_1312 = view_779 = None
        view_1313 = torch.ops.aten.view.default(mul_444, [16384, 14336]);  mul_444 = None
        permute_649 = torch.ops.aten.permute.default(view_1313, [1, 0])
        mm_355 = torch.ops.aten.mm.default(permute_649, view_775);  permute_649 = None
        permute_651 = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_356 = torch.ops.aten.mm.default(view_1313, permute_651);  view_1313 = permute_651 = None
        view_1314 = torch.ops.aten.view.default(mm_356, [2, 8192, 4096]);  mm_356 = None
        convert_element_type_1569 = torch.ops.prims.convert_element_type.default(mm_355, torch.float32);  mm_355 = None
        reduce_scatter_tensor_84 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1569, 'avg', 256, '0');  convert_element_type_1569 = None
        wait_tensor_375 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_84);  reduce_scatter_tensor_84 = None
        convert_element_type_1570 = torch.ops.prims.convert_element_type.default(mul_445, torch.float32);  mul_445 = None
        neg_9 = torch.ops.aten.neg.default(convert_element_type_752)
        exp_9 = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_192 = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(add_192);  add_192 = None
        mul_446 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_447 = torch.ops.aten.mul.Tensor(convert_element_type_1570, mul_446);  convert_element_type_1570 = None
        sub_28 = torch.ops.aten.sub.Tensor(1, mul_446);  mul_446 = None
        mul_448 = torch.ops.aten.mul.Tensor(convert_element_type_752, sub_28);  convert_element_type_752 = sub_28 = None
        add_193 = torch.ops.aten.add.Tensor(mul_448, 1);  mul_448 = None
        mul_449 = torch.ops.aten.mul.Tensor(mul_447, add_193);  mul_447 = add_193 = None
        convert_element_type_1572 = torch.ops.prims.convert_element_type.default(mul_449, torch.bfloat16);  mul_449 = None
        view_1315 = torch.ops.aten.view.default(convert_element_type_1572, [16384, 14336]);  convert_element_type_1572 = None
        permute_653 = torch.ops.aten.permute.default(view_1315, [1, 0])
        mm_357 = torch.ops.aten.mm.default(permute_653, view_775);  permute_653 = view_775 = None
        convert_element_type_749 = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        all_gather_into_tensor_205 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_749, 256, '0');  convert_element_type_749 = None
        wait_tensor_205 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_205);  all_gather_into_tensor_205 = None
        permute_250 = torch.ops.aten.permute.default(wait_tensor_205, [1, 0]);  wait_tensor_205 = None
        permute_655 = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_358 = torch.ops.aten.mm.default(view_1315, permute_655);  view_1315 = permute_655 = None
        view_1316 = torch.ops.aten.view.default(mm_358, [2, 8192, 4096]);  mm_358 = None
        add_194 = torch.ops.aten.add.Tensor(view_1314, view_1316);  view_1314 = view_1316 = None
        convert_element_type_1577 = torch.ops.prims.convert_element_type.default(mm_357, torch.float32);  mm_357 = None
        reduce_scatter_tensor_85 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1577, 'avg', 256, '0');  convert_element_type_1577 = None
        wait_tensor_376 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_85);  reduce_scatter_tensor_85 = None
        convert_element_type_1578 = torch.ops.prims.convert_element_type.default(add_194, torch.float32);  add_194 = None
        convert_element_type_1580 = torch.ops.prims.convert_element_type.default(wait_tensor_204, torch.float32);  wait_tensor_204 = None
        mul_450 = torch.ops.aten.mul.Tensor(convert_element_type_1578, convert_element_type_1580);  convert_element_type_1580 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_180, mul_450)
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_452, [2], True);  mul_452 = None
        div_19 = torch.ops.aten.div.Tensor(mul_180, 4096)
        mul_453 = torch.ops.aten.mul.Tensor(div_19, sum_57);  div_19 = sum_57 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_450, mul_453);  mul_450 = mul_453 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_45);  sub_29 = rsqrt_45 = None
        mul_455 = torch.ops.aten.mul.Tensor(convert_element_type_1578, mul_180);  convert_element_type_1578 = mul_180 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1]);  mul_455 = None
        convert_element_type_1581 = torch.ops.prims.convert_element_type.default(mul_454, torch.bfloat16);  mul_454 = None
        add_195 = torch.ops.aten.add.Tensor(add_191, convert_element_type_1581);  add_191 = convert_element_type_1581 = None
        convert_element_type_default_46 = torch.ops.prims.convert_element_type.default(sum_58, torch.float32);  sum_58 = None
        reduce_scatter_tensor_86 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_46, 'avg', 256, '0');  convert_element_type_default_46 = None
        wait_tensor_377 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_86);  reduce_scatter_tensor_86 = None
        view_1317 = torch.ops.aten.view.default(add_195, [16384, 4096])
        permute_657 = torch.ops.aten.permute.default(view_1317, [1, 0])
        mm_359 = torch.ops.aten.mm.default(permute_657, view_771);  permute_657 = view_771 = None
        permute_659 = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        mm_360 = torch.ops.aten.mm.default(view_1317, permute_659);  view_1317 = permute_659 = None
        view_1318 = torch.ops.aten.view.default(mm_360, [2, 8192, 4096]);  mm_360 = None
        convert_element_type_1588 = torch.ops.prims.convert_element_type.default(mm_359, torch.float32);  mm_359 = None
        reduce_scatter_tensor_87 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1588, 'avg', 256, '0');  convert_element_type_1588 = None
        wait_tensor_378 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_87);  reduce_scatter_tensor_87 = None
        view_1319 = torch.ops.aten.view.default(view_1318, [2, 8192, 32, 128]);  view_1318 = None
        permute_661 = torch.ops.aten.permute.default(view_1319, [0, 2, 1, 3]);  view_1319 = None
        convert_element_type_727 = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16);  primals_202 = None
        all_gather_into_tensor_199 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_727, 256, '0');  convert_element_type_727 = None
        wait_tensor_199 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_199);  all_gather_into_tensor_199 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_728, 2)
        mean_44 = torch.ops.aten.mean.dim(pow_45, [2], True);  pow_45 = None
        add_88 = torch.ops.aten.add.Scalar(mean_44, 1e-05);  mean_44 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_176 = torch.ops.aten.mul.Tensor(convert_element_type_728, rsqrt_44);  convert_element_type_728 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, wait_tensor_199)
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        view_751 = torch.ops.aten.view.default(convert_element_type_729, [16384, 4096]);  convert_element_type_729 = None
        view_752 = torch.ops.aten.view.default(mm_154, [2, 8192, 4096]);  mm_154 = None
        convert_element_type_733 = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        all_gather_into_tensor_201 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_733, 256, '0');  convert_element_type_733 = None
        wait_tensor_201 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_201);  all_gather_into_tensor_201 = None
        permute_243 = torch.ops.aten.permute.default(wait_tensor_201, [1, 0]);  wait_tensor_201 = None
        mm_155 = torch.ops.aten.mm.default(view_751, permute_243)
        view_755 = torch.ops.aten.view.default(mm_155, [2, 8192, 1024]);  mm_155 = None
        view_758 = torch.ops.aten.view.default(mm_156, [2, 8192, 1024]);  mm_156 = None
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
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_661, permute_245, permute_246, permute_247, getitem_198, getitem_199, getitem_204, getitem_205, None, None, None, 8192, 8192, 0.0, True);  permute_661 = permute_245 = permute_246 = permute_247 = getitem_198 = getitem_199 = getitem_204 = getitem_205 = None
        getitem_315 = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_316 = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_317 = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None
        permute_662 = torch.ops.aten.permute.default(getitem_317, [0, 2, 1, 3]);  getitem_317 = None
        permute_663 = torch.ops.aten.permute.default(getitem_316, [0, 2, 1, 3]);  getitem_316 = None
        permute_664 = torch.ops.aten.permute.default(getitem_315, [0, 2, 1, 3]);  getitem_315 = None
        view_1320 = torch.ops.aten.view.default(permute_662, [2, 8192, 8, 4, 128]);  permute_662 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(view_1320, [3], True);  view_1320 = None
        squeeze_18 = torch.ops.aten.squeeze.dim(sum_59, 3);  sum_59 = None
        view_1321 = torch.ops.aten.view.default(permute_663, [2, 8192, 8, 4, 128]);  permute_663 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(view_1321, [3], True);  view_1321 = None
        squeeze_19 = torch.ops.aten.squeeze.dim(sum_60, 3);  sum_60 = None
        convert_element_type_1589 = torch.ops.prims.convert_element_type.default(squeeze_19, torch.float32);  squeeze_19 = None
        convert_element_type_1590 = torch.ops.prims.convert_element_type.default(permute_664, torch.float32);  permute_664 = None
        view_1322 = torch.ops.aten.view.default(convert_element_type_1589, [2, 8192, 8, 64, 2]);  convert_element_type_1589 = None
        view_as_complex_82 = torch.ops.aten.view_as_complex.default(view_1322);  view_1322 = None
        mul_456 = torch.ops.aten.mul.Tensor(view_as_complex_82, _conj);  view_as_complex_82 = None
        view_1323 = torch.ops.aten.view.default(convert_element_type_1590, [2, 8192, 32, 64, 2]);  convert_element_type_1590 = None
        view_as_complex_83 = torch.ops.aten.view_as_complex.default(view_1323);  view_1323 = None
        mul_457 = torch.ops.aten.mul.Tensor(view_as_complex_83, _conj);  view_as_complex_83 = None
        view_as_real_82 = torch.ops.aten.view_as_real.default(mul_456);  mul_456 = None
        view_1324 = torch.ops.aten.view.default(view_as_real_82, [2, 8192, 8, 128]);  view_as_real_82 = None
        convert_element_type_1591 = torch.ops.prims.convert_element_type.default(view_1324, torch.bfloat16);  view_1324 = None
        view_as_real_83 = torch.ops.aten.view_as_real.default(mul_457);  mul_457 = None
        view_1325 = torch.ops.aten.view.default(view_as_real_83, [2, 8192, 32, 128]);  view_as_real_83 = None
        convert_element_type_1592 = torch.ops.prims.convert_element_type.default(view_1325, torch.bfloat16);  view_1325 = None
        view_1326 = torch.ops.aten.view.default(squeeze_18, [2, 8192, 1024]);  squeeze_18 = None
        view_1327 = torch.ops.aten.view.default(convert_element_type_1591, [2, 8192, 1024]);  convert_element_type_1591 = None
        view_1328 = torch.ops.aten.view.default(convert_element_type_1592, [2, 8192, 4096]);  convert_element_type_1592 = None
        view_1329 = torch.ops.aten.view.default(view_1326, [16384, 1024]);  view_1326 = None
        permute_665 = torch.ops.aten.permute.default(view_1329, [1, 0])
        mm_361 = torch.ops.aten.mm.default(permute_665, view_751);  permute_665 = None
        convert_element_type_736 = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        all_gather_into_tensor_202 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_736, 256, '0');  convert_element_type_736 = None
        wait_tensor_202 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_202);  all_gather_into_tensor_202 = None
        permute_244 = torch.ops.aten.permute.default(wait_tensor_202, [1, 0]);  wait_tensor_202 = None
        permute_667 = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_362 = torch.ops.aten.mm.default(view_1329, permute_667);  view_1329 = permute_667 = None
        view_1330 = torch.ops.aten.view.default(mm_362, [2, 8192, 4096]);  mm_362 = None
        convert_element_type_1597 = torch.ops.prims.convert_element_type.default(mm_361, torch.float32);  mm_361 = None
        reduce_scatter_tensor_88 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1597, 'avg', 256, '0');  convert_element_type_1597 = None
        wait_tensor_379 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_88);  reduce_scatter_tensor_88 = None
        view_1331 = torch.ops.aten.view.default(view_1327, [16384, 1024]);  view_1327 = None
        permute_669 = torch.ops.aten.permute.default(view_1331, [1, 0])
        mm_363 = torch.ops.aten.mm.default(permute_669, view_751);  permute_669 = None
        permute_671 = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_364 = torch.ops.aten.mm.default(view_1331, permute_671);  view_1331 = permute_671 = None
        view_1332 = torch.ops.aten.view.default(mm_364, [2, 8192, 4096]);  mm_364 = None
        add_196 = torch.ops.aten.add.Tensor(view_1330, view_1332);  view_1330 = view_1332 = None
        convert_element_type_1602 = torch.ops.prims.convert_element_type.default(mm_363, torch.float32);  mm_363 = None
        reduce_scatter_tensor_89 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1602, 'avg', 256, '0');  convert_element_type_1602 = None
        wait_tensor_380 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_89);  reduce_scatter_tensor_89 = None
        view_1333 = torch.ops.aten.view.default(view_1328, [16384, 4096]);  view_1328 = None
        permute_673 = torch.ops.aten.permute.default(view_1333, [1, 0])
        mm_365 = torch.ops.aten.mm.default(permute_673, view_751);  permute_673 = view_751 = None
        convert_element_type_730 = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        all_gather_into_tensor_200 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_730, 256, '0');  convert_element_type_730 = None
        wait_tensor_200 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_200);  all_gather_into_tensor_200 = None
        permute_242 = torch.ops.aten.permute.default(wait_tensor_200, [1, 0]);  wait_tensor_200 = None
        permute_675 = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_366 = torch.ops.aten.mm.default(view_1333, permute_675);  view_1333 = permute_675 = None
        view_1334 = torch.ops.aten.view.default(mm_366, [2, 8192, 4096]);  mm_366 = None
        add_197 = torch.ops.aten.add.Tensor(add_196, view_1334);  add_196 = view_1334 = None
        convert_element_type_1607 = torch.ops.prims.convert_element_type.default(mm_365, torch.float32);  mm_365 = None
        reduce_scatter_tensor_90 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1607, 'avg', 256, '0');  convert_element_type_1607 = None
        wait_tensor_381 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_90);  reduce_scatter_tensor_90 = None
        convert_element_type_1608 = torch.ops.prims.convert_element_type.default(add_197, torch.float32);  add_197 = None
        convert_element_type_1610 = torch.ops.prims.convert_element_type.default(wait_tensor_199, torch.float32);  wait_tensor_199 = None
        mul_458 = torch.ops.aten.mul.Tensor(convert_element_type_1608, convert_element_type_1610);  convert_element_type_1610 = None
        mul_460 = torch.ops.aten.mul.Tensor(mul_176, mul_458)
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_460, [2], True);  mul_460 = None
        div_20 = torch.ops.aten.div.Tensor(mul_176, 4096)
        mul_461 = torch.ops.aten.mul.Tensor(div_20, sum_61);  div_20 = sum_61 = None
        sub_30 = torch.ops.aten.sub.Tensor(mul_458, mul_461);  mul_458 = mul_461 = None
        mul_462 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_44);  sub_30 = rsqrt_44 = None
        mul_463 = torch.ops.aten.mul.Tensor(convert_element_type_1608, mul_176);  convert_element_type_1608 = mul_176 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1]);  mul_463 = None
        convert_element_type_1611 = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        add_198 = torch.ops.aten.add.Tensor(add_195, convert_element_type_1611);  add_195 = convert_element_type_1611 = None
        convert_element_type_default_45 = torch.ops.prims.convert_element_type.default(sum_62, torch.float32);  sum_62 = None
        reduce_scatter_tensor_91 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_45, 'avg', 256, '0');  convert_element_type_default_45 = None
        wait_tensor_382 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_91);  reduce_scatter_tensor_91 = None
        view_1335 = torch.ops.aten.view.default(add_198, [16384, 4096])
        permute_677 = torch.ops.aten.permute.default(view_1335, [1, 0])
        permute_237 = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3])
        view_735 = torch.ops.aten.view.default(permute_237, [2, 8192, -1]);  permute_237 = None
        convert_element_type_710 = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        all_gather_into_tensor_194 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_710, 256, '0');  convert_element_type_710 = None
        wait_tensor_194 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_194);  all_gather_into_tensor_194 = None
        permute_238 = torch.ops.aten.permute.default(wait_tensor_194, [1, 0]);  wait_tensor_194 = None
        view_737 = torch.ops.aten.view.default(view_735, [16384, 4096]);  view_735 = None
        mm_150 = torch.ops.aten.mm.default(view_737, permute_238)
        view_738 = torch.ops.aten.view.default(mm_150, [2, 8192, 4096]);  mm_150 = None
        add_85 = torch.ops.aten.add.Tensor(add_83, view_738);  view_738 = None
        convert_element_type_713 = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        all_gather_into_tensor_195 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_713, 256, '0');  convert_element_type_713 = None
        wait_tensor_195 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_195);  all_gather_into_tensor_195 = None
        convert_element_type_714 = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_714, 2)
        mean_43 = torch.ops.aten.mean.dim(pow_44, [2], True);  pow_44 = None
        add_86 = torch.ops.aten.add.Scalar(mean_43, 1e-05);  mean_43 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_172 = torch.ops.aten.mul.Tensor(convert_element_type_714, rsqrt_43);  convert_element_type_714 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, wait_tensor_195)
        convert_element_type_715 = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        view_741 = torch.ops.aten.view.default(convert_element_type_715, [16384, 4096]);  convert_element_type_715 = None
        view_742 = torch.ops.aten.view.default(mm_151, [2, 8192, 14336]);  mm_151 = None
        convert_element_type_719 = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(convert_element_type_719)
        mul_174 = torch.ops.aten.mul.Tensor(convert_element_type_719, sigmoid_21);  sigmoid_21 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        convert_element_type_721 = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        all_gather_into_tensor_197 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_721, 256, '0');  convert_element_type_721 = None
        wait_tensor_197 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_197);  all_gather_into_tensor_197 = None
        permute_240 = torch.ops.aten.permute.default(wait_tensor_197, [1, 0]);  wait_tensor_197 = None
        mm_152 = torch.ops.aten.mm.default(view_741, permute_240)
        view_745 = torch.ops.aten.view.default(mm_152, [2, 8192, 14336]);  mm_152 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_720, view_745)
        view_747 = torch.ops.aten.view.default(mul_175, [16384, 14336]);  mul_175 = None
        mm_367 = torch.ops.aten.mm.default(permute_677, view_747);  permute_677 = view_747 = None
        convert_element_type_724 = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        all_gather_into_tensor_198 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_724, 256, '0');  convert_element_type_724 = None
        wait_tensor_198 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_198);  all_gather_into_tensor_198 = None
        permute_241 = torch.ops.aten.permute.default(wait_tensor_198, [1, 0]);  wait_tensor_198 = None
        permute_679 = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_368 = torch.ops.aten.mm.default(view_1335, permute_679);  view_1335 = permute_679 = None
        view_1336 = torch.ops.aten.view.default(mm_368, [2, 8192, 14336]);  mm_368 = None
        convert_element_type_1618 = torch.ops.prims.convert_element_type.default(mm_367, torch.float32);  mm_367 = None
        reduce_scatter_tensor_92 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1618, 'avg', 256, '0');  convert_element_type_1618 = None
        wait_tensor_383 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_92);  reduce_scatter_tensor_92 = None
        mul_464 = torch.ops.aten.mul.Tensor(view_1336, convert_element_type_720);  convert_element_type_720 = None
        mul_465 = torch.ops.aten.mul.Tensor(view_1336, view_745);  view_1336 = view_745 = None
        view_1337 = torch.ops.aten.view.default(mul_464, [16384, 14336]);  mul_464 = None
        permute_681 = torch.ops.aten.permute.default(view_1337, [1, 0])
        mm_369 = torch.ops.aten.mm.default(permute_681, view_741);  permute_681 = None
        permute_683 = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_370 = torch.ops.aten.mm.default(view_1337, permute_683);  view_1337 = permute_683 = None
        view_1338 = torch.ops.aten.view.default(mm_370, [2, 8192, 4096]);  mm_370 = None
        convert_element_type_1623 = torch.ops.prims.convert_element_type.default(mm_369, torch.float32);  mm_369 = None
        reduce_scatter_tensor_93 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1623, 'avg', 256, '0');  convert_element_type_1623 = None
        wait_tensor_384 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_93);  reduce_scatter_tensor_93 = None
        convert_element_type_1624 = torch.ops.prims.convert_element_type.default(mul_465, torch.float32);  mul_465 = None
        neg_10 = torch.ops.aten.neg.default(convert_element_type_719)
        exp_10 = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_199 = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(add_199);  add_199 = None
        mul_466 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_467 = torch.ops.aten.mul.Tensor(convert_element_type_1624, mul_466);  convert_element_type_1624 = None
        sub_31 = torch.ops.aten.sub.Tensor(1, mul_466);  mul_466 = None
        mul_468 = torch.ops.aten.mul.Tensor(convert_element_type_719, sub_31);  convert_element_type_719 = sub_31 = None
        add_200 = torch.ops.aten.add.Tensor(mul_468, 1);  mul_468 = None
        mul_469 = torch.ops.aten.mul.Tensor(mul_467, add_200);  mul_467 = add_200 = None
        convert_element_type_1626 = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16);  mul_469 = None
        view_1339 = torch.ops.aten.view.default(convert_element_type_1626, [16384, 14336]);  convert_element_type_1626 = None
        permute_685 = torch.ops.aten.permute.default(view_1339, [1, 0])
        mm_371 = torch.ops.aten.mm.default(permute_685, view_741);  permute_685 = view_741 = None
        convert_element_type_716 = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        all_gather_into_tensor_196 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_716, 256, '0');  convert_element_type_716 = None
        wait_tensor_196 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_196);  all_gather_into_tensor_196 = None
        permute_239 = torch.ops.aten.permute.default(wait_tensor_196, [1, 0]);  wait_tensor_196 = None
        permute_687 = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_372 = torch.ops.aten.mm.default(view_1339, permute_687);  view_1339 = permute_687 = None
        view_1340 = torch.ops.aten.view.default(mm_372, [2, 8192, 4096]);  mm_372 = None
        add_201 = torch.ops.aten.add.Tensor(view_1338, view_1340);  view_1338 = view_1340 = None
        convert_element_type_1631 = torch.ops.prims.convert_element_type.default(mm_371, torch.float32);  mm_371 = None
        reduce_scatter_tensor_94 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1631, 'avg', 256, '0');  convert_element_type_1631 = None
        wait_tensor_385 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_94);  reduce_scatter_tensor_94 = None
        convert_element_type_1632 = torch.ops.prims.convert_element_type.default(add_201, torch.float32);  add_201 = None
        convert_element_type_1634 = torch.ops.prims.convert_element_type.default(wait_tensor_195, torch.float32);  wait_tensor_195 = None
        mul_470 = torch.ops.aten.mul.Tensor(convert_element_type_1632, convert_element_type_1634);  convert_element_type_1634 = None
        mul_472 = torch.ops.aten.mul.Tensor(mul_172, mul_470)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        div_21 = torch.ops.aten.div.Tensor(mul_172, 4096)
        mul_473 = torch.ops.aten.mul.Tensor(div_21, sum_63);  div_21 = sum_63 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_470, mul_473);  mul_470 = mul_473 = None
        mul_474 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_43);  sub_32 = rsqrt_43 = None
        mul_475 = torch.ops.aten.mul.Tensor(convert_element_type_1632, mul_172);  convert_element_type_1632 = mul_172 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        convert_element_type_1635 = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None
        add_202 = torch.ops.aten.add.Tensor(add_198, convert_element_type_1635);  add_198 = convert_element_type_1635 = None
        convert_element_type_default_44 = torch.ops.prims.convert_element_type.default(sum_64, torch.float32);  sum_64 = None
        reduce_scatter_tensor_95 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_44, 'avg', 256, '0');  convert_element_type_default_44 = None
        wait_tensor_386 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_95);  reduce_scatter_tensor_95 = None
        view_1341 = torch.ops.aten.view.default(add_202, [16384, 4096])
        permute_689 = torch.ops.aten.permute.default(view_1341, [1, 0])
        mm_373 = torch.ops.aten.mm.default(permute_689, view_737);  permute_689 = view_737 = None
        permute_691 = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        mm_374 = torch.ops.aten.mm.default(view_1341, permute_691);  view_1341 = permute_691 = None
        view_1342 = torch.ops.aten.view.default(mm_374, [2, 8192, 4096]);  mm_374 = None
        convert_element_type_1642 = torch.ops.prims.convert_element_type.default(mm_373, torch.float32);  mm_373 = None
        reduce_scatter_tensor_96 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1642, 'avg', 256, '0');  convert_element_type_1642 = None
        wait_tensor_387 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_96);  reduce_scatter_tensor_96 = None
        view_1343 = torch.ops.aten.view.default(view_1342, [2, 8192, 32, 128]);  view_1342 = None
        permute_693 = torch.ops.aten.permute.default(view_1343, [0, 2, 1, 3]);  view_1343 = None
        convert_element_type_694 = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        all_gather_into_tensor_190 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_694, 256, '0');  convert_element_type_694 = None
        wait_tensor_190 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_190);  all_gather_into_tensor_190 = None
        convert_element_type_695 = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 2)
        mean_42 = torch.ops.aten.mean.dim(pow_43, [2], True);  pow_43 = None
        add_84 = torch.ops.aten.add.Scalar(mean_42, 1e-05);  mean_42 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_168 = torch.ops.aten.mul.Tensor(convert_element_type_695, rsqrt_42);  convert_element_type_695 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, wait_tensor_190)
        convert_element_type_696 = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        view_717 = torch.ops.aten.view.default(convert_element_type_696, [16384, 4096]);  convert_element_type_696 = None
        view_718 = torch.ops.aten.view.default(mm_147, [2, 8192, 4096]);  mm_147 = None
        convert_element_type_700 = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        all_gather_into_tensor_192 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_700, 256, '0');  convert_element_type_700 = None
        wait_tensor_192 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_192);  all_gather_into_tensor_192 = None
        permute_232 = torch.ops.aten.permute.default(wait_tensor_192, [1, 0]);  wait_tensor_192 = None
        mm_148 = torch.ops.aten.mm.default(view_717, permute_232)
        view_721 = torch.ops.aten.view.default(mm_148, [2, 8192, 1024]);  mm_148 = None
        view_724 = torch.ops.aten.view.default(mm_149, [2, 8192, 1024]);  mm_149 = None
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
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_693, permute_234, permute_235, permute_236, getitem_189, getitem_190, getitem_195, getitem_196, None, None, None, 8192, 8192, 0.0, True);  permute_693 = permute_234 = permute_235 = permute_236 = getitem_189 = getitem_190 = getitem_195 = getitem_196 = None
        getitem_318 = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_319 = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_320 = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None
        permute_694 = torch.ops.aten.permute.default(getitem_320, [0, 2, 1, 3]);  getitem_320 = None
        permute_695 = torch.ops.aten.permute.default(getitem_319, [0, 2, 1, 3]);  getitem_319 = None
        permute_696 = torch.ops.aten.permute.default(getitem_318, [0, 2, 1, 3]);  getitem_318 = None
        view_1344 = torch.ops.aten.view.default(permute_694, [2, 8192, 8, 4, 128]);  permute_694 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(view_1344, [3], True);  view_1344 = None
        squeeze_20 = torch.ops.aten.squeeze.dim(sum_65, 3);  sum_65 = None
        view_1345 = torch.ops.aten.view.default(permute_695, [2, 8192, 8, 4, 128]);  permute_695 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(view_1345, [3], True);  view_1345 = None
        squeeze_21 = torch.ops.aten.squeeze.dim(sum_66, 3);  sum_66 = None
        convert_element_type_1643 = torch.ops.prims.convert_element_type.default(squeeze_21, torch.float32);  squeeze_21 = None
        convert_element_type_1644 = torch.ops.prims.convert_element_type.default(permute_696, torch.float32);  permute_696 = None
        view_1346 = torch.ops.aten.view.default(convert_element_type_1643, [2, 8192, 8, 64, 2]);  convert_element_type_1643 = None
        view_as_complex_84 = torch.ops.aten.view_as_complex.default(view_1346);  view_1346 = None
        mul_476 = torch.ops.aten.mul.Tensor(view_as_complex_84, _conj);  view_as_complex_84 = None
        view_1347 = torch.ops.aten.view.default(convert_element_type_1644, [2, 8192, 32, 64, 2]);  convert_element_type_1644 = None
        view_as_complex_85 = torch.ops.aten.view_as_complex.default(view_1347);  view_1347 = None
        mul_477 = torch.ops.aten.mul.Tensor(view_as_complex_85, _conj);  view_as_complex_85 = None
        view_as_real_84 = torch.ops.aten.view_as_real.default(mul_476);  mul_476 = None
        view_1348 = torch.ops.aten.view.default(view_as_real_84, [2, 8192, 8, 128]);  view_as_real_84 = None
        convert_element_type_1645 = torch.ops.prims.convert_element_type.default(view_1348, torch.bfloat16);  view_1348 = None
        view_as_real_85 = torch.ops.aten.view_as_real.default(mul_477);  mul_477 = None
        view_1349 = torch.ops.aten.view.default(view_as_real_85, [2, 8192, 32, 128]);  view_as_real_85 = None
        convert_element_type_1646 = torch.ops.prims.convert_element_type.default(view_1349, torch.bfloat16);  view_1349 = None
        view_1350 = torch.ops.aten.view.default(squeeze_20, [2, 8192, 1024]);  squeeze_20 = None
        view_1351 = torch.ops.aten.view.default(convert_element_type_1645, [2, 8192, 1024]);  convert_element_type_1645 = None
        view_1352 = torch.ops.aten.view.default(convert_element_type_1646, [2, 8192, 4096]);  convert_element_type_1646 = None
        view_1353 = torch.ops.aten.view.default(view_1350, [16384, 1024]);  view_1350 = None
        permute_697 = torch.ops.aten.permute.default(view_1353, [1, 0])
        mm_375 = torch.ops.aten.mm.default(permute_697, view_717);  permute_697 = None
        convert_element_type_703 = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        all_gather_into_tensor_193 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_703, 256, '0');  convert_element_type_703 = None
        wait_tensor_193 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_193);  all_gather_into_tensor_193 = None
        permute_233 = torch.ops.aten.permute.default(wait_tensor_193, [1, 0]);  wait_tensor_193 = None
        permute_699 = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_376 = torch.ops.aten.mm.default(view_1353, permute_699);  view_1353 = permute_699 = None
        view_1354 = torch.ops.aten.view.default(mm_376, [2, 8192, 4096]);  mm_376 = None
        convert_element_type_1651 = torch.ops.prims.convert_element_type.default(mm_375, torch.float32);  mm_375 = None
        reduce_scatter_tensor_97 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1651, 'avg', 256, '0');  convert_element_type_1651 = None
        wait_tensor_388 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_97);  reduce_scatter_tensor_97 = None
        view_1355 = torch.ops.aten.view.default(view_1351, [16384, 1024]);  view_1351 = None
        permute_701 = torch.ops.aten.permute.default(view_1355, [1, 0])
        mm_377 = torch.ops.aten.mm.default(permute_701, view_717);  permute_701 = None
        permute_703 = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        mm_378 = torch.ops.aten.mm.default(view_1355, permute_703);  view_1355 = permute_703 = None
        view_1356 = torch.ops.aten.view.default(mm_378, [2, 8192, 4096]);  mm_378 = None
        add_203 = torch.ops.aten.add.Tensor(view_1354, view_1356);  view_1354 = view_1356 = None
        convert_element_type_1656 = torch.ops.prims.convert_element_type.default(mm_377, torch.float32);  mm_377 = None
        reduce_scatter_tensor_98 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1656, 'avg', 256, '0');  convert_element_type_1656 = None
        wait_tensor_389 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_98);  reduce_scatter_tensor_98 = None
        view_1357 = torch.ops.aten.view.default(view_1352, [16384, 4096]);  view_1352 = None
        permute_705 = torch.ops.aten.permute.default(view_1357, [1, 0])
        mm_379 = torch.ops.aten.mm.default(permute_705, view_717);  permute_705 = view_717 = None
        convert_element_type_697 = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        all_gather_into_tensor_191 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_697, 256, '0');  convert_element_type_697 = None
        wait_tensor_191 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_191);  all_gather_into_tensor_191 = None
        permute_231 = torch.ops.aten.permute.default(wait_tensor_191, [1, 0]);  wait_tensor_191 = None
        permute_707 = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_380 = torch.ops.aten.mm.default(view_1357, permute_707);  view_1357 = permute_707 = None
        view_1358 = torch.ops.aten.view.default(mm_380, [2, 8192, 4096]);  mm_380 = None
        add_204 = torch.ops.aten.add.Tensor(add_203, view_1358);  add_203 = view_1358 = None
        convert_element_type_1661 = torch.ops.prims.convert_element_type.default(mm_379, torch.float32);  mm_379 = None
        reduce_scatter_tensor_99 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1661, 'avg', 256, '0');  convert_element_type_1661 = None
        wait_tensor_390 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_99);  reduce_scatter_tensor_99 = None
        convert_element_type_1662 = torch.ops.prims.convert_element_type.default(add_204, torch.float32);  add_204 = None
        convert_element_type_1664 = torch.ops.prims.convert_element_type.default(wait_tensor_190, torch.float32);  wait_tensor_190 = None
        mul_478 = torch.ops.aten.mul.Tensor(convert_element_type_1662, convert_element_type_1664);  convert_element_type_1664 = None
        mul_480 = torch.ops.aten.mul.Tensor(mul_168, mul_478)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
        div_22 = torch.ops.aten.div.Tensor(mul_168, 4096)
        mul_481 = torch.ops.aten.mul.Tensor(div_22, sum_67);  div_22 = sum_67 = None
        sub_33 = torch.ops.aten.sub.Tensor(mul_478, mul_481);  mul_478 = mul_481 = None
        mul_482 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_42);  sub_33 = rsqrt_42 = None
        mul_483 = torch.ops.aten.mul.Tensor(convert_element_type_1662, mul_168);  convert_element_type_1662 = mul_168 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1]);  mul_483 = None
        convert_element_type_1665 = torch.ops.prims.convert_element_type.default(mul_482, torch.bfloat16);  mul_482 = None
        add_205 = torch.ops.aten.add.Tensor(add_202, convert_element_type_1665);  add_202 = convert_element_type_1665 = None
        convert_element_type_default_43 = torch.ops.prims.convert_element_type.default(sum_68, torch.float32);  sum_68 = None
        reduce_scatter_tensor_100 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_43, 'avg', 256, '0');  convert_element_type_default_43 = None
        wait_tensor_391 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_100);  reduce_scatter_tensor_100 = None
        view_1359 = torch.ops.aten.view.default(add_205, [16384, 4096])
        permute_709 = torch.ops.aten.permute.default(view_1359, [1, 0])
        permute_226 = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3])
        view_701 = torch.ops.aten.view.default(permute_226, [2, 8192, -1]);  permute_226 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        all_gather_into_tensor_185 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_677, 256, '0');  convert_element_type_677 = None
        wait_tensor_185 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_185);  all_gather_into_tensor_185 = None
        permute_227 = torch.ops.aten.permute.default(wait_tensor_185, [1, 0]);  wait_tensor_185 = None
        view_703 = torch.ops.aten.view.default(view_701, [16384, 4096]);  view_701 = None
        mm_143 = torch.ops.aten.mm.default(view_703, permute_227)
        view_704 = torch.ops.aten.view.default(mm_143, [2, 8192, 4096]);  mm_143 = None
        add_81 = torch.ops.aten.add.Tensor(add_79, view_704);  view_704 = None
        convert_element_type_680 = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        all_gather_into_tensor_186 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_680, 256, '0');  convert_element_type_680 = None
        wait_tensor_186 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_186);  all_gather_into_tensor_186 = None
        convert_element_type_681 = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_681, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_42, [2], True);  pow_42 = None
        add_82 = torch.ops.aten.add.Scalar(mean_41, 1e-05);  mean_41 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_164 = torch.ops.aten.mul.Tensor(convert_element_type_681, rsqrt_41);  convert_element_type_681 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, wait_tensor_186)
        convert_element_type_682 = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        view_707 = torch.ops.aten.view.default(convert_element_type_682, [16384, 4096]);  convert_element_type_682 = None
        view_708 = torch.ops.aten.view.default(mm_144, [2, 8192, 14336]);  mm_144 = None
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_708, torch.float32);  view_708 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(convert_element_type_686)
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_686, sigmoid_20);  sigmoid_20 = None
        convert_element_type_687 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        convert_element_type_688 = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        all_gather_into_tensor_188 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_688, 256, '0');  convert_element_type_688 = None
        wait_tensor_188 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_188);  all_gather_into_tensor_188 = None
        permute_229 = torch.ops.aten.permute.default(wait_tensor_188, [1, 0]);  wait_tensor_188 = None
        mm_145 = torch.ops.aten.mm.default(view_707, permute_229)
        view_711 = torch.ops.aten.view.default(mm_145, [2, 8192, 14336]);  mm_145 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_687, view_711)
        view_713 = torch.ops.aten.view.default(mul_167, [16384, 14336]);  mul_167 = None
        mm_381 = torch.ops.aten.mm.default(permute_709, view_713);  permute_709 = view_713 = None
        convert_element_type_691 = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        all_gather_into_tensor_189 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_691, 256, '0');  convert_element_type_691 = None
        wait_tensor_189 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_189);  all_gather_into_tensor_189 = None
        permute_230 = torch.ops.aten.permute.default(wait_tensor_189, [1, 0]);  wait_tensor_189 = None
        permute_711 = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_382 = torch.ops.aten.mm.default(view_1359, permute_711);  view_1359 = permute_711 = None
        view_1360 = torch.ops.aten.view.default(mm_382, [2, 8192, 14336]);  mm_382 = None
        convert_element_type_1672 = torch.ops.prims.convert_element_type.default(mm_381, torch.float32);  mm_381 = None
        reduce_scatter_tensor_101 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1672, 'avg', 256, '0');  convert_element_type_1672 = None
        wait_tensor_392 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_101);  reduce_scatter_tensor_101 = None
        mul_484 = torch.ops.aten.mul.Tensor(view_1360, convert_element_type_687);  convert_element_type_687 = None
        mul_485 = torch.ops.aten.mul.Tensor(view_1360, view_711);  view_1360 = view_711 = None
        view_1361 = torch.ops.aten.view.default(mul_484, [16384, 14336]);  mul_484 = None
        permute_713 = torch.ops.aten.permute.default(view_1361, [1, 0])
        mm_383 = torch.ops.aten.mm.default(permute_713, view_707);  permute_713 = None
        permute_715 = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_384 = torch.ops.aten.mm.default(view_1361, permute_715);  view_1361 = permute_715 = None
        view_1362 = torch.ops.aten.view.default(mm_384, [2, 8192, 4096]);  mm_384 = None
        convert_element_type_1677 = torch.ops.prims.convert_element_type.default(mm_383, torch.float32);  mm_383 = None
        reduce_scatter_tensor_102 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1677, 'avg', 256, '0');  convert_element_type_1677 = None
        wait_tensor_393 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_102);  reduce_scatter_tensor_102 = None
        convert_element_type_1678 = torch.ops.prims.convert_element_type.default(mul_485, torch.float32);  mul_485 = None
        neg_11 = torch.ops.aten.neg.default(convert_element_type_686)
        exp_11 = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_206 = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_206);  add_206 = None
        mul_486 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_487 = torch.ops.aten.mul.Tensor(convert_element_type_1678, mul_486);  convert_element_type_1678 = None
        sub_34 = torch.ops.aten.sub.Tensor(1, mul_486);  mul_486 = None
        mul_488 = torch.ops.aten.mul.Tensor(convert_element_type_686, sub_34);  convert_element_type_686 = sub_34 = None
        add_207 = torch.ops.aten.add.Tensor(mul_488, 1);  mul_488 = None
        mul_489 = torch.ops.aten.mul.Tensor(mul_487, add_207);  mul_487 = add_207 = None
        convert_element_type_1680 = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16);  mul_489 = None
        view_1363 = torch.ops.aten.view.default(convert_element_type_1680, [16384, 14336]);  convert_element_type_1680 = None
        permute_717 = torch.ops.aten.permute.default(view_1363, [1, 0])
        mm_385 = torch.ops.aten.mm.default(permute_717, view_707);  permute_717 = view_707 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        all_gather_into_tensor_187 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_683, 256, '0');  convert_element_type_683 = None
        wait_tensor_187 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_187);  all_gather_into_tensor_187 = None
        permute_228 = torch.ops.aten.permute.default(wait_tensor_187, [1, 0]);  wait_tensor_187 = None
        permute_719 = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_386 = torch.ops.aten.mm.default(view_1363, permute_719);  view_1363 = permute_719 = None
        view_1364 = torch.ops.aten.view.default(mm_386, [2, 8192, 4096]);  mm_386 = None
        add_208 = torch.ops.aten.add.Tensor(view_1362, view_1364);  view_1362 = view_1364 = None
        convert_element_type_1685 = torch.ops.prims.convert_element_type.default(mm_385, torch.float32);  mm_385 = None
        reduce_scatter_tensor_103 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1685, 'avg', 256, '0');  convert_element_type_1685 = None
        wait_tensor_394 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_103);  reduce_scatter_tensor_103 = None
        convert_element_type_1686 = torch.ops.prims.convert_element_type.default(add_208, torch.float32);  add_208 = None
        convert_element_type_1688 = torch.ops.prims.convert_element_type.default(wait_tensor_186, torch.float32);  wait_tensor_186 = None
        mul_490 = torch.ops.aten.mul.Tensor(convert_element_type_1686, convert_element_type_1688);  convert_element_type_1688 = None
        mul_492 = torch.ops.aten.mul.Tensor(mul_164, mul_490)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_492, [2], True);  mul_492 = None
        div_23 = torch.ops.aten.div.Tensor(mul_164, 4096)
        mul_493 = torch.ops.aten.mul.Tensor(div_23, sum_69);  div_23 = sum_69 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_490, mul_493);  mul_490 = mul_493 = None
        mul_494 = torch.ops.aten.mul.Tensor(sub_35, rsqrt_41);  sub_35 = rsqrt_41 = None
        mul_495 = torch.ops.aten.mul.Tensor(convert_element_type_1686, mul_164);  convert_element_type_1686 = mul_164 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_495, [0, 1]);  mul_495 = None
        convert_element_type_1689 = torch.ops.prims.convert_element_type.default(mul_494, torch.bfloat16);  mul_494 = None
        add_209 = torch.ops.aten.add.Tensor(add_205, convert_element_type_1689);  add_205 = convert_element_type_1689 = None
        convert_element_type_default_42 = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None
        reduce_scatter_tensor_104 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_42, 'avg', 256, '0');  convert_element_type_default_42 = None
        wait_tensor_395 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_104);  reduce_scatter_tensor_104 = None
        view_1365 = torch.ops.aten.view.default(add_209, [16384, 4096])
        permute_721 = torch.ops.aten.permute.default(view_1365, [1, 0])
        mm_387 = torch.ops.aten.mm.default(permute_721, view_703);  permute_721 = view_703 = None
        permute_723 = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
        mm_388 = torch.ops.aten.mm.default(view_1365, permute_723);  view_1365 = permute_723 = None
        view_1366 = torch.ops.aten.view.default(mm_388, [2, 8192, 4096]);  mm_388 = None
        convert_element_type_1696 = torch.ops.prims.convert_element_type.default(mm_387, torch.float32);  mm_387 = None
        reduce_scatter_tensor_105 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1696, 'avg', 256, '0');  convert_element_type_1696 = None
        wait_tensor_396 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_105);  reduce_scatter_tensor_105 = None
        view_1367 = torch.ops.aten.view.default(view_1366, [2, 8192, 32, 128]);  view_1366 = None
        permute_725 = torch.ops.aten.permute.default(view_1367, [0, 2, 1, 3]);  view_1367 = None
        convert_element_type_661 = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        all_gather_into_tensor_181 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_661, 256, '0');  convert_element_type_661 = None
        wait_tensor_181 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_181);  all_gather_into_tensor_181 = None
        convert_element_type_662 = torch.ops.prims.convert_element_type.default(add_79, torch.float32);  add_79 = None
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_662, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_41, [2], True);  pow_41 = None
        add_80 = torch.ops.aten.add.Scalar(mean_40, 1e-05);  mean_40 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_662, rsqrt_40);  convert_element_type_662 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, wait_tensor_181)
        convert_element_type_663 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        view_683 = torch.ops.aten.view.default(convert_element_type_663, [16384, 4096]);  convert_element_type_663 = None
        view_684 = torch.ops.aten.view.default(mm_140, [2, 8192, 4096]);  mm_140 = None
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        all_gather_into_tensor_183 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_667, 256, '0');  convert_element_type_667 = None
        wait_tensor_183 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_183);  all_gather_into_tensor_183 = None
        permute_221 = torch.ops.aten.permute.default(wait_tensor_183, [1, 0]);  wait_tensor_183 = None
        mm_141 = torch.ops.aten.mm.default(view_683, permute_221)
        view_687 = torch.ops.aten.view.default(mm_141, [2, 8192, 1024]);  mm_141 = None
        view_690 = torch.ops.aten.view.default(mm_142, [2, 8192, 1024]);  mm_142 = None
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
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_725, permute_223, permute_224, permute_225, getitem_180, getitem_181, getitem_186, getitem_187, None, None, None, 8192, 8192, 0.0, True);  permute_725 = permute_223 = permute_224 = permute_225 = getitem_180 = getitem_181 = getitem_186 = getitem_187 = None
        getitem_321 = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_322 = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_323 = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None
        permute_726 = torch.ops.aten.permute.default(getitem_323, [0, 2, 1, 3]);  getitem_323 = None
        permute_727 = torch.ops.aten.permute.default(getitem_322, [0, 2, 1, 3]);  getitem_322 = None
        permute_728 = torch.ops.aten.permute.default(getitem_321, [0, 2, 1, 3]);  getitem_321 = None
        view_1368 = torch.ops.aten.view.default(permute_726, [2, 8192, 8, 4, 128]);  permute_726 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(view_1368, [3], True);  view_1368 = None
        squeeze_22 = torch.ops.aten.squeeze.dim(sum_71, 3);  sum_71 = None
        view_1369 = torch.ops.aten.view.default(permute_727, [2, 8192, 8, 4, 128]);  permute_727 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_1369, [3], True);  view_1369 = None
        squeeze_23 = torch.ops.aten.squeeze.dim(sum_72, 3);  sum_72 = None
        convert_element_type_1697 = torch.ops.prims.convert_element_type.default(squeeze_23, torch.float32);  squeeze_23 = None
        convert_element_type_1698 = torch.ops.prims.convert_element_type.default(permute_728, torch.float32);  permute_728 = None
        view_1370 = torch.ops.aten.view.default(convert_element_type_1697, [2, 8192, 8, 64, 2]);  convert_element_type_1697 = None
        view_as_complex_86 = torch.ops.aten.view_as_complex.default(view_1370);  view_1370 = None
        mul_496 = torch.ops.aten.mul.Tensor(view_as_complex_86, _conj);  view_as_complex_86 = None
        view_1371 = torch.ops.aten.view.default(convert_element_type_1698, [2, 8192, 32, 64, 2]);  convert_element_type_1698 = None
        view_as_complex_87 = torch.ops.aten.view_as_complex.default(view_1371);  view_1371 = None
        mul_497 = torch.ops.aten.mul.Tensor(view_as_complex_87, _conj);  view_as_complex_87 = None
        view_as_real_86 = torch.ops.aten.view_as_real.default(mul_496);  mul_496 = None
        view_1372 = torch.ops.aten.view.default(view_as_real_86, [2, 8192, 8, 128]);  view_as_real_86 = None
        convert_element_type_1699 = torch.ops.prims.convert_element_type.default(view_1372, torch.bfloat16);  view_1372 = None
        view_as_real_87 = torch.ops.aten.view_as_real.default(mul_497);  mul_497 = None
        view_1373 = torch.ops.aten.view.default(view_as_real_87, [2, 8192, 32, 128]);  view_as_real_87 = None
        convert_element_type_1700 = torch.ops.prims.convert_element_type.default(view_1373, torch.bfloat16);  view_1373 = None
        view_1374 = torch.ops.aten.view.default(squeeze_22, [2, 8192, 1024]);  squeeze_22 = None
        view_1375 = torch.ops.aten.view.default(convert_element_type_1699, [2, 8192, 1024]);  convert_element_type_1699 = None
        view_1376 = torch.ops.aten.view.default(convert_element_type_1700, [2, 8192, 4096]);  convert_element_type_1700 = None
        view_1377 = torch.ops.aten.view.default(view_1374, [16384, 1024]);  view_1374 = None
        permute_729 = torch.ops.aten.permute.default(view_1377, [1, 0])
        mm_389 = torch.ops.aten.mm.default(permute_729, view_683);  permute_729 = None
        convert_element_type_670 = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        all_gather_into_tensor_184 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_670, 256, '0');  convert_element_type_670 = None
        wait_tensor_184 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_184);  all_gather_into_tensor_184 = None
        permute_222 = torch.ops.aten.permute.default(wait_tensor_184, [1, 0]);  wait_tensor_184 = None
        permute_731 = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_390 = torch.ops.aten.mm.default(view_1377, permute_731);  view_1377 = permute_731 = None
        view_1378 = torch.ops.aten.view.default(mm_390, [2, 8192, 4096]);  mm_390 = None
        convert_element_type_1705 = torch.ops.prims.convert_element_type.default(mm_389, torch.float32);  mm_389 = None
        reduce_scatter_tensor_106 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1705, 'avg', 256, '0');  convert_element_type_1705 = None
        wait_tensor_397 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_106);  reduce_scatter_tensor_106 = None
        view_1379 = torch.ops.aten.view.default(view_1375, [16384, 1024]);  view_1375 = None
        permute_733 = torch.ops.aten.permute.default(view_1379, [1, 0])
        mm_391 = torch.ops.aten.mm.default(permute_733, view_683);  permute_733 = None
        permute_735 = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        mm_392 = torch.ops.aten.mm.default(view_1379, permute_735);  view_1379 = permute_735 = None
        view_1380 = torch.ops.aten.view.default(mm_392, [2, 8192, 4096]);  mm_392 = None
        add_210 = torch.ops.aten.add.Tensor(view_1378, view_1380);  view_1378 = view_1380 = None
        convert_element_type_1710 = torch.ops.prims.convert_element_type.default(mm_391, torch.float32);  mm_391 = None
        reduce_scatter_tensor_107 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1710, 'avg', 256, '0');  convert_element_type_1710 = None
        wait_tensor_398 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_107);  reduce_scatter_tensor_107 = None
        view_1381 = torch.ops.aten.view.default(view_1376, [16384, 4096]);  view_1376 = None
        permute_737 = torch.ops.aten.permute.default(view_1381, [1, 0])
        mm_393 = torch.ops.aten.mm.default(permute_737, view_683);  permute_737 = view_683 = None
        convert_element_type_664 = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        all_gather_into_tensor_182 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_664, 256, '0');  convert_element_type_664 = None
        wait_tensor_182 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_182);  all_gather_into_tensor_182 = None
        permute_220 = torch.ops.aten.permute.default(wait_tensor_182, [1, 0]);  wait_tensor_182 = None
        permute_739 = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_394 = torch.ops.aten.mm.default(view_1381, permute_739);  view_1381 = permute_739 = None
        view_1382 = torch.ops.aten.view.default(mm_394, [2, 8192, 4096]);  mm_394 = None
        add_211 = torch.ops.aten.add.Tensor(add_210, view_1382);  add_210 = view_1382 = None
        convert_element_type_1715 = torch.ops.prims.convert_element_type.default(mm_393, torch.float32);  mm_393 = None
        reduce_scatter_tensor_108 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1715, 'avg', 256, '0');  convert_element_type_1715 = None
        wait_tensor_399 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_108);  reduce_scatter_tensor_108 = None
        convert_element_type_1716 = torch.ops.prims.convert_element_type.default(add_211, torch.float32);  add_211 = None
        convert_element_type_1718 = torch.ops.prims.convert_element_type.default(wait_tensor_181, torch.float32);  wait_tensor_181 = None
        mul_498 = torch.ops.aten.mul.Tensor(convert_element_type_1716, convert_element_type_1718);  convert_element_type_1718 = None
        mul_500 = torch.ops.aten.mul.Tensor(mul_160, mul_498)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
        div_24 = torch.ops.aten.div.Tensor(mul_160, 4096)
        mul_501 = torch.ops.aten.mul.Tensor(div_24, sum_73);  div_24 = sum_73 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_498, mul_501);  mul_498 = mul_501 = None
        mul_502 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_40);  sub_36 = rsqrt_40 = None
        mul_503 = torch.ops.aten.mul.Tensor(convert_element_type_1716, mul_160);  convert_element_type_1716 = mul_160 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
        convert_element_type_1719 = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None
        add_212 = torch.ops.aten.add.Tensor(add_209, convert_element_type_1719);  add_209 = convert_element_type_1719 = None
        convert_element_type_default_41 = torch.ops.prims.convert_element_type.default(sum_74, torch.float32);  sum_74 = None
        reduce_scatter_tensor_109 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_41, 'avg', 256, '0');  convert_element_type_default_41 = None
        wait_tensor_400 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_109);  reduce_scatter_tensor_109 = None
        view_1383 = torch.ops.aten.view.default(add_212, [16384, 4096])
        permute_741 = torch.ops.aten.permute.default(view_1383, [1, 0])
        permute_215 = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3])
        view_667 = torch.ops.aten.view.default(permute_215, [2, 8192, -1]);  permute_215 = None
        convert_element_type_644 = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        all_gather_into_tensor_176 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_644, 256, '0');  convert_element_type_644 = None
        wait_tensor_176 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_176);  all_gather_into_tensor_176 = None
        permute_216 = torch.ops.aten.permute.default(wait_tensor_176, [1, 0]);  wait_tensor_176 = None
        view_669 = torch.ops.aten.view.default(view_667, [16384, 4096]);  view_667 = None
        mm_136 = torch.ops.aten.mm.default(view_669, permute_216)
        view_670 = torch.ops.aten.view.default(mm_136, [2, 8192, 4096]);  mm_136 = None
        add_77 = torch.ops.aten.add.Tensor(add_75, view_670);  view_670 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        all_gather_into_tensor_177 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_647, 256, '0');  convert_element_type_647 = None
        wait_tensor_177 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_177);  all_gather_into_tensor_177 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_648, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_40, [2], True);  pow_40 = None
        add_78 = torch.ops.aten.add.Scalar(mean_39, 1e-05);  mean_39 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_156 = torch.ops.aten.mul.Tensor(convert_element_type_648, rsqrt_39);  convert_element_type_648 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, wait_tensor_177)
        convert_element_type_649 = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        view_673 = torch.ops.aten.view.default(convert_element_type_649, [16384, 4096]);  convert_element_type_649 = None
        view_674 = torch.ops.aten.view.default(mm_137, [2, 8192, 14336]);  mm_137 = None
        convert_element_type_653 = torch.ops.prims.convert_element_type.default(view_674, torch.float32);  view_674 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(convert_element_type_653)
        mul_158 = torch.ops.aten.mul.Tensor(convert_element_type_653, sigmoid_19);  sigmoid_19 = None
        convert_element_type_654 = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        convert_element_type_655 = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        all_gather_into_tensor_179 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_655, 256, '0');  convert_element_type_655 = None
        wait_tensor_179 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_179);  all_gather_into_tensor_179 = None
        permute_218 = torch.ops.aten.permute.default(wait_tensor_179, [1, 0]);  wait_tensor_179 = None
        mm_138 = torch.ops.aten.mm.default(view_673, permute_218)
        view_677 = torch.ops.aten.view.default(mm_138, [2, 8192, 14336]);  mm_138 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_654, view_677)
        view_679 = torch.ops.aten.view.default(mul_159, [16384, 14336]);  mul_159 = None
        mm_395 = torch.ops.aten.mm.default(permute_741, view_679);  permute_741 = view_679 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        all_gather_into_tensor_180 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_658, 256, '0');  convert_element_type_658 = None
        wait_tensor_180 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_180);  all_gather_into_tensor_180 = None
        permute_219 = torch.ops.aten.permute.default(wait_tensor_180, [1, 0]);  wait_tensor_180 = None
        permute_743 = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_396 = torch.ops.aten.mm.default(view_1383, permute_743);  view_1383 = permute_743 = None
        view_1384 = torch.ops.aten.view.default(mm_396, [2, 8192, 14336]);  mm_396 = None
        convert_element_type_1726 = torch.ops.prims.convert_element_type.default(mm_395, torch.float32);  mm_395 = None
        reduce_scatter_tensor_110 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1726, 'avg', 256, '0');  convert_element_type_1726 = None
        wait_tensor_401 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_110);  reduce_scatter_tensor_110 = None
        mul_504 = torch.ops.aten.mul.Tensor(view_1384, convert_element_type_654);  convert_element_type_654 = None
        mul_505 = torch.ops.aten.mul.Tensor(view_1384, view_677);  view_1384 = view_677 = None
        view_1385 = torch.ops.aten.view.default(mul_504, [16384, 14336]);  mul_504 = None
        permute_745 = torch.ops.aten.permute.default(view_1385, [1, 0])
        mm_397 = torch.ops.aten.mm.default(permute_745, view_673);  permute_745 = None
        permute_747 = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_398 = torch.ops.aten.mm.default(view_1385, permute_747);  view_1385 = permute_747 = None
        view_1386 = torch.ops.aten.view.default(mm_398, [2, 8192, 4096]);  mm_398 = None
        convert_element_type_1731 = torch.ops.prims.convert_element_type.default(mm_397, torch.float32);  mm_397 = None
        reduce_scatter_tensor_111 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1731, 'avg', 256, '0');  convert_element_type_1731 = None
        wait_tensor_402 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_111);  reduce_scatter_tensor_111 = None
        convert_element_type_1732 = torch.ops.prims.convert_element_type.default(mul_505, torch.float32);  mul_505 = None
        neg_12 = torch.ops.aten.neg.default(convert_element_type_653)
        exp_12 = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_213 = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(add_213);  add_213 = None
        mul_506 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_507 = torch.ops.aten.mul.Tensor(convert_element_type_1732, mul_506);  convert_element_type_1732 = None
        sub_37 = torch.ops.aten.sub.Tensor(1, mul_506);  mul_506 = None
        mul_508 = torch.ops.aten.mul.Tensor(convert_element_type_653, sub_37);  convert_element_type_653 = sub_37 = None
        add_214 = torch.ops.aten.add.Tensor(mul_508, 1);  mul_508 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_507, add_214);  mul_507 = add_214 = None
        convert_element_type_1734 = torch.ops.prims.convert_element_type.default(mul_509, torch.bfloat16);  mul_509 = None
        view_1387 = torch.ops.aten.view.default(convert_element_type_1734, [16384, 14336]);  convert_element_type_1734 = None
        permute_749 = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_399 = torch.ops.aten.mm.default(permute_749, view_673);  permute_749 = view_673 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        all_gather_into_tensor_178 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_650, 256, '0');  convert_element_type_650 = None
        wait_tensor_178 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_178);  all_gather_into_tensor_178 = None
        permute_217 = torch.ops.aten.permute.default(wait_tensor_178, [1, 0]);  wait_tensor_178 = None
        permute_751 = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_400 = torch.ops.aten.mm.default(view_1387, permute_751);  view_1387 = permute_751 = None
        view_1388 = torch.ops.aten.view.default(mm_400, [2, 8192, 4096]);  mm_400 = None
        add_215 = torch.ops.aten.add.Tensor(view_1386, view_1388);  view_1386 = view_1388 = None
        convert_element_type_1739 = torch.ops.prims.convert_element_type.default(mm_399, torch.float32);  mm_399 = None
        reduce_scatter_tensor_112 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1739, 'avg', 256, '0');  convert_element_type_1739 = None
        wait_tensor_403 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_112);  reduce_scatter_tensor_112 = None
        convert_element_type_1740 = torch.ops.prims.convert_element_type.default(add_215, torch.float32);  add_215 = None
        convert_element_type_1742 = torch.ops.prims.convert_element_type.default(wait_tensor_177, torch.float32);  wait_tensor_177 = None
        mul_510 = torch.ops.aten.mul.Tensor(convert_element_type_1740, convert_element_type_1742);  convert_element_type_1742 = None
        mul_512 = torch.ops.aten.mul.Tensor(mul_156, mul_510)
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_512, [2], True);  mul_512 = None
        div_25 = torch.ops.aten.div.Tensor(mul_156, 4096)
        mul_513 = torch.ops.aten.mul.Tensor(div_25, sum_75);  div_25 = sum_75 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_510, mul_513);  mul_510 = mul_513 = None
        mul_514 = torch.ops.aten.mul.Tensor(sub_38, rsqrt_39);  sub_38 = rsqrt_39 = None
        mul_515 = torch.ops.aten.mul.Tensor(convert_element_type_1740, mul_156);  convert_element_type_1740 = mul_156 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_515, [0, 1]);  mul_515 = None
        convert_element_type_1743 = torch.ops.prims.convert_element_type.default(mul_514, torch.bfloat16);  mul_514 = None
        add_216 = torch.ops.aten.add.Tensor(add_212, convert_element_type_1743);  add_212 = convert_element_type_1743 = None
        convert_element_type_default_40 = torch.ops.prims.convert_element_type.default(sum_76, torch.float32);  sum_76 = None
        reduce_scatter_tensor_113 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_40, 'avg', 256, '0');  convert_element_type_default_40 = None
        wait_tensor_404 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_113);  reduce_scatter_tensor_113 = None
        view_1389 = torch.ops.aten.view.default(add_216, [16384, 4096])
        permute_753 = torch.ops.aten.permute.default(view_1389, [1, 0])
        mm_401 = torch.ops.aten.mm.default(permute_753, view_669);  permute_753 = view_669 = None
        permute_755 = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        mm_402 = torch.ops.aten.mm.default(view_1389, permute_755);  view_1389 = permute_755 = None
        view_1390 = torch.ops.aten.view.default(mm_402, [2, 8192, 4096]);  mm_402 = None
        convert_element_type_1750 = torch.ops.prims.convert_element_type.default(mm_401, torch.float32);  mm_401 = None
        reduce_scatter_tensor_114 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1750, 'avg', 256, '0');  convert_element_type_1750 = None
        wait_tensor_405 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_114);  reduce_scatter_tensor_114 = None
        view_1391 = torch.ops.aten.view.default(view_1390, [2, 8192, 32, 128]);  view_1390 = None
        permute_757 = torch.ops.aten.permute.default(view_1391, [0, 2, 1, 3]);  view_1391 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        all_gather_into_tensor_172 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_628, 256, '0');  convert_element_type_628 = None
        wait_tensor_172 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_172);  all_gather_into_tensor_172 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_629, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_39, [2], True);  pow_39 = None
        add_76 = torch.ops.aten.add.Scalar(mean_38, 1e-05);  mean_38 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_629, rsqrt_38);  convert_element_type_629 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, wait_tensor_172)
        convert_element_type_630 = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None
        view_649 = torch.ops.aten.view.default(convert_element_type_630, [16384, 4096]);  convert_element_type_630 = None
        view_650 = torch.ops.aten.view.default(mm_133, [2, 8192, 4096]);  mm_133 = None
        convert_element_type_634 = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        all_gather_into_tensor_174 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_634, 256, '0');  convert_element_type_634 = None
        wait_tensor_174 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_174);  all_gather_into_tensor_174 = None
        permute_210 = torch.ops.aten.permute.default(wait_tensor_174, [1, 0]);  wait_tensor_174 = None
        mm_134 = torch.ops.aten.mm.default(view_649, permute_210)
        view_653 = torch.ops.aten.view.default(mm_134, [2, 8192, 1024]);  mm_134 = None
        view_656 = torch.ops.aten.view.default(mm_135, [2, 8192, 1024]);  mm_135 = None
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
        _scaled_dot_product_cudnn_attention_backward_12 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_757, permute_212, permute_213, permute_214, getitem_171, getitem_172, getitem_177, getitem_178, None, None, None, 8192, 8192, 0.0, True);  permute_757 = permute_212 = permute_213 = permute_214 = getitem_171 = getitem_172 = getitem_177 = getitem_178 = None
        getitem_324 = _scaled_dot_product_cudnn_attention_backward_12[0]
        getitem_325 = _scaled_dot_product_cudnn_attention_backward_12[1]
        getitem_326 = _scaled_dot_product_cudnn_attention_backward_12[2];  _scaled_dot_product_cudnn_attention_backward_12 = None
        permute_758 = torch.ops.aten.permute.default(getitem_326, [0, 2, 1, 3]);  getitem_326 = None
        permute_759 = torch.ops.aten.permute.default(getitem_325, [0, 2, 1, 3]);  getitem_325 = None
        permute_760 = torch.ops.aten.permute.default(getitem_324, [0, 2, 1, 3]);  getitem_324 = None
        view_1392 = torch.ops.aten.view.default(permute_758, [2, 8192, 8, 4, 128]);  permute_758 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(view_1392, [3], True);  view_1392 = None
        squeeze_24 = torch.ops.aten.squeeze.dim(sum_77, 3);  sum_77 = None
        view_1393 = torch.ops.aten.view.default(permute_759, [2, 8192, 8, 4, 128]);  permute_759 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(view_1393, [3], True);  view_1393 = None
        squeeze_25 = torch.ops.aten.squeeze.dim(sum_78, 3);  sum_78 = None
        convert_element_type_1751 = torch.ops.prims.convert_element_type.default(squeeze_25, torch.float32);  squeeze_25 = None
        convert_element_type_1752 = torch.ops.prims.convert_element_type.default(permute_760, torch.float32);  permute_760 = None
        view_1394 = torch.ops.aten.view.default(convert_element_type_1751, [2, 8192, 8, 64, 2]);  convert_element_type_1751 = None
        view_as_complex_88 = torch.ops.aten.view_as_complex.default(view_1394);  view_1394 = None
        mul_516 = torch.ops.aten.mul.Tensor(view_as_complex_88, _conj);  view_as_complex_88 = None
        view_1395 = torch.ops.aten.view.default(convert_element_type_1752, [2, 8192, 32, 64, 2]);  convert_element_type_1752 = None
        view_as_complex_89 = torch.ops.aten.view_as_complex.default(view_1395);  view_1395 = None
        mul_517 = torch.ops.aten.mul.Tensor(view_as_complex_89, _conj);  view_as_complex_89 = None
        view_as_real_88 = torch.ops.aten.view_as_real.default(mul_516);  mul_516 = None
        view_1396 = torch.ops.aten.view.default(view_as_real_88, [2, 8192, 8, 128]);  view_as_real_88 = None
        convert_element_type_1753 = torch.ops.prims.convert_element_type.default(view_1396, torch.bfloat16);  view_1396 = None
        view_as_real_89 = torch.ops.aten.view_as_real.default(mul_517);  mul_517 = None
        view_1397 = torch.ops.aten.view.default(view_as_real_89, [2, 8192, 32, 128]);  view_as_real_89 = None
        convert_element_type_1754 = torch.ops.prims.convert_element_type.default(view_1397, torch.bfloat16);  view_1397 = None
        view_1398 = torch.ops.aten.view.default(squeeze_24, [2, 8192, 1024]);  squeeze_24 = None
        view_1399 = torch.ops.aten.view.default(convert_element_type_1753, [2, 8192, 1024]);  convert_element_type_1753 = None
        view_1400 = torch.ops.aten.view.default(convert_element_type_1754, [2, 8192, 4096]);  convert_element_type_1754 = None
        view_1401 = torch.ops.aten.view.default(view_1398, [16384, 1024]);  view_1398 = None
        permute_761 = torch.ops.aten.permute.default(view_1401, [1, 0])
        mm_403 = torch.ops.aten.mm.default(permute_761, view_649);  permute_761 = None
        convert_element_type_637 = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        all_gather_into_tensor_175 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_637, 256, '0');  convert_element_type_637 = None
        wait_tensor_175 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_175);  all_gather_into_tensor_175 = None
        permute_211 = torch.ops.aten.permute.default(wait_tensor_175, [1, 0]);  wait_tensor_175 = None
        permute_763 = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_404 = torch.ops.aten.mm.default(view_1401, permute_763);  view_1401 = permute_763 = None
        view_1402 = torch.ops.aten.view.default(mm_404, [2, 8192, 4096]);  mm_404 = None
        convert_element_type_1759 = torch.ops.prims.convert_element_type.default(mm_403, torch.float32);  mm_403 = None
        reduce_scatter_tensor_115 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1759, 'avg', 256, '0');  convert_element_type_1759 = None
        wait_tensor_406 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_115);  reduce_scatter_tensor_115 = None
        view_1403 = torch.ops.aten.view.default(view_1399, [16384, 1024]);  view_1399 = None
        permute_765 = torch.ops.aten.permute.default(view_1403, [1, 0])
        mm_405 = torch.ops.aten.mm.default(permute_765, view_649);  permute_765 = None
        permute_767 = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        mm_406 = torch.ops.aten.mm.default(view_1403, permute_767);  view_1403 = permute_767 = None
        view_1404 = torch.ops.aten.view.default(mm_406, [2, 8192, 4096]);  mm_406 = None
        add_217 = torch.ops.aten.add.Tensor(view_1402, view_1404);  view_1402 = view_1404 = None
        convert_element_type_1764 = torch.ops.prims.convert_element_type.default(mm_405, torch.float32);  mm_405 = None
        reduce_scatter_tensor_116 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1764, 'avg', 256, '0');  convert_element_type_1764 = None
        wait_tensor_407 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_116);  reduce_scatter_tensor_116 = None
        view_1405 = torch.ops.aten.view.default(view_1400, [16384, 4096]);  view_1400 = None
        permute_769 = torch.ops.aten.permute.default(view_1405, [1, 0])
        mm_407 = torch.ops.aten.mm.default(permute_769, view_649);  permute_769 = view_649 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        all_gather_into_tensor_173 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_631, 256, '0');  convert_element_type_631 = None
        wait_tensor_173 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_173);  all_gather_into_tensor_173 = None
        permute_209 = torch.ops.aten.permute.default(wait_tensor_173, [1, 0]);  wait_tensor_173 = None
        permute_771 = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_408 = torch.ops.aten.mm.default(view_1405, permute_771);  view_1405 = permute_771 = None
        view_1406 = torch.ops.aten.view.default(mm_408, [2, 8192, 4096]);  mm_408 = None
        add_218 = torch.ops.aten.add.Tensor(add_217, view_1406);  add_217 = view_1406 = None
        convert_element_type_1769 = torch.ops.prims.convert_element_type.default(mm_407, torch.float32);  mm_407 = None
        reduce_scatter_tensor_117 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1769, 'avg', 256, '0');  convert_element_type_1769 = None
        wait_tensor_408 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_117);  reduce_scatter_tensor_117 = None
        convert_element_type_1770 = torch.ops.prims.convert_element_type.default(add_218, torch.float32);  add_218 = None
        convert_element_type_1772 = torch.ops.prims.convert_element_type.default(wait_tensor_172, torch.float32);  wait_tensor_172 = None
        mul_518 = torch.ops.aten.mul.Tensor(convert_element_type_1770, convert_element_type_1772);  convert_element_type_1772 = None
        mul_520 = torch.ops.aten.mul.Tensor(mul_152, mul_518)
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_520, [2], True);  mul_520 = None
        div_26 = torch.ops.aten.div.Tensor(mul_152, 4096)
        mul_521 = torch.ops.aten.mul.Tensor(div_26, sum_79);  div_26 = sum_79 = None
        sub_39 = torch.ops.aten.sub.Tensor(mul_518, mul_521);  mul_518 = mul_521 = None
        mul_522 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_38);  sub_39 = rsqrt_38 = None
        mul_523 = torch.ops.aten.mul.Tensor(convert_element_type_1770, mul_152);  convert_element_type_1770 = mul_152 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_523, [0, 1]);  mul_523 = None
        convert_element_type_1773 = torch.ops.prims.convert_element_type.default(mul_522, torch.bfloat16);  mul_522 = None
        add_219 = torch.ops.aten.add.Tensor(add_216, convert_element_type_1773);  add_216 = convert_element_type_1773 = None
        convert_element_type_default_39 = torch.ops.prims.convert_element_type.default(sum_80, torch.float32);  sum_80 = None
        reduce_scatter_tensor_118 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_39, 'avg', 256, '0');  convert_element_type_default_39 = None
        wait_tensor_409 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_118);  reduce_scatter_tensor_118 = None
        view_1407 = torch.ops.aten.view.default(add_219, [16384, 4096])
        permute_773 = torch.ops.aten.permute.default(view_1407, [1, 0])
        permute_204 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3])
        view_633 = torch.ops.aten.view.default(permute_204, [2, 8192, -1]);  permute_204 = None
        convert_element_type_611 = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        all_gather_into_tensor_167 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_611, 256, '0');  convert_element_type_611 = None
        wait_tensor_167 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_167);  all_gather_into_tensor_167 = None
        permute_205 = torch.ops.aten.permute.default(wait_tensor_167, [1, 0]);  wait_tensor_167 = None
        view_635 = torch.ops.aten.view.default(view_633, [16384, 4096]);  view_633 = None
        mm_129 = torch.ops.aten.mm.default(view_635, permute_205)
        view_636 = torch.ops.aten.view.default(mm_129, [2, 8192, 4096]);  mm_129 = None
        add_73 = torch.ops.aten.add.Tensor(add_71, view_636);  view_636 = None
        convert_element_type_614 = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        all_gather_into_tensor_168 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_614, 256, '0');  convert_element_type_614 = None
        wait_tensor_168 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_168);  all_gather_into_tensor_168 = None
        convert_element_type_615 = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_615, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_38, [2], True);  pow_38 = None
        add_74 = torch.ops.aten.add.Scalar(mean_37, 1e-05);  mean_37 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_148 = torch.ops.aten.mul.Tensor(convert_element_type_615, rsqrt_37);  convert_element_type_615 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, wait_tensor_168)
        convert_element_type_616 = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        view_639 = torch.ops.aten.view.default(convert_element_type_616, [16384, 4096]);  convert_element_type_616 = None
        view_640 = torch.ops.aten.view.default(mm_130, [2, 8192, 14336]);  mm_130 = None
        convert_element_type_620 = torch.ops.prims.convert_element_type.default(view_640, torch.float32);  view_640 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(convert_element_type_620)
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_620, sigmoid_18);  sigmoid_18 = None
        convert_element_type_621 = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        all_gather_into_tensor_170 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_622, 256, '0');  convert_element_type_622 = None
        wait_tensor_170 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_170);  all_gather_into_tensor_170 = None
        permute_207 = torch.ops.aten.permute.default(wait_tensor_170, [1, 0]);  wait_tensor_170 = None
        mm_131 = torch.ops.aten.mm.default(view_639, permute_207)
        view_643 = torch.ops.aten.view.default(mm_131, [2, 8192, 14336]);  mm_131 = None
        mul_151 = torch.ops.aten.mul.Tensor(convert_element_type_621, view_643)
        view_645 = torch.ops.aten.view.default(mul_151, [16384, 14336]);  mul_151 = None
        mm_409 = torch.ops.aten.mm.default(permute_773, view_645);  permute_773 = view_645 = None
        convert_element_type_625 = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        all_gather_into_tensor_171 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_625, 256, '0');  convert_element_type_625 = None
        wait_tensor_171 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_171);  all_gather_into_tensor_171 = None
        permute_208 = torch.ops.aten.permute.default(wait_tensor_171, [1, 0]);  wait_tensor_171 = None
        permute_775 = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_410 = torch.ops.aten.mm.default(view_1407, permute_775);  view_1407 = permute_775 = None
        view_1408 = torch.ops.aten.view.default(mm_410, [2, 8192, 14336]);  mm_410 = None
        convert_element_type_1780 = torch.ops.prims.convert_element_type.default(mm_409, torch.float32);  mm_409 = None
        reduce_scatter_tensor_119 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1780, 'avg', 256, '0');  convert_element_type_1780 = None
        wait_tensor_410 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_119);  reduce_scatter_tensor_119 = None
        mul_524 = torch.ops.aten.mul.Tensor(view_1408, convert_element_type_621);  convert_element_type_621 = None
        mul_525 = torch.ops.aten.mul.Tensor(view_1408, view_643);  view_1408 = view_643 = None
        view_1409 = torch.ops.aten.view.default(mul_524, [16384, 14336]);  mul_524 = None
        permute_777 = torch.ops.aten.permute.default(view_1409, [1, 0])
        mm_411 = torch.ops.aten.mm.default(permute_777, view_639);  permute_777 = None
        permute_779 = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_412 = torch.ops.aten.mm.default(view_1409, permute_779);  view_1409 = permute_779 = None
        view_1410 = torch.ops.aten.view.default(mm_412, [2, 8192, 4096]);  mm_412 = None
        convert_element_type_1785 = torch.ops.prims.convert_element_type.default(mm_411, torch.float32);  mm_411 = None
        reduce_scatter_tensor_120 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1785, 'avg', 256, '0');  convert_element_type_1785 = None
        wait_tensor_411 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_120);  reduce_scatter_tensor_120 = None
        convert_element_type_1786 = torch.ops.prims.convert_element_type.default(mul_525, torch.float32);  mul_525 = None
        neg_13 = torch.ops.aten.neg.default(convert_element_type_620)
        exp_13 = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_220 = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(add_220);  add_220 = None
        mul_526 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_527 = torch.ops.aten.mul.Tensor(convert_element_type_1786, mul_526);  convert_element_type_1786 = None
        sub_40 = torch.ops.aten.sub.Tensor(1, mul_526);  mul_526 = None
        mul_528 = torch.ops.aten.mul.Tensor(convert_element_type_620, sub_40);  convert_element_type_620 = sub_40 = None
        add_221 = torch.ops.aten.add.Tensor(mul_528, 1);  mul_528 = None
        mul_529 = torch.ops.aten.mul.Tensor(mul_527, add_221);  mul_527 = add_221 = None
        convert_element_type_1788 = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None
        view_1411 = torch.ops.aten.view.default(convert_element_type_1788, [16384, 14336]);  convert_element_type_1788 = None
        permute_781 = torch.ops.aten.permute.default(view_1411, [1, 0])
        mm_413 = torch.ops.aten.mm.default(permute_781, view_639);  permute_781 = view_639 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        all_gather_into_tensor_169 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_617, 256, '0');  convert_element_type_617 = None
        wait_tensor_169 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_169);  all_gather_into_tensor_169 = None
        permute_206 = torch.ops.aten.permute.default(wait_tensor_169, [1, 0]);  wait_tensor_169 = None
        permute_783 = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_414 = torch.ops.aten.mm.default(view_1411, permute_783);  view_1411 = permute_783 = None
        view_1412 = torch.ops.aten.view.default(mm_414, [2, 8192, 4096]);  mm_414 = None
        add_222 = torch.ops.aten.add.Tensor(view_1410, view_1412);  view_1410 = view_1412 = None
        convert_element_type_1793 = torch.ops.prims.convert_element_type.default(mm_413, torch.float32);  mm_413 = None
        reduce_scatter_tensor_121 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1793, 'avg', 256, '0');  convert_element_type_1793 = None
        wait_tensor_412 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_121);  reduce_scatter_tensor_121 = None
        convert_element_type_1794 = torch.ops.prims.convert_element_type.default(add_222, torch.float32);  add_222 = None
        convert_element_type_1796 = torch.ops.prims.convert_element_type.default(wait_tensor_168, torch.float32);  wait_tensor_168 = None
        mul_530 = torch.ops.aten.mul.Tensor(convert_element_type_1794, convert_element_type_1796);  convert_element_type_1796 = None
        mul_532 = torch.ops.aten.mul.Tensor(mul_148, mul_530)
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_532, [2], True);  mul_532 = None
        div_27 = torch.ops.aten.div.Tensor(mul_148, 4096)
        mul_533 = torch.ops.aten.mul.Tensor(div_27, sum_81);  div_27 = sum_81 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_530, mul_533);  mul_530 = mul_533 = None
        mul_534 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_37);  sub_41 = rsqrt_37 = None
        mul_535 = torch.ops.aten.mul.Tensor(convert_element_type_1794, mul_148);  convert_element_type_1794 = mul_148 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_535, [0, 1]);  mul_535 = None
        convert_element_type_1797 = torch.ops.prims.convert_element_type.default(mul_534, torch.bfloat16);  mul_534 = None
        add_223 = torch.ops.aten.add.Tensor(add_219, convert_element_type_1797);  add_219 = convert_element_type_1797 = None
        convert_element_type_default_38 = torch.ops.prims.convert_element_type.default(sum_82, torch.float32);  sum_82 = None
        reduce_scatter_tensor_122 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_38, 'avg', 256, '0');  convert_element_type_default_38 = None
        wait_tensor_413 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_122);  reduce_scatter_tensor_122 = None
        view_1413 = torch.ops.aten.view.default(add_223, [16384, 4096])
        permute_785 = torch.ops.aten.permute.default(view_1413, [1, 0])
        mm_415 = torch.ops.aten.mm.default(permute_785, view_635);  permute_785 = view_635 = None
        permute_787 = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        mm_416 = torch.ops.aten.mm.default(view_1413, permute_787);  view_1413 = permute_787 = None
        view_1414 = torch.ops.aten.view.default(mm_416, [2, 8192, 4096]);  mm_416 = None
        convert_element_type_1804 = torch.ops.prims.convert_element_type.default(mm_415, torch.float32);  mm_415 = None
        reduce_scatter_tensor_123 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1804, 'avg', 256, '0');  convert_element_type_1804 = None
        wait_tensor_414 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_123);  reduce_scatter_tensor_123 = None
        view_1415 = torch.ops.aten.view.default(view_1414, [2, 8192, 32, 128]);  view_1414 = None
        permute_789 = torch.ops.aten.permute.default(view_1415, [0, 2, 1, 3]);  view_1415 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        all_gather_into_tensor_163 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_595, 256, '0');  convert_element_type_595 = None
        wait_tensor_163 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_163);  all_gather_into_tensor_163 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_596, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_37, [2], True);  pow_37 = None
        add_72 = torch.ops.aten.add.Scalar(mean_36, 1e-05);  mean_36 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_596, rsqrt_36);  convert_element_type_596 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, wait_tensor_163)
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        view_615 = torch.ops.aten.view.default(convert_element_type_597, [16384, 4096]);  convert_element_type_597 = None
        view_616 = torch.ops.aten.view.default(mm_126, [2, 8192, 4096]);  mm_126 = None
        convert_element_type_601 = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        all_gather_into_tensor_165 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_601, 256, '0');  convert_element_type_601 = None
        wait_tensor_165 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_165);  all_gather_into_tensor_165 = None
        permute_199 = torch.ops.aten.permute.default(wait_tensor_165, [1, 0]);  wait_tensor_165 = None
        mm_127 = torch.ops.aten.mm.default(view_615, permute_199)
        view_619 = torch.ops.aten.view.default(mm_127, [2, 8192, 1024]);  mm_127 = None
        view_622 = torch.ops.aten.view.default(mm_128, [2, 8192, 1024]);  mm_128 = None
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
        _scaled_dot_product_cudnn_attention_backward_13 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_789, permute_201, permute_202, permute_203, getitem_162, getitem_163, getitem_168, getitem_169, None, None, None, 8192, 8192, 0.0, True);  permute_789 = permute_201 = permute_202 = permute_203 = getitem_162 = getitem_163 = getitem_168 = getitem_169 = None
        getitem_327 = _scaled_dot_product_cudnn_attention_backward_13[0]
        getitem_328 = _scaled_dot_product_cudnn_attention_backward_13[1]
        getitem_329 = _scaled_dot_product_cudnn_attention_backward_13[2];  _scaled_dot_product_cudnn_attention_backward_13 = None
        permute_790 = torch.ops.aten.permute.default(getitem_329, [0, 2, 1, 3]);  getitem_329 = None
        permute_791 = torch.ops.aten.permute.default(getitem_328, [0, 2, 1, 3]);  getitem_328 = None
        permute_792 = torch.ops.aten.permute.default(getitem_327, [0, 2, 1, 3]);  getitem_327 = None
        view_1416 = torch.ops.aten.view.default(permute_790, [2, 8192, 8, 4, 128]);  permute_790 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(view_1416, [3], True);  view_1416 = None
        squeeze_26 = torch.ops.aten.squeeze.dim(sum_83, 3);  sum_83 = None
        view_1417 = torch.ops.aten.view.default(permute_791, [2, 8192, 8, 4, 128]);  permute_791 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(view_1417, [3], True);  view_1417 = None
        squeeze_27 = torch.ops.aten.squeeze.dim(sum_84, 3);  sum_84 = None
        convert_element_type_1805 = torch.ops.prims.convert_element_type.default(squeeze_27, torch.float32);  squeeze_27 = None
        convert_element_type_1806 = torch.ops.prims.convert_element_type.default(permute_792, torch.float32);  permute_792 = None
        view_1418 = torch.ops.aten.view.default(convert_element_type_1805, [2, 8192, 8, 64, 2]);  convert_element_type_1805 = None
        view_as_complex_90 = torch.ops.aten.view_as_complex.default(view_1418);  view_1418 = None
        mul_536 = torch.ops.aten.mul.Tensor(view_as_complex_90, _conj);  view_as_complex_90 = None
        view_1419 = torch.ops.aten.view.default(convert_element_type_1806, [2, 8192, 32, 64, 2]);  convert_element_type_1806 = None
        view_as_complex_91 = torch.ops.aten.view_as_complex.default(view_1419);  view_1419 = None
        mul_537 = torch.ops.aten.mul.Tensor(view_as_complex_91, _conj);  view_as_complex_91 = None
        view_as_real_90 = torch.ops.aten.view_as_real.default(mul_536);  mul_536 = None
        view_1420 = torch.ops.aten.view.default(view_as_real_90, [2, 8192, 8, 128]);  view_as_real_90 = None
        convert_element_type_1807 = torch.ops.prims.convert_element_type.default(view_1420, torch.bfloat16);  view_1420 = None
        view_as_real_91 = torch.ops.aten.view_as_real.default(mul_537);  mul_537 = None
        view_1421 = torch.ops.aten.view.default(view_as_real_91, [2, 8192, 32, 128]);  view_as_real_91 = None
        convert_element_type_1808 = torch.ops.prims.convert_element_type.default(view_1421, torch.bfloat16);  view_1421 = None
        view_1422 = torch.ops.aten.view.default(squeeze_26, [2, 8192, 1024]);  squeeze_26 = None
        view_1423 = torch.ops.aten.view.default(convert_element_type_1807, [2, 8192, 1024]);  convert_element_type_1807 = None
        view_1424 = torch.ops.aten.view.default(convert_element_type_1808, [2, 8192, 4096]);  convert_element_type_1808 = None
        view_1425 = torch.ops.aten.view.default(view_1422, [16384, 1024]);  view_1422 = None
        permute_793 = torch.ops.aten.permute.default(view_1425, [1, 0])
        mm_417 = torch.ops.aten.mm.default(permute_793, view_615);  permute_793 = None
        convert_element_type_604 = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        all_gather_into_tensor_166 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_604, 256, '0');  convert_element_type_604 = None
        wait_tensor_166 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_166);  all_gather_into_tensor_166 = None
        permute_200 = torch.ops.aten.permute.default(wait_tensor_166, [1, 0]);  wait_tensor_166 = None
        permute_795 = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_418 = torch.ops.aten.mm.default(view_1425, permute_795);  view_1425 = permute_795 = None
        view_1426 = torch.ops.aten.view.default(mm_418, [2, 8192, 4096]);  mm_418 = None
        convert_element_type_1813 = torch.ops.prims.convert_element_type.default(mm_417, torch.float32);  mm_417 = None
        reduce_scatter_tensor_124 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1813, 'avg', 256, '0');  convert_element_type_1813 = None
        wait_tensor_415 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_124);  reduce_scatter_tensor_124 = None
        view_1427 = torch.ops.aten.view.default(view_1423, [16384, 1024]);  view_1423 = None
        permute_797 = torch.ops.aten.permute.default(view_1427, [1, 0])
        mm_419 = torch.ops.aten.mm.default(permute_797, view_615);  permute_797 = None
        permute_799 = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        mm_420 = torch.ops.aten.mm.default(view_1427, permute_799);  view_1427 = permute_799 = None
        view_1428 = torch.ops.aten.view.default(mm_420, [2, 8192, 4096]);  mm_420 = None
        add_224 = torch.ops.aten.add.Tensor(view_1426, view_1428);  view_1426 = view_1428 = None
        convert_element_type_1818 = torch.ops.prims.convert_element_type.default(mm_419, torch.float32);  mm_419 = None
        reduce_scatter_tensor_125 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1818, 'avg', 256, '0');  convert_element_type_1818 = None
        wait_tensor_416 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_125);  reduce_scatter_tensor_125 = None
        view_1429 = torch.ops.aten.view.default(view_1424, [16384, 4096]);  view_1424 = None
        permute_801 = torch.ops.aten.permute.default(view_1429, [1, 0])
        mm_421 = torch.ops.aten.mm.default(permute_801, view_615);  permute_801 = view_615 = None
        convert_element_type_598 = torch.ops.prims.convert_element_type.default(primals_167, torch.bfloat16);  primals_167 = None
        all_gather_into_tensor_164 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_598, 256, '0');  convert_element_type_598 = None
        wait_tensor_164 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_164);  all_gather_into_tensor_164 = None
        permute_198 = torch.ops.aten.permute.default(wait_tensor_164, [1, 0]);  wait_tensor_164 = None
        permute_803 = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_422 = torch.ops.aten.mm.default(view_1429, permute_803);  view_1429 = permute_803 = None
        view_1430 = torch.ops.aten.view.default(mm_422, [2, 8192, 4096]);  mm_422 = None
        add_225 = torch.ops.aten.add.Tensor(add_224, view_1430);  add_224 = view_1430 = None
        convert_element_type_1823 = torch.ops.prims.convert_element_type.default(mm_421, torch.float32);  mm_421 = None
        reduce_scatter_tensor_126 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1823, 'avg', 256, '0');  convert_element_type_1823 = None
        wait_tensor_417 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_126);  reduce_scatter_tensor_126 = None
        convert_element_type_1824 = torch.ops.prims.convert_element_type.default(add_225, torch.float32);  add_225 = None
        convert_element_type_1826 = torch.ops.prims.convert_element_type.default(wait_tensor_163, torch.float32);  wait_tensor_163 = None
        mul_538 = torch.ops.aten.mul.Tensor(convert_element_type_1824, convert_element_type_1826);  convert_element_type_1826 = None
        mul_540 = torch.ops.aten.mul.Tensor(mul_144, mul_538)
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_540, [2], True);  mul_540 = None
        div_28 = torch.ops.aten.div.Tensor(mul_144, 4096)
        mul_541 = torch.ops.aten.mul.Tensor(div_28, sum_85);  div_28 = sum_85 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_538, mul_541);  mul_538 = mul_541 = None
        mul_542 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_36);  sub_42 = rsqrt_36 = None
        mul_543 = torch.ops.aten.mul.Tensor(convert_element_type_1824, mul_144);  convert_element_type_1824 = mul_144 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1]);  mul_543 = None
        convert_element_type_1827 = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None
        add_226 = torch.ops.aten.add.Tensor(add_223, convert_element_type_1827);  add_223 = convert_element_type_1827 = None
        convert_element_type_default_37 = torch.ops.prims.convert_element_type.default(sum_86, torch.float32);  sum_86 = None
        reduce_scatter_tensor_127 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_37, 'avg', 256, '0');  convert_element_type_default_37 = None
        wait_tensor_418 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_127);  reduce_scatter_tensor_127 = None
        view_1431 = torch.ops.aten.view.default(add_226, [16384, 4096])
        permute_805 = torch.ops.aten.permute.default(view_1431, [1, 0])
        permute_193 = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3])
        view_599 = torch.ops.aten.view.default(permute_193, [2, 8192, -1]);  permute_193 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        all_gather_into_tensor_158 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_578, 256, '0');  convert_element_type_578 = None
        wait_tensor_158 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_158);  all_gather_into_tensor_158 = None
        permute_194 = torch.ops.aten.permute.default(wait_tensor_158, [1, 0]);  wait_tensor_158 = None
        view_601 = torch.ops.aten.view.default(view_599, [16384, 4096]);  view_599 = None
        mm_122 = torch.ops.aten.mm.default(view_601, permute_194)
        view_602 = torch.ops.aten.view.default(mm_122, [2, 8192, 4096]);  mm_122 = None
        add_69 = torch.ops.aten.add.Tensor(add_67, view_602);  view_602 = None
        convert_element_type_581 = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        all_gather_into_tensor_159 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_581, 256, '0');  convert_element_type_581 = None
        wait_tensor_159 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_159);  all_gather_into_tensor_159 = None
        convert_element_type_582 = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_582, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_36, [2], True);  pow_36 = None
        add_70 = torch.ops.aten.add.Scalar(mean_35, 1e-05);  mean_35 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_582, rsqrt_35);  convert_element_type_582 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, wait_tensor_159)
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        view_605 = torch.ops.aten.view.default(convert_element_type_583, [16384, 4096]);  convert_element_type_583 = None
        view_606 = torch.ops.aten.view.default(mm_123, [2, 8192, 14336]);  mm_123 = None
        convert_element_type_587 = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(convert_element_type_587)
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_587, sigmoid_17);  sigmoid_17 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        convert_element_type_589 = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        all_gather_into_tensor_161 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_589, 256, '0');  convert_element_type_589 = None
        wait_tensor_161 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_161);  all_gather_into_tensor_161 = None
        permute_196 = torch.ops.aten.permute.default(wait_tensor_161, [1, 0]);  wait_tensor_161 = None
        mm_124 = torch.ops.aten.mm.default(view_605, permute_196)
        view_609 = torch.ops.aten.view.default(mm_124, [2, 8192, 14336]);  mm_124 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_588, view_609)
        view_611 = torch.ops.aten.view.default(mul_143, [16384, 14336]);  mul_143 = None
        mm_423 = torch.ops.aten.mm.default(permute_805, view_611);  permute_805 = view_611 = None
        convert_element_type_592 = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        all_gather_into_tensor_162 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_592, 256, '0');  convert_element_type_592 = None
        wait_tensor_162 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_162);  all_gather_into_tensor_162 = None
        permute_197 = torch.ops.aten.permute.default(wait_tensor_162, [1, 0]);  wait_tensor_162 = None
        permute_807 = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_424 = torch.ops.aten.mm.default(view_1431, permute_807);  view_1431 = permute_807 = None
        view_1432 = torch.ops.aten.view.default(mm_424, [2, 8192, 14336]);  mm_424 = None
        convert_element_type_1834 = torch.ops.prims.convert_element_type.default(mm_423, torch.float32);  mm_423 = None
        reduce_scatter_tensor_128 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1834, 'avg', 256, '0');  convert_element_type_1834 = None
        wait_tensor_419 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_128);  reduce_scatter_tensor_128 = None
        mul_544 = torch.ops.aten.mul.Tensor(view_1432, convert_element_type_588);  convert_element_type_588 = None
        mul_545 = torch.ops.aten.mul.Tensor(view_1432, view_609);  view_1432 = view_609 = None
        view_1433 = torch.ops.aten.view.default(mul_544, [16384, 14336]);  mul_544 = None
        permute_809 = torch.ops.aten.permute.default(view_1433, [1, 0])
        mm_425 = torch.ops.aten.mm.default(permute_809, view_605);  permute_809 = None
        permute_811 = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_426 = torch.ops.aten.mm.default(view_1433, permute_811);  view_1433 = permute_811 = None
        view_1434 = torch.ops.aten.view.default(mm_426, [2, 8192, 4096]);  mm_426 = None
        convert_element_type_1839 = torch.ops.prims.convert_element_type.default(mm_425, torch.float32);  mm_425 = None
        reduce_scatter_tensor_129 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1839, 'avg', 256, '0');  convert_element_type_1839 = None
        wait_tensor_420 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_129);  reduce_scatter_tensor_129 = None
        convert_element_type_1840 = torch.ops.prims.convert_element_type.default(mul_545, torch.float32);  mul_545 = None
        neg_14 = torch.ops.aten.neg.default(convert_element_type_587)
        exp_14 = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_227 = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_227);  add_227 = None
        mul_546 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_547 = torch.ops.aten.mul.Tensor(convert_element_type_1840, mul_546);  convert_element_type_1840 = None
        sub_43 = torch.ops.aten.sub.Tensor(1, mul_546);  mul_546 = None
        mul_548 = torch.ops.aten.mul.Tensor(convert_element_type_587, sub_43);  convert_element_type_587 = sub_43 = None
        add_228 = torch.ops.aten.add.Tensor(mul_548, 1);  mul_548 = None
        mul_549 = torch.ops.aten.mul.Tensor(mul_547, add_228);  mul_547 = add_228 = None
        convert_element_type_1842 = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None
        view_1435 = torch.ops.aten.view.default(convert_element_type_1842, [16384, 14336]);  convert_element_type_1842 = None
        permute_813 = torch.ops.aten.permute.default(view_1435, [1, 0])
        mm_427 = torch.ops.aten.mm.default(permute_813, view_605);  permute_813 = view_605 = None
        convert_element_type_584 = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        all_gather_into_tensor_160 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_584, 256, '0');  convert_element_type_584 = None
        wait_tensor_160 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_160);  all_gather_into_tensor_160 = None
        permute_195 = torch.ops.aten.permute.default(wait_tensor_160, [1, 0]);  wait_tensor_160 = None
        permute_815 = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_428 = torch.ops.aten.mm.default(view_1435, permute_815);  view_1435 = permute_815 = None
        view_1436 = torch.ops.aten.view.default(mm_428, [2, 8192, 4096]);  mm_428 = None
        add_229 = torch.ops.aten.add.Tensor(view_1434, view_1436);  view_1434 = view_1436 = None
        convert_element_type_1847 = torch.ops.prims.convert_element_type.default(mm_427, torch.float32);  mm_427 = None
        reduce_scatter_tensor_130 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1847, 'avg', 256, '0');  convert_element_type_1847 = None
        wait_tensor_421 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_130);  reduce_scatter_tensor_130 = None
        convert_element_type_1848 = torch.ops.prims.convert_element_type.default(add_229, torch.float32);  add_229 = None
        convert_element_type_1850 = torch.ops.prims.convert_element_type.default(wait_tensor_159, torch.float32);  wait_tensor_159 = None
        mul_550 = torch.ops.aten.mul.Tensor(convert_element_type_1848, convert_element_type_1850);  convert_element_type_1850 = None
        mul_552 = torch.ops.aten.mul.Tensor(mul_140, mul_550)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_552, [2], True);  mul_552 = None
        div_29 = torch.ops.aten.div.Tensor(mul_140, 4096)
        mul_553 = torch.ops.aten.mul.Tensor(div_29, sum_87);  div_29 = sum_87 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_550, mul_553);  mul_550 = mul_553 = None
        mul_554 = torch.ops.aten.mul.Tensor(sub_44, rsqrt_35);  sub_44 = rsqrt_35 = None
        mul_555 = torch.ops.aten.mul.Tensor(convert_element_type_1848, mul_140);  convert_element_type_1848 = mul_140 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_555, [0, 1]);  mul_555 = None
        convert_element_type_1851 = torch.ops.prims.convert_element_type.default(mul_554, torch.bfloat16);  mul_554 = None
        add_230 = torch.ops.aten.add.Tensor(add_226, convert_element_type_1851);  add_226 = convert_element_type_1851 = None
        convert_element_type_default_36 = torch.ops.prims.convert_element_type.default(sum_88, torch.float32);  sum_88 = None
        reduce_scatter_tensor_131 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_36, 'avg', 256, '0');  convert_element_type_default_36 = None
        wait_tensor_422 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_131);  reduce_scatter_tensor_131 = None
        view_1437 = torch.ops.aten.view.default(add_230, [16384, 4096])
        permute_817 = torch.ops.aten.permute.default(view_1437, [1, 0])
        mm_429 = torch.ops.aten.mm.default(permute_817, view_601);  permute_817 = view_601 = None
        permute_819 = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        mm_430 = torch.ops.aten.mm.default(view_1437, permute_819);  view_1437 = permute_819 = None
        view_1438 = torch.ops.aten.view.default(mm_430, [2, 8192, 4096]);  mm_430 = None
        convert_element_type_1858 = torch.ops.prims.convert_element_type.default(mm_429, torch.float32);  mm_429 = None
        reduce_scatter_tensor_132 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1858, 'avg', 256, '0');  convert_element_type_1858 = None
        wait_tensor_423 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_132);  reduce_scatter_tensor_132 = None
        view_1439 = torch.ops.aten.view.default(view_1438, [2, 8192, 32, 128]);  view_1438 = None
        permute_821 = torch.ops.aten.permute.default(view_1439, [0, 2, 1, 3]);  view_1439 = None
        convert_element_type_562 = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        all_gather_into_tensor_154 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_562, 256, '0');  convert_element_type_562 = None
        wait_tensor_154 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_154);  all_gather_into_tensor_154 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_35, [2], True);  pow_35 = None
        add_68 = torch.ops.aten.add.Scalar(mean_34, 1e-05);  mean_34 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_563, rsqrt_34);  convert_element_type_563 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, wait_tensor_154)
        convert_element_type_564 = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        view_581 = torch.ops.aten.view.default(convert_element_type_564, [16384, 4096]);  convert_element_type_564 = None
        view_582 = torch.ops.aten.view.default(mm_119, [2, 8192, 4096]);  mm_119 = None
        convert_element_type_568 = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        all_gather_into_tensor_156 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_568, 256, '0');  convert_element_type_568 = None
        wait_tensor_156 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_156);  all_gather_into_tensor_156 = None
        permute_188 = torch.ops.aten.permute.default(wait_tensor_156, [1, 0]);  wait_tensor_156 = None
        mm_120 = torch.ops.aten.mm.default(view_581, permute_188)
        view_585 = torch.ops.aten.view.default(mm_120, [2, 8192, 1024]);  mm_120 = None
        view_588 = torch.ops.aten.view.default(mm_121, [2, 8192, 1024]);  mm_121 = None
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
        _scaled_dot_product_cudnn_attention_backward_14 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_821, permute_190, permute_191, permute_192, getitem_153, getitem_154, getitem_159, getitem_160, None, None, None, 8192, 8192, 0.0, True);  permute_821 = permute_190 = permute_191 = permute_192 = getitem_153 = getitem_154 = getitem_159 = getitem_160 = None
        getitem_330 = _scaled_dot_product_cudnn_attention_backward_14[0]
        getitem_331 = _scaled_dot_product_cudnn_attention_backward_14[1]
        getitem_332 = _scaled_dot_product_cudnn_attention_backward_14[2];  _scaled_dot_product_cudnn_attention_backward_14 = None
        permute_822 = torch.ops.aten.permute.default(getitem_332, [0, 2, 1, 3]);  getitem_332 = None
        permute_823 = torch.ops.aten.permute.default(getitem_331, [0, 2, 1, 3]);  getitem_331 = None
        permute_824 = torch.ops.aten.permute.default(getitem_330, [0, 2, 1, 3]);  getitem_330 = None
        view_1440 = torch.ops.aten.view.default(permute_822, [2, 8192, 8, 4, 128]);  permute_822 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(view_1440, [3], True);  view_1440 = None
        squeeze_28 = torch.ops.aten.squeeze.dim(sum_89, 3);  sum_89 = None
        view_1441 = torch.ops.aten.view.default(permute_823, [2, 8192, 8, 4, 128]);  permute_823 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(view_1441, [3], True);  view_1441 = None
        squeeze_29 = torch.ops.aten.squeeze.dim(sum_90, 3);  sum_90 = None
        convert_element_type_1859 = torch.ops.prims.convert_element_type.default(squeeze_29, torch.float32);  squeeze_29 = None
        convert_element_type_1860 = torch.ops.prims.convert_element_type.default(permute_824, torch.float32);  permute_824 = None
        view_1442 = torch.ops.aten.view.default(convert_element_type_1859, [2, 8192, 8, 64, 2]);  convert_element_type_1859 = None
        view_as_complex_92 = torch.ops.aten.view_as_complex.default(view_1442);  view_1442 = None
        mul_556 = torch.ops.aten.mul.Tensor(view_as_complex_92, _conj);  view_as_complex_92 = None
        view_1443 = torch.ops.aten.view.default(convert_element_type_1860, [2, 8192, 32, 64, 2]);  convert_element_type_1860 = None
        view_as_complex_93 = torch.ops.aten.view_as_complex.default(view_1443);  view_1443 = None
        mul_557 = torch.ops.aten.mul.Tensor(view_as_complex_93, _conj);  view_as_complex_93 = None
        view_as_real_92 = torch.ops.aten.view_as_real.default(mul_556);  mul_556 = None
        view_1444 = torch.ops.aten.view.default(view_as_real_92, [2, 8192, 8, 128]);  view_as_real_92 = None
        convert_element_type_1861 = torch.ops.prims.convert_element_type.default(view_1444, torch.bfloat16);  view_1444 = None
        view_as_real_93 = torch.ops.aten.view_as_real.default(mul_557);  mul_557 = None
        view_1445 = torch.ops.aten.view.default(view_as_real_93, [2, 8192, 32, 128]);  view_as_real_93 = None
        convert_element_type_1862 = torch.ops.prims.convert_element_type.default(view_1445, torch.bfloat16);  view_1445 = None
        view_1446 = torch.ops.aten.view.default(squeeze_28, [2, 8192, 1024]);  squeeze_28 = None
        view_1447 = torch.ops.aten.view.default(convert_element_type_1861, [2, 8192, 1024]);  convert_element_type_1861 = None
        view_1448 = torch.ops.aten.view.default(convert_element_type_1862, [2, 8192, 4096]);  convert_element_type_1862 = None
        view_1449 = torch.ops.aten.view.default(view_1446, [16384, 1024]);  view_1446 = None
        permute_825 = torch.ops.aten.permute.default(view_1449, [1, 0])
        mm_431 = torch.ops.aten.mm.default(permute_825, view_581);  permute_825 = None
        convert_element_type_571 = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        all_gather_into_tensor_157 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_571, 256, '0');  convert_element_type_571 = None
        wait_tensor_157 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_157);  all_gather_into_tensor_157 = None
        permute_189 = torch.ops.aten.permute.default(wait_tensor_157, [1, 0]);  wait_tensor_157 = None
        permute_827 = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_432 = torch.ops.aten.mm.default(view_1449, permute_827);  view_1449 = permute_827 = None
        view_1450 = torch.ops.aten.view.default(mm_432, [2, 8192, 4096]);  mm_432 = None
        convert_element_type_1867 = torch.ops.prims.convert_element_type.default(mm_431, torch.float32);  mm_431 = None
        reduce_scatter_tensor_133 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1867, 'avg', 256, '0');  convert_element_type_1867 = None
        wait_tensor_424 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_133);  reduce_scatter_tensor_133 = None
        view_1451 = torch.ops.aten.view.default(view_1447, [16384, 1024]);  view_1447 = None
        permute_829 = torch.ops.aten.permute.default(view_1451, [1, 0])
        mm_433 = torch.ops.aten.mm.default(permute_829, view_581);  permute_829 = None
        permute_831 = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        mm_434 = torch.ops.aten.mm.default(view_1451, permute_831);  view_1451 = permute_831 = None
        view_1452 = torch.ops.aten.view.default(mm_434, [2, 8192, 4096]);  mm_434 = None
        add_231 = torch.ops.aten.add.Tensor(view_1450, view_1452);  view_1450 = view_1452 = None
        convert_element_type_1872 = torch.ops.prims.convert_element_type.default(mm_433, torch.float32);  mm_433 = None
        reduce_scatter_tensor_134 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1872, 'avg', 256, '0');  convert_element_type_1872 = None
        wait_tensor_425 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_134);  reduce_scatter_tensor_134 = None
        view_1453 = torch.ops.aten.view.default(view_1448, [16384, 4096]);  view_1448 = None
        permute_833 = torch.ops.aten.permute.default(view_1453, [1, 0])
        mm_435 = torch.ops.aten.mm.default(permute_833, view_581);  permute_833 = view_581 = None
        convert_element_type_565 = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        all_gather_into_tensor_155 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_565, 256, '0');  convert_element_type_565 = None
        wait_tensor_155 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_155);  all_gather_into_tensor_155 = None
        permute_187 = torch.ops.aten.permute.default(wait_tensor_155, [1, 0]);  wait_tensor_155 = None
        permute_835 = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_436 = torch.ops.aten.mm.default(view_1453, permute_835);  view_1453 = permute_835 = None
        view_1454 = torch.ops.aten.view.default(mm_436, [2, 8192, 4096]);  mm_436 = None
        add_232 = torch.ops.aten.add.Tensor(add_231, view_1454);  add_231 = view_1454 = None
        convert_element_type_1877 = torch.ops.prims.convert_element_type.default(mm_435, torch.float32);  mm_435 = None
        reduce_scatter_tensor_135 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1877, 'avg', 256, '0');  convert_element_type_1877 = None
        wait_tensor_426 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_135);  reduce_scatter_tensor_135 = None
        convert_element_type_1878 = torch.ops.prims.convert_element_type.default(add_232, torch.float32);  add_232 = None
        convert_element_type_1880 = torch.ops.prims.convert_element_type.default(wait_tensor_154, torch.float32);  wait_tensor_154 = None
        mul_558 = torch.ops.aten.mul.Tensor(convert_element_type_1878, convert_element_type_1880);  convert_element_type_1880 = None
        mul_560 = torch.ops.aten.mul.Tensor(mul_136, mul_558)
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        div_30 = torch.ops.aten.div.Tensor(mul_136, 4096)
        mul_561 = torch.ops.aten.mul.Tensor(div_30, sum_91);  div_30 = sum_91 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_558, mul_561);  mul_558 = mul_561 = None
        mul_562 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_34);  sub_45 = rsqrt_34 = None
        mul_563 = torch.ops.aten.mul.Tensor(convert_element_type_1878, mul_136);  convert_element_type_1878 = mul_136 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_563, [0, 1]);  mul_563 = None
        convert_element_type_1881 = torch.ops.prims.convert_element_type.default(mul_562, torch.bfloat16);  mul_562 = None
        add_233 = torch.ops.aten.add.Tensor(add_230, convert_element_type_1881);  add_230 = convert_element_type_1881 = None
        convert_element_type_default_35 = torch.ops.prims.convert_element_type.default(sum_92, torch.float32);  sum_92 = None
        reduce_scatter_tensor_136 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_35, 'avg', 256, '0');  convert_element_type_default_35 = None
        wait_tensor_427 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_136);  reduce_scatter_tensor_136 = None
        view_1455 = torch.ops.aten.view.default(add_233, [16384, 4096])
        permute_837 = torch.ops.aten.permute.default(view_1455, [1, 0])
        permute_182 = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3])
        view_565 = torch.ops.aten.view.default(permute_182, [2, 8192, -1]);  permute_182 = None
        convert_element_type_545 = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        all_gather_into_tensor_149 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_545, 256, '0');  convert_element_type_545 = None
        wait_tensor_149 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_149);  all_gather_into_tensor_149 = None
        permute_183 = torch.ops.aten.permute.default(wait_tensor_149, [1, 0]);  wait_tensor_149 = None
        view_567 = torch.ops.aten.view.default(view_565, [16384, 4096]);  view_565 = None
        mm_115 = torch.ops.aten.mm.default(view_567, permute_183)
        view_568 = torch.ops.aten.view.default(mm_115, [2, 8192, 4096]);  mm_115 = None
        add_65 = torch.ops.aten.add.Tensor(add_63, view_568);  view_568 = None
        convert_element_type_548 = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        all_gather_into_tensor_150 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_548, 256, '0');  convert_element_type_548 = None
        wait_tensor_150 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_150);  all_gather_into_tensor_150 = None
        convert_element_type_549 = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_549, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_34, [2], True);  pow_34 = None
        add_66 = torch.ops.aten.add.Scalar(mean_33, 1e-05);  mean_33 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_132 = torch.ops.aten.mul.Tensor(convert_element_type_549, rsqrt_33);  convert_element_type_549 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, wait_tensor_150)
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        view_571 = torch.ops.aten.view.default(convert_element_type_550, [16384, 4096]);  convert_element_type_550 = None
        view_572 = torch.ops.aten.view.default(mm_116, [2, 8192, 14336]);  mm_116 = None
        convert_element_type_554 = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(convert_element_type_554)
        mul_134 = torch.ops.aten.mul.Tensor(convert_element_type_554, sigmoid_16);  sigmoid_16 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        all_gather_into_tensor_152 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_556, 256, '0');  convert_element_type_556 = None
        wait_tensor_152 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_152);  all_gather_into_tensor_152 = None
        permute_185 = torch.ops.aten.permute.default(wait_tensor_152, [1, 0]);  wait_tensor_152 = None
        mm_117 = torch.ops.aten.mm.default(view_571, permute_185)
        view_575 = torch.ops.aten.view.default(mm_117, [2, 8192, 14336]);  mm_117 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_555, view_575)
        view_577 = torch.ops.aten.view.default(mul_135, [16384, 14336]);  mul_135 = None
        mm_437 = torch.ops.aten.mm.default(permute_837, view_577);  permute_837 = view_577 = None
        convert_element_type_559 = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        all_gather_into_tensor_153 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_559, 256, '0');  convert_element_type_559 = None
        wait_tensor_153 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_153);  all_gather_into_tensor_153 = None
        permute_186 = torch.ops.aten.permute.default(wait_tensor_153, [1, 0]);  wait_tensor_153 = None
        permute_839 = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_438 = torch.ops.aten.mm.default(view_1455, permute_839);  view_1455 = permute_839 = None
        view_1456 = torch.ops.aten.view.default(mm_438, [2, 8192, 14336]);  mm_438 = None
        convert_element_type_1888 = torch.ops.prims.convert_element_type.default(mm_437, torch.float32);  mm_437 = None
        reduce_scatter_tensor_137 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1888, 'avg', 256, '0');  convert_element_type_1888 = None
        wait_tensor_428 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_137);  reduce_scatter_tensor_137 = None
        mul_564 = torch.ops.aten.mul.Tensor(view_1456, convert_element_type_555);  convert_element_type_555 = None
        mul_565 = torch.ops.aten.mul.Tensor(view_1456, view_575);  view_1456 = view_575 = None
        view_1457 = torch.ops.aten.view.default(mul_564, [16384, 14336]);  mul_564 = None
        permute_841 = torch.ops.aten.permute.default(view_1457, [1, 0])
        mm_439 = torch.ops.aten.mm.default(permute_841, view_571);  permute_841 = None
        permute_843 = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_440 = torch.ops.aten.mm.default(view_1457, permute_843);  view_1457 = permute_843 = None
        view_1458 = torch.ops.aten.view.default(mm_440, [2, 8192, 4096]);  mm_440 = None
        convert_element_type_1893 = torch.ops.prims.convert_element_type.default(mm_439, torch.float32);  mm_439 = None
        reduce_scatter_tensor_138 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1893, 'avg', 256, '0');  convert_element_type_1893 = None
        wait_tensor_429 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_138);  reduce_scatter_tensor_138 = None
        convert_element_type_1894 = torch.ops.prims.convert_element_type.default(mul_565, torch.float32);  mul_565 = None
        neg_15 = torch.ops.aten.neg.default(convert_element_type_554)
        exp_15 = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_234 = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(add_234);  add_234 = None
        mul_566 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_567 = torch.ops.aten.mul.Tensor(convert_element_type_1894, mul_566);  convert_element_type_1894 = None
        sub_46 = torch.ops.aten.sub.Tensor(1, mul_566);  mul_566 = None
        mul_568 = torch.ops.aten.mul.Tensor(convert_element_type_554, sub_46);  convert_element_type_554 = sub_46 = None
        add_235 = torch.ops.aten.add.Tensor(mul_568, 1);  mul_568 = None
        mul_569 = torch.ops.aten.mul.Tensor(mul_567, add_235);  mul_567 = add_235 = None
        convert_element_type_1896 = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None
        view_1459 = torch.ops.aten.view.default(convert_element_type_1896, [16384, 14336]);  convert_element_type_1896 = None
        permute_845 = torch.ops.aten.permute.default(view_1459, [1, 0])
        mm_441 = torch.ops.aten.mm.default(permute_845, view_571);  permute_845 = view_571 = None
        convert_element_type_551 = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        all_gather_into_tensor_151 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_551, 256, '0');  convert_element_type_551 = None
        wait_tensor_151 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_151);  all_gather_into_tensor_151 = None
        permute_184 = torch.ops.aten.permute.default(wait_tensor_151, [1, 0]);  wait_tensor_151 = None
        permute_847 = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_442 = torch.ops.aten.mm.default(view_1459, permute_847);  view_1459 = permute_847 = None
        view_1460 = torch.ops.aten.view.default(mm_442, [2, 8192, 4096]);  mm_442 = None
        add_236 = torch.ops.aten.add.Tensor(view_1458, view_1460);  view_1458 = view_1460 = None
        convert_element_type_1901 = torch.ops.prims.convert_element_type.default(mm_441, torch.float32);  mm_441 = None
        reduce_scatter_tensor_139 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1901, 'avg', 256, '0');  convert_element_type_1901 = None
        wait_tensor_430 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_139);  reduce_scatter_tensor_139 = None
        convert_element_type_1902 = torch.ops.prims.convert_element_type.default(add_236, torch.float32);  add_236 = None
        convert_element_type_1904 = torch.ops.prims.convert_element_type.default(wait_tensor_150, torch.float32);  wait_tensor_150 = None
        mul_570 = torch.ops.aten.mul.Tensor(convert_element_type_1902, convert_element_type_1904);  convert_element_type_1904 = None
        mul_572 = torch.ops.aten.mul.Tensor(mul_132, mul_570)
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
        div_31 = torch.ops.aten.div.Tensor(mul_132, 4096)
        mul_573 = torch.ops.aten.mul.Tensor(div_31, sum_93);  div_31 = sum_93 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_570, mul_573);  mul_570 = mul_573 = None
        mul_574 = torch.ops.aten.mul.Tensor(sub_47, rsqrt_33);  sub_47 = rsqrt_33 = None
        mul_575 = torch.ops.aten.mul.Tensor(convert_element_type_1902, mul_132);  convert_element_type_1902 = mul_132 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
        convert_element_type_1905 = torch.ops.prims.convert_element_type.default(mul_574, torch.bfloat16);  mul_574 = None
        add_237 = torch.ops.aten.add.Tensor(add_233, convert_element_type_1905);  add_233 = convert_element_type_1905 = None
        convert_element_type_default_34 = torch.ops.prims.convert_element_type.default(sum_94, torch.float32);  sum_94 = None
        reduce_scatter_tensor_140 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_34, 'avg', 256, '0');  convert_element_type_default_34 = None
        wait_tensor_431 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_140);  reduce_scatter_tensor_140 = None
        view_1461 = torch.ops.aten.view.default(add_237, [16384, 4096])
        permute_849 = torch.ops.aten.permute.default(view_1461, [1, 0])
        mm_443 = torch.ops.aten.mm.default(permute_849, view_567);  permute_849 = view_567 = None
        permute_851 = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        mm_444 = torch.ops.aten.mm.default(view_1461, permute_851);  view_1461 = permute_851 = None
        view_1462 = torch.ops.aten.view.default(mm_444, [2, 8192, 4096]);  mm_444 = None
        convert_element_type_1912 = torch.ops.prims.convert_element_type.default(mm_443, torch.float32);  mm_443 = None
        reduce_scatter_tensor_141 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1912, 'avg', 256, '0');  convert_element_type_1912 = None
        wait_tensor_432 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_141);  reduce_scatter_tensor_141 = None
        view_1463 = torch.ops.aten.view.default(view_1462, [2, 8192, 32, 128]);  view_1462 = None
        permute_853 = torch.ops.aten.permute.default(view_1463, [0, 2, 1, 3]);  view_1463 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        all_gather_into_tensor_145 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_529, 256, '0');  convert_element_type_529 = None
        wait_tensor_145 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_145);  all_gather_into_tensor_145 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_530, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_33, [2], True);  pow_33 = None
        add_64 = torch.ops.aten.add.Scalar(mean_32, 1e-05);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_32);  convert_element_type_530 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, wait_tensor_145)
        convert_element_type_531 = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        view_547 = torch.ops.aten.view.default(convert_element_type_531, [16384, 4096]);  convert_element_type_531 = None
        view_548 = torch.ops.aten.view.default(mm_112, [2, 8192, 4096]);  mm_112 = None
        convert_element_type_535 = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        all_gather_into_tensor_147 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_535, 256, '0');  convert_element_type_535 = None
        wait_tensor_147 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_147);  all_gather_into_tensor_147 = None
        permute_177 = torch.ops.aten.permute.default(wait_tensor_147, [1, 0]);  wait_tensor_147 = None
        mm_113 = torch.ops.aten.mm.default(view_547, permute_177)
        view_551 = torch.ops.aten.view.default(mm_113, [2, 8192, 1024]);  mm_113 = None
        view_554 = torch.ops.aten.view.default(mm_114, [2, 8192, 1024]);  mm_114 = None
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
        _scaled_dot_product_cudnn_attention_backward_15 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_853, permute_179, permute_180, permute_181, getitem_144, getitem_145, getitem_150, getitem_151, None, None, None, 8192, 8192, 0.0, True);  permute_853 = permute_179 = permute_180 = permute_181 = getitem_144 = getitem_145 = getitem_150 = getitem_151 = None
        getitem_333 = _scaled_dot_product_cudnn_attention_backward_15[0]
        getitem_334 = _scaled_dot_product_cudnn_attention_backward_15[1]
        getitem_335 = _scaled_dot_product_cudnn_attention_backward_15[2];  _scaled_dot_product_cudnn_attention_backward_15 = None
        permute_854 = torch.ops.aten.permute.default(getitem_335, [0, 2, 1, 3]);  getitem_335 = None
        permute_855 = torch.ops.aten.permute.default(getitem_334, [0, 2, 1, 3]);  getitem_334 = None
        permute_856 = torch.ops.aten.permute.default(getitem_333, [0, 2, 1, 3]);  getitem_333 = None
        view_1464 = torch.ops.aten.view.default(permute_854, [2, 8192, 8, 4, 128]);  permute_854 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(view_1464, [3], True);  view_1464 = None
        squeeze_30 = torch.ops.aten.squeeze.dim(sum_95, 3);  sum_95 = None
        view_1465 = torch.ops.aten.view.default(permute_855, [2, 8192, 8, 4, 128]);  permute_855 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(view_1465, [3], True);  view_1465 = None
        squeeze_31 = torch.ops.aten.squeeze.dim(sum_96, 3);  sum_96 = None
        convert_element_type_1913 = torch.ops.prims.convert_element_type.default(squeeze_31, torch.float32);  squeeze_31 = None
        convert_element_type_1914 = torch.ops.prims.convert_element_type.default(permute_856, torch.float32);  permute_856 = None
        view_1466 = torch.ops.aten.view.default(convert_element_type_1913, [2, 8192, 8, 64, 2]);  convert_element_type_1913 = None
        view_as_complex_94 = torch.ops.aten.view_as_complex.default(view_1466);  view_1466 = None
        mul_576 = torch.ops.aten.mul.Tensor(view_as_complex_94, _conj);  view_as_complex_94 = None
        view_1467 = torch.ops.aten.view.default(convert_element_type_1914, [2, 8192, 32, 64, 2]);  convert_element_type_1914 = None
        view_as_complex_95 = torch.ops.aten.view_as_complex.default(view_1467);  view_1467 = None
        mul_577 = torch.ops.aten.mul.Tensor(view_as_complex_95, _conj);  view_as_complex_95 = None
        view_as_real_94 = torch.ops.aten.view_as_real.default(mul_576);  mul_576 = None
        view_1468 = torch.ops.aten.view.default(view_as_real_94, [2, 8192, 8, 128]);  view_as_real_94 = None
        convert_element_type_1915 = torch.ops.prims.convert_element_type.default(view_1468, torch.bfloat16);  view_1468 = None
        view_as_real_95 = torch.ops.aten.view_as_real.default(mul_577);  mul_577 = None
        view_1469 = torch.ops.aten.view.default(view_as_real_95, [2, 8192, 32, 128]);  view_as_real_95 = None
        convert_element_type_1916 = torch.ops.prims.convert_element_type.default(view_1469, torch.bfloat16);  view_1469 = None
        view_1470 = torch.ops.aten.view.default(squeeze_30, [2, 8192, 1024]);  squeeze_30 = None
        view_1471 = torch.ops.aten.view.default(convert_element_type_1915, [2, 8192, 1024]);  convert_element_type_1915 = None
        view_1472 = torch.ops.aten.view.default(convert_element_type_1916, [2, 8192, 4096]);  convert_element_type_1916 = None
        view_1473 = torch.ops.aten.view.default(view_1470, [16384, 1024]);  view_1470 = None
        permute_857 = torch.ops.aten.permute.default(view_1473, [1, 0])
        mm_445 = torch.ops.aten.mm.default(permute_857, view_547);  permute_857 = None
        convert_element_type_538 = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        all_gather_into_tensor_148 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_538, 256, '0');  convert_element_type_538 = None
        wait_tensor_148 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_148);  all_gather_into_tensor_148 = None
        permute_178 = torch.ops.aten.permute.default(wait_tensor_148, [1, 0]);  wait_tensor_148 = None
        permute_859 = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_446 = torch.ops.aten.mm.default(view_1473, permute_859);  view_1473 = permute_859 = None
        view_1474 = torch.ops.aten.view.default(mm_446, [2, 8192, 4096]);  mm_446 = None
        convert_element_type_1921 = torch.ops.prims.convert_element_type.default(mm_445, torch.float32);  mm_445 = None
        reduce_scatter_tensor_142 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1921, 'avg', 256, '0');  convert_element_type_1921 = None
        wait_tensor_433 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_142);  reduce_scatter_tensor_142 = None
        view_1475 = torch.ops.aten.view.default(view_1471, [16384, 1024]);  view_1471 = None
        permute_861 = torch.ops.aten.permute.default(view_1475, [1, 0])
        mm_447 = torch.ops.aten.mm.default(permute_861, view_547);  permute_861 = None
        permute_863 = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        mm_448 = torch.ops.aten.mm.default(view_1475, permute_863);  view_1475 = permute_863 = None
        view_1476 = torch.ops.aten.view.default(mm_448, [2, 8192, 4096]);  mm_448 = None
        add_238 = torch.ops.aten.add.Tensor(view_1474, view_1476);  view_1474 = view_1476 = None
        convert_element_type_1926 = torch.ops.prims.convert_element_type.default(mm_447, torch.float32);  mm_447 = None
        reduce_scatter_tensor_143 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1926, 'avg', 256, '0');  convert_element_type_1926 = None
        wait_tensor_434 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_143);  reduce_scatter_tensor_143 = None
        view_1477 = torch.ops.aten.view.default(view_1472, [16384, 4096]);  view_1472 = None
        permute_865 = torch.ops.aten.permute.default(view_1477, [1, 0])
        mm_449 = torch.ops.aten.mm.default(permute_865, view_547);  permute_865 = view_547 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        all_gather_into_tensor_146 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_532, 256, '0');  convert_element_type_532 = None
        wait_tensor_146 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_146);  all_gather_into_tensor_146 = None
        permute_176 = torch.ops.aten.permute.default(wait_tensor_146, [1, 0]);  wait_tensor_146 = None
        permute_867 = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_450 = torch.ops.aten.mm.default(view_1477, permute_867);  view_1477 = permute_867 = None
        view_1478 = torch.ops.aten.view.default(mm_450, [2, 8192, 4096]);  mm_450 = None
        add_239 = torch.ops.aten.add.Tensor(add_238, view_1478);  add_238 = view_1478 = None
        convert_element_type_1931 = torch.ops.prims.convert_element_type.default(mm_449, torch.float32);  mm_449 = None
        reduce_scatter_tensor_144 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1931, 'avg', 256, '0');  convert_element_type_1931 = None
        wait_tensor_435 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_144);  reduce_scatter_tensor_144 = None
        convert_element_type_1932 = torch.ops.prims.convert_element_type.default(add_239, torch.float32);  add_239 = None
        convert_element_type_1934 = torch.ops.prims.convert_element_type.default(wait_tensor_145, torch.float32);  wait_tensor_145 = None
        mul_578 = torch.ops.aten.mul.Tensor(convert_element_type_1932, convert_element_type_1934);  convert_element_type_1934 = None
        mul_580 = torch.ops.aten.mul.Tensor(mul_128, mul_578)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_580, [2], True);  mul_580 = None
        div_32 = torch.ops.aten.div.Tensor(mul_128, 4096)
        mul_581 = torch.ops.aten.mul.Tensor(div_32, sum_97);  div_32 = sum_97 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_578, mul_581);  mul_578 = mul_581 = None
        mul_582 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_583 = torch.ops.aten.mul.Tensor(convert_element_type_1932, mul_128);  convert_element_type_1932 = mul_128 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_583, [0, 1]);  mul_583 = None
        convert_element_type_1935 = torch.ops.prims.convert_element_type.default(mul_582, torch.bfloat16);  mul_582 = None
        add_240 = torch.ops.aten.add.Tensor(add_237, convert_element_type_1935);  add_237 = convert_element_type_1935 = None
        convert_element_type_default_33 = torch.ops.prims.convert_element_type.default(sum_98, torch.float32);  sum_98 = None
        reduce_scatter_tensor_145 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_33, 'avg', 256, '0');  convert_element_type_default_33 = None
        wait_tensor_436 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_145);  reduce_scatter_tensor_145 = None
        view_1479 = torch.ops.aten.view.default(add_240, [16384, 4096])
        permute_869 = torch.ops.aten.permute.default(view_1479, [1, 0])
        permute_171 = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3])
        view_531 = torch.ops.aten.view.default(permute_171, [2, 8192, -1]);  permute_171 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        all_gather_into_tensor_140 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_512, 256, '0');  convert_element_type_512 = None
        wait_tensor_140 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_140);  all_gather_into_tensor_140 = None
        permute_172 = torch.ops.aten.permute.default(wait_tensor_140, [1, 0]);  wait_tensor_140 = None
        view_533 = torch.ops.aten.view.default(view_531, [16384, 4096]);  view_531 = None
        mm_108 = torch.ops.aten.mm.default(view_533, permute_172)
        view_534 = torch.ops.aten.view.default(mm_108, [2, 8192, 4096]);  mm_108 = None
        add_61 = torch.ops.aten.add.Tensor(add_59, view_534);  view_534 = None
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        all_gather_into_tensor_141 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_515, 256, '0');  convert_element_type_515 = None
        wait_tensor_141 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_141);  all_gather_into_tensor_141 = None
        convert_element_type_516 = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_516, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_62 = torch.ops.aten.add.Scalar(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_516, rsqrt_31);  convert_element_type_516 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, wait_tensor_141)
        convert_element_type_517 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        view_537 = torch.ops.aten.view.default(convert_element_type_517, [16384, 4096]);  convert_element_type_517 = None
        view_538 = torch.ops.aten.view.default(mm_109, [2, 8192, 14336]);  mm_109 = None
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(view_538, torch.float32);  view_538 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_521)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_521, sigmoid_15);  sigmoid_15 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        all_gather_into_tensor_143 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_523, 256, '0');  convert_element_type_523 = None
        wait_tensor_143 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_143);  all_gather_into_tensor_143 = None
        permute_174 = torch.ops.aten.permute.default(wait_tensor_143, [1, 0]);  wait_tensor_143 = None
        mm_110 = torch.ops.aten.mm.default(view_537, permute_174)
        view_541 = torch.ops.aten.view.default(mm_110, [2, 8192, 14336]);  mm_110 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_522, view_541)
        view_543 = torch.ops.aten.view.default(mul_127, [16384, 14336]);  mul_127 = None
        mm_451 = torch.ops.aten.mm.default(permute_869, view_543);  permute_869 = view_543 = None
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        all_gather_into_tensor_144 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_526, 256, '0');  convert_element_type_526 = None
        wait_tensor_144 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_144);  all_gather_into_tensor_144 = None
        permute_175 = torch.ops.aten.permute.default(wait_tensor_144, [1, 0]);  wait_tensor_144 = None
        permute_871 = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_452 = torch.ops.aten.mm.default(view_1479, permute_871);  view_1479 = permute_871 = None
        view_1480 = torch.ops.aten.view.default(mm_452, [2, 8192, 14336]);  mm_452 = None
        convert_element_type_1942 = torch.ops.prims.convert_element_type.default(mm_451, torch.float32);  mm_451 = None
        reduce_scatter_tensor_146 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1942, 'avg', 256, '0');  convert_element_type_1942 = None
        wait_tensor_437 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_146);  reduce_scatter_tensor_146 = None
        mul_584 = torch.ops.aten.mul.Tensor(view_1480, convert_element_type_522);  convert_element_type_522 = None
        mul_585 = torch.ops.aten.mul.Tensor(view_1480, view_541);  view_1480 = view_541 = None
        view_1481 = torch.ops.aten.view.default(mul_584, [16384, 14336]);  mul_584 = None
        permute_873 = torch.ops.aten.permute.default(view_1481, [1, 0])
        mm_453 = torch.ops.aten.mm.default(permute_873, view_537);  permute_873 = None
        permute_875 = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_454 = torch.ops.aten.mm.default(view_1481, permute_875);  view_1481 = permute_875 = None
        view_1482 = torch.ops.aten.view.default(mm_454, [2, 8192, 4096]);  mm_454 = None
        convert_element_type_1947 = torch.ops.prims.convert_element_type.default(mm_453, torch.float32);  mm_453 = None
        reduce_scatter_tensor_147 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1947, 'avg', 256, '0');  convert_element_type_1947 = None
        wait_tensor_438 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_147);  reduce_scatter_tensor_147 = None
        convert_element_type_1948 = torch.ops.prims.convert_element_type.default(mul_585, torch.float32);  mul_585 = None
        neg_16 = torch.ops.aten.neg.default(convert_element_type_521)
        exp_16 = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_241 = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(add_241);  add_241 = None
        mul_586 = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        mul_587 = torch.ops.aten.mul.Tensor(convert_element_type_1948, mul_586);  convert_element_type_1948 = None
        sub_49 = torch.ops.aten.sub.Tensor(1, mul_586);  mul_586 = None
        mul_588 = torch.ops.aten.mul.Tensor(convert_element_type_521, sub_49);  convert_element_type_521 = sub_49 = None
        add_242 = torch.ops.aten.add.Tensor(mul_588, 1);  mul_588 = None
        mul_589 = torch.ops.aten.mul.Tensor(mul_587, add_242);  mul_587 = add_242 = None
        convert_element_type_1950 = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16);  mul_589 = None
        view_1483 = torch.ops.aten.view.default(convert_element_type_1950, [16384, 14336]);  convert_element_type_1950 = None
        permute_877 = torch.ops.aten.permute.default(view_1483, [1, 0])
        mm_455 = torch.ops.aten.mm.default(permute_877, view_537);  permute_877 = view_537 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        all_gather_into_tensor_142 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_518, 256, '0');  convert_element_type_518 = None
        wait_tensor_142 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_142);  all_gather_into_tensor_142 = None
        permute_173 = torch.ops.aten.permute.default(wait_tensor_142, [1, 0]);  wait_tensor_142 = None
        permute_879 = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_456 = torch.ops.aten.mm.default(view_1483, permute_879);  view_1483 = permute_879 = None
        view_1484 = torch.ops.aten.view.default(mm_456, [2, 8192, 4096]);  mm_456 = None
        add_243 = torch.ops.aten.add.Tensor(view_1482, view_1484);  view_1482 = view_1484 = None
        convert_element_type_1955 = torch.ops.prims.convert_element_type.default(mm_455, torch.float32);  mm_455 = None
        reduce_scatter_tensor_148 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1955, 'avg', 256, '0');  convert_element_type_1955 = None
        wait_tensor_439 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_148);  reduce_scatter_tensor_148 = None
        convert_element_type_1956 = torch.ops.prims.convert_element_type.default(add_243, torch.float32);  add_243 = None
        convert_element_type_1958 = torch.ops.prims.convert_element_type.default(wait_tensor_141, torch.float32);  wait_tensor_141 = None
        mul_590 = torch.ops.aten.mul.Tensor(convert_element_type_1956, convert_element_type_1958);  convert_element_type_1958 = None
        mul_592 = torch.ops.aten.mul.Tensor(mul_124, mul_590)
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_592, [2], True);  mul_592 = None
        div_33 = torch.ops.aten.div.Tensor(mul_124, 4096)
        mul_593 = torch.ops.aten.mul.Tensor(div_33, sum_99);  div_33 = sum_99 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_590, mul_593);  mul_590 = mul_593 = None
        mul_594 = torch.ops.aten.mul.Tensor(sub_50, rsqrt_31);  sub_50 = rsqrt_31 = None
        mul_595 = torch.ops.aten.mul.Tensor(convert_element_type_1956, mul_124);  convert_element_type_1956 = mul_124 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1]);  mul_595 = None
        convert_element_type_1959 = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None
        add_244 = torch.ops.aten.add.Tensor(add_240, convert_element_type_1959);  add_240 = convert_element_type_1959 = None
        convert_element_type_default_32 = torch.ops.prims.convert_element_type.default(sum_100, torch.float32);  sum_100 = None
        reduce_scatter_tensor_149 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_32, 'avg', 256, '0');  convert_element_type_default_32 = None
        wait_tensor_440 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_149);  reduce_scatter_tensor_149 = None
        view_1485 = torch.ops.aten.view.default(add_244, [16384, 4096])
        permute_881 = torch.ops.aten.permute.default(view_1485, [1, 0])
        mm_457 = torch.ops.aten.mm.default(permute_881, view_533);  permute_881 = view_533 = None
        permute_883 = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        mm_458 = torch.ops.aten.mm.default(view_1485, permute_883);  view_1485 = permute_883 = None
        view_1486 = torch.ops.aten.view.default(mm_458, [2, 8192, 4096]);  mm_458 = None
        convert_element_type_1966 = torch.ops.prims.convert_element_type.default(mm_457, torch.float32);  mm_457 = None
        reduce_scatter_tensor_150 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1966, 'avg', 256, '0');  convert_element_type_1966 = None
        wait_tensor_441 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_150);  reduce_scatter_tensor_150 = None
        view_1487 = torch.ops.aten.view.default(view_1486, [2, 8192, 32, 128]);  view_1486 = None
        permute_885 = torch.ops.aten.permute.default(view_1487, [0, 2, 1, 3]);  view_1487 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        all_gather_into_tensor_136 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_496, 256, '0');  convert_element_type_496 = None
        wait_tensor_136 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_136);  all_gather_into_tensor_136 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [2], True);  pow_31 = None
        add_60 = torch.ops.aten.add.Scalar(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_30);  convert_element_type_497 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, wait_tensor_136)
        convert_element_type_498 = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        view_513 = torch.ops.aten.view.default(convert_element_type_498, [16384, 4096]);  convert_element_type_498 = None
        view_514 = torch.ops.aten.view.default(mm_105, [2, 8192, 4096]);  mm_105 = None
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        all_gather_into_tensor_138 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_502, 256, '0');  convert_element_type_502 = None
        wait_tensor_138 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_138);  all_gather_into_tensor_138 = None
        permute_166 = torch.ops.aten.permute.default(wait_tensor_138, [1, 0]);  wait_tensor_138 = None
        mm_106 = torch.ops.aten.mm.default(view_513, permute_166)
        view_517 = torch.ops.aten.view.default(mm_106, [2, 8192, 1024]);  mm_106 = None
        view_520 = torch.ops.aten.view.default(mm_107, [2, 8192, 1024]);  mm_107 = None
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
        _scaled_dot_product_cudnn_attention_backward_16 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_885, permute_168, permute_169, permute_170, getitem_135, getitem_136, getitem_141, getitem_142, None, None, None, 8192, 8192, 0.0, True);  permute_885 = permute_168 = permute_169 = permute_170 = getitem_135 = getitem_136 = getitem_141 = getitem_142 = None
        getitem_336 = _scaled_dot_product_cudnn_attention_backward_16[0]
        getitem_337 = _scaled_dot_product_cudnn_attention_backward_16[1]
        getitem_338 = _scaled_dot_product_cudnn_attention_backward_16[2];  _scaled_dot_product_cudnn_attention_backward_16 = None
        permute_886 = torch.ops.aten.permute.default(getitem_338, [0, 2, 1, 3]);  getitem_338 = None
        permute_887 = torch.ops.aten.permute.default(getitem_337, [0, 2, 1, 3]);  getitem_337 = None
        permute_888 = torch.ops.aten.permute.default(getitem_336, [0, 2, 1, 3]);  getitem_336 = None
        view_1488 = torch.ops.aten.view.default(permute_886, [2, 8192, 8, 4, 128]);  permute_886 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(view_1488, [3], True);  view_1488 = None
        squeeze_32 = torch.ops.aten.squeeze.dim(sum_101, 3);  sum_101 = None
        view_1489 = torch.ops.aten.view.default(permute_887, [2, 8192, 8, 4, 128]);  permute_887 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(view_1489, [3], True);  view_1489 = None
        squeeze_33 = torch.ops.aten.squeeze.dim(sum_102, 3);  sum_102 = None
        convert_element_type_1967 = torch.ops.prims.convert_element_type.default(squeeze_33, torch.float32);  squeeze_33 = None
        convert_element_type_1968 = torch.ops.prims.convert_element_type.default(permute_888, torch.float32);  permute_888 = None
        view_1490 = torch.ops.aten.view.default(convert_element_type_1967, [2, 8192, 8, 64, 2]);  convert_element_type_1967 = None
        view_as_complex_96 = torch.ops.aten.view_as_complex.default(view_1490);  view_1490 = None
        mul_596 = torch.ops.aten.mul.Tensor(view_as_complex_96, _conj);  view_as_complex_96 = None
        view_1491 = torch.ops.aten.view.default(convert_element_type_1968, [2, 8192, 32, 64, 2]);  convert_element_type_1968 = None
        view_as_complex_97 = torch.ops.aten.view_as_complex.default(view_1491);  view_1491 = None
        mul_597 = torch.ops.aten.mul.Tensor(view_as_complex_97, _conj);  view_as_complex_97 = None
        view_as_real_96 = torch.ops.aten.view_as_real.default(mul_596);  mul_596 = None
        view_1492 = torch.ops.aten.view.default(view_as_real_96, [2, 8192, 8, 128]);  view_as_real_96 = None
        convert_element_type_1969 = torch.ops.prims.convert_element_type.default(view_1492, torch.bfloat16);  view_1492 = None
        view_as_real_97 = torch.ops.aten.view_as_real.default(mul_597);  mul_597 = None
        view_1493 = torch.ops.aten.view.default(view_as_real_97, [2, 8192, 32, 128]);  view_as_real_97 = None
        convert_element_type_1970 = torch.ops.prims.convert_element_type.default(view_1493, torch.bfloat16);  view_1493 = None
        view_1494 = torch.ops.aten.view.default(squeeze_32, [2, 8192, 1024]);  squeeze_32 = None
        view_1495 = torch.ops.aten.view.default(convert_element_type_1969, [2, 8192, 1024]);  convert_element_type_1969 = None
        view_1496 = torch.ops.aten.view.default(convert_element_type_1970, [2, 8192, 4096]);  convert_element_type_1970 = None
        view_1497 = torch.ops.aten.view.default(view_1494, [16384, 1024]);  view_1494 = None
        permute_889 = torch.ops.aten.permute.default(view_1497, [1, 0])
        mm_459 = torch.ops.aten.mm.default(permute_889, view_513);  permute_889 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        all_gather_into_tensor_139 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_505, 256, '0');  convert_element_type_505 = None
        wait_tensor_139 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_139);  all_gather_into_tensor_139 = None
        permute_167 = torch.ops.aten.permute.default(wait_tensor_139, [1, 0]);  wait_tensor_139 = None
        permute_891 = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_460 = torch.ops.aten.mm.default(view_1497, permute_891);  view_1497 = permute_891 = None
        view_1498 = torch.ops.aten.view.default(mm_460, [2, 8192, 4096]);  mm_460 = None
        convert_element_type_1975 = torch.ops.prims.convert_element_type.default(mm_459, torch.float32);  mm_459 = None
        reduce_scatter_tensor_151 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1975, 'avg', 256, '0');  convert_element_type_1975 = None
        wait_tensor_442 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_151);  reduce_scatter_tensor_151 = None
        view_1499 = torch.ops.aten.view.default(view_1495, [16384, 1024]);  view_1495 = None
        permute_893 = torch.ops.aten.permute.default(view_1499, [1, 0])
        mm_461 = torch.ops.aten.mm.default(permute_893, view_513);  permute_893 = None
        permute_895 = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_462 = torch.ops.aten.mm.default(view_1499, permute_895);  view_1499 = permute_895 = None
        view_1500 = torch.ops.aten.view.default(mm_462, [2, 8192, 4096]);  mm_462 = None
        add_245 = torch.ops.aten.add.Tensor(view_1498, view_1500);  view_1498 = view_1500 = None
        convert_element_type_1980 = torch.ops.prims.convert_element_type.default(mm_461, torch.float32);  mm_461 = None
        reduce_scatter_tensor_152 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1980, 'avg', 256, '0');  convert_element_type_1980 = None
        wait_tensor_443 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_152);  reduce_scatter_tensor_152 = None
        view_1501 = torch.ops.aten.view.default(view_1496, [16384, 4096]);  view_1496 = None
        permute_897 = torch.ops.aten.permute.default(view_1501, [1, 0])
        mm_463 = torch.ops.aten.mm.default(permute_897, view_513);  permute_897 = view_513 = None
        convert_element_type_499 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        all_gather_into_tensor_137 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_499, 256, '0');  convert_element_type_499 = None
        wait_tensor_137 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_137);  all_gather_into_tensor_137 = None
        permute_165 = torch.ops.aten.permute.default(wait_tensor_137, [1, 0]);  wait_tensor_137 = None
        permute_899 = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_464 = torch.ops.aten.mm.default(view_1501, permute_899);  view_1501 = permute_899 = None
        view_1502 = torch.ops.aten.view.default(mm_464, [2, 8192, 4096]);  mm_464 = None
        add_246 = torch.ops.aten.add.Tensor(add_245, view_1502);  add_245 = view_1502 = None
        convert_element_type_1985 = torch.ops.prims.convert_element_type.default(mm_463, torch.float32);  mm_463 = None
        reduce_scatter_tensor_153 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1985, 'avg', 256, '0');  convert_element_type_1985 = None
        wait_tensor_444 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_153);  reduce_scatter_tensor_153 = None
        convert_element_type_1986 = torch.ops.prims.convert_element_type.default(add_246, torch.float32);  add_246 = None
        convert_element_type_1988 = torch.ops.prims.convert_element_type.default(wait_tensor_136, torch.float32);  wait_tensor_136 = None
        mul_598 = torch.ops.aten.mul.Tensor(convert_element_type_1986, convert_element_type_1988);  convert_element_type_1988 = None
        mul_600 = torch.ops.aten.mul.Tensor(mul_120, mul_598)
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_600, [2], True);  mul_600 = None
        div_34 = torch.ops.aten.div.Tensor(mul_120, 4096)
        mul_601 = torch.ops.aten.mul.Tensor(div_34, sum_103);  div_34 = sum_103 = None
        sub_51 = torch.ops.aten.sub.Tensor(mul_598, mul_601);  mul_598 = mul_601 = None
        mul_602 = torch.ops.aten.mul.Tensor(sub_51, rsqrt_30);  sub_51 = rsqrt_30 = None
        mul_603 = torch.ops.aten.mul.Tensor(convert_element_type_1986, mul_120);  convert_element_type_1986 = mul_120 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_603, [0, 1]);  mul_603 = None
        convert_element_type_1989 = torch.ops.prims.convert_element_type.default(mul_602, torch.bfloat16);  mul_602 = None
        add_247 = torch.ops.aten.add.Tensor(add_244, convert_element_type_1989);  add_244 = convert_element_type_1989 = None
        convert_element_type_default_31 = torch.ops.prims.convert_element_type.default(sum_104, torch.float32);  sum_104 = None
        reduce_scatter_tensor_154 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_31, 'avg', 256, '0');  convert_element_type_default_31 = None
        wait_tensor_445 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_154);  reduce_scatter_tensor_154 = None
        view_1503 = torch.ops.aten.view.default(add_247, [16384, 4096])
        permute_901 = torch.ops.aten.permute.default(view_1503, [1, 0])
        permute_160 = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_497 = torch.ops.aten.view.default(permute_160, [2, 8192, -1]);  permute_160 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        all_gather_into_tensor_131 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_479, 256, '0');  convert_element_type_479 = None
        wait_tensor_131 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_131);  all_gather_into_tensor_131 = None
        permute_161 = torch.ops.aten.permute.default(wait_tensor_131, [1, 0]);  wait_tensor_131 = None
        view_499 = torch.ops.aten.view.default(view_497, [16384, 4096]);  view_497 = None
        mm_101 = torch.ops.aten.mm.default(view_499, permute_161)
        view_500 = torch.ops.aten.view.default(mm_101, [2, 8192, 4096]);  mm_101 = None
        add_57 = torch.ops.aten.add.Tensor(add_55, view_500);  view_500 = None
        convert_element_type_482 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        all_gather_into_tensor_132 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_482, 256, '0');  convert_element_type_482 = None
        wait_tensor_132 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_132);  all_gather_into_tensor_132 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_58 = torch.ops.aten.add.Scalar(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_29);  convert_element_type_483 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, wait_tensor_132)
        convert_element_type_484 = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        view_503 = torch.ops.aten.view.default(convert_element_type_484, [16384, 4096]);  convert_element_type_484 = None
        view_504 = torch.ops.aten.view.default(mm_102, [2, 8192, 14336]);  mm_102 = None
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(view_504, torch.float32);  view_504 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_488)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_488, sigmoid_14);  sigmoid_14 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        all_gather_into_tensor_134 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_490, 256, '0');  convert_element_type_490 = None
        wait_tensor_134 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_134);  all_gather_into_tensor_134 = None
        permute_163 = torch.ops.aten.permute.default(wait_tensor_134, [1, 0]);  wait_tensor_134 = None
        mm_103 = torch.ops.aten.mm.default(view_503, permute_163)
        view_507 = torch.ops.aten.view.default(mm_103, [2, 8192, 14336]);  mm_103 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_489, view_507)
        view_509 = torch.ops.aten.view.default(mul_119, [16384, 14336]);  mul_119 = None
        mm_465 = torch.ops.aten.mm.default(permute_901, view_509);  permute_901 = view_509 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        all_gather_into_tensor_135 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_493, 256, '0');  convert_element_type_493 = None
        wait_tensor_135 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_135);  all_gather_into_tensor_135 = None
        permute_164 = torch.ops.aten.permute.default(wait_tensor_135, [1, 0]);  wait_tensor_135 = None
        permute_903 = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_466 = torch.ops.aten.mm.default(view_1503, permute_903);  view_1503 = permute_903 = None
        view_1504 = torch.ops.aten.view.default(mm_466, [2, 8192, 14336]);  mm_466 = None
        convert_element_type_1996 = torch.ops.prims.convert_element_type.default(mm_465, torch.float32);  mm_465 = None
        reduce_scatter_tensor_155 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1996, 'avg', 256, '0');  convert_element_type_1996 = None
        wait_tensor_446 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_155);  reduce_scatter_tensor_155 = None
        mul_604 = torch.ops.aten.mul.Tensor(view_1504, convert_element_type_489);  convert_element_type_489 = None
        mul_605 = torch.ops.aten.mul.Tensor(view_1504, view_507);  view_1504 = view_507 = None
        view_1505 = torch.ops.aten.view.default(mul_604, [16384, 14336]);  mul_604 = None
        permute_905 = torch.ops.aten.permute.default(view_1505, [1, 0])
        mm_467 = torch.ops.aten.mm.default(permute_905, view_503);  permute_905 = None
        permute_907 = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_468 = torch.ops.aten.mm.default(view_1505, permute_907);  view_1505 = permute_907 = None
        view_1506 = torch.ops.aten.view.default(mm_468, [2, 8192, 4096]);  mm_468 = None
        convert_element_type_2001 = torch.ops.prims.convert_element_type.default(mm_467, torch.float32);  mm_467 = None
        reduce_scatter_tensor_156 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2001, 'avg', 256, '0');  convert_element_type_2001 = None
        wait_tensor_447 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_156);  reduce_scatter_tensor_156 = None
        convert_element_type_2002 = torch.ops.prims.convert_element_type.default(mul_605, torch.float32);  mul_605 = None
        neg_17 = torch.ops.aten.neg.default(convert_element_type_488)
        exp_17 = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_248 = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(add_248);  add_248 = None
        mul_606 = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        mul_607 = torch.ops.aten.mul.Tensor(convert_element_type_2002, mul_606);  convert_element_type_2002 = None
        sub_52 = torch.ops.aten.sub.Tensor(1, mul_606);  mul_606 = None
        mul_608 = torch.ops.aten.mul.Tensor(convert_element_type_488, sub_52);  convert_element_type_488 = sub_52 = None
        add_249 = torch.ops.aten.add.Tensor(mul_608, 1);  mul_608 = None
        mul_609 = torch.ops.aten.mul.Tensor(mul_607, add_249);  mul_607 = add_249 = None
        convert_element_type_2004 = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16);  mul_609 = None
        view_1507 = torch.ops.aten.view.default(convert_element_type_2004, [16384, 14336]);  convert_element_type_2004 = None
        permute_909 = torch.ops.aten.permute.default(view_1507, [1, 0])
        mm_469 = torch.ops.aten.mm.default(permute_909, view_503);  permute_909 = view_503 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        all_gather_into_tensor_133 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_485, 256, '0');  convert_element_type_485 = None
        wait_tensor_133 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_133);  all_gather_into_tensor_133 = None
        permute_162 = torch.ops.aten.permute.default(wait_tensor_133, [1, 0]);  wait_tensor_133 = None
        permute_911 = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_470 = torch.ops.aten.mm.default(view_1507, permute_911);  view_1507 = permute_911 = None
        view_1508 = torch.ops.aten.view.default(mm_470, [2, 8192, 4096]);  mm_470 = None
        add_250 = torch.ops.aten.add.Tensor(view_1506, view_1508);  view_1506 = view_1508 = None
        convert_element_type_2009 = torch.ops.prims.convert_element_type.default(mm_469, torch.float32);  mm_469 = None
        reduce_scatter_tensor_157 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2009, 'avg', 256, '0');  convert_element_type_2009 = None
        wait_tensor_448 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_157);  reduce_scatter_tensor_157 = None
        convert_element_type_2010 = torch.ops.prims.convert_element_type.default(add_250, torch.float32);  add_250 = None
        convert_element_type_2012 = torch.ops.prims.convert_element_type.default(wait_tensor_132, torch.float32);  wait_tensor_132 = None
        mul_610 = torch.ops.aten.mul.Tensor(convert_element_type_2010, convert_element_type_2012);  convert_element_type_2012 = None
        mul_612 = torch.ops.aten.mul.Tensor(mul_116, mul_610)
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_612, [2], True);  mul_612 = None
        div_35 = torch.ops.aten.div.Tensor(mul_116, 4096)
        mul_613 = torch.ops.aten.mul.Tensor(div_35, sum_105);  div_35 = sum_105 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_610, mul_613);  mul_610 = mul_613 = None
        mul_614 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_29);  sub_53 = rsqrt_29 = None
        mul_615 = torch.ops.aten.mul.Tensor(convert_element_type_2010, mul_116);  convert_element_type_2010 = mul_116 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(mul_615, [0, 1]);  mul_615 = None
        convert_element_type_2013 = torch.ops.prims.convert_element_type.default(mul_614, torch.bfloat16);  mul_614 = None
        add_251 = torch.ops.aten.add.Tensor(add_247, convert_element_type_2013);  add_247 = convert_element_type_2013 = None
        convert_element_type_default_30 = torch.ops.prims.convert_element_type.default(sum_106, torch.float32);  sum_106 = None
        reduce_scatter_tensor_158 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_30, 'avg', 256, '0');  convert_element_type_default_30 = None
        wait_tensor_449 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_158);  reduce_scatter_tensor_158 = None
        view_1509 = torch.ops.aten.view.default(add_251, [16384, 4096])
        permute_913 = torch.ops.aten.permute.default(view_1509, [1, 0])
        mm_471 = torch.ops.aten.mm.default(permute_913, view_499);  permute_913 = view_499 = None
        permute_915 = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        mm_472 = torch.ops.aten.mm.default(view_1509, permute_915);  view_1509 = permute_915 = None
        view_1510 = torch.ops.aten.view.default(mm_472, [2, 8192, 4096]);  mm_472 = None
        convert_element_type_2020 = torch.ops.prims.convert_element_type.default(mm_471, torch.float32);  mm_471 = None
        reduce_scatter_tensor_159 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2020, 'avg', 256, '0');  convert_element_type_2020 = None
        wait_tensor_450 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_159);  reduce_scatter_tensor_159 = None
        view_1511 = torch.ops.aten.view.default(view_1510, [2, 8192, 32, 128]);  view_1510 = None
        permute_917 = torch.ops.aten.permute.default(view_1511, [0, 2, 1, 3]);  view_1511 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        all_gather_into_tensor_127 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_463, 256, '0');  convert_element_type_463 = None
        wait_tensor_127 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_127);  all_gather_into_tensor_127 = None
        convert_element_type_464 = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [2], True);  pow_29 = None
        add_56 = torch.ops.aten.add.Scalar(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_28);  convert_element_type_464 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, wait_tensor_127)
        convert_element_type_465 = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        view_479 = torch.ops.aten.view.default(convert_element_type_465, [16384, 4096]);  convert_element_type_465 = None
        view_480 = torch.ops.aten.view.default(mm_98, [2, 8192, 4096]);  mm_98 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        all_gather_into_tensor_129 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_469, 256, '0');  convert_element_type_469 = None
        wait_tensor_129 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_129);  all_gather_into_tensor_129 = None
        permute_155 = torch.ops.aten.permute.default(wait_tensor_129, [1, 0]);  wait_tensor_129 = None
        mm_99 = torch.ops.aten.mm.default(view_479, permute_155)
        view_483 = torch.ops.aten.view.default(mm_99, [2, 8192, 1024]);  mm_99 = None
        view_486 = torch.ops.aten.view.default(mm_100, [2, 8192, 1024]);  mm_100 = None
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
        _scaled_dot_product_cudnn_attention_backward_17 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_917, permute_157, permute_158, permute_159, getitem_126, getitem_127, getitem_132, getitem_133, None, None, None, 8192, 8192, 0.0, True);  permute_917 = permute_157 = permute_158 = permute_159 = getitem_126 = getitem_127 = getitem_132 = getitem_133 = None
        getitem_339 = _scaled_dot_product_cudnn_attention_backward_17[0]
        getitem_340 = _scaled_dot_product_cudnn_attention_backward_17[1]
        getitem_341 = _scaled_dot_product_cudnn_attention_backward_17[2];  _scaled_dot_product_cudnn_attention_backward_17 = None
        permute_918 = torch.ops.aten.permute.default(getitem_341, [0, 2, 1, 3]);  getitem_341 = None
        permute_919 = torch.ops.aten.permute.default(getitem_340, [0, 2, 1, 3]);  getitem_340 = None
        permute_920 = torch.ops.aten.permute.default(getitem_339, [0, 2, 1, 3]);  getitem_339 = None
        view_1512 = torch.ops.aten.view.default(permute_918, [2, 8192, 8, 4, 128]);  permute_918 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(view_1512, [3], True);  view_1512 = None
        squeeze_34 = torch.ops.aten.squeeze.dim(sum_107, 3);  sum_107 = None
        view_1513 = torch.ops.aten.view.default(permute_919, [2, 8192, 8, 4, 128]);  permute_919 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(view_1513, [3], True);  view_1513 = None
        squeeze_35 = torch.ops.aten.squeeze.dim(sum_108, 3);  sum_108 = None
        convert_element_type_2021 = torch.ops.prims.convert_element_type.default(squeeze_35, torch.float32);  squeeze_35 = None
        convert_element_type_2022 = torch.ops.prims.convert_element_type.default(permute_920, torch.float32);  permute_920 = None
        view_1514 = torch.ops.aten.view.default(convert_element_type_2021, [2, 8192, 8, 64, 2]);  convert_element_type_2021 = None
        view_as_complex_98 = torch.ops.aten.view_as_complex.default(view_1514);  view_1514 = None
        mul_616 = torch.ops.aten.mul.Tensor(view_as_complex_98, _conj);  view_as_complex_98 = None
        view_1515 = torch.ops.aten.view.default(convert_element_type_2022, [2, 8192, 32, 64, 2]);  convert_element_type_2022 = None
        view_as_complex_99 = torch.ops.aten.view_as_complex.default(view_1515);  view_1515 = None
        mul_617 = torch.ops.aten.mul.Tensor(view_as_complex_99, _conj);  view_as_complex_99 = None
        view_as_real_98 = torch.ops.aten.view_as_real.default(mul_616);  mul_616 = None
        view_1516 = torch.ops.aten.view.default(view_as_real_98, [2, 8192, 8, 128]);  view_as_real_98 = None
        convert_element_type_2023 = torch.ops.prims.convert_element_type.default(view_1516, torch.bfloat16);  view_1516 = None
        view_as_real_99 = torch.ops.aten.view_as_real.default(mul_617);  mul_617 = None
        view_1517 = torch.ops.aten.view.default(view_as_real_99, [2, 8192, 32, 128]);  view_as_real_99 = None
        convert_element_type_2024 = torch.ops.prims.convert_element_type.default(view_1517, torch.bfloat16);  view_1517 = None
        view_1518 = torch.ops.aten.view.default(squeeze_34, [2, 8192, 1024]);  squeeze_34 = None
        view_1519 = torch.ops.aten.view.default(convert_element_type_2023, [2, 8192, 1024]);  convert_element_type_2023 = None
        view_1520 = torch.ops.aten.view.default(convert_element_type_2024, [2, 8192, 4096]);  convert_element_type_2024 = None
        view_1521 = torch.ops.aten.view.default(view_1518, [16384, 1024]);  view_1518 = None
        permute_921 = torch.ops.aten.permute.default(view_1521, [1, 0])
        mm_473 = torch.ops.aten.mm.default(permute_921, view_479);  permute_921 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        all_gather_into_tensor_130 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_472, 256, '0');  convert_element_type_472 = None
        wait_tensor_130 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_130);  all_gather_into_tensor_130 = None
        permute_156 = torch.ops.aten.permute.default(wait_tensor_130, [1, 0]);  wait_tensor_130 = None
        permute_923 = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_474 = torch.ops.aten.mm.default(view_1521, permute_923);  view_1521 = permute_923 = None
        view_1522 = torch.ops.aten.view.default(mm_474, [2, 8192, 4096]);  mm_474 = None
        convert_element_type_2029 = torch.ops.prims.convert_element_type.default(mm_473, torch.float32);  mm_473 = None
        reduce_scatter_tensor_160 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2029, 'avg', 256, '0');  convert_element_type_2029 = None
        wait_tensor_451 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_160);  reduce_scatter_tensor_160 = None
        view_1523 = torch.ops.aten.view.default(view_1519, [16384, 1024]);  view_1519 = None
        permute_925 = torch.ops.aten.permute.default(view_1523, [1, 0])
        mm_475 = torch.ops.aten.mm.default(permute_925, view_479);  permute_925 = None
        permute_927 = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_476 = torch.ops.aten.mm.default(view_1523, permute_927);  view_1523 = permute_927 = None
        view_1524 = torch.ops.aten.view.default(mm_476, [2, 8192, 4096]);  mm_476 = None
        add_252 = torch.ops.aten.add.Tensor(view_1522, view_1524);  view_1522 = view_1524 = None
        convert_element_type_2034 = torch.ops.prims.convert_element_type.default(mm_475, torch.float32);  mm_475 = None
        reduce_scatter_tensor_161 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2034, 'avg', 256, '0');  convert_element_type_2034 = None
        wait_tensor_452 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_161);  reduce_scatter_tensor_161 = None
        view_1525 = torch.ops.aten.view.default(view_1520, [16384, 4096]);  view_1520 = None
        permute_929 = torch.ops.aten.permute.default(view_1525, [1, 0])
        mm_477 = torch.ops.aten.mm.default(permute_929, view_479);  permute_929 = view_479 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        all_gather_into_tensor_128 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_466, 256, '0');  convert_element_type_466 = None
        wait_tensor_128 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_128);  all_gather_into_tensor_128 = None
        permute_154 = torch.ops.aten.permute.default(wait_tensor_128, [1, 0]);  wait_tensor_128 = None
        permute_931 = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_478 = torch.ops.aten.mm.default(view_1525, permute_931);  view_1525 = permute_931 = None
        view_1526 = torch.ops.aten.view.default(mm_478, [2, 8192, 4096]);  mm_478 = None
        add_253 = torch.ops.aten.add.Tensor(add_252, view_1526);  add_252 = view_1526 = None
        convert_element_type_2039 = torch.ops.prims.convert_element_type.default(mm_477, torch.float32);  mm_477 = None
        reduce_scatter_tensor_162 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2039, 'avg', 256, '0');  convert_element_type_2039 = None
        wait_tensor_453 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_162);  reduce_scatter_tensor_162 = None
        convert_element_type_2040 = torch.ops.prims.convert_element_type.default(add_253, torch.float32);  add_253 = None
        convert_element_type_2042 = torch.ops.prims.convert_element_type.default(wait_tensor_127, torch.float32);  wait_tensor_127 = None
        mul_618 = torch.ops.aten.mul.Tensor(convert_element_type_2040, convert_element_type_2042);  convert_element_type_2042 = None
        mul_620 = torch.ops.aten.mul.Tensor(mul_112, mul_618)
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
        div_36 = torch.ops.aten.div.Tensor(mul_112, 4096)
        mul_621 = torch.ops.aten.mul.Tensor(div_36, sum_109);  div_36 = sum_109 = None
        sub_54 = torch.ops.aten.sub.Tensor(mul_618, mul_621);  mul_618 = mul_621 = None
        mul_622 = torch.ops.aten.mul.Tensor(sub_54, rsqrt_28);  sub_54 = rsqrt_28 = None
        mul_623 = torch.ops.aten.mul.Tensor(convert_element_type_2040, mul_112);  convert_element_type_2040 = mul_112 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1]);  mul_623 = None
        convert_element_type_2043 = torch.ops.prims.convert_element_type.default(mul_622, torch.bfloat16);  mul_622 = None
        add_254 = torch.ops.aten.add.Tensor(add_251, convert_element_type_2043);  add_251 = convert_element_type_2043 = None
        convert_element_type_default_29 = torch.ops.prims.convert_element_type.default(sum_110, torch.float32);  sum_110 = None
        reduce_scatter_tensor_163 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_29, 'avg', 256, '0');  convert_element_type_default_29 = None
        wait_tensor_454 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_163);  reduce_scatter_tensor_163 = None
        view_1527 = torch.ops.aten.view.default(add_254, [16384, 4096])
        permute_933 = torch.ops.aten.permute.default(view_1527, [1, 0])
        permute_149 = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_463 = torch.ops.aten.view.default(permute_149, [2, 8192, -1]);  permute_149 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        all_gather_into_tensor_122 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_446, 256, '0');  convert_element_type_446 = None
        wait_tensor_122 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_122);  all_gather_into_tensor_122 = None
        permute_150 = torch.ops.aten.permute.default(wait_tensor_122, [1, 0]);  wait_tensor_122 = None
        view_465 = torch.ops.aten.view.default(view_463, [16384, 4096]);  view_463 = None
        mm_94 = torch.ops.aten.mm.default(view_465, permute_150)
        view_466 = torch.ops.aten.view.default(mm_94, [2, 8192, 4096]);  mm_94 = None
        add_53 = torch.ops.aten.add.Tensor(add_51, view_466);  view_466 = None
        convert_element_type_449 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        all_gather_into_tensor_123 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_449, 256, '0');  convert_element_type_449 = None
        wait_tensor_123 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_123);  all_gather_into_tensor_123 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_450, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [2], True);  pow_28 = None
        add_54 = torch.ops.aten.add.Scalar(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_27);  convert_element_type_450 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, wait_tensor_123)
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_469 = torch.ops.aten.view.default(convert_element_type_451, [16384, 4096]);  convert_element_type_451 = None
        view_470 = torch.ops.aten.view.default(mm_95, [2, 8192, 14336]);  mm_95 = None
        convert_element_type_455 = torch.ops.prims.convert_element_type.default(view_470, torch.float32);  view_470 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_455)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_13);  sigmoid_13 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        all_gather_into_tensor_125 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_457, 256, '0');  convert_element_type_457 = None
        wait_tensor_125 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_125);  all_gather_into_tensor_125 = None
        permute_152 = torch.ops.aten.permute.default(wait_tensor_125, [1, 0]);  wait_tensor_125 = None
        mm_96 = torch.ops.aten.mm.default(view_469, permute_152)
        view_473 = torch.ops.aten.view.default(mm_96, [2, 8192, 14336]);  mm_96 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_456, view_473)
        view_475 = torch.ops.aten.view.default(mul_111, [16384, 14336]);  mul_111 = None
        mm_479 = torch.ops.aten.mm.default(permute_933, view_475);  permute_933 = view_475 = None
        convert_element_type_460 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        all_gather_into_tensor_126 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_460, 256, '0');  convert_element_type_460 = None
        wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_126);  all_gather_into_tensor_126 = None
        permute_153 = torch.ops.aten.permute.default(wait_tensor_126, [1, 0]);  wait_tensor_126 = None
        permute_935 = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_480 = torch.ops.aten.mm.default(view_1527, permute_935);  view_1527 = permute_935 = None
        view_1528 = torch.ops.aten.view.default(mm_480, [2, 8192, 14336]);  mm_480 = None
        convert_element_type_2050 = torch.ops.prims.convert_element_type.default(mm_479, torch.float32);  mm_479 = None
        reduce_scatter_tensor_164 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2050, 'avg', 256, '0');  convert_element_type_2050 = None
        wait_tensor_455 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_164);  reduce_scatter_tensor_164 = None
        mul_624 = torch.ops.aten.mul.Tensor(view_1528, convert_element_type_456);  convert_element_type_456 = None
        mul_625 = torch.ops.aten.mul.Tensor(view_1528, view_473);  view_1528 = view_473 = None
        view_1529 = torch.ops.aten.view.default(mul_624, [16384, 14336]);  mul_624 = None
        permute_937 = torch.ops.aten.permute.default(view_1529, [1, 0])
        mm_481 = torch.ops.aten.mm.default(permute_937, view_469);  permute_937 = None
        permute_939 = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_482 = torch.ops.aten.mm.default(view_1529, permute_939);  view_1529 = permute_939 = None
        view_1530 = torch.ops.aten.view.default(mm_482, [2, 8192, 4096]);  mm_482 = None
        convert_element_type_2055 = torch.ops.prims.convert_element_type.default(mm_481, torch.float32);  mm_481 = None
        reduce_scatter_tensor_165 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2055, 'avg', 256, '0');  convert_element_type_2055 = None
        wait_tensor_456 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_165);  reduce_scatter_tensor_165 = None
        convert_element_type_2056 = torch.ops.prims.convert_element_type.default(mul_625, torch.float32);  mul_625 = None
        neg_18 = torch.ops.aten.neg.default(convert_element_type_455)
        exp_18 = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_255 = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(add_255);  add_255 = None
        mul_626 = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        mul_627 = torch.ops.aten.mul.Tensor(convert_element_type_2056, mul_626);  convert_element_type_2056 = None
        sub_55 = torch.ops.aten.sub.Tensor(1, mul_626);  mul_626 = None
        mul_628 = torch.ops.aten.mul.Tensor(convert_element_type_455, sub_55);  convert_element_type_455 = sub_55 = None
        add_256 = torch.ops.aten.add.Tensor(mul_628, 1);  mul_628 = None
        mul_629 = torch.ops.aten.mul.Tensor(mul_627, add_256);  mul_627 = add_256 = None
        convert_element_type_2058 = torch.ops.prims.convert_element_type.default(mul_629, torch.bfloat16);  mul_629 = None
        view_1531 = torch.ops.aten.view.default(convert_element_type_2058, [16384, 14336]);  convert_element_type_2058 = None
        permute_941 = torch.ops.aten.permute.default(view_1531, [1, 0])
        mm_483 = torch.ops.aten.mm.default(permute_941, view_469);  permute_941 = view_469 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        all_gather_into_tensor_124 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_452, 256, '0');  convert_element_type_452 = None
        wait_tensor_124 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_124);  all_gather_into_tensor_124 = None
        permute_151 = torch.ops.aten.permute.default(wait_tensor_124, [1, 0]);  wait_tensor_124 = None
        permute_943 = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_484 = torch.ops.aten.mm.default(view_1531, permute_943);  view_1531 = permute_943 = None
        view_1532 = torch.ops.aten.view.default(mm_484, [2, 8192, 4096]);  mm_484 = None
        add_257 = torch.ops.aten.add.Tensor(view_1530, view_1532);  view_1530 = view_1532 = None
        convert_element_type_2063 = torch.ops.prims.convert_element_type.default(mm_483, torch.float32);  mm_483 = None
        reduce_scatter_tensor_166 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2063, 'avg', 256, '0');  convert_element_type_2063 = None
        wait_tensor_457 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_166);  reduce_scatter_tensor_166 = None
        convert_element_type_2064 = torch.ops.prims.convert_element_type.default(add_257, torch.float32);  add_257 = None
        convert_element_type_2066 = torch.ops.prims.convert_element_type.default(wait_tensor_123, torch.float32);  wait_tensor_123 = None
        mul_630 = torch.ops.aten.mul.Tensor(convert_element_type_2064, convert_element_type_2066);  convert_element_type_2066 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_108, mul_630)
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_632, [2], True);  mul_632 = None
        div_37 = torch.ops.aten.div.Tensor(mul_108, 4096)
        mul_633 = torch.ops.aten.mul.Tensor(div_37, sum_111);  div_37 = sum_111 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_630, mul_633);  mul_630 = mul_633 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_56, rsqrt_27);  sub_56 = rsqrt_27 = None
        mul_635 = torch.ops.aten.mul.Tensor(convert_element_type_2064, mul_108);  convert_element_type_2064 = mul_108 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_635, [0, 1]);  mul_635 = None
        convert_element_type_2067 = torch.ops.prims.convert_element_type.default(mul_634, torch.bfloat16);  mul_634 = None
        add_258 = torch.ops.aten.add.Tensor(add_254, convert_element_type_2067);  add_254 = convert_element_type_2067 = None
        convert_element_type_default_28 = torch.ops.prims.convert_element_type.default(sum_112, torch.float32);  sum_112 = None
        reduce_scatter_tensor_167 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_28, 'avg', 256, '0');  convert_element_type_default_28 = None
        wait_tensor_458 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_167);  reduce_scatter_tensor_167 = None
        view_1533 = torch.ops.aten.view.default(add_258, [16384, 4096])
        permute_945 = torch.ops.aten.permute.default(view_1533, [1, 0])
        mm_485 = torch.ops.aten.mm.default(permute_945, view_465);  permute_945 = view_465 = None
        permute_947 = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        mm_486 = torch.ops.aten.mm.default(view_1533, permute_947);  view_1533 = permute_947 = None
        view_1534 = torch.ops.aten.view.default(mm_486, [2, 8192, 4096]);  mm_486 = None
        convert_element_type_2074 = torch.ops.prims.convert_element_type.default(mm_485, torch.float32);  mm_485 = None
        reduce_scatter_tensor_168 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2074, 'avg', 256, '0');  convert_element_type_2074 = None
        wait_tensor_459 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_168);  reduce_scatter_tensor_168 = None
        view_1535 = torch.ops.aten.view.default(view_1534, [2, 8192, 32, 128]);  view_1534 = None
        permute_949 = torch.ops.aten.permute.default(view_1535, [0, 2, 1, 3]);  view_1535 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        all_gather_into_tensor_118 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_430, 256, '0');  convert_element_type_430 = None
        wait_tensor_118 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_118);  all_gather_into_tensor_118 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_52 = torch.ops.aten.add.Scalar(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_431, rsqrt_26);  convert_element_type_431 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, wait_tensor_118)
        convert_element_type_432 = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        view_445 = torch.ops.aten.view.default(convert_element_type_432, [16384, 4096]);  convert_element_type_432 = None
        view_446 = torch.ops.aten.view.default(mm_91, [2, 8192, 4096]);  mm_91 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        all_gather_into_tensor_120 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_436, 256, '0');  convert_element_type_436 = None
        wait_tensor_120 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_120);  all_gather_into_tensor_120 = None
        permute_144 = torch.ops.aten.permute.default(wait_tensor_120, [1, 0]);  wait_tensor_120 = None
        mm_92 = torch.ops.aten.mm.default(view_445, permute_144)
        view_449 = torch.ops.aten.view.default(mm_92, [2, 8192, 1024]);  mm_92 = None
        view_452 = torch.ops.aten.view.default(mm_93, [2, 8192, 1024]);  mm_93 = None
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
        _scaled_dot_product_cudnn_attention_backward_18 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_949, permute_146, permute_147, permute_148, getitem_117, getitem_118, getitem_123, getitem_124, None, None, None, 8192, 8192, 0.0, True);  permute_949 = permute_146 = permute_147 = permute_148 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_342 = _scaled_dot_product_cudnn_attention_backward_18[0]
        getitem_343 = _scaled_dot_product_cudnn_attention_backward_18[1]
        getitem_344 = _scaled_dot_product_cudnn_attention_backward_18[2];  _scaled_dot_product_cudnn_attention_backward_18 = None
        permute_950 = torch.ops.aten.permute.default(getitem_344, [0, 2, 1, 3]);  getitem_344 = None
        permute_951 = torch.ops.aten.permute.default(getitem_343, [0, 2, 1, 3]);  getitem_343 = None
        permute_952 = torch.ops.aten.permute.default(getitem_342, [0, 2, 1, 3]);  getitem_342 = None
        view_1536 = torch.ops.aten.view.default(permute_950, [2, 8192, 8, 4, 128]);  permute_950 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(view_1536, [3], True);  view_1536 = None
        squeeze_36 = torch.ops.aten.squeeze.dim(sum_113, 3);  sum_113 = None
        view_1537 = torch.ops.aten.view.default(permute_951, [2, 8192, 8, 4, 128]);  permute_951 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(view_1537, [3], True);  view_1537 = None
        squeeze_37 = torch.ops.aten.squeeze.dim(sum_114, 3);  sum_114 = None
        convert_element_type_2075 = torch.ops.prims.convert_element_type.default(squeeze_37, torch.float32);  squeeze_37 = None
        convert_element_type_2076 = torch.ops.prims.convert_element_type.default(permute_952, torch.float32);  permute_952 = None
        view_1538 = torch.ops.aten.view.default(convert_element_type_2075, [2, 8192, 8, 64, 2]);  convert_element_type_2075 = None
        view_as_complex_100 = torch.ops.aten.view_as_complex.default(view_1538);  view_1538 = None
        mul_636 = torch.ops.aten.mul.Tensor(view_as_complex_100, _conj);  view_as_complex_100 = None
        view_1539 = torch.ops.aten.view.default(convert_element_type_2076, [2, 8192, 32, 64, 2]);  convert_element_type_2076 = None
        view_as_complex_101 = torch.ops.aten.view_as_complex.default(view_1539);  view_1539 = None
        mul_637 = torch.ops.aten.mul.Tensor(view_as_complex_101, _conj);  view_as_complex_101 = None
        view_as_real_100 = torch.ops.aten.view_as_real.default(mul_636);  mul_636 = None
        view_1540 = torch.ops.aten.view.default(view_as_real_100, [2, 8192, 8, 128]);  view_as_real_100 = None
        convert_element_type_2077 = torch.ops.prims.convert_element_type.default(view_1540, torch.bfloat16);  view_1540 = None
        view_as_real_101 = torch.ops.aten.view_as_real.default(mul_637);  mul_637 = None
        view_1541 = torch.ops.aten.view.default(view_as_real_101, [2, 8192, 32, 128]);  view_as_real_101 = None
        convert_element_type_2078 = torch.ops.prims.convert_element_type.default(view_1541, torch.bfloat16);  view_1541 = None
        view_1542 = torch.ops.aten.view.default(squeeze_36, [2, 8192, 1024]);  squeeze_36 = None
        view_1543 = torch.ops.aten.view.default(convert_element_type_2077, [2, 8192, 1024]);  convert_element_type_2077 = None
        view_1544 = torch.ops.aten.view.default(convert_element_type_2078, [2, 8192, 4096]);  convert_element_type_2078 = None
        view_1545 = torch.ops.aten.view.default(view_1542, [16384, 1024]);  view_1542 = None
        permute_953 = torch.ops.aten.permute.default(view_1545, [1, 0])
        mm_487 = torch.ops.aten.mm.default(permute_953, view_445);  permute_953 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        all_gather_into_tensor_121 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_439, 256, '0');  convert_element_type_439 = None
        wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_121);  all_gather_into_tensor_121 = None
        permute_145 = torch.ops.aten.permute.default(wait_tensor_121, [1, 0]);  wait_tensor_121 = None
        permute_955 = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_488 = torch.ops.aten.mm.default(view_1545, permute_955);  view_1545 = permute_955 = None
        view_1546 = torch.ops.aten.view.default(mm_488, [2, 8192, 4096]);  mm_488 = None
        convert_element_type_2083 = torch.ops.prims.convert_element_type.default(mm_487, torch.float32);  mm_487 = None
        reduce_scatter_tensor_169 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2083, 'avg', 256, '0');  convert_element_type_2083 = None
        wait_tensor_460 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_169);  reduce_scatter_tensor_169 = None
        view_1547 = torch.ops.aten.view.default(view_1543, [16384, 1024]);  view_1543 = None
        permute_957 = torch.ops.aten.permute.default(view_1547, [1, 0])
        mm_489 = torch.ops.aten.mm.default(permute_957, view_445);  permute_957 = None
        permute_959 = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_490 = torch.ops.aten.mm.default(view_1547, permute_959);  view_1547 = permute_959 = None
        view_1548 = torch.ops.aten.view.default(mm_490, [2, 8192, 4096]);  mm_490 = None
        add_259 = torch.ops.aten.add.Tensor(view_1546, view_1548);  view_1546 = view_1548 = None
        convert_element_type_2088 = torch.ops.prims.convert_element_type.default(mm_489, torch.float32);  mm_489 = None
        reduce_scatter_tensor_170 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2088, 'avg', 256, '0');  convert_element_type_2088 = None
        wait_tensor_461 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_170);  reduce_scatter_tensor_170 = None
        view_1549 = torch.ops.aten.view.default(view_1544, [16384, 4096]);  view_1544 = None
        permute_961 = torch.ops.aten.permute.default(view_1549, [1, 0])
        mm_491 = torch.ops.aten.mm.default(permute_961, view_445);  permute_961 = view_445 = None
        convert_element_type_433 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        all_gather_into_tensor_119 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_433, 256, '0');  convert_element_type_433 = None
        wait_tensor_119 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_119);  all_gather_into_tensor_119 = None
        permute_143 = torch.ops.aten.permute.default(wait_tensor_119, [1, 0]);  wait_tensor_119 = None
        permute_963 = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_492 = torch.ops.aten.mm.default(view_1549, permute_963);  view_1549 = permute_963 = None
        view_1550 = torch.ops.aten.view.default(mm_492, [2, 8192, 4096]);  mm_492 = None
        add_260 = torch.ops.aten.add.Tensor(add_259, view_1550);  add_259 = view_1550 = None
        convert_element_type_2093 = torch.ops.prims.convert_element_type.default(mm_491, torch.float32);  mm_491 = None
        reduce_scatter_tensor_171 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2093, 'avg', 256, '0');  convert_element_type_2093 = None
        wait_tensor_462 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_171);  reduce_scatter_tensor_171 = None
        convert_element_type_2094 = torch.ops.prims.convert_element_type.default(add_260, torch.float32);  add_260 = None
        convert_element_type_2096 = torch.ops.prims.convert_element_type.default(wait_tensor_118, torch.float32);  wait_tensor_118 = None
        mul_638 = torch.ops.aten.mul.Tensor(convert_element_type_2094, convert_element_type_2096);  convert_element_type_2096 = None
        mul_640 = torch.ops.aten.mul.Tensor(mul_104, mul_638)
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_640, [2], True);  mul_640 = None
        div_38 = torch.ops.aten.div.Tensor(mul_104, 4096)
        mul_641 = torch.ops.aten.mul.Tensor(div_38, sum_115);  div_38 = sum_115 = None
        sub_57 = torch.ops.aten.sub.Tensor(mul_638, mul_641);  mul_638 = mul_641 = None
        mul_642 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_26);  sub_57 = rsqrt_26 = None
        mul_643 = torch.ops.aten.mul.Tensor(convert_element_type_2094, mul_104);  convert_element_type_2094 = mul_104 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
        convert_element_type_2097 = torch.ops.prims.convert_element_type.default(mul_642, torch.bfloat16);  mul_642 = None
        add_261 = torch.ops.aten.add.Tensor(add_258, convert_element_type_2097);  add_258 = convert_element_type_2097 = None
        convert_element_type_default_27 = torch.ops.prims.convert_element_type.default(sum_116, torch.float32);  sum_116 = None
        reduce_scatter_tensor_172 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_27, 'avg', 256, '0');  convert_element_type_default_27 = None
        wait_tensor_463 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_172);  reduce_scatter_tensor_172 = None
        view_1551 = torch.ops.aten.view.default(add_261, [16384, 4096])
        permute_965 = torch.ops.aten.permute.default(view_1551, [1, 0])
        permute_138 = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])
        view_429 = torch.ops.aten.view.default(permute_138, [2, 8192, -1]);  permute_138 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        all_gather_into_tensor_113 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_413, 256, '0');  convert_element_type_413 = None
        wait_tensor_113 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_113);  all_gather_into_tensor_113 = None
        permute_139 = torch.ops.aten.permute.default(wait_tensor_113, [1, 0]);  wait_tensor_113 = None
        view_431 = torch.ops.aten.view.default(view_429, [16384, 4096]);  view_429 = None
        mm_87 = torch.ops.aten.mm.default(view_431, permute_139)
        view_432 = torch.ops.aten.view.default(mm_87, [2, 8192, 4096]);  mm_87 = None
        add_49 = torch.ops.aten.add.Tensor(add_47, view_432);  view_432 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        all_gather_into_tensor_114 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_416, 256, '0');  convert_element_type_416 = None
        wait_tensor_114 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_114);  all_gather_into_tensor_114 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_417, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [2], True);  pow_26 = None
        add_50 = torch.ops.aten.add.Scalar(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_25);  convert_element_type_417 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, wait_tensor_114)
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        view_435 = torch.ops.aten.view.default(convert_element_type_418, [16384, 4096]);  convert_element_type_418 = None
        view_436 = torch.ops.aten.view.default(mm_88, [2, 8192, 14336]);  mm_88 = None
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_422)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_422, sigmoid_12);  sigmoid_12 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        all_gather_into_tensor_116 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_424, 256, '0');  convert_element_type_424 = None
        wait_tensor_116 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_116);  all_gather_into_tensor_116 = None
        permute_141 = torch.ops.aten.permute.default(wait_tensor_116, [1, 0]);  wait_tensor_116 = None
        mm_89 = torch.ops.aten.mm.default(view_435, permute_141)
        view_439 = torch.ops.aten.view.default(mm_89, [2, 8192, 14336]);  mm_89 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_423, view_439)
        view_441 = torch.ops.aten.view.default(mul_103, [16384, 14336]);  mul_103 = None
        mm_493 = torch.ops.aten.mm.default(permute_965, view_441);  permute_965 = view_441 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        all_gather_into_tensor_117 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_427, 256, '0');  convert_element_type_427 = None
        wait_tensor_117 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_117);  all_gather_into_tensor_117 = None
        permute_142 = torch.ops.aten.permute.default(wait_tensor_117, [1, 0]);  wait_tensor_117 = None
        permute_967 = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_494 = torch.ops.aten.mm.default(view_1551, permute_967);  view_1551 = permute_967 = None
        view_1552 = torch.ops.aten.view.default(mm_494, [2, 8192, 14336]);  mm_494 = None
        convert_element_type_2104 = torch.ops.prims.convert_element_type.default(mm_493, torch.float32);  mm_493 = None
        reduce_scatter_tensor_173 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2104, 'avg', 256, '0');  convert_element_type_2104 = None
        wait_tensor_464 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_173);  reduce_scatter_tensor_173 = None
        mul_644 = torch.ops.aten.mul.Tensor(view_1552, convert_element_type_423);  convert_element_type_423 = None
        mul_645 = torch.ops.aten.mul.Tensor(view_1552, view_439);  view_1552 = view_439 = None
        view_1553 = torch.ops.aten.view.default(mul_644, [16384, 14336]);  mul_644 = None
        permute_969 = torch.ops.aten.permute.default(view_1553, [1, 0])
        mm_495 = torch.ops.aten.mm.default(permute_969, view_435);  permute_969 = None
        permute_971 = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_496 = torch.ops.aten.mm.default(view_1553, permute_971);  view_1553 = permute_971 = None
        view_1554 = torch.ops.aten.view.default(mm_496, [2, 8192, 4096]);  mm_496 = None
        convert_element_type_2109 = torch.ops.prims.convert_element_type.default(mm_495, torch.float32);  mm_495 = None
        reduce_scatter_tensor_174 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2109, 'avg', 256, '0');  convert_element_type_2109 = None
        wait_tensor_465 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_174);  reduce_scatter_tensor_174 = None
        convert_element_type_2110 = torch.ops.prims.convert_element_type.default(mul_645, torch.float32);  mul_645 = None
        neg_19 = torch.ops.aten.neg.default(convert_element_type_422)
        exp_19 = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_262 = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(add_262);  add_262 = None
        mul_646 = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        mul_647 = torch.ops.aten.mul.Tensor(convert_element_type_2110, mul_646);  convert_element_type_2110 = None
        sub_58 = torch.ops.aten.sub.Tensor(1, mul_646);  mul_646 = None
        mul_648 = torch.ops.aten.mul.Tensor(convert_element_type_422, sub_58);  convert_element_type_422 = sub_58 = None
        add_263 = torch.ops.aten.add.Tensor(mul_648, 1);  mul_648 = None
        mul_649 = torch.ops.aten.mul.Tensor(mul_647, add_263);  mul_647 = add_263 = None
        convert_element_type_2112 = torch.ops.prims.convert_element_type.default(mul_649, torch.bfloat16);  mul_649 = None
        view_1555 = torch.ops.aten.view.default(convert_element_type_2112, [16384, 14336]);  convert_element_type_2112 = None
        permute_973 = torch.ops.aten.permute.default(view_1555, [1, 0])
        mm_497 = torch.ops.aten.mm.default(permute_973, view_435);  permute_973 = view_435 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        all_gather_into_tensor_115 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_419, 256, '0');  convert_element_type_419 = None
        wait_tensor_115 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_115);  all_gather_into_tensor_115 = None
        permute_140 = torch.ops.aten.permute.default(wait_tensor_115, [1, 0]);  wait_tensor_115 = None
        permute_975 = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_498 = torch.ops.aten.mm.default(view_1555, permute_975);  view_1555 = permute_975 = None
        view_1556 = torch.ops.aten.view.default(mm_498, [2, 8192, 4096]);  mm_498 = None
        add_264 = torch.ops.aten.add.Tensor(view_1554, view_1556);  view_1554 = view_1556 = None
        convert_element_type_2117 = torch.ops.prims.convert_element_type.default(mm_497, torch.float32);  mm_497 = None
        reduce_scatter_tensor_175 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2117, 'avg', 256, '0');  convert_element_type_2117 = None
        wait_tensor_466 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_175);  reduce_scatter_tensor_175 = None
        convert_element_type_2118 = torch.ops.prims.convert_element_type.default(add_264, torch.float32);  add_264 = None
        convert_element_type_2120 = torch.ops.prims.convert_element_type.default(wait_tensor_114, torch.float32);  wait_tensor_114 = None
        mul_650 = torch.ops.aten.mul.Tensor(convert_element_type_2118, convert_element_type_2120);  convert_element_type_2120 = None
        mul_652 = torch.ops.aten.mul.Tensor(mul_100, mul_650)
        sum_117 = torch.ops.aten.sum.dim_IntList(mul_652, [2], True);  mul_652 = None
        div_39 = torch.ops.aten.div.Tensor(mul_100, 4096)
        mul_653 = torch.ops.aten.mul.Tensor(div_39, sum_117);  div_39 = sum_117 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_650, mul_653);  mul_650 = mul_653 = None
        mul_654 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_25);  sub_59 = rsqrt_25 = None
        mul_655 = torch.ops.aten.mul.Tensor(convert_element_type_2118, mul_100);  convert_element_type_2118 = mul_100 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_655, [0, 1]);  mul_655 = None
        convert_element_type_2121 = torch.ops.prims.convert_element_type.default(mul_654, torch.bfloat16);  mul_654 = None
        add_265 = torch.ops.aten.add.Tensor(add_261, convert_element_type_2121);  add_261 = convert_element_type_2121 = None
        convert_element_type_default_26 = torch.ops.prims.convert_element_type.default(sum_118, torch.float32);  sum_118 = None
        reduce_scatter_tensor_176 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_26, 'avg', 256, '0');  convert_element_type_default_26 = None
        wait_tensor_467 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_176);  reduce_scatter_tensor_176 = None
        view_1557 = torch.ops.aten.view.default(add_265, [16384, 4096])
        permute_977 = torch.ops.aten.permute.default(view_1557, [1, 0])
        mm_499 = torch.ops.aten.mm.default(permute_977, view_431);  permute_977 = view_431 = None
        permute_979 = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        mm_500 = torch.ops.aten.mm.default(view_1557, permute_979);  view_1557 = permute_979 = None
        view_1558 = torch.ops.aten.view.default(mm_500, [2, 8192, 4096]);  mm_500 = None
        convert_element_type_2128 = torch.ops.prims.convert_element_type.default(mm_499, torch.float32);  mm_499 = None
        reduce_scatter_tensor_177 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2128, 'avg', 256, '0');  convert_element_type_2128 = None
        wait_tensor_468 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_177);  reduce_scatter_tensor_177 = None
        view_1559 = torch.ops.aten.view.default(view_1558, [2, 8192, 32, 128]);  view_1558 = None
        permute_981 = torch.ops.aten.permute.default(view_1559, [0, 2, 1, 3]);  view_1559 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        all_gather_into_tensor_109 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_397, 256, '0');  convert_element_type_397 = None
        wait_tensor_109 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_109);  all_gather_into_tensor_109 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_48 = torch.ops.aten.add.Scalar(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_24);  convert_element_type_398 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, wait_tensor_109)
        convert_element_type_399 = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None
        view_411 = torch.ops.aten.view.default(convert_element_type_399, [16384, 4096]);  convert_element_type_399 = None
        view_412 = torch.ops.aten.view.default(mm_84, [2, 8192, 4096]);  mm_84 = None
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        all_gather_into_tensor_111 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_403, 256, '0');  convert_element_type_403 = None
        wait_tensor_111 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_111);  all_gather_into_tensor_111 = None
        permute_133 = torch.ops.aten.permute.default(wait_tensor_111, [1, 0]);  wait_tensor_111 = None
        mm_85 = torch.ops.aten.mm.default(view_411, permute_133)
        view_415 = torch.ops.aten.view.default(mm_85, [2, 8192, 1024]);  mm_85 = None
        view_418 = torch.ops.aten.view.default(mm_86, [2, 8192, 1024]);  mm_86 = None
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
        _scaled_dot_product_cudnn_attention_backward_19 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_981, permute_135, permute_136, permute_137, getitem_108, getitem_109, getitem_114, getitem_115, None, None, None, 8192, 8192, 0.0, True);  permute_981 = permute_135 = permute_136 = permute_137 = getitem_108 = getitem_109 = getitem_114 = getitem_115 = None
        getitem_345 = _scaled_dot_product_cudnn_attention_backward_19[0]
        getitem_346 = _scaled_dot_product_cudnn_attention_backward_19[1]
        getitem_347 = _scaled_dot_product_cudnn_attention_backward_19[2];  _scaled_dot_product_cudnn_attention_backward_19 = None
        permute_982 = torch.ops.aten.permute.default(getitem_347, [0, 2, 1, 3]);  getitem_347 = None
        permute_983 = torch.ops.aten.permute.default(getitem_346, [0, 2, 1, 3]);  getitem_346 = None
        permute_984 = torch.ops.aten.permute.default(getitem_345, [0, 2, 1, 3]);  getitem_345 = None
        view_1560 = torch.ops.aten.view.default(permute_982, [2, 8192, 8, 4, 128]);  permute_982 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(view_1560, [3], True);  view_1560 = None
        squeeze_38 = torch.ops.aten.squeeze.dim(sum_119, 3);  sum_119 = None
        view_1561 = torch.ops.aten.view.default(permute_983, [2, 8192, 8, 4, 128]);  permute_983 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(view_1561, [3], True);  view_1561 = None
        squeeze_39 = torch.ops.aten.squeeze.dim(sum_120, 3);  sum_120 = None
        convert_element_type_2129 = torch.ops.prims.convert_element_type.default(squeeze_39, torch.float32);  squeeze_39 = None
        convert_element_type_2130 = torch.ops.prims.convert_element_type.default(permute_984, torch.float32);  permute_984 = None
        view_1562 = torch.ops.aten.view.default(convert_element_type_2129, [2, 8192, 8, 64, 2]);  convert_element_type_2129 = None
        view_as_complex_102 = torch.ops.aten.view_as_complex.default(view_1562);  view_1562 = None
        mul_656 = torch.ops.aten.mul.Tensor(view_as_complex_102, _conj);  view_as_complex_102 = None
        view_1563 = torch.ops.aten.view.default(convert_element_type_2130, [2, 8192, 32, 64, 2]);  convert_element_type_2130 = None
        view_as_complex_103 = torch.ops.aten.view_as_complex.default(view_1563);  view_1563 = None
        mul_657 = torch.ops.aten.mul.Tensor(view_as_complex_103, _conj);  view_as_complex_103 = None
        view_as_real_102 = torch.ops.aten.view_as_real.default(mul_656);  mul_656 = None
        view_1564 = torch.ops.aten.view.default(view_as_real_102, [2, 8192, 8, 128]);  view_as_real_102 = None
        convert_element_type_2131 = torch.ops.prims.convert_element_type.default(view_1564, torch.bfloat16);  view_1564 = None
        view_as_real_103 = torch.ops.aten.view_as_real.default(mul_657);  mul_657 = None
        view_1565 = torch.ops.aten.view.default(view_as_real_103, [2, 8192, 32, 128]);  view_as_real_103 = None
        convert_element_type_2132 = torch.ops.prims.convert_element_type.default(view_1565, torch.bfloat16);  view_1565 = None
        view_1566 = torch.ops.aten.view.default(squeeze_38, [2, 8192, 1024]);  squeeze_38 = None
        view_1567 = torch.ops.aten.view.default(convert_element_type_2131, [2, 8192, 1024]);  convert_element_type_2131 = None
        view_1568 = torch.ops.aten.view.default(convert_element_type_2132, [2, 8192, 4096]);  convert_element_type_2132 = None
        view_1569 = torch.ops.aten.view.default(view_1566, [16384, 1024]);  view_1566 = None
        permute_985 = torch.ops.aten.permute.default(view_1569, [1, 0])
        mm_501 = torch.ops.aten.mm.default(permute_985, view_411);  permute_985 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        all_gather_into_tensor_112 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_406, 256, '0');  convert_element_type_406 = None
        wait_tensor_112 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_112);  all_gather_into_tensor_112 = None
        permute_134 = torch.ops.aten.permute.default(wait_tensor_112, [1, 0]);  wait_tensor_112 = None
        permute_987 = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_502 = torch.ops.aten.mm.default(view_1569, permute_987);  view_1569 = permute_987 = None
        view_1570 = torch.ops.aten.view.default(mm_502, [2, 8192, 4096]);  mm_502 = None
        convert_element_type_2137 = torch.ops.prims.convert_element_type.default(mm_501, torch.float32);  mm_501 = None
        reduce_scatter_tensor_178 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2137, 'avg', 256, '0');  convert_element_type_2137 = None
        wait_tensor_469 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_178);  reduce_scatter_tensor_178 = None
        view_1571 = torch.ops.aten.view.default(view_1567, [16384, 1024]);  view_1567 = None
        permute_989 = torch.ops.aten.permute.default(view_1571, [1, 0])
        mm_503 = torch.ops.aten.mm.default(permute_989, view_411);  permute_989 = None
        permute_991 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_504 = torch.ops.aten.mm.default(view_1571, permute_991);  view_1571 = permute_991 = None
        view_1572 = torch.ops.aten.view.default(mm_504, [2, 8192, 4096]);  mm_504 = None
        add_266 = torch.ops.aten.add.Tensor(view_1570, view_1572);  view_1570 = view_1572 = None
        convert_element_type_2142 = torch.ops.prims.convert_element_type.default(mm_503, torch.float32);  mm_503 = None
        reduce_scatter_tensor_179 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2142, 'avg', 256, '0');  convert_element_type_2142 = None
        wait_tensor_470 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_179);  reduce_scatter_tensor_179 = None
        view_1573 = torch.ops.aten.view.default(view_1568, [16384, 4096]);  view_1568 = None
        permute_993 = torch.ops.aten.permute.default(view_1573, [1, 0])
        mm_505 = torch.ops.aten.mm.default(permute_993, view_411);  permute_993 = view_411 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        all_gather_into_tensor_110 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_400, 256, '0');  convert_element_type_400 = None
        wait_tensor_110 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_110);  all_gather_into_tensor_110 = None
        permute_132 = torch.ops.aten.permute.default(wait_tensor_110, [1, 0]);  wait_tensor_110 = None
        permute_995 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_506 = torch.ops.aten.mm.default(view_1573, permute_995);  view_1573 = permute_995 = None
        view_1574 = torch.ops.aten.view.default(mm_506, [2, 8192, 4096]);  mm_506 = None
        add_267 = torch.ops.aten.add.Tensor(add_266, view_1574);  add_266 = view_1574 = None
        convert_element_type_2147 = torch.ops.prims.convert_element_type.default(mm_505, torch.float32);  mm_505 = None
        reduce_scatter_tensor_180 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2147, 'avg', 256, '0');  convert_element_type_2147 = None
        wait_tensor_471 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_180);  reduce_scatter_tensor_180 = None
        convert_element_type_2148 = torch.ops.prims.convert_element_type.default(add_267, torch.float32);  add_267 = None
        convert_element_type_2150 = torch.ops.prims.convert_element_type.default(wait_tensor_109, torch.float32);  wait_tensor_109 = None
        mul_658 = torch.ops.aten.mul.Tensor(convert_element_type_2148, convert_element_type_2150);  convert_element_type_2150 = None
        mul_660 = torch.ops.aten.mul.Tensor(mul_96, mul_658)
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_660, [2], True);  mul_660 = None
        div_40 = torch.ops.aten.div.Tensor(mul_96, 4096)
        mul_661 = torch.ops.aten.mul.Tensor(div_40, sum_121);  div_40 = sum_121 = None
        sub_60 = torch.ops.aten.sub.Tensor(mul_658, mul_661);  mul_658 = mul_661 = None
        mul_662 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_24);  sub_60 = rsqrt_24 = None
        mul_663 = torch.ops.aten.mul.Tensor(convert_element_type_2148, mul_96);  convert_element_type_2148 = mul_96 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(mul_663, [0, 1]);  mul_663 = None
        convert_element_type_2151 = torch.ops.prims.convert_element_type.default(mul_662, torch.bfloat16);  mul_662 = None
        add_268 = torch.ops.aten.add.Tensor(add_265, convert_element_type_2151);  add_265 = convert_element_type_2151 = None
        convert_element_type_default_25 = torch.ops.prims.convert_element_type.default(sum_122, torch.float32);  sum_122 = None
        reduce_scatter_tensor_181 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_25, 'avg', 256, '0');  convert_element_type_default_25 = None
        wait_tensor_472 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_181);  reduce_scatter_tensor_181 = None
        view_1575 = torch.ops.aten.view.default(add_268, [16384, 4096])
        permute_997 = torch.ops.aten.permute.default(view_1575, [1, 0])
        permute_127 = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3])
        view_395 = torch.ops.aten.view.default(permute_127, [2, 8192, -1]);  permute_127 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        all_gather_into_tensor_104 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_380, 256, '0');  convert_element_type_380 = None
        wait_tensor_104 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_104);  all_gather_into_tensor_104 = None
        permute_128 = torch.ops.aten.permute.default(wait_tensor_104, [1, 0]);  wait_tensor_104 = None
        view_397 = torch.ops.aten.view.default(view_395, [16384, 4096]);  view_395 = None
        mm_80 = torch.ops.aten.mm.default(view_397, permute_128)
        view_398 = torch.ops.aten.view.default(mm_80, [2, 8192, 4096]);  mm_80 = None
        add_45 = torch.ops.aten.add.Tensor(add_43, view_398);  view_398 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        all_gather_into_tensor_105 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_383, 256, '0');  convert_element_type_383 = None
        wait_tensor_105 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_105);  all_gather_into_tensor_105 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [2], True);  pow_24 = None
        add_46 = torch.ops.aten.add.Scalar(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_23);  convert_element_type_384 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, wait_tensor_105)
        convert_element_type_385 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        view_401 = torch.ops.aten.view.default(convert_element_type_385, [16384, 4096]);  convert_element_type_385 = None
        view_402 = torch.ops.aten.view.default(mm_81, [2, 8192, 14336]);  mm_81 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_389)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_389, sigmoid_11);  sigmoid_11 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        all_gather_into_tensor_107 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_391, 256, '0');  convert_element_type_391 = None
        wait_tensor_107 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_107);  all_gather_into_tensor_107 = None
        permute_130 = torch.ops.aten.permute.default(wait_tensor_107, [1, 0]);  wait_tensor_107 = None
        mm_82 = torch.ops.aten.mm.default(view_401, permute_130)
        view_405 = torch.ops.aten.view.default(mm_82, [2, 8192, 14336]);  mm_82 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_390, view_405)
        view_407 = torch.ops.aten.view.default(mul_95, [16384, 14336]);  mul_95 = None
        mm_507 = torch.ops.aten.mm.default(permute_997, view_407);  permute_997 = view_407 = None
        convert_element_type_394 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        all_gather_into_tensor_108 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_394, 256, '0');  convert_element_type_394 = None
        wait_tensor_108 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_108);  all_gather_into_tensor_108 = None
        permute_131 = torch.ops.aten.permute.default(wait_tensor_108, [1, 0]);  wait_tensor_108 = None
        permute_999 = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_508 = torch.ops.aten.mm.default(view_1575, permute_999);  view_1575 = permute_999 = None
        view_1576 = torch.ops.aten.view.default(mm_508, [2, 8192, 14336]);  mm_508 = None
        convert_element_type_2158 = torch.ops.prims.convert_element_type.default(mm_507, torch.float32);  mm_507 = None
        reduce_scatter_tensor_182 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2158, 'avg', 256, '0');  convert_element_type_2158 = None
        wait_tensor_473 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_182);  reduce_scatter_tensor_182 = None
        mul_664 = torch.ops.aten.mul.Tensor(view_1576, convert_element_type_390);  convert_element_type_390 = None
        mul_665 = torch.ops.aten.mul.Tensor(view_1576, view_405);  view_1576 = view_405 = None
        view_1577 = torch.ops.aten.view.default(mul_664, [16384, 14336]);  mul_664 = None
        permute_1001 = torch.ops.aten.permute.default(view_1577, [1, 0])
        mm_509 = torch.ops.aten.mm.default(permute_1001, view_401);  permute_1001 = None
        permute_1003 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_510 = torch.ops.aten.mm.default(view_1577, permute_1003);  view_1577 = permute_1003 = None
        view_1578 = torch.ops.aten.view.default(mm_510, [2, 8192, 4096]);  mm_510 = None
        convert_element_type_2163 = torch.ops.prims.convert_element_type.default(mm_509, torch.float32);  mm_509 = None
        reduce_scatter_tensor_183 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2163, 'avg', 256, '0');  convert_element_type_2163 = None
        wait_tensor_474 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_183);  reduce_scatter_tensor_183 = None
        convert_element_type_2164 = torch.ops.prims.convert_element_type.default(mul_665, torch.float32);  mul_665 = None
        neg_20 = torch.ops.aten.neg.default(convert_element_type_389)
        exp_20 = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_269 = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(add_269);  add_269 = None
        mul_666 = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        mul_667 = torch.ops.aten.mul.Tensor(convert_element_type_2164, mul_666);  convert_element_type_2164 = None
        sub_61 = torch.ops.aten.sub.Tensor(1, mul_666);  mul_666 = None
        mul_668 = torch.ops.aten.mul.Tensor(convert_element_type_389, sub_61);  convert_element_type_389 = sub_61 = None
        add_270 = torch.ops.aten.add.Tensor(mul_668, 1);  mul_668 = None
        mul_669 = torch.ops.aten.mul.Tensor(mul_667, add_270);  mul_667 = add_270 = None
        convert_element_type_2166 = torch.ops.prims.convert_element_type.default(mul_669, torch.bfloat16);  mul_669 = None
        view_1579 = torch.ops.aten.view.default(convert_element_type_2166, [16384, 14336]);  convert_element_type_2166 = None
        permute_1005 = torch.ops.aten.permute.default(view_1579, [1, 0])
        mm_511 = torch.ops.aten.mm.default(permute_1005, view_401);  permute_1005 = view_401 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        all_gather_into_tensor_106 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_386, 256, '0');  convert_element_type_386 = None
        wait_tensor_106 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_106);  all_gather_into_tensor_106 = None
        permute_129 = torch.ops.aten.permute.default(wait_tensor_106, [1, 0]);  wait_tensor_106 = None
        permute_1007 = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_512 = torch.ops.aten.mm.default(view_1579, permute_1007);  view_1579 = permute_1007 = None
        view_1580 = torch.ops.aten.view.default(mm_512, [2, 8192, 4096]);  mm_512 = None
        add_271 = torch.ops.aten.add.Tensor(view_1578, view_1580);  view_1578 = view_1580 = None
        convert_element_type_2171 = torch.ops.prims.convert_element_type.default(mm_511, torch.float32);  mm_511 = None
        reduce_scatter_tensor_184 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2171, 'avg', 256, '0');  convert_element_type_2171 = None
        wait_tensor_475 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_184);  reduce_scatter_tensor_184 = None
        convert_element_type_2172 = torch.ops.prims.convert_element_type.default(add_271, torch.float32);  add_271 = None
        convert_element_type_2174 = torch.ops.prims.convert_element_type.default(wait_tensor_105, torch.float32);  wait_tensor_105 = None
        mul_670 = torch.ops.aten.mul.Tensor(convert_element_type_2172, convert_element_type_2174);  convert_element_type_2174 = None
        mul_672 = torch.ops.aten.mul.Tensor(mul_92, mul_670)
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_672, [2], True);  mul_672 = None
        div_41 = torch.ops.aten.div.Tensor(mul_92, 4096)
        mul_673 = torch.ops.aten.mul.Tensor(div_41, sum_123);  div_41 = sum_123 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_670, mul_673);  mul_670 = mul_673 = None
        mul_674 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_23);  sub_62 = rsqrt_23 = None
        mul_675 = torch.ops.aten.mul.Tensor(convert_element_type_2172, mul_92);  convert_element_type_2172 = mul_92 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(mul_675, [0, 1]);  mul_675 = None
        convert_element_type_2175 = torch.ops.prims.convert_element_type.default(mul_674, torch.bfloat16);  mul_674 = None
        add_272 = torch.ops.aten.add.Tensor(add_268, convert_element_type_2175);  add_268 = convert_element_type_2175 = None
        convert_element_type_default_24 = torch.ops.prims.convert_element_type.default(sum_124, torch.float32);  sum_124 = None
        reduce_scatter_tensor_185 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_24, 'avg', 256, '0');  convert_element_type_default_24 = None
        wait_tensor_476 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_185);  reduce_scatter_tensor_185 = None
        view_1581 = torch.ops.aten.view.default(add_272, [16384, 4096])
        permute_1009 = torch.ops.aten.permute.default(view_1581, [1, 0])
        mm_513 = torch.ops.aten.mm.default(permute_1009, view_397);  permute_1009 = view_397 = None
        permute_1011 = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        mm_514 = torch.ops.aten.mm.default(view_1581, permute_1011);  view_1581 = permute_1011 = None
        view_1582 = torch.ops.aten.view.default(mm_514, [2, 8192, 4096]);  mm_514 = None
        convert_element_type_2182 = torch.ops.prims.convert_element_type.default(mm_513, torch.float32);  mm_513 = None
        reduce_scatter_tensor_186 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2182, 'avg', 256, '0');  convert_element_type_2182 = None
        wait_tensor_477 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_186);  reduce_scatter_tensor_186 = None
        view_1583 = torch.ops.aten.view.default(view_1582, [2, 8192, 32, 128]);  view_1582 = None
        permute_1013 = torch.ops.aten.permute.default(view_1583, [0, 2, 1, 3]);  view_1583 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        all_gather_into_tensor_100 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_364, 256, '0');  convert_element_type_364 = None
        wait_tensor_100 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_100);  all_gather_into_tensor_100 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_365, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [2], True);  pow_23 = None
        add_44 = torch.ops.aten.add.Scalar(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_365, rsqrt_22);  convert_element_type_365 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, wait_tensor_100)
        convert_element_type_366 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_377 = torch.ops.aten.view.default(convert_element_type_366, [16384, 4096]);  convert_element_type_366 = None
        view_378 = torch.ops.aten.view.default(mm_77, [2, 8192, 4096]);  mm_77 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        all_gather_into_tensor_102 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_370, 256, '0');  convert_element_type_370 = None
        wait_tensor_102 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_102);  all_gather_into_tensor_102 = None
        permute_122 = torch.ops.aten.permute.default(wait_tensor_102, [1, 0]);  wait_tensor_102 = None
        mm_78 = torch.ops.aten.mm.default(view_377, permute_122)
        view_381 = torch.ops.aten.view.default(mm_78, [2, 8192, 1024]);  mm_78 = None
        view_384 = torch.ops.aten.view.default(mm_79, [2, 8192, 1024]);  mm_79 = None
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
        _scaled_dot_product_cudnn_attention_backward_20 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1013, permute_124, permute_125, permute_126, getitem_99, getitem_100, getitem_105, getitem_106, None, None, None, 8192, 8192, 0.0, True);  permute_1013 = permute_124 = permute_125 = permute_126 = getitem_99 = getitem_100 = getitem_105 = getitem_106 = None
        getitem_348 = _scaled_dot_product_cudnn_attention_backward_20[0]
        getitem_349 = _scaled_dot_product_cudnn_attention_backward_20[1]
        getitem_350 = _scaled_dot_product_cudnn_attention_backward_20[2];  _scaled_dot_product_cudnn_attention_backward_20 = None
        permute_1014 = torch.ops.aten.permute.default(getitem_350, [0, 2, 1, 3]);  getitem_350 = None
        permute_1015 = torch.ops.aten.permute.default(getitem_349, [0, 2, 1, 3]);  getitem_349 = None
        permute_1016 = torch.ops.aten.permute.default(getitem_348, [0, 2, 1, 3]);  getitem_348 = None
        view_1584 = torch.ops.aten.view.default(permute_1014, [2, 8192, 8, 4, 128]);  permute_1014 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(view_1584, [3], True);  view_1584 = None
        squeeze_40 = torch.ops.aten.squeeze.dim(sum_125, 3);  sum_125 = None
        view_1585 = torch.ops.aten.view.default(permute_1015, [2, 8192, 8, 4, 128]);  permute_1015 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(view_1585, [3], True);  view_1585 = None
        squeeze_41 = torch.ops.aten.squeeze.dim(sum_126, 3);  sum_126 = None
        convert_element_type_2183 = torch.ops.prims.convert_element_type.default(squeeze_41, torch.float32);  squeeze_41 = None
        convert_element_type_2184 = torch.ops.prims.convert_element_type.default(permute_1016, torch.float32);  permute_1016 = None
        view_1586 = torch.ops.aten.view.default(convert_element_type_2183, [2, 8192, 8, 64, 2]);  convert_element_type_2183 = None
        view_as_complex_104 = torch.ops.aten.view_as_complex.default(view_1586);  view_1586 = None
        mul_676 = torch.ops.aten.mul.Tensor(view_as_complex_104, _conj);  view_as_complex_104 = None
        view_1587 = torch.ops.aten.view.default(convert_element_type_2184, [2, 8192, 32, 64, 2]);  convert_element_type_2184 = None
        view_as_complex_105 = torch.ops.aten.view_as_complex.default(view_1587);  view_1587 = None
        mul_677 = torch.ops.aten.mul.Tensor(view_as_complex_105, _conj);  view_as_complex_105 = None
        view_as_real_104 = torch.ops.aten.view_as_real.default(mul_676);  mul_676 = None
        view_1588 = torch.ops.aten.view.default(view_as_real_104, [2, 8192, 8, 128]);  view_as_real_104 = None
        convert_element_type_2185 = torch.ops.prims.convert_element_type.default(view_1588, torch.bfloat16);  view_1588 = None
        view_as_real_105 = torch.ops.aten.view_as_real.default(mul_677);  mul_677 = None
        view_1589 = torch.ops.aten.view.default(view_as_real_105, [2, 8192, 32, 128]);  view_as_real_105 = None
        convert_element_type_2186 = torch.ops.prims.convert_element_type.default(view_1589, torch.bfloat16);  view_1589 = None
        view_1590 = torch.ops.aten.view.default(squeeze_40, [2, 8192, 1024]);  squeeze_40 = None
        view_1591 = torch.ops.aten.view.default(convert_element_type_2185, [2, 8192, 1024]);  convert_element_type_2185 = None
        view_1592 = torch.ops.aten.view.default(convert_element_type_2186, [2, 8192, 4096]);  convert_element_type_2186 = None
        view_1593 = torch.ops.aten.view.default(view_1590, [16384, 1024]);  view_1590 = None
        permute_1017 = torch.ops.aten.permute.default(view_1593, [1, 0])
        mm_515 = torch.ops.aten.mm.default(permute_1017, view_377);  permute_1017 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        all_gather_into_tensor_103 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_373, 256, '0');  convert_element_type_373 = None
        wait_tensor_103 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_103);  all_gather_into_tensor_103 = None
        permute_123 = torch.ops.aten.permute.default(wait_tensor_103, [1, 0]);  wait_tensor_103 = None
        permute_1019 = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_516 = torch.ops.aten.mm.default(view_1593, permute_1019);  view_1593 = permute_1019 = None
        view_1594 = torch.ops.aten.view.default(mm_516, [2, 8192, 4096]);  mm_516 = None
        convert_element_type_2191 = torch.ops.prims.convert_element_type.default(mm_515, torch.float32);  mm_515 = None
        reduce_scatter_tensor_187 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2191, 'avg', 256, '0');  convert_element_type_2191 = None
        wait_tensor_478 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_187);  reduce_scatter_tensor_187 = None
        view_1595 = torch.ops.aten.view.default(view_1591, [16384, 1024]);  view_1591 = None
        permute_1021 = torch.ops.aten.permute.default(view_1595, [1, 0])
        mm_517 = torch.ops.aten.mm.default(permute_1021, view_377);  permute_1021 = None
        permute_1023 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_518 = torch.ops.aten.mm.default(view_1595, permute_1023);  view_1595 = permute_1023 = None
        view_1596 = torch.ops.aten.view.default(mm_518, [2, 8192, 4096]);  mm_518 = None
        add_273 = torch.ops.aten.add.Tensor(view_1594, view_1596);  view_1594 = view_1596 = None
        convert_element_type_2196 = torch.ops.prims.convert_element_type.default(mm_517, torch.float32);  mm_517 = None
        reduce_scatter_tensor_188 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2196, 'avg', 256, '0');  convert_element_type_2196 = None
        wait_tensor_479 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_188);  reduce_scatter_tensor_188 = None
        view_1597 = torch.ops.aten.view.default(view_1592, [16384, 4096]);  view_1592 = None
        permute_1025 = torch.ops.aten.permute.default(view_1597, [1, 0])
        mm_519 = torch.ops.aten.mm.default(permute_1025, view_377);  permute_1025 = view_377 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        all_gather_into_tensor_101 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_367, 256, '0');  convert_element_type_367 = None
        wait_tensor_101 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_101);  all_gather_into_tensor_101 = None
        permute_121 = torch.ops.aten.permute.default(wait_tensor_101, [1, 0]);  wait_tensor_101 = None
        permute_1027 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_520 = torch.ops.aten.mm.default(view_1597, permute_1027);  view_1597 = permute_1027 = None
        view_1598 = torch.ops.aten.view.default(mm_520, [2, 8192, 4096]);  mm_520 = None
        add_274 = torch.ops.aten.add.Tensor(add_273, view_1598);  add_273 = view_1598 = None
        convert_element_type_2201 = torch.ops.prims.convert_element_type.default(mm_519, torch.float32);  mm_519 = None
        reduce_scatter_tensor_189 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2201, 'avg', 256, '0');  convert_element_type_2201 = None
        wait_tensor_480 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_189);  reduce_scatter_tensor_189 = None
        convert_element_type_2202 = torch.ops.prims.convert_element_type.default(add_274, torch.float32);  add_274 = None
        convert_element_type_2204 = torch.ops.prims.convert_element_type.default(wait_tensor_100, torch.float32);  wait_tensor_100 = None
        mul_678 = torch.ops.aten.mul.Tensor(convert_element_type_2202, convert_element_type_2204);  convert_element_type_2204 = None
        mul_680 = torch.ops.aten.mul.Tensor(mul_88, mul_678)
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_680, [2], True);  mul_680 = None
        div_42 = torch.ops.aten.div.Tensor(mul_88, 4096)
        mul_681 = torch.ops.aten.mul.Tensor(div_42, sum_127);  div_42 = sum_127 = None
        sub_63 = torch.ops.aten.sub.Tensor(mul_678, mul_681);  mul_678 = mul_681 = None
        mul_682 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_22);  sub_63 = rsqrt_22 = None
        mul_683 = torch.ops.aten.mul.Tensor(convert_element_type_2202, mul_88);  convert_element_type_2202 = mul_88 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(mul_683, [0, 1]);  mul_683 = None
        convert_element_type_2205 = torch.ops.prims.convert_element_type.default(mul_682, torch.bfloat16);  mul_682 = None
        add_275 = torch.ops.aten.add.Tensor(add_272, convert_element_type_2205);  add_272 = convert_element_type_2205 = None
        convert_element_type_default_23 = torch.ops.prims.convert_element_type.default(sum_128, torch.float32);  sum_128 = None
        reduce_scatter_tensor_190 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_23, 'avg', 256, '0');  convert_element_type_default_23 = None
        wait_tensor_481 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_190);  reduce_scatter_tensor_190 = None
        view_1599 = torch.ops.aten.view.default(add_275, [16384, 4096])
        permute_1029 = torch.ops.aten.permute.default(view_1599, [1, 0])
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3])
        view_361 = torch.ops.aten.view.default(permute_116, [2, 8192, -1]);  permute_116 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        all_gather_into_tensor_95 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_347, 256, '0');  convert_element_type_347 = None
        wait_tensor_95 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_95);  all_gather_into_tensor_95 = None
        permute_117 = torch.ops.aten.permute.default(wait_tensor_95, [1, 0]);  wait_tensor_95 = None
        view_363 = torch.ops.aten.view.default(view_361, [16384, 4096]);  view_361 = None
        mm_73 = torch.ops.aten.mm.default(view_363, permute_117)
        view_364 = torch.ops.aten.view.default(mm_73, [2, 8192, 4096]);  mm_73 = None
        add_41 = torch.ops.aten.add.Tensor(add_39, view_364);  view_364 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        all_gather_into_tensor_96 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_350, 256, '0');  convert_element_type_350 = None
        wait_tensor_96 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_96);  all_gather_into_tensor_96 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_42 = torch.ops.aten.add.Scalar(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_21);  convert_element_type_351 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, wait_tensor_96)
        convert_element_type_352 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        view_367 = torch.ops.aten.view.default(convert_element_type_352, [16384, 4096]);  convert_element_type_352 = None
        view_368 = torch.ops.aten.view.default(mm_74, [2, 8192, 14336]);  mm_74 = None
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_356)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_356, sigmoid_10);  sigmoid_10 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        all_gather_into_tensor_98 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_358, 256, '0');  convert_element_type_358 = None
        wait_tensor_98 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_98);  all_gather_into_tensor_98 = None
        permute_119 = torch.ops.aten.permute.default(wait_tensor_98, [1, 0]);  wait_tensor_98 = None
        mm_75 = torch.ops.aten.mm.default(view_367, permute_119)
        view_371 = torch.ops.aten.view.default(mm_75, [2, 8192, 14336]);  mm_75 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_357, view_371)
        view_373 = torch.ops.aten.view.default(mul_87, [16384, 14336]);  mul_87 = None
        mm_521 = torch.ops.aten.mm.default(permute_1029, view_373);  permute_1029 = view_373 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        all_gather_into_tensor_99 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_361, 256, '0');  convert_element_type_361 = None
        wait_tensor_99 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_99);  all_gather_into_tensor_99 = None
        permute_120 = torch.ops.aten.permute.default(wait_tensor_99, [1, 0]);  wait_tensor_99 = None
        permute_1031 = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_522 = torch.ops.aten.mm.default(view_1599, permute_1031);  view_1599 = permute_1031 = None
        view_1600 = torch.ops.aten.view.default(mm_522, [2, 8192, 14336]);  mm_522 = None
        convert_element_type_2212 = torch.ops.prims.convert_element_type.default(mm_521, torch.float32);  mm_521 = None
        reduce_scatter_tensor_191 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2212, 'avg', 256, '0');  convert_element_type_2212 = None
        wait_tensor_482 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_191);  reduce_scatter_tensor_191 = None
        mul_684 = torch.ops.aten.mul.Tensor(view_1600, convert_element_type_357);  convert_element_type_357 = None
        mul_685 = torch.ops.aten.mul.Tensor(view_1600, view_371);  view_1600 = view_371 = None
        view_1601 = torch.ops.aten.view.default(mul_684, [16384, 14336]);  mul_684 = None
        permute_1033 = torch.ops.aten.permute.default(view_1601, [1, 0])
        mm_523 = torch.ops.aten.mm.default(permute_1033, view_367);  permute_1033 = None
        permute_1035 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_524 = torch.ops.aten.mm.default(view_1601, permute_1035);  view_1601 = permute_1035 = None
        view_1602 = torch.ops.aten.view.default(mm_524, [2, 8192, 4096]);  mm_524 = None
        convert_element_type_2217 = torch.ops.prims.convert_element_type.default(mm_523, torch.float32);  mm_523 = None
        reduce_scatter_tensor_192 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2217, 'avg', 256, '0');  convert_element_type_2217 = None
        wait_tensor_483 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_192);  reduce_scatter_tensor_192 = None
        convert_element_type_2218 = torch.ops.prims.convert_element_type.default(mul_685, torch.float32);  mul_685 = None
        neg_21 = torch.ops.aten.neg.default(convert_element_type_356)
        exp_21 = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_276 = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(add_276);  add_276 = None
        mul_686 = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        mul_687 = torch.ops.aten.mul.Tensor(convert_element_type_2218, mul_686);  convert_element_type_2218 = None
        sub_64 = torch.ops.aten.sub.Tensor(1, mul_686);  mul_686 = None
        mul_688 = torch.ops.aten.mul.Tensor(convert_element_type_356, sub_64);  convert_element_type_356 = sub_64 = None
        add_277 = torch.ops.aten.add.Tensor(mul_688, 1);  mul_688 = None
        mul_689 = torch.ops.aten.mul.Tensor(mul_687, add_277);  mul_687 = add_277 = None
        convert_element_type_2220 = torch.ops.prims.convert_element_type.default(mul_689, torch.bfloat16);  mul_689 = None
        view_1603 = torch.ops.aten.view.default(convert_element_type_2220, [16384, 14336]);  convert_element_type_2220 = None
        permute_1037 = torch.ops.aten.permute.default(view_1603, [1, 0])
        mm_525 = torch.ops.aten.mm.default(permute_1037, view_367);  permute_1037 = view_367 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        all_gather_into_tensor_97 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_353, 256, '0');  convert_element_type_353 = None
        wait_tensor_97 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_97);  all_gather_into_tensor_97 = None
        permute_118 = torch.ops.aten.permute.default(wait_tensor_97, [1, 0]);  wait_tensor_97 = None
        permute_1039 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_526 = torch.ops.aten.mm.default(view_1603, permute_1039);  view_1603 = permute_1039 = None
        view_1604 = torch.ops.aten.view.default(mm_526, [2, 8192, 4096]);  mm_526 = None
        add_278 = torch.ops.aten.add.Tensor(view_1602, view_1604);  view_1602 = view_1604 = None
        convert_element_type_2225 = torch.ops.prims.convert_element_type.default(mm_525, torch.float32);  mm_525 = None
        reduce_scatter_tensor_193 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2225, 'avg', 256, '0');  convert_element_type_2225 = None
        wait_tensor_484 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_193);  reduce_scatter_tensor_193 = None
        convert_element_type_2226 = torch.ops.prims.convert_element_type.default(add_278, torch.float32);  add_278 = None
        convert_element_type_2228 = torch.ops.prims.convert_element_type.default(wait_tensor_96, torch.float32);  wait_tensor_96 = None
        mul_690 = torch.ops.aten.mul.Tensor(convert_element_type_2226, convert_element_type_2228);  convert_element_type_2228 = None
        mul_692 = torch.ops.aten.mul.Tensor(mul_84, mul_690)
        sum_129 = torch.ops.aten.sum.dim_IntList(mul_692, [2], True);  mul_692 = None
        div_43 = torch.ops.aten.div.Tensor(mul_84, 4096)
        mul_693 = torch.ops.aten.mul.Tensor(div_43, sum_129);  div_43 = sum_129 = None
        sub_65 = torch.ops.aten.sub.Tensor(mul_690, mul_693);  mul_690 = mul_693 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_21);  sub_65 = rsqrt_21 = None
        mul_695 = torch.ops.aten.mul.Tensor(convert_element_type_2226, mul_84);  convert_element_type_2226 = mul_84 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(mul_695, [0, 1]);  mul_695 = None
        convert_element_type_2229 = torch.ops.prims.convert_element_type.default(mul_694, torch.bfloat16);  mul_694 = None
        add_279 = torch.ops.aten.add.Tensor(add_275, convert_element_type_2229);  add_275 = convert_element_type_2229 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(sum_130, torch.float32);  sum_130 = None
        reduce_scatter_tensor_194 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_22, 'avg', 256, '0');  convert_element_type_default_22 = None
        wait_tensor_485 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_194);  reduce_scatter_tensor_194 = None
        view_1605 = torch.ops.aten.view.default(add_279, [16384, 4096])
        permute_1041 = torch.ops.aten.permute.default(view_1605, [1, 0])
        mm_527 = torch.ops.aten.mm.default(permute_1041, view_363);  permute_1041 = view_363 = None
        permute_1043 = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        mm_528 = torch.ops.aten.mm.default(view_1605, permute_1043);  view_1605 = permute_1043 = None
        view_1606 = torch.ops.aten.view.default(mm_528, [2, 8192, 4096]);  mm_528 = None
        convert_element_type_2236 = torch.ops.prims.convert_element_type.default(mm_527, torch.float32);  mm_527 = None
        reduce_scatter_tensor_195 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2236, 'avg', 256, '0');  convert_element_type_2236 = None
        wait_tensor_486 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_195);  reduce_scatter_tensor_195 = None
        view_1607 = torch.ops.aten.view.default(view_1606, [2, 8192, 32, 128]);  view_1606 = None
        permute_1045 = torch.ops.aten.permute.default(view_1607, [0, 2, 1, 3]);  view_1607 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        all_gather_into_tensor_91 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_331, 256, '0');  convert_element_type_331 = None
        wait_tensor_91 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_91);  all_gather_into_tensor_91 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [2], True);  pow_21 = None
        add_40 = torch.ops.aten.add.Scalar(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_20);  convert_element_type_332 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, wait_tensor_91)
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        view_343 = torch.ops.aten.view.default(convert_element_type_333, [16384, 4096]);  convert_element_type_333 = None
        view_344 = torch.ops.aten.view.default(mm_70, [2, 8192, 4096]);  mm_70 = None
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        all_gather_into_tensor_93 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_337, 256, '0');  convert_element_type_337 = None
        wait_tensor_93 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_93);  all_gather_into_tensor_93 = None
        permute_111 = torch.ops.aten.permute.default(wait_tensor_93, [1, 0]);  wait_tensor_93 = None
        mm_71 = torch.ops.aten.mm.default(view_343, permute_111)
        view_347 = torch.ops.aten.view.default(mm_71, [2, 8192, 1024]);  mm_71 = None
        view_350 = torch.ops.aten.view.default(mm_72, [2, 8192, 1024]);  mm_72 = None
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
        _scaled_dot_product_cudnn_attention_backward_21 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1045, permute_113, permute_114, permute_115, getitem_90, getitem_91, getitem_96, getitem_97, None, None, None, 8192, 8192, 0.0, True);  permute_1045 = permute_113 = permute_114 = permute_115 = getitem_90 = getitem_91 = getitem_96 = getitem_97 = None
        getitem_351 = _scaled_dot_product_cudnn_attention_backward_21[0]
        getitem_352 = _scaled_dot_product_cudnn_attention_backward_21[1]
        getitem_353 = _scaled_dot_product_cudnn_attention_backward_21[2];  _scaled_dot_product_cudnn_attention_backward_21 = None
        permute_1046 = torch.ops.aten.permute.default(getitem_353, [0, 2, 1, 3]);  getitem_353 = None
        permute_1047 = torch.ops.aten.permute.default(getitem_352, [0, 2, 1, 3]);  getitem_352 = None
        permute_1048 = torch.ops.aten.permute.default(getitem_351, [0, 2, 1, 3]);  getitem_351 = None
        view_1608 = torch.ops.aten.view.default(permute_1046, [2, 8192, 8, 4, 128]);  permute_1046 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(view_1608, [3], True);  view_1608 = None
        squeeze_42 = torch.ops.aten.squeeze.dim(sum_131, 3);  sum_131 = None
        view_1609 = torch.ops.aten.view.default(permute_1047, [2, 8192, 8, 4, 128]);  permute_1047 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(view_1609, [3], True);  view_1609 = None
        squeeze_43 = torch.ops.aten.squeeze.dim(sum_132, 3);  sum_132 = None
        convert_element_type_2237 = torch.ops.prims.convert_element_type.default(squeeze_43, torch.float32);  squeeze_43 = None
        convert_element_type_2238 = torch.ops.prims.convert_element_type.default(permute_1048, torch.float32);  permute_1048 = None
        view_1610 = torch.ops.aten.view.default(convert_element_type_2237, [2, 8192, 8, 64, 2]);  convert_element_type_2237 = None
        view_as_complex_106 = torch.ops.aten.view_as_complex.default(view_1610);  view_1610 = None
        mul_696 = torch.ops.aten.mul.Tensor(view_as_complex_106, _conj);  view_as_complex_106 = None
        view_1611 = torch.ops.aten.view.default(convert_element_type_2238, [2, 8192, 32, 64, 2]);  convert_element_type_2238 = None
        view_as_complex_107 = torch.ops.aten.view_as_complex.default(view_1611);  view_1611 = None
        mul_697 = torch.ops.aten.mul.Tensor(view_as_complex_107, _conj);  view_as_complex_107 = None
        view_as_real_106 = torch.ops.aten.view_as_real.default(mul_696);  mul_696 = None
        view_1612 = torch.ops.aten.view.default(view_as_real_106, [2, 8192, 8, 128]);  view_as_real_106 = None
        convert_element_type_2239 = torch.ops.prims.convert_element_type.default(view_1612, torch.bfloat16);  view_1612 = None
        view_as_real_107 = torch.ops.aten.view_as_real.default(mul_697);  mul_697 = None
        view_1613 = torch.ops.aten.view.default(view_as_real_107, [2, 8192, 32, 128]);  view_as_real_107 = None
        convert_element_type_2240 = torch.ops.prims.convert_element_type.default(view_1613, torch.bfloat16);  view_1613 = None
        view_1614 = torch.ops.aten.view.default(squeeze_42, [2, 8192, 1024]);  squeeze_42 = None
        view_1615 = torch.ops.aten.view.default(convert_element_type_2239, [2, 8192, 1024]);  convert_element_type_2239 = None
        view_1616 = torch.ops.aten.view.default(convert_element_type_2240, [2, 8192, 4096]);  convert_element_type_2240 = None
        view_1617 = torch.ops.aten.view.default(view_1614, [16384, 1024]);  view_1614 = None
        permute_1049 = torch.ops.aten.permute.default(view_1617, [1, 0])
        mm_529 = torch.ops.aten.mm.default(permute_1049, view_343);  permute_1049 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        all_gather_into_tensor_94 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_340, 256, '0');  convert_element_type_340 = None
        wait_tensor_94 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_94);  all_gather_into_tensor_94 = None
        permute_112 = torch.ops.aten.permute.default(wait_tensor_94, [1, 0]);  wait_tensor_94 = None
        permute_1051 = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_530 = torch.ops.aten.mm.default(view_1617, permute_1051);  view_1617 = permute_1051 = None
        view_1618 = torch.ops.aten.view.default(mm_530, [2, 8192, 4096]);  mm_530 = None
        convert_element_type_2245 = torch.ops.prims.convert_element_type.default(mm_529, torch.float32);  mm_529 = None
        reduce_scatter_tensor_196 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2245, 'avg', 256, '0');  convert_element_type_2245 = None
        wait_tensor_487 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_196);  reduce_scatter_tensor_196 = None
        view_1619 = torch.ops.aten.view.default(view_1615, [16384, 1024]);  view_1615 = None
        permute_1053 = torch.ops.aten.permute.default(view_1619, [1, 0])
        mm_531 = torch.ops.aten.mm.default(permute_1053, view_343);  permute_1053 = None
        permute_1055 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_532 = torch.ops.aten.mm.default(view_1619, permute_1055);  view_1619 = permute_1055 = None
        view_1620 = torch.ops.aten.view.default(mm_532, [2, 8192, 4096]);  mm_532 = None
        add_280 = torch.ops.aten.add.Tensor(view_1618, view_1620);  view_1618 = view_1620 = None
        convert_element_type_2250 = torch.ops.prims.convert_element_type.default(mm_531, torch.float32);  mm_531 = None
        reduce_scatter_tensor_197 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2250, 'avg', 256, '0');  convert_element_type_2250 = None
        wait_tensor_488 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_197);  reduce_scatter_tensor_197 = None
        view_1621 = torch.ops.aten.view.default(view_1616, [16384, 4096]);  view_1616 = None
        permute_1057 = torch.ops.aten.permute.default(view_1621, [1, 0])
        mm_533 = torch.ops.aten.mm.default(permute_1057, view_343);  permute_1057 = view_343 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        all_gather_into_tensor_92 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_334, 256, '0');  convert_element_type_334 = None
        wait_tensor_92 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_92);  all_gather_into_tensor_92 = None
        permute_110 = torch.ops.aten.permute.default(wait_tensor_92, [1, 0]);  wait_tensor_92 = None
        permute_1059 = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_534 = torch.ops.aten.mm.default(view_1621, permute_1059);  view_1621 = permute_1059 = None
        view_1622 = torch.ops.aten.view.default(mm_534, [2, 8192, 4096]);  mm_534 = None
        add_281 = torch.ops.aten.add.Tensor(add_280, view_1622);  add_280 = view_1622 = None
        convert_element_type_2255 = torch.ops.prims.convert_element_type.default(mm_533, torch.float32);  mm_533 = None
        reduce_scatter_tensor_198 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2255, 'avg', 256, '0');  convert_element_type_2255 = None
        wait_tensor_489 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_198);  reduce_scatter_tensor_198 = None
        convert_element_type_2256 = torch.ops.prims.convert_element_type.default(add_281, torch.float32);  add_281 = None
        convert_element_type_2258 = torch.ops.prims.convert_element_type.default(wait_tensor_91, torch.float32);  wait_tensor_91 = None
        mul_698 = torch.ops.aten.mul.Tensor(convert_element_type_2256, convert_element_type_2258);  convert_element_type_2258 = None
        mul_700 = torch.ops.aten.mul.Tensor(mul_80, mul_698)
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_700, [2], True);  mul_700 = None
        div_44 = torch.ops.aten.div.Tensor(mul_80, 4096)
        mul_701 = torch.ops.aten.mul.Tensor(div_44, sum_133);  div_44 = sum_133 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_698, mul_701);  mul_698 = mul_701 = None
        mul_702 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_20);  sub_66 = rsqrt_20 = None
        mul_703 = torch.ops.aten.mul.Tensor(convert_element_type_2256, mul_80);  convert_element_type_2256 = mul_80 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1]);  mul_703 = None
        convert_element_type_2259 = torch.ops.prims.convert_element_type.default(mul_702, torch.bfloat16);  mul_702 = None
        add_282 = torch.ops.aten.add.Tensor(add_279, convert_element_type_2259);  add_279 = convert_element_type_2259 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(sum_134, torch.float32);  sum_134 = None
        reduce_scatter_tensor_199 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_21, 'avg', 256, '0');  convert_element_type_default_21 = None
        wait_tensor_490 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_199);  reduce_scatter_tensor_199 = None
        view_1623 = torch.ops.aten.view.default(add_282, [16384, 4096])
        permute_1061 = torch.ops.aten.permute.default(view_1623, [1, 0])
        permute_105 = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3])
        view_327 = torch.ops.aten.view.default(permute_105, [2, 8192, -1]);  permute_105 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        all_gather_into_tensor_86 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_314, 256, '0');  convert_element_type_314 = None
        wait_tensor_86 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_86);  all_gather_into_tensor_86 = None
        permute_106 = torch.ops.aten.permute.default(wait_tensor_86, [1, 0]);  wait_tensor_86 = None
        view_329 = torch.ops.aten.view.default(view_327, [16384, 4096]);  view_327 = None
        mm_66 = torch.ops.aten.mm.default(view_329, permute_106)
        view_330 = torch.ops.aten.view.default(mm_66, [2, 8192, 4096]);  mm_66 = None
        add_37 = torch.ops.aten.add.Tensor(add_35, view_330);  view_330 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        all_gather_into_tensor_87 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_317, 256, '0');  convert_element_type_317 = None
        wait_tensor_87 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_87);  all_gather_into_tensor_87 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_38 = torch.ops.aten.add.Scalar(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_19);  convert_element_type_318 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, wait_tensor_87)
        convert_element_type_319 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        view_333 = torch.ops.aten.view.default(convert_element_type_319, [16384, 4096]);  convert_element_type_319 = None
        view_334 = torch.ops.aten.view.default(mm_67, [2, 8192, 14336]);  mm_67 = None
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(view_334, torch.float32);  view_334 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_323)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_323, sigmoid_9);  sigmoid_9 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        all_gather_into_tensor_89 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_325, 256, '0');  convert_element_type_325 = None
        wait_tensor_89 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_89);  all_gather_into_tensor_89 = None
        permute_108 = torch.ops.aten.permute.default(wait_tensor_89, [1, 0]);  wait_tensor_89 = None
        mm_68 = torch.ops.aten.mm.default(view_333, permute_108)
        view_337 = torch.ops.aten.view.default(mm_68, [2, 8192, 14336]);  mm_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_324, view_337)
        view_339 = torch.ops.aten.view.default(mul_79, [16384, 14336]);  mul_79 = None
        mm_535 = torch.ops.aten.mm.default(permute_1061, view_339);  permute_1061 = view_339 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        all_gather_into_tensor_90 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_328, 256, '0');  convert_element_type_328 = None
        wait_tensor_90 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_90);  all_gather_into_tensor_90 = None
        permute_109 = torch.ops.aten.permute.default(wait_tensor_90, [1, 0]);  wait_tensor_90 = None
        permute_1063 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_536 = torch.ops.aten.mm.default(view_1623, permute_1063);  view_1623 = permute_1063 = None
        view_1624 = torch.ops.aten.view.default(mm_536, [2, 8192, 14336]);  mm_536 = None
        convert_element_type_2266 = torch.ops.prims.convert_element_type.default(mm_535, torch.float32);  mm_535 = None
        reduce_scatter_tensor_200 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2266, 'avg', 256, '0');  convert_element_type_2266 = None
        wait_tensor_491 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_200);  reduce_scatter_tensor_200 = None
        mul_704 = torch.ops.aten.mul.Tensor(view_1624, convert_element_type_324);  convert_element_type_324 = None
        mul_705 = torch.ops.aten.mul.Tensor(view_1624, view_337);  view_1624 = view_337 = None
        view_1625 = torch.ops.aten.view.default(mul_704, [16384, 14336]);  mul_704 = None
        permute_1065 = torch.ops.aten.permute.default(view_1625, [1, 0])
        mm_537 = torch.ops.aten.mm.default(permute_1065, view_333);  permute_1065 = None
        permute_1067 = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_538 = torch.ops.aten.mm.default(view_1625, permute_1067);  view_1625 = permute_1067 = None
        view_1626 = torch.ops.aten.view.default(mm_538, [2, 8192, 4096]);  mm_538 = None
        convert_element_type_2271 = torch.ops.prims.convert_element_type.default(mm_537, torch.float32);  mm_537 = None
        reduce_scatter_tensor_201 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2271, 'avg', 256, '0');  convert_element_type_2271 = None
        wait_tensor_492 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_201);  reduce_scatter_tensor_201 = None
        convert_element_type_2272 = torch.ops.prims.convert_element_type.default(mul_705, torch.float32);  mul_705 = None
        neg_22 = torch.ops.aten.neg.default(convert_element_type_323)
        exp_22 = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_283 = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(add_283);  add_283 = None
        mul_706 = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        mul_707 = torch.ops.aten.mul.Tensor(convert_element_type_2272, mul_706);  convert_element_type_2272 = None
        sub_67 = torch.ops.aten.sub.Tensor(1, mul_706);  mul_706 = None
        mul_708 = torch.ops.aten.mul.Tensor(convert_element_type_323, sub_67);  convert_element_type_323 = sub_67 = None
        add_284 = torch.ops.aten.add.Tensor(mul_708, 1);  mul_708 = None
        mul_709 = torch.ops.aten.mul.Tensor(mul_707, add_284);  mul_707 = add_284 = None
        convert_element_type_2274 = torch.ops.prims.convert_element_type.default(mul_709, torch.bfloat16);  mul_709 = None
        view_1627 = torch.ops.aten.view.default(convert_element_type_2274, [16384, 14336]);  convert_element_type_2274 = None
        permute_1069 = torch.ops.aten.permute.default(view_1627, [1, 0])
        mm_539 = torch.ops.aten.mm.default(permute_1069, view_333);  permute_1069 = view_333 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        all_gather_into_tensor_88 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_320, 256, '0');  convert_element_type_320 = None
        wait_tensor_88 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_88);  all_gather_into_tensor_88 = None
        permute_107 = torch.ops.aten.permute.default(wait_tensor_88, [1, 0]);  wait_tensor_88 = None
        permute_1071 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_540 = torch.ops.aten.mm.default(view_1627, permute_1071);  view_1627 = permute_1071 = None
        view_1628 = torch.ops.aten.view.default(mm_540, [2, 8192, 4096]);  mm_540 = None
        add_285 = torch.ops.aten.add.Tensor(view_1626, view_1628);  view_1626 = view_1628 = None
        convert_element_type_2279 = torch.ops.prims.convert_element_type.default(mm_539, torch.float32);  mm_539 = None
        reduce_scatter_tensor_202 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2279, 'avg', 256, '0');  convert_element_type_2279 = None
        wait_tensor_493 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_202);  reduce_scatter_tensor_202 = None
        convert_element_type_2280 = torch.ops.prims.convert_element_type.default(add_285, torch.float32);  add_285 = None
        convert_element_type_2282 = torch.ops.prims.convert_element_type.default(wait_tensor_87, torch.float32);  wait_tensor_87 = None
        mul_710 = torch.ops.aten.mul.Tensor(convert_element_type_2280, convert_element_type_2282);  convert_element_type_2282 = None
        mul_712 = torch.ops.aten.mul.Tensor(mul_76, mul_710)
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_712, [2], True);  mul_712 = None
        div_45 = torch.ops.aten.div.Tensor(mul_76, 4096)
        mul_713 = torch.ops.aten.mul.Tensor(div_45, sum_135);  div_45 = sum_135 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_710, mul_713);  mul_710 = mul_713 = None
        mul_714 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_19);  sub_68 = rsqrt_19 = None
        mul_715 = torch.ops.aten.mul.Tensor(convert_element_type_2280, mul_76);  convert_element_type_2280 = mul_76 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(mul_715, [0, 1]);  mul_715 = None
        convert_element_type_2283 = torch.ops.prims.convert_element_type.default(mul_714, torch.bfloat16);  mul_714 = None
        add_286 = torch.ops.aten.add.Tensor(add_282, convert_element_type_2283);  add_282 = convert_element_type_2283 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(sum_136, torch.float32);  sum_136 = None
        reduce_scatter_tensor_203 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_20, 'avg', 256, '0');  convert_element_type_default_20 = None
        wait_tensor_494 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_203);  reduce_scatter_tensor_203 = None
        view_1629 = torch.ops.aten.view.default(add_286, [16384, 4096])
        permute_1073 = torch.ops.aten.permute.default(view_1629, [1, 0])
        mm_541 = torch.ops.aten.mm.default(permute_1073, view_329);  permute_1073 = view_329 = None
        permute_1075 = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        mm_542 = torch.ops.aten.mm.default(view_1629, permute_1075);  view_1629 = permute_1075 = None
        view_1630 = torch.ops.aten.view.default(mm_542, [2, 8192, 4096]);  mm_542 = None
        convert_element_type_2290 = torch.ops.prims.convert_element_type.default(mm_541, torch.float32);  mm_541 = None
        reduce_scatter_tensor_204 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2290, 'avg', 256, '0');  convert_element_type_2290 = None
        wait_tensor_495 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_204);  reduce_scatter_tensor_204 = None
        view_1631 = torch.ops.aten.view.default(view_1630, [2, 8192, 32, 128]);  view_1630 = None
        permute_1077 = torch.ops.aten.permute.default(view_1631, [0, 2, 1, 3]);  view_1631 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        all_gather_into_tensor_82 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_298, 256, '0');  convert_element_type_298 = None
        wait_tensor_82 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_82);  all_gather_into_tensor_82 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [2], True);  pow_19 = None
        add_36 = torch.ops.aten.add.Scalar(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_18);  convert_element_type_299 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, wait_tensor_82)
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        view_309 = torch.ops.aten.view.default(convert_element_type_300, [16384, 4096]);  convert_element_type_300 = None
        view_310 = torch.ops.aten.view.default(mm_63, [2, 8192, 4096]);  mm_63 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        all_gather_into_tensor_84 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_304, 256, '0');  convert_element_type_304 = None
        wait_tensor_84 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_84);  all_gather_into_tensor_84 = None
        permute_100 = torch.ops.aten.permute.default(wait_tensor_84, [1, 0]);  wait_tensor_84 = None
        mm_64 = torch.ops.aten.mm.default(view_309, permute_100)
        view_313 = torch.ops.aten.view.default(mm_64, [2, 8192, 1024]);  mm_64 = None
        view_316 = torch.ops.aten.view.default(mm_65, [2, 8192, 1024]);  mm_65 = None
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
        _scaled_dot_product_cudnn_attention_backward_22 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1077, permute_102, permute_103, permute_104, getitem_81, getitem_82, getitem_87, getitem_88, None, None, None, 8192, 8192, 0.0, True);  permute_1077 = permute_102 = permute_103 = permute_104 = getitem_81 = getitem_82 = getitem_87 = getitem_88 = None
        getitem_354 = _scaled_dot_product_cudnn_attention_backward_22[0]
        getitem_355 = _scaled_dot_product_cudnn_attention_backward_22[1]
        getitem_356 = _scaled_dot_product_cudnn_attention_backward_22[2];  _scaled_dot_product_cudnn_attention_backward_22 = None
        permute_1078 = torch.ops.aten.permute.default(getitem_356, [0, 2, 1, 3]);  getitem_356 = None
        permute_1079 = torch.ops.aten.permute.default(getitem_355, [0, 2, 1, 3]);  getitem_355 = None
        permute_1080 = torch.ops.aten.permute.default(getitem_354, [0, 2, 1, 3]);  getitem_354 = None
        view_1632 = torch.ops.aten.view.default(permute_1078, [2, 8192, 8, 4, 128]);  permute_1078 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(view_1632, [3], True);  view_1632 = None
        squeeze_44 = torch.ops.aten.squeeze.dim(sum_137, 3);  sum_137 = None
        view_1633 = torch.ops.aten.view.default(permute_1079, [2, 8192, 8, 4, 128]);  permute_1079 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(view_1633, [3], True);  view_1633 = None
        squeeze_45 = torch.ops.aten.squeeze.dim(sum_138, 3);  sum_138 = None
        convert_element_type_2291 = torch.ops.prims.convert_element_type.default(squeeze_45, torch.float32);  squeeze_45 = None
        convert_element_type_2292 = torch.ops.prims.convert_element_type.default(permute_1080, torch.float32);  permute_1080 = None
        view_1634 = torch.ops.aten.view.default(convert_element_type_2291, [2, 8192, 8, 64, 2]);  convert_element_type_2291 = None
        view_as_complex_108 = torch.ops.aten.view_as_complex.default(view_1634);  view_1634 = None
        mul_716 = torch.ops.aten.mul.Tensor(view_as_complex_108, _conj);  view_as_complex_108 = None
        view_1635 = torch.ops.aten.view.default(convert_element_type_2292, [2, 8192, 32, 64, 2]);  convert_element_type_2292 = None
        view_as_complex_109 = torch.ops.aten.view_as_complex.default(view_1635);  view_1635 = None
        mul_717 = torch.ops.aten.mul.Tensor(view_as_complex_109, _conj);  view_as_complex_109 = None
        view_as_real_108 = torch.ops.aten.view_as_real.default(mul_716);  mul_716 = None
        view_1636 = torch.ops.aten.view.default(view_as_real_108, [2, 8192, 8, 128]);  view_as_real_108 = None
        convert_element_type_2293 = torch.ops.prims.convert_element_type.default(view_1636, torch.bfloat16);  view_1636 = None
        view_as_real_109 = torch.ops.aten.view_as_real.default(mul_717);  mul_717 = None
        view_1637 = torch.ops.aten.view.default(view_as_real_109, [2, 8192, 32, 128]);  view_as_real_109 = None
        convert_element_type_2294 = torch.ops.prims.convert_element_type.default(view_1637, torch.bfloat16);  view_1637 = None
        view_1638 = torch.ops.aten.view.default(squeeze_44, [2, 8192, 1024]);  squeeze_44 = None
        view_1639 = torch.ops.aten.view.default(convert_element_type_2293, [2, 8192, 1024]);  convert_element_type_2293 = None
        view_1640 = torch.ops.aten.view.default(convert_element_type_2294, [2, 8192, 4096]);  convert_element_type_2294 = None
        view_1641 = torch.ops.aten.view.default(view_1638, [16384, 1024]);  view_1638 = None
        permute_1081 = torch.ops.aten.permute.default(view_1641, [1, 0])
        mm_543 = torch.ops.aten.mm.default(permute_1081, view_309);  permute_1081 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        all_gather_into_tensor_85 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_307, 256, '0');  convert_element_type_307 = None
        wait_tensor_85 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_85);  all_gather_into_tensor_85 = None
        permute_101 = torch.ops.aten.permute.default(wait_tensor_85, [1, 0]);  wait_tensor_85 = None
        permute_1083 = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_544 = torch.ops.aten.mm.default(view_1641, permute_1083);  view_1641 = permute_1083 = None
        view_1642 = torch.ops.aten.view.default(mm_544, [2, 8192, 4096]);  mm_544 = None
        convert_element_type_2299 = torch.ops.prims.convert_element_type.default(mm_543, torch.float32);  mm_543 = None
        reduce_scatter_tensor_205 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2299, 'avg', 256, '0');  convert_element_type_2299 = None
        wait_tensor_496 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_205);  reduce_scatter_tensor_205 = None
        view_1643 = torch.ops.aten.view.default(view_1639, [16384, 1024]);  view_1639 = None
        permute_1085 = torch.ops.aten.permute.default(view_1643, [1, 0])
        mm_545 = torch.ops.aten.mm.default(permute_1085, view_309);  permute_1085 = None
        permute_1087 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_546 = torch.ops.aten.mm.default(view_1643, permute_1087);  view_1643 = permute_1087 = None
        view_1644 = torch.ops.aten.view.default(mm_546, [2, 8192, 4096]);  mm_546 = None
        add_287 = torch.ops.aten.add.Tensor(view_1642, view_1644);  view_1642 = view_1644 = None
        convert_element_type_2304 = torch.ops.prims.convert_element_type.default(mm_545, torch.float32);  mm_545 = None
        reduce_scatter_tensor_206 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2304, 'avg', 256, '0');  convert_element_type_2304 = None
        wait_tensor_497 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_206);  reduce_scatter_tensor_206 = None
        view_1645 = torch.ops.aten.view.default(view_1640, [16384, 4096]);  view_1640 = None
        permute_1089 = torch.ops.aten.permute.default(view_1645, [1, 0])
        mm_547 = torch.ops.aten.mm.default(permute_1089, view_309);  permute_1089 = view_309 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        all_gather_into_tensor_83 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_301, 256, '0');  convert_element_type_301 = None
        wait_tensor_83 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_83);  all_gather_into_tensor_83 = None
        permute_99 = torch.ops.aten.permute.default(wait_tensor_83, [1, 0]);  wait_tensor_83 = None
        permute_1091 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_548 = torch.ops.aten.mm.default(view_1645, permute_1091);  view_1645 = permute_1091 = None
        view_1646 = torch.ops.aten.view.default(mm_548, [2, 8192, 4096]);  mm_548 = None
        add_288 = torch.ops.aten.add.Tensor(add_287, view_1646);  add_287 = view_1646 = None
        convert_element_type_2309 = torch.ops.prims.convert_element_type.default(mm_547, torch.float32);  mm_547 = None
        reduce_scatter_tensor_207 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2309, 'avg', 256, '0');  convert_element_type_2309 = None
        wait_tensor_498 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_207);  reduce_scatter_tensor_207 = None
        convert_element_type_2310 = torch.ops.prims.convert_element_type.default(add_288, torch.float32);  add_288 = None
        convert_element_type_2312 = torch.ops.prims.convert_element_type.default(wait_tensor_82, torch.float32);  wait_tensor_82 = None
        mul_718 = torch.ops.aten.mul.Tensor(convert_element_type_2310, convert_element_type_2312);  convert_element_type_2312 = None
        mul_720 = torch.ops.aten.mul.Tensor(mul_72, mul_718)
        sum_139 = torch.ops.aten.sum.dim_IntList(mul_720, [2], True);  mul_720 = None
        div_46 = torch.ops.aten.div.Tensor(mul_72, 4096)
        mul_721 = torch.ops.aten.mul.Tensor(div_46, sum_139);  div_46 = sum_139 = None
        sub_69 = torch.ops.aten.sub.Tensor(mul_718, mul_721);  mul_718 = mul_721 = None
        mul_722 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_18);  sub_69 = rsqrt_18 = None
        mul_723 = torch.ops.aten.mul.Tensor(convert_element_type_2310, mul_72);  convert_element_type_2310 = mul_72 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_723, [0, 1]);  mul_723 = None
        convert_element_type_2313 = torch.ops.prims.convert_element_type.default(mul_722, torch.bfloat16);  mul_722 = None
        add_289 = torch.ops.aten.add.Tensor(add_286, convert_element_type_2313);  add_286 = convert_element_type_2313 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(sum_140, torch.float32);  sum_140 = None
        reduce_scatter_tensor_208 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_19, 'avg', 256, '0');  convert_element_type_default_19 = None
        wait_tensor_499 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_208);  reduce_scatter_tensor_208 = None
        view_1647 = torch.ops.aten.view.default(add_289, [16384, 4096])
        permute_1093 = torch.ops.aten.permute.default(view_1647, [1, 0])
        permute_94 = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3])
        view_293 = torch.ops.aten.view.default(permute_94, [2, 8192, -1]);  permute_94 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        all_gather_into_tensor_77 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_281, 256, '0');  convert_element_type_281 = None
        wait_tensor_77 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_77);  all_gather_into_tensor_77 = None
        permute_95 = torch.ops.aten.permute.default(wait_tensor_77, [1, 0]);  wait_tensor_77 = None
        view_295 = torch.ops.aten.view.default(view_293, [16384, 4096]);  view_293 = None
        mm_59 = torch.ops.aten.mm.default(view_295, permute_95)
        view_296 = torch.ops.aten.view.default(mm_59, [2, 8192, 4096]);  mm_59 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, view_296);  view_296 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        all_gather_into_tensor_78 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_284, 256, '0');  convert_element_type_284 = None
        wait_tensor_78 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_78);  all_gather_into_tensor_78 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [2], True);  pow_18 = None
        add_34 = torch.ops.aten.add.Scalar(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_285, rsqrt_17);  convert_element_type_285 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, wait_tensor_78)
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_299 = torch.ops.aten.view.default(convert_element_type_286, [16384, 4096]);  convert_element_type_286 = None
        view_300 = torch.ops.aten.view.default(mm_60, [2, 8192, 14336]);  mm_60 = None
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_290)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_290, sigmoid_8);  sigmoid_8 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        all_gather_into_tensor_80 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_292, 256, '0');  convert_element_type_292 = None
        wait_tensor_80 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_80);  all_gather_into_tensor_80 = None
        permute_97 = torch.ops.aten.permute.default(wait_tensor_80, [1, 0]);  wait_tensor_80 = None
        mm_61 = torch.ops.aten.mm.default(view_299, permute_97)
        view_303 = torch.ops.aten.view.default(mm_61, [2, 8192, 14336]);  mm_61 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_291, view_303)
        view_305 = torch.ops.aten.view.default(mul_71, [16384, 14336]);  mul_71 = None
        mm_549 = torch.ops.aten.mm.default(permute_1093, view_305);  permute_1093 = view_305 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        all_gather_into_tensor_81 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_295, 256, '0');  convert_element_type_295 = None
        wait_tensor_81 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_81);  all_gather_into_tensor_81 = None
        permute_98 = torch.ops.aten.permute.default(wait_tensor_81, [1, 0]);  wait_tensor_81 = None
        permute_1095 = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_550 = torch.ops.aten.mm.default(view_1647, permute_1095);  view_1647 = permute_1095 = None
        view_1648 = torch.ops.aten.view.default(mm_550, [2, 8192, 14336]);  mm_550 = None
        convert_element_type_2320 = torch.ops.prims.convert_element_type.default(mm_549, torch.float32);  mm_549 = None
        reduce_scatter_tensor_209 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2320, 'avg', 256, '0');  convert_element_type_2320 = None
        wait_tensor_500 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_209);  reduce_scatter_tensor_209 = None
        mul_724 = torch.ops.aten.mul.Tensor(view_1648, convert_element_type_291);  convert_element_type_291 = None
        mul_725 = torch.ops.aten.mul.Tensor(view_1648, view_303);  view_1648 = view_303 = None
        view_1649 = torch.ops.aten.view.default(mul_724, [16384, 14336]);  mul_724 = None
        permute_1097 = torch.ops.aten.permute.default(view_1649, [1, 0])
        mm_551 = torch.ops.aten.mm.default(permute_1097, view_299);  permute_1097 = None
        permute_1099 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_552 = torch.ops.aten.mm.default(view_1649, permute_1099);  view_1649 = permute_1099 = None
        view_1650 = torch.ops.aten.view.default(mm_552, [2, 8192, 4096]);  mm_552 = None
        convert_element_type_2325 = torch.ops.prims.convert_element_type.default(mm_551, torch.float32);  mm_551 = None
        reduce_scatter_tensor_210 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2325, 'avg', 256, '0');  convert_element_type_2325 = None
        wait_tensor_501 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_210);  reduce_scatter_tensor_210 = None
        convert_element_type_2326 = torch.ops.prims.convert_element_type.default(mul_725, torch.float32);  mul_725 = None
        neg_23 = torch.ops.aten.neg.default(convert_element_type_290)
        exp_23 = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_290 = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(add_290);  add_290 = None
        mul_726 = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        mul_727 = torch.ops.aten.mul.Tensor(convert_element_type_2326, mul_726);  convert_element_type_2326 = None
        sub_70 = torch.ops.aten.sub.Tensor(1, mul_726);  mul_726 = None
        mul_728 = torch.ops.aten.mul.Tensor(convert_element_type_290, sub_70);  convert_element_type_290 = sub_70 = None
        add_291 = torch.ops.aten.add.Tensor(mul_728, 1);  mul_728 = None
        mul_729 = torch.ops.aten.mul.Tensor(mul_727, add_291);  mul_727 = add_291 = None
        convert_element_type_2328 = torch.ops.prims.convert_element_type.default(mul_729, torch.bfloat16);  mul_729 = None
        view_1651 = torch.ops.aten.view.default(convert_element_type_2328, [16384, 14336]);  convert_element_type_2328 = None
        permute_1101 = torch.ops.aten.permute.default(view_1651, [1, 0])
        mm_553 = torch.ops.aten.mm.default(permute_1101, view_299);  permute_1101 = view_299 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        all_gather_into_tensor_79 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_287, 256, '0');  convert_element_type_287 = None
        wait_tensor_79 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_79);  all_gather_into_tensor_79 = None
        permute_96 = torch.ops.aten.permute.default(wait_tensor_79, [1, 0]);  wait_tensor_79 = None
        permute_1103 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_554 = torch.ops.aten.mm.default(view_1651, permute_1103);  view_1651 = permute_1103 = None
        view_1652 = torch.ops.aten.view.default(mm_554, [2, 8192, 4096]);  mm_554 = None
        add_292 = torch.ops.aten.add.Tensor(view_1650, view_1652);  view_1650 = view_1652 = None
        convert_element_type_2333 = torch.ops.prims.convert_element_type.default(mm_553, torch.float32);  mm_553 = None
        reduce_scatter_tensor_211 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2333, 'avg', 256, '0');  convert_element_type_2333 = None
        wait_tensor_502 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_211);  reduce_scatter_tensor_211 = None
        convert_element_type_2334 = torch.ops.prims.convert_element_type.default(add_292, torch.float32);  add_292 = None
        convert_element_type_2336 = torch.ops.prims.convert_element_type.default(wait_tensor_78, torch.float32);  wait_tensor_78 = None
        mul_730 = torch.ops.aten.mul.Tensor(convert_element_type_2334, convert_element_type_2336);  convert_element_type_2336 = None
        mul_732 = torch.ops.aten.mul.Tensor(mul_68, mul_730)
        sum_141 = torch.ops.aten.sum.dim_IntList(mul_732, [2], True);  mul_732 = None
        div_47 = torch.ops.aten.div.Tensor(mul_68, 4096)
        mul_733 = torch.ops.aten.mul.Tensor(div_47, sum_141);  div_47 = sum_141 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_730, mul_733);  mul_730 = mul_733 = None
        mul_734 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
        mul_735 = torch.ops.aten.mul.Tensor(convert_element_type_2334, mul_68);  convert_element_type_2334 = mul_68 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(mul_735, [0, 1]);  mul_735 = None
        convert_element_type_2337 = torch.ops.prims.convert_element_type.default(mul_734, torch.bfloat16);  mul_734 = None
        add_293 = torch.ops.aten.add.Tensor(add_289, convert_element_type_2337);  add_289 = convert_element_type_2337 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(sum_142, torch.float32);  sum_142 = None
        reduce_scatter_tensor_212 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_18, 'avg', 256, '0');  convert_element_type_default_18 = None
        wait_tensor_503 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_212);  reduce_scatter_tensor_212 = None
        view_1653 = torch.ops.aten.view.default(add_293, [16384, 4096])
        permute_1105 = torch.ops.aten.permute.default(view_1653, [1, 0])
        mm_555 = torch.ops.aten.mm.default(permute_1105, view_295);  permute_1105 = view_295 = None
        permute_1107 = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        mm_556 = torch.ops.aten.mm.default(view_1653, permute_1107);  view_1653 = permute_1107 = None
        view_1654 = torch.ops.aten.view.default(mm_556, [2, 8192, 4096]);  mm_556 = None
        convert_element_type_2344 = torch.ops.prims.convert_element_type.default(mm_555, torch.float32);  mm_555 = None
        reduce_scatter_tensor_213 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2344, 'avg', 256, '0');  convert_element_type_2344 = None
        wait_tensor_504 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_213);  reduce_scatter_tensor_213 = None
        view_1655 = torch.ops.aten.view.default(view_1654, [2, 8192, 32, 128]);  view_1654 = None
        permute_1109 = torch.ops.aten.permute.default(view_1655, [0, 2, 1, 3]);  view_1655 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        all_gather_into_tensor_73 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_265, 256, '0');  convert_element_type_265 = None
        wait_tensor_73 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_73);  all_gather_into_tensor_73 = None
        convert_element_type_266 = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_32 = torch.ops.aten.add.Scalar(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_16);  convert_element_type_266 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, wait_tensor_73)
        convert_element_type_267 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        view_275 = torch.ops.aten.view.default(convert_element_type_267, [16384, 4096]);  convert_element_type_267 = None
        view_276 = torch.ops.aten.view.default(mm_56, [2, 8192, 4096]);  mm_56 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        all_gather_into_tensor_75 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_271, 256, '0');  convert_element_type_271 = None
        wait_tensor_75 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_75);  all_gather_into_tensor_75 = None
        permute_89 = torch.ops.aten.permute.default(wait_tensor_75, [1, 0]);  wait_tensor_75 = None
        mm_57 = torch.ops.aten.mm.default(view_275, permute_89)
        view_279 = torch.ops.aten.view.default(mm_57, [2, 8192, 1024]);  mm_57 = None
        view_282 = torch.ops.aten.view.default(mm_58, [2, 8192, 1024]);  mm_58 = None
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
        _scaled_dot_product_cudnn_attention_backward_23 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1109, permute_91, permute_92, permute_93, getitem_72, getitem_73, getitem_78, getitem_79, None, None, None, 8192, 8192, 0.0, True);  permute_1109 = permute_91 = permute_92 = permute_93 = getitem_72 = getitem_73 = getitem_78 = getitem_79 = None
        getitem_357 = _scaled_dot_product_cudnn_attention_backward_23[0]
        getitem_358 = _scaled_dot_product_cudnn_attention_backward_23[1]
        getitem_359 = _scaled_dot_product_cudnn_attention_backward_23[2];  _scaled_dot_product_cudnn_attention_backward_23 = None
        permute_1110 = torch.ops.aten.permute.default(getitem_359, [0, 2, 1, 3]);  getitem_359 = None
        permute_1111 = torch.ops.aten.permute.default(getitem_358, [0, 2, 1, 3]);  getitem_358 = None
        permute_1112 = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3]);  getitem_357 = None
        view_1656 = torch.ops.aten.view.default(permute_1110, [2, 8192, 8, 4, 128]);  permute_1110 = None
        sum_143 = torch.ops.aten.sum.dim_IntList(view_1656, [3], True);  view_1656 = None
        squeeze_46 = torch.ops.aten.squeeze.dim(sum_143, 3);  sum_143 = None
        view_1657 = torch.ops.aten.view.default(permute_1111, [2, 8192, 8, 4, 128]);  permute_1111 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(view_1657, [3], True);  view_1657 = None
        squeeze_47 = torch.ops.aten.squeeze.dim(sum_144, 3);  sum_144 = None
        convert_element_type_2345 = torch.ops.prims.convert_element_type.default(squeeze_47, torch.float32);  squeeze_47 = None
        convert_element_type_2346 = torch.ops.prims.convert_element_type.default(permute_1112, torch.float32);  permute_1112 = None
        view_1658 = torch.ops.aten.view.default(convert_element_type_2345, [2, 8192, 8, 64, 2]);  convert_element_type_2345 = None
        view_as_complex_110 = torch.ops.aten.view_as_complex.default(view_1658);  view_1658 = None
        mul_736 = torch.ops.aten.mul.Tensor(view_as_complex_110, _conj);  view_as_complex_110 = None
        view_1659 = torch.ops.aten.view.default(convert_element_type_2346, [2, 8192, 32, 64, 2]);  convert_element_type_2346 = None
        view_as_complex_111 = torch.ops.aten.view_as_complex.default(view_1659);  view_1659 = None
        mul_737 = torch.ops.aten.mul.Tensor(view_as_complex_111, _conj);  view_as_complex_111 = None
        view_as_real_110 = torch.ops.aten.view_as_real.default(mul_736);  mul_736 = None
        view_1660 = torch.ops.aten.view.default(view_as_real_110, [2, 8192, 8, 128]);  view_as_real_110 = None
        convert_element_type_2347 = torch.ops.prims.convert_element_type.default(view_1660, torch.bfloat16);  view_1660 = None
        view_as_real_111 = torch.ops.aten.view_as_real.default(mul_737);  mul_737 = None
        view_1661 = torch.ops.aten.view.default(view_as_real_111, [2, 8192, 32, 128]);  view_as_real_111 = None
        convert_element_type_2348 = torch.ops.prims.convert_element_type.default(view_1661, torch.bfloat16);  view_1661 = None
        view_1662 = torch.ops.aten.view.default(squeeze_46, [2, 8192, 1024]);  squeeze_46 = None
        view_1663 = torch.ops.aten.view.default(convert_element_type_2347, [2, 8192, 1024]);  convert_element_type_2347 = None
        view_1664 = torch.ops.aten.view.default(convert_element_type_2348, [2, 8192, 4096]);  convert_element_type_2348 = None
        view_1665 = torch.ops.aten.view.default(view_1662, [16384, 1024]);  view_1662 = None
        permute_1113 = torch.ops.aten.permute.default(view_1665, [1, 0])
        mm_557 = torch.ops.aten.mm.default(permute_1113, view_275);  permute_1113 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        all_gather_into_tensor_76 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_274, 256, '0');  convert_element_type_274 = None
        wait_tensor_76 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_76);  all_gather_into_tensor_76 = None
        permute_90 = torch.ops.aten.permute.default(wait_tensor_76, [1, 0]);  wait_tensor_76 = None
        permute_1115 = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_558 = torch.ops.aten.mm.default(view_1665, permute_1115);  view_1665 = permute_1115 = None
        view_1666 = torch.ops.aten.view.default(mm_558, [2, 8192, 4096]);  mm_558 = None
        convert_element_type_2353 = torch.ops.prims.convert_element_type.default(mm_557, torch.float32);  mm_557 = None
        reduce_scatter_tensor_214 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2353, 'avg', 256, '0');  convert_element_type_2353 = None
        wait_tensor_505 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_214);  reduce_scatter_tensor_214 = None
        view_1667 = torch.ops.aten.view.default(view_1663, [16384, 1024]);  view_1663 = None
        permute_1117 = torch.ops.aten.permute.default(view_1667, [1, 0])
        mm_559 = torch.ops.aten.mm.default(permute_1117, view_275);  permute_1117 = None
        permute_1119 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_560 = torch.ops.aten.mm.default(view_1667, permute_1119);  view_1667 = permute_1119 = None
        view_1668 = torch.ops.aten.view.default(mm_560, [2, 8192, 4096]);  mm_560 = None
        add_294 = torch.ops.aten.add.Tensor(view_1666, view_1668);  view_1666 = view_1668 = None
        convert_element_type_2358 = torch.ops.prims.convert_element_type.default(mm_559, torch.float32);  mm_559 = None
        reduce_scatter_tensor_215 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2358, 'avg', 256, '0');  convert_element_type_2358 = None
        wait_tensor_506 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_215);  reduce_scatter_tensor_215 = None
        view_1669 = torch.ops.aten.view.default(view_1664, [16384, 4096]);  view_1664 = None
        permute_1121 = torch.ops.aten.permute.default(view_1669, [1, 0])
        mm_561 = torch.ops.aten.mm.default(permute_1121, view_275);  permute_1121 = view_275 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        all_gather_into_tensor_74 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_268, 256, '0');  convert_element_type_268 = None
        wait_tensor_74 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_74);  all_gather_into_tensor_74 = None
        permute_88 = torch.ops.aten.permute.default(wait_tensor_74, [1, 0]);  wait_tensor_74 = None
        permute_1123 = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_562 = torch.ops.aten.mm.default(view_1669, permute_1123);  view_1669 = permute_1123 = None
        view_1670 = torch.ops.aten.view.default(mm_562, [2, 8192, 4096]);  mm_562 = None
        add_295 = torch.ops.aten.add.Tensor(add_294, view_1670);  add_294 = view_1670 = None
        convert_element_type_2363 = torch.ops.prims.convert_element_type.default(mm_561, torch.float32);  mm_561 = None
        reduce_scatter_tensor_216 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2363, 'avg', 256, '0');  convert_element_type_2363 = None
        wait_tensor_507 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_216);  reduce_scatter_tensor_216 = None
        convert_element_type_2364 = torch.ops.prims.convert_element_type.default(add_295, torch.float32);  add_295 = None
        convert_element_type_2366 = torch.ops.prims.convert_element_type.default(wait_tensor_73, torch.float32);  wait_tensor_73 = None
        mul_738 = torch.ops.aten.mul.Tensor(convert_element_type_2364, convert_element_type_2366);  convert_element_type_2366 = None
        mul_740 = torch.ops.aten.mul.Tensor(mul_64, mul_738)
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_740, [2], True);  mul_740 = None
        div_48 = torch.ops.aten.div.Tensor(mul_64, 4096)
        mul_741 = torch.ops.aten.mul.Tensor(div_48, sum_145);  div_48 = sum_145 = None
        sub_72 = torch.ops.aten.sub.Tensor(mul_738, mul_741);  mul_738 = mul_741 = None
        mul_742 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_16);  sub_72 = rsqrt_16 = None
        mul_743 = torch.ops.aten.mul.Tensor(convert_element_type_2364, mul_64);  convert_element_type_2364 = mul_64 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_743, [0, 1]);  mul_743 = None
        convert_element_type_2367 = torch.ops.prims.convert_element_type.default(mul_742, torch.bfloat16);  mul_742 = None
        add_296 = torch.ops.aten.add.Tensor(add_293, convert_element_type_2367);  add_293 = convert_element_type_2367 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(sum_146, torch.float32);  sum_146 = None
        reduce_scatter_tensor_217 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_17, 'avg', 256, '0');  convert_element_type_default_17 = None
        wait_tensor_508 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_217);  reduce_scatter_tensor_217 = None
        view_1671 = torch.ops.aten.view.default(add_296, [16384, 4096])
        permute_1125 = torch.ops.aten.permute.default(view_1671, [1, 0])
        permute_83 = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3])
        view_259 = torch.ops.aten.view.default(permute_83, [2, 8192, -1]);  permute_83 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        all_gather_into_tensor_68 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_248, 256, '0');  convert_element_type_248 = None
        wait_tensor_68 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_68);  all_gather_into_tensor_68 = None
        permute_84 = torch.ops.aten.permute.default(wait_tensor_68, [1, 0]);  wait_tensor_68 = None
        view_261 = torch.ops.aten.view.default(view_259, [16384, 4096]);  view_259 = None
        mm_52 = torch.ops.aten.mm.default(view_261, permute_84)
        view_262 = torch.ops.aten.view.default(mm_52, [2, 8192, 4096]);  mm_52 = None
        add_29 = torch.ops.aten.add.Tensor(add_27, view_262);  view_262 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        all_gather_into_tensor_69 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_251, 256, '0');  convert_element_type_251 = None
        wait_tensor_69 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_69);  all_gather_into_tensor_69 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [2], True);  pow_16 = None
        add_30 = torch.ops.aten.add.Scalar(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_15);  convert_element_type_252 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, wait_tensor_69)
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        view_265 = torch.ops.aten.view.default(convert_element_type_253, [16384, 4096]);  convert_element_type_253 = None
        view_266 = torch.ops.aten.view.default(mm_53, [2, 8192, 14336]);  mm_53 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(view_266, torch.float32);  view_266 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_7);  sigmoid_7 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        all_gather_into_tensor_71 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_259, 256, '0');  convert_element_type_259 = None
        wait_tensor_71 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_71);  all_gather_into_tensor_71 = None
        permute_86 = torch.ops.aten.permute.default(wait_tensor_71, [1, 0]);  wait_tensor_71 = None
        mm_54 = torch.ops.aten.mm.default(view_265, permute_86)
        view_269 = torch.ops.aten.view.default(mm_54, [2, 8192, 14336]);  mm_54 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_258, view_269)
        view_271 = torch.ops.aten.view.default(mul_63, [16384, 14336]);  mul_63 = None
        mm_563 = torch.ops.aten.mm.default(permute_1125, view_271);  permute_1125 = view_271 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        all_gather_into_tensor_72 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_262, 256, '0');  convert_element_type_262 = None
        wait_tensor_72 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_72);  all_gather_into_tensor_72 = None
        permute_87 = torch.ops.aten.permute.default(wait_tensor_72, [1, 0]);  wait_tensor_72 = None
        permute_1127 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_564 = torch.ops.aten.mm.default(view_1671, permute_1127);  view_1671 = permute_1127 = None
        view_1672 = torch.ops.aten.view.default(mm_564, [2, 8192, 14336]);  mm_564 = None
        convert_element_type_2374 = torch.ops.prims.convert_element_type.default(mm_563, torch.float32);  mm_563 = None
        reduce_scatter_tensor_218 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2374, 'avg', 256, '0');  convert_element_type_2374 = None
        wait_tensor_509 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_218);  reduce_scatter_tensor_218 = None
        mul_744 = torch.ops.aten.mul.Tensor(view_1672, convert_element_type_258);  convert_element_type_258 = None
        mul_745 = torch.ops.aten.mul.Tensor(view_1672, view_269);  view_1672 = view_269 = None
        view_1673 = torch.ops.aten.view.default(mul_744, [16384, 14336]);  mul_744 = None
        permute_1129 = torch.ops.aten.permute.default(view_1673, [1, 0])
        mm_565 = torch.ops.aten.mm.default(permute_1129, view_265);  permute_1129 = None
        permute_1131 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_566 = torch.ops.aten.mm.default(view_1673, permute_1131);  view_1673 = permute_1131 = None
        view_1674 = torch.ops.aten.view.default(mm_566, [2, 8192, 4096]);  mm_566 = None
        convert_element_type_2379 = torch.ops.prims.convert_element_type.default(mm_565, torch.float32);  mm_565 = None
        reduce_scatter_tensor_219 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2379, 'avg', 256, '0');  convert_element_type_2379 = None
        wait_tensor_510 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_219);  reduce_scatter_tensor_219 = None
        convert_element_type_2380 = torch.ops.prims.convert_element_type.default(mul_745, torch.float32);  mul_745 = None
        neg_24 = torch.ops.aten.neg.default(convert_element_type_257)
        exp_24 = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_297 = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(add_297);  add_297 = None
        mul_746 = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        mul_747 = torch.ops.aten.mul.Tensor(convert_element_type_2380, mul_746);  convert_element_type_2380 = None
        sub_73 = torch.ops.aten.sub.Tensor(1, mul_746);  mul_746 = None
        mul_748 = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_73);  convert_element_type_257 = sub_73 = None
        add_298 = torch.ops.aten.add.Tensor(mul_748, 1);  mul_748 = None
        mul_749 = torch.ops.aten.mul.Tensor(mul_747, add_298);  mul_747 = add_298 = None
        convert_element_type_2382 = torch.ops.prims.convert_element_type.default(mul_749, torch.bfloat16);  mul_749 = None
        view_1675 = torch.ops.aten.view.default(convert_element_type_2382, [16384, 14336]);  convert_element_type_2382 = None
        permute_1133 = torch.ops.aten.permute.default(view_1675, [1, 0])
        mm_567 = torch.ops.aten.mm.default(permute_1133, view_265);  permute_1133 = view_265 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        all_gather_into_tensor_70 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_254, 256, '0');  convert_element_type_254 = None
        wait_tensor_70 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_70);  all_gather_into_tensor_70 = None
        permute_85 = torch.ops.aten.permute.default(wait_tensor_70, [1, 0]);  wait_tensor_70 = None
        permute_1135 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_568 = torch.ops.aten.mm.default(view_1675, permute_1135);  view_1675 = permute_1135 = None
        view_1676 = torch.ops.aten.view.default(mm_568, [2, 8192, 4096]);  mm_568 = None
        add_299 = torch.ops.aten.add.Tensor(view_1674, view_1676);  view_1674 = view_1676 = None
        convert_element_type_2387 = torch.ops.prims.convert_element_type.default(mm_567, torch.float32);  mm_567 = None
        reduce_scatter_tensor_220 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2387, 'avg', 256, '0');  convert_element_type_2387 = None
        wait_tensor_511 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_220);  reduce_scatter_tensor_220 = None
        convert_element_type_2388 = torch.ops.prims.convert_element_type.default(add_299, torch.float32);  add_299 = None
        convert_element_type_2390 = torch.ops.prims.convert_element_type.default(wait_tensor_69, torch.float32);  wait_tensor_69 = None
        mul_750 = torch.ops.aten.mul.Tensor(convert_element_type_2388, convert_element_type_2390);  convert_element_type_2390 = None
        mul_752 = torch.ops.aten.mul.Tensor(mul_60, mul_750)
        sum_147 = torch.ops.aten.sum.dim_IntList(mul_752, [2], True);  mul_752 = None
        div_49 = torch.ops.aten.div.Tensor(mul_60, 4096)
        mul_753 = torch.ops.aten.mul.Tensor(div_49, sum_147);  div_49 = sum_147 = None
        sub_74 = torch.ops.aten.sub.Tensor(mul_750, mul_753);  mul_750 = mul_753 = None
        mul_754 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_15);  sub_74 = rsqrt_15 = None
        mul_755 = torch.ops.aten.mul.Tensor(convert_element_type_2388, mul_60);  convert_element_type_2388 = mul_60 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(mul_755, [0, 1]);  mul_755 = None
        convert_element_type_2391 = torch.ops.prims.convert_element_type.default(mul_754, torch.bfloat16);  mul_754 = None
        add_300 = torch.ops.aten.add.Tensor(add_296, convert_element_type_2391);  add_296 = convert_element_type_2391 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(sum_148, torch.float32);  sum_148 = None
        reduce_scatter_tensor_221 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_16, 'avg', 256, '0');  convert_element_type_default_16 = None
        wait_tensor_512 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_221);  reduce_scatter_tensor_221 = None
        view_1677 = torch.ops.aten.view.default(add_300, [16384, 4096])
        permute_1137 = torch.ops.aten.permute.default(view_1677, [1, 0])
        mm_569 = torch.ops.aten.mm.default(permute_1137, view_261);  permute_1137 = view_261 = None
        permute_1139 = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        mm_570 = torch.ops.aten.mm.default(view_1677, permute_1139);  view_1677 = permute_1139 = None
        view_1678 = torch.ops.aten.view.default(mm_570, [2, 8192, 4096]);  mm_570 = None
        convert_element_type_2398 = torch.ops.prims.convert_element_type.default(mm_569, torch.float32);  mm_569 = None
        reduce_scatter_tensor_222 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2398, 'avg', 256, '0');  convert_element_type_2398 = None
        wait_tensor_513 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_222);  reduce_scatter_tensor_222 = None
        view_1679 = torch.ops.aten.view.default(view_1678, [2, 8192, 32, 128]);  view_1678 = None
        permute_1141 = torch.ops.aten.permute.default(view_1679, [0, 2, 1, 3]);  view_1679 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        all_gather_into_tensor_64 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_232, 256, '0');  convert_element_type_232 = None
        wait_tensor_64 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_64);  all_gather_into_tensor_64 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_28 = torch.ops.aten.add.Scalar(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_14);  convert_element_type_233 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, wait_tensor_64)
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        view_241 = torch.ops.aten.view.default(convert_element_type_234, [16384, 4096]);  convert_element_type_234 = None
        view_242 = torch.ops.aten.view.default(mm_49, [2, 8192, 4096]);  mm_49 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        all_gather_into_tensor_66 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_238, 256, '0');  convert_element_type_238 = None
        wait_tensor_66 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_66);  all_gather_into_tensor_66 = None
        permute_78 = torch.ops.aten.permute.default(wait_tensor_66, [1, 0]);  wait_tensor_66 = None
        mm_50 = torch.ops.aten.mm.default(view_241, permute_78)
        view_245 = torch.ops.aten.view.default(mm_50, [2, 8192, 1024]);  mm_50 = None
        view_248 = torch.ops.aten.view.default(mm_51, [2, 8192, 1024]);  mm_51 = None
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
        _scaled_dot_product_cudnn_attention_backward_24 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1141, permute_80, permute_81, permute_82, getitem_63, getitem_64, getitem_69, getitem_70, None, None, None, 8192, 8192, 0.0, True);  permute_1141 = permute_80 = permute_81 = permute_82 = getitem_63 = getitem_64 = getitem_69 = getitem_70 = None
        getitem_360 = _scaled_dot_product_cudnn_attention_backward_24[0]
        getitem_361 = _scaled_dot_product_cudnn_attention_backward_24[1]
        getitem_362 = _scaled_dot_product_cudnn_attention_backward_24[2];  _scaled_dot_product_cudnn_attention_backward_24 = None
        permute_1142 = torch.ops.aten.permute.default(getitem_362, [0, 2, 1, 3]);  getitem_362 = None
        permute_1143 = torch.ops.aten.permute.default(getitem_361, [0, 2, 1, 3]);  getitem_361 = None
        permute_1144 = torch.ops.aten.permute.default(getitem_360, [0, 2, 1, 3]);  getitem_360 = None
        view_1680 = torch.ops.aten.view.default(permute_1142, [2, 8192, 8, 4, 128]);  permute_1142 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(view_1680, [3], True);  view_1680 = None
        squeeze_48 = torch.ops.aten.squeeze.dim(sum_149, 3);  sum_149 = None
        view_1681 = torch.ops.aten.view.default(permute_1143, [2, 8192, 8, 4, 128]);  permute_1143 = None
        sum_150 = torch.ops.aten.sum.dim_IntList(view_1681, [3], True);  view_1681 = None
        squeeze_49 = torch.ops.aten.squeeze.dim(sum_150, 3);  sum_150 = None
        convert_element_type_2399 = torch.ops.prims.convert_element_type.default(squeeze_49, torch.float32);  squeeze_49 = None
        convert_element_type_2400 = torch.ops.prims.convert_element_type.default(permute_1144, torch.float32);  permute_1144 = None
        view_1682 = torch.ops.aten.view.default(convert_element_type_2399, [2, 8192, 8, 64, 2]);  convert_element_type_2399 = None
        view_as_complex_112 = torch.ops.aten.view_as_complex.default(view_1682);  view_1682 = None
        mul_756 = torch.ops.aten.mul.Tensor(view_as_complex_112, _conj);  view_as_complex_112 = None
        view_1683 = torch.ops.aten.view.default(convert_element_type_2400, [2, 8192, 32, 64, 2]);  convert_element_type_2400 = None
        view_as_complex_113 = torch.ops.aten.view_as_complex.default(view_1683);  view_1683 = None
        mul_757 = torch.ops.aten.mul.Tensor(view_as_complex_113, _conj);  view_as_complex_113 = None
        view_as_real_112 = torch.ops.aten.view_as_real.default(mul_756);  mul_756 = None
        view_1684 = torch.ops.aten.view.default(view_as_real_112, [2, 8192, 8, 128]);  view_as_real_112 = None
        convert_element_type_2401 = torch.ops.prims.convert_element_type.default(view_1684, torch.bfloat16);  view_1684 = None
        view_as_real_113 = torch.ops.aten.view_as_real.default(mul_757);  mul_757 = None
        view_1685 = torch.ops.aten.view.default(view_as_real_113, [2, 8192, 32, 128]);  view_as_real_113 = None
        convert_element_type_2402 = torch.ops.prims.convert_element_type.default(view_1685, torch.bfloat16);  view_1685 = None
        view_1686 = torch.ops.aten.view.default(squeeze_48, [2, 8192, 1024]);  squeeze_48 = None
        view_1687 = torch.ops.aten.view.default(convert_element_type_2401, [2, 8192, 1024]);  convert_element_type_2401 = None
        view_1688 = torch.ops.aten.view.default(convert_element_type_2402, [2, 8192, 4096]);  convert_element_type_2402 = None
        view_1689 = torch.ops.aten.view.default(view_1686, [16384, 1024]);  view_1686 = None
        permute_1145 = torch.ops.aten.permute.default(view_1689, [1, 0])
        mm_571 = torch.ops.aten.mm.default(permute_1145, view_241);  permute_1145 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        all_gather_into_tensor_67 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_241, 256, '0');  convert_element_type_241 = None
        wait_tensor_67 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_67);  all_gather_into_tensor_67 = None
        permute_79 = torch.ops.aten.permute.default(wait_tensor_67, [1, 0]);  wait_tensor_67 = None
        permute_1147 = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_572 = torch.ops.aten.mm.default(view_1689, permute_1147);  view_1689 = permute_1147 = None
        view_1690 = torch.ops.aten.view.default(mm_572, [2, 8192, 4096]);  mm_572 = None
        convert_element_type_2407 = torch.ops.prims.convert_element_type.default(mm_571, torch.float32);  mm_571 = None
        reduce_scatter_tensor_223 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2407, 'avg', 256, '0');  convert_element_type_2407 = None
        wait_tensor_514 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_223);  reduce_scatter_tensor_223 = None
        view_1691 = torch.ops.aten.view.default(view_1687, [16384, 1024]);  view_1687 = None
        permute_1149 = torch.ops.aten.permute.default(view_1691, [1, 0])
        mm_573 = torch.ops.aten.mm.default(permute_1149, view_241);  permute_1149 = None
        permute_1151 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_574 = torch.ops.aten.mm.default(view_1691, permute_1151);  view_1691 = permute_1151 = None
        view_1692 = torch.ops.aten.view.default(mm_574, [2, 8192, 4096]);  mm_574 = None
        add_301 = torch.ops.aten.add.Tensor(view_1690, view_1692);  view_1690 = view_1692 = None
        convert_element_type_2412 = torch.ops.prims.convert_element_type.default(mm_573, torch.float32);  mm_573 = None
        reduce_scatter_tensor_224 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2412, 'avg', 256, '0');  convert_element_type_2412 = None
        wait_tensor_515 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_224);  reduce_scatter_tensor_224 = None
        view_1693 = torch.ops.aten.view.default(view_1688, [16384, 4096]);  view_1688 = None
        permute_1153 = torch.ops.aten.permute.default(view_1693, [1, 0])
        mm_575 = torch.ops.aten.mm.default(permute_1153, view_241);  permute_1153 = view_241 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        all_gather_into_tensor_65 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_235, 256, '0');  convert_element_type_235 = None
        wait_tensor_65 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_65);  all_gather_into_tensor_65 = None
        permute_77 = torch.ops.aten.permute.default(wait_tensor_65, [1, 0]);  wait_tensor_65 = None
        permute_1155 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_576 = torch.ops.aten.mm.default(view_1693, permute_1155);  view_1693 = permute_1155 = None
        view_1694 = torch.ops.aten.view.default(mm_576, [2, 8192, 4096]);  mm_576 = None
        add_302 = torch.ops.aten.add.Tensor(add_301, view_1694);  add_301 = view_1694 = None
        convert_element_type_2417 = torch.ops.prims.convert_element_type.default(mm_575, torch.float32);  mm_575 = None
        reduce_scatter_tensor_225 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2417, 'avg', 256, '0');  convert_element_type_2417 = None
        wait_tensor_516 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_225);  reduce_scatter_tensor_225 = None
        convert_element_type_2418 = torch.ops.prims.convert_element_type.default(add_302, torch.float32);  add_302 = None
        convert_element_type_2420 = torch.ops.prims.convert_element_type.default(wait_tensor_64, torch.float32);  wait_tensor_64 = None
        mul_758 = torch.ops.aten.mul.Tensor(convert_element_type_2418, convert_element_type_2420);  convert_element_type_2420 = None
        mul_760 = torch.ops.aten.mul.Tensor(mul_56, mul_758)
        sum_151 = torch.ops.aten.sum.dim_IntList(mul_760, [2], True);  mul_760 = None
        div_50 = torch.ops.aten.div.Tensor(mul_56, 4096)
        mul_761 = torch.ops.aten.mul.Tensor(div_50, sum_151);  div_50 = sum_151 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_758, mul_761);  mul_758 = mul_761 = None
        mul_762 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_14);  sub_75 = rsqrt_14 = None
        mul_763 = torch.ops.aten.mul.Tensor(convert_element_type_2418, mul_56);  convert_element_type_2418 = mul_56 = None
        sum_152 = torch.ops.aten.sum.dim_IntList(mul_763, [0, 1]);  mul_763 = None
        convert_element_type_2421 = torch.ops.prims.convert_element_type.default(mul_762, torch.bfloat16);  mul_762 = None
        add_303 = torch.ops.aten.add.Tensor(add_300, convert_element_type_2421);  add_300 = convert_element_type_2421 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(sum_152, torch.float32);  sum_152 = None
        reduce_scatter_tensor_226 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_15, 'avg', 256, '0');  convert_element_type_default_15 = None
        wait_tensor_517 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_226);  reduce_scatter_tensor_226 = None
        view_1695 = torch.ops.aten.view.default(add_303, [16384, 4096])
        permute_1157 = torch.ops.aten.permute.default(view_1695, [1, 0])
        permute_72 = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3])
        view_225 = torch.ops.aten.view.default(permute_72, [2, 8192, -1]);  permute_72 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        all_gather_into_tensor_59 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_215, 256, '0');  convert_element_type_215 = None
        wait_tensor_59 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_59);  all_gather_into_tensor_59 = None
        permute_73 = torch.ops.aten.permute.default(wait_tensor_59, [1, 0]);  wait_tensor_59 = None
        view_227 = torch.ops.aten.view.default(view_225, [16384, 4096]);  view_225 = None
        mm_45 = torch.ops.aten.mm.default(view_227, permute_73)
        view_228 = torch.ops.aten.view.default(mm_45, [2, 8192, 4096]);  mm_45 = None
        add_25 = torch.ops.aten.add.Tensor(add_23, view_228);  view_228 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        all_gather_into_tensor_60 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_218, 256, '0');  convert_element_type_218 = None
        wait_tensor_60 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_60);  all_gather_into_tensor_60 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [2], True);  pow_14 = None
        add_26 = torch.ops.aten.add.Scalar(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_13);  convert_element_type_219 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, wait_tensor_60)
        convert_element_type_220 = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_231 = torch.ops.aten.view.default(convert_element_type_220, [16384, 4096]);  convert_element_type_220 = None
        view_232 = torch.ops.aten.view.default(mm_46, [2, 8192, 14336]);  mm_46 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_224)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_224, sigmoid_6);  sigmoid_6 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        all_gather_into_tensor_62 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_226, 256, '0');  convert_element_type_226 = None
        wait_tensor_62 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_62);  all_gather_into_tensor_62 = None
        permute_75 = torch.ops.aten.permute.default(wait_tensor_62, [1, 0]);  wait_tensor_62 = None
        mm_47 = torch.ops.aten.mm.default(view_231, permute_75)
        view_235 = torch.ops.aten.view.default(mm_47, [2, 8192, 14336]);  mm_47 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_225, view_235)
        view_237 = torch.ops.aten.view.default(mul_55, [16384, 14336]);  mul_55 = None
        mm_577 = torch.ops.aten.mm.default(permute_1157, view_237);  permute_1157 = view_237 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        all_gather_into_tensor_63 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_229, 256, '0');  convert_element_type_229 = None
        wait_tensor_63 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_63);  all_gather_into_tensor_63 = None
        permute_76 = torch.ops.aten.permute.default(wait_tensor_63, [1, 0]);  wait_tensor_63 = None
        permute_1159 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_578 = torch.ops.aten.mm.default(view_1695, permute_1159);  view_1695 = permute_1159 = None
        view_1696 = torch.ops.aten.view.default(mm_578, [2, 8192, 14336]);  mm_578 = None
        convert_element_type_2428 = torch.ops.prims.convert_element_type.default(mm_577, torch.float32);  mm_577 = None
        reduce_scatter_tensor_227 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2428, 'avg', 256, '0');  convert_element_type_2428 = None
        wait_tensor_518 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_227);  reduce_scatter_tensor_227 = None
        mul_764 = torch.ops.aten.mul.Tensor(view_1696, convert_element_type_225);  convert_element_type_225 = None
        mul_765 = torch.ops.aten.mul.Tensor(view_1696, view_235);  view_1696 = view_235 = None
        view_1697 = torch.ops.aten.view.default(mul_764, [16384, 14336]);  mul_764 = None
        permute_1161 = torch.ops.aten.permute.default(view_1697, [1, 0])
        mm_579 = torch.ops.aten.mm.default(permute_1161, view_231);  permute_1161 = None
        permute_1163 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_580 = torch.ops.aten.mm.default(view_1697, permute_1163);  view_1697 = permute_1163 = None
        view_1698 = torch.ops.aten.view.default(mm_580, [2, 8192, 4096]);  mm_580 = None
        convert_element_type_2433 = torch.ops.prims.convert_element_type.default(mm_579, torch.float32);  mm_579 = None
        reduce_scatter_tensor_228 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2433, 'avg', 256, '0');  convert_element_type_2433 = None
        wait_tensor_519 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_228);  reduce_scatter_tensor_228 = None
        convert_element_type_2434 = torch.ops.prims.convert_element_type.default(mul_765, torch.float32);  mul_765 = None
        neg_25 = torch.ops.aten.neg.default(convert_element_type_224)
        exp_25 = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_304 = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(add_304);  add_304 = None
        mul_766 = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        mul_767 = torch.ops.aten.mul.Tensor(convert_element_type_2434, mul_766);  convert_element_type_2434 = None
        sub_76 = torch.ops.aten.sub.Tensor(1, mul_766);  mul_766 = None
        mul_768 = torch.ops.aten.mul.Tensor(convert_element_type_224, sub_76);  convert_element_type_224 = sub_76 = None
        add_305 = torch.ops.aten.add.Tensor(mul_768, 1);  mul_768 = None
        mul_769 = torch.ops.aten.mul.Tensor(mul_767, add_305);  mul_767 = add_305 = None
        convert_element_type_2436 = torch.ops.prims.convert_element_type.default(mul_769, torch.bfloat16);  mul_769 = None
        view_1699 = torch.ops.aten.view.default(convert_element_type_2436, [16384, 14336]);  convert_element_type_2436 = None
        permute_1165 = torch.ops.aten.permute.default(view_1699, [1, 0])
        mm_581 = torch.ops.aten.mm.default(permute_1165, view_231);  permute_1165 = view_231 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        all_gather_into_tensor_61 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_221, 256, '0');  convert_element_type_221 = None
        wait_tensor_61 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_61);  all_gather_into_tensor_61 = None
        permute_74 = torch.ops.aten.permute.default(wait_tensor_61, [1, 0]);  wait_tensor_61 = None
        permute_1167 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_582 = torch.ops.aten.mm.default(view_1699, permute_1167);  view_1699 = permute_1167 = None
        view_1700 = torch.ops.aten.view.default(mm_582, [2, 8192, 4096]);  mm_582 = None
        add_306 = torch.ops.aten.add.Tensor(view_1698, view_1700);  view_1698 = view_1700 = None
        convert_element_type_2441 = torch.ops.prims.convert_element_type.default(mm_581, torch.float32);  mm_581 = None
        reduce_scatter_tensor_229 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2441, 'avg', 256, '0');  convert_element_type_2441 = None
        wait_tensor_520 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_229);  reduce_scatter_tensor_229 = None
        convert_element_type_2442 = torch.ops.prims.convert_element_type.default(add_306, torch.float32);  add_306 = None
        convert_element_type_2444 = torch.ops.prims.convert_element_type.default(wait_tensor_60, torch.float32);  wait_tensor_60 = None
        mul_770 = torch.ops.aten.mul.Tensor(convert_element_type_2442, convert_element_type_2444);  convert_element_type_2444 = None
        mul_772 = torch.ops.aten.mul.Tensor(mul_52, mul_770)
        sum_153 = torch.ops.aten.sum.dim_IntList(mul_772, [2], True);  mul_772 = None
        div_51 = torch.ops.aten.div.Tensor(mul_52, 4096)
        mul_773 = torch.ops.aten.mul.Tensor(div_51, sum_153);  div_51 = sum_153 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_770, mul_773);  mul_770 = mul_773 = None
        mul_774 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_13);  sub_77 = rsqrt_13 = None
        mul_775 = torch.ops.aten.mul.Tensor(convert_element_type_2442, mul_52);  convert_element_type_2442 = mul_52 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(mul_775, [0, 1]);  mul_775 = None
        convert_element_type_2445 = torch.ops.prims.convert_element_type.default(mul_774, torch.bfloat16);  mul_774 = None
        add_307 = torch.ops.aten.add.Tensor(add_303, convert_element_type_2445);  add_303 = convert_element_type_2445 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(sum_154, torch.float32);  sum_154 = None
        reduce_scatter_tensor_230 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_14, 'avg', 256, '0');  convert_element_type_default_14 = None
        wait_tensor_521 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_230);  reduce_scatter_tensor_230 = None
        view_1701 = torch.ops.aten.view.default(add_307, [16384, 4096])
        permute_1169 = torch.ops.aten.permute.default(view_1701, [1, 0])
        mm_583 = torch.ops.aten.mm.default(permute_1169, view_227);  permute_1169 = view_227 = None
        permute_1171 = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        mm_584 = torch.ops.aten.mm.default(view_1701, permute_1171);  view_1701 = permute_1171 = None
        view_1702 = torch.ops.aten.view.default(mm_584, [2, 8192, 4096]);  mm_584 = None
        convert_element_type_2452 = torch.ops.prims.convert_element_type.default(mm_583, torch.float32);  mm_583 = None
        reduce_scatter_tensor_231 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2452, 'avg', 256, '0');  convert_element_type_2452 = None
        wait_tensor_522 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_231);  reduce_scatter_tensor_231 = None
        view_1703 = torch.ops.aten.view.default(view_1702, [2, 8192, 32, 128]);  view_1702 = None
        permute_1173 = torch.ops.aten.permute.default(view_1703, [0, 2, 1, 3]);  view_1703 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        all_gather_into_tensor_55 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_199, 256, '0');  convert_element_type_199 = None
        wait_tensor_55 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_55);  all_gather_into_tensor_55 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [2], True);  pow_13 = None
        add_24 = torch.ops.aten.add.Scalar(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_12);  convert_element_type_200 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, wait_tensor_55)
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_207 = torch.ops.aten.view.default(convert_element_type_201, [16384, 4096]);  convert_element_type_201 = None
        view_208 = torch.ops.aten.view.default(mm_42, [2, 8192, 4096]);  mm_42 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        all_gather_into_tensor_57 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_205, 256, '0');  convert_element_type_205 = None
        wait_tensor_57 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_57);  all_gather_into_tensor_57 = None
        permute_67 = torch.ops.aten.permute.default(wait_tensor_57, [1, 0]);  wait_tensor_57 = None
        mm_43 = torch.ops.aten.mm.default(view_207, permute_67)
        view_211 = torch.ops.aten.view.default(mm_43, [2, 8192, 1024]);  mm_43 = None
        view_214 = torch.ops.aten.view.default(mm_44, [2, 8192, 1024]);  mm_44 = None
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
        _scaled_dot_product_cudnn_attention_backward_25 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1173, permute_69, permute_70, permute_71, getitem_54, getitem_55, getitem_60, getitem_61, None, None, None, 8192, 8192, 0.0, True);  permute_1173 = permute_69 = permute_70 = permute_71 = getitem_54 = getitem_55 = getitem_60 = getitem_61 = None
        getitem_363 = _scaled_dot_product_cudnn_attention_backward_25[0]
        getitem_364 = _scaled_dot_product_cudnn_attention_backward_25[1]
        getitem_365 = _scaled_dot_product_cudnn_attention_backward_25[2];  _scaled_dot_product_cudnn_attention_backward_25 = None
        permute_1174 = torch.ops.aten.permute.default(getitem_365, [0, 2, 1, 3]);  getitem_365 = None
        permute_1175 = torch.ops.aten.permute.default(getitem_364, [0, 2, 1, 3]);  getitem_364 = None
        permute_1176 = torch.ops.aten.permute.default(getitem_363, [0, 2, 1, 3]);  getitem_363 = None
        view_1704 = torch.ops.aten.view.default(permute_1174, [2, 8192, 8, 4, 128]);  permute_1174 = None
        sum_155 = torch.ops.aten.sum.dim_IntList(view_1704, [3], True);  view_1704 = None
        squeeze_50 = torch.ops.aten.squeeze.dim(sum_155, 3);  sum_155 = None
        view_1705 = torch.ops.aten.view.default(permute_1175, [2, 8192, 8, 4, 128]);  permute_1175 = None
        sum_156 = torch.ops.aten.sum.dim_IntList(view_1705, [3], True);  view_1705 = None
        squeeze_51 = torch.ops.aten.squeeze.dim(sum_156, 3);  sum_156 = None
        convert_element_type_2453 = torch.ops.prims.convert_element_type.default(squeeze_51, torch.float32);  squeeze_51 = None
        convert_element_type_2454 = torch.ops.prims.convert_element_type.default(permute_1176, torch.float32);  permute_1176 = None
        view_1706 = torch.ops.aten.view.default(convert_element_type_2453, [2, 8192, 8, 64, 2]);  convert_element_type_2453 = None
        view_as_complex_114 = torch.ops.aten.view_as_complex.default(view_1706);  view_1706 = None
        mul_776 = torch.ops.aten.mul.Tensor(view_as_complex_114, _conj);  view_as_complex_114 = None
        view_1707 = torch.ops.aten.view.default(convert_element_type_2454, [2, 8192, 32, 64, 2]);  convert_element_type_2454 = None
        view_as_complex_115 = torch.ops.aten.view_as_complex.default(view_1707);  view_1707 = None
        mul_777 = torch.ops.aten.mul.Tensor(view_as_complex_115, _conj);  view_as_complex_115 = None
        view_as_real_114 = torch.ops.aten.view_as_real.default(mul_776);  mul_776 = None
        view_1708 = torch.ops.aten.view.default(view_as_real_114, [2, 8192, 8, 128]);  view_as_real_114 = None
        convert_element_type_2455 = torch.ops.prims.convert_element_type.default(view_1708, torch.bfloat16);  view_1708 = None
        view_as_real_115 = torch.ops.aten.view_as_real.default(mul_777);  mul_777 = None
        view_1709 = torch.ops.aten.view.default(view_as_real_115, [2, 8192, 32, 128]);  view_as_real_115 = None
        convert_element_type_2456 = torch.ops.prims.convert_element_type.default(view_1709, torch.bfloat16);  view_1709 = None
        view_1710 = torch.ops.aten.view.default(squeeze_50, [2, 8192, 1024]);  squeeze_50 = None
        view_1711 = torch.ops.aten.view.default(convert_element_type_2455, [2, 8192, 1024]);  convert_element_type_2455 = None
        view_1712 = torch.ops.aten.view.default(convert_element_type_2456, [2, 8192, 4096]);  convert_element_type_2456 = None
        view_1713 = torch.ops.aten.view.default(view_1710, [16384, 1024]);  view_1710 = None
        permute_1177 = torch.ops.aten.permute.default(view_1713, [1, 0])
        mm_585 = torch.ops.aten.mm.default(permute_1177, view_207);  permute_1177 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        all_gather_into_tensor_58 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_208, 256, '0');  convert_element_type_208 = None
        wait_tensor_58 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_58);  all_gather_into_tensor_58 = None
        permute_68 = torch.ops.aten.permute.default(wait_tensor_58, [1, 0]);  wait_tensor_58 = None
        permute_1179 = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_586 = torch.ops.aten.mm.default(view_1713, permute_1179);  view_1713 = permute_1179 = None
        view_1714 = torch.ops.aten.view.default(mm_586, [2, 8192, 4096]);  mm_586 = None
        convert_element_type_2461 = torch.ops.prims.convert_element_type.default(mm_585, torch.float32);  mm_585 = None
        reduce_scatter_tensor_232 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2461, 'avg', 256, '0');  convert_element_type_2461 = None
        wait_tensor_523 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_232);  reduce_scatter_tensor_232 = None
        view_1715 = torch.ops.aten.view.default(view_1711, [16384, 1024]);  view_1711 = None
        permute_1181 = torch.ops.aten.permute.default(view_1715, [1, 0])
        mm_587 = torch.ops.aten.mm.default(permute_1181, view_207);  permute_1181 = None
        permute_1183 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_588 = torch.ops.aten.mm.default(view_1715, permute_1183);  view_1715 = permute_1183 = None
        view_1716 = torch.ops.aten.view.default(mm_588, [2, 8192, 4096]);  mm_588 = None
        add_308 = torch.ops.aten.add.Tensor(view_1714, view_1716);  view_1714 = view_1716 = None
        convert_element_type_2466 = torch.ops.prims.convert_element_type.default(mm_587, torch.float32);  mm_587 = None
        reduce_scatter_tensor_233 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2466, 'avg', 256, '0');  convert_element_type_2466 = None
        wait_tensor_524 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_233);  reduce_scatter_tensor_233 = None
        view_1717 = torch.ops.aten.view.default(view_1712, [16384, 4096]);  view_1712 = None
        permute_1185 = torch.ops.aten.permute.default(view_1717, [1, 0])
        mm_589 = torch.ops.aten.mm.default(permute_1185, view_207);  permute_1185 = view_207 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        all_gather_into_tensor_56 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_202, 256, '0');  convert_element_type_202 = None
        wait_tensor_56 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_56);  all_gather_into_tensor_56 = None
        permute_66 = torch.ops.aten.permute.default(wait_tensor_56, [1, 0]);  wait_tensor_56 = None
        permute_1187 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_590 = torch.ops.aten.mm.default(view_1717, permute_1187);  view_1717 = permute_1187 = None
        view_1718 = torch.ops.aten.view.default(mm_590, [2, 8192, 4096]);  mm_590 = None
        add_309 = torch.ops.aten.add.Tensor(add_308, view_1718);  add_308 = view_1718 = None
        convert_element_type_2471 = torch.ops.prims.convert_element_type.default(mm_589, torch.float32);  mm_589 = None
        reduce_scatter_tensor_234 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2471, 'avg', 256, '0');  convert_element_type_2471 = None
        wait_tensor_525 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_234);  reduce_scatter_tensor_234 = None
        convert_element_type_2472 = torch.ops.prims.convert_element_type.default(add_309, torch.float32);  add_309 = None
        convert_element_type_2474 = torch.ops.prims.convert_element_type.default(wait_tensor_55, torch.float32);  wait_tensor_55 = None
        mul_778 = torch.ops.aten.mul.Tensor(convert_element_type_2472, convert_element_type_2474);  convert_element_type_2474 = None
        mul_780 = torch.ops.aten.mul.Tensor(mul_48, mul_778)
        sum_157 = torch.ops.aten.sum.dim_IntList(mul_780, [2], True);  mul_780 = None
        div_52 = torch.ops.aten.div.Tensor(mul_48, 4096)
        mul_781 = torch.ops.aten.mul.Tensor(div_52, sum_157);  div_52 = sum_157 = None
        sub_78 = torch.ops.aten.sub.Tensor(mul_778, mul_781);  mul_778 = mul_781 = None
        mul_782 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_12);  sub_78 = rsqrt_12 = None
        mul_783 = torch.ops.aten.mul.Tensor(convert_element_type_2472, mul_48);  convert_element_type_2472 = mul_48 = None
        sum_158 = torch.ops.aten.sum.dim_IntList(mul_783, [0, 1]);  mul_783 = None
        convert_element_type_2475 = torch.ops.prims.convert_element_type.default(mul_782, torch.bfloat16);  mul_782 = None
        add_310 = torch.ops.aten.add.Tensor(add_307, convert_element_type_2475);  add_307 = convert_element_type_2475 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(sum_158, torch.float32);  sum_158 = None
        reduce_scatter_tensor_235 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_13, 'avg', 256, '0');  convert_element_type_default_13 = None
        wait_tensor_526 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_235);  reduce_scatter_tensor_235 = None
        view_1719 = torch.ops.aten.view.default(add_310, [16384, 4096])
        permute_1189 = torch.ops.aten.permute.default(view_1719, [1, 0])
        permute_61 = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3])
        view_191 = torch.ops.aten.view.default(permute_61, [2, 8192, -1]);  permute_61 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        all_gather_into_tensor_50 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_182, 256, '0');  convert_element_type_182 = None
        wait_tensor_50 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_50);  all_gather_into_tensor_50 = None
        permute_62 = torch.ops.aten.permute.default(wait_tensor_50, [1, 0]);  wait_tensor_50 = None
        view_193 = torch.ops.aten.view.default(view_191, [16384, 4096]);  view_191 = None
        mm_38 = torch.ops.aten.mm.default(view_193, permute_62)
        view_194 = torch.ops.aten.view.default(mm_38, [2, 8192, 4096]);  mm_38 = None
        add_21 = torch.ops.aten.add.Tensor(add_19, view_194);  view_194 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        all_gather_into_tensor_51 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_185, 256, '0');  convert_element_type_185 = None
        wait_tensor_51 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_51);  all_gather_into_tensor_51 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_22 = torch.ops.aten.add.Scalar(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_186, rsqrt_11);  convert_element_type_186 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, wait_tensor_51)
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        view_197 = torch.ops.aten.view.default(convert_element_type_187, [16384, 4096]);  convert_element_type_187 = None
        view_198 = torch.ops.aten.view.default(mm_39, [2, 8192, 14336]);  mm_39 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_191)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_191, sigmoid_5);  sigmoid_5 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        all_gather_into_tensor_53 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_193, 256, '0');  convert_element_type_193 = None
        wait_tensor_53 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_53);  all_gather_into_tensor_53 = None
        permute_64 = torch.ops.aten.permute.default(wait_tensor_53, [1, 0]);  wait_tensor_53 = None
        mm_40 = torch.ops.aten.mm.default(view_197, permute_64)
        view_201 = torch.ops.aten.view.default(mm_40, [2, 8192, 14336]);  mm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_192, view_201)
        view_203 = torch.ops.aten.view.default(mul_47, [16384, 14336]);  mul_47 = None
        mm_591 = torch.ops.aten.mm.default(permute_1189, view_203);  permute_1189 = view_203 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        all_gather_into_tensor_54 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_196, 256, '0');  convert_element_type_196 = None
        wait_tensor_54 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_54);  all_gather_into_tensor_54 = None
        permute_65 = torch.ops.aten.permute.default(wait_tensor_54, [1, 0]);  wait_tensor_54 = None
        permute_1191 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_592 = torch.ops.aten.mm.default(view_1719, permute_1191);  view_1719 = permute_1191 = None
        view_1720 = torch.ops.aten.view.default(mm_592, [2, 8192, 14336]);  mm_592 = None
        convert_element_type_2482 = torch.ops.prims.convert_element_type.default(mm_591, torch.float32);  mm_591 = None
        reduce_scatter_tensor_236 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2482, 'avg', 256, '0');  convert_element_type_2482 = None
        wait_tensor_527 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_236);  reduce_scatter_tensor_236 = None
        mul_784 = torch.ops.aten.mul.Tensor(view_1720, convert_element_type_192);  convert_element_type_192 = None
        mul_785 = torch.ops.aten.mul.Tensor(view_1720, view_201);  view_1720 = view_201 = None
        view_1721 = torch.ops.aten.view.default(mul_784, [16384, 14336]);  mul_784 = None
        permute_1193 = torch.ops.aten.permute.default(view_1721, [1, 0])
        mm_593 = torch.ops.aten.mm.default(permute_1193, view_197);  permute_1193 = None
        permute_1195 = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_594 = torch.ops.aten.mm.default(view_1721, permute_1195);  view_1721 = permute_1195 = None
        view_1722 = torch.ops.aten.view.default(mm_594, [2, 8192, 4096]);  mm_594 = None
        convert_element_type_2487 = torch.ops.prims.convert_element_type.default(mm_593, torch.float32);  mm_593 = None
        reduce_scatter_tensor_237 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2487, 'avg', 256, '0');  convert_element_type_2487 = None
        wait_tensor_528 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_237);  reduce_scatter_tensor_237 = None
        convert_element_type_2488 = torch.ops.prims.convert_element_type.default(mul_785, torch.float32);  mul_785 = None
        neg_26 = torch.ops.aten.neg.default(convert_element_type_191)
        exp_26 = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_311 = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(add_311);  add_311 = None
        mul_786 = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        mul_787 = torch.ops.aten.mul.Tensor(convert_element_type_2488, mul_786);  convert_element_type_2488 = None
        sub_79 = torch.ops.aten.sub.Tensor(1, mul_786);  mul_786 = None
        mul_788 = torch.ops.aten.mul.Tensor(convert_element_type_191, sub_79);  convert_element_type_191 = sub_79 = None
        add_312 = torch.ops.aten.add.Tensor(mul_788, 1);  mul_788 = None
        mul_789 = torch.ops.aten.mul.Tensor(mul_787, add_312);  mul_787 = add_312 = None
        convert_element_type_2490 = torch.ops.prims.convert_element_type.default(mul_789, torch.bfloat16);  mul_789 = None
        view_1723 = torch.ops.aten.view.default(convert_element_type_2490, [16384, 14336]);  convert_element_type_2490 = None
        permute_1197 = torch.ops.aten.permute.default(view_1723, [1, 0])
        mm_595 = torch.ops.aten.mm.default(permute_1197, view_197);  permute_1197 = view_197 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        all_gather_into_tensor_52 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_188, 256, '0');  convert_element_type_188 = None
        wait_tensor_52 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_52);  all_gather_into_tensor_52 = None
        permute_63 = torch.ops.aten.permute.default(wait_tensor_52, [1, 0]);  wait_tensor_52 = None
        permute_1199 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_596 = torch.ops.aten.mm.default(view_1723, permute_1199);  view_1723 = permute_1199 = None
        view_1724 = torch.ops.aten.view.default(mm_596, [2, 8192, 4096]);  mm_596 = None
        add_313 = torch.ops.aten.add.Tensor(view_1722, view_1724);  view_1722 = view_1724 = None
        convert_element_type_2495 = torch.ops.prims.convert_element_type.default(mm_595, torch.float32);  mm_595 = None
        reduce_scatter_tensor_238 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2495, 'avg', 256, '0');  convert_element_type_2495 = None
        wait_tensor_529 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_238);  reduce_scatter_tensor_238 = None
        convert_element_type_2496 = torch.ops.prims.convert_element_type.default(add_313, torch.float32);  add_313 = None
        convert_element_type_2498 = torch.ops.prims.convert_element_type.default(wait_tensor_51, torch.float32);  wait_tensor_51 = None
        mul_790 = torch.ops.aten.mul.Tensor(convert_element_type_2496, convert_element_type_2498);  convert_element_type_2498 = None
        mul_792 = torch.ops.aten.mul.Tensor(mul_44, mul_790)
        sum_159 = torch.ops.aten.sum.dim_IntList(mul_792, [2], True);  mul_792 = None
        div_53 = torch.ops.aten.div.Tensor(mul_44, 4096)
        mul_793 = torch.ops.aten.mul.Tensor(div_53, sum_159);  div_53 = sum_159 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_790, mul_793);  mul_790 = mul_793 = None
        mul_794 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_11);  sub_80 = rsqrt_11 = None
        mul_795 = torch.ops.aten.mul.Tensor(convert_element_type_2496, mul_44);  convert_element_type_2496 = mul_44 = None
        sum_160 = torch.ops.aten.sum.dim_IntList(mul_795, [0, 1]);  mul_795 = None
        convert_element_type_2499 = torch.ops.prims.convert_element_type.default(mul_794, torch.bfloat16);  mul_794 = None
        add_314 = torch.ops.aten.add.Tensor(add_310, convert_element_type_2499);  add_310 = convert_element_type_2499 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(sum_160, torch.float32);  sum_160 = None
        reduce_scatter_tensor_239 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_12, 'avg', 256, '0');  convert_element_type_default_12 = None
        wait_tensor_530 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_239);  reduce_scatter_tensor_239 = None
        view_1725 = torch.ops.aten.view.default(add_314, [16384, 4096])
        permute_1201 = torch.ops.aten.permute.default(view_1725, [1, 0])
        mm_597 = torch.ops.aten.mm.default(permute_1201, view_193);  permute_1201 = view_193 = None
        permute_1203 = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_598 = torch.ops.aten.mm.default(view_1725, permute_1203);  view_1725 = permute_1203 = None
        view_1726 = torch.ops.aten.view.default(mm_598, [2, 8192, 4096]);  mm_598 = None
        convert_element_type_2506 = torch.ops.prims.convert_element_type.default(mm_597, torch.float32);  mm_597 = None
        reduce_scatter_tensor_240 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2506, 'avg', 256, '0');  convert_element_type_2506 = None
        wait_tensor_531 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_240);  reduce_scatter_tensor_240 = None
        view_1727 = torch.ops.aten.view.default(view_1726, [2, 8192, 32, 128]);  view_1726 = None
        permute_1205 = torch.ops.aten.permute.default(view_1727, [0, 2, 1, 3]);  view_1727 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        all_gather_into_tensor_46 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_166, 256, '0');  convert_element_type_166 = None
        wait_tensor_46 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_46);  all_gather_into_tensor_46 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
        add_20 = torch.ops.aten.add.Scalar(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_167, rsqrt_10);  convert_element_type_167 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, wait_tensor_46)
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        view_173 = torch.ops.aten.view.default(convert_element_type_168, [16384, 4096]);  convert_element_type_168 = None
        view_174 = torch.ops.aten.view.default(mm_35, [2, 8192, 4096]);  mm_35 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        all_gather_into_tensor_48 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_172, 256, '0');  convert_element_type_172 = None
        wait_tensor_48 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_48);  all_gather_into_tensor_48 = None
        permute_56 = torch.ops.aten.permute.default(wait_tensor_48, [1, 0]);  wait_tensor_48 = None
        mm_36 = torch.ops.aten.mm.default(view_173, permute_56)
        view_177 = torch.ops.aten.view.default(mm_36, [2, 8192, 1024]);  mm_36 = None
        view_180 = torch.ops.aten.view.default(mm_37, [2, 8192, 1024]);  mm_37 = None
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
        _scaled_dot_product_cudnn_attention_backward_26 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1205, permute_58, permute_59, permute_60, getitem_45, getitem_46, getitem_51, getitem_52, None, None, None, 8192, 8192, 0.0, True);  permute_1205 = permute_58 = permute_59 = permute_60 = getitem_45 = getitem_46 = getitem_51 = getitem_52 = None
        getitem_366 = _scaled_dot_product_cudnn_attention_backward_26[0]
        getitem_367 = _scaled_dot_product_cudnn_attention_backward_26[1]
        getitem_368 = _scaled_dot_product_cudnn_attention_backward_26[2];  _scaled_dot_product_cudnn_attention_backward_26 = None
        permute_1206 = torch.ops.aten.permute.default(getitem_368, [0, 2, 1, 3]);  getitem_368 = None
        permute_1207 = torch.ops.aten.permute.default(getitem_367, [0, 2, 1, 3]);  getitem_367 = None
        permute_1208 = torch.ops.aten.permute.default(getitem_366, [0, 2, 1, 3]);  getitem_366 = None
        view_1728 = torch.ops.aten.view.default(permute_1206, [2, 8192, 8, 4, 128]);  permute_1206 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(view_1728, [3], True);  view_1728 = None
        squeeze_52 = torch.ops.aten.squeeze.dim(sum_161, 3);  sum_161 = None
        view_1729 = torch.ops.aten.view.default(permute_1207, [2, 8192, 8, 4, 128]);  permute_1207 = None
        sum_162 = torch.ops.aten.sum.dim_IntList(view_1729, [3], True);  view_1729 = None
        squeeze_53 = torch.ops.aten.squeeze.dim(sum_162, 3);  sum_162 = None
        convert_element_type_2507 = torch.ops.prims.convert_element_type.default(squeeze_53, torch.float32);  squeeze_53 = None
        convert_element_type_2508 = torch.ops.prims.convert_element_type.default(permute_1208, torch.float32);  permute_1208 = None
        view_1730 = torch.ops.aten.view.default(convert_element_type_2507, [2, 8192, 8, 64, 2]);  convert_element_type_2507 = None
        view_as_complex_116 = torch.ops.aten.view_as_complex.default(view_1730);  view_1730 = None
        mul_796 = torch.ops.aten.mul.Tensor(view_as_complex_116, _conj);  view_as_complex_116 = None
        view_1731 = torch.ops.aten.view.default(convert_element_type_2508, [2, 8192, 32, 64, 2]);  convert_element_type_2508 = None
        view_as_complex_117 = torch.ops.aten.view_as_complex.default(view_1731);  view_1731 = None
        mul_797 = torch.ops.aten.mul.Tensor(view_as_complex_117, _conj);  view_as_complex_117 = None
        view_as_real_116 = torch.ops.aten.view_as_real.default(mul_796);  mul_796 = None
        view_1732 = torch.ops.aten.view.default(view_as_real_116, [2, 8192, 8, 128]);  view_as_real_116 = None
        convert_element_type_2509 = torch.ops.prims.convert_element_type.default(view_1732, torch.bfloat16);  view_1732 = None
        view_as_real_117 = torch.ops.aten.view_as_real.default(mul_797);  mul_797 = None
        view_1733 = torch.ops.aten.view.default(view_as_real_117, [2, 8192, 32, 128]);  view_as_real_117 = None
        convert_element_type_2510 = torch.ops.prims.convert_element_type.default(view_1733, torch.bfloat16);  view_1733 = None
        view_1734 = torch.ops.aten.view.default(squeeze_52, [2, 8192, 1024]);  squeeze_52 = None
        view_1735 = torch.ops.aten.view.default(convert_element_type_2509, [2, 8192, 1024]);  convert_element_type_2509 = None
        view_1736 = torch.ops.aten.view.default(convert_element_type_2510, [2, 8192, 4096]);  convert_element_type_2510 = None
        view_1737 = torch.ops.aten.view.default(view_1734, [16384, 1024]);  view_1734 = None
        permute_1209 = torch.ops.aten.permute.default(view_1737, [1, 0])
        mm_599 = torch.ops.aten.mm.default(permute_1209, view_173);  permute_1209 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        all_gather_into_tensor_49 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_175, 256, '0');  convert_element_type_175 = None
        wait_tensor_49 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_49);  all_gather_into_tensor_49 = None
        permute_57 = torch.ops.aten.permute.default(wait_tensor_49, [1, 0]);  wait_tensor_49 = None
        permute_1211 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_600 = torch.ops.aten.mm.default(view_1737, permute_1211);  view_1737 = permute_1211 = None
        view_1738 = torch.ops.aten.view.default(mm_600, [2, 8192, 4096]);  mm_600 = None
        convert_element_type_2515 = torch.ops.prims.convert_element_type.default(mm_599, torch.float32);  mm_599 = None
        reduce_scatter_tensor_241 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2515, 'avg', 256, '0');  convert_element_type_2515 = None
        wait_tensor_532 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_241);  reduce_scatter_tensor_241 = None
        view_1739 = torch.ops.aten.view.default(view_1735, [16384, 1024]);  view_1735 = None
        permute_1213 = torch.ops.aten.permute.default(view_1739, [1, 0])
        mm_601 = torch.ops.aten.mm.default(permute_1213, view_173);  permute_1213 = None
        permute_1215 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_602 = torch.ops.aten.mm.default(view_1739, permute_1215);  view_1739 = permute_1215 = None
        view_1740 = torch.ops.aten.view.default(mm_602, [2, 8192, 4096]);  mm_602 = None
        add_315 = torch.ops.aten.add.Tensor(view_1738, view_1740);  view_1738 = view_1740 = None
        convert_element_type_2520 = torch.ops.prims.convert_element_type.default(mm_601, torch.float32);  mm_601 = None
        reduce_scatter_tensor_242 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2520, 'avg', 256, '0');  convert_element_type_2520 = None
        wait_tensor_533 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_242);  reduce_scatter_tensor_242 = None
        view_1741 = torch.ops.aten.view.default(view_1736, [16384, 4096]);  view_1736 = None
        permute_1217 = torch.ops.aten.permute.default(view_1741, [1, 0])
        mm_603 = torch.ops.aten.mm.default(permute_1217, view_173);  permute_1217 = view_173 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        all_gather_into_tensor_47 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_169, 256, '0');  convert_element_type_169 = None
        wait_tensor_47 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_47);  all_gather_into_tensor_47 = None
        permute_55 = torch.ops.aten.permute.default(wait_tensor_47, [1, 0]);  wait_tensor_47 = None
        permute_1219 = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_604 = torch.ops.aten.mm.default(view_1741, permute_1219);  view_1741 = permute_1219 = None
        view_1742 = torch.ops.aten.view.default(mm_604, [2, 8192, 4096]);  mm_604 = None
        add_316 = torch.ops.aten.add.Tensor(add_315, view_1742);  add_315 = view_1742 = None
        convert_element_type_2525 = torch.ops.prims.convert_element_type.default(mm_603, torch.float32);  mm_603 = None
        reduce_scatter_tensor_243 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2525, 'avg', 256, '0');  convert_element_type_2525 = None
        wait_tensor_534 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_243);  reduce_scatter_tensor_243 = None
        convert_element_type_2526 = torch.ops.prims.convert_element_type.default(add_316, torch.float32);  add_316 = None
        convert_element_type_2528 = torch.ops.prims.convert_element_type.default(wait_tensor_46, torch.float32);  wait_tensor_46 = None
        mul_798 = torch.ops.aten.mul.Tensor(convert_element_type_2526, convert_element_type_2528);  convert_element_type_2528 = None
        mul_800 = torch.ops.aten.mul.Tensor(mul_40, mul_798)
        sum_163 = torch.ops.aten.sum.dim_IntList(mul_800, [2], True);  mul_800 = None
        div_54 = torch.ops.aten.div.Tensor(mul_40, 4096)
        mul_801 = torch.ops.aten.mul.Tensor(div_54, sum_163);  div_54 = sum_163 = None
        sub_81 = torch.ops.aten.sub.Tensor(mul_798, mul_801);  mul_798 = mul_801 = None
        mul_802 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_10);  sub_81 = rsqrt_10 = None
        mul_803 = torch.ops.aten.mul.Tensor(convert_element_type_2526, mul_40);  convert_element_type_2526 = mul_40 = None
        sum_164 = torch.ops.aten.sum.dim_IntList(mul_803, [0, 1]);  mul_803 = None
        convert_element_type_2529 = torch.ops.prims.convert_element_type.default(mul_802, torch.bfloat16);  mul_802 = None
        add_317 = torch.ops.aten.add.Tensor(add_314, convert_element_type_2529);  add_314 = convert_element_type_2529 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(sum_164, torch.float32);  sum_164 = None
        reduce_scatter_tensor_244 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_11, 'avg', 256, '0');  convert_element_type_default_11 = None
        wait_tensor_535 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_244);  reduce_scatter_tensor_244 = None
        view_1743 = torch.ops.aten.view.default(add_317, [16384, 4096])
        permute_1221 = torch.ops.aten.permute.default(view_1743, [1, 0])
        permute_50 = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3])
        view_157 = torch.ops.aten.view.default(permute_50, [2, 8192, -1]);  permute_50 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        all_gather_into_tensor_41 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_149, 256, '0');  convert_element_type_149 = None
        wait_tensor_41 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_41);  all_gather_into_tensor_41 = None
        permute_51 = torch.ops.aten.permute.default(wait_tensor_41, [1, 0]);  wait_tensor_41 = None
        view_159 = torch.ops.aten.view.default(view_157, [16384, 4096]);  view_157 = None
        mm_31 = torch.ops.aten.mm.default(view_159, permute_51)
        view_160 = torch.ops.aten.view.default(mm_31, [2, 8192, 4096]);  mm_31 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, view_160);  view_160 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        all_gather_into_tensor_42 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_152, 256, '0');  convert_element_type_152 = None
        wait_tensor_42 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_42);  all_gather_into_tensor_42 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_18 = torch.ops.aten.add.Scalar(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_9);  convert_element_type_153 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, wait_tensor_42)
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        view_163 = torch.ops.aten.view.default(convert_element_type_154, [16384, 4096]);  convert_element_type_154 = None
        view_164 = torch.ops.aten.view.default(mm_32, [2, 8192, 14336]);  mm_32 = None
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_158)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_158, sigmoid_4);  sigmoid_4 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        all_gather_into_tensor_44 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_160, 256, '0');  convert_element_type_160 = None
        wait_tensor_44 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_44);  all_gather_into_tensor_44 = None
        permute_53 = torch.ops.aten.permute.default(wait_tensor_44, [1, 0]);  wait_tensor_44 = None
        mm_33 = torch.ops.aten.mm.default(view_163, permute_53)
        view_167 = torch.ops.aten.view.default(mm_33, [2, 8192, 14336]);  mm_33 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_159, view_167)
        view_169 = torch.ops.aten.view.default(mul_39, [16384, 14336]);  mul_39 = None
        mm_605 = torch.ops.aten.mm.default(permute_1221, view_169);  permute_1221 = view_169 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        all_gather_into_tensor_45 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_163, 256, '0');  convert_element_type_163 = None
        wait_tensor_45 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_45);  all_gather_into_tensor_45 = None
        permute_54 = torch.ops.aten.permute.default(wait_tensor_45, [1, 0]);  wait_tensor_45 = None
        permute_1223 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_606 = torch.ops.aten.mm.default(view_1743, permute_1223);  view_1743 = permute_1223 = None
        view_1744 = torch.ops.aten.view.default(mm_606, [2, 8192, 14336]);  mm_606 = None
        convert_element_type_2536 = torch.ops.prims.convert_element_type.default(mm_605, torch.float32);  mm_605 = None
        reduce_scatter_tensor_245 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2536, 'avg', 256, '0');  convert_element_type_2536 = None
        wait_tensor_536 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_245);  reduce_scatter_tensor_245 = None
        mul_804 = torch.ops.aten.mul.Tensor(view_1744, convert_element_type_159);  convert_element_type_159 = None
        mul_805 = torch.ops.aten.mul.Tensor(view_1744, view_167);  view_1744 = view_167 = None
        view_1745 = torch.ops.aten.view.default(mul_804, [16384, 14336]);  mul_804 = None
        permute_1225 = torch.ops.aten.permute.default(view_1745, [1, 0])
        mm_607 = torch.ops.aten.mm.default(permute_1225, view_163);  permute_1225 = None
        permute_1227 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_608 = torch.ops.aten.mm.default(view_1745, permute_1227);  view_1745 = permute_1227 = None
        view_1746 = torch.ops.aten.view.default(mm_608, [2, 8192, 4096]);  mm_608 = None
        convert_element_type_2541 = torch.ops.prims.convert_element_type.default(mm_607, torch.float32);  mm_607 = None
        reduce_scatter_tensor_246 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2541, 'avg', 256, '0');  convert_element_type_2541 = None
        wait_tensor_537 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_246);  reduce_scatter_tensor_246 = None
        convert_element_type_2542 = torch.ops.prims.convert_element_type.default(mul_805, torch.float32);  mul_805 = None
        neg_27 = torch.ops.aten.neg.default(convert_element_type_158)
        exp_27 = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_318 = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(add_318);  add_318 = None
        mul_806 = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        mul_807 = torch.ops.aten.mul.Tensor(convert_element_type_2542, mul_806);  convert_element_type_2542 = None
        sub_82 = torch.ops.aten.sub.Tensor(1, mul_806);  mul_806 = None
        mul_808 = torch.ops.aten.mul.Tensor(convert_element_type_158, sub_82);  convert_element_type_158 = sub_82 = None
        add_319 = torch.ops.aten.add.Tensor(mul_808, 1);  mul_808 = None
        mul_809 = torch.ops.aten.mul.Tensor(mul_807, add_319);  mul_807 = add_319 = None
        convert_element_type_2544 = torch.ops.prims.convert_element_type.default(mul_809, torch.bfloat16);  mul_809 = None
        view_1747 = torch.ops.aten.view.default(convert_element_type_2544, [16384, 14336]);  convert_element_type_2544 = None
        permute_1229 = torch.ops.aten.permute.default(view_1747, [1, 0])
        mm_609 = torch.ops.aten.mm.default(permute_1229, view_163);  permute_1229 = view_163 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        all_gather_into_tensor_43 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_155, 256, '0');  convert_element_type_155 = None
        wait_tensor_43 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_43);  all_gather_into_tensor_43 = None
        permute_52 = torch.ops.aten.permute.default(wait_tensor_43, [1, 0]);  wait_tensor_43 = None
        permute_1231 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_610 = torch.ops.aten.mm.default(view_1747, permute_1231);  view_1747 = permute_1231 = None
        view_1748 = torch.ops.aten.view.default(mm_610, [2, 8192, 4096]);  mm_610 = None
        add_320 = torch.ops.aten.add.Tensor(view_1746, view_1748);  view_1746 = view_1748 = None
        convert_element_type_2549 = torch.ops.prims.convert_element_type.default(mm_609, torch.float32);  mm_609 = None
        reduce_scatter_tensor_247 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2549, 'avg', 256, '0');  convert_element_type_2549 = None
        wait_tensor_538 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_247);  reduce_scatter_tensor_247 = None
        convert_element_type_2550 = torch.ops.prims.convert_element_type.default(add_320, torch.float32);  add_320 = None
        convert_element_type_2552 = torch.ops.prims.convert_element_type.default(wait_tensor_42, torch.float32);  wait_tensor_42 = None
        mul_810 = torch.ops.aten.mul.Tensor(convert_element_type_2550, convert_element_type_2552);  convert_element_type_2552 = None
        mul_812 = torch.ops.aten.mul.Tensor(mul_36, mul_810)
        sum_165 = torch.ops.aten.sum.dim_IntList(mul_812, [2], True);  mul_812 = None
        div_55 = torch.ops.aten.div.Tensor(mul_36, 4096)
        mul_813 = torch.ops.aten.mul.Tensor(div_55, sum_165);  div_55 = sum_165 = None
        sub_83 = torch.ops.aten.sub.Tensor(mul_810, mul_813);  mul_810 = mul_813 = None
        mul_814 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_9);  sub_83 = rsqrt_9 = None
        mul_815 = torch.ops.aten.mul.Tensor(convert_element_type_2550, mul_36);  convert_element_type_2550 = mul_36 = None
        sum_166 = torch.ops.aten.sum.dim_IntList(mul_815, [0, 1]);  mul_815 = None
        convert_element_type_2553 = torch.ops.prims.convert_element_type.default(mul_814, torch.bfloat16);  mul_814 = None
        add_321 = torch.ops.aten.add.Tensor(add_317, convert_element_type_2553);  add_317 = convert_element_type_2553 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(sum_166, torch.float32);  sum_166 = None
        reduce_scatter_tensor_248 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_10, 'avg', 256, '0');  convert_element_type_default_10 = None
        wait_tensor_539 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_248);  reduce_scatter_tensor_248 = None
        view_1749 = torch.ops.aten.view.default(add_321, [16384, 4096])
        permute_1233 = torch.ops.aten.permute.default(view_1749, [1, 0])
        mm_611 = torch.ops.aten.mm.default(permute_1233, view_159);  permute_1233 = view_159 = None
        permute_1235 = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        mm_612 = torch.ops.aten.mm.default(view_1749, permute_1235);  view_1749 = permute_1235 = None
        view_1750 = torch.ops.aten.view.default(mm_612, [2, 8192, 4096]);  mm_612 = None
        convert_element_type_2560 = torch.ops.prims.convert_element_type.default(mm_611, torch.float32);  mm_611 = None
        reduce_scatter_tensor_249 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2560, 'avg', 256, '0');  convert_element_type_2560 = None
        wait_tensor_540 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_249);  reduce_scatter_tensor_249 = None
        view_1751 = torch.ops.aten.view.default(view_1750, [2, 8192, 32, 128]);  view_1750 = None
        permute_1237 = torch.ops.aten.permute.default(view_1751, [0, 2, 1, 3]);  view_1751 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        all_gather_into_tensor_37 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_133, 256, '0');  convert_element_type_133 = None
        wait_tensor_37 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_37);  all_gather_into_tensor_37 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_16 = torch.ops.aten.add.Scalar(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_8);  convert_element_type_134 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, wait_tensor_37)
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        view_139 = torch.ops.aten.view.default(convert_element_type_135, [16384, 4096]);  convert_element_type_135 = None
        view_140 = torch.ops.aten.view.default(mm_28, [2, 8192, 4096]);  mm_28 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        all_gather_into_tensor_39 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_139, 256, '0');  convert_element_type_139 = None
        wait_tensor_39 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_39);  all_gather_into_tensor_39 = None
        permute_45 = torch.ops.aten.permute.default(wait_tensor_39, [1, 0]);  wait_tensor_39 = None
        mm_29 = torch.ops.aten.mm.default(view_139, permute_45)
        view_143 = torch.ops.aten.view.default(mm_29, [2, 8192, 1024]);  mm_29 = None
        view_146 = torch.ops.aten.view.default(mm_30, [2, 8192, 1024]);  mm_30 = None
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
        _scaled_dot_product_cudnn_attention_backward_27 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1237, permute_47, permute_48, permute_49, getitem_36, getitem_37, getitem_42, getitem_43, None, None, None, 8192, 8192, 0.0, True);  permute_1237 = permute_47 = permute_48 = permute_49 = getitem_36 = getitem_37 = getitem_42 = getitem_43 = None
        getitem_369 = _scaled_dot_product_cudnn_attention_backward_27[0]
        getitem_370 = _scaled_dot_product_cudnn_attention_backward_27[1]
        getitem_371 = _scaled_dot_product_cudnn_attention_backward_27[2];  _scaled_dot_product_cudnn_attention_backward_27 = None
        permute_1238 = torch.ops.aten.permute.default(getitem_371, [0, 2, 1, 3]);  getitem_371 = None
        permute_1239 = torch.ops.aten.permute.default(getitem_370, [0, 2, 1, 3]);  getitem_370 = None
        permute_1240 = torch.ops.aten.permute.default(getitem_369, [0, 2, 1, 3]);  getitem_369 = None
        view_1752 = torch.ops.aten.view.default(permute_1238, [2, 8192, 8, 4, 128]);  permute_1238 = None
        sum_167 = torch.ops.aten.sum.dim_IntList(view_1752, [3], True);  view_1752 = None
        squeeze_54 = torch.ops.aten.squeeze.dim(sum_167, 3);  sum_167 = None
        view_1753 = torch.ops.aten.view.default(permute_1239, [2, 8192, 8, 4, 128]);  permute_1239 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(view_1753, [3], True);  view_1753 = None
        squeeze_55 = torch.ops.aten.squeeze.dim(sum_168, 3);  sum_168 = None
        convert_element_type_2561 = torch.ops.prims.convert_element_type.default(squeeze_55, torch.float32);  squeeze_55 = None
        convert_element_type_2562 = torch.ops.prims.convert_element_type.default(permute_1240, torch.float32);  permute_1240 = None
        view_1754 = torch.ops.aten.view.default(convert_element_type_2561, [2, 8192, 8, 64, 2]);  convert_element_type_2561 = None
        view_as_complex_118 = torch.ops.aten.view_as_complex.default(view_1754);  view_1754 = None
        mul_816 = torch.ops.aten.mul.Tensor(view_as_complex_118, _conj);  view_as_complex_118 = None
        view_1755 = torch.ops.aten.view.default(convert_element_type_2562, [2, 8192, 32, 64, 2]);  convert_element_type_2562 = None
        view_as_complex_119 = torch.ops.aten.view_as_complex.default(view_1755);  view_1755 = None
        mul_817 = torch.ops.aten.mul.Tensor(view_as_complex_119, _conj);  view_as_complex_119 = None
        view_as_real_118 = torch.ops.aten.view_as_real.default(mul_816);  mul_816 = None
        view_1756 = torch.ops.aten.view.default(view_as_real_118, [2, 8192, 8, 128]);  view_as_real_118 = None
        convert_element_type_2563 = torch.ops.prims.convert_element_type.default(view_1756, torch.bfloat16);  view_1756 = None
        view_as_real_119 = torch.ops.aten.view_as_real.default(mul_817);  mul_817 = None
        view_1757 = torch.ops.aten.view.default(view_as_real_119, [2, 8192, 32, 128]);  view_as_real_119 = None
        convert_element_type_2564 = torch.ops.prims.convert_element_type.default(view_1757, torch.bfloat16);  view_1757 = None
        view_1758 = torch.ops.aten.view.default(squeeze_54, [2, 8192, 1024]);  squeeze_54 = None
        view_1759 = torch.ops.aten.view.default(convert_element_type_2563, [2, 8192, 1024]);  convert_element_type_2563 = None
        view_1760 = torch.ops.aten.view.default(convert_element_type_2564, [2, 8192, 4096]);  convert_element_type_2564 = None
        view_1761 = torch.ops.aten.view.default(view_1758, [16384, 1024]);  view_1758 = None
        permute_1241 = torch.ops.aten.permute.default(view_1761, [1, 0])
        mm_613 = torch.ops.aten.mm.default(permute_1241, view_139);  permute_1241 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        all_gather_into_tensor_40 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_142, 256, '0');  convert_element_type_142 = None
        wait_tensor_40 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_40);  all_gather_into_tensor_40 = None
        permute_46 = torch.ops.aten.permute.default(wait_tensor_40, [1, 0]);  wait_tensor_40 = None
        permute_1243 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_614 = torch.ops.aten.mm.default(view_1761, permute_1243);  view_1761 = permute_1243 = None
        view_1762 = torch.ops.aten.view.default(mm_614, [2, 8192, 4096]);  mm_614 = None
        convert_element_type_2569 = torch.ops.prims.convert_element_type.default(mm_613, torch.float32);  mm_613 = None
        reduce_scatter_tensor_250 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2569, 'avg', 256, '0');  convert_element_type_2569 = None
        wait_tensor_541 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_250);  reduce_scatter_tensor_250 = None
        view_1763 = torch.ops.aten.view.default(view_1759, [16384, 1024]);  view_1759 = None
        permute_1245 = torch.ops.aten.permute.default(view_1763, [1, 0])
        mm_615 = torch.ops.aten.mm.default(permute_1245, view_139);  permute_1245 = None
        permute_1247 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_616 = torch.ops.aten.mm.default(view_1763, permute_1247);  view_1763 = permute_1247 = None
        view_1764 = torch.ops.aten.view.default(mm_616, [2, 8192, 4096]);  mm_616 = None
        add_322 = torch.ops.aten.add.Tensor(view_1762, view_1764);  view_1762 = view_1764 = None
        convert_element_type_2574 = torch.ops.prims.convert_element_type.default(mm_615, torch.float32);  mm_615 = None
        reduce_scatter_tensor_251 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2574, 'avg', 256, '0');  convert_element_type_2574 = None
        wait_tensor_542 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_251);  reduce_scatter_tensor_251 = None
        view_1765 = torch.ops.aten.view.default(view_1760, [16384, 4096]);  view_1760 = None
        permute_1249 = torch.ops.aten.permute.default(view_1765, [1, 0])
        mm_617 = torch.ops.aten.mm.default(permute_1249, view_139);  permute_1249 = view_139 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        all_gather_into_tensor_38 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_136, 256, '0');  convert_element_type_136 = None
        wait_tensor_38 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_38);  all_gather_into_tensor_38 = None
        permute_44 = torch.ops.aten.permute.default(wait_tensor_38, [1, 0]);  wait_tensor_38 = None
        permute_1251 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_618 = torch.ops.aten.mm.default(view_1765, permute_1251);  view_1765 = permute_1251 = None
        view_1766 = torch.ops.aten.view.default(mm_618, [2, 8192, 4096]);  mm_618 = None
        add_323 = torch.ops.aten.add.Tensor(add_322, view_1766);  add_322 = view_1766 = None
        convert_element_type_2579 = torch.ops.prims.convert_element_type.default(mm_617, torch.float32);  mm_617 = None
        reduce_scatter_tensor_252 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2579, 'avg', 256, '0');  convert_element_type_2579 = None
        wait_tensor_543 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_252);  reduce_scatter_tensor_252 = None
        convert_element_type_2580 = torch.ops.prims.convert_element_type.default(add_323, torch.float32);  add_323 = None
        convert_element_type_2582 = torch.ops.prims.convert_element_type.default(wait_tensor_37, torch.float32);  wait_tensor_37 = None
        mul_818 = torch.ops.aten.mul.Tensor(convert_element_type_2580, convert_element_type_2582);  convert_element_type_2582 = None
        mul_820 = torch.ops.aten.mul.Tensor(mul_32, mul_818)
        sum_169 = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
        div_56 = torch.ops.aten.div.Tensor(mul_32, 4096)
        mul_821 = torch.ops.aten.mul.Tensor(div_56, sum_169);  div_56 = sum_169 = None
        sub_84 = torch.ops.aten.sub.Tensor(mul_818, mul_821);  mul_818 = mul_821 = None
        mul_822 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_8);  sub_84 = rsqrt_8 = None
        mul_823 = torch.ops.aten.mul.Tensor(convert_element_type_2580, mul_32);  convert_element_type_2580 = mul_32 = None
        sum_170 = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
        convert_element_type_2583 = torch.ops.prims.convert_element_type.default(mul_822, torch.bfloat16);  mul_822 = None
        add_324 = torch.ops.aten.add.Tensor(add_321, convert_element_type_2583);  add_321 = convert_element_type_2583 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(sum_170, torch.float32);  sum_170 = None
        reduce_scatter_tensor_253 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_9, 'avg', 256, '0');  convert_element_type_default_9 = None
        wait_tensor_544 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_253);  reduce_scatter_tensor_253 = None
        view_1767 = torch.ops.aten.view.default(add_324, [16384, 4096])
        permute_1253 = torch.ops.aten.permute.default(view_1767, [1, 0])
        permute_39 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_123 = torch.ops.aten.view.default(permute_39, [2, 8192, -1]);  permute_39 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        all_gather_into_tensor_32 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_116, 256, '0');  convert_element_type_116 = None
        wait_tensor_32 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_32);  all_gather_into_tensor_32 = None
        permute_40 = torch.ops.aten.permute.default(wait_tensor_32, [1, 0]);  wait_tensor_32 = None
        view_125 = torch.ops.aten.view.default(view_123, [16384, 4096]);  view_123 = None
        mm_24 = torch.ops.aten.mm.default(view_125, permute_40)
        view_126 = torch.ops.aten.view.default(mm_24, [2, 8192, 4096]);  mm_24 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, view_126);  view_126 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        all_gather_into_tensor_33 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_119, 256, '0');  convert_element_type_119 = None
        wait_tensor_33 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_33);  all_gather_into_tensor_33 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_14 = torch.ops.aten.add.Scalar(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_7);  convert_element_type_120 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, wait_tensor_33)
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_129 = torch.ops.aten.view.default(convert_element_type_121, [16384, 4096]);  convert_element_type_121 = None
        view_130 = torch.ops.aten.view.default(mm_25, [2, 8192, 14336]);  mm_25 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_125, sigmoid_3);  sigmoid_3 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        all_gather_into_tensor_35 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_127, 256, '0');  convert_element_type_127 = None
        wait_tensor_35 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_35);  all_gather_into_tensor_35 = None
        permute_42 = torch.ops.aten.permute.default(wait_tensor_35, [1, 0]);  wait_tensor_35 = None
        mm_26 = torch.ops.aten.mm.default(view_129, permute_42)
        view_133 = torch.ops.aten.view.default(mm_26, [2, 8192, 14336]);  mm_26 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_126, view_133)
        view_135 = torch.ops.aten.view.default(mul_31, [16384, 14336]);  mul_31 = None
        mm_619 = torch.ops.aten.mm.default(permute_1253, view_135);  permute_1253 = view_135 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        all_gather_into_tensor_36 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_130, 256, '0');  convert_element_type_130 = None
        wait_tensor_36 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_36);  all_gather_into_tensor_36 = None
        permute_43 = torch.ops.aten.permute.default(wait_tensor_36, [1, 0]);  wait_tensor_36 = None
        permute_1255 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_620 = torch.ops.aten.mm.default(view_1767, permute_1255);  view_1767 = permute_1255 = None
        view_1768 = torch.ops.aten.view.default(mm_620, [2, 8192, 14336]);  mm_620 = None
        convert_element_type_2590 = torch.ops.prims.convert_element_type.default(mm_619, torch.float32);  mm_619 = None
        reduce_scatter_tensor_254 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2590, 'avg', 256, '0');  convert_element_type_2590 = None
        wait_tensor_545 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_254);  reduce_scatter_tensor_254 = None
        mul_824 = torch.ops.aten.mul.Tensor(view_1768, convert_element_type_126);  convert_element_type_126 = None
        mul_825 = torch.ops.aten.mul.Tensor(view_1768, view_133);  view_1768 = view_133 = None
        view_1769 = torch.ops.aten.view.default(mul_824, [16384, 14336]);  mul_824 = None
        permute_1257 = torch.ops.aten.permute.default(view_1769, [1, 0])
        mm_621 = torch.ops.aten.mm.default(permute_1257, view_129);  permute_1257 = None
        permute_1259 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_622 = torch.ops.aten.mm.default(view_1769, permute_1259);  view_1769 = permute_1259 = None
        view_1770 = torch.ops.aten.view.default(mm_622, [2, 8192, 4096]);  mm_622 = None
        convert_element_type_2595 = torch.ops.prims.convert_element_type.default(mm_621, torch.float32);  mm_621 = None
        reduce_scatter_tensor_255 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2595, 'avg', 256, '0');  convert_element_type_2595 = None
        wait_tensor_546 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_255);  reduce_scatter_tensor_255 = None
        convert_element_type_2596 = torch.ops.prims.convert_element_type.default(mul_825, torch.float32);  mul_825 = None
        neg_28 = torch.ops.aten.neg.default(convert_element_type_125)
        exp_28 = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_325 = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(add_325);  add_325 = None
        mul_826 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        mul_827 = torch.ops.aten.mul.Tensor(convert_element_type_2596, mul_826);  convert_element_type_2596 = None
        sub_85 = torch.ops.aten.sub.Tensor(1, mul_826);  mul_826 = None
        mul_828 = torch.ops.aten.mul.Tensor(convert_element_type_125, sub_85);  convert_element_type_125 = sub_85 = None
        add_326 = torch.ops.aten.add.Tensor(mul_828, 1);  mul_828 = None
        mul_829 = torch.ops.aten.mul.Tensor(mul_827, add_326);  mul_827 = add_326 = None
        convert_element_type_2598 = torch.ops.prims.convert_element_type.default(mul_829, torch.bfloat16);  mul_829 = None
        view_1771 = torch.ops.aten.view.default(convert_element_type_2598, [16384, 14336]);  convert_element_type_2598 = None
        permute_1261 = torch.ops.aten.permute.default(view_1771, [1, 0])
        mm_623 = torch.ops.aten.mm.default(permute_1261, view_129);  permute_1261 = view_129 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        all_gather_into_tensor_34 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_122, 256, '0');  convert_element_type_122 = None
        wait_tensor_34 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_34);  all_gather_into_tensor_34 = None
        permute_41 = torch.ops.aten.permute.default(wait_tensor_34, [1, 0]);  wait_tensor_34 = None
        permute_1263 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_624 = torch.ops.aten.mm.default(view_1771, permute_1263);  view_1771 = permute_1263 = None
        view_1772 = torch.ops.aten.view.default(mm_624, [2, 8192, 4096]);  mm_624 = None
        add_327 = torch.ops.aten.add.Tensor(view_1770, view_1772);  view_1770 = view_1772 = None
        convert_element_type_2603 = torch.ops.prims.convert_element_type.default(mm_623, torch.float32);  mm_623 = None
        reduce_scatter_tensor_256 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2603, 'avg', 256, '0');  convert_element_type_2603 = None
        wait_tensor_547 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_256);  reduce_scatter_tensor_256 = None
        convert_element_type_2604 = torch.ops.prims.convert_element_type.default(add_327, torch.float32);  add_327 = None
        convert_element_type_2606 = torch.ops.prims.convert_element_type.default(wait_tensor_33, torch.float32);  wait_tensor_33 = None
        mul_830 = torch.ops.aten.mul.Tensor(convert_element_type_2604, convert_element_type_2606);  convert_element_type_2606 = None
        mul_832 = torch.ops.aten.mul.Tensor(mul_28, mul_830)
        sum_171 = torch.ops.aten.sum.dim_IntList(mul_832, [2], True);  mul_832 = None
        div_57 = torch.ops.aten.div.Tensor(mul_28, 4096)
        mul_833 = torch.ops.aten.mul.Tensor(div_57, sum_171);  div_57 = sum_171 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_830, mul_833);  mul_830 = mul_833 = None
        mul_834 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_7);  sub_86 = rsqrt_7 = None
        mul_835 = torch.ops.aten.mul.Tensor(convert_element_type_2604, mul_28);  convert_element_type_2604 = mul_28 = None
        sum_172 = torch.ops.aten.sum.dim_IntList(mul_835, [0, 1]);  mul_835 = None
        convert_element_type_2607 = torch.ops.prims.convert_element_type.default(mul_834, torch.bfloat16);  mul_834 = None
        add_328 = torch.ops.aten.add.Tensor(add_324, convert_element_type_2607);  add_324 = convert_element_type_2607 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(sum_172, torch.float32);  sum_172 = None
        reduce_scatter_tensor_257 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_8, 'avg', 256, '0');  convert_element_type_default_8 = None
        wait_tensor_548 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_257);  reduce_scatter_tensor_257 = None
        view_1773 = torch.ops.aten.view.default(add_328, [16384, 4096])
        permute_1265 = torch.ops.aten.permute.default(view_1773, [1, 0])
        mm_625 = torch.ops.aten.mm.default(permute_1265, view_125);  permute_1265 = view_125 = None
        permute_1267 = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_626 = torch.ops.aten.mm.default(view_1773, permute_1267);  view_1773 = permute_1267 = None
        view_1774 = torch.ops.aten.view.default(mm_626, [2, 8192, 4096]);  mm_626 = None
        convert_element_type_2614 = torch.ops.prims.convert_element_type.default(mm_625, torch.float32);  mm_625 = None
        reduce_scatter_tensor_258 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2614, 'avg', 256, '0');  convert_element_type_2614 = None
        wait_tensor_549 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_258);  reduce_scatter_tensor_258 = None
        view_1775 = torch.ops.aten.view.default(view_1774, [2, 8192, 32, 128]);  view_1774 = None
        permute_1269 = torch.ops.aten.permute.default(view_1775, [0, 2, 1, 3]);  view_1775 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        all_gather_into_tensor_28 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_100, 256, '0');  convert_element_type_100 = None
        wait_tensor_28 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_28);  all_gather_into_tensor_28 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_12 = torch.ops.aten.add.Scalar(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_6);  convert_element_type_101 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, wait_tensor_28)
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        view_105 = torch.ops.aten.view.default(convert_element_type_102, [16384, 4096]);  convert_element_type_102 = None
        view_106 = torch.ops.aten.view.default(mm_21, [2, 8192, 4096]);  mm_21 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        all_gather_into_tensor_30 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_106, 256, '0');  convert_element_type_106 = None
        wait_tensor_30 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_30);  all_gather_into_tensor_30 = None
        permute_34 = torch.ops.aten.permute.default(wait_tensor_30, [1, 0]);  wait_tensor_30 = None
        mm_22 = torch.ops.aten.mm.default(view_105, permute_34)
        view_109 = torch.ops.aten.view.default(mm_22, [2, 8192, 1024]);  mm_22 = None
        view_112 = torch.ops.aten.view.default(mm_23, [2, 8192, 1024]);  mm_23 = None
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
        _scaled_dot_product_cudnn_attention_backward_28 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1269, permute_36, permute_37, permute_38, getitem_27, getitem_28, getitem_33, getitem_34, None, None, None, 8192, 8192, 0.0, True);  permute_1269 = permute_36 = permute_37 = permute_38 = getitem_27 = getitem_28 = getitem_33 = getitem_34 = None
        getitem_372 = _scaled_dot_product_cudnn_attention_backward_28[0]
        getitem_373 = _scaled_dot_product_cudnn_attention_backward_28[1]
        getitem_374 = _scaled_dot_product_cudnn_attention_backward_28[2];  _scaled_dot_product_cudnn_attention_backward_28 = None
        permute_1270 = torch.ops.aten.permute.default(getitem_374, [0, 2, 1, 3]);  getitem_374 = None
        permute_1271 = torch.ops.aten.permute.default(getitem_373, [0, 2, 1, 3]);  getitem_373 = None
        permute_1272 = torch.ops.aten.permute.default(getitem_372, [0, 2, 1, 3]);  getitem_372 = None
        view_1776 = torch.ops.aten.view.default(permute_1270, [2, 8192, 8, 4, 128]);  permute_1270 = None
        sum_173 = torch.ops.aten.sum.dim_IntList(view_1776, [3], True);  view_1776 = None
        squeeze_56 = torch.ops.aten.squeeze.dim(sum_173, 3);  sum_173 = None
        view_1777 = torch.ops.aten.view.default(permute_1271, [2, 8192, 8, 4, 128]);  permute_1271 = None
        sum_174 = torch.ops.aten.sum.dim_IntList(view_1777, [3], True);  view_1777 = None
        squeeze_57 = torch.ops.aten.squeeze.dim(sum_174, 3);  sum_174 = None
        convert_element_type_2615 = torch.ops.prims.convert_element_type.default(squeeze_57, torch.float32);  squeeze_57 = None
        convert_element_type_2616 = torch.ops.prims.convert_element_type.default(permute_1272, torch.float32);  permute_1272 = None
        view_1778 = torch.ops.aten.view.default(convert_element_type_2615, [2, 8192, 8, 64, 2]);  convert_element_type_2615 = None
        view_as_complex_120 = torch.ops.aten.view_as_complex.default(view_1778);  view_1778 = None
        mul_836 = torch.ops.aten.mul.Tensor(view_as_complex_120, _conj);  view_as_complex_120 = None
        view_1779 = torch.ops.aten.view.default(convert_element_type_2616, [2, 8192, 32, 64, 2]);  convert_element_type_2616 = None
        view_as_complex_121 = torch.ops.aten.view_as_complex.default(view_1779);  view_1779 = None
        mul_837 = torch.ops.aten.mul.Tensor(view_as_complex_121, _conj);  view_as_complex_121 = None
        view_as_real_120 = torch.ops.aten.view_as_real.default(mul_836);  mul_836 = None
        view_1780 = torch.ops.aten.view.default(view_as_real_120, [2, 8192, 8, 128]);  view_as_real_120 = None
        convert_element_type_2617 = torch.ops.prims.convert_element_type.default(view_1780, torch.bfloat16);  view_1780 = None
        view_as_real_121 = torch.ops.aten.view_as_real.default(mul_837);  mul_837 = None
        view_1781 = torch.ops.aten.view.default(view_as_real_121, [2, 8192, 32, 128]);  view_as_real_121 = None
        convert_element_type_2618 = torch.ops.prims.convert_element_type.default(view_1781, torch.bfloat16);  view_1781 = None
        view_1782 = torch.ops.aten.view.default(squeeze_56, [2, 8192, 1024]);  squeeze_56 = None
        view_1783 = torch.ops.aten.view.default(convert_element_type_2617, [2, 8192, 1024]);  convert_element_type_2617 = None
        view_1784 = torch.ops.aten.view.default(convert_element_type_2618, [2, 8192, 4096]);  convert_element_type_2618 = None
        view_1785 = torch.ops.aten.view.default(view_1782, [16384, 1024]);  view_1782 = None
        permute_1273 = torch.ops.aten.permute.default(view_1785, [1, 0])
        mm_627 = torch.ops.aten.mm.default(permute_1273, view_105);  permute_1273 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        all_gather_into_tensor_31 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_109, 256, '0');  convert_element_type_109 = None
        wait_tensor_31 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_31);  all_gather_into_tensor_31 = None
        permute_35 = torch.ops.aten.permute.default(wait_tensor_31, [1, 0]);  wait_tensor_31 = None
        permute_1275 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_628 = torch.ops.aten.mm.default(view_1785, permute_1275);  view_1785 = permute_1275 = None
        view_1786 = torch.ops.aten.view.default(mm_628, [2, 8192, 4096]);  mm_628 = None
        convert_element_type_2623 = torch.ops.prims.convert_element_type.default(mm_627, torch.float32);  mm_627 = None
        reduce_scatter_tensor_259 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2623, 'avg', 256, '0');  convert_element_type_2623 = None
        wait_tensor_550 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_259);  reduce_scatter_tensor_259 = None
        view_1787 = torch.ops.aten.view.default(view_1783, [16384, 1024]);  view_1783 = None
        permute_1277 = torch.ops.aten.permute.default(view_1787, [1, 0])
        mm_629 = torch.ops.aten.mm.default(permute_1277, view_105);  permute_1277 = None
        permute_1279 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_630 = torch.ops.aten.mm.default(view_1787, permute_1279);  view_1787 = permute_1279 = None
        view_1788 = torch.ops.aten.view.default(mm_630, [2, 8192, 4096]);  mm_630 = None
        add_329 = torch.ops.aten.add.Tensor(view_1786, view_1788);  view_1786 = view_1788 = None
        convert_element_type_2628 = torch.ops.prims.convert_element_type.default(mm_629, torch.float32);  mm_629 = None
        reduce_scatter_tensor_260 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2628, 'avg', 256, '0');  convert_element_type_2628 = None
        wait_tensor_551 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_260);  reduce_scatter_tensor_260 = None
        view_1789 = torch.ops.aten.view.default(view_1784, [16384, 4096]);  view_1784 = None
        permute_1281 = torch.ops.aten.permute.default(view_1789, [1, 0])
        mm_631 = torch.ops.aten.mm.default(permute_1281, view_105);  permute_1281 = view_105 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        all_gather_into_tensor_29 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_103, 256, '0');  convert_element_type_103 = None
        wait_tensor_29 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_29);  all_gather_into_tensor_29 = None
        permute_33 = torch.ops.aten.permute.default(wait_tensor_29, [1, 0]);  wait_tensor_29 = None
        permute_1283 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_632 = torch.ops.aten.mm.default(view_1789, permute_1283);  view_1789 = permute_1283 = None
        view_1790 = torch.ops.aten.view.default(mm_632, [2, 8192, 4096]);  mm_632 = None
        add_330 = torch.ops.aten.add.Tensor(add_329, view_1790);  add_329 = view_1790 = None
        convert_element_type_2633 = torch.ops.prims.convert_element_type.default(mm_631, torch.float32);  mm_631 = None
        reduce_scatter_tensor_261 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2633, 'avg', 256, '0');  convert_element_type_2633 = None
        wait_tensor_552 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_261);  reduce_scatter_tensor_261 = None
        convert_element_type_2634 = torch.ops.prims.convert_element_type.default(add_330, torch.float32);  add_330 = None
        convert_element_type_2636 = torch.ops.prims.convert_element_type.default(wait_tensor_28, torch.float32);  wait_tensor_28 = None
        mul_838 = torch.ops.aten.mul.Tensor(convert_element_type_2634, convert_element_type_2636);  convert_element_type_2636 = None
        mul_840 = torch.ops.aten.mul.Tensor(mul_24, mul_838)
        sum_175 = torch.ops.aten.sum.dim_IntList(mul_840, [2], True);  mul_840 = None
        div_58 = torch.ops.aten.div.Tensor(mul_24, 4096)
        mul_841 = torch.ops.aten.mul.Tensor(div_58, sum_175);  div_58 = sum_175 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_838, mul_841);  mul_838 = mul_841 = None
        mul_842 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_6);  sub_87 = rsqrt_6 = None
        mul_843 = torch.ops.aten.mul.Tensor(convert_element_type_2634, mul_24);  convert_element_type_2634 = mul_24 = None
        sum_176 = torch.ops.aten.sum.dim_IntList(mul_843, [0, 1]);  mul_843 = None
        convert_element_type_2637 = torch.ops.prims.convert_element_type.default(mul_842, torch.bfloat16);  mul_842 = None
        add_331 = torch.ops.aten.add.Tensor(add_328, convert_element_type_2637);  add_328 = convert_element_type_2637 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(sum_176, torch.float32);  sum_176 = None
        reduce_scatter_tensor_262 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_7, 'avg', 256, '0');  convert_element_type_default_7 = None
        wait_tensor_553 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_262);  reduce_scatter_tensor_262 = None
        view_1791 = torch.ops.aten.view.default(add_331, [16384, 4096])
        permute_1285 = torch.ops.aten.permute.default(view_1791, [1, 0])
        permute_28 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])
        view_89 = torch.ops.aten.view.default(permute_28, [2, 8192, -1]);  permute_28 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        all_gather_into_tensor_23 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_83, 256, '0');  convert_element_type_83 = None
        wait_tensor_23 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_23);  all_gather_into_tensor_23 = None
        permute_29 = torch.ops.aten.permute.default(wait_tensor_23, [1, 0]);  wait_tensor_23 = None
        view_91 = torch.ops.aten.view.default(view_89, [16384, 4096]);  view_89 = None
        mm_17 = torch.ops.aten.mm.default(view_91, permute_29)
        view_92 = torch.ops.aten.view.default(mm_17, [2, 8192, 4096]);  mm_17 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, view_92);  view_92 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        all_gather_into_tensor_24 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_86, 256, '0');  convert_element_type_86 = None
        wait_tensor_24 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_24);  all_gather_into_tensor_24 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_10 = torch.ops.aten.add.Scalar(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_5);  convert_element_type_87 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, wait_tensor_24)
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_95 = torch.ops.aten.view.default(convert_element_type_88, [16384, 4096]);  convert_element_type_88 = None
        view_96 = torch.ops.aten.view.default(mm_18, [2, 8192, 14336]);  mm_18 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_92)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_92, sigmoid_2);  sigmoid_2 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        all_gather_into_tensor_26 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_94, 256, '0');  convert_element_type_94 = None
        wait_tensor_26 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_26);  all_gather_into_tensor_26 = None
        permute_31 = torch.ops.aten.permute.default(wait_tensor_26, [1, 0]);  wait_tensor_26 = None
        mm_19 = torch.ops.aten.mm.default(view_95, permute_31)
        view_99 = torch.ops.aten.view.default(mm_19, [2, 8192, 14336]);  mm_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_93, view_99)
        view_101 = torch.ops.aten.view.default(mul_23, [16384, 14336]);  mul_23 = None
        mm_633 = torch.ops.aten.mm.default(permute_1285, view_101);  permute_1285 = view_101 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        all_gather_into_tensor_27 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_97, 256, '0');  convert_element_type_97 = None
        wait_tensor_27 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_27);  all_gather_into_tensor_27 = None
        permute_32 = torch.ops.aten.permute.default(wait_tensor_27, [1, 0]);  wait_tensor_27 = None
        permute_1287 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_634 = torch.ops.aten.mm.default(view_1791, permute_1287);  view_1791 = permute_1287 = None
        view_1792 = torch.ops.aten.view.default(mm_634, [2, 8192, 14336]);  mm_634 = None
        convert_element_type_2644 = torch.ops.prims.convert_element_type.default(mm_633, torch.float32);  mm_633 = None
        reduce_scatter_tensor_263 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2644, 'avg', 256, '0');  convert_element_type_2644 = None
        wait_tensor_554 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_263);  reduce_scatter_tensor_263 = None
        mul_844 = torch.ops.aten.mul.Tensor(view_1792, convert_element_type_93);  convert_element_type_93 = None
        mul_845 = torch.ops.aten.mul.Tensor(view_1792, view_99);  view_1792 = view_99 = None
        view_1793 = torch.ops.aten.view.default(mul_844, [16384, 14336]);  mul_844 = None
        permute_1289 = torch.ops.aten.permute.default(view_1793, [1, 0])
        mm_635 = torch.ops.aten.mm.default(permute_1289, view_95);  permute_1289 = None
        permute_1291 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_636 = torch.ops.aten.mm.default(view_1793, permute_1291);  view_1793 = permute_1291 = None
        view_1794 = torch.ops.aten.view.default(mm_636, [2, 8192, 4096]);  mm_636 = None
        convert_element_type_2649 = torch.ops.prims.convert_element_type.default(mm_635, torch.float32);  mm_635 = None
        reduce_scatter_tensor_264 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2649, 'avg', 256, '0');  convert_element_type_2649 = None
        wait_tensor_555 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_264);  reduce_scatter_tensor_264 = None
        convert_element_type_2650 = torch.ops.prims.convert_element_type.default(mul_845, torch.float32);  mul_845 = None
        neg_29 = torch.ops.aten.neg.default(convert_element_type_92)
        exp_29 = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_332 = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(add_332);  add_332 = None
        mul_846 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        mul_847 = torch.ops.aten.mul.Tensor(convert_element_type_2650, mul_846);  convert_element_type_2650 = None
        sub_88 = torch.ops.aten.sub.Tensor(1, mul_846);  mul_846 = None
        mul_848 = torch.ops.aten.mul.Tensor(convert_element_type_92, sub_88);  convert_element_type_92 = sub_88 = None
        add_333 = torch.ops.aten.add.Tensor(mul_848, 1);  mul_848 = None
        mul_849 = torch.ops.aten.mul.Tensor(mul_847, add_333);  mul_847 = add_333 = None
        convert_element_type_2652 = torch.ops.prims.convert_element_type.default(mul_849, torch.bfloat16);  mul_849 = None
        view_1795 = torch.ops.aten.view.default(convert_element_type_2652, [16384, 14336]);  convert_element_type_2652 = None
        permute_1293 = torch.ops.aten.permute.default(view_1795, [1, 0])
        mm_637 = torch.ops.aten.mm.default(permute_1293, view_95);  permute_1293 = view_95 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        all_gather_into_tensor_25 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_89, 256, '0');  convert_element_type_89 = None
        wait_tensor_25 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_25);  all_gather_into_tensor_25 = None
        permute_30 = torch.ops.aten.permute.default(wait_tensor_25, [1, 0]);  wait_tensor_25 = None
        permute_1295 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_638 = torch.ops.aten.mm.default(view_1795, permute_1295);  view_1795 = permute_1295 = None
        view_1796 = torch.ops.aten.view.default(mm_638, [2, 8192, 4096]);  mm_638 = None
        add_334 = torch.ops.aten.add.Tensor(view_1794, view_1796);  view_1794 = view_1796 = None
        convert_element_type_2657 = torch.ops.prims.convert_element_type.default(mm_637, torch.float32);  mm_637 = None
        reduce_scatter_tensor_265 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2657, 'avg', 256, '0');  convert_element_type_2657 = None
        wait_tensor_556 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_265);  reduce_scatter_tensor_265 = None
        convert_element_type_2658 = torch.ops.prims.convert_element_type.default(add_334, torch.float32);  add_334 = None
        convert_element_type_2660 = torch.ops.prims.convert_element_type.default(wait_tensor_24, torch.float32);  wait_tensor_24 = None
        mul_850 = torch.ops.aten.mul.Tensor(convert_element_type_2658, convert_element_type_2660);  convert_element_type_2660 = None
        mul_852 = torch.ops.aten.mul.Tensor(mul_20, mul_850)
        sum_177 = torch.ops.aten.sum.dim_IntList(mul_852, [2], True);  mul_852 = None
        div_59 = torch.ops.aten.div.Tensor(mul_20, 4096)
        mul_853 = torch.ops.aten.mul.Tensor(div_59, sum_177);  div_59 = sum_177 = None
        sub_89 = torch.ops.aten.sub.Tensor(mul_850, mul_853);  mul_850 = mul_853 = None
        mul_854 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_5);  sub_89 = rsqrt_5 = None
        mul_855 = torch.ops.aten.mul.Tensor(convert_element_type_2658, mul_20);  convert_element_type_2658 = mul_20 = None
        sum_178 = torch.ops.aten.sum.dim_IntList(mul_855, [0, 1]);  mul_855 = None
        convert_element_type_2661 = torch.ops.prims.convert_element_type.default(mul_854, torch.bfloat16);  mul_854 = None
        add_335 = torch.ops.aten.add.Tensor(add_331, convert_element_type_2661);  add_331 = convert_element_type_2661 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(sum_178, torch.float32);  sum_178 = None
        reduce_scatter_tensor_266 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_6, 'avg', 256, '0');  convert_element_type_default_6 = None
        wait_tensor_557 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_266);  reduce_scatter_tensor_266 = None
        view_1797 = torch.ops.aten.view.default(add_335, [16384, 4096])
        permute_1297 = torch.ops.aten.permute.default(view_1797, [1, 0])
        mm_639 = torch.ops.aten.mm.default(permute_1297, view_91);  permute_1297 = view_91 = None
        permute_1299 = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_640 = torch.ops.aten.mm.default(view_1797, permute_1299);  view_1797 = permute_1299 = None
        view_1798 = torch.ops.aten.view.default(mm_640, [2, 8192, 4096]);  mm_640 = None
        convert_element_type_2668 = torch.ops.prims.convert_element_type.default(mm_639, torch.float32);  mm_639 = None
        reduce_scatter_tensor_267 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2668, 'avg', 256, '0');  convert_element_type_2668 = None
        wait_tensor_558 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_267);  reduce_scatter_tensor_267 = None
        view_1799 = torch.ops.aten.view.default(view_1798, [2, 8192, 32, 128]);  view_1798 = None
        permute_1301 = torch.ops.aten.permute.default(view_1799, [0, 2, 1, 3]);  view_1799 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        all_gather_into_tensor_19 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_67, 256, '0');  convert_element_type_67 = None
        wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_19);  all_gather_into_tensor_19 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_68, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_8 = torch.ops.aten.add.Scalar(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_68, rsqrt_4);  convert_element_type_68 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, wait_tensor_19)
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_71 = torch.ops.aten.view.default(convert_element_type_69, [16384, 4096]);  convert_element_type_69 = None
        view_72 = torch.ops.aten.view.default(mm_14, [2, 8192, 4096]);  mm_14 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        all_gather_into_tensor_21 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_73, 256, '0');  convert_element_type_73 = None
        wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_21);  all_gather_into_tensor_21 = None
        permute_23 = torch.ops.aten.permute.default(wait_tensor_21, [1, 0]);  wait_tensor_21 = None
        mm_15 = torch.ops.aten.mm.default(view_71, permute_23)
        view_75 = torch.ops.aten.view.default(mm_15, [2, 8192, 1024]);  mm_15 = None
        view_78 = torch.ops.aten.view.default(mm_16, [2, 8192, 1024]);  mm_16 = None
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
        _scaled_dot_product_cudnn_attention_backward_29 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1301, permute_25, permute_26, permute_27, getitem_18, getitem_19, getitem_24, getitem_25, None, None, None, 8192, 8192, 0.0, True);  permute_1301 = permute_25 = permute_26 = permute_27 = getitem_18 = getitem_19 = getitem_24 = getitem_25 = None
        getitem_375 = _scaled_dot_product_cudnn_attention_backward_29[0]
        getitem_376 = _scaled_dot_product_cudnn_attention_backward_29[1]
        getitem_377 = _scaled_dot_product_cudnn_attention_backward_29[2];  _scaled_dot_product_cudnn_attention_backward_29 = None
        permute_1302 = torch.ops.aten.permute.default(getitem_377, [0, 2, 1, 3]);  getitem_377 = None
        permute_1303 = torch.ops.aten.permute.default(getitem_376, [0, 2, 1, 3]);  getitem_376 = None
        permute_1304 = torch.ops.aten.permute.default(getitem_375, [0, 2, 1, 3]);  getitem_375 = None
        view_1800 = torch.ops.aten.view.default(permute_1302, [2, 8192, 8, 4, 128]);  permute_1302 = None
        sum_179 = torch.ops.aten.sum.dim_IntList(view_1800, [3], True);  view_1800 = None
        squeeze_58 = torch.ops.aten.squeeze.dim(sum_179, 3);  sum_179 = None
        view_1801 = torch.ops.aten.view.default(permute_1303, [2, 8192, 8, 4, 128]);  permute_1303 = None
        sum_180 = torch.ops.aten.sum.dim_IntList(view_1801, [3], True);  view_1801 = None
        squeeze_59 = torch.ops.aten.squeeze.dim(sum_180, 3);  sum_180 = None
        convert_element_type_2669 = torch.ops.prims.convert_element_type.default(squeeze_59, torch.float32);  squeeze_59 = None
        convert_element_type_2670 = torch.ops.prims.convert_element_type.default(permute_1304, torch.float32);  permute_1304 = None
        view_1802 = torch.ops.aten.view.default(convert_element_type_2669, [2, 8192, 8, 64, 2]);  convert_element_type_2669 = None
        view_as_complex_122 = torch.ops.aten.view_as_complex.default(view_1802);  view_1802 = None
        mul_856 = torch.ops.aten.mul.Tensor(view_as_complex_122, _conj);  view_as_complex_122 = None
        view_1803 = torch.ops.aten.view.default(convert_element_type_2670, [2, 8192, 32, 64, 2]);  convert_element_type_2670 = None
        view_as_complex_123 = torch.ops.aten.view_as_complex.default(view_1803);  view_1803 = None
        mul_857 = torch.ops.aten.mul.Tensor(view_as_complex_123, _conj);  view_as_complex_123 = None
        view_as_real_122 = torch.ops.aten.view_as_real.default(mul_856);  mul_856 = None
        view_1804 = torch.ops.aten.view.default(view_as_real_122, [2, 8192, 8, 128]);  view_as_real_122 = None
        convert_element_type_2671 = torch.ops.prims.convert_element_type.default(view_1804, torch.bfloat16);  view_1804 = None
        view_as_real_123 = torch.ops.aten.view_as_real.default(mul_857);  mul_857 = None
        view_1805 = torch.ops.aten.view.default(view_as_real_123, [2, 8192, 32, 128]);  view_as_real_123 = None
        convert_element_type_2672 = torch.ops.prims.convert_element_type.default(view_1805, torch.bfloat16);  view_1805 = None
        view_1806 = torch.ops.aten.view.default(squeeze_58, [2, 8192, 1024]);  squeeze_58 = None
        view_1807 = torch.ops.aten.view.default(convert_element_type_2671, [2, 8192, 1024]);  convert_element_type_2671 = None
        view_1808 = torch.ops.aten.view.default(convert_element_type_2672, [2, 8192, 4096]);  convert_element_type_2672 = None
        view_1809 = torch.ops.aten.view.default(view_1806, [16384, 1024]);  view_1806 = None
        permute_1305 = torch.ops.aten.permute.default(view_1809, [1, 0])
        mm_641 = torch.ops.aten.mm.default(permute_1305, view_71);  permute_1305 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        all_gather_into_tensor_22 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_76, 256, '0');  convert_element_type_76 = None
        wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_22);  all_gather_into_tensor_22 = None
        permute_24 = torch.ops.aten.permute.default(wait_tensor_22, [1, 0]);  wait_tensor_22 = None
        permute_1307 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_642 = torch.ops.aten.mm.default(view_1809, permute_1307);  view_1809 = permute_1307 = None
        view_1810 = torch.ops.aten.view.default(mm_642, [2, 8192, 4096]);  mm_642 = None
        convert_element_type_2677 = torch.ops.prims.convert_element_type.default(mm_641, torch.float32);  mm_641 = None
        reduce_scatter_tensor_268 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2677, 'avg', 256, '0');  convert_element_type_2677 = None
        wait_tensor_559 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_268);  reduce_scatter_tensor_268 = None
        view_1811 = torch.ops.aten.view.default(view_1807, [16384, 1024]);  view_1807 = None
        permute_1309 = torch.ops.aten.permute.default(view_1811, [1, 0])
        mm_643 = torch.ops.aten.mm.default(permute_1309, view_71);  permute_1309 = None
        permute_1311 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_644 = torch.ops.aten.mm.default(view_1811, permute_1311);  view_1811 = permute_1311 = None
        view_1812 = torch.ops.aten.view.default(mm_644, [2, 8192, 4096]);  mm_644 = None
        add_336 = torch.ops.aten.add.Tensor(view_1810, view_1812);  view_1810 = view_1812 = None
        convert_element_type_2682 = torch.ops.prims.convert_element_type.default(mm_643, torch.float32);  mm_643 = None
        reduce_scatter_tensor_269 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2682, 'avg', 256, '0');  convert_element_type_2682 = None
        wait_tensor_560 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_269);  reduce_scatter_tensor_269 = None
        view_1813 = torch.ops.aten.view.default(view_1808, [16384, 4096]);  view_1808 = None
        permute_1313 = torch.ops.aten.permute.default(view_1813, [1, 0])
        mm_645 = torch.ops.aten.mm.default(permute_1313, view_71);  permute_1313 = view_71 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        all_gather_into_tensor_20 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_70, 256, '0');  convert_element_type_70 = None
        wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_20);  all_gather_into_tensor_20 = None
        permute_22 = torch.ops.aten.permute.default(wait_tensor_20, [1, 0]);  wait_tensor_20 = None
        permute_1315 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_646 = torch.ops.aten.mm.default(view_1813, permute_1315);  view_1813 = permute_1315 = None
        view_1814 = torch.ops.aten.view.default(mm_646, [2, 8192, 4096]);  mm_646 = None
        add_337 = torch.ops.aten.add.Tensor(add_336, view_1814);  add_336 = view_1814 = None
        convert_element_type_2687 = torch.ops.prims.convert_element_type.default(mm_645, torch.float32);  mm_645 = None
        reduce_scatter_tensor_270 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2687, 'avg', 256, '0');  convert_element_type_2687 = None
        wait_tensor_561 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_270);  reduce_scatter_tensor_270 = None
        convert_element_type_2688 = torch.ops.prims.convert_element_type.default(add_337, torch.float32);  add_337 = None
        convert_element_type_2690 = torch.ops.prims.convert_element_type.default(wait_tensor_19, torch.float32);  wait_tensor_19 = None
        mul_858 = torch.ops.aten.mul.Tensor(convert_element_type_2688, convert_element_type_2690);  convert_element_type_2690 = None
        mul_860 = torch.ops.aten.mul.Tensor(mul_16, mul_858)
        sum_181 = torch.ops.aten.sum.dim_IntList(mul_860, [2], True);  mul_860 = None
        div_60 = torch.ops.aten.div.Tensor(mul_16, 4096)
        mul_861 = torch.ops.aten.mul.Tensor(div_60, sum_181);  div_60 = sum_181 = None
        sub_90 = torch.ops.aten.sub.Tensor(mul_858, mul_861);  mul_858 = mul_861 = None
        mul_862 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_4);  sub_90 = rsqrt_4 = None
        mul_863 = torch.ops.aten.mul.Tensor(convert_element_type_2688, mul_16);  convert_element_type_2688 = mul_16 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(mul_863, [0, 1]);  mul_863 = None
        convert_element_type_2691 = torch.ops.prims.convert_element_type.default(mul_862, torch.bfloat16);  mul_862 = None
        add_338 = torch.ops.aten.add.Tensor(add_335, convert_element_type_2691);  add_335 = convert_element_type_2691 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(sum_182, torch.float32);  sum_182 = None
        reduce_scatter_tensor_271 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_5, 'avg', 256, '0');  convert_element_type_default_5 = None
        wait_tensor_562 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_271);  reduce_scatter_tensor_271 = None
        view_1815 = torch.ops.aten.view.default(add_338, [16384, 4096])
        permute_1317 = torch.ops.aten.permute.default(view_1815, [1, 0])
        permute_17 = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3])
        view_55 = torch.ops.aten.view.default(permute_17, [2, 8192, -1]);  permute_17 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        all_gather_into_tensor_14 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_50, 256, '0');  convert_element_type_50 = None
        wait_tensor_14 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        permute_18 = torch.ops.aten.permute.default(wait_tensor_14, [1, 0]);  wait_tensor_14 = None
        view_57 = torch.ops.aten.view.default(view_55, [16384, 4096]);  view_55 = None
        mm_10 = torch.ops.aten.mm.default(view_57, permute_18)
        view_58 = torch.ops.aten.view.default(mm_10, [2, 8192, 4096]);  mm_10 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, view_58);  view_58 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        all_gather_into_tensor_15 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_53, 256, '0');  convert_element_type_53 = None
        wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_6 = torch.ops.aten.add.Scalar(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_3);  convert_element_type_54 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, wait_tensor_15)
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        view_61 = torch.ops.aten.view.default(convert_element_type_55, [16384, 4096]);  convert_element_type_55 = None
        view_62 = torch.ops.aten.view.default(mm_11, [2, 8192, 14336]);  mm_11 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_59, sigmoid_1);  sigmoid_1 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        all_gather_into_tensor_17 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_61, 256, '0');  convert_element_type_61 = None
        wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_17);  all_gather_into_tensor_17 = None
        permute_20 = torch.ops.aten.permute.default(wait_tensor_17, [1, 0]);  wait_tensor_17 = None
        mm_12 = torch.ops.aten.mm.default(view_61, permute_20)
        view_65 = torch.ops.aten.view.default(mm_12, [2, 8192, 14336]);  mm_12 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_60, view_65)
        view_67 = torch.ops.aten.view.default(mul_15, [16384, 14336]);  mul_15 = None
        mm_647 = torch.ops.aten.mm.default(permute_1317, view_67);  permute_1317 = view_67 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        all_gather_into_tensor_18 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_64, 256, '0');  convert_element_type_64 = None
        wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_18);  all_gather_into_tensor_18 = None
        permute_21 = torch.ops.aten.permute.default(wait_tensor_18, [1, 0]);  wait_tensor_18 = None
        permute_1319 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_648 = torch.ops.aten.mm.default(view_1815, permute_1319);  view_1815 = permute_1319 = None
        view_1816 = torch.ops.aten.view.default(mm_648, [2, 8192, 14336]);  mm_648 = None
        convert_element_type_2698 = torch.ops.prims.convert_element_type.default(mm_647, torch.float32);  mm_647 = None
        reduce_scatter_tensor_272 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2698, 'avg', 256, '0');  convert_element_type_2698 = None
        wait_tensor_563 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_272);  reduce_scatter_tensor_272 = None
        mul_864 = torch.ops.aten.mul.Tensor(view_1816, convert_element_type_60);  convert_element_type_60 = None
        mul_865 = torch.ops.aten.mul.Tensor(view_1816, view_65);  view_1816 = view_65 = None
        view_1817 = torch.ops.aten.view.default(mul_864, [16384, 14336]);  mul_864 = None
        permute_1321 = torch.ops.aten.permute.default(view_1817, [1, 0])
        mm_649 = torch.ops.aten.mm.default(permute_1321, view_61);  permute_1321 = None
        permute_1323 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_650 = torch.ops.aten.mm.default(view_1817, permute_1323);  view_1817 = permute_1323 = None
        view_1818 = torch.ops.aten.view.default(mm_650, [2, 8192, 4096]);  mm_650 = None
        convert_element_type_2703 = torch.ops.prims.convert_element_type.default(mm_649, torch.float32);  mm_649 = None
        reduce_scatter_tensor_273 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2703, 'avg', 256, '0');  convert_element_type_2703 = None
        wait_tensor_564 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_273);  reduce_scatter_tensor_273 = None
        convert_element_type_2704 = torch.ops.prims.convert_element_type.default(mul_865, torch.float32);  mul_865 = None
        neg_30 = torch.ops.aten.neg.default(convert_element_type_59)
        exp_30 = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_339 = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(add_339);  add_339 = None
        mul_866 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        mul_867 = torch.ops.aten.mul.Tensor(convert_element_type_2704, mul_866);  convert_element_type_2704 = None
        sub_91 = torch.ops.aten.sub.Tensor(1, mul_866);  mul_866 = None
        mul_868 = torch.ops.aten.mul.Tensor(convert_element_type_59, sub_91);  convert_element_type_59 = sub_91 = None
        add_340 = torch.ops.aten.add.Tensor(mul_868, 1);  mul_868 = None
        mul_869 = torch.ops.aten.mul.Tensor(mul_867, add_340);  mul_867 = add_340 = None
        convert_element_type_2706 = torch.ops.prims.convert_element_type.default(mul_869, torch.bfloat16);  mul_869 = None
        view_1819 = torch.ops.aten.view.default(convert_element_type_2706, [16384, 14336]);  convert_element_type_2706 = None
        permute_1325 = torch.ops.aten.permute.default(view_1819, [1, 0])
        mm_651 = torch.ops.aten.mm.default(permute_1325, view_61);  permute_1325 = view_61 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        all_gather_into_tensor_16 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_56, 256, '0');  convert_element_type_56 = None
        wait_tensor_16 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        permute_19 = torch.ops.aten.permute.default(wait_tensor_16, [1, 0]);  wait_tensor_16 = None
        permute_1327 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_652 = torch.ops.aten.mm.default(view_1819, permute_1327);  view_1819 = permute_1327 = None
        view_1820 = torch.ops.aten.view.default(mm_652, [2, 8192, 4096]);  mm_652 = None
        add_341 = torch.ops.aten.add.Tensor(view_1818, view_1820);  view_1818 = view_1820 = None
        convert_element_type_2711 = torch.ops.prims.convert_element_type.default(mm_651, torch.float32);  mm_651 = None
        reduce_scatter_tensor_274 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2711, 'avg', 256, '0');  convert_element_type_2711 = None
        wait_tensor_565 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_274);  reduce_scatter_tensor_274 = None
        convert_element_type_2712 = torch.ops.prims.convert_element_type.default(add_341, torch.float32);  add_341 = None
        convert_element_type_2714 = torch.ops.prims.convert_element_type.default(wait_tensor_15, torch.float32);  wait_tensor_15 = None
        mul_870 = torch.ops.aten.mul.Tensor(convert_element_type_2712, convert_element_type_2714);  convert_element_type_2714 = None
        mul_872 = torch.ops.aten.mul.Tensor(mul_12, mul_870)
        sum_183 = torch.ops.aten.sum.dim_IntList(mul_872, [2], True);  mul_872 = None
        div_61 = torch.ops.aten.div.Tensor(mul_12, 4096)
        mul_873 = torch.ops.aten.mul.Tensor(div_61, sum_183);  div_61 = sum_183 = None
        sub_92 = torch.ops.aten.sub.Tensor(mul_870, mul_873);  mul_870 = mul_873 = None
        mul_874 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_3);  sub_92 = rsqrt_3 = None
        mul_875 = torch.ops.aten.mul.Tensor(convert_element_type_2712, mul_12);  convert_element_type_2712 = mul_12 = None
        sum_184 = torch.ops.aten.sum.dim_IntList(mul_875, [0, 1]);  mul_875 = None
        convert_element_type_2715 = torch.ops.prims.convert_element_type.default(mul_874, torch.bfloat16);  mul_874 = None
        add_342 = torch.ops.aten.add.Tensor(add_338, convert_element_type_2715);  add_338 = convert_element_type_2715 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sum_184, torch.float32);  sum_184 = None
        reduce_scatter_tensor_275 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_4, 'avg', 256, '0');  convert_element_type_default_4 = None
        wait_tensor_566 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_275);  reduce_scatter_tensor_275 = None
        view_1821 = torch.ops.aten.view.default(add_342, [16384, 4096])
        permute_1329 = torch.ops.aten.permute.default(view_1821, [1, 0])
        mm_653 = torch.ops.aten.mm.default(permute_1329, view_57);  permute_1329 = view_57 = None
        permute_1331 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_654 = torch.ops.aten.mm.default(view_1821, permute_1331);  view_1821 = permute_1331 = None
        view_1822 = torch.ops.aten.view.default(mm_654, [2, 8192, 4096]);  mm_654 = None
        convert_element_type_2722 = torch.ops.prims.convert_element_type.default(mm_653, torch.float32);  mm_653 = None
        reduce_scatter_tensor_276 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2722, 'avg', 256, '0');  convert_element_type_2722 = None
        wait_tensor_567 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_276);  reduce_scatter_tensor_276 = None
        view_1823 = torch.ops.aten.view.default(view_1822, [2, 8192, 32, 128]);  view_1822 = None
        permute_1333 = torch.ops.aten.permute.default(view_1823, [0, 2, 1, 3]);  view_1823 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        all_gather_into_tensor_10 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_34, 256, '0');  convert_element_type_34 = None
        wait_tensor_10 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_4 = torch.ops.aten.add.Scalar(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_2);  convert_element_type_35 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, wait_tensor_10)
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_37 = torch.ops.aten.view.default(convert_element_type_36, [16384, 4096]);  convert_element_type_36 = None
        view_38 = torch.ops.aten.view.default(mm_7, [2, 8192, 4096]);  mm_7 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        all_gather_into_tensor_12 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_40, 256, '0');  convert_element_type_40 = None
        wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        permute_12 = torch.ops.aten.permute.default(wait_tensor_12, [1, 0]);  wait_tensor_12 = None
        mm_8 = torch.ops.aten.mm.default(view_37, permute_12)
        view_41 = torch.ops.aten.view.default(mm_8, [2, 8192, 1024]);  mm_8 = None
        view_44 = torch.ops.aten.view.default(mm_9, [2, 8192, 1024]);  mm_9 = None
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
        _scaled_dot_product_cudnn_attention_backward_30 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1333, permute_14, permute_15, permute_16, getitem_9, getitem_10, getitem_15, getitem_16, None, None, None, 8192, 8192, 0.0, True);  permute_1333 = permute_14 = permute_15 = permute_16 = getitem_9 = getitem_10 = getitem_15 = getitem_16 = None
        getitem_378 = _scaled_dot_product_cudnn_attention_backward_30[0]
        getitem_379 = _scaled_dot_product_cudnn_attention_backward_30[1]
        getitem_380 = _scaled_dot_product_cudnn_attention_backward_30[2];  _scaled_dot_product_cudnn_attention_backward_30 = None
        permute_1334 = torch.ops.aten.permute.default(getitem_380, [0, 2, 1, 3]);  getitem_380 = None
        permute_1335 = torch.ops.aten.permute.default(getitem_379, [0, 2, 1, 3]);  getitem_379 = None
        permute_1336 = torch.ops.aten.permute.default(getitem_378, [0, 2, 1, 3]);  getitem_378 = None
        view_1824 = torch.ops.aten.view.default(permute_1334, [2, 8192, 8, 4, 128]);  permute_1334 = None
        sum_185 = torch.ops.aten.sum.dim_IntList(view_1824, [3], True);  view_1824 = None
        squeeze_60 = torch.ops.aten.squeeze.dim(sum_185, 3);  sum_185 = None
        view_1825 = torch.ops.aten.view.default(permute_1335, [2, 8192, 8, 4, 128]);  permute_1335 = None
        sum_186 = torch.ops.aten.sum.dim_IntList(view_1825, [3], True);  view_1825 = None
        squeeze_61 = torch.ops.aten.squeeze.dim(sum_186, 3);  sum_186 = None
        convert_element_type_2723 = torch.ops.prims.convert_element_type.default(squeeze_61, torch.float32);  squeeze_61 = None
        convert_element_type_2724 = torch.ops.prims.convert_element_type.default(permute_1336, torch.float32);  permute_1336 = None
        view_1826 = torch.ops.aten.view.default(convert_element_type_2723, [2, 8192, 8, 64, 2]);  convert_element_type_2723 = None
        view_as_complex_124 = torch.ops.aten.view_as_complex.default(view_1826);  view_1826 = None
        mul_876 = torch.ops.aten.mul.Tensor(view_as_complex_124, _conj);  view_as_complex_124 = None
        view_1827 = torch.ops.aten.view.default(convert_element_type_2724, [2, 8192, 32, 64, 2]);  convert_element_type_2724 = None
        view_as_complex_125 = torch.ops.aten.view_as_complex.default(view_1827);  view_1827 = None
        mul_877 = torch.ops.aten.mul.Tensor(view_as_complex_125, _conj);  view_as_complex_125 = None
        view_as_real_124 = torch.ops.aten.view_as_real.default(mul_876);  mul_876 = None
        view_1828 = torch.ops.aten.view.default(view_as_real_124, [2, 8192, 8, 128]);  view_as_real_124 = None
        convert_element_type_2725 = torch.ops.prims.convert_element_type.default(view_1828, torch.bfloat16);  view_1828 = None
        view_as_real_125 = torch.ops.aten.view_as_real.default(mul_877);  mul_877 = None
        view_1829 = torch.ops.aten.view.default(view_as_real_125, [2, 8192, 32, 128]);  view_as_real_125 = None
        convert_element_type_2726 = torch.ops.prims.convert_element_type.default(view_1829, torch.bfloat16);  view_1829 = None
        view_1830 = torch.ops.aten.view.default(squeeze_60, [2, 8192, 1024]);  squeeze_60 = None
        view_1831 = torch.ops.aten.view.default(convert_element_type_2725, [2, 8192, 1024]);  convert_element_type_2725 = None
        view_1832 = torch.ops.aten.view.default(convert_element_type_2726, [2, 8192, 4096]);  convert_element_type_2726 = None
        view_1833 = torch.ops.aten.view.default(view_1830, [16384, 1024]);  view_1830 = None
        permute_1337 = torch.ops.aten.permute.default(view_1833, [1, 0])
        mm_655 = torch.ops.aten.mm.default(permute_1337, view_37);  permute_1337 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        all_gather_into_tensor_13 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_43, 256, '0');  convert_element_type_43 = None
        wait_tensor_13 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_13);  all_gather_into_tensor_13 = None
        permute_13 = torch.ops.aten.permute.default(wait_tensor_13, [1, 0]);  wait_tensor_13 = None
        permute_1339 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_656 = torch.ops.aten.mm.default(view_1833, permute_1339);  view_1833 = permute_1339 = None
        view_1834 = torch.ops.aten.view.default(mm_656, [2, 8192, 4096]);  mm_656 = None
        convert_element_type_2731 = torch.ops.prims.convert_element_type.default(mm_655, torch.float32);  mm_655 = None
        reduce_scatter_tensor_277 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2731, 'avg', 256, '0');  convert_element_type_2731 = None
        wait_tensor_568 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_277);  reduce_scatter_tensor_277 = None
        view_1835 = torch.ops.aten.view.default(view_1831, [16384, 1024]);  view_1831 = None
        permute_1341 = torch.ops.aten.permute.default(view_1835, [1, 0])
        mm_657 = torch.ops.aten.mm.default(permute_1341, view_37);  permute_1341 = None
        permute_1343 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_658 = torch.ops.aten.mm.default(view_1835, permute_1343);  view_1835 = permute_1343 = None
        view_1836 = torch.ops.aten.view.default(mm_658, [2, 8192, 4096]);  mm_658 = None
        add_343 = torch.ops.aten.add.Tensor(view_1834, view_1836);  view_1834 = view_1836 = None
        convert_element_type_2736 = torch.ops.prims.convert_element_type.default(mm_657, torch.float32);  mm_657 = None
        reduce_scatter_tensor_278 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2736, 'avg', 256, '0');  convert_element_type_2736 = None
        wait_tensor_569 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_278);  reduce_scatter_tensor_278 = None
        view_1837 = torch.ops.aten.view.default(view_1832, [16384, 4096]);  view_1832 = None
        permute_1345 = torch.ops.aten.permute.default(view_1837, [1, 0])
        mm_659 = torch.ops.aten.mm.default(permute_1345, view_37);  permute_1345 = view_37 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        all_gather_into_tensor_11 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_37, 256, '0');  convert_element_type_37 = None
        wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        permute_11 = torch.ops.aten.permute.default(wait_tensor_11, [1, 0]);  wait_tensor_11 = None
        permute_1347 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_660 = torch.ops.aten.mm.default(view_1837, permute_1347);  view_1837 = permute_1347 = None
        view_1838 = torch.ops.aten.view.default(mm_660, [2, 8192, 4096]);  mm_660 = None
        add_344 = torch.ops.aten.add.Tensor(add_343, view_1838);  add_343 = view_1838 = None
        convert_element_type_2741 = torch.ops.prims.convert_element_type.default(mm_659, torch.float32);  mm_659 = None
        reduce_scatter_tensor_279 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2741, 'avg', 256, '0');  convert_element_type_2741 = None
        wait_tensor_570 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_279);  reduce_scatter_tensor_279 = None
        convert_element_type_2742 = torch.ops.prims.convert_element_type.default(add_344, torch.float32);  add_344 = None
        convert_element_type_2744 = torch.ops.prims.convert_element_type.default(wait_tensor_10, torch.float32);  wait_tensor_10 = None
        mul_878 = torch.ops.aten.mul.Tensor(convert_element_type_2742, convert_element_type_2744);  convert_element_type_2744 = None
        mul_880 = torch.ops.aten.mul.Tensor(mul_8, mul_878)
        sum_187 = torch.ops.aten.sum.dim_IntList(mul_880, [2], True);  mul_880 = None
        div_62 = torch.ops.aten.div.Tensor(mul_8, 4096)
        mul_881 = torch.ops.aten.mul.Tensor(div_62, sum_187);  div_62 = sum_187 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_878, mul_881);  mul_878 = mul_881 = None
        mul_882 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_2);  sub_93 = rsqrt_2 = None
        mul_883 = torch.ops.aten.mul.Tensor(convert_element_type_2742, mul_8);  convert_element_type_2742 = mul_8 = None
        sum_188 = torch.ops.aten.sum.dim_IntList(mul_883, [0, 1]);  mul_883 = None
        convert_element_type_2745 = torch.ops.prims.convert_element_type.default(mul_882, torch.bfloat16);  mul_882 = None
        add_345 = torch.ops.aten.add.Tensor(add_342, convert_element_type_2745);  add_342 = convert_element_type_2745 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sum_188, torch.float32);  sum_188 = None
        reduce_scatter_tensor_280 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_3, 'avg', 256, '0');  convert_element_type_default_3 = None
        wait_tensor_571 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_280);  reduce_scatter_tensor_280 = None
        view_1839 = torch.ops.aten.view.default(add_345, [16384, 4096])
        permute_1349 = torch.ops.aten.permute.default(view_1839, [1, 0])
        permute_6 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])
        view_21 = torch.ops.aten.view.default(permute_6, [2, 8192, -1]);  permute_6 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        all_gather_into_tensor_5 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_17, 256, '0');  convert_element_type_17 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        permute_7 = torch.ops.aten.permute.default(wait_tensor_5, [1, 0]);  wait_tensor_5 = None
        view_23 = torch.ops.aten.view.default(view_21, [16384, 4096]);  view_21 = None
        mm_3 = torch.ops.aten.mm.default(view_23, permute_7)
        view_24 = torch.ops.aten.view.default(mm_3, [2, 8192, 4096]);  mm_3 = None
        add_1 = torch.ops.aten.add.Tensor(embedding, view_24);  view_24 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        all_gather_into_tensor_6 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_20, 256, '0');  convert_element_type_20 = None
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Scalar(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_1);  convert_element_type_21 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, wait_tensor_6)
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_27 = torch.ops.aten.view.default(convert_element_type_22, [16384, 4096]);  convert_element_type_22 = None
        view_28 = torch.ops.aten.view.default(mm_4, [2, 8192, 14336]);  mm_4 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_26)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_26, sigmoid);  sigmoid = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        all_gather_into_tensor_8 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_28, 256, '0');  convert_element_type_28 = None
        wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        permute_9 = torch.ops.aten.permute.default(wait_tensor_8, [1, 0]);  wait_tensor_8 = None
        mm_5 = torch.ops.aten.mm.default(view_27, permute_9)
        view_31 = torch.ops.aten.view.default(mm_5, [2, 8192, 14336]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_27, view_31)
        view_33 = torch.ops.aten.view.default(mul_7, [16384, 14336]);  mul_7 = None
        mm_661 = torch.ops.aten.mm.default(permute_1349, view_33);  permute_1349 = view_33 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        all_gather_into_tensor_9 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_31, 256, '0');  convert_element_type_31 = None
        wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        permute_10 = torch.ops.aten.permute.default(wait_tensor_9, [1, 0]);  wait_tensor_9 = None
        permute_1351 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_662 = torch.ops.aten.mm.default(view_1839, permute_1351);  view_1839 = permute_1351 = None
        view_1840 = torch.ops.aten.view.default(mm_662, [2, 8192, 14336]);  mm_662 = None
        convert_element_type_2752 = torch.ops.prims.convert_element_type.default(mm_661, torch.float32);  mm_661 = None
        reduce_scatter_tensor_281 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2752, 'avg', 256, '0');  convert_element_type_2752 = None
        wait_tensor_572 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_281);  reduce_scatter_tensor_281 = None
        mul_884 = torch.ops.aten.mul.Tensor(view_1840, convert_element_type_27);  convert_element_type_27 = None
        mul_885 = torch.ops.aten.mul.Tensor(view_1840, view_31);  view_1840 = view_31 = None
        view_1841 = torch.ops.aten.view.default(mul_884, [16384, 14336]);  mul_884 = None
        permute_1353 = torch.ops.aten.permute.default(view_1841, [1, 0])
        mm_663 = torch.ops.aten.mm.default(permute_1353, view_27);  permute_1353 = None
        permute_1355 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_664 = torch.ops.aten.mm.default(view_1841, permute_1355);  view_1841 = permute_1355 = None
        view_1842 = torch.ops.aten.view.default(mm_664, [2, 8192, 4096]);  mm_664 = None
        convert_element_type_2757 = torch.ops.prims.convert_element_type.default(mm_663, torch.float32);  mm_663 = None
        reduce_scatter_tensor_282 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2757, 'avg', 256, '0');  convert_element_type_2757 = None
        wait_tensor_573 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_282);  reduce_scatter_tensor_282 = None
        convert_element_type_2758 = torch.ops.prims.convert_element_type.default(mul_885, torch.float32);  mul_885 = None
        neg_31 = torch.ops.aten.neg.default(convert_element_type_26)
        exp_31 = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_346 = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(add_346);  add_346 = None
        mul_886 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        mul_887 = torch.ops.aten.mul.Tensor(convert_element_type_2758, mul_886);  convert_element_type_2758 = None
        sub_94 = torch.ops.aten.sub.Tensor(1, mul_886);  mul_886 = None
        mul_888 = torch.ops.aten.mul.Tensor(convert_element_type_26, sub_94);  convert_element_type_26 = sub_94 = None
        add_347 = torch.ops.aten.add.Tensor(mul_888, 1);  mul_888 = None
        mul_889 = torch.ops.aten.mul.Tensor(mul_887, add_347);  mul_887 = add_347 = None
        convert_element_type_2760 = torch.ops.prims.convert_element_type.default(mul_889, torch.bfloat16);  mul_889 = None
        view_1843 = torch.ops.aten.view.default(convert_element_type_2760, [16384, 14336]);  convert_element_type_2760 = None
        permute_1357 = torch.ops.aten.permute.default(view_1843, [1, 0])
        mm_665 = torch.ops.aten.mm.default(permute_1357, view_27);  permute_1357 = view_27 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        all_gather_into_tensor_7 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_23, 256, '0');  convert_element_type_23 = None
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        permute_8 = torch.ops.aten.permute.default(wait_tensor_7, [1, 0]);  wait_tensor_7 = None
        permute_1359 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_666 = torch.ops.aten.mm.default(view_1843, permute_1359);  view_1843 = permute_1359 = None
        view_1844 = torch.ops.aten.view.default(mm_666, [2, 8192, 4096]);  mm_666 = None
        add_348 = torch.ops.aten.add.Tensor(view_1842, view_1844);  view_1842 = view_1844 = None
        convert_element_type_2765 = torch.ops.prims.convert_element_type.default(mm_665, torch.float32);  mm_665 = None
        reduce_scatter_tensor_283 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2765, 'avg', 256, '0');  convert_element_type_2765 = None
        wait_tensor_574 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_283);  reduce_scatter_tensor_283 = None
        convert_element_type_2766 = torch.ops.prims.convert_element_type.default(add_348, torch.float32);  add_348 = None
        convert_element_type_2768 = torch.ops.prims.convert_element_type.default(wait_tensor_6, torch.float32);  wait_tensor_6 = None
        mul_890 = torch.ops.aten.mul.Tensor(convert_element_type_2766, convert_element_type_2768);  convert_element_type_2768 = None
        mul_892 = torch.ops.aten.mul.Tensor(mul_4, mul_890)
        sum_189 = torch.ops.aten.sum.dim_IntList(mul_892, [2], True);  mul_892 = None
        div_63 = torch.ops.aten.div.Tensor(mul_4, 4096)
        mul_893 = torch.ops.aten.mul.Tensor(div_63, sum_189);  div_63 = sum_189 = None
        sub_95 = torch.ops.aten.sub.Tensor(mul_890, mul_893);  mul_890 = mul_893 = None
        mul_894 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_1);  sub_95 = rsqrt_1 = None
        mul_895 = torch.ops.aten.mul.Tensor(convert_element_type_2766, mul_4);  convert_element_type_2766 = mul_4 = None
        sum_190 = torch.ops.aten.sum.dim_IntList(mul_895, [0, 1]);  mul_895 = None
        convert_element_type_2769 = torch.ops.prims.convert_element_type.default(mul_894, torch.bfloat16);  mul_894 = None
        add_349 = torch.ops.aten.add.Tensor(add_345, convert_element_type_2769);  add_345 = convert_element_type_2769 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sum_190, torch.float32);  sum_190 = None
        reduce_scatter_tensor_284 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_2, 'avg', 256, '0');  convert_element_type_default_2 = None
        wait_tensor_575 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_284);  reduce_scatter_tensor_284 = None
        view_1845 = torch.ops.aten.view.default(add_349, [16384, 4096])
        permute_1361 = torch.ops.aten.permute.default(view_1845, [1, 0])
        mm_667 = torch.ops.aten.mm.default(permute_1361, view_23);  permute_1361 = view_23 = None
        permute_1363 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_668 = torch.ops.aten.mm.default(view_1845, permute_1363);  view_1845 = permute_1363 = None
        view_1846 = torch.ops.aten.view.default(mm_668, [2, 8192, 4096]);  mm_668 = None
        convert_element_type_2776 = torch.ops.prims.convert_element_type.default(mm_667, torch.float32);  mm_667 = None
        reduce_scatter_tensor_285 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2776, 'avg', 256, '0');  convert_element_type_2776 = None
        wait_tensor_576 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_285);  reduce_scatter_tensor_285 = None
        view_1847 = torch.ops.aten.view.default(view_1846, [2, 8192, 32, 128]);  view_1846 = None
        permute_1365 = torch.ops.aten.permute.default(view_1847, [0, 2, 1, 3]);  view_1847 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 256, '0');  convert_element_type_1 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, wait_tensor_1)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view_3 = torch.ops.aten.view.default(convert_element_type_3, [16384, 4096]);  convert_element_type_3 = None
        view_4 = torch.ops.aten.view.default(mm, [2, 8192, 4096]);  mm = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        all_gather_into_tensor_3 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_7, 256, '0');  convert_element_type_7 = None
        wait_tensor_3 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        permute_1 = torch.ops.aten.permute.default(wait_tensor_3, [1, 0]);  wait_tensor_3 = None
        mm_1 = torch.ops.aten.mm.default(view_3, permute_1)
        view_7 = torch.ops.aten.view.default(mm_1, [2, 8192, 1024]);  mm_1 = None
        view_10 = torch.ops.aten.view.default(mm_2, [2, 8192, 1024]);  mm_2 = None
        view_11 = torch.ops.aten.view.default(view_4, [2, 8192, -1, 128]);  view_4 = None
        view_12 = torch.ops.aten.view.default(view_7, [2, 8192, -1, 128]);  view_7 = None
        view_13 = torch.ops.aten.view.default(view_10, [2, 8192, -1, 128]);  view_10 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_13, [2, 8192, 32, -1, 2]);  convert_element_type_13 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_14);  view_14 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        view_15 = torch.ops.aten.view.default(convert_element_type_14, [2, 8192, 8, -1, 2]);  convert_element_type_14 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_15);  view_15 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_16);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_17 = torch.ops.aten.view.default(view_as_real, [2, 8192, 32, 128]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_16);  view_as_complex_1 = view_16 = None
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
        _scaled_dot_product_cudnn_attention_backward_31 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_1365, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, None, None, None, 8192, 8192, 0.0, True);  permute_1365 = permute_3 = permute_4 = permute_5 = getitem = getitem_1 = getitem_6 = getitem_7 = None
        getitem_381 = _scaled_dot_product_cudnn_attention_backward_31[0]
        getitem_382 = _scaled_dot_product_cudnn_attention_backward_31[1]
        getitem_383 = _scaled_dot_product_cudnn_attention_backward_31[2];  _scaled_dot_product_cudnn_attention_backward_31 = None
        permute_1366 = torch.ops.aten.permute.default(getitem_383, [0, 2, 1, 3]);  getitem_383 = None
        permute_1367 = torch.ops.aten.permute.default(getitem_382, [0, 2, 1, 3]);  getitem_382 = None
        permute_1368 = torch.ops.aten.permute.default(getitem_381, [0, 2, 1, 3]);  getitem_381 = None
        view_1848 = torch.ops.aten.view.default(permute_1366, [2, 8192, 8, 4, 128]);  permute_1366 = None
        sum_191 = torch.ops.aten.sum.dim_IntList(view_1848, [3], True);  view_1848 = None
        squeeze_62 = torch.ops.aten.squeeze.dim(sum_191, 3);  sum_191 = None
        view_1849 = torch.ops.aten.view.default(permute_1367, [2, 8192, 8, 4, 128]);  permute_1367 = None
        sum_192 = torch.ops.aten.sum.dim_IntList(view_1849, [3], True);  view_1849 = None
        squeeze_63 = torch.ops.aten.squeeze.dim(sum_192, 3);  sum_192 = None
        convert_element_type_2777 = torch.ops.prims.convert_element_type.default(squeeze_63, torch.float32);  squeeze_63 = None
        convert_element_type_2778 = torch.ops.prims.convert_element_type.default(permute_1368, torch.float32);  permute_1368 = None
        view_1850 = torch.ops.aten.view.default(convert_element_type_2777, [2, 8192, 8, 64, 2]);  convert_element_type_2777 = None
        view_as_complex_126 = torch.ops.aten.view_as_complex.default(view_1850);  view_1850 = None
        mul_896 = torch.ops.aten.mul.Tensor(view_as_complex_126, _conj);  view_as_complex_126 = None
        view_1851 = torch.ops.aten.view.default(convert_element_type_2778, [2, 8192, 32, 64, 2]);  convert_element_type_2778 = None
        view_as_complex_127 = torch.ops.aten.view_as_complex.default(view_1851);  view_1851 = None
        mul_897 = torch.ops.aten.mul.Tensor(view_as_complex_127, _conj);  view_as_complex_127 = _conj = None
        view_as_real_126 = torch.ops.aten.view_as_real.default(mul_896);  mul_896 = None
        view_1852 = torch.ops.aten.view.default(view_as_real_126, [2, 8192, 8, 128]);  view_as_real_126 = None
        convert_element_type_2779 = torch.ops.prims.convert_element_type.default(view_1852, torch.bfloat16);  view_1852 = None
        view_as_real_127 = torch.ops.aten.view_as_real.default(mul_897);  mul_897 = None
        view_1853 = torch.ops.aten.view.default(view_as_real_127, [2, 8192, 32, 128]);  view_as_real_127 = None
        convert_element_type_2780 = torch.ops.prims.convert_element_type.default(view_1853, torch.bfloat16);  view_1853 = None
        view_1854 = torch.ops.aten.view.default(squeeze_62, [2, 8192, 1024]);  squeeze_62 = None
        view_1855 = torch.ops.aten.view.default(convert_element_type_2779, [2, 8192, 1024]);  convert_element_type_2779 = None
        view_1856 = torch.ops.aten.view.default(convert_element_type_2780, [2, 8192, 4096]);  convert_element_type_2780 = None
        view_1857 = torch.ops.aten.view.default(view_1854, [16384, 1024]);  view_1854 = None
        permute_1369 = torch.ops.aten.permute.default(view_1857, [1, 0])
        mm_669 = torch.ops.aten.mm.default(permute_1369, view_3);  permute_1369 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        all_gather_into_tensor_4 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_10, 256, '0');  convert_element_type_10 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        permute_2 = torch.ops.aten.permute.default(wait_tensor_4, [1, 0]);  wait_tensor_4 = None
        permute_1371 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_670 = torch.ops.aten.mm.default(view_1857, permute_1371);  view_1857 = permute_1371 = None
        view_1858 = torch.ops.aten.view.default(mm_670, [2, 8192, 4096]);  mm_670 = None
        convert_element_type_2785 = torch.ops.prims.convert_element_type.default(mm_669, torch.float32);  mm_669 = None
        reduce_scatter_tensor_286 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2785, 'avg', 256, '0');  convert_element_type_2785 = None
        wait_tensor_577 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_286);  reduce_scatter_tensor_286 = None
        view_1859 = torch.ops.aten.view.default(view_1855, [16384, 1024]);  view_1855 = None
        permute_1373 = torch.ops.aten.permute.default(view_1859, [1, 0])
        mm_671 = torch.ops.aten.mm.default(permute_1373, view_3);  permute_1373 = None
        permute_1375 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_672 = torch.ops.aten.mm.default(view_1859, permute_1375);  view_1859 = permute_1375 = None
        view_1860 = torch.ops.aten.view.default(mm_672, [2, 8192, 4096]);  mm_672 = None
        add_350 = torch.ops.aten.add.Tensor(view_1858, view_1860);  view_1858 = view_1860 = None
        convert_element_type_2790 = torch.ops.prims.convert_element_type.default(mm_671, torch.float32);  mm_671 = None
        reduce_scatter_tensor_287 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2790, 'avg', 256, '0');  convert_element_type_2790 = None
        wait_tensor_578 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_287);  reduce_scatter_tensor_287 = None
        view_1861 = torch.ops.aten.view.default(view_1856, [16384, 4096]);  view_1856 = None
        permute_1377 = torch.ops.aten.permute.default(view_1861, [1, 0])
        mm_673 = torch.ops.aten.mm.default(permute_1377, view_3);  permute_1377 = view_3 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        all_gather_into_tensor_2 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_4, 256, '0');  convert_element_type_4 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        permute = torch.ops.aten.permute.default(wait_tensor_2, [1, 0]);  wait_tensor_2 = None
        permute_1379 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_674 = torch.ops.aten.mm.default(view_1861, permute_1379);  view_1861 = permute_1379 = None
        view_1862 = torch.ops.aten.view.default(mm_674, [2, 8192, 4096]);  mm_674 = None
        add_351 = torch.ops.aten.add.Tensor(add_350, view_1862);  add_350 = view_1862 = None
        convert_element_type_2795 = torch.ops.prims.convert_element_type.default(mm_673, torch.float32);  mm_673 = None
        reduce_scatter_tensor_288 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_2795, 'avg', 256, '0');  convert_element_type_2795 = None
        wait_tensor_579 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_288);  reduce_scatter_tensor_288 = None
        convert_element_type_2796 = torch.ops.prims.convert_element_type.default(add_351, torch.float32);  add_351 = None
        convert_element_type_2798 = torch.ops.prims.convert_element_type.default(wait_tensor_1, torch.float32);  wait_tensor_1 = None
        mul_898 = torch.ops.aten.mul.Tensor(convert_element_type_2796, convert_element_type_2798);  convert_element_type_2798 = None
        mul_900 = torch.ops.aten.mul.Tensor(mul, mul_898)
        sum_193 = torch.ops.aten.sum.dim_IntList(mul_900, [2], True);  mul_900 = None
        div_64 = torch.ops.aten.div.Tensor(mul, 4096)
        mul_901 = torch.ops.aten.mul.Tensor(div_64, sum_193);  div_64 = sum_193 = None
        sub_96 = torch.ops.aten.sub.Tensor(mul_898, mul_901);  mul_898 = mul_901 = None
        mul_902 = torch.ops.aten.mul.Tensor(sub_96, rsqrt);  sub_96 = rsqrt = None
        mul_903 = torch.ops.aten.mul.Tensor(convert_element_type_2796, mul);  convert_element_type_2796 = mul = None
        sum_194 = torch.ops.aten.sum.dim_IntList(mul_903, [0, 1]);  mul_903 = None
        convert_element_type_2799 = torch.ops.prims.convert_element_type.default(mul_902, torch.bfloat16);  mul_902 = None
        add_352 = torch.ops.aten.add.Tensor(add_349, convert_element_type_2799);  add_349 = convert_element_type_2799 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sum_194, torch.float32);  sum_194 = None
        reduce_scatter_tensor_289 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default_1, 'avg', 256, '0');  convert_element_type_default_1 = None
        wait_tensor_580 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_289);  reduce_scatter_tensor_289 = None
        convert_element_type_2802 = torch.ops.prims.convert_element_type.default(add_352, torch.float32);  add_352 = None
        eq = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(unsqueeze_64, full_default, convert_element_type_2802);  unsqueeze_64 = full_default = convert_element_type_2802 = None
        full_default_1 = torch.ops.aten.full.default([128256, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default_1, [primals_2], where, True);  full_default_1 = primals_2 = where = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(index_put, torch.float32);  index_put = None
        reduce_scatter_tensor_290 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_default, 'avg', 256, '0');  convert_element_type_default = None
        wait_tensor_581 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_290);  reduce_scatter_tensor_290 = None
        return (wait_tensor_581, None, None, wait_tensor_580, wait_tensor_579, wait_tensor_578, wait_tensor_577, wait_tensor_576, wait_tensor_575, wait_tensor_574, wait_tensor_573, wait_tensor_572, wait_tensor_571, wait_tensor_570, wait_tensor_569, wait_tensor_568, wait_tensor_567, wait_tensor_566, wait_tensor_565, wait_tensor_564, wait_tensor_563, wait_tensor_562, wait_tensor_561, wait_tensor_560, wait_tensor_559, wait_tensor_558, wait_tensor_557, wait_tensor_556, wait_tensor_555, wait_tensor_554, wait_tensor_553, wait_tensor_552, wait_tensor_551, wait_tensor_550, wait_tensor_549, wait_tensor_548, wait_tensor_547, wait_tensor_546, wait_tensor_545, wait_tensor_544, wait_tensor_543, wait_tensor_542, wait_tensor_541, wait_tensor_540, wait_tensor_539, wait_tensor_538, wait_tensor_537, wait_tensor_536, wait_tensor_535, wait_tensor_534, wait_tensor_533, wait_tensor_532, wait_tensor_531, wait_tensor_530, wait_tensor_529, wait_tensor_528, wait_tensor_527, wait_tensor_526, wait_tensor_525, wait_tensor_524, wait_tensor_523, wait_tensor_522, wait_tensor_521, wait_tensor_520, wait_tensor_519, wait_tensor_518, wait_tensor_517, wait_tensor_516, wait_tensor_515, wait_tensor_514, wait_tensor_513, wait_tensor_512, wait_tensor_511, wait_tensor_510, wait_tensor_509, wait_tensor_508, wait_tensor_507, wait_tensor_506, wait_tensor_505, wait_tensor_504, wait_tensor_503, wait_tensor_502, wait_tensor_501, wait_tensor_500, wait_tensor_499, wait_tensor_498, wait_tensor_497, wait_tensor_496, wait_tensor_495, wait_tensor_494, wait_tensor_493, wait_tensor_492, wait_tensor_491, wait_tensor_490, wait_tensor_489, wait_tensor_488, wait_tensor_487, wait_tensor_486, wait_tensor_485, wait_tensor_484, wait_tensor_483, wait_tensor_482, wait_tensor_481, wait_tensor_480, wait_tensor_479, wait_tensor_478, wait_tensor_477, wait_tensor_476, wait_tensor_475, wait_tensor_474, wait_tensor_473, wait_tensor_472, wait_tensor_471, wait_tensor_470, wait_tensor_469, wait_tensor_468, wait_tensor_467, wait_tensor_466, wait_tensor_465, wait_tensor_464, wait_tensor_463, wait_tensor_462, wait_tensor_461, wait_tensor_460, wait_tensor_459, wait_tensor_458, wait_tensor_457, wait_tensor_456, wait_tensor_455, wait_tensor_454, wait_tensor_453, wait_tensor_452, wait_tensor_451, wait_tensor_450, wait_tensor_449, wait_tensor_448, wait_tensor_447, wait_tensor_446, wait_tensor_445, wait_tensor_444, wait_tensor_443, wait_tensor_442, wait_tensor_441, wait_tensor_440, wait_tensor_439, wait_tensor_438, wait_tensor_437, wait_tensor_436, wait_tensor_435, wait_tensor_434, wait_tensor_433, wait_tensor_432, wait_tensor_431, wait_tensor_430, wait_tensor_429, wait_tensor_428, wait_tensor_427, wait_tensor_426, wait_tensor_425, wait_tensor_424, wait_tensor_423, wait_tensor_422, wait_tensor_421, wait_tensor_420, wait_tensor_419, wait_tensor_418, wait_tensor_417, wait_tensor_416, wait_tensor_415, wait_tensor_414, wait_tensor_413, wait_tensor_412, wait_tensor_411, wait_tensor_410, wait_tensor_409, wait_tensor_408, wait_tensor_407, wait_tensor_406, wait_tensor_405, wait_tensor_404, wait_tensor_403, wait_tensor_402, wait_tensor_401, wait_tensor_400, wait_tensor_399, wait_tensor_398, wait_tensor_397, wait_tensor_396, wait_tensor_395, wait_tensor_394, wait_tensor_393, wait_tensor_392, wait_tensor_391, wait_tensor_390, wait_tensor_389, wait_tensor_388, wait_tensor_387, wait_tensor_386, wait_tensor_385, wait_tensor_384, wait_tensor_383, wait_tensor_382, wait_tensor_381, wait_tensor_380, wait_tensor_379, wait_tensor_378, wait_tensor_377, wait_tensor_376, wait_tensor_375, wait_tensor_374, wait_tensor_373, wait_tensor_372, wait_tensor_371, wait_tensor_370, wait_tensor_369, wait_tensor_368, wait_tensor_367, wait_tensor_366, wait_tensor_365, wait_tensor_364, wait_tensor_363, wait_tensor_362, wait_tensor_361, wait_tensor_360, wait_tensor_359, wait_tensor_358, wait_tensor_357, wait_tensor_356, wait_tensor_355, wait_tensor_354, wait_tensor_353, wait_tensor_352, wait_tensor_351, wait_tensor_350, wait_tensor_349, wait_tensor_348, wait_tensor_347, wait_tensor_346, wait_tensor_345, wait_tensor_344, wait_tensor_343, wait_tensor_342, wait_tensor_341, wait_tensor_340, wait_tensor_339, wait_tensor_338, wait_tensor_337, wait_tensor_336, wait_tensor_335, wait_tensor_334, wait_tensor_333, wait_tensor_332, wait_tensor_331, wait_tensor_330, wait_tensor_329, wait_tensor_328, wait_tensor_327, wait_tensor_326, wait_tensor_325, wait_tensor_324, wait_tensor_323, wait_tensor_322, wait_tensor_321, wait_tensor_320, wait_tensor_319, wait_tensor_318, wait_tensor_317, wait_tensor_316, wait_tensor_315, wait_tensor_314, wait_tensor_313, wait_tensor_312, wait_tensor_311, wait_tensor_310, wait_tensor_309, wait_tensor_308, wait_tensor_307, wait_tensor_306, wait_tensor_305, wait_tensor_304, wait_tensor_303, wait_tensor_302, wait_tensor_301, wait_tensor_300, wait_tensor_299, wait_tensor_298, wait_tensor_297, wait_tensor_296, wait_tensor_295, wait_tensor_294, wait_tensor_293, wait_tensor_292, wait_tensor_291)
        
def load_args(reader):
    buf0 = reader.storage(None, 8208384, device=device(type='cuda', index=0))
    reader.tensor(buf0, (501, 4096), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf2, (8192, 64), dtype=torch.complex64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16, 4096), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4, 4096), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (4, 4096), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (16, 4096), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf8, (16,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf9, (56, 4096), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf10, (56, 4096), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 14336), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16, 4096), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4, 4096), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (4, 4096), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (16, 4096), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf18, (56, 4096), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf19, (56, 4096), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf20, (16, 14336), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf21, (16,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf22, (16, 4096), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf23, (4, 4096), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf24, (4, 4096), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (16, 4096), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf26, (16,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf27, (56, 4096), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf28, (56, 4096), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf29, (16, 14336), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf30, (16,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf31, (16, 4096), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4, 4096), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4, 4096), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf34, (16, 4096), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf35, (16,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf36, (56, 4096), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf37, (56, 4096), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf38, (16, 14336), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf39, (16,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf40, (16, 4096), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (4, 4096), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf42, (4, 4096), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf43, (16, 4096), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf44, (16,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf45, (56, 4096), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf46, (56, 4096), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf47, (16, 14336), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf48, (16,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf49, (16, 4096), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf50, (4, 4096), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (4, 4096), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf52, (16, 4096), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf53, (16,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf54, (56, 4096), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf55, (56, 4096), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf56, (16, 14336), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf57, (16,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf58, (16, 4096), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf59, (4, 4096), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf60, (4, 4096), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf61, (16, 4096), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf62, (16,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf63, (56, 4096), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf64, (56, 4096), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf65, (16, 14336), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf66, (16,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (16, 4096), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf68, (4, 4096), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf69, (4, 4096), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf70, (16, 4096), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf71, (16,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf72, (56, 4096), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf73, (56, 4096), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf74, (16, 14336), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf75, (16,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf76, (16, 4096), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (4, 4096), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (4, 4096), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf79, (16, 4096), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf80, (16,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf81, (56, 4096), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf82, (56, 4096), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf83, (16, 14336), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf84, (16,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (16, 4096), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf86, (4, 4096), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (4, 4096), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf88, (16, 4096), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf89, (16,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf90, (56, 4096), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf91, (56, 4096), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf92, (16, 14336), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf93, (16,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf94, (16, 4096), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (4, 4096), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf96, (4, 4096), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf97, (16, 4096), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf98, (16,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf99, (56, 4096), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf100, (56, 4096), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf101, (16, 14336), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf102, (16,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf103, (16, 4096), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf104, (4, 4096), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (4, 4096), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf106, (16, 4096), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf107, (16,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf108, (56, 4096), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf109, (56, 4096), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf110, (16, 14336), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf111, (16,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf112, (16, 4096), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (4, 4096), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (4, 4096), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf115, (16, 4096), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf116, (16,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf117, (56, 4096), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf118, (56, 4096), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf119, (16, 14336), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf120, (16,), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf121, (16, 4096), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (4, 4096), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (4, 4096), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf124, (16, 4096), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf125, (16,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf126, (56, 4096), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf127, (56, 4096), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf128, (16, 14336), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf129, (16,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (16, 4096), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (4, 4096), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf132, (4, 4096), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf133, (16, 4096), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf134, (16,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf135, (56, 4096), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf136, (56, 4096), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf137, (16, 14336), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf138, (16,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (16, 4096), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (4, 4096), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf141, (4, 4096), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf142, (16, 4096), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf143, (16,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf144, (56, 4096), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf145, (56, 4096), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf146, (16, 14336), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf147, (16,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf148, (16, 4096), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf149, (4, 4096), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf150, (4, 4096), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf151, (16, 4096), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf152, (16,), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf153, (56, 4096), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf154, (56, 4096), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf155, (16, 14336), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf156, (16,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (16, 4096), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf158, (4, 4096), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (4, 4096), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf160, (16, 4096), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf161, (16,), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf162, (56, 4096), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf163, (56, 4096), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf164, (16, 14336), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf165, (16,), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf166, (16, 4096), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf167, (4, 4096), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf168, (4, 4096), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf169, (16, 4096), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf170, (16,), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf171, (56, 4096), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf172, (56, 4096), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf173, (16, 14336), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf174, (16,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf175, (16, 4096), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4, 4096), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (4, 4096), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf178, (16, 4096), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf179, (16,), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf180, (56, 4096), is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf181, (56, 4096), is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf182, (16, 14336), is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf183, (16,), is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf184, (16, 4096), is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf185, (4, 4096), is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf186, (4, 4096), is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf187, (16, 4096), is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf188, (16,), is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf189, (56, 4096), is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf190, (56, 4096), is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf191, (16, 14336), is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf192, (16,), is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (16, 4096), is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4, 4096), is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf195, (4, 4096), is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf196, (16, 4096), is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf197, (16,), is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf198, (56, 4096), is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf199, (56, 4096), is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf200, (16, 14336), is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf201, (16,), is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf202, (16, 4096), is_leaf=True)  # primals_203
    buf203 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf203, (4, 4096), is_leaf=True)  # primals_204
    buf204 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf204, (4, 4096), is_leaf=True)  # primals_205
    buf205 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (16, 4096), is_leaf=True)  # primals_206
    buf206 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf206, (16,), is_leaf=True)  # primals_207
    buf207 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf207, (56, 4096), is_leaf=True)  # primals_208
    buf208 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf208, (56, 4096), is_leaf=True)  # primals_209
    buf209 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf209, (16, 14336), is_leaf=True)  # primals_210
    buf210 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf210, (16,), is_leaf=True)  # primals_211
    buf211 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf211, (16, 4096), is_leaf=True)  # primals_212
    buf212 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf212, (4, 4096), is_leaf=True)  # primals_213
    buf213 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf213, (4, 4096), is_leaf=True)  # primals_214
    buf214 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf214, (16, 4096), is_leaf=True)  # primals_215
    buf215 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf215, (16,), is_leaf=True)  # primals_216
    buf216 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf216, (56, 4096), is_leaf=True)  # primals_217
    buf217 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf217, (56, 4096), is_leaf=True)  # primals_218
    buf218 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf218, (16, 14336), is_leaf=True)  # primals_219
    buf219 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf219, (16,), is_leaf=True)  # primals_220
    buf220 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf220, (16, 4096), is_leaf=True)  # primals_221
    buf221 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf221, (4, 4096), is_leaf=True)  # primals_222
    buf222 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf222, (4, 4096), is_leaf=True)  # primals_223
    buf223 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (16, 4096), is_leaf=True)  # primals_224
    buf224 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf224, (16,), is_leaf=True)  # primals_225
    buf225 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf225, (56, 4096), is_leaf=True)  # primals_226
    buf226 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf226, (56, 4096), is_leaf=True)  # primals_227
    buf227 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf227, (16, 14336), is_leaf=True)  # primals_228
    buf228 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf228, (16,), is_leaf=True)  # primals_229
    buf229 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf229, (16, 4096), is_leaf=True)  # primals_230
    buf230 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf230, (4, 4096), is_leaf=True)  # primals_231
    buf231 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf231, (4, 4096), is_leaf=True)  # primals_232
    buf232 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf232, (16, 4096), is_leaf=True)  # primals_233
    buf233 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf233, (16,), is_leaf=True)  # primals_234
    buf234 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf234, (56, 4096), is_leaf=True)  # primals_235
    buf235 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf235, (56, 4096), is_leaf=True)  # primals_236
    buf236 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf236, (16, 14336), is_leaf=True)  # primals_237
    buf237 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf237, (16,), is_leaf=True)  # primals_238
    buf238 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf238, (16, 4096), is_leaf=True)  # primals_239
    buf239 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf239, (4, 4096), is_leaf=True)  # primals_240
    buf240 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf240, (4, 4096), is_leaf=True)  # primals_241
    buf241 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf241, (16, 4096), is_leaf=True)  # primals_242
    buf242 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf242, (16,), is_leaf=True)  # primals_243
    buf243 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf243, (56, 4096), is_leaf=True)  # primals_244
    buf244 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf244, (56, 4096), is_leaf=True)  # primals_245
    buf245 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf245, (16, 14336), is_leaf=True)  # primals_246
    buf246 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf246, (16,), is_leaf=True)  # primals_247
    buf247 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf247, (16, 4096), is_leaf=True)  # primals_248
    buf248 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf248, (4, 4096), is_leaf=True)  # primals_249
    buf249 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf249, (4, 4096), is_leaf=True)  # primals_250
    buf250 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf250, (16, 4096), is_leaf=True)  # primals_251
    buf251 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf251, (16,), is_leaf=True)  # primals_252
    buf252 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf252, (56, 4096), is_leaf=True)  # primals_253
    buf253 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf253, (56, 4096), is_leaf=True)  # primals_254
    buf254 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf254, (16, 14336), is_leaf=True)  # primals_255
    buf255 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf255, (16,), is_leaf=True)  # primals_256
    buf256 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf256, (16, 4096), is_leaf=True)  # primals_257
    buf257 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf257, (4, 4096), is_leaf=True)  # primals_258
    buf258 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf258, (4, 4096), is_leaf=True)  # primals_259
    buf259 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf259, (16, 4096), is_leaf=True)  # primals_260
    buf260 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf260, (16,), is_leaf=True)  # primals_261
    buf261 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf261, (56, 4096), is_leaf=True)  # primals_262
    buf262 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf262, (56, 4096), is_leaf=True)  # primals_263
    buf263 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf263, (16, 14336), is_leaf=True)  # primals_264
    buf264 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf264, (16,), is_leaf=True)  # primals_265
    buf265 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf265, (16, 4096), is_leaf=True)  # primals_266
    buf266 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf266, (4, 4096), is_leaf=True)  # primals_267
    buf267 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf267, (4, 4096), is_leaf=True)  # primals_268
    buf268 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf268, (16, 4096), is_leaf=True)  # primals_269
    buf269 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf269, (16,), is_leaf=True)  # primals_270
    buf270 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf270, (56, 4096), is_leaf=True)  # primals_271
    buf271 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf271, (56, 4096), is_leaf=True)  # primals_272
    buf272 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf272, (16, 14336), is_leaf=True)  # primals_273
    buf273 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf273, (16,), is_leaf=True)  # primals_274
    buf274 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf274, (16, 4096), is_leaf=True)  # primals_275
    buf275 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf275, (4, 4096), is_leaf=True)  # primals_276
    buf276 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf276, (4, 4096), is_leaf=True)  # primals_277
    buf277 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf277, (16, 4096), is_leaf=True)  # primals_278
    buf278 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf278, (16,), is_leaf=True)  # primals_279
    buf279 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf279, (56, 4096), is_leaf=True)  # primals_280
    buf280 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf280, (56, 4096), is_leaf=True)  # primals_281
    buf281 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf281, (16, 14336), is_leaf=True)  # primals_282
    buf282 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf282, (16,), is_leaf=True)  # primals_283
    buf283 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf283, (16, 4096), is_leaf=True)  # primals_284
    buf284 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf284, (4, 4096), is_leaf=True)  # primals_285
    buf285 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf285, (4, 4096), is_leaf=True)  # primals_286
    buf286 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf286, (16, 4096), is_leaf=True)  # primals_287
    buf287 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf287, (16,), is_leaf=True)  # primals_288
    buf288 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf288, (56, 4096), is_leaf=True)  # primals_289
    buf289 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf289, (56, 4096), is_leaf=True)  # primals_290
    buf290 = reader.storage(None, 917504, device=device(type='cuda', index=0))
    reader.tensor(buf290, (16, 14336), is_leaf=True)  # primals_291
    buf291 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf291, (16,), is_leaf=True)  # primals_292
    buf292 = reader.storage(None, 8208384, device=device(type='cuda', index=0))
    reader.tensor(buf292, (501, 4096), is_leaf=True)  # primals_293
    buf293 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf293, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # embedding
    buf294 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf294, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf295 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf295, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf296 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf296, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem
    buf297 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf297, (2, 32, 8192, 1), is_leaf=True)  # getitem_1
    buf298 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf298, (), dtype=torch.int64, is_leaf=True)  # getitem_6
    buf299 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf299, (), dtype=torch.int64, is_leaf=True)  # getitem_7
    buf300 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf300, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf301 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf301, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_3
    buf302 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf302, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_7
    buf303 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf303, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_9
    buf304 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf304, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_9
    buf305 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf305, (2, 32, 8192, 1), is_leaf=True)  # getitem_10
    buf306 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf306, (), dtype=torch.int64, is_leaf=True)  # getitem_15
    buf307 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf307, (), dtype=torch.int64, is_leaf=True)  # getitem_16
    buf308 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf308, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_11
    buf309 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf309, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_7
    buf310 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf310, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_14
    buf311 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf311, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_16
    buf312 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf312, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_18
    buf313 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf313, (2, 32, 8192, 1), is_leaf=True)  # getitem_19
    buf314 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf314, (), dtype=torch.int64, is_leaf=True)  # getitem_24
    buf315 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf315, (), dtype=torch.int64, is_leaf=True)  # getitem_25
    buf316 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf316, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_18
    buf317 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf317, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_11
    buf318 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf318, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_21
    buf319 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf319, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_23
    buf320 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf320, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_27
    buf321 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf321, (2, 32, 8192, 1), is_leaf=True)  # getitem_28
    buf322 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf322, (), dtype=torch.int64, is_leaf=True)  # getitem_33
    buf323 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf323, (), dtype=torch.int64, is_leaf=True)  # getitem_34
    buf324 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf324, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_25
    buf325 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf325, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_15
    buf326 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf326, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_28
    buf327 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf327, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_30
    buf328 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf328, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_36
    buf329 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf329, (2, 32, 8192, 1), is_leaf=True)  # getitem_37
    buf330 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf330, (), dtype=torch.int64, is_leaf=True)  # getitem_42
    buf331 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf331, (), dtype=torch.int64, is_leaf=True)  # getitem_43
    buf332 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf332, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_32
    buf333 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf333, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_19
    buf334 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf334, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_35
    buf335 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf335, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_37
    buf336 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf336, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_45
    buf337 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf337, (2, 32, 8192, 1), is_leaf=True)  # getitem_46
    buf338 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf338, (), dtype=torch.int64, is_leaf=True)  # getitem_51
    buf339 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf339, (), dtype=torch.int64, is_leaf=True)  # getitem_52
    buf340 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf340, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_39
    buf341 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf341, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_23
    buf342 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf342, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_42
    buf343 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf343, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_44
    buf344 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf344, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_54
    buf345 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf345, (2, 32, 8192, 1), is_leaf=True)  # getitem_55
    buf346 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf346, (), dtype=torch.int64, is_leaf=True)  # getitem_60
    buf347 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf347, (), dtype=torch.int64, is_leaf=True)  # getitem_61
    buf348 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf348, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_46
    buf349 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf349, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_27
    buf350 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf350, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_49
    buf351 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf351, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_51
    buf352 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf352, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_63
    buf353 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf353, (2, 32, 8192, 1), is_leaf=True)  # getitem_64
    buf354 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf354, (), dtype=torch.int64, is_leaf=True)  # getitem_69
    buf355 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf355, (), dtype=torch.int64, is_leaf=True)  # getitem_70
    buf356 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf356, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_53
    buf357 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf357, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_31
    buf358 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf358, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_56
    buf359 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf359, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_58
    buf360 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf360, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_72
    buf361 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf361, (2, 32, 8192, 1), is_leaf=True)  # getitem_73
    buf362 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf362, (), dtype=torch.int64, is_leaf=True)  # getitem_78
    buf363 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf363, (), dtype=torch.int64, is_leaf=True)  # getitem_79
    buf364 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf364, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_60
    buf365 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf365, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_35
    buf366 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf366, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_63
    buf367 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf367, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_65
    buf368 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf368, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_81
    buf369 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf369, (2, 32, 8192, 1), is_leaf=True)  # getitem_82
    buf370 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf370, (), dtype=torch.int64, is_leaf=True)  # getitem_87
    buf371 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf371, (), dtype=torch.int64, is_leaf=True)  # getitem_88
    buf372 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf372, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_67
    buf373 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf373, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_39
    buf374 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf374, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_70
    buf375 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf375, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_72
    buf376 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf376, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_90
    buf377 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf377, (2, 32, 8192, 1), is_leaf=True)  # getitem_91
    buf378 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf378, (), dtype=torch.int64, is_leaf=True)  # getitem_96
    buf379 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf379, (), dtype=torch.int64, is_leaf=True)  # getitem_97
    buf380 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf380, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_74
    buf381 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf381, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_43
    buf382 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf382, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_77
    buf383 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf383, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_79
    buf384 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf384, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_99
    buf385 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf385, (2, 32, 8192, 1), is_leaf=True)  # getitem_100
    buf386 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf386, (), dtype=torch.int64, is_leaf=True)  # getitem_105
    buf387 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf387, (), dtype=torch.int64, is_leaf=True)  # getitem_106
    buf388 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf388, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_81
    buf389 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf389, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_47
    buf390 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf390, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_84
    buf391 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf391, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_86
    buf392 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf392, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_108
    buf393 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf393, (2, 32, 8192, 1), is_leaf=True)  # getitem_109
    buf394 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf394, (), dtype=torch.int64, is_leaf=True)  # getitem_114
    buf395 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf395, (), dtype=torch.int64, is_leaf=True)  # getitem_115
    buf396 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf396, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_88
    buf397 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf397, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_51
    buf398 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf398, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_91
    buf399 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf399, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_93
    buf400 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf400, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_117
    buf401 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf401, (2, 32, 8192, 1), is_leaf=True)  # getitem_118
    buf402 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf402, (), dtype=torch.int64, is_leaf=True)  # getitem_123
    buf403 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf403, (), dtype=torch.int64, is_leaf=True)  # getitem_124
    buf404 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf404, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_95
    buf405 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf405, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_55
    buf406 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf406, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_98
    buf407 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf407, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_100
    buf408 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf408, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_126
    buf409 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf409, (2, 32, 8192, 1), is_leaf=True)  # getitem_127
    buf410 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf410, (), dtype=torch.int64, is_leaf=True)  # getitem_132
    buf411 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf411, (), dtype=torch.int64, is_leaf=True)  # getitem_133
    buf412 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf412, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_102
    buf413 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf413, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_59
    buf414 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf414, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_105
    buf415 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf415, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_107
    buf416 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf416, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_135
    buf417 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf417, (2, 32, 8192, 1), is_leaf=True)  # getitem_136
    buf418 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf418, (), dtype=torch.int64, is_leaf=True)  # getitem_141
    buf419 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf419, (), dtype=torch.int64, is_leaf=True)  # getitem_142
    buf420 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf420, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_109
    buf421 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf421, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_63
    buf422 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf422, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_112
    buf423 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf423, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_114
    buf424 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf424, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_144
    buf425 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf425, (2, 32, 8192, 1), is_leaf=True)  # getitem_145
    buf426 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf426, (), dtype=torch.int64, is_leaf=True)  # getitem_150
    buf427 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf427, (), dtype=torch.int64, is_leaf=True)  # getitem_151
    buf428 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf428, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_116
    buf429 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf429, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_67
    buf430 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf430, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_119
    buf431 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf431, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_121
    buf432 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf432, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_153
    buf433 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf433, (2, 32, 8192, 1), is_leaf=True)  # getitem_154
    buf434 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf434, (), dtype=torch.int64, is_leaf=True)  # getitem_159
    buf435 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf435, (), dtype=torch.int64, is_leaf=True)  # getitem_160
    buf436 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf436, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_123
    buf437 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf437, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_71
    buf438 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf438, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_126
    buf439 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf439, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_128
    buf440 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf440, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_162
    buf441 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf441, (2, 32, 8192, 1), is_leaf=True)  # getitem_163
    buf442 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf442, (), dtype=torch.int64, is_leaf=True)  # getitem_168
    buf443 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf443, (), dtype=torch.int64, is_leaf=True)  # getitem_169
    buf444 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf444, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_130
    buf445 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf445, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_75
    buf446 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf446, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_133
    buf447 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf447, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_135
    buf448 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf448, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_171
    buf449 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf449, (2, 32, 8192, 1), is_leaf=True)  # getitem_172
    buf450 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf450, (), dtype=torch.int64, is_leaf=True)  # getitem_177
    buf451 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf451, (), dtype=torch.int64, is_leaf=True)  # getitem_178
    buf452 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf452, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_137
    buf453 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf453, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_79
    buf454 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf454, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_140
    buf455 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf455, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_142
    buf456 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf456, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_180
    buf457 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf457, (2, 32, 8192, 1), is_leaf=True)  # getitem_181
    buf458 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf458, (), dtype=torch.int64, is_leaf=True)  # getitem_186
    buf459 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf459, (), dtype=torch.int64, is_leaf=True)  # getitem_187
    buf460 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf460, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_144
    buf461 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf461, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_83
    buf462 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf462, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_147
    buf463 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf463, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_149
    buf464 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf464, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_189
    buf465 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf465, (2, 32, 8192, 1), is_leaf=True)  # getitem_190
    buf466 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf466, (), dtype=torch.int64, is_leaf=True)  # getitem_195
    buf467 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf467, (), dtype=torch.int64, is_leaf=True)  # getitem_196
    buf468 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf468, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_151
    buf469 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf469, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_87
    buf470 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf470, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_154
    buf471 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf471, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_156
    buf472 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf472, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_198
    buf473 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf473, (2, 32, 8192, 1), is_leaf=True)  # getitem_199
    buf474 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf474, (), dtype=torch.int64, is_leaf=True)  # getitem_204
    buf475 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf475, (), dtype=torch.int64, is_leaf=True)  # getitem_205
    buf476 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf476, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_158
    buf477 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf477, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_91
    buf478 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf478, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_161
    buf479 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf479, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_163
    buf480 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf480, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_207
    buf481 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf481, (2, 32, 8192, 1), is_leaf=True)  # getitem_208
    buf482 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf482, (), dtype=torch.int64, is_leaf=True)  # getitem_213
    buf483 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf483, (), dtype=torch.int64, is_leaf=True)  # getitem_214
    buf484 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf484, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_165
    buf485 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf485, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_95
    buf486 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf486, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_168
    buf487 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf487, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_170
    buf488 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf488, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_216
    buf489 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf489, (2, 32, 8192, 1), is_leaf=True)  # getitem_217
    buf490 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf490, (), dtype=torch.int64, is_leaf=True)  # getitem_222
    buf491 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf491, (), dtype=torch.int64, is_leaf=True)  # getitem_223
    buf492 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf492, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_172
    buf493 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf493, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_99
    buf494 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf494, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_175
    buf495 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf495, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_177
    buf496 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf496, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_225
    buf497 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf497, (2, 32, 8192, 1), is_leaf=True)  # getitem_226
    buf498 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf498, (), dtype=torch.int64, is_leaf=True)  # getitem_231
    buf499 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf499, (), dtype=torch.int64, is_leaf=True)  # getitem_232
    buf500 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf500, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_179
    buf501 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf501, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_103
    buf502 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf502, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_182
    buf503 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf503, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_184
    buf504 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf504, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_234
    buf505 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf505, (2, 32, 8192, 1), is_leaf=True)  # getitem_235
    buf506 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf506, (), dtype=torch.int64, is_leaf=True)  # getitem_240
    buf507 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf507, (), dtype=torch.int64, is_leaf=True)  # getitem_241
    buf508 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf508, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_186
    buf509 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf509, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_107
    buf510 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf510, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_189
    buf511 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf511, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_191
    buf512 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf512, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_243
    buf513 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf513, (2, 32, 8192, 1), is_leaf=True)  # getitem_244
    buf514 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf514, (), dtype=torch.int64, is_leaf=True)  # getitem_249
    buf515 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf515, (), dtype=torch.int64, is_leaf=True)  # getitem_250
    buf516 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf516, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_193
    buf517 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf517, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_111
    buf518 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf518, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_196
    buf519 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf519, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_198
    buf520 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf520, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_252
    buf521 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf521, (2, 32, 8192, 1), is_leaf=True)  # getitem_253
    buf522 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf522, (), dtype=torch.int64, is_leaf=True)  # getitem_258
    buf523 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf523, (), dtype=torch.int64, is_leaf=True)  # getitem_259
    buf524 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf524, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_200
    buf525 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf525, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_115
    buf526 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf526, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_203
    buf527 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf527, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_205
    buf528 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf528, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_261
    buf529 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf529, (2, 32, 8192, 1), is_leaf=True)  # getitem_262
    buf530 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf530, (), dtype=torch.int64, is_leaf=True)  # getitem_267
    buf531 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf531, (), dtype=torch.int64, is_leaf=True)  # getitem_268
    buf532 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf532, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_207
    buf533 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf533, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_119
    buf534 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf534, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_210
    buf535 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf535, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_212
    buf536 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf536, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_270
    buf537 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf537, (2, 32, 8192, 1), is_leaf=True)  # getitem_271
    buf538 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf538, (), dtype=torch.int64, is_leaf=True)  # getitem_276
    buf539 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf539, (), dtype=torch.int64, is_leaf=True)  # getitem_277
    buf540 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf540, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_214
    buf541 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf541, (2, 8192, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_123
    buf542 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf542, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_217
    buf543 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf543, (16384, 1024), dtype=torch.bfloat16, is_leaf=True)  # mm_219
    buf544 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf544, (2, 32, 8192, 128), (33554432, 128, 4096, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_279
    buf545 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf545, (2, 32, 8192, 1), is_leaf=True)  # getitem_280
    buf546 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf546, (), dtype=torch.int64, is_leaf=True)  # getitem_285
    buf547 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf547, (), dtype=torch.int64, is_leaf=True)  # getitem_286
    buf548 = reader.storage(None, 469762048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf548, (16384, 14336), dtype=torch.bfloat16, is_leaf=True)  # mm_221
    buf549 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf549, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # mm_223
    buf550 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf550, (2, 8192, 1), is_leaf=True)  # rsqrt_64
    buf551 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf551, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # view_1091
    buf552 = reader.storage(None, 4202692608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf552, (2, 8192, 128256), dtype=torch.bfloat16, is_leaf=True)  # tangents_1

load_args._version = 0

def get_pg_config():
    return {'0': {'size': 256, 'rank': 0}}

def get_colls_estimations_file():
    return "colls32_8.table"

