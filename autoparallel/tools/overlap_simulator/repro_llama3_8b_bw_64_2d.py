import torch
from torch.nn import *
from torch import tensor, device


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, wait_tensor_1, mm, mm_2, getitem_80, getitem_81, getitem_86, getitem_87, reduce_scatter_tensor_1, mm_4, add_3, mm_7, mm_9, getitem_121, getitem_122, getitem_127, getitem_128, reduce_scatter_tensor_3, mm_11, add_7, mm_14, mm_16, getitem_162, getitem_163, getitem_168, getitem_169, reduce_scatter_tensor_5, mm_18, add_11, mm_21, mm_23, getitem_203, getitem_204, getitem_209, getitem_210, reduce_scatter_tensor_7, mm_25, add_15, mm_28, mm_30, getitem_244, getitem_245, getitem_250, getitem_251, reduce_scatter_tensor_9, mm_32, add_19, mm_35, mm_37, getitem_285, getitem_286, getitem_291, getitem_292, reduce_scatter_tensor_11, mm_39, add_23, mm_42, mm_44, getitem_326, getitem_327, getitem_332, getitem_333, reduce_scatter_tensor_13, mm_46, add_27, mm_49, mm_51, getitem_367, getitem_368, getitem_373, getitem_374, reduce_scatter_tensor_15, mm_53, add_31, mm_56, mm_58, getitem_408, getitem_409, getitem_414, getitem_415, reduce_scatter_tensor_17, mm_60, add_35, mm_63, mm_65, getitem_449, getitem_450, getitem_455, getitem_456, reduce_scatter_tensor_19, mm_67, add_39, mm_70, mm_72, getitem_490, getitem_491, getitem_496, getitem_497, reduce_scatter_tensor_21, mm_74, add_43, mm_77, mm_79, getitem_531, getitem_532, getitem_537, getitem_538, reduce_scatter_tensor_23, mm_81, add_47, mm_84, mm_86, getitem_572, getitem_573, getitem_578, getitem_579, reduce_scatter_tensor_25, mm_88, add_51, mm_91, mm_93, getitem_613, getitem_614, getitem_619, getitem_620, reduce_scatter_tensor_27, mm_95, add_55, mm_98, mm_100, getitem_654, getitem_655, getitem_660, getitem_661, reduce_scatter_tensor_29, mm_102, add_59, mm_105, mm_107, getitem_695, getitem_696, getitem_701, getitem_702, reduce_scatter_tensor_31, mm_109, reduce_scatter_tensor_32, rsqrt_32, view_1167, tangents_1):
        view_1169 = torch.ops.aten.view.default(tangents_1, [16384, 16032]);  tangents_1 = None
        permute_177 = torch.ops.aten.permute.default(view_1169, [1, 0])
        mm_113 = torch.ops.aten.mm.default(permute_177, view_1167);  permute_177 = view_1167 = None
        convert_element_type_532 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        all_gather_into_tensor_179 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_532, 8, '0');  convert_element_type_532 = None
        wait_tensor_212 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_179);  all_gather_into_tensor_179 = None
        permute_176 = torch.ops.aten.permute.default(wait_tensor_212, [1, 0]);  wait_tensor_212 = None
        permute_179 = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_114 = torch.ops.aten.mm.default(view_1169, permute_179);  view_1169 = permute_179 = None
        view_1170 = torch.ops.aten.view.default(mm_114, [2, 8192, 4096]);  mm_114 = None
        convert_element_type_539 = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        reduce_scatter_tensor_33 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_539, 'avg', 8, '0');  convert_element_type_539 = None
        wait_tensor_213 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_33);  reduce_scatter_tensor_33 = None
        split_74 = torch.ops.aten.split.Tensor(view_1170, 1024, 1);  view_1170 = None
        getitem_736 = split_74[0]
        getitem_737 = split_74[1]
        getitem_738 = split_74[2]
        getitem_739 = split_74[3]
        getitem_740 = split_74[4]
        getitem_741 = split_74[5]
        getitem_742 = split_74[6]
        getitem_743 = split_74[7];  split_74 = None
        cat_66 = torch.ops.aten.cat.default([getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743]);  getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = None
        reduce_scatter_tensor_34 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_66, 'sum', 8, '1');  cat_66 = None
        wait_tensor_214 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_34);  reduce_scatter_tensor_34 = None
        convert_element_type_540 = torch.ops.prims.convert_element_type.default(wait_tensor_214, torch.float32);  wait_tensor_214 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        all_gather_into_tensor_177 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_529, 8, '0');  convert_element_type_529 = None
        wait_tensor_210 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_177);  all_gather_into_tensor_177 = None
        convert_element_type_542 = torch.ops.prims.convert_element_type.default(wait_tensor_210, torch.float32);  wait_tensor_210 = None
        mul_130 = torch.ops.aten.mul.Tensor(convert_element_type_540, convert_element_type_542);  convert_element_type_542 = None
        wait_tensor_203 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_31);  reduce_scatter_tensor_31 = None
        add_61 = torch.ops.aten.add.Tensor(add_59, wait_tensor_203);  wait_tensor_203 = None
        wait_tensor_209 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_32);  reduce_scatter_tensor_32 = None
        add_63 = torch.ops.aten.add.Tensor(add_61, wait_tensor_209);  wait_tensor_209 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_530, rsqrt_32);  convert_element_type_530 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_128, mul_130)
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
        div = torch.ops.aten.div.Tensor(mul_128, 4096)
        mul_133 = torch.ops.aten.mul.Tensor(div, sum_1);  div = sum_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_130, mul_133);  mul_130 = mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_32);  sub_1 = rsqrt_32 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_540, mul_128);  convert_element_type_540 = mul_128 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
        convert_element_type_543 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        all_reduce = torch.ops._c10d_functional.all_reduce.default(convert_element_type_544, 'sum', '1');  convert_element_type_544 = None
        wait_tensor_215 = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        convert_element_type_545 = torch.ops.prims.convert_element_type.default(wait_tensor_215, torch.float32);  wait_tensor_215 = None
        reduce_scatter_tensor_35 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_545, 'avg', 8, '0');  convert_element_type_545 = None
        wait_tensor_216 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_35);  reduce_scatter_tensor_35 = None
        all_gather_into_tensor_180 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_543, 8, '1')
        wait_tensor_217 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_180);  all_gather_into_tensor_180 = None
        split_75 = torch.ops.aten.split.Tensor(wait_tensor_217, 2);  wait_tensor_217 = None
        getitem_744 = split_75[0]
        getitem_745 = split_75[1]
        getitem_746 = split_75[2]
        getitem_747 = split_75[3]
        getitem_748 = split_75[4]
        getitem_749 = split_75[5]
        getitem_750 = split_75[6]
        getitem_751 = split_75[7];  split_75 = None
        cat_67 = torch.ops.aten.cat.default([getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751], 1);  getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = None
        view_1171 = torch.ops.aten.view.default(cat_67, [16384, 4096]);  cat_67 = None
        permute_181 = torch.ops.aten.permute.default(view_1171, [1, 0])
        convert_element_type_515 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        all_gather_into_tensor_172 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_515, 8, '0');  convert_element_type_515 = None
        wait_tensor_204 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_172);  all_gather_into_tensor_172 = None
        convert_element_type_516 = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_516, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [2], True);  pow_32 = None
        add_62 = torch.ops.aten.add.Scalar(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_516, rsqrt_31);  convert_element_type_516 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, wait_tensor_204)
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
        view_1140 = torch.ops.aten.view.default(cat_63, [16384, 4096]);  cat_63 = None
        view_1141 = torch.ops.aten.view.default(mm_109, [2, 8192, 1792]);  mm_109 = None
        convert_element_type_521 = torch.ops.prims.convert_element_type.default(view_1141, torch.float32);  view_1141 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_521)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_521, sigmoid_15);  sigmoid_15 = None
        convert_element_type_522 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        all_gather_into_tensor_175 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_523, 8, '0');  convert_element_type_523 = None
        wait_tensor_207 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_175);  all_gather_into_tensor_175 = None
        permute_174 = torch.ops.aten.permute.default(wait_tensor_207, [1, 0]);  wait_tensor_207 = None
        mm_110 = torch.ops.aten.mm.default(view_1140, permute_174)
        view_1148 = torch.ops.aten.view.default(mm_110, [2, 8192, 1792]);  mm_110 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_522, view_1148)
        view_1155 = torch.ops.aten.view.default(mul_127, [16384, 1792]);  mul_127 = None
        mm_115 = torch.ops.aten.mm.default(permute_181, view_1155);  permute_181 = view_1155 = None
        convert_element_type_526 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        all_gather_into_tensor_176 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_526, 8, '0');  convert_element_type_526 = None
        wait_tensor_208 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_176);  all_gather_into_tensor_176 = None
        permute_175 = torch.ops.aten.permute.default(wait_tensor_208, [1, 0]);  wait_tensor_208 = None
        permute_183 = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_116 = torch.ops.aten.mm.default(view_1171, permute_183);  view_1171 = permute_183 = None
        view_1172 = torch.ops.aten.view.default(mm_116, [2, 8192, 1792]);  mm_116 = None
        convert_element_type_550 = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        reduce_scatter_tensor_36 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_550, 'avg', 8, '0');  convert_element_type_550 = None
        wait_tensor_218 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_36);  reduce_scatter_tensor_36 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_1172, convert_element_type_522);  convert_element_type_522 = None
        mul_137 = torch.ops.aten.mul.Tensor(view_1172, view_1148);  view_1172 = view_1148 = None
        view_1173 = torch.ops.aten.view.default(mul_136, [16384, 1792]);  mul_136 = None
        permute_185 = torch.ops.aten.permute.default(view_1173, [1, 0])
        mm_117 = torch.ops.aten.mm.default(permute_185, view_1140);  permute_185 = None
        permute_187 = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_118 = torch.ops.aten.mm.default(view_1173, permute_187);  view_1173 = permute_187 = None
        view_1174 = torch.ops.aten.view.default(mm_118, [2, 8192, 4096]);  mm_118 = None
        convert_element_type_555 = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        reduce_scatter_tensor_37 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_555, 'avg', 8, '0');  convert_element_type_555 = None
        wait_tensor_219 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_37);  reduce_scatter_tensor_37 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(mul_137, torch.float32);  mul_137 = None
        neg = torch.ops.aten.neg.default(convert_element_type_521)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add_65 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal = torch.ops.aten.reciprocal.default(add_65);  add_65 = None
        mul_138 = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_556, mul_138);  convert_element_type_556 = None
        sub_2 = torch.ops.aten.sub.Tensor(1, mul_138);  mul_138 = None
        mul_140 = torch.ops.aten.mul.Tensor(convert_element_type_521, sub_2);  convert_element_type_521 = sub_2 = None
        add_66 = torch.ops.aten.add.Tensor(mul_140, 1);  mul_140 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, add_66);  mul_139 = add_66 = None
        convert_element_type_558 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        view_1175 = torch.ops.aten.view.default(convert_element_type_558, [16384, 1792]);  convert_element_type_558 = None
        permute_189 = torch.ops.aten.permute.default(view_1175, [1, 0])
        mm_119 = torch.ops.aten.mm.default(permute_189, view_1140);  permute_189 = view_1140 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        all_gather_into_tensor_174 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_518, 8, '0');  convert_element_type_518 = None
        wait_tensor_206 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_174);  all_gather_into_tensor_174 = None
        permute_173 = torch.ops.aten.permute.default(wait_tensor_206, [1, 0]);  wait_tensor_206 = None
        permute_191 = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_120 = torch.ops.aten.mm.default(view_1175, permute_191);  view_1175 = permute_191 = None
        view_1176 = torch.ops.aten.view.default(mm_120, [2, 8192, 4096]);  mm_120 = None
        add_67 = torch.ops.aten.add.Tensor(view_1174, view_1176);  view_1174 = view_1176 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        reduce_scatter_tensor_38 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_563, 'avg', 8, '0');  convert_element_type_563 = None
        wait_tensor_220 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_38);  reduce_scatter_tensor_38 = None
        split_76 = torch.ops.aten.split.Tensor(add_67, 1024, 1);  add_67 = None
        getitem_752 = split_76[0]
        getitem_753 = split_76[1]
        getitem_754 = split_76[2]
        getitem_755 = split_76[3]
        getitem_756 = split_76[4]
        getitem_757 = split_76[5]
        getitem_758 = split_76[6]
        getitem_759 = split_76[7];  split_76 = None
        cat_68 = torch.ops.aten.cat.default([getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759]);  getitem_752 = getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = None
        reduce_scatter_tensor_39 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_68, 'sum', 8, '1');  cat_68 = None
        wait_tensor_221 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_39);  reduce_scatter_tensor_39 = None
        convert_element_type_564 = torch.ops.prims.convert_element_type.default(wait_tensor_221, torch.float32);  wait_tensor_221 = None
        convert_element_type_566 = torch.ops.prims.convert_element_type.default(wait_tensor_204, torch.float32);  wait_tensor_204 = None
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_564, convert_element_type_566);  convert_element_type_566 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_124, mul_142)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
        div_1 = torch.ops.aten.div.Tensor(mul_124, 4096)
        mul_145 = torch.ops.aten.mul.Tensor(div_1, sum_3);  div_1 = sum_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_142, mul_145);  mul_142 = mul_145 = None
        mul_146 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_31);  sub_3 = rsqrt_31 = None
        mul_147 = torch.ops.aten.mul.Tensor(convert_element_type_564, mul_124);  convert_element_type_564 = mul_124 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
        convert_element_type_567 = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None
        convert_element_type_568 = torch.ops.prims.convert_element_type.default(sum_4, torch.bfloat16);  sum_4 = None
        all_reduce_1 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_568, 'sum', '1');  convert_element_type_568 = None
        wait_tensor_222 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        convert_element_type_569 = torch.ops.prims.convert_element_type.default(wait_tensor_222, torch.float32);  wait_tensor_222 = None
        reduce_scatter_tensor_40 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_569, 'avg', 8, '0');  convert_element_type_569 = None
        wait_tensor_223 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_40);  reduce_scatter_tensor_40 = None
        add_68 = torch.ops.aten.add.Tensor(convert_element_type_543, convert_element_type_567);  convert_element_type_543 = convert_element_type_567 = None
        all_gather_into_tensor_181 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_68, 8, '1')
        wait_tensor_224 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_181);  all_gather_into_tensor_181 = None
        split_77 = torch.ops.aten.split.Tensor(wait_tensor_224, 2);  wait_tensor_224 = None
        getitem_760 = split_77[0]
        getitem_761 = split_77[1]
        getitem_762 = split_77[2]
        getitem_763 = split_77[3]
        getitem_764 = split_77[4]
        getitem_765 = split_77[5]
        getitem_766 = split_77[6]
        getitem_767 = split_77[7];  split_77 = None
        cat_69 = torch.ops.aten.cat.default([getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767], 1);  getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = None
        view_1177 = torch.ops.aten.view.default(cat_69, [16384, 4096]);  cat_69 = None
        permute_193 = torch.ops.aten.permute.default(view_1177, [1, 0])
        permute_171 = torch.ops.aten.permute.default(getitem_695, [0, 2, 1, 3])
        view_1122 = torch.ops.aten.view.default(permute_171, [2, 8192, -1]);  permute_171 = None
        view_1128 = torch.ops.aten.view.default(view_1122, [16384, 512]);  view_1122 = None
        mm_121 = torch.ops.aten.mm.default(permute_193, view_1128);  permute_193 = view_1128 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        all_gather_into_tensor_171 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_512, 8, '0');  convert_element_type_512 = None
        wait_tensor_202 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_171);  all_gather_into_tensor_171 = None
        permute_172 = torch.ops.aten.permute.default(wait_tensor_202, [1, 0]);  wait_tensor_202 = None
        permute_195 = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        mm_122 = torch.ops.aten.mm.default(view_1177, permute_195);  view_1177 = permute_195 = None
        view_1178 = torch.ops.aten.view.default(mm_122, [2, 8192, 512]);  mm_122 = None
        convert_element_type_574 = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        reduce_scatter_tensor_41 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_574, 'avg', 8, '0');  convert_element_type_574 = None
        wait_tensor_225 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_41);  reduce_scatter_tensor_41 = None
        view_1179 = torch.ops.aten.view.default(view_1178, [2, 8192, 4, 128]);  view_1178 = None
        permute_197 = torch.ops.aten.permute.default(view_1179, [0, 2, 1, 3]);  view_1179 = None
        view_37 = torch.ops.aten.view.default(primals_3, [1, 8192, 1, 64]);  primals_3 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        all_gather_into_tensor_166 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_496, 8, '0');  convert_element_type_496 = None
        wait_tensor_197 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_166);  all_gather_into_tensor_166 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_497, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [2], True);  pow_31 = None
        add_60 = torch.ops.aten.add.Scalar(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_497, rsqrt_30);  convert_element_type_497 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, wait_tensor_197)
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
        view_1095 = torch.ops.aten.view.default(cat_61, [16384, 4096]);  cat_61 = None
        view_1096 = torch.ops.aten.view.default(mm_105, [2, 8192, 512]);  mm_105 = None
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        all_gather_into_tensor_169 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_502, 8, '0');  convert_element_type_502 = None
        wait_tensor_200 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_169);  all_gather_into_tensor_169 = None
        permute_166 = torch.ops.aten.permute.default(wait_tensor_200, [1, 0]);  wait_tensor_200 = None
        mm_106 = torch.ops.aten.mm.default(view_1095, permute_166)
        view_1103 = torch.ops.aten.view.default(mm_106, [2, 8192, 128]);  mm_106 = None
        view_1110 = torch.ops.aten.view.default(mm_107, [2, 8192, 128]);  mm_107 = None
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
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_197, permute_168, permute_169, permute_170, getitem_695, getitem_696, getitem_701, getitem_702, None, None, None, 8192, 8192, 0.0, True);  permute_197 = permute_168 = permute_169 = permute_170 = getitem_695 = getitem_696 = getitem_701 = getitem_702 = None
        getitem_768 = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_769 = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_770 = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None
        permute_198 = torch.ops.aten.permute.default(getitem_770, [0, 2, 1, 3]);  getitem_770 = None
        permute_199 = torch.ops.aten.permute.default(getitem_769, [0, 2, 1, 3]);  getitem_769 = None
        permute_200 = torch.ops.aten.permute.default(getitem_768, [0, 2, 1, 3]);  getitem_768 = None
        view_1180 = torch.ops.aten.view.default(permute_198, [2, 8192, 1, 4, 128]);  permute_198 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_1180, [3], True);  view_1180 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_5, 3);  sum_5 = None
        view_1181 = torch.ops.aten.view.default(permute_199, [2, 8192, 1, 4, 128]);  permute_199 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_1181, [3], True);  view_1181 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_6, 3);  sum_6 = None
        convert_element_type_575 = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_576 = torch.ops.prims.convert_element_type.default(permute_200, torch.float32);  permute_200 = None
        view_1182 = torch.ops.aten.view.default(convert_element_type_575, [2, 8192, 1, 64, 2]);  convert_element_type_575 = None
        view_as_complex_32 = torch.ops.aten.view_as_complex.default(view_1182);  view_1182 = None
        _conj = torch.ops.aten._conj.default(view_37)
        mul_148 = torch.ops.aten.mul.Tensor(view_as_complex_32, _conj);  view_as_complex_32 = None
        view_1183 = torch.ops.aten.view.default(convert_element_type_576, [2, 8192, 4, 64, 2]);  convert_element_type_576 = None
        view_as_complex_33 = torch.ops.aten.view_as_complex.default(view_1183);  view_1183 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_as_complex_33, _conj);  view_as_complex_33 = None
        view_as_real_32 = torch.ops.aten.view_as_real.default(mul_148);  mul_148 = None
        view_1184 = torch.ops.aten.view.default(view_as_real_32, [2, 8192, 1, 128]);  view_as_real_32 = None
        convert_element_type_577 = torch.ops.prims.convert_element_type.default(view_1184, torch.bfloat16);  view_1184 = None
        view_as_real_33 = torch.ops.aten.view_as_real.default(mul_149);  mul_149 = None
        view_1185 = torch.ops.aten.view.default(view_as_real_33, [2, 8192, 4, 128]);  view_as_real_33 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(view_1185, torch.bfloat16);  view_1185 = None
        view_1186 = torch.ops.aten.view.default(squeeze, [2, 8192, 128]);  squeeze = None
        view_1187 = torch.ops.aten.view.default(convert_element_type_577, [2, 8192, 128]);  convert_element_type_577 = None
        view_1188 = torch.ops.aten.view.default(convert_element_type_578, [2, 8192, 512]);  convert_element_type_578 = None
        view_1189 = torch.ops.aten.view.default(view_1186, [16384, 128]);  view_1186 = None
        permute_201 = torch.ops.aten.permute.default(view_1189, [1, 0])
        mm_123 = torch.ops.aten.mm.default(permute_201, view_1095);  permute_201 = None
        convert_element_type_505 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        all_gather_into_tensor_170 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_505, 8, '0');  convert_element_type_505 = None
        wait_tensor_201 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_170);  all_gather_into_tensor_170 = None
        permute_167 = torch.ops.aten.permute.default(wait_tensor_201, [1, 0]);  wait_tensor_201 = None
        permute_203 = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_124 = torch.ops.aten.mm.default(view_1189, permute_203);  view_1189 = permute_203 = None
        view_1190 = torch.ops.aten.view.default(mm_124, [2, 8192, 4096]);  mm_124 = None
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        reduce_scatter_tensor_42 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_583, 'avg', 8, '0');  convert_element_type_583 = None
        wait_tensor_226 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_42);  reduce_scatter_tensor_42 = None
        view_1191 = torch.ops.aten.view.default(view_1187, [16384, 128]);  view_1187 = None
        permute_205 = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_125 = torch.ops.aten.mm.default(permute_205, view_1095);  permute_205 = None
        permute_207 = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_126 = torch.ops.aten.mm.default(view_1191, permute_207);  view_1191 = permute_207 = None
        view_1192 = torch.ops.aten.view.default(mm_126, [2, 8192, 4096]);  mm_126 = None
        add_69 = torch.ops.aten.add.Tensor(view_1190, view_1192);  view_1190 = view_1192 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        reduce_scatter_tensor_43 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_588, 'avg', 8, '0');  convert_element_type_588 = None
        wait_tensor_227 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_43);  reduce_scatter_tensor_43 = None
        view_1193 = torch.ops.aten.view.default(view_1188, [16384, 512]);  view_1188 = None
        permute_209 = torch.ops.aten.permute.default(view_1193, [1, 0])
        mm_127 = torch.ops.aten.mm.default(permute_209, view_1095);  permute_209 = view_1095 = None
        convert_element_type_499 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        all_gather_into_tensor_168 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_499, 8, '0');  convert_element_type_499 = None
        wait_tensor_199 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_168);  all_gather_into_tensor_168 = None
        permute_165 = torch.ops.aten.permute.default(wait_tensor_199, [1, 0]);  wait_tensor_199 = None
        permute_211 = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_128 = torch.ops.aten.mm.default(view_1193, permute_211);  view_1193 = permute_211 = None
        view_1194 = torch.ops.aten.view.default(mm_128, [2, 8192, 4096]);  mm_128 = None
        add_70 = torch.ops.aten.add.Tensor(add_69, view_1194);  add_69 = view_1194 = None
        convert_element_type_593 = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        reduce_scatter_tensor_44 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_593, 'avg', 8, '0');  convert_element_type_593 = None
        wait_tensor_228 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_44);  reduce_scatter_tensor_44 = None
        split_78 = torch.ops.aten.split.Tensor(add_70, 1024, 1);  add_70 = None
        getitem_771 = split_78[0]
        getitem_772 = split_78[1]
        getitem_773 = split_78[2]
        getitem_774 = split_78[3]
        getitem_775 = split_78[4]
        getitem_776 = split_78[5]
        getitem_777 = split_78[6]
        getitem_778 = split_78[7];  split_78 = None
        cat_70 = torch.ops.aten.cat.default([getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778]);  getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = getitem_777 = getitem_778 = None
        reduce_scatter_tensor_45 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_70, 'sum', 8, '1');  cat_70 = None
        wait_tensor_229 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_45);  reduce_scatter_tensor_45 = None
        convert_element_type_594 = torch.ops.prims.convert_element_type.default(wait_tensor_229, torch.float32);  wait_tensor_229 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(wait_tensor_197, torch.float32);  wait_tensor_197 = None
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_594, convert_element_type_596);  convert_element_type_596 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_120, mul_150)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
        div_2 = torch.ops.aten.div.Tensor(mul_120, 4096)
        mul_153 = torch.ops.aten.mul.Tensor(div_2, sum_7);  div_2 = sum_7 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_150, mul_153);  mul_150 = mul_153 = None
        mul_154 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_30);  sub_4 = rsqrt_30 = None
        mul_155 = torch.ops.aten.mul.Tensor(convert_element_type_594, mul_120);  convert_element_type_594 = mul_120 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        convert_element_type_598 = torch.ops.prims.convert_element_type.default(sum_8, torch.bfloat16);  sum_8 = None
        all_reduce_2 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_598, 'sum', '1');  convert_element_type_598 = None
        wait_tensor_230 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        convert_element_type_599 = torch.ops.prims.convert_element_type.default(wait_tensor_230, torch.float32);  wait_tensor_230 = None
        reduce_scatter_tensor_46 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_599, 'avg', 8, '0');  convert_element_type_599 = None
        wait_tensor_231 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_46);  reduce_scatter_tensor_46 = None
        add_71 = torch.ops.aten.add.Tensor(add_68, convert_element_type_597);  add_68 = convert_element_type_597 = None
        all_gather_into_tensor_182 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_71, 8, '1')
        wait_tensor_232 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_182);  all_gather_into_tensor_182 = None
        split_79 = torch.ops.aten.split.Tensor(wait_tensor_232, 2);  wait_tensor_232 = None
        getitem_779 = split_79[0]
        getitem_780 = split_79[1]
        getitem_781 = split_79[2]
        getitem_782 = split_79[3]
        getitem_783 = split_79[4]
        getitem_784 = split_79[5]
        getitem_785 = split_79[6]
        getitem_786 = split_79[7];  split_79 = None
        cat_71 = torch.ops.aten.cat.default([getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786], 1);  getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = getitem_786 = None
        view_1195 = torch.ops.aten.view.default(cat_71, [16384, 4096]);  cat_71 = None
        permute_213 = torch.ops.aten.permute.default(view_1195, [1, 0])
        wait_tensor_190 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_29);  reduce_scatter_tensor_29 = None
        add_57 = torch.ops.aten.add.Tensor(add_55, wait_tensor_190);  wait_tensor_190 = None
        convert_element_type_482 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        all_gather_into_tensor_161 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_482, 8, '0');  convert_element_type_482 = None
        wait_tensor_191 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_161);  all_gather_into_tensor_161 = None
        convert_element_type_483 = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_483, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [2], True);  pow_30 = None
        add_58 = torch.ops.aten.add.Scalar(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_116 = torch.ops.aten.mul.Tensor(convert_element_type_483, rsqrt_29);  convert_element_type_483 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, wait_tensor_191)
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
        view_1068 = torch.ops.aten.view.default(cat_59, [16384, 4096]);  cat_59 = None
        view_1069 = torch.ops.aten.view.default(mm_102, [2, 8192, 1792]);  mm_102 = None
        convert_element_type_488 = torch.ops.prims.convert_element_type.default(view_1069, torch.float32);  view_1069 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_488)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_488, sigmoid_14);  sigmoid_14 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        all_gather_into_tensor_164 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_490, 8, '0');  convert_element_type_490 = None
        wait_tensor_194 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_164);  all_gather_into_tensor_164 = None
        permute_163 = torch.ops.aten.permute.default(wait_tensor_194, [1, 0]);  wait_tensor_194 = None
        mm_103 = torch.ops.aten.mm.default(view_1068, permute_163)
        view_1076 = torch.ops.aten.view.default(mm_103, [2, 8192, 1792]);  mm_103 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_489, view_1076)
        view_1083 = torch.ops.aten.view.default(mul_119, [16384, 1792]);  mul_119 = None
        mm_129 = torch.ops.aten.mm.default(permute_213, view_1083);  permute_213 = view_1083 = None
        convert_element_type_493 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        all_gather_into_tensor_165 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_493, 8, '0');  convert_element_type_493 = None
        wait_tensor_195 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_165);  all_gather_into_tensor_165 = None
        permute_164 = torch.ops.aten.permute.default(wait_tensor_195, [1, 0]);  wait_tensor_195 = None
        permute_215 = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_130 = torch.ops.aten.mm.default(view_1195, permute_215);  view_1195 = permute_215 = None
        view_1196 = torch.ops.aten.view.default(mm_130, [2, 8192, 1792]);  mm_130 = None
        convert_element_type_604 = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        reduce_scatter_tensor_47 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_604, 'avg', 8, '0');  convert_element_type_604 = None
        wait_tensor_233 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_47);  reduce_scatter_tensor_47 = None
        mul_156 = torch.ops.aten.mul.Tensor(view_1196, convert_element_type_489);  convert_element_type_489 = None
        mul_157 = torch.ops.aten.mul.Tensor(view_1196, view_1076);  view_1196 = view_1076 = None
        view_1197 = torch.ops.aten.view.default(mul_156, [16384, 1792]);  mul_156 = None
        permute_217 = torch.ops.aten.permute.default(view_1197, [1, 0])
        mm_131 = torch.ops.aten.mm.default(permute_217, view_1068);  permute_217 = None
        permute_219 = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_132 = torch.ops.aten.mm.default(view_1197, permute_219);  view_1197 = permute_219 = None
        view_1198 = torch.ops.aten.view.default(mm_132, [2, 8192, 4096]);  mm_132 = None
        convert_element_type_609 = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        reduce_scatter_tensor_48 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_609, 'avg', 8, '0');  convert_element_type_609 = None
        wait_tensor_234 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_48);  reduce_scatter_tensor_48 = None
        convert_element_type_610 = torch.ops.prims.convert_element_type.default(mul_157, torch.float32);  mul_157 = None
        neg_1 = torch.ops.aten.neg.default(convert_element_type_488)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_72 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(add_72);  add_72 = None
        mul_158 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_610, mul_158);  convert_element_type_610 = None
        sub_5 = torch.ops.aten.sub.Tensor(1, mul_158);  mul_158 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_488, sub_5);  convert_element_type_488 = sub_5 = None
        add_73 = torch.ops.aten.add.Tensor(mul_160, 1);  mul_160 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_159, add_73);  mul_159 = add_73 = None
        convert_element_type_612 = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        view_1199 = torch.ops.aten.view.default(convert_element_type_612, [16384, 1792]);  convert_element_type_612 = None
        permute_221 = torch.ops.aten.permute.default(view_1199, [1, 0])
        mm_133 = torch.ops.aten.mm.default(permute_221, view_1068);  permute_221 = view_1068 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        all_gather_into_tensor_163 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_485, 8, '0');  convert_element_type_485 = None
        wait_tensor_193 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_163);  all_gather_into_tensor_163 = None
        permute_162 = torch.ops.aten.permute.default(wait_tensor_193, [1, 0]);  wait_tensor_193 = None
        permute_223 = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_134 = torch.ops.aten.mm.default(view_1199, permute_223);  view_1199 = permute_223 = None
        view_1200 = torch.ops.aten.view.default(mm_134, [2, 8192, 4096]);  mm_134 = None
        add_74 = torch.ops.aten.add.Tensor(view_1198, view_1200);  view_1198 = view_1200 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        reduce_scatter_tensor_49 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_617, 'avg', 8, '0');  convert_element_type_617 = None
        wait_tensor_235 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_49);  reduce_scatter_tensor_49 = None
        split_80 = torch.ops.aten.split.Tensor(add_74, 1024, 1);  add_74 = None
        getitem_787 = split_80[0]
        getitem_788 = split_80[1]
        getitem_789 = split_80[2]
        getitem_790 = split_80[3]
        getitem_791 = split_80[4]
        getitem_792 = split_80[5]
        getitem_793 = split_80[6]
        getitem_794 = split_80[7];  split_80 = None
        cat_72 = torch.ops.aten.cat.default([getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794]);  getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = None
        reduce_scatter_tensor_50 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_72, 'sum', 8, '1');  cat_72 = None
        wait_tensor_236 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_50);  reduce_scatter_tensor_50 = None
        convert_element_type_618 = torch.ops.prims.convert_element_type.default(wait_tensor_236, torch.float32);  wait_tensor_236 = None
        convert_element_type_620 = torch.ops.prims.convert_element_type.default(wait_tensor_191, torch.float32);  wait_tensor_191 = None
        mul_162 = torch.ops.aten.mul.Tensor(convert_element_type_618, convert_element_type_620);  convert_element_type_620 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_116, mul_162)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_164, [2], True);  mul_164 = None
        div_3 = torch.ops.aten.div.Tensor(mul_116, 4096)
        mul_165 = torch.ops.aten.mul.Tensor(div_3, sum_9);  div_3 = sum_9 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_162, mul_165);  mul_162 = mul_165 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_29);  sub_6 = rsqrt_29 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_618, mul_116);  convert_element_type_618 = mul_116 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1]);  mul_167 = None
        convert_element_type_621 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(sum_10, torch.bfloat16);  sum_10 = None
        all_reduce_3 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_622, 'sum', '1');  convert_element_type_622 = None
        wait_tensor_237 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = None
        convert_element_type_623 = torch.ops.prims.convert_element_type.default(wait_tensor_237, torch.float32);  wait_tensor_237 = None
        reduce_scatter_tensor_51 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_623, 'avg', 8, '0');  convert_element_type_623 = None
        wait_tensor_238 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_51);  reduce_scatter_tensor_51 = None
        add_75 = torch.ops.aten.add.Tensor(add_71, convert_element_type_621);  add_71 = convert_element_type_621 = None
        all_gather_into_tensor_183 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_75, 8, '1')
        wait_tensor_239 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_183);  all_gather_into_tensor_183 = None
        split_81 = torch.ops.aten.split.Tensor(wait_tensor_239, 2);  wait_tensor_239 = None
        getitem_795 = split_81[0]
        getitem_796 = split_81[1]
        getitem_797 = split_81[2]
        getitem_798 = split_81[3]
        getitem_799 = split_81[4]
        getitem_800 = split_81[5]
        getitem_801 = split_81[6]
        getitem_802 = split_81[7];  split_81 = None
        cat_73 = torch.ops.aten.cat.default([getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802], 1);  getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = None
        view_1201 = torch.ops.aten.view.default(cat_73, [16384, 4096]);  cat_73 = None
        permute_225 = torch.ops.aten.permute.default(view_1201, [1, 0])
        permute_160 = torch.ops.aten.permute.default(getitem_654, [0, 2, 1, 3])
        view_1050 = torch.ops.aten.view.default(permute_160, [2, 8192, -1]);  permute_160 = None
        view_1056 = torch.ops.aten.view.default(view_1050, [16384, 512]);  view_1050 = None
        mm_135 = torch.ops.aten.mm.default(permute_225, view_1056);  permute_225 = view_1056 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        all_gather_into_tensor_160 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_479, 8, '0');  convert_element_type_479 = None
        wait_tensor_189 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_160);  all_gather_into_tensor_160 = None
        permute_161 = torch.ops.aten.permute.default(wait_tensor_189, [1, 0]);  wait_tensor_189 = None
        permute_227 = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        mm_136 = torch.ops.aten.mm.default(view_1201, permute_227);  view_1201 = permute_227 = None
        view_1202 = torch.ops.aten.view.default(mm_136, [2, 8192, 512]);  mm_136 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        reduce_scatter_tensor_52 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_628, 'avg', 8, '0');  convert_element_type_628 = None
        wait_tensor_240 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_52);  reduce_scatter_tensor_52 = None
        view_1203 = torch.ops.aten.view.default(view_1202, [2, 8192, 4, 128]);  view_1202 = None
        permute_229 = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        all_gather_into_tensor_155 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_463, 8, '0');  convert_element_type_463 = None
        wait_tensor_184 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_155);  all_gather_into_tensor_155 = None
        convert_element_type_464 = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_464, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [2], True);  pow_29 = None
        add_56 = torch.ops.aten.add.Scalar(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_464, rsqrt_28);  convert_element_type_464 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, wait_tensor_184)
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
        view_1023 = torch.ops.aten.view.default(cat_57, [16384, 4096]);  cat_57 = None
        view_1024 = torch.ops.aten.view.default(mm_98, [2, 8192, 512]);  mm_98 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        all_gather_into_tensor_158 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_469, 8, '0');  convert_element_type_469 = None
        wait_tensor_187 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_158);  all_gather_into_tensor_158 = None
        permute_155 = torch.ops.aten.permute.default(wait_tensor_187, [1, 0]);  wait_tensor_187 = None
        mm_99 = torch.ops.aten.mm.default(view_1023, permute_155)
        view_1031 = torch.ops.aten.view.default(mm_99, [2, 8192, 128]);  mm_99 = None
        view_1038 = torch.ops.aten.view.default(mm_100, [2, 8192, 128]);  mm_100 = None
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
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_229, permute_157, permute_158, permute_159, getitem_654, getitem_655, getitem_660, getitem_661, None, None, None, 8192, 8192, 0.0, True);  permute_229 = permute_157 = permute_158 = permute_159 = getitem_654 = getitem_655 = getitem_660 = getitem_661 = None
        getitem_803 = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_804 = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_805 = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None
        permute_230 = torch.ops.aten.permute.default(getitem_805, [0, 2, 1, 3]);  getitem_805 = None
        permute_231 = torch.ops.aten.permute.default(getitem_804, [0, 2, 1, 3]);  getitem_804 = None
        permute_232 = torch.ops.aten.permute.default(getitem_803, [0, 2, 1, 3]);  getitem_803 = None
        view_1204 = torch.ops.aten.view.default(permute_230, [2, 8192, 1, 4, 128]);  permute_230 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_1204, [3], True);  view_1204 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(sum_11, 3);  sum_11 = None
        view_1205 = torch.ops.aten.view.default(permute_231, [2, 8192, 1, 4, 128]);  permute_231 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_1205, [3], True);  view_1205 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(sum_12, 3);  sum_12 = None
        convert_element_type_629 = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        convert_element_type_630 = torch.ops.prims.convert_element_type.default(permute_232, torch.float32);  permute_232 = None
        view_1206 = torch.ops.aten.view.default(convert_element_type_629, [2, 8192, 1, 64, 2]);  convert_element_type_629 = None
        view_as_complex_34 = torch.ops.aten.view_as_complex.default(view_1206);  view_1206 = None
        mul_168 = torch.ops.aten.mul.Tensor(view_as_complex_34, _conj);  view_as_complex_34 = None
        view_1207 = torch.ops.aten.view.default(convert_element_type_630, [2, 8192, 4, 64, 2]);  convert_element_type_630 = None
        view_as_complex_35 = torch.ops.aten.view_as_complex.default(view_1207);  view_1207 = None
        mul_169 = torch.ops.aten.mul.Tensor(view_as_complex_35, _conj);  view_as_complex_35 = None
        view_as_real_34 = torch.ops.aten.view_as_real.default(mul_168);  mul_168 = None
        view_1208 = torch.ops.aten.view.default(view_as_real_34, [2, 8192, 1, 128]);  view_as_real_34 = None
        convert_element_type_631 = torch.ops.prims.convert_element_type.default(view_1208, torch.bfloat16);  view_1208 = None
        view_as_real_35 = torch.ops.aten.view_as_real.default(mul_169);  mul_169 = None
        view_1209 = torch.ops.aten.view.default(view_as_real_35, [2, 8192, 4, 128]);  view_as_real_35 = None
        convert_element_type_632 = torch.ops.prims.convert_element_type.default(view_1209, torch.bfloat16);  view_1209 = None
        view_1210 = torch.ops.aten.view.default(squeeze_2, [2, 8192, 128]);  squeeze_2 = None
        view_1211 = torch.ops.aten.view.default(convert_element_type_631, [2, 8192, 128]);  convert_element_type_631 = None
        view_1212 = torch.ops.aten.view.default(convert_element_type_632, [2, 8192, 512]);  convert_element_type_632 = None
        view_1213 = torch.ops.aten.view.default(view_1210, [16384, 128]);  view_1210 = None
        permute_233 = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_137 = torch.ops.aten.mm.default(permute_233, view_1023);  permute_233 = None
        convert_element_type_472 = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        all_gather_into_tensor_159 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_472, 8, '0');  convert_element_type_472 = None
        wait_tensor_188 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_159);  all_gather_into_tensor_159 = None
        permute_156 = torch.ops.aten.permute.default(wait_tensor_188, [1, 0]);  wait_tensor_188 = None
        permute_235 = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_138 = torch.ops.aten.mm.default(view_1213, permute_235);  view_1213 = permute_235 = None
        view_1214 = torch.ops.aten.view.default(mm_138, [2, 8192, 4096]);  mm_138 = None
        convert_element_type_637 = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        reduce_scatter_tensor_53 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_637, 'avg', 8, '0');  convert_element_type_637 = None
        wait_tensor_241 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_53);  reduce_scatter_tensor_53 = None
        view_1215 = torch.ops.aten.view.default(view_1211, [16384, 128]);  view_1211 = None
        permute_237 = torch.ops.aten.permute.default(view_1215, [1, 0])
        mm_139 = torch.ops.aten.mm.default(permute_237, view_1023);  permute_237 = None
        permute_239 = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_140 = torch.ops.aten.mm.default(view_1215, permute_239);  view_1215 = permute_239 = None
        view_1216 = torch.ops.aten.view.default(mm_140, [2, 8192, 4096]);  mm_140 = None
        add_76 = torch.ops.aten.add.Tensor(view_1214, view_1216);  view_1214 = view_1216 = None
        convert_element_type_642 = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        reduce_scatter_tensor_54 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_642, 'avg', 8, '0');  convert_element_type_642 = None
        wait_tensor_242 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_54);  reduce_scatter_tensor_54 = None
        view_1217 = torch.ops.aten.view.default(view_1212, [16384, 512]);  view_1212 = None
        permute_241 = torch.ops.aten.permute.default(view_1217, [1, 0])
        mm_141 = torch.ops.aten.mm.default(permute_241, view_1023);  permute_241 = view_1023 = None
        convert_element_type_466 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        all_gather_into_tensor_157 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_466, 8, '0');  convert_element_type_466 = None
        wait_tensor_186 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_157);  all_gather_into_tensor_157 = None
        permute_154 = torch.ops.aten.permute.default(wait_tensor_186, [1, 0]);  wait_tensor_186 = None
        permute_243 = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_142 = torch.ops.aten.mm.default(view_1217, permute_243);  view_1217 = permute_243 = None
        view_1218 = torch.ops.aten.view.default(mm_142, [2, 8192, 4096]);  mm_142 = None
        add_77 = torch.ops.aten.add.Tensor(add_76, view_1218);  add_76 = view_1218 = None
        convert_element_type_647 = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        reduce_scatter_tensor_55 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_647, 'avg', 8, '0');  convert_element_type_647 = None
        wait_tensor_243 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_55);  reduce_scatter_tensor_55 = None
        split_82 = torch.ops.aten.split.Tensor(add_77, 1024, 1);  add_77 = None
        getitem_806 = split_82[0]
        getitem_807 = split_82[1]
        getitem_808 = split_82[2]
        getitem_809 = split_82[3]
        getitem_810 = split_82[4]
        getitem_811 = split_82[5]
        getitem_812 = split_82[6]
        getitem_813 = split_82[7];  split_82 = None
        cat_74 = torch.ops.aten.cat.default([getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813]);  getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = None
        reduce_scatter_tensor_56 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_74, 'sum', 8, '1');  cat_74 = None
        wait_tensor_244 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_56);  reduce_scatter_tensor_56 = None
        convert_element_type_648 = torch.ops.prims.convert_element_type.default(wait_tensor_244, torch.float32);  wait_tensor_244 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(wait_tensor_184, torch.float32);  wait_tensor_184 = None
        mul_170 = torch.ops.aten.mul.Tensor(convert_element_type_648, convert_element_type_650);  convert_element_type_650 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_112, mul_170)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        div_4 = torch.ops.aten.div.Tensor(mul_112, 4096)
        mul_173 = torch.ops.aten.mul.Tensor(div_4, sum_13);  div_4 = sum_13 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_170, mul_173);  mul_170 = mul_173 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_28);  sub_7 = rsqrt_28 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_648, mul_112);  convert_element_type_648 = mul_112 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        convert_element_type_651 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        convert_element_type_652 = torch.ops.prims.convert_element_type.default(sum_14, torch.bfloat16);  sum_14 = None
        all_reduce_4 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_652, 'sum', '1');  convert_element_type_652 = None
        wait_tensor_245 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4);  all_reduce_4 = None
        convert_element_type_653 = torch.ops.prims.convert_element_type.default(wait_tensor_245, torch.float32);  wait_tensor_245 = None
        reduce_scatter_tensor_57 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_653, 'avg', 8, '0');  convert_element_type_653 = None
        wait_tensor_246 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_57);  reduce_scatter_tensor_57 = None
        add_78 = torch.ops.aten.add.Tensor(add_75, convert_element_type_651);  add_75 = convert_element_type_651 = None
        all_gather_into_tensor_184 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_78, 8, '1')
        wait_tensor_247 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_184);  all_gather_into_tensor_184 = None
        split_83 = torch.ops.aten.split.Tensor(wait_tensor_247, 2);  wait_tensor_247 = None
        getitem_814 = split_83[0]
        getitem_815 = split_83[1]
        getitem_816 = split_83[2]
        getitem_817 = split_83[3]
        getitem_818 = split_83[4]
        getitem_819 = split_83[5]
        getitem_820 = split_83[6]
        getitem_821 = split_83[7];  split_83 = None
        cat_75 = torch.ops.aten.cat.default([getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821], 1);  getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = None
        view_1219 = torch.ops.aten.view.default(cat_75, [16384, 4096]);  cat_75 = None
        permute_245 = torch.ops.aten.permute.default(view_1219, [1, 0])
        wait_tensor_177 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_27);  reduce_scatter_tensor_27 = None
        add_53 = torch.ops.aten.add.Tensor(add_51, wait_tensor_177);  wait_tensor_177 = None
        convert_element_type_449 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        all_gather_into_tensor_150 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_449, 8, '0');  convert_element_type_449 = None
        wait_tensor_178 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_150);  all_gather_into_tensor_150 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_450, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [2], True);  pow_28 = None
        add_54 = torch.ops.aten.add.Scalar(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_450, rsqrt_27);  convert_element_type_450 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, wait_tensor_178)
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
        view_996 = torch.ops.aten.view.default(cat_55, [16384, 4096]);  cat_55 = None
        view_997 = torch.ops.aten.view.default(mm_95, [2, 8192, 1792]);  mm_95 = None
        convert_element_type_455 = torch.ops.prims.convert_element_type.default(view_997, torch.float32);  view_997 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_455)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_13);  sigmoid_13 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        all_gather_into_tensor_153 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_457, 8, '0');  convert_element_type_457 = None
        wait_tensor_181 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_153);  all_gather_into_tensor_153 = None
        permute_152 = torch.ops.aten.permute.default(wait_tensor_181, [1, 0]);  wait_tensor_181 = None
        mm_96 = torch.ops.aten.mm.default(view_996, permute_152)
        view_1004 = torch.ops.aten.view.default(mm_96, [2, 8192, 1792]);  mm_96 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_456, view_1004)
        view_1011 = torch.ops.aten.view.default(mul_111, [16384, 1792]);  mul_111 = None
        mm_143 = torch.ops.aten.mm.default(permute_245, view_1011);  permute_245 = view_1011 = None
        convert_element_type_460 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        all_gather_into_tensor_154 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_460, 8, '0');  convert_element_type_460 = None
        wait_tensor_182 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_154);  all_gather_into_tensor_154 = None
        permute_153 = torch.ops.aten.permute.default(wait_tensor_182, [1, 0]);  wait_tensor_182 = None
        permute_247 = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_144 = torch.ops.aten.mm.default(view_1219, permute_247);  view_1219 = permute_247 = None
        view_1220 = torch.ops.aten.view.default(mm_144, [2, 8192, 1792]);  mm_144 = None
        convert_element_type_658 = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        reduce_scatter_tensor_58 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_658, 'avg', 8, '0');  convert_element_type_658 = None
        wait_tensor_248 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_58);  reduce_scatter_tensor_58 = None
        mul_176 = torch.ops.aten.mul.Tensor(view_1220, convert_element_type_456);  convert_element_type_456 = None
        mul_177 = torch.ops.aten.mul.Tensor(view_1220, view_1004);  view_1220 = view_1004 = None
        view_1221 = torch.ops.aten.view.default(mul_176, [16384, 1792]);  mul_176 = None
        permute_249 = torch.ops.aten.permute.default(view_1221, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_249, view_996);  permute_249 = None
        permute_251 = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_146 = torch.ops.aten.mm.default(view_1221, permute_251);  view_1221 = permute_251 = None
        view_1222 = torch.ops.aten.view.default(mm_146, [2, 8192, 4096]);  mm_146 = None
        convert_element_type_663 = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        reduce_scatter_tensor_59 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_663, 'avg', 8, '0');  convert_element_type_663 = None
        wait_tensor_249 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_59);  reduce_scatter_tensor_59 = None
        convert_element_type_664 = torch.ops.prims.convert_element_type.default(mul_177, torch.float32);  mul_177 = None
        neg_2 = torch.ops.aten.neg.default(convert_element_type_455)
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_79 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(add_79);  add_79 = None
        mul_178 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_179 = torch.ops.aten.mul.Tensor(convert_element_type_664, mul_178);  convert_element_type_664 = None
        sub_8 = torch.ops.aten.sub.Tensor(1, mul_178);  mul_178 = None
        mul_180 = torch.ops.aten.mul.Tensor(convert_element_type_455, sub_8);  convert_element_type_455 = sub_8 = None
        add_80 = torch.ops.aten.add.Tensor(mul_180, 1);  mul_180 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, add_80);  mul_179 = add_80 = None
        convert_element_type_666 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_1223 = torch.ops.aten.view.default(convert_element_type_666, [16384, 1792]);  convert_element_type_666 = None
        permute_253 = torch.ops.aten.permute.default(view_1223, [1, 0])
        mm_147 = torch.ops.aten.mm.default(permute_253, view_996);  permute_253 = view_996 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        all_gather_into_tensor_152 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_452, 8, '0');  convert_element_type_452 = None
        wait_tensor_180 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_152);  all_gather_into_tensor_152 = None
        permute_151 = torch.ops.aten.permute.default(wait_tensor_180, [1, 0]);  wait_tensor_180 = None
        permute_255 = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_148 = torch.ops.aten.mm.default(view_1223, permute_255);  view_1223 = permute_255 = None
        view_1224 = torch.ops.aten.view.default(mm_148, [2, 8192, 4096]);  mm_148 = None
        add_81 = torch.ops.aten.add.Tensor(view_1222, view_1224);  view_1222 = view_1224 = None
        convert_element_type_671 = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        reduce_scatter_tensor_60 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_671, 'avg', 8, '0');  convert_element_type_671 = None
        wait_tensor_250 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_60);  reduce_scatter_tensor_60 = None
        split_84 = torch.ops.aten.split.Tensor(add_81, 1024, 1);  add_81 = None
        getitem_822 = split_84[0]
        getitem_823 = split_84[1]
        getitem_824 = split_84[2]
        getitem_825 = split_84[3]
        getitem_826 = split_84[4]
        getitem_827 = split_84[5]
        getitem_828 = split_84[6]
        getitem_829 = split_84[7];  split_84 = None
        cat_76 = torch.ops.aten.cat.default([getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829]);  getitem_822 = getitem_823 = getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = None
        reduce_scatter_tensor_61 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_76, 'sum', 8, '1');  cat_76 = None
        wait_tensor_251 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_61);  reduce_scatter_tensor_61 = None
        convert_element_type_672 = torch.ops.prims.convert_element_type.default(wait_tensor_251, torch.float32);  wait_tensor_251 = None
        convert_element_type_674 = torch.ops.prims.convert_element_type.default(wait_tensor_178, torch.float32);  wait_tensor_178 = None
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_672, convert_element_type_674);  convert_element_type_674 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_108, mul_182)
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_184, [2], True);  mul_184 = None
        div_5 = torch.ops.aten.div.Tensor(mul_108, 4096)
        mul_185 = torch.ops.aten.mul.Tensor(div_5, sum_15);  div_5 = sum_15 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_182, mul_185);  mul_182 = mul_185 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_27);  sub_9 = rsqrt_27 = None
        mul_187 = torch.ops.aten.mul.Tensor(convert_element_type_672, mul_108);  convert_element_type_672 = mul_108 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1]);  mul_187 = None
        convert_element_type_675 = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None
        convert_element_type_676 = torch.ops.prims.convert_element_type.default(sum_16, torch.bfloat16);  sum_16 = None
        all_reduce_5 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_676, 'sum', '1');  convert_element_type_676 = None
        wait_tensor_252 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_5);  all_reduce_5 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(wait_tensor_252, torch.float32);  wait_tensor_252 = None
        reduce_scatter_tensor_62 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_677, 'avg', 8, '0');  convert_element_type_677 = None
        wait_tensor_253 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_62);  reduce_scatter_tensor_62 = None
        add_82 = torch.ops.aten.add.Tensor(add_78, convert_element_type_675);  add_78 = convert_element_type_675 = None
        all_gather_into_tensor_185 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_82, 8, '1')
        wait_tensor_254 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_185);  all_gather_into_tensor_185 = None
        split_85 = torch.ops.aten.split.Tensor(wait_tensor_254, 2);  wait_tensor_254 = None
        getitem_830 = split_85[0]
        getitem_831 = split_85[1]
        getitem_832 = split_85[2]
        getitem_833 = split_85[3]
        getitem_834 = split_85[4]
        getitem_835 = split_85[5]
        getitem_836 = split_85[6]
        getitem_837 = split_85[7];  split_85 = None
        cat_77 = torch.ops.aten.cat.default([getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837], 1);  getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = None
        view_1225 = torch.ops.aten.view.default(cat_77, [16384, 4096]);  cat_77 = None
        permute_257 = torch.ops.aten.permute.default(view_1225, [1, 0])
        permute_149 = torch.ops.aten.permute.default(getitem_613, [0, 2, 1, 3])
        view_978 = torch.ops.aten.view.default(permute_149, [2, 8192, -1]);  permute_149 = None
        view_984 = torch.ops.aten.view.default(view_978, [16384, 512]);  view_978 = None
        mm_149 = torch.ops.aten.mm.default(permute_257, view_984);  permute_257 = view_984 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        all_gather_into_tensor_149 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_446, 8, '0');  convert_element_type_446 = None
        wait_tensor_176 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_149);  all_gather_into_tensor_149 = None
        permute_150 = torch.ops.aten.permute.default(wait_tensor_176, [1, 0]);  wait_tensor_176 = None
        permute_259 = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        mm_150 = torch.ops.aten.mm.default(view_1225, permute_259);  view_1225 = permute_259 = None
        view_1226 = torch.ops.aten.view.default(mm_150, [2, 8192, 512]);  mm_150 = None
        convert_element_type_682 = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        reduce_scatter_tensor_63 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_682, 'avg', 8, '0');  convert_element_type_682 = None
        wait_tensor_255 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_63);  reduce_scatter_tensor_63 = None
        view_1227 = torch.ops.aten.view.default(view_1226, [2, 8192, 4, 128]);  view_1226 = None
        permute_261 = torch.ops.aten.permute.default(view_1227, [0, 2, 1, 3]);  view_1227 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        all_gather_into_tensor_144 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_430, 8, '0');  convert_element_type_430 = None
        wait_tensor_171 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_144);  all_gather_into_tensor_144 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_431, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [2], True);  pow_27 = None
        add_52 = torch.ops.aten.add.Scalar(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_431, rsqrt_26);  convert_element_type_431 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, wait_tensor_171)
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
        view_951 = torch.ops.aten.view.default(cat_53, [16384, 4096]);  cat_53 = None
        view_952 = torch.ops.aten.view.default(mm_91, [2, 8192, 512]);  mm_91 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        all_gather_into_tensor_147 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_436, 8, '0');  convert_element_type_436 = None
        wait_tensor_174 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_147);  all_gather_into_tensor_147 = None
        permute_144 = torch.ops.aten.permute.default(wait_tensor_174, [1, 0]);  wait_tensor_174 = None
        mm_92 = torch.ops.aten.mm.default(view_951, permute_144)
        view_959 = torch.ops.aten.view.default(mm_92, [2, 8192, 128]);  mm_92 = None
        view_966 = torch.ops.aten.view.default(mm_93, [2, 8192, 128]);  mm_93 = None
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
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_261, permute_146, permute_147, permute_148, getitem_613, getitem_614, getitem_619, getitem_620, None, None, None, 8192, 8192, 0.0, True);  permute_261 = permute_146 = permute_147 = permute_148 = getitem_613 = getitem_614 = getitem_619 = getitem_620 = None
        getitem_838 = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_839 = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_840 = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None
        permute_262 = torch.ops.aten.permute.default(getitem_840, [0, 2, 1, 3]);  getitem_840 = None
        permute_263 = torch.ops.aten.permute.default(getitem_839, [0, 2, 1, 3]);  getitem_839 = None
        permute_264 = torch.ops.aten.permute.default(getitem_838, [0, 2, 1, 3]);  getitem_838 = None
        view_1228 = torch.ops.aten.view.default(permute_262, [2, 8192, 1, 4, 128]);  permute_262 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_1228, [3], True);  view_1228 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(sum_17, 3);  sum_17 = None
        view_1229 = torch.ops.aten.view.default(permute_263, [2, 8192, 1, 4, 128]);  permute_263 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_1229, [3], True);  view_1229 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(sum_18, 3);  sum_18 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(squeeze_5, torch.float32);  squeeze_5 = None
        convert_element_type_684 = torch.ops.prims.convert_element_type.default(permute_264, torch.float32);  permute_264 = None
        view_1230 = torch.ops.aten.view.default(convert_element_type_683, [2, 8192, 1, 64, 2]);  convert_element_type_683 = None
        view_as_complex_36 = torch.ops.aten.view_as_complex.default(view_1230);  view_1230 = None
        mul_188 = torch.ops.aten.mul.Tensor(view_as_complex_36, _conj);  view_as_complex_36 = None
        view_1231 = torch.ops.aten.view.default(convert_element_type_684, [2, 8192, 4, 64, 2]);  convert_element_type_684 = None
        view_as_complex_37 = torch.ops.aten.view_as_complex.default(view_1231);  view_1231 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_as_complex_37, _conj);  view_as_complex_37 = None
        view_as_real_36 = torch.ops.aten.view_as_real.default(mul_188);  mul_188 = None
        view_1232 = torch.ops.aten.view.default(view_as_real_36, [2, 8192, 1, 128]);  view_as_real_36 = None
        convert_element_type_685 = torch.ops.prims.convert_element_type.default(view_1232, torch.bfloat16);  view_1232 = None
        view_as_real_37 = torch.ops.aten.view_as_real.default(mul_189);  mul_189 = None
        view_1233 = torch.ops.aten.view.default(view_as_real_37, [2, 8192, 4, 128]);  view_as_real_37 = None
        convert_element_type_686 = torch.ops.prims.convert_element_type.default(view_1233, torch.bfloat16);  view_1233 = None
        view_1234 = torch.ops.aten.view.default(squeeze_4, [2, 8192, 128]);  squeeze_4 = None
        view_1235 = torch.ops.aten.view.default(convert_element_type_685, [2, 8192, 128]);  convert_element_type_685 = None
        view_1236 = torch.ops.aten.view.default(convert_element_type_686, [2, 8192, 512]);  convert_element_type_686 = None
        view_1237 = torch.ops.aten.view.default(view_1234, [16384, 128]);  view_1234 = None
        permute_265 = torch.ops.aten.permute.default(view_1237, [1, 0])
        mm_151 = torch.ops.aten.mm.default(permute_265, view_951);  permute_265 = None
        convert_element_type_439 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        all_gather_into_tensor_148 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_439, 8, '0');  convert_element_type_439 = None
        wait_tensor_175 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_148);  all_gather_into_tensor_148 = None
        permute_145 = torch.ops.aten.permute.default(wait_tensor_175, [1, 0]);  wait_tensor_175 = None
        permute_267 = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_152 = torch.ops.aten.mm.default(view_1237, permute_267);  view_1237 = permute_267 = None
        view_1238 = torch.ops.aten.view.default(mm_152, [2, 8192, 4096]);  mm_152 = None
        convert_element_type_691 = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        reduce_scatter_tensor_64 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_691, 'avg', 8, '0');  convert_element_type_691 = None
        wait_tensor_256 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_64);  reduce_scatter_tensor_64 = None
        view_1239 = torch.ops.aten.view.default(view_1235, [16384, 128]);  view_1235 = None
        permute_269 = torch.ops.aten.permute.default(view_1239, [1, 0])
        mm_153 = torch.ops.aten.mm.default(permute_269, view_951);  permute_269 = None
        permute_271 = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_154 = torch.ops.aten.mm.default(view_1239, permute_271);  view_1239 = permute_271 = None
        view_1240 = torch.ops.aten.view.default(mm_154, [2, 8192, 4096]);  mm_154 = None
        add_83 = torch.ops.aten.add.Tensor(view_1238, view_1240);  view_1238 = view_1240 = None
        convert_element_type_696 = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        reduce_scatter_tensor_65 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_696, 'avg', 8, '0');  convert_element_type_696 = None
        wait_tensor_257 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_65);  reduce_scatter_tensor_65 = None
        view_1241 = torch.ops.aten.view.default(view_1236, [16384, 512]);  view_1236 = None
        permute_273 = torch.ops.aten.permute.default(view_1241, [1, 0])
        mm_155 = torch.ops.aten.mm.default(permute_273, view_951);  permute_273 = view_951 = None
        convert_element_type_433 = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        all_gather_into_tensor_146 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_433, 8, '0');  convert_element_type_433 = None
        wait_tensor_173 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_146);  all_gather_into_tensor_146 = None
        permute_143 = torch.ops.aten.permute.default(wait_tensor_173, [1, 0]);  wait_tensor_173 = None
        permute_275 = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_156 = torch.ops.aten.mm.default(view_1241, permute_275);  view_1241 = permute_275 = None
        view_1242 = torch.ops.aten.view.default(mm_156, [2, 8192, 4096]);  mm_156 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, view_1242);  add_83 = view_1242 = None
        convert_element_type_701 = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        reduce_scatter_tensor_66 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_701, 'avg', 8, '0');  convert_element_type_701 = None
        wait_tensor_258 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_66);  reduce_scatter_tensor_66 = None
        split_86 = torch.ops.aten.split.Tensor(add_84, 1024, 1);  add_84 = None
        getitem_841 = split_86[0]
        getitem_842 = split_86[1]
        getitem_843 = split_86[2]
        getitem_844 = split_86[3]
        getitem_845 = split_86[4]
        getitem_846 = split_86[5]
        getitem_847 = split_86[6]
        getitem_848 = split_86[7];  split_86 = None
        cat_78 = torch.ops.aten.cat.default([getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848]);  getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = None
        reduce_scatter_tensor_67 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_78, 'sum', 8, '1');  cat_78 = None
        wait_tensor_259 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_67);  reduce_scatter_tensor_67 = None
        convert_element_type_702 = torch.ops.prims.convert_element_type.default(wait_tensor_259, torch.float32);  wait_tensor_259 = None
        convert_element_type_704 = torch.ops.prims.convert_element_type.default(wait_tensor_171, torch.float32);  wait_tensor_171 = None
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_702, convert_element_type_704);  convert_element_type_704 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_104, mul_190)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
        div_6 = torch.ops.aten.div.Tensor(mul_104, 4096)
        mul_193 = torch.ops.aten.mul.Tensor(div_6, sum_19);  div_6 = sum_19 = None
        sub_10 = torch.ops.aten.sub.Tensor(mul_190, mul_193);  mul_190 = mul_193 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_26);  sub_10 = rsqrt_26 = None
        mul_195 = torch.ops.aten.mul.Tensor(convert_element_type_702, mul_104);  convert_element_type_702 = mul_104 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
        convert_element_type_705 = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        convert_element_type_706 = torch.ops.prims.convert_element_type.default(sum_20, torch.bfloat16);  sum_20 = None
        all_reduce_6 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_706, 'sum', '1');  convert_element_type_706 = None
        wait_tensor_260 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_6);  all_reduce_6 = None
        convert_element_type_707 = torch.ops.prims.convert_element_type.default(wait_tensor_260, torch.float32);  wait_tensor_260 = None
        reduce_scatter_tensor_68 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_707, 'avg', 8, '0');  convert_element_type_707 = None
        wait_tensor_261 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_68);  reduce_scatter_tensor_68 = None
        add_85 = torch.ops.aten.add.Tensor(add_82, convert_element_type_705);  add_82 = convert_element_type_705 = None
        all_gather_into_tensor_186 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_85, 8, '1')
        wait_tensor_262 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_186);  all_gather_into_tensor_186 = None
        split_87 = torch.ops.aten.split.Tensor(wait_tensor_262, 2);  wait_tensor_262 = None
        getitem_849 = split_87[0]
        getitem_850 = split_87[1]
        getitem_851 = split_87[2]
        getitem_852 = split_87[3]
        getitem_853 = split_87[4]
        getitem_854 = split_87[5]
        getitem_855 = split_87[6]
        getitem_856 = split_87[7];  split_87 = None
        cat_79 = torch.ops.aten.cat.default([getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856], 1);  getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = None
        view_1243 = torch.ops.aten.view.default(cat_79, [16384, 4096]);  cat_79 = None
        permute_277 = torch.ops.aten.permute.default(view_1243, [1, 0])
        wait_tensor_164 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_25);  reduce_scatter_tensor_25 = None
        add_49 = torch.ops.aten.add.Tensor(add_47, wait_tensor_164);  wait_tensor_164 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        all_gather_into_tensor_139 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_416, 8, '0');  convert_element_type_416 = None
        wait_tensor_165 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_139);  all_gather_into_tensor_139 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(add_49, torch.float32);  add_49 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_417, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [2], True);  pow_26 = None
        add_50 = torch.ops.aten.add.Scalar(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_100 = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_25);  convert_element_type_417 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, wait_tensor_165)
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
        view_924 = torch.ops.aten.view.default(cat_51, [16384, 4096]);  cat_51 = None
        view_925 = torch.ops.aten.view.default(mm_88, [2, 8192, 1792]);  mm_88 = None
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_925, torch.float32);  view_925 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_422)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_422, sigmoid_12);  sigmoid_12 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        all_gather_into_tensor_142 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_424, 8, '0');  convert_element_type_424 = None
        wait_tensor_168 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_142);  all_gather_into_tensor_142 = None
        permute_141 = torch.ops.aten.permute.default(wait_tensor_168, [1, 0]);  wait_tensor_168 = None
        mm_89 = torch.ops.aten.mm.default(view_924, permute_141)
        view_932 = torch.ops.aten.view.default(mm_89, [2, 8192, 1792]);  mm_89 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_423, view_932)
        view_939 = torch.ops.aten.view.default(mul_103, [16384, 1792]);  mul_103 = None
        mm_157 = torch.ops.aten.mm.default(permute_277, view_939);  permute_277 = view_939 = None
        convert_element_type_427 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        all_gather_into_tensor_143 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_427, 8, '0');  convert_element_type_427 = None
        wait_tensor_169 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_143);  all_gather_into_tensor_143 = None
        permute_142 = torch.ops.aten.permute.default(wait_tensor_169, [1, 0]);  wait_tensor_169 = None
        permute_279 = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_158 = torch.ops.aten.mm.default(view_1243, permute_279);  view_1243 = permute_279 = None
        view_1244 = torch.ops.aten.view.default(mm_158, [2, 8192, 1792]);  mm_158 = None
        convert_element_type_712 = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        reduce_scatter_tensor_69 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_712, 'avg', 8, '0');  convert_element_type_712 = None
        wait_tensor_263 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_69);  reduce_scatter_tensor_69 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_1244, convert_element_type_423);  convert_element_type_423 = None
        mul_197 = torch.ops.aten.mul.Tensor(view_1244, view_932);  view_1244 = view_932 = None
        view_1245 = torch.ops.aten.view.default(mul_196, [16384, 1792]);  mul_196 = None
        permute_281 = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_159 = torch.ops.aten.mm.default(permute_281, view_924);  permute_281 = None
        permute_283 = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_160 = torch.ops.aten.mm.default(view_1245, permute_283);  view_1245 = permute_283 = None
        view_1246 = torch.ops.aten.view.default(mm_160, [2, 8192, 4096]);  mm_160 = None
        convert_element_type_717 = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        reduce_scatter_tensor_70 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_717, 'avg', 8, '0');  convert_element_type_717 = None
        wait_tensor_264 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_70);  reduce_scatter_tensor_70 = None
        convert_element_type_718 = torch.ops.prims.convert_element_type.default(mul_197, torch.float32);  mul_197 = None
        neg_3 = torch.ops.aten.neg.default(convert_element_type_422)
        exp_3 = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_86 = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_718, mul_198);  convert_element_type_718 = None
        sub_11 = torch.ops.aten.sub.Tensor(1, mul_198);  mul_198 = None
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_422, sub_11);  convert_element_type_422 = sub_11 = None
        add_87 = torch.ops.aten.add.Tensor(mul_200, 1);  mul_200 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_199, add_87);  mul_199 = add_87 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        view_1247 = torch.ops.aten.view.default(convert_element_type_720, [16384, 1792]);  convert_element_type_720 = None
        permute_285 = torch.ops.aten.permute.default(view_1247, [1, 0])
        mm_161 = torch.ops.aten.mm.default(permute_285, view_924);  permute_285 = view_924 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        all_gather_into_tensor_141 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_419, 8, '0');  convert_element_type_419 = None
        wait_tensor_167 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_141);  all_gather_into_tensor_141 = None
        permute_140 = torch.ops.aten.permute.default(wait_tensor_167, [1, 0]);  wait_tensor_167 = None
        permute_287 = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_162 = torch.ops.aten.mm.default(view_1247, permute_287);  view_1247 = permute_287 = None
        view_1248 = torch.ops.aten.view.default(mm_162, [2, 8192, 4096]);  mm_162 = None
        add_88 = torch.ops.aten.add.Tensor(view_1246, view_1248);  view_1246 = view_1248 = None
        convert_element_type_725 = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        reduce_scatter_tensor_71 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_725, 'avg', 8, '0');  convert_element_type_725 = None
        wait_tensor_265 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_71);  reduce_scatter_tensor_71 = None
        split_88 = torch.ops.aten.split.Tensor(add_88, 1024, 1);  add_88 = None
        getitem_857 = split_88[0]
        getitem_858 = split_88[1]
        getitem_859 = split_88[2]
        getitem_860 = split_88[3]
        getitem_861 = split_88[4]
        getitem_862 = split_88[5]
        getitem_863 = split_88[6]
        getitem_864 = split_88[7];  split_88 = None
        cat_80 = torch.ops.aten.cat.default([getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864]);  getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = getitem_864 = None
        reduce_scatter_tensor_72 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_80, 'sum', 8, '1');  cat_80 = None
        wait_tensor_266 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_72);  reduce_scatter_tensor_72 = None
        convert_element_type_726 = torch.ops.prims.convert_element_type.default(wait_tensor_266, torch.float32);  wait_tensor_266 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(wait_tensor_165, torch.float32);  wait_tensor_165 = None
        mul_202 = torch.ops.aten.mul.Tensor(convert_element_type_726, convert_element_type_728);  convert_element_type_728 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_100, mul_202)
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        div_7 = torch.ops.aten.div.Tensor(mul_100, 4096)
        mul_205 = torch.ops.aten.mul.Tensor(div_7, sum_21);  div_7 = sum_21 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_202, mul_205);  mul_202 = mul_205 = None
        mul_206 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_25);  sub_12 = rsqrt_25 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_726, mul_100);  convert_element_type_726 = mul_100 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        convert_element_type_730 = torch.ops.prims.convert_element_type.default(sum_22, torch.bfloat16);  sum_22 = None
        all_reduce_7 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_730, 'sum', '1');  convert_element_type_730 = None
        wait_tensor_267 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_7);  all_reduce_7 = None
        convert_element_type_731 = torch.ops.prims.convert_element_type.default(wait_tensor_267, torch.float32);  wait_tensor_267 = None
        reduce_scatter_tensor_73 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_731, 'avg', 8, '0');  convert_element_type_731 = None
        wait_tensor_268 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_73);  reduce_scatter_tensor_73 = None
        add_89 = torch.ops.aten.add.Tensor(add_85, convert_element_type_729);  add_85 = convert_element_type_729 = None
        all_gather_into_tensor_187 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_89, 8, '1')
        wait_tensor_269 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_187);  all_gather_into_tensor_187 = None
        split_89 = torch.ops.aten.split.Tensor(wait_tensor_269, 2);  wait_tensor_269 = None
        getitem_865 = split_89[0]
        getitem_866 = split_89[1]
        getitem_867 = split_89[2]
        getitem_868 = split_89[3]
        getitem_869 = split_89[4]
        getitem_870 = split_89[5]
        getitem_871 = split_89[6]
        getitem_872 = split_89[7];  split_89 = None
        cat_81 = torch.ops.aten.cat.default([getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872], 1);  getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = None
        view_1249 = torch.ops.aten.view.default(cat_81, [16384, 4096]);  cat_81 = None
        permute_289 = torch.ops.aten.permute.default(view_1249, [1, 0])
        permute_138 = torch.ops.aten.permute.default(getitem_572, [0, 2, 1, 3])
        view_906 = torch.ops.aten.view.default(permute_138, [2, 8192, -1]);  permute_138 = None
        view_912 = torch.ops.aten.view.default(view_906, [16384, 512]);  view_906 = None
        mm_163 = torch.ops.aten.mm.default(permute_289, view_912);  permute_289 = view_912 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        all_gather_into_tensor_138 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_413, 8, '0');  convert_element_type_413 = None
        wait_tensor_163 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_138);  all_gather_into_tensor_138 = None
        permute_139 = torch.ops.aten.permute.default(wait_tensor_163, [1, 0]);  wait_tensor_163 = None
        permute_291 = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        mm_164 = torch.ops.aten.mm.default(view_1249, permute_291);  view_1249 = permute_291 = None
        view_1250 = torch.ops.aten.view.default(mm_164, [2, 8192, 512]);  mm_164 = None
        convert_element_type_736 = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        reduce_scatter_tensor_74 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_736, 'avg', 8, '0');  convert_element_type_736 = None
        wait_tensor_270 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_74);  reduce_scatter_tensor_74 = None
        view_1251 = torch.ops.aten.view.default(view_1250, [2, 8192, 4, 128]);  view_1250 = None
        permute_293 = torch.ops.aten.permute.default(view_1251, [0, 2, 1, 3]);  view_1251 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        all_gather_into_tensor_133 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_397, 8, '0');  convert_element_type_397 = None
        wait_tensor_158 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_133);  all_gather_into_tensor_133 = None
        convert_element_type_398 = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [2], True);  pow_25 = None
        add_48 = torch.ops.aten.add.Scalar(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_398, rsqrt_24);  convert_element_type_398 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, wait_tensor_158)
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
        view_879 = torch.ops.aten.view.default(cat_49, [16384, 4096]);  cat_49 = None
        view_880 = torch.ops.aten.view.default(mm_84, [2, 8192, 512]);  mm_84 = None
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        all_gather_into_tensor_136 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_403, 8, '0');  convert_element_type_403 = None
        wait_tensor_161 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_136);  all_gather_into_tensor_136 = None
        permute_133 = torch.ops.aten.permute.default(wait_tensor_161, [1, 0]);  wait_tensor_161 = None
        mm_85 = torch.ops.aten.mm.default(view_879, permute_133)
        view_887 = torch.ops.aten.view.default(mm_85, [2, 8192, 128]);  mm_85 = None
        view_894 = torch.ops.aten.view.default(mm_86, [2, 8192, 128]);  mm_86 = None
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
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_293, permute_135, permute_136, permute_137, getitem_572, getitem_573, getitem_578, getitem_579, None, None, None, 8192, 8192, 0.0, True);  permute_293 = permute_135 = permute_136 = permute_137 = getitem_572 = getitem_573 = getitem_578 = getitem_579 = None
        getitem_873 = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_874 = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_875 = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None
        permute_294 = torch.ops.aten.permute.default(getitem_875, [0, 2, 1, 3]);  getitem_875 = None
        permute_295 = torch.ops.aten.permute.default(getitem_874, [0, 2, 1, 3]);  getitem_874 = None
        permute_296 = torch.ops.aten.permute.default(getitem_873, [0, 2, 1, 3]);  getitem_873 = None
        view_1252 = torch.ops.aten.view.default(permute_294, [2, 8192, 1, 4, 128]);  permute_294 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_1252, [3], True);  view_1252 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(sum_23, 3);  sum_23 = None
        view_1253 = torch.ops.aten.view.default(permute_295, [2, 8192, 1, 4, 128]);  permute_295 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_1253, [3], True);  view_1253 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(sum_24, 3);  sum_24 = None
        convert_element_type_737 = torch.ops.prims.convert_element_type.default(squeeze_7, torch.float32);  squeeze_7 = None
        convert_element_type_738 = torch.ops.prims.convert_element_type.default(permute_296, torch.float32);  permute_296 = None
        view_1254 = torch.ops.aten.view.default(convert_element_type_737, [2, 8192, 1, 64, 2]);  convert_element_type_737 = None
        view_as_complex_38 = torch.ops.aten.view_as_complex.default(view_1254);  view_1254 = None
        mul_208 = torch.ops.aten.mul.Tensor(view_as_complex_38, _conj);  view_as_complex_38 = None
        view_1255 = torch.ops.aten.view.default(convert_element_type_738, [2, 8192, 4, 64, 2]);  convert_element_type_738 = None
        view_as_complex_39 = torch.ops.aten.view_as_complex.default(view_1255);  view_1255 = None
        mul_209 = torch.ops.aten.mul.Tensor(view_as_complex_39, _conj);  view_as_complex_39 = None
        view_as_real_38 = torch.ops.aten.view_as_real.default(mul_208);  mul_208 = None
        view_1256 = torch.ops.aten.view.default(view_as_real_38, [2, 8192, 1, 128]);  view_as_real_38 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(view_1256, torch.bfloat16);  view_1256 = None
        view_as_real_39 = torch.ops.aten.view_as_real.default(mul_209);  mul_209 = None
        view_1257 = torch.ops.aten.view.default(view_as_real_39, [2, 8192, 4, 128]);  view_as_real_39 = None
        convert_element_type_740 = torch.ops.prims.convert_element_type.default(view_1257, torch.bfloat16);  view_1257 = None
        view_1258 = torch.ops.aten.view.default(squeeze_6, [2, 8192, 128]);  squeeze_6 = None
        view_1259 = torch.ops.aten.view.default(convert_element_type_739, [2, 8192, 128]);  convert_element_type_739 = None
        view_1260 = torch.ops.aten.view.default(convert_element_type_740, [2, 8192, 512]);  convert_element_type_740 = None
        view_1261 = torch.ops.aten.view.default(view_1258, [16384, 128]);  view_1258 = None
        permute_297 = torch.ops.aten.permute.default(view_1261, [1, 0])
        mm_165 = torch.ops.aten.mm.default(permute_297, view_879);  permute_297 = None
        convert_element_type_406 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        all_gather_into_tensor_137 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_406, 8, '0');  convert_element_type_406 = None
        wait_tensor_162 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_137);  all_gather_into_tensor_137 = None
        permute_134 = torch.ops.aten.permute.default(wait_tensor_162, [1, 0]);  wait_tensor_162 = None
        permute_299 = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_166 = torch.ops.aten.mm.default(view_1261, permute_299);  view_1261 = permute_299 = None
        view_1262 = torch.ops.aten.view.default(mm_166, [2, 8192, 4096]);  mm_166 = None
        convert_element_type_745 = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        reduce_scatter_tensor_75 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_745, 'avg', 8, '0');  convert_element_type_745 = None
        wait_tensor_271 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_75);  reduce_scatter_tensor_75 = None
        view_1263 = torch.ops.aten.view.default(view_1259, [16384, 128]);  view_1259 = None
        permute_301 = torch.ops.aten.permute.default(view_1263, [1, 0])
        mm_167 = torch.ops.aten.mm.default(permute_301, view_879);  permute_301 = None
        permute_303 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_168 = torch.ops.aten.mm.default(view_1263, permute_303);  view_1263 = permute_303 = None
        view_1264 = torch.ops.aten.view.default(mm_168, [2, 8192, 4096]);  mm_168 = None
        add_90 = torch.ops.aten.add.Tensor(view_1262, view_1264);  view_1262 = view_1264 = None
        convert_element_type_750 = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        reduce_scatter_tensor_76 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_750, 'avg', 8, '0');  convert_element_type_750 = None
        wait_tensor_272 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_76);  reduce_scatter_tensor_76 = None
        view_1265 = torch.ops.aten.view.default(view_1260, [16384, 512]);  view_1260 = None
        permute_305 = torch.ops.aten.permute.default(view_1265, [1, 0])
        mm_169 = torch.ops.aten.mm.default(permute_305, view_879);  permute_305 = view_879 = None
        convert_element_type_400 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        all_gather_into_tensor_135 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_400, 8, '0');  convert_element_type_400 = None
        wait_tensor_160 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_135);  all_gather_into_tensor_135 = None
        permute_132 = torch.ops.aten.permute.default(wait_tensor_160, [1, 0]);  wait_tensor_160 = None
        permute_307 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_170 = torch.ops.aten.mm.default(view_1265, permute_307);  view_1265 = permute_307 = None
        view_1266 = torch.ops.aten.view.default(mm_170, [2, 8192, 4096]);  mm_170 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, view_1266);  add_90 = view_1266 = None
        convert_element_type_755 = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        reduce_scatter_tensor_77 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_755, 'avg', 8, '0');  convert_element_type_755 = None
        wait_tensor_273 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_77);  reduce_scatter_tensor_77 = None
        split_90 = torch.ops.aten.split.Tensor(add_91, 1024, 1);  add_91 = None
        getitem_876 = split_90[0]
        getitem_877 = split_90[1]
        getitem_878 = split_90[2]
        getitem_879 = split_90[3]
        getitem_880 = split_90[4]
        getitem_881 = split_90[5]
        getitem_882 = split_90[6]
        getitem_883 = split_90[7];  split_90 = None
        cat_82 = torch.ops.aten.cat.default([getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883]);  getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = None
        reduce_scatter_tensor_78 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_82, 'sum', 8, '1');  cat_82 = None
        wait_tensor_274 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_78);  reduce_scatter_tensor_78 = None
        convert_element_type_756 = torch.ops.prims.convert_element_type.default(wait_tensor_274, torch.float32);  wait_tensor_274 = None
        convert_element_type_758 = torch.ops.prims.convert_element_type.default(wait_tensor_158, torch.float32);  wait_tensor_158 = None
        mul_210 = torch.ops.aten.mul.Tensor(convert_element_type_756, convert_element_type_758);  convert_element_type_758 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_96, mul_210)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
        div_8 = torch.ops.aten.div.Tensor(mul_96, 4096)
        mul_213 = torch.ops.aten.mul.Tensor(div_8, sum_25);  div_8 = sum_25 = None
        sub_13 = torch.ops.aten.sub.Tensor(mul_210, mul_213);  mul_210 = mul_213 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_24);  sub_13 = rsqrt_24 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_756, mul_96);  convert_element_type_756 = mul_96 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
        convert_element_type_759 = torch.ops.prims.convert_element_type.default(mul_214, torch.bfloat16);  mul_214 = None
        convert_element_type_760 = torch.ops.prims.convert_element_type.default(sum_26, torch.bfloat16);  sum_26 = None
        all_reduce_8 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_760, 'sum', '1');  convert_element_type_760 = None
        wait_tensor_275 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_8);  all_reduce_8 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(wait_tensor_275, torch.float32);  wait_tensor_275 = None
        reduce_scatter_tensor_79 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_761, 'avg', 8, '0');  convert_element_type_761 = None
        wait_tensor_276 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_79);  reduce_scatter_tensor_79 = None
        add_92 = torch.ops.aten.add.Tensor(add_89, convert_element_type_759);  add_89 = convert_element_type_759 = None
        all_gather_into_tensor_188 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_92, 8, '1')
        wait_tensor_277 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_188);  all_gather_into_tensor_188 = None
        split_91 = torch.ops.aten.split.Tensor(wait_tensor_277, 2);  wait_tensor_277 = None
        getitem_884 = split_91[0]
        getitem_885 = split_91[1]
        getitem_886 = split_91[2]
        getitem_887 = split_91[3]
        getitem_888 = split_91[4]
        getitem_889 = split_91[5]
        getitem_890 = split_91[6]
        getitem_891 = split_91[7];  split_91 = None
        cat_83 = torch.ops.aten.cat.default([getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891], 1);  getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = None
        view_1267 = torch.ops.aten.view.default(cat_83, [16384, 4096]);  cat_83 = None
        permute_309 = torch.ops.aten.permute.default(view_1267, [1, 0])
        wait_tensor_151 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_23);  reduce_scatter_tensor_23 = None
        add_45 = torch.ops.aten.add.Tensor(add_43, wait_tensor_151);  wait_tensor_151 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        all_gather_into_tensor_128 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_383, 8, '0');  convert_element_type_383 = None
        wait_tensor_152 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_128);  all_gather_into_tensor_128 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_384, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [2], True);  pow_24 = None
        add_46 = torch.ops.aten.add.Scalar(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_23);  convert_element_type_384 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, wait_tensor_152)
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
        view_852 = torch.ops.aten.view.default(cat_47, [16384, 4096]);  cat_47 = None
        view_853 = torch.ops.aten.view.default(mm_81, [2, 8192, 1792]);  mm_81 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(view_853, torch.float32);  view_853 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_389)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_389, sigmoid_11);  sigmoid_11 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        all_gather_into_tensor_131 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_391, 8, '0');  convert_element_type_391 = None
        wait_tensor_155 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_131);  all_gather_into_tensor_131 = None
        permute_130 = torch.ops.aten.permute.default(wait_tensor_155, [1, 0]);  wait_tensor_155 = None
        mm_82 = torch.ops.aten.mm.default(view_852, permute_130)
        view_860 = torch.ops.aten.view.default(mm_82, [2, 8192, 1792]);  mm_82 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_390, view_860)
        view_867 = torch.ops.aten.view.default(mul_95, [16384, 1792]);  mul_95 = None
        mm_171 = torch.ops.aten.mm.default(permute_309, view_867);  permute_309 = view_867 = None
        convert_element_type_394 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        all_gather_into_tensor_132 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_394, 8, '0');  convert_element_type_394 = None
        wait_tensor_156 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_132);  all_gather_into_tensor_132 = None
        permute_131 = torch.ops.aten.permute.default(wait_tensor_156, [1, 0]);  wait_tensor_156 = None
        permute_311 = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_172 = torch.ops.aten.mm.default(view_1267, permute_311);  view_1267 = permute_311 = None
        view_1268 = torch.ops.aten.view.default(mm_172, [2, 8192, 1792]);  mm_172 = None
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        reduce_scatter_tensor_80 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_766, 'avg', 8, '0');  convert_element_type_766 = None
        wait_tensor_278 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_80);  reduce_scatter_tensor_80 = None
        mul_216 = torch.ops.aten.mul.Tensor(view_1268, convert_element_type_390);  convert_element_type_390 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_1268, view_860);  view_1268 = view_860 = None
        view_1269 = torch.ops.aten.view.default(mul_216, [16384, 1792]);  mul_216 = None
        permute_313 = torch.ops.aten.permute.default(view_1269, [1, 0])
        mm_173 = torch.ops.aten.mm.default(permute_313, view_852);  permute_313 = None
        permute_315 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_174 = torch.ops.aten.mm.default(view_1269, permute_315);  view_1269 = permute_315 = None
        view_1270 = torch.ops.aten.view.default(mm_174, [2, 8192, 4096]);  mm_174 = None
        convert_element_type_771 = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        reduce_scatter_tensor_81 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_771, 'avg', 8, '0');  convert_element_type_771 = None
        wait_tensor_279 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_81);  reduce_scatter_tensor_81 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(mul_217, torch.float32);  mul_217 = None
        neg_4 = torch.ops.aten.neg.default(convert_element_type_389)
        exp_4 = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_93 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(add_93);  add_93 = None
        mul_218 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_219 = torch.ops.aten.mul.Tensor(convert_element_type_772, mul_218);  convert_element_type_772 = None
        sub_14 = torch.ops.aten.sub.Tensor(1, mul_218);  mul_218 = None
        mul_220 = torch.ops.aten.mul.Tensor(convert_element_type_389, sub_14);  convert_element_type_389 = sub_14 = None
        add_94 = torch.ops.aten.add.Tensor(mul_220, 1);  mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_219, add_94);  mul_219 = add_94 = None
        convert_element_type_774 = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        view_1271 = torch.ops.aten.view.default(convert_element_type_774, [16384, 1792]);  convert_element_type_774 = None
        permute_317 = torch.ops.aten.permute.default(view_1271, [1, 0])
        mm_175 = torch.ops.aten.mm.default(permute_317, view_852);  permute_317 = view_852 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        all_gather_into_tensor_130 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_386, 8, '0');  convert_element_type_386 = None
        wait_tensor_154 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_130);  all_gather_into_tensor_130 = None
        permute_129 = torch.ops.aten.permute.default(wait_tensor_154, [1, 0]);  wait_tensor_154 = None
        permute_319 = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_176 = torch.ops.aten.mm.default(view_1271, permute_319);  view_1271 = permute_319 = None
        view_1272 = torch.ops.aten.view.default(mm_176, [2, 8192, 4096]);  mm_176 = None
        add_95 = torch.ops.aten.add.Tensor(view_1270, view_1272);  view_1270 = view_1272 = None
        convert_element_type_779 = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        reduce_scatter_tensor_82 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_779, 'avg', 8, '0');  convert_element_type_779 = None
        wait_tensor_280 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_82);  reduce_scatter_tensor_82 = None
        split_92 = torch.ops.aten.split.Tensor(add_95, 1024, 1);  add_95 = None
        getitem_892 = split_92[0]
        getitem_893 = split_92[1]
        getitem_894 = split_92[2]
        getitem_895 = split_92[3]
        getitem_896 = split_92[4]
        getitem_897 = split_92[5]
        getitem_898 = split_92[6]
        getitem_899 = split_92[7];  split_92 = None
        cat_84 = torch.ops.aten.cat.default([getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899]);  getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = None
        reduce_scatter_tensor_83 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_84, 'sum', 8, '1');  cat_84 = None
        wait_tensor_281 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_83);  reduce_scatter_tensor_83 = None
        convert_element_type_780 = torch.ops.prims.convert_element_type.default(wait_tensor_281, torch.float32);  wait_tensor_281 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(wait_tensor_152, torch.float32);  wait_tensor_152 = None
        mul_222 = torch.ops.aten.mul.Tensor(convert_element_type_780, convert_element_type_782);  convert_element_type_782 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_92, mul_222)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        div_9 = torch.ops.aten.div.Tensor(mul_92, 4096)
        mul_225 = torch.ops.aten.mul.Tensor(div_9, sum_27);  div_9 = sum_27 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_222, mul_225);  mul_222 = mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_23);  sub_15 = rsqrt_23 = None
        mul_227 = torch.ops.aten.mul.Tensor(convert_element_type_780, mul_92);  convert_element_type_780 = mul_92 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        convert_element_type_783 = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        convert_element_type_784 = torch.ops.prims.convert_element_type.default(sum_28, torch.bfloat16);  sum_28 = None
        all_reduce_9 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_784, 'sum', '1');  convert_element_type_784 = None
        wait_tensor_282 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_9);  all_reduce_9 = None
        convert_element_type_785 = torch.ops.prims.convert_element_type.default(wait_tensor_282, torch.float32);  wait_tensor_282 = None
        reduce_scatter_tensor_84 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_785, 'avg', 8, '0');  convert_element_type_785 = None
        wait_tensor_283 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_84);  reduce_scatter_tensor_84 = None
        add_96 = torch.ops.aten.add.Tensor(add_92, convert_element_type_783);  add_92 = convert_element_type_783 = None
        all_gather_into_tensor_189 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_96, 8, '1')
        wait_tensor_284 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_189);  all_gather_into_tensor_189 = None
        split_93 = torch.ops.aten.split.Tensor(wait_tensor_284, 2);  wait_tensor_284 = None
        getitem_900 = split_93[0]
        getitem_901 = split_93[1]
        getitem_902 = split_93[2]
        getitem_903 = split_93[3]
        getitem_904 = split_93[4]
        getitem_905 = split_93[5]
        getitem_906 = split_93[6]
        getitem_907 = split_93[7];  split_93 = None
        cat_85 = torch.ops.aten.cat.default([getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907], 1);  getitem_900 = getitem_901 = getitem_902 = getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = None
        view_1273 = torch.ops.aten.view.default(cat_85, [16384, 4096]);  cat_85 = None
        permute_321 = torch.ops.aten.permute.default(view_1273, [1, 0])
        permute_127 = torch.ops.aten.permute.default(getitem_531, [0, 2, 1, 3])
        view_834 = torch.ops.aten.view.default(permute_127, [2, 8192, -1]);  permute_127 = None
        view_840 = torch.ops.aten.view.default(view_834, [16384, 512]);  view_834 = None
        mm_177 = torch.ops.aten.mm.default(permute_321, view_840);  permute_321 = view_840 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        all_gather_into_tensor_127 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_380, 8, '0');  convert_element_type_380 = None
        wait_tensor_150 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_127);  all_gather_into_tensor_127 = None
        permute_128 = torch.ops.aten.permute.default(wait_tensor_150, [1, 0]);  wait_tensor_150 = None
        permute_323 = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        mm_178 = torch.ops.aten.mm.default(view_1273, permute_323);  view_1273 = permute_323 = None
        view_1274 = torch.ops.aten.view.default(mm_178, [2, 8192, 512]);  mm_178 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        reduce_scatter_tensor_85 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_790, 'avg', 8, '0');  convert_element_type_790 = None
        wait_tensor_285 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_85);  reduce_scatter_tensor_85 = None
        view_1275 = torch.ops.aten.view.default(view_1274, [2, 8192, 4, 128]);  view_1274 = None
        permute_325 = torch.ops.aten.permute.default(view_1275, [0, 2, 1, 3]);  view_1275 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        all_gather_into_tensor_122 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_364, 8, '0');  convert_element_type_364 = None
        wait_tensor_145 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_122);  all_gather_into_tensor_122 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_365, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [2], True);  pow_23 = None
        add_44 = torch.ops.aten.add.Scalar(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_365, rsqrt_22);  convert_element_type_365 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, wait_tensor_145)
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
        view_807 = torch.ops.aten.view.default(cat_45, [16384, 4096]);  cat_45 = None
        view_808 = torch.ops.aten.view.default(mm_77, [2, 8192, 512]);  mm_77 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        all_gather_into_tensor_125 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_370, 8, '0');  convert_element_type_370 = None
        wait_tensor_148 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_125);  all_gather_into_tensor_125 = None
        permute_122 = torch.ops.aten.permute.default(wait_tensor_148, [1, 0]);  wait_tensor_148 = None
        mm_78 = torch.ops.aten.mm.default(view_807, permute_122)
        view_815 = torch.ops.aten.view.default(mm_78, [2, 8192, 128]);  mm_78 = None
        view_822 = torch.ops.aten.view.default(mm_79, [2, 8192, 128]);  mm_79 = None
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
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_325, permute_124, permute_125, permute_126, getitem_531, getitem_532, getitem_537, getitem_538, None, None, None, 8192, 8192, 0.0, True);  permute_325 = permute_124 = permute_125 = permute_126 = getitem_531 = getitem_532 = getitem_537 = getitem_538 = None
        getitem_908 = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_909 = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_910 = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None
        permute_326 = torch.ops.aten.permute.default(getitem_910, [0, 2, 1, 3]);  getitem_910 = None
        permute_327 = torch.ops.aten.permute.default(getitem_909, [0, 2, 1, 3]);  getitem_909 = None
        permute_328 = torch.ops.aten.permute.default(getitem_908, [0, 2, 1, 3]);  getitem_908 = None
        view_1276 = torch.ops.aten.view.default(permute_326, [2, 8192, 1, 4, 128]);  permute_326 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(view_1276, [3], True);  view_1276 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(sum_29, 3);  sum_29 = None
        view_1277 = torch.ops.aten.view.default(permute_327, [2, 8192, 1, 4, 128]);  permute_327 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(view_1277, [3], True);  view_1277 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(sum_30, 3);  sum_30 = None
        convert_element_type_791 = torch.ops.prims.convert_element_type.default(squeeze_9, torch.float32);  squeeze_9 = None
        convert_element_type_792 = torch.ops.prims.convert_element_type.default(permute_328, torch.float32);  permute_328 = None
        view_1278 = torch.ops.aten.view.default(convert_element_type_791, [2, 8192, 1, 64, 2]);  convert_element_type_791 = None
        view_as_complex_40 = torch.ops.aten.view_as_complex.default(view_1278);  view_1278 = None
        mul_228 = torch.ops.aten.mul.Tensor(view_as_complex_40, _conj);  view_as_complex_40 = None
        view_1279 = torch.ops.aten.view.default(convert_element_type_792, [2, 8192, 4, 64, 2]);  convert_element_type_792 = None
        view_as_complex_41 = torch.ops.aten.view_as_complex.default(view_1279);  view_1279 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_as_complex_41, _conj);  view_as_complex_41 = None
        view_as_real_40 = torch.ops.aten.view_as_real.default(mul_228);  mul_228 = None
        view_1280 = torch.ops.aten.view.default(view_as_real_40, [2, 8192, 1, 128]);  view_as_real_40 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(view_1280, torch.bfloat16);  view_1280 = None
        view_as_real_41 = torch.ops.aten.view_as_real.default(mul_229);  mul_229 = None
        view_1281 = torch.ops.aten.view.default(view_as_real_41, [2, 8192, 4, 128]);  view_as_real_41 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(view_1281, torch.bfloat16);  view_1281 = None
        view_1282 = torch.ops.aten.view.default(squeeze_8, [2, 8192, 128]);  squeeze_8 = None
        view_1283 = torch.ops.aten.view.default(convert_element_type_793, [2, 8192, 128]);  convert_element_type_793 = None
        view_1284 = torch.ops.aten.view.default(convert_element_type_794, [2, 8192, 512]);  convert_element_type_794 = None
        view_1285 = torch.ops.aten.view.default(view_1282, [16384, 128]);  view_1282 = None
        permute_329 = torch.ops.aten.permute.default(view_1285, [1, 0])
        mm_179 = torch.ops.aten.mm.default(permute_329, view_807);  permute_329 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        all_gather_into_tensor_126 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_373, 8, '0');  convert_element_type_373 = None
        wait_tensor_149 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_126);  all_gather_into_tensor_126 = None
        permute_123 = torch.ops.aten.permute.default(wait_tensor_149, [1, 0]);  wait_tensor_149 = None
        permute_331 = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_180 = torch.ops.aten.mm.default(view_1285, permute_331);  view_1285 = permute_331 = None
        view_1286 = torch.ops.aten.view.default(mm_180, [2, 8192, 4096]);  mm_180 = None
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        reduce_scatter_tensor_86 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_799, 'avg', 8, '0');  convert_element_type_799 = None
        wait_tensor_286 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_86);  reduce_scatter_tensor_86 = None
        view_1287 = torch.ops.aten.view.default(view_1283, [16384, 128]);  view_1283 = None
        permute_333 = torch.ops.aten.permute.default(view_1287, [1, 0])
        mm_181 = torch.ops.aten.mm.default(permute_333, view_807);  permute_333 = None
        permute_335 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_182 = torch.ops.aten.mm.default(view_1287, permute_335);  view_1287 = permute_335 = None
        view_1288 = torch.ops.aten.view.default(mm_182, [2, 8192, 4096]);  mm_182 = None
        add_97 = torch.ops.aten.add.Tensor(view_1286, view_1288);  view_1286 = view_1288 = None
        convert_element_type_804 = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        reduce_scatter_tensor_87 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_804, 'avg', 8, '0');  convert_element_type_804 = None
        wait_tensor_287 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_87);  reduce_scatter_tensor_87 = None
        view_1289 = torch.ops.aten.view.default(view_1284, [16384, 512]);  view_1284 = None
        permute_337 = torch.ops.aten.permute.default(view_1289, [1, 0])
        mm_183 = torch.ops.aten.mm.default(permute_337, view_807);  permute_337 = view_807 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        all_gather_into_tensor_124 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_367, 8, '0');  convert_element_type_367 = None
        wait_tensor_147 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_124);  all_gather_into_tensor_124 = None
        permute_121 = torch.ops.aten.permute.default(wait_tensor_147, [1, 0]);  wait_tensor_147 = None
        permute_339 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_184 = torch.ops.aten.mm.default(view_1289, permute_339);  view_1289 = permute_339 = None
        view_1290 = torch.ops.aten.view.default(mm_184, [2, 8192, 4096]);  mm_184 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, view_1290);  add_97 = view_1290 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        reduce_scatter_tensor_88 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_809, 'avg', 8, '0');  convert_element_type_809 = None
        wait_tensor_288 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_88);  reduce_scatter_tensor_88 = None
        split_94 = torch.ops.aten.split.Tensor(add_98, 1024, 1);  add_98 = None
        getitem_911 = split_94[0]
        getitem_912 = split_94[1]
        getitem_913 = split_94[2]
        getitem_914 = split_94[3]
        getitem_915 = split_94[4]
        getitem_916 = split_94[5]
        getitem_917 = split_94[6]
        getitem_918 = split_94[7];  split_94 = None
        cat_86 = torch.ops.aten.cat.default([getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918]);  getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = getitem_917 = getitem_918 = None
        reduce_scatter_tensor_89 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_86, 'sum', 8, '1');  cat_86 = None
        wait_tensor_289 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_89);  reduce_scatter_tensor_89 = None
        convert_element_type_810 = torch.ops.prims.convert_element_type.default(wait_tensor_289, torch.float32);  wait_tensor_289 = None
        convert_element_type_812 = torch.ops.prims.convert_element_type.default(wait_tensor_145, torch.float32);  wait_tensor_145 = None
        mul_230 = torch.ops.aten.mul.Tensor(convert_element_type_810, convert_element_type_812);  convert_element_type_812 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_88, mul_230)
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_232, [2], True);  mul_232 = None
        div_10 = torch.ops.aten.div.Tensor(mul_88, 4096)
        mul_233 = torch.ops.aten.mul.Tensor(div_10, sum_31);  div_10 = sum_31 = None
        sub_16 = torch.ops.aten.sub.Tensor(mul_230, mul_233);  mul_230 = mul_233 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_22);  sub_16 = rsqrt_22 = None
        mul_235 = torch.ops.aten.mul.Tensor(convert_element_type_810, mul_88);  convert_element_type_810 = mul_88 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1]);  mul_235 = None
        convert_element_type_813 = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        convert_element_type_814 = torch.ops.prims.convert_element_type.default(sum_32, torch.bfloat16);  sum_32 = None
        all_reduce_10 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_814, 'sum', '1');  convert_element_type_814 = None
        wait_tensor_290 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_10);  all_reduce_10 = None
        convert_element_type_815 = torch.ops.prims.convert_element_type.default(wait_tensor_290, torch.float32);  wait_tensor_290 = None
        reduce_scatter_tensor_90 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_815, 'avg', 8, '0');  convert_element_type_815 = None
        wait_tensor_291 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_90);  reduce_scatter_tensor_90 = None
        add_99 = torch.ops.aten.add.Tensor(add_96, convert_element_type_813);  add_96 = convert_element_type_813 = None
        all_gather_into_tensor_190 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_99, 8, '1')
        wait_tensor_292 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_190);  all_gather_into_tensor_190 = None
        split_95 = torch.ops.aten.split.Tensor(wait_tensor_292, 2);  wait_tensor_292 = None
        getitem_919 = split_95[0]
        getitem_920 = split_95[1]
        getitem_921 = split_95[2]
        getitem_922 = split_95[3]
        getitem_923 = split_95[4]
        getitem_924 = split_95[5]
        getitem_925 = split_95[6]
        getitem_926 = split_95[7];  split_95 = None
        cat_87 = torch.ops.aten.cat.default([getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926], 1);  getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = None
        view_1291 = torch.ops.aten.view.default(cat_87, [16384, 4096]);  cat_87 = None
        permute_341 = torch.ops.aten.permute.default(view_1291, [1, 0])
        wait_tensor_138 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_21);  reduce_scatter_tensor_21 = None
        add_41 = torch.ops.aten.add.Tensor(add_39, wait_tensor_138);  wait_tensor_138 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        all_gather_into_tensor_117 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_350, 8, '0');  convert_element_type_350 = None
        wait_tensor_139 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_117);  all_gather_into_tensor_117 = None
        convert_element_type_351 = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_351, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [2], True);  pow_22 = None
        add_42 = torch.ops.aten.add.Scalar(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_351, rsqrt_21);  convert_element_type_351 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, wait_tensor_139)
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
        view_780 = torch.ops.aten.view.default(cat_43, [16384, 4096]);  cat_43 = None
        view_781 = torch.ops.aten.view.default(mm_74, [2, 8192, 1792]);  mm_74 = None
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_356)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_356, sigmoid_10);  sigmoid_10 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        all_gather_into_tensor_120 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_358, 8, '0');  convert_element_type_358 = None
        wait_tensor_142 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_120);  all_gather_into_tensor_120 = None
        permute_119 = torch.ops.aten.permute.default(wait_tensor_142, [1, 0]);  wait_tensor_142 = None
        mm_75 = torch.ops.aten.mm.default(view_780, permute_119)
        view_788 = torch.ops.aten.view.default(mm_75, [2, 8192, 1792]);  mm_75 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_357, view_788)
        view_795 = torch.ops.aten.view.default(mul_87, [16384, 1792]);  mul_87 = None
        mm_185 = torch.ops.aten.mm.default(permute_341, view_795);  permute_341 = view_795 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        all_gather_into_tensor_121 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_361, 8, '0');  convert_element_type_361 = None
        wait_tensor_143 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_121);  all_gather_into_tensor_121 = None
        permute_120 = torch.ops.aten.permute.default(wait_tensor_143, [1, 0]);  wait_tensor_143 = None
        permute_343 = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_186 = torch.ops.aten.mm.default(view_1291, permute_343);  view_1291 = permute_343 = None
        view_1292 = torch.ops.aten.view.default(mm_186, [2, 8192, 1792]);  mm_186 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        reduce_scatter_tensor_91 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_820, 'avg', 8, '0');  convert_element_type_820 = None
        wait_tensor_293 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_91);  reduce_scatter_tensor_91 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_1292, convert_element_type_357);  convert_element_type_357 = None
        mul_237 = torch.ops.aten.mul.Tensor(view_1292, view_788);  view_1292 = view_788 = None
        view_1293 = torch.ops.aten.view.default(mul_236, [16384, 1792]);  mul_236 = None
        permute_345 = torch.ops.aten.permute.default(view_1293, [1, 0])
        mm_187 = torch.ops.aten.mm.default(permute_345, view_780);  permute_345 = None
        permute_347 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_188 = torch.ops.aten.mm.default(view_1293, permute_347);  view_1293 = permute_347 = None
        view_1294 = torch.ops.aten.view.default(mm_188, [2, 8192, 4096]);  mm_188 = None
        convert_element_type_825 = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        reduce_scatter_tensor_92 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_825, 'avg', 8, '0');  convert_element_type_825 = None
        wait_tensor_294 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_92);  reduce_scatter_tensor_92 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(mul_237, torch.float32);  mul_237 = None
        neg_5 = torch.ops.aten.neg.default(convert_element_type_356)
        exp_5 = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_100 = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(add_100);  add_100 = None
        mul_238 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_826, mul_238);  convert_element_type_826 = None
        sub_17 = torch.ops.aten.sub.Tensor(1, mul_238);  mul_238 = None
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_356, sub_17);  convert_element_type_356 = sub_17 = None
        add_101 = torch.ops.aten.add.Tensor(mul_240, 1);  mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_239, add_101);  mul_239 = add_101 = None
        convert_element_type_828 = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        view_1295 = torch.ops.aten.view.default(convert_element_type_828, [16384, 1792]);  convert_element_type_828 = None
        permute_349 = torch.ops.aten.permute.default(view_1295, [1, 0])
        mm_189 = torch.ops.aten.mm.default(permute_349, view_780);  permute_349 = view_780 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        all_gather_into_tensor_119 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_353, 8, '0');  convert_element_type_353 = None
        wait_tensor_141 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_119);  all_gather_into_tensor_119 = None
        permute_118 = torch.ops.aten.permute.default(wait_tensor_141, [1, 0]);  wait_tensor_141 = None
        permute_351 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_190 = torch.ops.aten.mm.default(view_1295, permute_351);  view_1295 = permute_351 = None
        view_1296 = torch.ops.aten.view.default(mm_190, [2, 8192, 4096]);  mm_190 = None
        add_102 = torch.ops.aten.add.Tensor(view_1294, view_1296);  view_1294 = view_1296 = None
        convert_element_type_833 = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        reduce_scatter_tensor_93 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_833, 'avg', 8, '0');  convert_element_type_833 = None
        wait_tensor_295 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_93);  reduce_scatter_tensor_93 = None
        split_96 = torch.ops.aten.split.Tensor(add_102, 1024, 1);  add_102 = None
        getitem_927 = split_96[0]
        getitem_928 = split_96[1]
        getitem_929 = split_96[2]
        getitem_930 = split_96[3]
        getitem_931 = split_96[4]
        getitem_932 = split_96[5]
        getitem_933 = split_96[6]
        getitem_934 = split_96[7];  split_96 = None
        cat_88 = torch.ops.aten.cat.default([getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934]);  getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = getitem_933 = getitem_934 = None
        reduce_scatter_tensor_94 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_88, 'sum', 8, '1');  cat_88 = None
        wait_tensor_296 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_94);  reduce_scatter_tensor_94 = None
        convert_element_type_834 = torch.ops.prims.convert_element_type.default(wait_tensor_296, torch.float32);  wait_tensor_296 = None
        convert_element_type_836 = torch.ops.prims.convert_element_type.default(wait_tensor_139, torch.float32);  wait_tensor_139 = None
        mul_242 = torch.ops.aten.mul.Tensor(convert_element_type_834, convert_element_type_836);  convert_element_type_836 = None
        mul_244 = torch.ops.aten.mul.Tensor(mul_84, mul_242)
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
        div_11 = torch.ops.aten.div.Tensor(mul_84, 4096)
        mul_245 = torch.ops.aten.mul.Tensor(div_11, sum_33);  div_11 = sum_33 = None
        sub_18 = torch.ops.aten.sub.Tensor(mul_242, mul_245);  mul_242 = mul_245 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_21);  sub_18 = rsqrt_21 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_834, mul_84);  convert_element_type_834 = mul_84 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
        convert_element_type_837 = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        convert_element_type_838 = torch.ops.prims.convert_element_type.default(sum_34, torch.bfloat16);  sum_34 = None
        all_reduce_11 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_838, 'sum', '1');  convert_element_type_838 = None
        wait_tensor_297 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_11);  all_reduce_11 = None
        convert_element_type_839 = torch.ops.prims.convert_element_type.default(wait_tensor_297, torch.float32);  wait_tensor_297 = None
        reduce_scatter_tensor_95 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_839, 'avg', 8, '0');  convert_element_type_839 = None
        wait_tensor_298 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_95);  reduce_scatter_tensor_95 = None
        add_103 = torch.ops.aten.add.Tensor(add_99, convert_element_type_837);  add_99 = convert_element_type_837 = None
        all_gather_into_tensor_191 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_103, 8, '1')
        wait_tensor_299 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_191);  all_gather_into_tensor_191 = None
        split_97 = torch.ops.aten.split.Tensor(wait_tensor_299, 2);  wait_tensor_299 = None
        getitem_935 = split_97[0]
        getitem_936 = split_97[1]
        getitem_937 = split_97[2]
        getitem_938 = split_97[3]
        getitem_939 = split_97[4]
        getitem_940 = split_97[5]
        getitem_941 = split_97[6]
        getitem_942 = split_97[7];  split_97 = None
        cat_89 = torch.ops.aten.cat.default([getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942], 1);  getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = getitem_941 = getitem_942 = None
        view_1297 = torch.ops.aten.view.default(cat_89, [16384, 4096]);  cat_89 = None
        permute_353 = torch.ops.aten.permute.default(view_1297, [1, 0])
        permute_116 = torch.ops.aten.permute.default(getitem_490, [0, 2, 1, 3])
        view_762 = torch.ops.aten.view.default(permute_116, [2, 8192, -1]);  permute_116 = None
        view_768 = torch.ops.aten.view.default(view_762, [16384, 512]);  view_762 = None
        mm_191 = torch.ops.aten.mm.default(permute_353, view_768);  permute_353 = view_768 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        all_gather_into_tensor_116 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_347, 8, '0');  convert_element_type_347 = None
        wait_tensor_137 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_116);  all_gather_into_tensor_116 = None
        permute_117 = torch.ops.aten.permute.default(wait_tensor_137, [1, 0]);  wait_tensor_137 = None
        permute_355 = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        mm_192 = torch.ops.aten.mm.default(view_1297, permute_355);  view_1297 = permute_355 = None
        view_1298 = torch.ops.aten.view.default(mm_192, [2, 8192, 512]);  mm_192 = None
        convert_element_type_844 = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        reduce_scatter_tensor_96 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_844, 'avg', 8, '0');  convert_element_type_844 = None
        wait_tensor_300 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_96);  reduce_scatter_tensor_96 = None
        view_1299 = torch.ops.aten.view.default(view_1298, [2, 8192, 4, 128]);  view_1298 = None
        permute_357 = torch.ops.aten.permute.default(view_1299, [0, 2, 1, 3]);  view_1299 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        all_gather_into_tensor_111 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_331, 8, '0');  convert_element_type_331 = None
        wait_tensor_132 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_111);  all_gather_into_tensor_111 = None
        convert_element_type_332 = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [2], True);  pow_21 = None
        add_40 = torch.ops.aten.add.Scalar(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_332, rsqrt_20);  convert_element_type_332 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, wait_tensor_132)
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
        view_735 = torch.ops.aten.view.default(cat_41, [16384, 4096]);  cat_41 = None
        view_736 = torch.ops.aten.view.default(mm_70, [2, 8192, 512]);  mm_70 = None
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        all_gather_into_tensor_114 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_337, 8, '0');  convert_element_type_337 = None
        wait_tensor_135 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_114);  all_gather_into_tensor_114 = None
        permute_111 = torch.ops.aten.permute.default(wait_tensor_135, [1, 0]);  wait_tensor_135 = None
        mm_71 = torch.ops.aten.mm.default(view_735, permute_111)
        view_743 = torch.ops.aten.view.default(mm_71, [2, 8192, 128]);  mm_71 = None
        view_750 = torch.ops.aten.view.default(mm_72, [2, 8192, 128]);  mm_72 = None
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
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_357, permute_113, permute_114, permute_115, getitem_490, getitem_491, getitem_496, getitem_497, None, None, None, 8192, 8192, 0.0, True);  permute_357 = permute_113 = permute_114 = permute_115 = getitem_490 = getitem_491 = getitem_496 = getitem_497 = None
        getitem_943 = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_944 = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_945 = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None
        permute_358 = torch.ops.aten.permute.default(getitem_945, [0, 2, 1, 3]);  getitem_945 = None
        permute_359 = torch.ops.aten.permute.default(getitem_944, [0, 2, 1, 3]);  getitem_944 = None
        permute_360 = torch.ops.aten.permute.default(getitem_943, [0, 2, 1, 3]);  getitem_943 = None
        view_1300 = torch.ops.aten.view.default(permute_358, [2, 8192, 1, 4, 128]);  permute_358 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(view_1300, [3], True);  view_1300 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(sum_35, 3);  sum_35 = None
        view_1301 = torch.ops.aten.view.default(permute_359, [2, 8192, 1, 4, 128]);  permute_359 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(view_1301, [3], True);  view_1301 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(sum_36, 3);  sum_36 = None
        convert_element_type_845 = torch.ops.prims.convert_element_type.default(squeeze_11, torch.float32);  squeeze_11 = None
        convert_element_type_846 = torch.ops.prims.convert_element_type.default(permute_360, torch.float32);  permute_360 = None
        view_1302 = torch.ops.aten.view.default(convert_element_type_845, [2, 8192, 1, 64, 2]);  convert_element_type_845 = None
        view_as_complex_42 = torch.ops.aten.view_as_complex.default(view_1302);  view_1302 = None
        mul_248 = torch.ops.aten.mul.Tensor(view_as_complex_42, _conj);  view_as_complex_42 = None
        view_1303 = torch.ops.aten.view.default(convert_element_type_846, [2, 8192, 4, 64, 2]);  convert_element_type_846 = None
        view_as_complex_43 = torch.ops.aten.view_as_complex.default(view_1303);  view_1303 = None
        mul_249 = torch.ops.aten.mul.Tensor(view_as_complex_43, _conj);  view_as_complex_43 = None
        view_as_real_42 = torch.ops.aten.view_as_real.default(mul_248);  mul_248 = None
        view_1304 = torch.ops.aten.view.default(view_as_real_42, [2, 8192, 1, 128]);  view_as_real_42 = None
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(view_1304, torch.bfloat16);  view_1304 = None
        view_as_real_43 = torch.ops.aten.view_as_real.default(mul_249);  mul_249 = None
        view_1305 = torch.ops.aten.view.default(view_as_real_43, [2, 8192, 4, 128]);  view_as_real_43 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(view_1305, torch.bfloat16);  view_1305 = None
        view_1306 = torch.ops.aten.view.default(squeeze_10, [2, 8192, 128]);  squeeze_10 = None
        view_1307 = torch.ops.aten.view.default(convert_element_type_847, [2, 8192, 128]);  convert_element_type_847 = None
        view_1308 = torch.ops.aten.view.default(convert_element_type_848, [2, 8192, 512]);  convert_element_type_848 = None
        view_1309 = torch.ops.aten.view.default(view_1306, [16384, 128]);  view_1306 = None
        permute_361 = torch.ops.aten.permute.default(view_1309, [1, 0])
        mm_193 = torch.ops.aten.mm.default(permute_361, view_735);  permute_361 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        all_gather_into_tensor_115 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_340, 8, '0');  convert_element_type_340 = None
        wait_tensor_136 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_115);  all_gather_into_tensor_115 = None
        permute_112 = torch.ops.aten.permute.default(wait_tensor_136, [1, 0]);  wait_tensor_136 = None
        permute_363 = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_194 = torch.ops.aten.mm.default(view_1309, permute_363);  view_1309 = permute_363 = None
        view_1310 = torch.ops.aten.view.default(mm_194, [2, 8192, 4096]);  mm_194 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        reduce_scatter_tensor_97 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_853, 'avg', 8, '0');  convert_element_type_853 = None
        wait_tensor_301 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_97);  reduce_scatter_tensor_97 = None
        view_1311 = torch.ops.aten.view.default(view_1307, [16384, 128]);  view_1307 = None
        permute_365 = torch.ops.aten.permute.default(view_1311, [1, 0])
        mm_195 = torch.ops.aten.mm.default(permute_365, view_735);  permute_365 = None
        permute_367 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_196 = torch.ops.aten.mm.default(view_1311, permute_367);  view_1311 = permute_367 = None
        view_1312 = torch.ops.aten.view.default(mm_196, [2, 8192, 4096]);  mm_196 = None
        add_104 = torch.ops.aten.add.Tensor(view_1310, view_1312);  view_1310 = view_1312 = None
        convert_element_type_858 = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        reduce_scatter_tensor_98 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_858, 'avg', 8, '0');  convert_element_type_858 = None
        wait_tensor_302 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_98);  reduce_scatter_tensor_98 = None
        view_1313 = torch.ops.aten.view.default(view_1308, [16384, 512]);  view_1308 = None
        permute_369 = torch.ops.aten.permute.default(view_1313, [1, 0])
        mm_197 = torch.ops.aten.mm.default(permute_369, view_735);  permute_369 = view_735 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        all_gather_into_tensor_113 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_334, 8, '0');  convert_element_type_334 = None
        wait_tensor_134 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_113);  all_gather_into_tensor_113 = None
        permute_110 = torch.ops.aten.permute.default(wait_tensor_134, [1, 0]);  wait_tensor_134 = None
        permute_371 = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_198 = torch.ops.aten.mm.default(view_1313, permute_371);  view_1313 = permute_371 = None
        view_1314 = torch.ops.aten.view.default(mm_198, [2, 8192, 4096]);  mm_198 = None
        add_105 = torch.ops.aten.add.Tensor(add_104, view_1314);  add_104 = view_1314 = None
        convert_element_type_863 = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        reduce_scatter_tensor_99 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_863, 'avg', 8, '0');  convert_element_type_863 = None
        wait_tensor_303 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_99);  reduce_scatter_tensor_99 = None
        split_98 = torch.ops.aten.split.Tensor(add_105, 1024, 1);  add_105 = None
        getitem_946 = split_98[0]
        getitem_947 = split_98[1]
        getitem_948 = split_98[2]
        getitem_949 = split_98[3]
        getitem_950 = split_98[4]
        getitem_951 = split_98[5]
        getitem_952 = split_98[6]
        getitem_953 = split_98[7];  split_98 = None
        cat_90 = torch.ops.aten.cat.default([getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953]);  getitem_946 = getitem_947 = getitem_948 = getitem_949 = getitem_950 = getitem_951 = getitem_952 = getitem_953 = None
        reduce_scatter_tensor_100 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_90, 'sum', 8, '1');  cat_90 = None
        wait_tensor_304 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_100);  reduce_scatter_tensor_100 = None
        convert_element_type_864 = torch.ops.prims.convert_element_type.default(wait_tensor_304, torch.float32);  wait_tensor_304 = None
        convert_element_type_866 = torch.ops.prims.convert_element_type.default(wait_tensor_132, torch.float32);  wait_tensor_132 = None
        mul_250 = torch.ops.aten.mul.Tensor(convert_element_type_864, convert_element_type_866);  convert_element_type_866 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_80, mul_250)
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        div_12 = torch.ops.aten.div.Tensor(mul_80, 4096)
        mul_253 = torch.ops.aten.mul.Tensor(div_12, sum_37);  div_12 = sum_37 = None
        sub_19 = torch.ops.aten.sub.Tensor(mul_250, mul_253);  mul_250 = mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_20);  sub_19 = rsqrt_20 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_864, mul_80);  convert_element_type_864 = mul_80 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        convert_element_type_867 = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        convert_element_type_868 = torch.ops.prims.convert_element_type.default(sum_38, torch.bfloat16);  sum_38 = None
        all_reduce_12 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_868, 'sum', '1');  convert_element_type_868 = None
        wait_tensor_305 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_12);  all_reduce_12 = None
        convert_element_type_869 = torch.ops.prims.convert_element_type.default(wait_tensor_305, torch.float32);  wait_tensor_305 = None
        reduce_scatter_tensor_101 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_869, 'avg', 8, '0');  convert_element_type_869 = None
        wait_tensor_306 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_101);  reduce_scatter_tensor_101 = None
        add_106 = torch.ops.aten.add.Tensor(add_103, convert_element_type_867);  add_103 = convert_element_type_867 = None
        all_gather_into_tensor_192 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_106, 8, '1')
        wait_tensor_307 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_192);  all_gather_into_tensor_192 = None
        split_99 = torch.ops.aten.split.Tensor(wait_tensor_307, 2);  wait_tensor_307 = None
        getitem_954 = split_99[0]
        getitem_955 = split_99[1]
        getitem_956 = split_99[2]
        getitem_957 = split_99[3]
        getitem_958 = split_99[4]
        getitem_959 = split_99[5]
        getitem_960 = split_99[6]
        getitem_961 = split_99[7];  split_99 = None
        cat_91 = torch.ops.aten.cat.default([getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961], 1);  getitem_954 = getitem_955 = getitem_956 = getitem_957 = getitem_958 = getitem_959 = getitem_960 = getitem_961 = None
        view_1315 = torch.ops.aten.view.default(cat_91, [16384, 4096]);  cat_91 = None
        permute_373 = torch.ops.aten.permute.default(view_1315, [1, 0])
        wait_tensor_125 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_19);  reduce_scatter_tensor_19 = None
        add_37 = torch.ops.aten.add.Tensor(add_35, wait_tensor_125);  wait_tensor_125 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        all_gather_into_tensor_106 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_317, 8, '0');  convert_element_type_317 = None
        wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_106);  all_gather_into_tensor_106 = None
        convert_element_type_318 = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_318, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [2], True);  pow_20 = None
        add_38 = torch.ops.aten.add.Scalar(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(convert_element_type_318, rsqrt_19);  convert_element_type_318 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, wait_tensor_126)
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
        view_708 = torch.ops.aten.view.default(cat_39, [16384, 4096]);  cat_39 = None
        view_709 = torch.ops.aten.view.default(mm_67, [2, 8192, 1792]);  mm_67 = None
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_323)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_323, sigmoid_9);  sigmoid_9 = None
        convert_element_type_324 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        all_gather_into_tensor_109 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_325, 8, '0');  convert_element_type_325 = None
        wait_tensor_129 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_109);  all_gather_into_tensor_109 = None
        permute_108 = torch.ops.aten.permute.default(wait_tensor_129, [1, 0]);  wait_tensor_129 = None
        mm_68 = torch.ops.aten.mm.default(view_708, permute_108)
        view_716 = torch.ops.aten.view.default(mm_68, [2, 8192, 1792]);  mm_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_324, view_716)
        view_723 = torch.ops.aten.view.default(mul_79, [16384, 1792]);  mul_79 = None
        mm_199 = torch.ops.aten.mm.default(permute_373, view_723);  permute_373 = view_723 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        all_gather_into_tensor_110 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_328, 8, '0');  convert_element_type_328 = None
        wait_tensor_130 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_110);  all_gather_into_tensor_110 = None
        permute_109 = torch.ops.aten.permute.default(wait_tensor_130, [1, 0]);  wait_tensor_130 = None
        permute_375 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_200 = torch.ops.aten.mm.default(view_1315, permute_375);  view_1315 = permute_375 = None
        view_1316 = torch.ops.aten.view.default(mm_200, [2, 8192, 1792]);  mm_200 = None
        convert_element_type_874 = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        reduce_scatter_tensor_102 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_874, 'avg', 8, '0');  convert_element_type_874 = None
        wait_tensor_308 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_102);  reduce_scatter_tensor_102 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_1316, convert_element_type_324);  convert_element_type_324 = None
        mul_257 = torch.ops.aten.mul.Tensor(view_1316, view_716);  view_1316 = view_716 = None
        view_1317 = torch.ops.aten.view.default(mul_256, [16384, 1792]);  mul_256 = None
        permute_377 = torch.ops.aten.permute.default(view_1317, [1, 0])
        mm_201 = torch.ops.aten.mm.default(permute_377, view_708);  permute_377 = None
        permute_379 = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_202 = torch.ops.aten.mm.default(view_1317, permute_379);  view_1317 = permute_379 = None
        view_1318 = torch.ops.aten.view.default(mm_202, [2, 8192, 4096]);  mm_202 = None
        convert_element_type_879 = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        reduce_scatter_tensor_103 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_879, 'avg', 8, '0');  convert_element_type_879 = None
        wait_tensor_309 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_103);  reduce_scatter_tensor_103 = None
        convert_element_type_880 = torch.ops.prims.convert_element_type.default(mul_257, torch.float32);  mul_257 = None
        neg_6 = torch.ops.aten.neg.default(convert_element_type_323)
        exp_6 = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_107 = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(add_107);  add_107 = None
        mul_258 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_259 = torch.ops.aten.mul.Tensor(convert_element_type_880, mul_258);  convert_element_type_880 = None
        sub_20 = torch.ops.aten.sub.Tensor(1, mul_258);  mul_258 = None
        mul_260 = torch.ops.aten.mul.Tensor(convert_element_type_323, sub_20);  convert_element_type_323 = sub_20 = None
        add_108 = torch.ops.aten.add.Tensor(mul_260, 1);  mul_260 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_259, add_108);  mul_259 = add_108 = None
        convert_element_type_882 = torch.ops.prims.convert_element_type.default(mul_261, torch.bfloat16);  mul_261 = None
        view_1319 = torch.ops.aten.view.default(convert_element_type_882, [16384, 1792]);  convert_element_type_882 = None
        permute_381 = torch.ops.aten.permute.default(view_1319, [1, 0])
        mm_203 = torch.ops.aten.mm.default(permute_381, view_708);  permute_381 = view_708 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        all_gather_into_tensor_108 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_320, 8, '0');  convert_element_type_320 = None
        wait_tensor_128 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_108);  all_gather_into_tensor_108 = None
        permute_107 = torch.ops.aten.permute.default(wait_tensor_128, [1, 0]);  wait_tensor_128 = None
        permute_383 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_204 = torch.ops.aten.mm.default(view_1319, permute_383);  view_1319 = permute_383 = None
        view_1320 = torch.ops.aten.view.default(mm_204, [2, 8192, 4096]);  mm_204 = None
        add_109 = torch.ops.aten.add.Tensor(view_1318, view_1320);  view_1318 = view_1320 = None
        convert_element_type_887 = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        reduce_scatter_tensor_104 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_887, 'avg', 8, '0');  convert_element_type_887 = None
        wait_tensor_310 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_104);  reduce_scatter_tensor_104 = None
        split_100 = torch.ops.aten.split.Tensor(add_109, 1024, 1);  add_109 = None
        getitem_962 = split_100[0]
        getitem_963 = split_100[1]
        getitem_964 = split_100[2]
        getitem_965 = split_100[3]
        getitem_966 = split_100[4]
        getitem_967 = split_100[5]
        getitem_968 = split_100[6]
        getitem_969 = split_100[7];  split_100 = None
        cat_92 = torch.ops.aten.cat.default([getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969]);  getitem_962 = getitem_963 = getitem_964 = getitem_965 = getitem_966 = getitem_967 = getitem_968 = getitem_969 = None
        reduce_scatter_tensor_105 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_92, 'sum', 8, '1');  cat_92 = None
        wait_tensor_311 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_105);  reduce_scatter_tensor_105 = None
        convert_element_type_888 = torch.ops.prims.convert_element_type.default(wait_tensor_311, torch.float32);  wait_tensor_311 = None
        convert_element_type_890 = torch.ops.prims.convert_element_type.default(wait_tensor_126, torch.float32);  wait_tensor_126 = None
        mul_262 = torch.ops.aten.mul.Tensor(convert_element_type_888, convert_element_type_890);  convert_element_type_890 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_76, mul_262)
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
        div_13 = torch.ops.aten.div.Tensor(mul_76, 4096)
        mul_265 = torch.ops.aten.mul.Tensor(div_13, sum_39);  div_13 = sum_39 = None
        sub_21 = torch.ops.aten.sub.Tensor(mul_262, mul_265);  mul_262 = mul_265 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_267 = torch.ops.aten.mul.Tensor(convert_element_type_888, mul_76);  convert_element_type_888 = mul_76 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1]);  mul_267 = None
        convert_element_type_891 = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None
        convert_element_type_892 = torch.ops.prims.convert_element_type.default(sum_40, torch.bfloat16);  sum_40 = None
        all_reduce_13 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_892, 'sum', '1');  convert_element_type_892 = None
        wait_tensor_312 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_13);  all_reduce_13 = None
        convert_element_type_893 = torch.ops.prims.convert_element_type.default(wait_tensor_312, torch.float32);  wait_tensor_312 = None
        reduce_scatter_tensor_106 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_893, 'avg', 8, '0');  convert_element_type_893 = None
        wait_tensor_313 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_106);  reduce_scatter_tensor_106 = None
        add_110 = torch.ops.aten.add.Tensor(add_106, convert_element_type_891);  add_106 = convert_element_type_891 = None
        all_gather_into_tensor_193 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_110, 8, '1')
        wait_tensor_314 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_193);  all_gather_into_tensor_193 = None
        split_101 = torch.ops.aten.split.Tensor(wait_tensor_314, 2);  wait_tensor_314 = None
        getitem_970 = split_101[0]
        getitem_971 = split_101[1]
        getitem_972 = split_101[2]
        getitem_973 = split_101[3]
        getitem_974 = split_101[4]
        getitem_975 = split_101[5]
        getitem_976 = split_101[6]
        getitem_977 = split_101[7];  split_101 = None
        cat_93 = torch.ops.aten.cat.default([getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977], 1);  getitem_970 = getitem_971 = getitem_972 = getitem_973 = getitem_974 = getitem_975 = getitem_976 = getitem_977 = None
        view_1321 = torch.ops.aten.view.default(cat_93, [16384, 4096]);  cat_93 = None
        permute_385 = torch.ops.aten.permute.default(view_1321, [1, 0])
        permute_105 = torch.ops.aten.permute.default(getitem_449, [0, 2, 1, 3])
        view_690 = torch.ops.aten.view.default(permute_105, [2, 8192, -1]);  permute_105 = None
        view_696 = torch.ops.aten.view.default(view_690, [16384, 512]);  view_690 = None
        mm_205 = torch.ops.aten.mm.default(permute_385, view_696);  permute_385 = view_696 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        all_gather_into_tensor_105 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_314, 8, '0');  convert_element_type_314 = None
        wait_tensor_124 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_105);  all_gather_into_tensor_105 = None
        permute_106 = torch.ops.aten.permute.default(wait_tensor_124, [1, 0]);  wait_tensor_124 = None
        permute_387 = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        mm_206 = torch.ops.aten.mm.default(view_1321, permute_387);  view_1321 = permute_387 = None
        view_1322 = torch.ops.aten.view.default(mm_206, [2, 8192, 512]);  mm_206 = None
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        reduce_scatter_tensor_107 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_898, 'avg', 8, '0');  convert_element_type_898 = None
        wait_tensor_315 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_107);  reduce_scatter_tensor_107 = None
        view_1323 = torch.ops.aten.view.default(view_1322, [2, 8192, 4, 128]);  view_1322 = None
        permute_389 = torch.ops.aten.permute.default(view_1323, [0, 2, 1, 3]);  view_1323 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        all_gather_into_tensor_100 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_298, 8, '0');  convert_element_type_298 = None
        wait_tensor_119 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_100);  all_gather_into_tensor_100 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [2], True);  pow_19 = None
        add_36 = torch.ops.aten.add.Scalar(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_299, rsqrt_18);  convert_element_type_299 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, wait_tensor_119)
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
        view_663 = torch.ops.aten.view.default(cat_37, [16384, 4096]);  cat_37 = None
        view_664 = torch.ops.aten.view.default(mm_63, [2, 8192, 512]);  mm_63 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        all_gather_into_tensor_103 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_304, 8, '0');  convert_element_type_304 = None
        wait_tensor_122 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_103);  all_gather_into_tensor_103 = None
        permute_100 = torch.ops.aten.permute.default(wait_tensor_122, [1, 0]);  wait_tensor_122 = None
        mm_64 = torch.ops.aten.mm.default(view_663, permute_100)
        view_671 = torch.ops.aten.view.default(mm_64, [2, 8192, 128]);  mm_64 = None
        view_678 = torch.ops.aten.view.default(mm_65, [2, 8192, 128]);  mm_65 = None
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
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_389, permute_102, permute_103, permute_104, getitem_449, getitem_450, getitem_455, getitem_456, None, None, None, 8192, 8192, 0.0, True);  permute_389 = permute_102 = permute_103 = permute_104 = getitem_449 = getitem_450 = getitem_455 = getitem_456 = None
        getitem_978 = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_979 = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_980 = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None
        permute_390 = torch.ops.aten.permute.default(getitem_980, [0, 2, 1, 3]);  getitem_980 = None
        permute_391 = torch.ops.aten.permute.default(getitem_979, [0, 2, 1, 3]);  getitem_979 = None
        permute_392 = torch.ops.aten.permute.default(getitem_978, [0, 2, 1, 3]);  getitem_978 = None
        view_1324 = torch.ops.aten.view.default(permute_390, [2, 8192, 1, 4, 128]);  permute_390 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(view_1324, [3], True);  view_1324 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(sum_41, 3);  sum_41 = None
        view_1325 = torch.ops.aten.view.default(permute_391, [2, 8192, 1, 4, 128]);  permute_391 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_1325, [3], True);  view_1325 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(sum_42, 3);  sum_42 = None
        convert_element_type_899 = torch.ops.prims.convert_element_type.default(squeeze_13, torch.float32);  squeeze_13 = None
        convert_element_type_900 = torch.ops.prims.convert_element_type.default(permute_392, torch.float32);  permute_392 = None
        view_1326 = torch.ops.aten.view.default(convert_element_type_899, [2, 8192, 1, 64, 2]);  convert_element_type_899 = None
        view_as_complex_44 = torch.ops.aten.view_as_complex.default(view_1326);  view_1326 = None
        mul_268 = torch.ops.aten.mul.Tensor(view_as_complex_44, _conj);  view_as_complex_44 = None
        view_1327 = torch.ops.aten.view.default(convert_element_type_900, [2, 8192, 4, 64, 2]);  convert_element_type_900 = None
        view_as_complex_45 = torch.ops.aten.view_as_complex.default(view_1327);  view_1327 = None
        mul_269 = torch.ops.aten.mul.Tensor(view_as_complex_45, _conj);  view_as_complex_45 = None
        view_as_real_44 = torch.ops.aten.view_as_real.default(mul_268);  mul_268 = None
        view_1328 = torch.ops.aten.view.default(view_as_real_44, [2, 8192, 1, 128]);  view_as_real_44 = None
        convert_element_type_901 = torch.ops.prims.convert_element_type.default(view_1328, torch.bfloat16);  view_1328 = None
        view_as_real_45 = torch.ops.aten.view_as_real.default(mul_269);  mul_269 = None
        view_1329 = torch.ops.aten.view.default(view_as_real_45, [2, 8192, 4, 128]);  view_as_real_45 = None
        convert_element_type_902 = torch.ops.prims.convert_element_type.default(view_1329, torch.bfloat16);  view_1329 = None
        view_1330 = torch.ops.aten.view.default(squeeze_12, [2, 8192, 128]);  squeeze_12 = None
        view_1331 = torch.ops.aten.view.default(convert_element_type_901, [2, 8192, 128]);  convert_element_type_901 = None
        view_1332 = torch.ops.aten.view.default(convert_element_type_902, [2, 8192, 512]);  convert_element_type_902 = None
        view_1333 = torch.ops.aten.view.default(view_1330, [16384, 128]);  view_1330 = None
        permute_393 = torch.ops.aten.permute.default(view_1333, [1, 0])
        mm_207 = torch.ops.aten.mm.default(permute_393, view_663);  permute_393 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        all_gather_into_tensor_104 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_307, 8, '0');  convert_element_type_307 = None
        wait_tensor_123 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_104);  all_gather_into_tensor_104 = None
        permute_101 = torch.ops.aten.permute.default(wait_tensor_123, [1, 0]);  wait_tensor_123 = None
        permute_395 = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_208 = torch.ops.aten.mm.default(view_1333, permute_395);  view_1333 = permute_395 = None
        view_1334 = torch.ops.aten.view.default(mm_208, [2, 8192, 4096]);  mm_208 = None
        convert_element_type_907 = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        reduce_scatter_tensor_108 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_907, 'avg', 8, '0');  convert_element_type_907 = None
        wait_tensor_316 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_108);  reduce_scatter_tensor_108 = None
        view_1335 = torch.ops.aten.view.default(view_1331, [16384, 128]);  view_1331 = None
        permute_397 = torch.ops.aten.permute.default(view_1335, [1, 0])
        mm_209 = torch.ops.aten.mm.default(permute_397, view_663);  permute_397 = None
        permute_399 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_210 = torch.ops.aten.mm.default(view_1335, permute_399);  view_1335 = permute_399 = None
        view_1336 = torch.ops.aten.view.default(mm_210, [2, 8192, 4096]);  mm_210 = None
        add_111 = torch.ops.aten.add.Tensor(view_1334, view_1336);  view_1334 = view_1336 = None
        convert_element_type_912 = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        reduce_scatter_tensor_109 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_912, 'avg', 8, '0');  convert_element_type_912 = None
        wait_tensor_317 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_109);  reduce_scatter_tensor_109 = None
        view_1337 = torch.ops.aten.view.default(view_1332, [16384, 512]);  view_1332 = None
        permute_401 = torch.ops.aten.permute.default(view_1337, [1, 0])
        mm_211 = torch.ops.aten.mm.default(permute_401, view_663);  permute_401 = view_663 = None
        convert_element_type_301 = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        all_gather_into_tensor_102 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_301, 8, '0');  convert_element_type_301 = None
        wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_102);  all_gather_into_tensor_102 = None
        permute_99 = torch.ops.aten.permute.default(wait_tensor_121, [1, 0]);  wait_tensor_121 = None
        permute_403 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_212 = torch.ops.aten.mm.default(view_1337, permute_403);  view_1337 = permute_403 = None
        view_1338 = torch.ops.aten.view.default(mm_212, [2, 8192, 4096]);  mm_212 = None
        add_112 = torch.ops.aten.add.Tensor(add_111, view_1338);  add_111 = view_1338 = None
        convert_element_type_917 = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        reduce_scatter_tensor_110 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_917, 'avg', 8, '0');  convert_element_type_917 = None
        wait_tensor_318 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_110);  reduce_scatter_tensor_110 = None
        split_102 = torch.ops.aten.split.Tensor(add_112, 1024, 1);  add_112 = None
        getitem_981 = split_102[0]
        getitem_982 = split_102[1]
        getitem_983 = split_102[2]
        getitem_984 = split_102[3]
        getitem_985 = split_102[4]
        getitem_986 = split_102[5]
        getitem_987 = split_102[6]
        getitem_988 = split_102[7];  split_102 = None
        cat_94 = torch.ops.aten.cat.default([getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988]);  getitem_981 = getitem_982 = getitem_983 = getitem_984 = getitem_985 = getitem_986 = getitem_987 = getitem_988 = None
        reduce_scatter_tensor_111 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_94, 'sum', 8, '1');  cat_94 = None
        wait_tensor_319 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_111);  reduce_scatter_tensor_111 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(wait_tensor_319, torch.float32);  wait_tensor_319 = None
        convert_element_type_920 = torch.ops.prims.convert_element_type.default(wait_tensor_119, torch.float32);  wait_tensor_119 = None
        mul_270 = torch.ops.aten.mul.Tensor(convert_element_type_918, convert_element_type_920);  convert_element_type_920 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_72, mul_270)
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        div_14 = torch.ops.aten.div.Tensor(mul_72, 4096)
        mul_273 = torch.ops.aten.mul.Tensor(div_14, sum_43);  div_14 = sum_43 = None
        sub_22 = torch.ops.aten.sub.Tensor(mul_270, mul_273);  mul_270 = mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = rsqrt_18 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_918, mul_72);  convert_element_type_918 = mul_72 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        convert_element_type_921 = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None
        convert_element_type_922 = torch.ops.prims.convert_element_type.default(sum_44, torch.bfloat16);  sum_44 = None
        all_reduce_14 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_922, 'sum', '1');  convert_element_type_922 = None
        wait_tensor_320 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_14);  all_reduce_14 = None
        convert_element_type_923 = torch.ops.prims.convert_element_type.default(wait_tensor_320, torch.float32);  wait_tensor_320 = None
        reduce_scatter_tensor_112 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_923, 'avg', 8, '0');  convert_element_type_923 = None
        wait_tensor_321 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_112);  reduce_scatter_tensor_112 = None
        add_113 = torch.ops.aten.add.Tensor(add_110, convert_element_type_921);  add_110 = convert_element_type_921 = None
        all_gather_into_tensor_194 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_113, 8, '1')
        wait_tensor_322 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_194);  all_gather_into_tensor_194 = None
        split_103 = torch.ops.aten.split.Tensor(wait_tensor_322, 2);  wait_tensor_322 = None
        getitem_989 = split_103[0]
        getitem_990 = split_103[1]
        getitem_991 = split_103[2]
        getitem_992 = split_103[3]
        getitem_993 = split_103[4]
        getitem_994 = split_103[5]
        getitem_995 = split_103[6]
        getitem_996 = split_103[7];  split_103 = None
        cat_95 = torch.ops.aten.cat.default([getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996], 1);  getitem_989 = getitem_990 = getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = None
        view_1339 = torch.ops.aten.view.default(cat_95, [16384, 4096]);  cat_95 = None
        permute_405 = torch.ops.aten.permute.default(view_1339, [1, 0])
        wait_tensor_112 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_17);  reduce_scatter_tensor_17 = None
        add_33 = torch.ops.aten.add.Tensor(add_31, wait_tensor_112);  wait_tensor_112 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        all_gather_into_tensor_95 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_284, 8, '0');  convert_element_type_284 = None
        wait_tensor_113 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_95);  all_gather_into_tensor_95 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [2], True);  pow_18 = None
        add_34 = torch.ops.aten.add.Scalar(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_285, rsqrt_17);  convert_element_type_285 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, wait_tensor_113)
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
        view_636 = torch.ops.aten.view.default(cat_35, [16384, 4096]);  cat_35 = None
        view_637 = torch.ops.aten.view.default(mm_60, [2, 8192, 1792]);  mm_60 = None
        convert_element_type_290 = torch.ops.prims.convert_element_type.default(view_637, torch.float32);  view_637 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_290)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_290, sigmoid_8);  sigmoid_8 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        all_gather_into_tensor_98 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_292, 8, '0');  convert_element_type_292 = None
        wait_tensor_116 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_98);  all_gather_into_tensor_98 = None
        permute_97 = torch.ops.aten.permute.default(wait_tensor_116, [1, 0]);  wait_tensor_116 = None
        mm_61 = torch.ops.aten.mm.default(view_636, permute_97)
        view_644 = torch.ops.aten.view.default(mm_61, [2, 8192, 1792]);  mm_61 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_291, view_644)
        view_651 = torch.ops.aten.view.default(mul_71, [16384, 1792]);  mul_71 = None
        mm_213 = torch.ops.aten.mm.default(permute_405, view_651);  permute_405 = view_651 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        all_gather_into_tensor_99 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_295, 8, '0');  convert_element_type_295 = None
        wait_tensor_117 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_99);  all_gather_into_tensor_99 = None
        permute_98 = torch.ops.aten.permute.default(wait_tensor_117, [1, 0]);  wait_tensor_117 = None
        permute_407 = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_214 = torch.ops.aten.mm.default(view_1339, permute_407);  view_1339 = permute_407 = None
        view_1340 = torch.ops.aten.view.default(mm_214, [2, 8192, 1792]);  mm_214 = None
        convert_element_type_928 = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        reduce_scatter_tensor_113 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_928, 'avg', 8, '0');  convert_element_type_928 = None
        wait_tensor_323 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_113);  reduce_scatter_tensor_113 = None
        mul_276 = torch.ops.aten.mul.Tensor(view_1340, convert_element_type_291);  convert_element_type_291 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_1340, view_644);  view_1340 = view_644 = None
        view_1341 = torch.ops.aten.view.default(mul_276, [16384, 1792]);  mul_276 = None
        permute_409 = torch.ops.aten.permute.default(view_1341, [1, 0])
        mm_215 = torch.ops.aten.mm.default(permute_409, view_636);  permute_409 = None
        permute_411 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_216 = torch.ops.aten.mm.default(view_1341, permute_411);  view_1341 = permute_411 = None
        view_1342 = torch.ops.aten.view.default(mm_216, [2, 8192, 4096]);  mm_216 = None
        convert_element_type_933 = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None
        reduce_scatter_tensor_114 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_933, 'avg', 8, '0');  convert_element_type_933 = None
        wait_tensor_324 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_114);  reduce_scatter_tensor_114 = None
        convert_element_type_934 = torch.ops.prims.convert_element_type.default(mul_277, torch.float32);  mul_277 = None
        neg_7 = torch.ops.aten.neg.default(convert_element_type_290)
        exp_7 = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_114 = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(add_114);  add_114 = None
        mul_278 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_279 = torch.ops.aten.mul.Tensor(convert_element_type_934, mul_278);  convert_element_type_934 = None
        sub_23 = torch.ops.aten.sub.Tensor(1, mul_278);  mul_278 = None
        mul_280 = torch.ops.aten.mul.Tensor(convert_element_type_290, sub_23);  convert_element_type_290 = sub_23 = None
        add_115 = torch.ops.aten.add.Tensor(mul_280, 1);  mul_280 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_279, add_115);  mul_279 = add_115 = None
        convert_element_type_936 = torch.ops.prims.convert_element_type.default(mul_281, torch.bfloat16);  mul_281 = None
        view_1343 = torch.ops.aten.view.default(convert_element_type_936, [16384, 1792]);  convert_element_type_936 = None
        permute_413 = torch.ops.aten.permute.default(view_1343, [1, 0])
        mm_217 = torch.ops.aten.mm.default(permute_413, view_636);  permute_413 = view_636 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        all_gather_into_tensor_97 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_287, 8, '0');  convert_element_type_287 = None
        wait_tensor_115 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_97);  all_gather_into_tensor_97 = None
        permute_96 = torch.ops.aten.permute.default(wait_tensor_115, [1, 0]);  wait_tensor_115 = None
        permute_415 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_218 = torch.ops.aten.mm.default(view_1343, permute_415);  view_1343 = permute_415 = None
        view_1344 = torch.ops.aten.view.default(mm_218, [2, 8192, 4096]);  mm_218 = None
        add_116 = torch.ops.aten.add.Tensor(view_1342, view_1344);  view_1342 = view_1344 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None
        reduce_scatter_tensor_115 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_941, 'avg', 8, '0');  convert_element_type_941 = None
        wait_tensor_325 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_115);  reduce_scatter_tensor_115 = None
        split_104 = torch.ops.aten.split.Tensor(add_116, 1024, 1);  add_116 = None
        getitem_997 = split_104[0]
        getitem_998 = split_104[1]
        getitem_999 = split_104[2]
        getitem_1000 = split_104[3]
        getitem_1001 = split_104[4]
        getitem_1002 = split_104[5]
        getitem_1003 = split_104[6]
        getitem_1004 = split_104[7];  split_104 = None
        cat_96 = torch.ops.aten.cat.default([getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004]);  getitem_997 = getitem_998 = getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = None
        reduce_scatter_tensor_116 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_96, 'sum', 8, '1');  cat_96 = None
        wait_tensor_326 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_116);  reduce_scatter_tensor_116 = None
        convert_element_type_942 = torch.ops.prims.convert_element_type.default(wait_tensor_326, torch.float32);  wait_tensor_326 = None
        convert_element_type_944 = torch.ops.prims.convert_element_type.default(wait_tensor_113, torch.float32);  wait_tensor_113 = None
        mul_282 = torch.ops.aten.mul.Tensor(convert_element_type_942, convert_element_type_944);  convert_element_type_944 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_68, mul_282)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        div_15 = torch.ops.aten.div.Tensor(mul_68, 4096)
        mul_285 = torch.ops.aten.mul.Tensor(div_15, sum_45);  div_15 = sum_45 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_282, mul_285);  mul_282 = mul_285 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = rsqrt_17 = None
        mul_287 = torch.ops.aten.mul.Tensor(convert_element_type_942, mul_68);  convert_element_type_942 = mul_68 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        convert_element_type_945 = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        convert_element_type_946 = torch.ops.prims.convert_element_type.default(sum_46, torch.bfloat16);  sum_46 = None
        all_reduce_15 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_946, 'sum', '1');  convert_element_type_946 = None
        wait_tensor_327 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_15);  all_reduce_15 = None
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(wait_tensor_327, torch.float32);  wait_tensor_327 = None
        reduce_scatter_tensor_117 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_947, 'avg', 8, '0');  convert_element_type_947 = None
        wait_tensor_328 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_117);  reduce_scatter_tensor_117 = None
        add_117 = torch.ops.aten.add.Tensor(add_113, convert_element_type_945);  add_113 = convert_element_type_945 = None
        all_gather_into_tensor_195 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_117, 8, '1')
        wait_tensor_329 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_195);  all_gather_into_tensor_195 = None
        split_105 = torch.ops.aten.split.Tensor(wait_tensor_329, 2);  wait_tensor_329 = None
        getitem_1005 = split_105[0]
        getitem_1006 = split_105[1]
        getitem_1007 = split_105[2]
        getitem_1008 = split_105[3]
        getitem_1009 = split_105[4]
        getitem_1010 = split_105[5]
        getitem_1011 = split_105[6]
        getitem_1012 = split_105[7];  split_105 = None
        cat_97 = torch.ops.aten.cat.default([getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012], 1);  getitem_1005 = getitem_1006 = getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = None
        view_1345 = torch.ops.aten.view.default(cat_97, [16384, 4096]);  cat_97 = None
        permute_417 = torch.ops.aten.permute.default(view_1345, [1, 0])
        permute_94 = torch.ops.aten.permute.default(getitem_408, [0, 2, 1, 3])
        view_618 = torch.ops.aten.view.default(permute_94, [2, 8192, -1]);  permute_94 = None
        view_624 = torch.ops.aten.view.default(view_618, [16384, 512]);  view_618 = None
        mm_219 = torch.ops.aten.mm.default(permute_417, view_624);  permute_417 = view_624 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        all_gather_into_tensor_94 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_281, 8, '0');  convert_element_type_281 = None
        wait_tensor_111 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_94);  all_gather_into_tensor_94 = None
        permute_95 = torch.ops.aten.permute.default(wait_tensor_111, [1, 0]);  wait_tensor_111 = None
        permute_419 = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        mm_220 = torch.ops.aten.mm.default(view_1345, permute_419);  view_1345 = permute_419 = None
        view_1346 = torch.ops.aten.view.default(mm_220, [2, 8192, 512]);  mm_220 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        reduce_scatter_tensor_118 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_952, 'avg', 8, '0');  convert_element_type_952 = None
        wait_tensor_330 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_118);  reduce_scatter_tensor_118 = None
        view_1347 = torch.ops.aten.view.default(view_1346, [2, 8192, 4, 128]);  view_1346 = None
        permute_421 = torch.ops.aten.permute.default(view_1347, [0, 2, 1, 3]);  view_1347 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        all_gather_into_tensor_89 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_265, 8, '0');  convert_element_type_265 = None
        wait_tensor_106 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_89);  all_gather_into_tensor_89 = None
        convert_element_type_266 = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [2], True);  pow_17 = None
        add_32 = torch.ops.aten.add.Scalar(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_266, rsqrt_16);  convert_element_type_266 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, wait_tensor_106)
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
        view_591 = torch.ops.aten.view.default(cat_33, [16384, 4096]);  cat_33 = None
        view_592 = torch.ops.aten.view.default(mm_56, [2, 8192, 512]);  mm_56 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        all_gather_into_tensor_92 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_271, 8, '0');  convert_element_type_271 = None
        wait_tensor_109 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_92);  all_gather_into_tensor_92 = None
        permute_89 = torch.ops.aten.permute.default(wait_tensor_109, [1, 0]);  wait_tensor_109 = None
        mm_57 = torch.ops.aten.mm.default(view_591, permute_89)
        view_599 = torch.ops.aten.view.default(mm_57, [2, 8192, 128]);  mm_57 = None
        view_606 = torch.ops.aten.view.default(mm_58, [2, 8192, 128]);  mm_58 = None
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
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_421, permute_91, permute_92, permute_93, getitem_408, getitem_409, getitem_414, getitem_415, None, None, None, 8192, 8192, 0.0, True);  permute_421 = permute_91 = permute_92 = permute_93 = getitem_408 = getitem_409 = getitem_414 = getitem_415 = None
        getitem_1013 = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_1014 = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_1015 = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None
        permute_422 = torch.ops.aten.permute.default(getitem_1015, [0, 2, 1, 3]);  getitem_1015 = None
        permute_423 = torch.ops.aten.permute.default(getitem_1014, [0, 2, 1, 3]);  getitem_1014 = None
        permute_424 = torch.ops.aten.permute.default(getitem_1013, [0, 2, 1, 3]);  getitem_1013 = None
        view_1348 = torch.ops.aten.view.default(permute_422, [2, 8192, 1, 4, 128]);  permute_422 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(view_1348, [3], True);  view_1348 = None
        squeeze_14 = torch.ops.aten.squeeze.dim(sum_47, 3);  sum_47 = None
        view_1349 = torch.ops.aten.view.default(permute_423, [2, 8192, 1, 4, 128]);  permute_423 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(view_1349, [3], True);  view_1349 = None
        squeeze_15 = torch.ops.aten.squeeze.dim(sum_48, 3);  sum_48 = None
        convert_element_type_953 = torch.ops.prims.convert_element_type.default(squeeze_15, torch.float32);  squeeze_15 = None
        convert_element_type_954 = torch.ops.prims.convert_element_type.default(permute_424, torch.float32);  permute_424 = None
        view_1350 = torch.ops.aten.view.default(convert_element_type_953, [2, 8192, 1, 64, 2]);  convert_element_type_953 = None
        view_as_complex_46 = torch.ops.aten.view_as_complex.default(view_1350);  view_1350 = None
        mul_288 = torch.ops.aten.mul.Tensor(view_as_complex_46, _conj);  view_as_complex_46 = None
        view_1351 = torch.ops.aten.view.default(convert_element_type_954, [2, 8192, 4, 64, 2]);  convert_element_type_954 = None
        view_as_complex_47 = torch.ops.aten.view_as_complex.default(view_1351);  view_1351 = None
        mul_289 = torch.ops.aten.mul.Tensor(view_as_complex_47, _conj);  view_as_complex_47 = None
        view_as_real_46 = torch.ops.aten.view_as_real.default(mul_288);  mul_288 = None
        view_1352 = torch.ops.aten.view.default(view_as_real_46, [2, 8192, 1, 128]);  view_as_real_46 = None
        convert_element_type_955 = torch.ops.prims.convert_element_type.default(view_1352, torch.bfloat16);  view_1352 = None
        view_as_real_47 = torch.ops.aten.view_as_real.default(mul_289);  mul_289 = None
        view_1353 = torch.ops.aten.view.default(view_as_real_47, [2, 8192, 4, 128]);  view_as_real_47 = None
        convert_element_type_956 = torch.ops.prims.convert_element_type.default(view_1353, torch.bfloat16);  view_1353 = None
        view_1354 = torch.ops.aten.view.default(squeeze_14, [2, 8192, 128]);  squeeze_14 = None
        view_1355 = torch.ops.aten.view.default(convert_element_type_955, [2, 8192, 128]);  convert_element_type_955 = None
        view_1356 = torch.ops.aten.view.default(convert_element_type_956, [2, 8192, 512]);  convert_element_type_956 = None
        view_1357 = torch.ops.aten.view.default(view_1354, [16384, 128]);  view_1354 = None
        permute_425 = torch.ops.aten.permute.default(view_1357, [1, 0])
        mm_221 = torch.ops.aten.mm.default(permute_425, view_591);  permute_425 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        all_gather_into_tensor_93 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_274, 8, '0');  convert_element_type_274 = None
        wait_tensor_110 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_93);  all_gather_into_tensor_93 = None
        permute_90 = torch.ops.aten.permute.default(wait_tensor_110, [1, 0]);  wait_tensor_110 = None
        permute_427 = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_222 = torch.ops.aten.mm.default(view_1357, permute_427);  view_1357 = permute_427 = None
        view_1358 = torch.ops.aten.view.default(mm_222, [2, 8192, 4096]);  mm_222 = None
        convert_element_type_961 = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        reduce_scatter_tensor_119 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_961, 'avg', 8, '0');  convert_element_type_961 = None
        wait_tensor_331 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_119);  reduce_scatter_tensor_119 = None
        view_1359 = torch.ops.aten.view.default(view_1355, [16384, 128]);  view_1355 = None
        permute_429 = torch.ops.aten.permute.default(view_1359, [1, 0])
        mm_223 = torch.ops.aten.mm.default(permute_429, view_591);  permute_429 = None
        permute_431 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_224 = torch.ops.aten.mm.default(view_1359, permute_431);  view_1359 = permute_431 = None
        view_1360 = torch.ops.aten.view.default(mm_224, [2, 8192, 4096]);  mm_224 = None
        add_118 = torch.ops.aten.add.Tensor(view_1358, view_1360);  view_1358 = view_1360 = None
        convert_element_type_966 = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        reduce_scatter_tensor_120 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_966, 'avg', 8, '0');  convert_element_type_966 = None
        wait_tensor_332 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_120);  reduce_scatter_tensor_120 = None
        view_1361 = torch.ops.aten.view.default(view_1356, [16384, 512]);  view_1356 = None
        permute_433 = torch.ops.aten.permute.default(view_1361, [1, 0])
        mm_225 = torch.ops.aten.mm.default(permute_433, view_591);  permute_433 = view_591 = None
        convert_element_type_268 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        all_gather_into_tensor_91 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_268, 8, '0');  convert_element_type_268 = None
        wait_tensor_108 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_91);  all_gather_into_tensor_91 = None
        permute_88 = torch.ops.aten.permute.default(wait_tensor_108, [1, 0]);  wait_tensor_108 = None
        permute_435 = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_226 = torch.ops.aten.mm.default(view_1361, permute_435);  view_1361 = permute_435 = None
        view_1362 = torch.ops.aten.view.default(mm_226, [2, 8192, 4096]);  mm_226 = None
        add_119 = torch.ops.aten.add.Tensor(add_118, view_1362);  add_118 = view_1362 = None
        convert_element_type_971 = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        reduce_scatter_tensor_121 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_971, 'avg', 8, '0');  convert_element_type_971 = None
        wait_tensor_333 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_121);  reduce_scatter_tensor_121 = None
        split_106 = torch.ops.aten.split.Tensor(add_119, 1024, 1);  add_119 = None
        getitem_1016 = split_106[0]
        getitem_1017 = split_106[1]
        getitem_1018 = split_106[2]
        getitem_1019 = split_106[3]
        getitem_1020 = split_106[4]
        getitem_1021 = split_106[5]
        getitem_1022 = split_106[6]
        getitem_1023 = split_106[7];  split_106 = None
        cat_98 = torch.ops.aten.cat.default([getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023]);  getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = None
        reduce_scatter_tensor_122 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_98, 'sum', 8, '1');  cat_98 = None
        wait_tensor_334 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_122);  reduce_scatter_tensor_122 = None
        convert_element_type_972 = torch.ops.prims.convert_element_type.default(wait_tensor_334, torch.float32);  wait_tensor_334 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(wait_tensor_106, torch.float32);  wait_tensor_106 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_972, convert_element_type_974);  convert_element_type_974 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_64, mul_290)
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        div_16 = torch.ops.aten.div.Tensor(mul_64, 4096)
        mul_293 = torch.ops.aten.mul.Tensor(div_16, sum_49);  div_16 = sum_49 = None
        sub_25 = torch.ops.aten.sub.Tensor(mul_290, mul_293);  mul_290 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
        mul_295 = torch.ops.aten.mul.Tensor(convert_element_type_972, mul_64);  convert_element_type_972 = mul_64 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        convert_element_type_975 = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None
        convert_element_type_976 = torch.ops.prims.convert_element_type.default(sum_50, torch.bfloat16);  sum_50 = None
        all_reduce_16 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_976, 'sum', '1');  convert_element_type_976 = None
        wait_tensor_335 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_16);  all_reduce_16 = None
        convert_element_type_977 = torch.ops.prims.convert_element_type.default(wait_tensor_335, torch.float32);  wait_tensor_335 = None
        reduce_scatter_tensor_123 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_977, 'avg', 8, '0');  convert_element_type_977 = None
        wait_tensor_336 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_123);  reduce_scatter_tensor_123 = None
        add_120 = torch.ops.aten.add.Tensor(add_117, convert_element_type_975);  add_117 = convert_element_type_975 = None
        all_gather_into_tensor_196 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_120, 8, '1')
        wait_tensor_337 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_196);  all_gather_into_tensor_196 = None
        split_107 = torch.ops.aten.split.Tensor(wait_tensor_337, 2);  wait_tensor_337 = None
        getitem_1024 = split_107[0]
        getitem_1025 = split_107[1]
        getitem_1026 = split_107[2]
        getitem_1027 = split_107[3]
        getitem_1028 = split_107[4]
        getitem_1029 = split_107[5]
        getitem_1030 = split_107[6]
        getitem_1031 = split_107[7];  split_107 = None
        cat_99 = torch.ops.aten.cat.default([getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031], 1);  getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = getitem_1030 = getitem_1031 = None
        view_1363 = torch.ops.aten.view.default(cat_99, [16384, 4096]);  cat_99 = None
        permute_437 = torch.ops.aten.permute.default(view_1363, [1, 0])
        wait_tensor_99 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_15);  reduce_scatter_tensor_15 = None
        add_29 = torch.ops.aten.add.Tensor(add_27, wait_tensor_99);  wait_tensor_99 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        all_gather_into_tensor_84 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_251, 8, '0');  convert_element_type_251 = None
        wait_tensor_100 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_84);  all_gather_into_tensor_84 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [2], True);  pow_16 = None
        add_30 = torch.ops.aten.add.Scalar(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_252, rsqrt_15);  convert_element_type_252 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, wait_tensor_100)
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
        view_564 = torch.ops.aten.view.default(cat_31, [16384, 4096]);  cat_31 = None
        view_565 = torch.ops.aten.view.default(mm_53, [2, 8192, 1792]);  mm_53 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(view_565, torch.float32);  view_565 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_257, sigmoid_7);  sigmoid_7 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        all_gather_into_tensor_87 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_259, 8, '0');  convert_element_type_259 = None
        wait_tensor_103 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_87);  all_gather_into_tensor_87 = None
        permute_86 = torch.ops.aten.permute.default(wait_tensor_103, [1, 0]);  wait_tensor_103 = None
        mm_54 = torch.ops.aten.mm.default(view_564, permute_86)
        view_572 = torch.ops.aten.view.default(mm_54, [2, 8192, 1792]);  mm_54 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_258, view_572)
        view_579 = torch.ops.aten.view.default(mul_63, [16384, 1792]);  mul_63 = None
        mm_227 = torch.ops.aten.mm.default(permute_437, view_579);  permute_437 = view_579 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        all_gather_into_tensor_88 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_262, 8, '0');  convert_element_type_262 = None
        wait_tensor_104 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_88);  all_gather_into_tensor_88 = None
        permute_87 = torch.ops.aten.permute.default(wait_tensor_104, [1, 0]);  wait_tensor_104 = None
        permute_439 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_228 = torch.ops.aten.mm.default(view_1363, permute_439);  view_1363 = permute_439 = None
        view_1364 = torch.ops.aten.view.default(mm_228, [2, 8192, 1792]);  mm_228 = None
        convert_element_type_982 = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None
        reduce_scatter_tensor_124 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_982, 'avg', 8, '0');  convert_element_type_982 = None
        wait_tensor_338 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_124);  reduce_scatter_tensor_124 = None
        mul_296 = torch.ops.aten.mul.Tensor(view_1364, convert_element_type_258);  convert_element_type_258 = None
        mul_297 = torch.ops.aten.mul.Tensor(view_1364, view_572);  view_1364 = view_572 = None
        view_1365 = torch.ops.aten.view.default(mul_296, [16384, 1792]);  mul_296 = None
        permute_441 = torch.ops.aten.permute.default(view_1365, [1, 0])
        mm_229 = torch.ops.aten.mm.default(permute_441, view_564);  permute_441 = None
        permute_443 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_230 = torch.ops.aten.mm.default(view_1365, permute_443);  view_1365 = permute_443 = None
        view_1366 = torch.ops.aten.view.default(mm_230, [2, 8192, 4096]);  mm_230 = None
        convert_element_type_987 = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None
        reduce_scatter_tensor_125 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_987, 'avg', 8, '0');  convert_element_type_987 = None
        wait_tensor_339 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_125);  reduce_scatter_tensor_125 = None
        convert_element_type_988 = torch.ops.prims.convert_element_type.default(mul_297, torch.float32);  mul_297 = None
        neg_8 = torch.ops.aten.neg.default(convert_element_type_257)
        exp_8 = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_121 = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(add_121);  add_121 = None
        mul_298 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_299 = torch.ops.aten.mul.Tensor(convert_element_type_988, mul_298);  convert_element_type_988 = None
        sub_26 = torch.ops.aten.sub.Tensor(1, mul_298);  mul_298 = None
        mul_300 = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_26);  convert_element_type_257 = sub_26 = None
        add_122 = torch.ops.aten.add.Tensor(mul_300, 1);  mul_300 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_299, add_122);  mul_299 = add_122 = None
        convert_element_type_990 = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16);  mul_301 = None
        view_1367 = torch.ops.aten.view.default(convert_element_type_990, [16384, 1792]);  convert_element_type_990 = None
        permute_445 = torch.ops.aten.permute.default(view_1367, [1, 0])
        mm_231 = torch.ops.aten.mm.default(permute_445, view_564);  permute_445 = view_564 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        all_gather_into_tensor_86 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_254, 8, '0');  convert_element_type_254 = None
        wait_tensor_102 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_86);  all_gather_into_tensor_86 = None
        permute_85 = torch.ops.aten.permute.default(wait_tensor_102, [1, 0]);  wait_tensor_102 = None
        permute_447 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_232 = torch.ops.aten.mm.default(view_1367, permute_447);  view_1367 = permute_447 = None
        view_1368 = torch.ops.aten.view.default(mm_232, [2, 8192, 4096]);  mm_232 = None
        add_123 = torch.ops.aten.add.Tensor(view_1366, view_1368);  view_1366 = view_1368 = None
        convert_element_type_995 = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        reduce_scatter_tensor_126 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_995, 'avg', 8, '0');  convert_element_type_995 = None
        wait_tensor_340 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_126);  reduce_scatter_tensor_126 = None
        split_108 = torch.ops.aten.split.Tensor(add_123, 1024, 1);  add_123 = None
        getitem_1032 = split_108[0]
        getitem_1033 = split_108[1]
        getitem_1034 = split_108[2]
        getitem_1035 = split_108[3]
        getitem_1036 = split_108[4]
        getitem_1037 = split_108[5]
        getitem_1038 = split_108[6]
        getitem_1039 = split_108[7];  split_108 = None
        cat_100 = torch.ops.aten.cat.default([getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039]);  getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = None
        reduce_scatter_tensor_127 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_100, 'sum', 8, '1');  cat_100 = None
        wait_tensor_341 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_127);  reduce_scatter_tensor_127 = None
        convert_element_type_996 = torch.ops.prims.convert_element_type.default(wait_tensor_341, torch.float32);  wait_tensor_341 = None
        convert_element_type_998 = torch.ops.prims.convert_element_type.default(wait_tensor_100, torch.float32);  wait_tensor_100 = None
        mul_302 = torch.ops.aten.mul.Tensor(convert_element_type_996, convert_element_type_998);  convert_element_type_998 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_60, mul_302)
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        div_17 = torch.ops.aten.div.Tensor(mul_60, 4096)
        mul_305 = torch.ops.aten.mul.Tensor(div_17, sum_51);  div_17 = sum_51 = None
        sub_27 = torch.ops.aten.sub.Tensor(mul_302, mul_305);  mul_302 = mul_305 = None
        mul_306 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_15);  sub_27 = rsqrt_15 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_996, mul_60);  convert_element_type_996 = mul_60 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        convert_element_type_999 = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        convert_element_type_1000 = torch.ops.prims.convert_element_type.default(sum_52, torch.bfloat16);  sum_52 = None
        all_reduce_17 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1000, 'sum', '1');  convert_element_type_1000 = None
        wait_tensor_342 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_17);  all_reduce_17 = None
        convert_element_type_1001 = torch.ops.prims.convert_element_type.default(wait_tensor_342, torch.float32);  wait_tensor_342 = None
        reduce_scatter_tensor_128 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1001, 'avg', 8, '0');  convert_element_type_1001 = None
        wait_tensor_343 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_128);  reduce_scatter_tensor_128 = None
        add_124 = torch.ops.aten.add.Tensor(add_120, convert_element_type_999);  add_120 = convert_element_type_999 = None
        all_gather_into_tensor_197 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_124, 8, '1')
        wait_tensor_344 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_197);  all_gather_into_tensor_197 = None
        split_109 = torch.ops.aten.split.Tensor(wait_tensor_344, 2);  wait_tensor_344 = None
        getitem_1040 = split_109[0]
        getitem_1041 = split_109[1]
        getitem_1042 = split_109[2]
        getitem_1043 = split_109[3]
        getitem_1044 = split_109[4]
        getitem_1045 = split_109[5]
        getitem_1046 = split_109[6]
        getitem_1047 = split_109[7];  split_109 = None
        cat_101 = torch.ops.aten.cat.default([getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047], 1);  getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = None
        view_1369 = torch.ops.aten.view.default(cat_101, [16384, 4096]);  cat_101 = None
        permute_449 = torch.ops.aten.permute.default(view_1369, [1, 0])
        permute_83 = torch.ops.aten.permute.default(getitem_367, [0, 2, 1, 3])
        view_546 = torch.ops.aten.view.default(permute_83, [2, 8192, -1]);  permute_83 = None
        view_552 = torch.ops.aten.view.default(view_546, [16384, 512]);  view_546 = None
        mm_233 = torch.ops.aten.mm.default(permute_449, view_552);  permute_449 = view_552 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        all_gather_into_tensor_83 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_248, 8, '0');  convert_element_type_248 = None
        wait_tensor_98 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_83);  all_gather_into_tensor_83 = None
        permute_84 = torch.ops.aten.permute.default(wait_tensor_98, [1, 0]);  wait_tensor_98 = None
        permute_451 = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        mm_234 = torch.ops.aten.mm.default(view_1369, permute_451);  view_1369 = permute_451 = None
        view_1370 = torch.ops.aten.view.default(mm_234, [2, 8192, 512]);  mm_234 = None
        convert_element_type_1006 = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        reduce_scatter_tensor_129 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1006, 'avg', 8, '0');  convert_element_type_1006 = None
        wait_tensor_345 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_129);  reduce_scatter_tensor_129 = None
        view_1371 = torch.ops.aten.view.default(view_1370, [2, 8192, 4, 128]);  view_1370 = None
        permute_453 = torch.ops.aten.permute.default(view_1371, [0, 2, 1, 3]);  view_1371 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        all_gather_into_tensor_78 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_232, 8, '0');  convert_element_type_232 = None
        wait_tensor_93 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_78);  all_gather_into_tensor_78 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [2], True);  pow_15 = None
        add_28 = torch.ops.aten.add.Scalar(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_14);  convert_element_type_233 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, wait_tensor_93)
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
        view_519 = torch.ops.aten.view.default(cat_29, [16384, 4096]);  cat_29 = None
        view_520 = torch.ops.aten.view.default(mm_49, [2, 8192, 512]);  mm_49 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        all_gather_into_tensor_81 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_238, 8, '0');  convert_element_type_238 = None
        wait_tensor_96 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_81);  all_gather_into_tensor_81 = None
        permute_78 = torch.ops.aten.permute.default(wait_tensor_96, [1, 0]);  wait_tensor_96 = None
        mm_50 = torch.ops.aten.mm.default(view_519, permute_78)
        view_527 = torch.ops.aten.view.default(mm_50, [2, 8192, 128]);  mm_50 = None
        view_534 = torch.ops.aten.view.default(mm_51, [2, 8192, 128]);  mm_51 = None
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
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_453, permute_80, permute_81, permute_82, getitem_367, getitem_368, getitem_373, getitem_374, None, None, None, 8192, 8192, 0.0, True);  permute_453 = permute_80 = permute_81 = permute_82 = getitem_367 = getitem_368 = getitem_373 = getitem_374 = None
        getitem_1048 = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_1049 = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_1050 = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None
        permute_454 = torch.ops.aten.permute.default(getitem_1050, [0, 2, 1, 3]);  getitem_1050 = None
        permute_455 = torch.ops.aten.permute.default(getitem_1049, [0, 2, 1, 3]);  getitem_1049 = None
        permute_456 = torch.ops.aten.permute.default(getitem_1048, [0, 2, 1, 3]);  getitem_1048 = None
        view_1372 = torch.ops.aten.view.default(permute_454, [2, 8192, 1, 4, 128]);  permute_454 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(view_1372, [3], True);  view_1372 = None
        squeeze_16 = torch.ops.aten.squeeze.dim(sum_53, 3);  sum_53 = None
        view_1373 = torch.ops.aten.view.default(permute_455, [2, 8192, 1, 4, 128]);  permute_455 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(view_1373, [3], True);  view_1373 = None
        squeeze_17 = torch.ops.aten.squeeze.dim(sum_54, 3);  sum_54 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(squeeze_17, torch.float32);  squeeze_17 = None
        convert_element_type_1008 = torch.ops.prims.convert_element_type.default(permute_456, torch.float32);  permute_456 = None
        view_1374 = torch.ops.aten.view.default(convert_element_type_1007, [2, 8192, 1, 64, 2]);  convert_element_type_1007 = None
        view_as_complex_48 = torch.ops.aten.view_as_complex.default(view_1374);  view_1374 = None
        mul_308 = torch.ops.aten.mul.Tensor(view_as_complex_48, _conj);  view_as_complex_48 = None
        view_1375 = torch.ops.aten.view.default(convert_element_type_1008, [2, 8192, 4, 64, 2]);  convert_element_type_1008 = None
        view_as_complex_49 = torch.ops.aten.view_as_complex.default(view_1375);  view_1375 = None
        mul_309 = torch.ops.aten.mul.Tensor(view_as_complex_49, _conj);  view_as_complex_49 = None
        view_as_real_48 = torch.ops.aten.view_as_real.default(mul_308);  mul_308 = None
        view_1376 = torch.ops.aten.view.default(view_as_real_48, [2, 8192, 1, 128]);  view_as_real_48 = None
        convert_element_type_1009 = torch.ops.prims.convert_element_type.default(view_1376, torch.bfloat16);  view_1376 = None
        view_as_real_49 = torch.ops.aten.view_as_real.default(mul_309);  mul_309 = None
        view_1377 = torch.ops.aten.view.default(view_as_real_49, [2, 8192, 4, 128]);  view_as_real_49 = None
        convert_element_type_1010 = torch.ops.prims.convert_element_type.default(view_1377, torch.bfloat16);  view_1377 = None
        view_1378 = torch.ops.aten.view.default(squeeze_16, [2, 8192, 128]);  squeeze_16 = None
        view_1379 = torch.ops.aten.view.default(convert_element_type_1009, [2, 8192, 128]);  convert_element_type_1009 = None
        view_1380 = torch.ops.aten.view.default(convert_element_type_1010, [2, 8192, 512]);  convert_element_type_1010 = None
        view_1381 = torch.ops.aten.view.default(view_1378, [16384, 128]);  view_1378 = None
        permute_457 = torch.ops.aten.permute.default(view_1381, [1, 0])
        mm_235 = torch.ops.aten.mm.default(permute_457, view_519);  permute_457 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        all_gather_into_tensor_82 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_241, 8, '0');  convert_element_type_241 = None
        wait_tensor_97 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_82);  all_gather_into_tensor_82 = None
        permute_79 = torch.ops.aten.permute.default(wait_tensor_97, [1, 0]);  wait_tensor_97 = None
        permute_459 = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_236 = torch.ops.aten.mm.default(view_1381, permute_459);  view_1381 = permute_459 = None
        view_1382 = torch.ops.aten.view.default(mm_236, [2, 8192, 4096]);  mm_236 = None
        convert_element_type_1015 = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        reduce_scatter_tensor_130 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1015, 'avg', 8, '0');  convert_element_type_1015 = None
        wait_tensor_346 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_130);  reduce_scatter_tensor_130 = None
        view_1383 = torch.ops.aten.view.default(view_1379, [16384, 128]);  view_1379 = None
        permute_461 = torch.ops.aten.permute.default(view_1383, [1, 0])
        mm_237 = torch.ops.aten.mm.default(permute_461, view_519);  permute_461 = None
        permute_463 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_238 = torch.ops.aten.mm.default(view_1383, permute_463);  view_1383 = permute_463 = None
        view_1384 = torch.ops.aten.view.default(mm_238, [2, 8192, 4096]);  mm_238 = None
        add_125 = torch.ops.aten.add.Tensor(view_1382, view_1384);  view_1382 = view_1384 = None
        convert_element_type_1020 = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        reduce_scatter_tensor_131 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1020, 'avg', 8, '0');  convert_element_type_1020 = None
        wait_tensor_347 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_131);  reduce_scatter_tensor_131 = None
        view_1385 = torch.ops.aten.view.default(view_1380, [16384, 512]);  view_1380 = None
        permute_465 = torch.ops.aten.permute.default(view_1385, [1, 0])
        mm_239 = torch.ops.aten.mm.default(permute_465, view_519);  permute_465 = view_519 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        all_gather_into_tensor_80 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_235, 8, '0');  convert_element_type_235 = None
        wait_tensor_95 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_80);  all_gather_into_tensor_80 = None
        permute_77 = torch.ops.aten.permute.default(wait_tensor_95, [1, 0]);  wait_tensor_95 = None
        permute_467 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_240 = torch.ops.aten.mm.default(view_1385, permute_467);  view_1385 = permute_467 = None
        view_1386 = torch.ops.aten.view.default(mm_240, [2, 8192, 4096]);  mm_240 = None
        add_126 = torch.ops.aten.add.Tensor(add_125, view_1386);  add_125 = view_1386 = None
        convert_element_type_1025 = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None
        reduce_scatter_tensor_132 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1025, 'avg', 8, '0');  convert_element_type_1025 = None
        wait_tensor_348 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_132);  reduce_scatter_tensor_132 = None
        split_110 = torch.ops.aten.split.Tensor(add_126, 1024, 1);  add_126 = None
        getitem_1051 = split_110[0]
        getitem_1052 = split_110[1]
        getitem_1053 = split_110[2]
        getitem_1054 = split_110[3]
        getitem_1055 = split_110[4]
        getitem_1056 = split_110[5]
        getitem_1057 = split_110[6]
        getitem_1058 = split_110[7];  split_110 = None
        cat_102 = torch.ops.aten.cat.default([getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058]);  getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = None
        reduce_scatter_tensor_133 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_102, 'sum', 8, '1');  cat_102 = None
        wait_tensor_349 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_133);  reduce_scatter_tensor_133 = None
        convert_element_type_1026 = torch.ops.prims.convert_element_type.default(wait_tensor_349, torch.float32);  wait_tensor_349 = None
        convert_element_type_1028 = torch.ops.prims.convert_element_type.default(wait_tensor_93, torch.float32);  wait_tensor_93 = None
        mul_310 = torch.ops.aten.mul.Tensor(convert_element_type_1026, convert_element_type_1028);  convert_element_type_1028 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_56, mul_310)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        div_18 = torch.ops.aten.div.Tensor(mul_56, 4096)
        mul_313 = torch.ops.aten.mul.Tensor(div_18, sum_55);  div_18 = sum_55 = None
        sub_28 = torch.ops.aten.sub.Tensor(mul_310, mul_313);  mul_310 = mul_313 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_14);  sub_28 = rsqrt_14 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_1026, mul_56);  convert_element_type_1026 = mul_56 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        convert_element_type_1029 = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16);  mul_314 = None
        convert_element_type_1030 = torch.ops.prims.convert_element_type.default(sum_56, torch.bfloat16);  sum_56 = None
        all_reduce_18 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1030, 'sum', '1');  convert_element_type_1030 = None
        wait_tensor_350 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_18);  all_reduce_18 = None
        convert_element_type_1031 = torch.ops.prims.convert_element_type.default(wait_tensor_350, torch.float32);  wait_tensor_350 = None
        reduce_scatter_tensor_134 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1031, 'avg', 8, '0');  convert_element_type_1031 = None
        wait_tensor_351 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_134);  reduce_scatter_tensor_134 = None
        add_127 = torch.ops.aten.add.Tensor(add_124, convert_element_type_1029);  add_124 = convert_element_type_1029 = None
        all_gather_into_tensor_198 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_127, 8, '1')
        wait_tensor_352 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_198);  all_gather_into_tensor_198 = None
        split_111 = torch.ops.aten.split.Tensor(wait_tensor_352, 2);  wait_tensor_352 = None
        getitem_1059 = split_111[0]
        getitem_1060 = split_111[1]
        getitem_1061 = split_111[2]
        getitem_1062 = split_111[3]
        getitem_1063 = split_111[4]
        getitem_1064 = split_111[5]
        getitem_1065 = split_111[6]
        getitem_1066 = split_111[7];  split_111 = None
        cat_103 = torch.ops.aten.cat.default([getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066], 1);  getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = None
        view_1387 = torch.ops.aten.view.default(cat_103, [16384, 4096]);  cat_103 = None
        permute_469 = torch.ops.aten.permute.default(view_1387, [1, 0])
        wait_tensor_86 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_13);  reduce_scatter_tensor_13 = None
        add_25 = torch.ops.aten.add.Tensor(add_23, wait_tensor_86);  wait_tensor_86 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        all_gather_into_tensor_73 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_218, 8, '0');  convert_element_type_218 = None
        wait_tensor_87 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_73);  all_gather_into_tensor_73 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_219, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [2], True);  pow_14 = None
        add_26 = torch.ops.aten.add.Scalar(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_219, rsqrt_13);  convert_element_type_219 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, wait_tensor_87)
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
        view_492 = torch.ops.aten.view.default(cat_27, [16384, 4096]);  cat_27 = None
        view_493 = torch.ops.aten.view.default(mm_46, [2, 8192, 1792]);  mm_46 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_224)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_224, sigmoid_6);  sigmoid_6 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        all_gather_into_tensor_76 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_226, 8, '0');  convert_element_type_226 = None
        wait_tensor_90 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_76);  all_gather_into_tensor_76 = None
        permute_75 = torch.ops.aten.permute.default(wait_tensor_90, [1, 0]);  wait_tensor_90 = None
        mm_47 = torch.ops.aten.mm.default(view_492, permute_75)
        view_500 = torch.ops.aten.view.default(mm_47, [2, 8192, 1792]);  mm_47 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_225, view_500)
        view_507 = torch.ops.aten.view.default(mul_55, [16384, 1792]);  mul_55 = None
        mm_241 = torch.ops.aten.mm.default(permute_469, view_507);  permute_469 = view_507 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        all_gather_into_tensor_77 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_229, 8, '0');  convert_element_type_229 = None
        wait_tensor_91 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_77);  all_gather_into_tensor_77 = None
        permute_76 = torch.ops.aten.permute.default(wait_tensor_91, [1, 0]);  wait_tensor_91 = None
        permute_471 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_242 = torch.ops.aten.mm.default(view_1387, permute_471);  view_1387 = permute_471 = None
        view_1388 = torch.ops.aten.view.default(mm_242, [2, 8192, 1792]);  mm_242 = None
        convert_element_type_1036 = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None
        reduce_scatter_tensor_135 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1036, 'avg', 8, '0');  convert_element_type_1036 = None
        wait_tensor_353 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_135);  reduce_scatter_tensor_135 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_1388, convert_element_type_225);  convert_element_type_225 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_1388, view_500);  view_1388 = view_500 = None
        view_1389 = torch.ops.aten.view.default(mul_316, [16384, 1792]);  mul_316 = None
        permute_473 = torch.ops.aten.permute.default(view_1389, [1, 0])
        mm_243 = torch.ops.aten.mm.default(permute_473, view_492);  permute_473 = None
        permute_475 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_244 = torch.ops.aten.mm.default(view_1389, permute_475);  view_1389 = permute_475 = None
        view_1390 = torch.ops.aten.view.default(mm_244, [2, 8192, 4096]);  mm_244 = None
        convert_element_type_1041 = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        reduce_scatter_tensor_136 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1041, 'avg', 8, '0');  convert_element_type_1041 = None
        wait_tensor_354 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_136);  reduce_scatter_tensor_136 = None
        convert_element_type_1042 = torch.ops.prims.convert_element_type.default(mul_317, torch.float32);  mul_317 = None
        neg_9 = torch.ops.aten.neg.default(convert_element_type_224)
        exp_9 = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_128 = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(add_128);  add_128 = None
        mul_318 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_319 = torch.ops.aten.mul.Tensor(convert_element_type_1042, mul_318);  convert_element_type_1042 = None
        sub_29 = torch.ops.aten.sub.Tensor(1, mul_318);  mul_318 = None
        mul_320 = torch.ops.aten.mul.Tensor(convert_element_type_224, sub_29);  convert_element_type_224 = sub_29 = None
        add_129 = torch.ops.aten.add.Tensor(mul_320, 1);  mul_320 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_319, add_129);  mul_319 = add_129 = None
        convert_element_type_1044 = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        view_1391 = torch.ops.aten.view.default(convert_element_type_1044, [16384, 1792]);  convert_element_type_1044 = None
        permute_477 = torch.ops.aten.permute.default(view_1391, [1, 0])
        mm_245 = torch.ops.aten.mm.default(permute_477, view_492);  permute_477 = view_492 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        all_gather_into_tensor_75 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_221, 8, '0');  convert_element_type_221 = None
        wait_tensor_89 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_75);  all_gather_into_tensor_75 = None
        permute_74 = torch.ops.aten.permute.default(wait_tensor_89, [1, 0]);  wait_tensor_89 = None
        permute_479 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_246 = torch.ops.aten.mm.default(view_1391, permute_479);  view_1391 = permute_479 = None
        view_1392 = torch.ops.aten.view.default(mm_246, [2, 8192, 4096]);  mm_246 = None
        add_130 = torch.ops.aten.add.Tensor(view_1390, view_1392);  view_1390 = view_1392 = None
        convert_element_type_1049 = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        reduce_scatter_tensor_137 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1049, 'avg', 8, '0');  convert_element_type_1049 = None
        wait_tensor_355 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_137);  reduce_scatter_tensor_137 = None
        split_112 = torch.ops.aten.split.Tensor(add_130, 1024, 1);  add_130 = None
        getitem_1067 = split_112[0]
        getitem_1068 = split_112[1]
        getitem_1069 = split_112[2]
        getitem_1070 = split_112[3]
        getitem_1071 = split_112[4]
        getitem_1072 = split_112[5]
        getitem_1073 = split_112[6]
        getitem_1074 = split_112[7];  split_112 = None
        cat_104 = torch.ops.aten.cat.default([getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074]);  getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = None
        reduce_scatter_tensor_138 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_104, 'sum', 8, '1');  cat_104 = None
        wait_tensor_356 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_138);  reduce_scatter_tensor_138 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(wait_tensor_356, torch.float32);  wait_tensor_356 = None
        convert_element_type_1052 = torch.ops.prims.convert_element_type.default(wait_tensor_87, torch.float32);  wait_tensor_87 = None
        mul_322 = torch.ops.aten.mul.Tensor(convert_element_type_1050, convert_element_type_1052);  convert_element_type_1052 = None
        mul_324 = torch.ops.aten.mul.Tensor(mul_52, mul_322)
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
        div_19 = torch.ops.aten.div.Tensor(mul_52, 4096)
        mul_325 = torch.ops.aten.mul.Tensor(div_19, sum_57);  div_19 = sum_57 = None
        sub_30 = torch.ops.aten.sub.Tensor(mul_322, mul_325);  mul_322 = mul_325 = None
        mul_326 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_13);  sub_30 = rsqrt_13 = None
        mul_327 = torch.ops.aten.mul.Tensor(convert_element_type_1050, mul_52);  convert_element_type_1050 = mul_52 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1]);  mul_327 = None
        convert_element_type_1053 = torch.ops.prims.convert_element_type.default(mul_326, torch.bfloat16);  mul_326 = None
        convert_element_type_1054 = torch.ops.prims.convert_element_type.default(sum_58, torch.bfloat16);  sum_58 = None
        all_reduce_19 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1054, 'sum', '1');  convert_element_type_1054 = None
        wait_tensor_357 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_19);  all_reduce_19 = None
        convert_element_type_1055 = torch.ops.prims.convert_element_type.default(wait_tensor_357, torch.float32);  wait_tensor_357 = None
        reduce_scatter_tensor_139 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1055, 'avg', 8, '0');  convert_element_type_1055 = None
        wait_tensor_358 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_139);  reduce_scatter_tensor_139 = None
        add_131 = torch.ops.aten.add.Tensor(add_127, convert_element_type_1053);  add_127 = convert_element_type_1053 = None
        all_gather_into_tensor_199 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_131, 8, '1')
        wait_tensor_359 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_199);  all_gather_into_tensor_199 = None
        split_113 = torch.ops.aten.split.Tensor(wait_tensor_359, 2);  wait_tensor_359 = None
        getitem_1075 = split_113[0]
        getitem_1076 = split_113[1]
        getitem_1077 = split_113[2]
        getitem_1078 = split_113[3]
        getitem_1079 = split_113[4]
        getitem_1080 = split_113[5]
        getitem_1081 = split_113[6]
        getitem_1082 = split_113[7];  split_113 = None
        cat_105 = torch.ops.aten.cat.default([getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082], 1);  getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = getitem_1081 = getitem_1082 = None
        view_1393 = torch.ops.aten.view.default(cat_105, [16384, 4096]);  cat_105 = None
        permute_481 = torch.ops.aten.permute.default(view_1393, [1, 0])
        permute_72 = torch.ops.aten.permute.default(getitem_326, [0, 2, 1, 3])
        view_474 = torch.ops.aten.view.default(permute_72, [2, 8192, -1]);  permute_72 = None
        view_480 = torch.ops.aten.view.default(view_474, [16384, 512]);  view_474 = None
        mm_247 = torch.ops.aten.mm.default(permute_481, view_480);  permute_481 = view_480 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        all_gather_into_tensor_72 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_215, 8, '0');  convert_element_type_215 = None
        wait_tensor_85 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_72);  all_gather_into_tensor_72 = None
        permute_73 = torch.ops.aten.permute.default(wait_tensor_85, [1, 0]);  wait_tensor_85 = None
        permute_483 = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        mm_248 = torch.ops.aten.mm.default(view_1393, permute_483);  view_1393 = permute_483 = None
        view_1394 = torch.ops.aten.view.default(mm_248, [2, 8192, 512]);  mm_248 = None
        convert_element_type_1060 = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        reduce_scatter_tensor_140 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1060, 'avg', 8, '0');  convert_element_type_1060 = None
        wait_tensor_360 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_140);  reduce_scatter_tensor_140 = None
        view_1395 = torch.ops.aten.view.default(view_1394, [2, 8192, 4, 128]);  view_1394 = None
        permute_485 = torch.ops.aten.permute.default(view_1395, [0, 2, 1, 3]);  view_1395 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        all_gather_into_tensor_67 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_199, 8, '0');  convert_element_type_199 = None
        wait_tensor_80 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_67);  all_gather_into_tensor_67 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [2], True);  pow_13 = None
        add_24 = torch.ops.aten.add.Scalar(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_12);  convert_element_type_200 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, wait_tensor_80)
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
        view_447 = torch.ops.aten.view.default(cat_25, [16384, 4096]);  cat_25 = None
        view_448 = torch.ops.aten.view.default(mm_42, [2, 8192, 512]);  mm_42 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        all_gather_into_tensor_70 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_205, 8, '0');  convert_element_type_205 = None
        wait_tensor_83 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_70);  all_gather_into_tensor_70 = None
        permute_67 = torch.ops.aten.permute.default(wait_tensor_83, [1, 0]);  wait_tensor_83 = None
        mm_43 = torch.ops.aten.mm.default(view_447, permute_67)
        view_455 = torch.ops.aten.view.default(mm_43, [2, 8192, 128]);  mm_43 = None
        view_462 = torch.ops.aten.view.default(mm_44, [2, 8192, 128]);  mm_44 = None
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
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_485, permute_69, permute_70, permute_71, getitem_326, getitem_327, getitem_332, getitem_333, None, None, None, 8192, 8192, 0.0, True);  permute_485 = permute_69 = permute_70 = permute_71 = getitem_326 = getitem_327 = getitem_332 = getitem_333 = None
        getitem_1083 = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_1084 = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_1085 = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None
        permute_486 = torch.ops.aten.permute.default(getitem_1085, [0, 2, 1, 3]);  getitem_1085 = None
        permute_487 = torch.ops.aten.permute.default(getitem_1084, [0, 2, 1, 3]);  getitem_1084 = None
        permute_488 = torch.ops.aten.permute.default(getitem_1083, [0, 2, 1, 3]);  getitem_1083 = None
        view_1396 = torch.ops.aten.view.default(permute_486, [2, 8192, 1, 4, 128]);  permute_486 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(view_1396, [3], True);  view_1396 = None
        squeeze_18 = torch.ops.aten.squeeze.dim(sum_59, 3);  sum_59 = None
        view_1397 = torch.ops.aten.view.default(permute_487, [2, 8192, 1, 4, 128]);  permute_487 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(view_1397, [3], True);  view_1397 = None
        squeeze_19 = torch.ops.aten.squeeze.dim(sum_60, 3);  sum_60 = None
        convert_element_type_1061 = torch.ops.prims.convert_element_type.default(squeeze_19, torch.float32);  squeeze_19 = None
        convert_element_type_1062 = torch.ops.prims.convert_element_type.default(permute_488, torch.float32);  permute_488 = None
        view_1398 = torch.ops.aten.view.default(convert_element_type_1061, [2, 8192, 1, 64, 2]);  convert_element_type_1061 = None
        view_as_complex_50 = torch.ops.aten.view_as_complex.default(view_1398);  view_1398 = None
        mul_328 = torch.ops.aten.mul.Tensor(view_as_complex_50, _conj);  view_as_complex_50 = None
        view_1399 = torch.ops.aten.view.default(convert_element_type_1062, [2, 8192, 4, 64, 2]);  convert_element_type_1062 = None
        view_as_complex_51 = torch.ops.aten.view_as_complex.default(view_1399);  view_1399 = None
        mul_329 = torch.ops.aten.mul.Tensor(view_as_complex_51, _conj);  view_as_complex_51 = None
        view_as_real_50 = torch.ops.aten.view_as_real.default(mul_328);  mul_328 = None
        view_1400 = torch.ops.aten.view.default(view_as_real_50, [2, 8192, 1, 128]);  view_as_real_50 = None
        convert_element_type_1063 = torch.ops.prims.convert_element_type.default(view_1400, torch.bfloat16);  view_1400 = None
        view_as_real_51 = torch.ops.aten.view_as_real.default(mul_329);  mul_329 = None
        view_1401 = torch.ops.aten.view.default(view_as_real_51, [2, 8192, 4, 128]);  view_as_real_51 = None
        convert_element_type_1064 = torch.ops.prims.convert_element_type.default(view_1401, torch.bfloat16);  view_1401 = None
        view_1402 = torch.ops.aten.view.default(squeeze_18, [2, 8192, 128]);  squeeze_18 = None
        view_1403 = torch.ops.aten.view.default(convert_element_type_1063, [2, 8192, 128]);  convert_element_type_1063 = None
        view_1404 = torch.ops.aten.view.default(convert_element_type_1064, [2, 8192, 512]);  convert_element_type_1064 = None
        view_1405 = torch.ops.aten.view.default(view_1402, [16384, 128]);  view_1402 = None
        permute_489 = torch.ops.aten.permute.default(view_1405, [1, 0])
        mm_249 = torch.ops.aten.mm.default(permute_489, view_447);  permute_489 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        all_gather_into_tensor_71 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_208, 8, '0');  convert_element_type_208 = None
        wait_tensor_84 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_71);  all_gather_into_tensor_71 = None
        permute_68 = torch.ops.aten.permute.default(wait_tensor_84, [1, 0]);  wait_tensor_84 = None
        permute_491 = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_250 = torch.ops.aten.mm.default(view_1405, permute_491);  view_1405 = permute_491 = None
        view_1406 = torch.ops.aten.view.default(mm_250, [2, 8192, 4096]);  mm_250 = None
        convert_element_type_1069 = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        reduce_scatter_tensor_141 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1069, 'avg', 8, '0');  convert_element_type_1069 = None
        wait_tensor_361 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_141);  reduce_scatter_tensor_141 = None
        view_1407 = torch.ops.aten.view.default(view_1403, [16384, 128]);  view_1403 = None
        permute_493 = torch.ops.aten.permute.default(view_1407, [1, 0])
        mm_251 = torch.ops.aten.mm.default(permute_493, view_447);  permute_493 = None
        permute_495 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_252 = torch.ops.aten.mm.default(view_1407, permute_495);  view_1407 = permute_495 = None
        view_1408 = torch.ops.aten.view.default(mm_252, [2, 8192, 4096]);  mm_252 = None
        add_132 = torch.ops.aten.add.Tensor(view_1406, view_1408);  view_1406 = view_1408 = None
        convert_element_type_1074 = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        reduce_scatter_tensor_142 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1074, 'avg', 8, '0');  convert_element_type_1074 = None
        wait_tensor_362 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_142);  reduce_scatter_tensor_142 = None
        view_1409 = torch.ops.aten.view.default(view_1404, [16384, 512]);  view_1404 = None
        permute_497 = torch.ops.aten.permute.default(view_1409, [1, 0])
        mm_253 = torch.ops.aten.mm.default(permute_497, view_447);  permute_497 = view_447 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        all_gather_into_tensor_69 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_202, 8, '0');  convert_element_type_202 = None
        wait_tensor_82 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_69);  all_gather_into_tensor_69 = None
        permute_66 = torch.ops.aten.permute.default(wait_tensor_82, [1, 0]);  wait_tensor_82 = None
        permute_499 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_254 = torch.ops.aten.mm.default(view_1409, permute_499);  view_1409 = permute_499 = None
        view_1410 = torch.ops.aten.view.default(mm_254, [2, 8192, 4096]);  mm_254 = None
        add_133 = torch.ops.aten.add.Tensor(add_132, view_1410);  add_132 = view_1410 = None
        convert_element_type_1079 = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None
        reduce_scatter_tensor_143 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1079, 'avg', 8, '0');  convert_element_type_1079 = None
        wait_tensor_363 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_143);  reduce_scatter_tensor_143 = None
        split_114 = torch.ops.aten.split.Tensor(add_133, 1024, 1);  add_133 = None
        getitem_1086 = split_114[0]
        getitem_1087 = split_114[1]
        getitem_1088 = split_114[2]
        getitem_1089 = split_114[3]
        getitem_1090 = split_114[4]
        getitem_1091 = split_114[5]
        getitem_1092 = split_114[6]
        getitem_1093 = split_114[7];  split_114 = None
        cat_106 = torch.ops.aten.cat.default([getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093]);  getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = None
        reduce_scatter_tensor_144 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_106, 'sum', 8, '1');  cat_106 = None
        wait_tensor_364 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_144);  reduce_scatter_tensor_144 = None
        convert_element_type_1080 = torch.ops.prims.convert_element_type.default(wait_tensor_364, torch.float32);  wait_tensor_364 = None
        convert_element_type_1082 = torch.ops.prims.convert_element_type.default(wait_tensor_80, torch.float32);  wait_tensor_80 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_1080, convert_element_type_1082);  convert_element_type_1082 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_48, mul_330)
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        div_20 = torch.ops.aten.div.Tensor(mul_48, 4096)
        mul_333 = torch.ops.aten.mul.Tensor(div_20, sum_61);  div_20 = sum_61 = None
        sub_31 = torch.ops.aten.sub.Tensor(mul_330, mul_333);  mul_330 = mul_333 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_31, rsqrt_12);  sub_31 = rsqrt_12 = None
        mul_335 = torch.ops.aten.mul.Tensor(convert_element_type_1080, mul_48);  convert_element_type_1080 = mul_48 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        convert_element_type_1083 = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        convert_element_type_1084 = torch.ops.prims.convert_element_type.default(sum_62, torch.bfloat16);  sum_62 = None
        all_reduce_20 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1084, 'sum', '1');  convert_element_type_1084 = None
        wait_tensor_365 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_20);  all_reduce_20 = None
        convert_element_type_1085 = torch.ops.prims.convert_element_type.default(wait_tensor_365, torch.float32);  wait_tensor_365 = None
        reduce_scatter_tensor_145 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1085, 'avg', 8, '0');  convert_element_type_1085 = None
        wait_tensor_366 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_145);  reduce_scatter_tensor_145 = None
        add_134 = torch.ops.aten.add.Tensor(add_131, convert_element_type_1083);  add_131 = convert_element_type_1083 = None
        all_gather_into_tensor_200 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_134, 8, '1')
        wait_tensor_367 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_200);  all_gather_into_tensor_200 = None
        split_115 = torch.ops.aten.split.Tensor(wait_tensor_367, 2);  wait_tensor_367 = None
        getitem_1094 = split_115[0]
        getitem_1095 = split_115[1]
        getitem_1096 = split_115[2]
        getitem_1097 = split_115[3]
        getitem_1098 = split_115[4]
        getitem_1099 = split_115[5]
        getitem_1100 = split_115[6]
        getitem_1101 = split_115[7];  split_115 = None
        cat_107 = torch.ops.aten.cat.default([getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101], 1);  getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = None
        view_1411 = torch.ops.aten.view.default(cat_107, [16384, 4096]);  cat_107 = None
        permute_501 = torch.ops.aten.permute.default(view_1411, [1, 0])
        wait_tensor_73 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_11);  reduce_scatter_tensor_11 = None
        add_21 = torch.ops.aten.add.Tensor(add_19, wait_tensor_73);  wait_tensor_73 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        all_gather_into_tensor_62 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_185, 8, '0');  convert_element_type_185 = None
        wait_tensor_74 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_62);  all_gather_into_tensor_62 = None
        convert_element_type_186 = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [2], True);  pow_12 = None
        add_22 = torch.ops.aten.add.Scalar(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_186, rsqrt_11);  convert_element_type_186 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, wait_tensor_74)
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
        view_420 = torch.ops.aten.view.default(cat_23, [16384, 4096]);  cat_23 = None
        view_421 = torch.ops.aten.view.default(mm_39, [2, 8192, 1792]);  mm_39 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_191)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_191, sigmoid_5);  sigmoid_5 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        all_gather_into_tensor_65 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_193, 8, '0');  convert_element_type_193 = None
        wait_tensor_77 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_65);  all_gather_into_tensor_65 = None
        permute_64 = torch.ops.aten.permute.default(wait_tensor_77, [1, 0]);  wait_tensor_77 = None
        mm_40 = torch.ops.aten.mm.default(view_420, permute_64)
        view_428 = torch.ops.aten.view.default(mm_40, [2, 8192, 1792]);  mm_40 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_192, view_428)
        view_435 = torch.ops.aten.view.default(mul_47, [16384, 1792]);  mul_47 = None
        mm_255 = torch.ops.aten.mm.default(permute_501, view_435);  permute_501 = view_435 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        all_gather_into_tensor_66 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_196, 8, '0');  convert_element_type_196 = None
        wait_tensor_78 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_66);  all_gather_into_tensor_66 = None
        permute_65 = torch.ops.aten.permute.default(wait_tensor_78, [1, 0]);  wait_tensor_78 = None
        permute_503 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_256 = torch.ops.aten.mm.default(view_1411, permute_503);  view_1411 = permute_503 = None
        view_1412 = torch.ops.aten.view.default(mm_256, [2, 8192, 1792]);  mm_256 = None
        convert_element_type_1090 = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        reduce_scatter_tensor_146 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1090, 'avg', 8, '0');  convert_element_type_1090 = None
        wait_tensor_368 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_146);  reduce_scatter_tensor_146 = None
        mul_336 = torch.ops.aten.mul.Tensor(view_1412, convert_element_type_192);  convert_element_type_192 = None
        mul_337 = torch.ops.aten.mul.Tensor(view_1412, view_428);  view_1412 = view_428 = None
        view_1413 = torch.ops.aten.view.default(mul_336, [16384, 1792]);  mul_336 = None
        permute_505 = torch.ops.aten.permute.default(view_1413, [1, 0])
        mm_257 = torch.ops.aten.mm.default(permute_505, view_420);  permute_505 = None
        permute_507 = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_258 = torch.ops.aten.mm.default(view_1413, permute_507);  view_1413 = permute_507 = None
        view_1414 = torch.ops.aten.view.default(mm_258, [2, 8192, 4096]);  mm_258 = None
        convert_element_type_1095 = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        reduce_scatter_tensor_147 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1095, 'avg', 8, '0');  convert_element_type_1095 = None
        wait_tensor_369 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_147);  reduce_scatter_tensor_147 = None
        convert_element_type_1096 = torch.ops.prims.convert_element_type.default(mul_337, torch.float32);  mul_337 = None
        neg_10 = torch.ops.aten.neg.default(convert_element_type_191)
        exp_10 = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_135 = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(add_135);  add_135 = None
        mul_338 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_339 = torch.ops.aten.mul.Tensor(convert_element_type_1096, mul_338);  convert_element_type_1096 = None
        sub_32 = torch.ops.aten.sub.Tensor(1, mul_338);  mul_338 = None
        mul_340 = torch.ops.aten.mul.Tensor(convert_element_type_191, sub_32);  convert_element_type_191 = sub_32 = None
        add_136 = torch.ops.aten.add.Tensor(mul_340, 1);  mul_340 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_339, add_136);  mul_339 = add_136 = None
        convert_element_type_1098 = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16);  mul_341 = None
        view_1415 = torch.ops.aten.view.default(convert_element_type_1098, [16384, 1792]);  convert_element_type_1098 = None
        permute_509 = torch.ops.aten.permute.default(view_1415, [1, 0])
        mm_259 = torch.ops.aten.mm.default(permute_509, view_420);  permute_509 = view_420 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        all_gather_into_tensor_64 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_188, 8, '0');  convert_element_type_188 = None
        wait_tensor_76 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_64);  all_gather_into_tensor_64 = None
        permute_63 = torch.ops.aten.permute.default(wait_tensor_76, [1, 0]);  wait_tensor_76 = None
        permute_511 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_260 = torch.ops.aten.mm.default(view_1415, permute_511);  view_1415 = permute_511 = None
        view_1416 = torch.ops.aten.view.default(mm_260, [2, 8192, 4096]);  mm_260 = None
        add_137 = torch.ops.aten.add.Tensor(view_1414, view_1416);  view_1414 = view_1416 = None
        convert_element_type_1103 = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        reduce_scatter_tensor_148 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1103, 'avg', 8, '0');  convert_element_type_1103 = None
        wait_tensor_370 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_148);  reduce_scatter_tensor_148 = None
        split_116 = torch.ops.aten.split.Tensor(add_137, 1024, 1);  add_137 = None
        getitem_1102 = split_116[0]
        getitem_1103 = split_116[1]
        getitem_1104 = split_116[2]
        getitem_1105 = split_116[3]
        getitem_1106 = split_116[4]
        getitem_1107 = split_116[5]
        getitem_1108 = split_116[6]
        getitem_1109 = split_116[7];  split_116 = None
        cat_108 = torch.ops.aten.cat.default([getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109]);  getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = None
        reduce_scatter_tensor_149 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_108, 'sum', 8, '1');  cat_108 = None
        wait_tensor_371 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_149);  reduce_scatter_tensor_149 = None
        convert_element_type_1104 = torch.ops.prims.convert_element_type.default(wait_tensor_371, torch.float32);  wait_tensor_371 = None
        convert_element_type_1106 = torch.ops.prims.convert_element_type.default(wait_tensor_74, torch.float32);  wait_tensor_74 = None
        mul_342 = torch.ops.aten.mul.Tensor(convert_element_type_1104, convert_element_type_1106);  convert_element_type_1106 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_44, mul_342)
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        div_21 = torch.ops.aten.div.Tensor(mul_44, 4096)
        mul_345 = torch.ops.aten.mul.Tensor(div_21, sum_63);  div_21 = sum_63 = None
        sub_33 = torch.ops.aten.sub.Tensor(mul_342, mul_345);  mul_342 = mul_345 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_11);  sub_33 = rsqrt_11 = None
        mul_347 = torch.ops.aten.mul.Tensor(convert_element_type_1104, mul_44);  convert_element_type_1104 = mul_44 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        convert_element_type_1107 = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        convert_element_type_1108 = torch.ops.prims.convert_element_type.default(sum_64, torch.bfloat16);  sum_64 = None
        all_reduce_21 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1108, 'sum', '1');  convert_element_type_1108 = None
        wait_tensor_372 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_21);  all_reduce_21 = None
        convert_element_type_1109 = torch.ops.prims.convert_element_type.default(wait_tensor_372, torch.float32);  wait_tensor_372 = None
        reduce_scatter_tensor_150 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1109, 'avg', 8, '0');  convert_element_type_1109 = None
        wait_tensor_373 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_150);  reduce_scatter_tensor_150 = None
        add_138 = torch.ops.aten.add.Tensor(add_134, convert_element_type_1107);  add_134 = convert_element_type_1107 = None
        all_gather_into_tensor_201 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_138, 8, '1')
        wait_tensor_374 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_201);  all_gather_into_tensor_201 = None
        split_117 = torch.ops.aten.split.Tensor(wait_tensor_374, 2);  wait_tensor_374 = None
        getitem_1110 = split_117[0]
        getitem_1111 = split_117[1]
        getitem_1112 = split_117[2]
        getitem_1113 = split_117[3]
        getitem_1114 = split_117[4]
        getitem_1115 = split_117[5]
        getitem_1116 = split_117[6]
        getitem_1117 = split_117[7];  split_117 = None
        cat_109 = torch.ops.aten.cat.default([getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117], 1);  getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = None
        view_1417 = torch.ops.aten.view.default(cat_109, [16384, 4096]);  cat_109 = None
        permute_513 = torch.ops.aten.permute.default(view_1417, [1, 0])
        permute_61 = torch.ops.aten.permute.default(getitem_285, [0, 2, 1, 3])
        view_402 = torch.ops.aten.view.default(permute_61, [2, 8192, -1]);  permute_61 = None
        view_408 = torch.ops.aten.view.default(view_402, [16384, 512]);  view_402 = None
        mm_261 = torch.ops.aten.mm.default(permute_513, view_408);  permute_513 = view_408 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        all_gather_into_tensor_61 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_182, 8, '0');  convert_element_type_182 = None
        wait_tensor_72 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_61);  all_gather_into_tensor_61 = None
        permute_62 = torch.ops.aten.permute.default(wait_tensor_72, [1, 0]);  wait_tensor_72 = None
        permute_515 = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_262 = torch.ops.aten.mm.default(view_1417, permute_515);  view_1417 = permute_515 = None
        view_1418 = torch.ops.aten.view.default(mm_262, [2, 8192, 512]);  mm_262 = None
        convert_element_type_1114 = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        reduce_scatter_tensor_151 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1114, 'avg', 8, '0');  convert_element_type_1114 = None
        wait_tensor_375 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_151);  reduce_scatter_tensor_151 = None
        view_1419 = torch.ops.aten.view.default(view_1418, [2, 8192, 4, 128]);  view_1418 = None
        permute_517 = torch.ops.aten.permute.default(view_1419, [0, 2, 1, 3]);  view_1419 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        all_gather_into_tensor_56 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_166, 8, '0');  convert_element_type_166 = None
        wait_tensor_67 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_56);  all_gather_into_tensor_56 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_167, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [2], True);  pow_11 = None
        add_20 = torch.ops.aten.add.Scalar(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_167, rsqrt_10);  convert_element_type_167 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, wait_tensor_67)
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
        view_375 = torch.ops.aten.view.default(cat_21, [16384, 4096]);  cat_21 = None
        view_376 = torch.ops.aten.view.default(mm_35, [2, 8192, 512]);  mm_35 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        all_gather_into_tensor_59 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_172, 8, '0');  convert_element_type_172 = None
        wait_tensor_70 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_59);  all_gather_into_tensor_59 = None
        permute_56 = torch.ops.aten.permute.default(wait_tensor_70, [1, 0]);  wait_tensor_70 = None
        mm_36 = torch.ops.aten.mm.default(view_375, permute_56)
        view_383 = torch.ops.aten.view.default(mm_36, [2, 8192, 128]);  mm_36 = None
        view_390 = torch.ops.aten.view.default(mm_37, [2, 8192, 128]);  mm_37 = None
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
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_517, permute_58, permute_59, permute_60, getitem_285, getitem_286, getitem_291, getitem_292, None, None, None, 8192, 8192, 0.0, True);  permute_517 = permute_58 = permute_59 = permute_60 = getitem_285 = getitem_286 = getitem_291 = getitem_292 = None
        getitem_1118 = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_1119 = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_1120 = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None
        permute_518 = torch.ops.aten.permute.default(getitem_1120, [0, 2, 1, 3]);  getitem_1120 = None
        permute_519 = torch.ops.aten.permute.default(getitem_1119, [0, 2, 1, 3]);  getitem_1119 = None
        permute_520 = torch.ops.aten.permute.default(getitem_1118, [0, 2, 1, 3]);  getitem_1118 = None
        view_1420 = torch.ops.aten.view.default(permute_518, [2, 8192, 1, 4, 128]);  permute_518 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(view_1420, [3], True);  view_1420 = None
        squeeze_20 = torch.ops.aten.squeeze.dim(sum_65, 3);  sum_65 = None
        view_1421 = torch.ops.aten.view.default(permute_519, [2, 8192, 1, 4, 128]);  permute_519 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(view_1421, [3], True);  view_1421 = None
        squeeze_21 = torch.ops.aten.squeeze.dim(sum_66, 3);  sum_66 = None
        convert_element_type_1115 = torch.ops.prims.convert_element_type.default(squeeze_21, torch.float32);  squeeze_21 = None
        convert_element_type_1116 = torch.ops.prims.convert_element_type.default(permute_520, torch.float32);  permute_520 = None
        view_1422 = torch.ops.aten.view.default(convert_element_type_1115, [2, 8192, 1, 64, 2]);  convert_element_type_1115 = None
        view_as_complex_52 = torch.ops.aten.view_as_complex.default(view_1422);  view_1422 = None
        mul_348 = torch.ops.aten.mul.Tensor(view_as_complex_52, _conj);  view_as_complex_52 = None
        view_1423 = torch.ops.aten.view.default(convert_element_type_1116, [2, 8192, 4, 64, 2]);  convert_element_type_1116 = None
        view_as_complex_53 = torch.ops.aten.view_as_complex.default(view_1423);  view_1423 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_as_complex_53, _conj);  view_as_complex_53 = None
        view_as_real_52 = torch.ops.aten.view_as_real.default(mul_348);  mul_348 = None
        view_1424 = torch.ops.aten.view.default(view_as_real_52, [2, 8192, 1, 128]);  view_as_real_52 = None
        convert_element_type_1117 = torch.ops.prims.convert_element_type.default(view_1424, torch.bfloat16);  view_1424 = None
        view_as_real_53 = torch.ops.aten.view_as_real.default(mul_349);  mul_349 = None
        view_1425 = torch.ops.aten.view.default(view_as_real_53, [2, 8192, 4, 128]);  view_as_real_53 = None
        convert_element_type_1118 = torch.ops.prims.convert_element_type.default(view_1425, torch.bfloat16);  view_1425 = None
        view_1426 = torch.ops.aten.view.default(squeeze_20, [2, 8192, 128]);  squeeze_20 = None
        view_1427 = torch.ops.aten.view.default(convert_element_type_1117, [2, 8192, 128]);  convert_element_type_1117 = None
        view_1428 = torch.ops.aten.view.default(convert_element_type_1118, [2, 8192, 512]);  convert_element_type_1118 = None
        view_1429 = torch.ops.aten.view.default(view_1426, [16384, 128]);  view_1426 = None
        permute_521 = torch.ops.aten.permute.default(view_1429, [1, 0])
        mm_263 = torch.ops.aten.mm.default(permute_521, view_375);  permute_521 = None
        convert_element_type_175 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        all_gather_into_tensor_60 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_175, 8, '0');  convert_element_type_175 = None
        wait_tensor_71 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_60);  all_gather_into_tensor_60 = None
        permute_57 = torch.ops.aten.permute.default(wait_tensor_71, [1, 0]);  wait_tensor_71 = None
        permute_523 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_264 = torch.ops.aten.mm.default(view_1429, permute_523);  view_1429 = permute_523 = None
        view_1430 = torch.ops.aten.view.default(mm_264, [2, 8192, 4096]);  mm_264 = None
        convert_element_type_1123 = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        reduce_scatter_tensor_152 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1123, 'avg', 8, '0');  convert_element_type_1123 = None
        wait_tensor_376 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_152);  reduce_scatter_tensor_152 = None
        view_1431 = torch.ops.aten.view.default(view_1427, [16384, 128]);  view_1427 = None
        permute_525 = torch.ops.aten.permute.default(view_1431, [1, 0])
        mm_265 = torch.ops.aten.mm.default(permute_525, view_375);  permute_525 = None
        permute_527 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_266 = torch.ops.aten.mm.default(view_1431, permute_527);  view_1431 = permute_527 = None
        view_1432 = torch.ops.aten.view.default(mm_266, [2, 8192, 4096]);  mm_266 = None
        add_139 = torch.ops.aten.add.Tensor(view_1430, view_1432);  view_1430 = view_1432 = None
        convert_element_type_1128 = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        reduce_scatter_tensor_153 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1128, 'avg', 8, '0');  convert_element_type_1128 = None
        wait_tensor_377 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_153);  reduce_scatter_tensor_153 = None
        view_1433 = torch.ops.aten.view.default(view_1428, [16384, 512]);  view_1428 = None
        permute_529 = torch.ops.aten.permute.default(view_1433, [1, 0])
        mm_267 = torch.ops.aten.mm.default(permute_529, view_375);  permute_529 = view_375 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        all_gather_into_tensor_58 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_169, 8, '0');  convert_element_type_169 = None
        wait_tensor_69 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_58);  all_gather_into_tensor_58 = None
        permute_55 = torch.ops.aten.permute.default(wait_tensor_69, [1, 0]);  wait_tensor_69 = None
        permute_531 = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_268 = torch.ops.aten.mm.default(view_1433, permute_531);  view_1433 = permute_531 = None
        view_1434 = torch.ops.aten.view.default(mm_268, [2, 8192, 4096]);  mm_268 = None
        add_140 = torch.ops.aten.add.Tensor(add_139, view_1434);  add_139 = view_1434 = None
        convert_element_type_1133 = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        reduce_scatter_tensor_154 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1133, 'avg', 8, '0');  convert_element_type_1133 = None
        wait_tensor_378 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_154);  reduce_scatter_tensor_154 = None
        split_118 = torch.ops.aten.split.Tensor(add_140, 1024, 1);  add_140 = None
        getitem_1121 = split_118[0]
        getitem_1122 = split_118[1]
        getitem_1123 = split_118[2]
        getitem_1124 = split_118[3]
        getitem_1125 = split_118[4]
        getitem_1126 = split_118[5]
        getitem_1127 = split_118[6]
        getitem_1128 = split_118[7];  split_118 = None
        cat_110 = torch.ops.aten.cat.default([getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128]);  getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = None
        reduce_scatter_tensor_155 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_110, 'sum', 8, '1');  cat_110 = None
        wait_tensor_379 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_155);  reduce_scatter_tensor_155 = None
        convert_element_type_1134 = torch.ops.prims.convert_element_type.default(wait_tensor_379, torch.float32);  wait_tensor_379 = None
        convert_element_type_1136 = torch.ops.prims.convert_element_type.default(wait_tensor_67, torch.float32);  wait_tensor_67 = None
        mul_350 = torch.ops.aten.mul.Tensor(convert_element_type_1134, convert_element_type_1136);  convert_element_type_1136 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_40, mul_350)
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        div_22 = torch.ops.aten.div.Tensor(mul_40, 4096)
        mul_353 = torch.ops.aten.mul.Tensor(div_22, sum_67);  div_22 = sum_67 = None
        sub_34 = torch.ops.aten.sub.Tensor(mul_350, mul_353);  mul_350 = mul_353 = None
        mul_354 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_10);  sub_34 = rsqrt_10 = None
        mul_355 = torch.ops.aten.mul.Tensor(convert_element_type_1134, mul_40);  convert_element_type_1134 = mul_40 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        convert_element_type_1137 = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        convert_element_type_1138 = torch.ops.prims.convert_element_type.default(sum_68, torch.bfloat16);  sum_68 = None
        all_reduce_22 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1138, 'sum', '1');  convert_element_type_1138 = None
        wait_tensor_380 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_22);  all_reduce_22 = None
        convert_element_type_1139 = torch.ops.prims.convert_element_type.default(wait_tensor_380, torch.float32);  wait_tensor_380 = None
        reduce_scatter_tensor_156 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1139, 'avg', 8, '0');  convert_element_type_1139 = None
        wait_tensor_381 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_156);  reduce_scatter_tensor_156 = None
        add_141 = torch.ops.aten.add.Tensor(add_138, convert_element_type_1137);  add_138 = convert_element_type_1137 = None
        all_gather_into_tensor_202 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_141, 8, '1')
        wait_tensor_382 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_202);  all_gather_into_tensor_202 = None
        split_119 = torch.ops.aten.split.Tensor(wait_tensor_382, 2);  wait_tensor_382 = None
        getitem_1129 = split_119[0]
        getitem_1130 = split_119[1]
        getitem_1131 = split_119[2]
        getitem_1132 = split_119[3]
        getitem_1133 = split_119[4]
        getitem_1134 = split_119[5]
        getitem_1135 = split_119[6]
        getitem_1136 = split_119[7];  split_119 = None
        cat_111 = torch.ops.aten.cat.default([getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136], 1);  getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = None
        view_1435 = torch.ops.aten.view.default(cat_111, [16384, 4096]);  cat_111 = None
        permute_533 = torch.ops.aten.permute.default(view_1435, [1, 0])
        wait_tensor_60 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_9);  reduce_scatter_tensor_9 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, wait_tensor_60);  wait_tensor_60 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        all_gather_into_tensor_51 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_152, 8, '0');  convert_element_type_152 = None
        wait_tensor_61 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_51);  all_gather_into_tensor_51 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [2], True);  pow_10 = None
        add_18 = torch.ops.aten.add.Scalar(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_9);  convert_element_type_153 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, wait_tensor_61)
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
        view_348 = torch.ops.aten.view.default(cat_19, [16384, 4096]);  cat_19 = None
        view_349 = torch.ops.aten.view.default(mm_32, [2, 8192, 1792]);  mm_32 = None
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_158)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_158, sigmoid_4);  sigmoid_4 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        all_gather_into_tensor_54 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_160, 8, '0');  convert_element_type_160 = None
        wait_tensor_64 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_54);  all_gather_into_tensor_54 = None
        permute_53 = torch.ops.aten.permute.default(wait_tensor_64, [1, 0]);  wait_tensor_64 = None
        mm_33 = torch.ops.aten.mm.default(view_348, permute_53)
        view_356 = torch.ops.aten.view.default(mm_33, [2, 8192, 1792]);  mm_33 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_159, view_356)
        view_363 = torch.ops.aten.view.default(mul_39, [16384, 1792]);  mul_39 = None
        mm_269 = torch.ops.aten.mm.default(permute_533, view_363);  permute_533 = view_363 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        all_gather_into_tensor_55 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_163, 8, '0');  convert_element_type_163 = None
        wait_tensor_65 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_55);  all_gather_into_tensor_55 = None
        permute_54 = torch.ops.aten.permute.default(wait_tensor_65, [1, 0]);  wait_tensor_65 = None
        permute_535 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_270 = torch.ops.aten.mm.default(view_1435, permute_535);  view_1435 = permute_535 = None
        view_1436 = torch.ops.aten.view.default(mm_270, [2, 8192, 1792]);  mm_270 = None
        convert_element_type_1144 = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        reduce_scatter_tensor_157 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1144, 'avg', 8, '0');  convert_element_type_1144 = None
        wait_tensor_383 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_157);  reduce_scatter_tensor_157 = None
        mul_356 = torch.ops.aten.mul.Tensor(view_1436, convert_element_type_159);  convert_element_type_159 = None
        mul_357 = torch.ops.aten.mul.Tensor(view_1436, view_356);  view_1436 = view_356 = None
        view_1437 = torch.ops.aten.view.default(mul_356, [16384, 1792]);  mul_356 = None
        permute_537 = torch.ops.aten.permute.default(view_1437, [1, 0])
        mm_271 = torch.ops.aten.mm.default(permute_537, view_348);  permute_537 = None
        permute_539 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_272 = torch.ops.aten.mm.default(view_1437, permute_539);  view_1437 = permute_539 = None
        view_1438 = torch.ops.aten.view.default(mm_272, [2, 8192, 4096]);  mm_272 = None
        convert_element_type_1149 = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        reduce_scatter_tensor_158 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1149, 'avg', 8, '0');  convert_element_type_1149 = None
        wait_tensor_384 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_158);  reduce_scatter_tensor_158 = None
        convert_element_type_1150 = torch.ops.prims.convert_element_type.default(mul_357, torch.float32);  mul_357 = None
        neg_11 = torch.ops.aten.neg.default(convert_element_type_158)
        exp_11 = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_142 = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_142);  add_142 = None
        mul_358 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_359 = torch.ops.aten.mul.Tensor(convert_element_type_1150, mul_358);  convert_element_type_1150 = None
        sub_35 = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
        mul_360 = torch.ops.aten.mul.Tensor(convert_element_type_158, sub_35);  convert_element_type_158 = sub_35 = None
        add_143 = torch.ops.aten.add.Tensor(mul_360, 1);  mul_360 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_359, add_143);  mul_359 = add_143 = None
        convert_element_type_1152 = torch.ops.prims.convert_element_type.default(mul_361, torch.bfloat16);  mul_361 = None
        view_1439 = torch.ops.aten.view.default(convert_element_type_1152, [16384, 1792]);  convert_element_type_1152 = None
        permute_541 = torch.ops.aten.permute.default(view_1439, [1, 0])
        mm_273 = torch.ops.aten.mm.default(permute_541, view_348);  permute_541 = view_348 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        all_gather_into_tensor_53 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_155, 8, '0');  convert_element_type_155 = None
        wait_tensor_63 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_53);  all_gather_into_tensor_53 = None
        permute_52 = torch.ops.aten.permute.default(wait_tensor_63, [1, 0]);  wait_tensor_63 = None
        permute_543 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_274 = torch.ops.aten.mm.default(view_1439, permute_543);  view_1439 = permute_543 = None
        view_1440 = torch.ops.aten.view.default(mm_274, [2, 8192, 4096]);  mm_274 = None
        add_144 = torch.ops.aten.add.Tensor(view_1438, view_1440);  view_1438 = view_1440 = None
        convert_element_type_1157 = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        reduce_scatter_tensor_159 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1157, 'avg', 8, '0');  convert_element_type_1157 = None
        wait_tensor_385 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_159);  reduce_scatter_tensor_159 = None
        split_120 = torch.ops.aten.split.Tensor(add_144, 1024, 1);  add_144 = None
        getitem_1137 = split_120[0]
        getitem_1138 = split_120[1]
        getitem_1139 = split_120[2]
        getitem_1140 = split_120[3]
        getitem_1141 = split_120[4]
        getitem_1142 = split_120[5]
        getitem_1143 = split_120[6]
        getitem_1144 = split_120[7];  split_120 = None
        cat_112 = torch.ops.aten.cat.default([getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144]);  getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = None
        reduce_scatter_tensor_160 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_112, 'sum', 8, '1');  cat_112 = None
        wait_tensor_386 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_160);  reduce_scatter_tensor_160 = None
        convert_element_type_1158 = torch.ops.prims.convert_element_type.default(wait_tensor_386, torch.float32);  wait_tensor_386 = None
        convert_element_type_1160 = torch.ops.prims.convert_element_type.default(wait_tensor_61, torch.float32);  wait_tensor_61 = None
        mul_362 = torch.ops.aten.mul.Tensor(convert_element_type_1158, convert_element_type_1160);  convert_element_type_1160 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_36, mul_362)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_364, [2], True);  mul_364 = None
        div_23 = torch.ops.aten.div.Tensor(mul_36, 4096)
        mul_365 = torch.ops.aten.mul.Tensor(div_23, sum_69);  div_23 = sum_69 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_362, mul_365);  mul_362 = mul_365 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_36, rsqrt_9);  sub_36 = rsqrt_9 = None
        mul_367 = torch.ops.aten.mul.Tensor(convert_element_type_1158, mul_36);  convert_element_type_1158 = mul_36 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1]);  mul_367 = None
        convert_element_type_1161 = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None
        convert_element_type_1162 = torch.ops.prims.convert_element_type.default(sum_70, torch.bfloat16);  sum_70 = None
        all_reduce_23 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1162, 'sum', '1');  convert_element_type_1162 = None
        wait_tensor_387 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_23);  all_reduce_23 = None
        convert_element_type_1163 = torch.ops.prims.convert_element_type.default(wait_tensor_387, torch.float32);  wait_tensor_387 = None
        reduce_scatter_tensor_161 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1163, 'avg', 8, '0');  convert_element_type_1163 = None
        wait_tensor_388 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_161);  reduce_scatter_tensor_161 = None
        add_145 = torch.ops.aten.add.Tensor(add_141, convert_element_type_1161);  add_141 = convert_element_type_1161 = None
        all_gather_into_tensor_203 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_145, 8, '1')
        wait_tensor_389 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_203);  all_gather_into_tensor_203 = None
        split_121 = torch.ops.aten.split.Tensor(wait_tensor_389, 2);  wait_tensor_389 = None
        getitem_1145 = split_121[0]
        getitem_1146 = split_121[1]
        getitem_1147 = split_121[2]
        getitem_1148 = split_121[3]
        getitem_1149 = split_121[4]
        getitem_1150 = split_121[5]
        getitem_1151 = split_121[6]
        getitem_1152 = split_121[7];  split_121 = None
        cat_113 = torch.ops.aten.cat.default([getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152], 1);  getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = None
        view_1441 = torch.ops.aten.view.default(cat_113, [16384, 4096]);  cat_113 = None
        permute_545 = torch.ops.aten.permute.default(view_1441, [1, 0])
        permute_50 = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3])
        view_330 = torch.ops.aten.view.default(permute_50, [2, 8192, -1]);  permute_50 = None
        view_336 = torch.ops.aten.view.default(view_330, [16384, 512]);  view_330 = None
        mm_275 = torch.ops.aten.mm.default(permute_545, view_336);  permute_545 = view_336 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        all_gather_into_tensor_50 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_149, 8, '0');  convert_element_type_149 = None
        wait_tensor_59 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_50);  all_gather_into_tensor_50 = None
        permute_51 = torch.ops.aten.permute.default(wait_tensor_59, [1, 0]);  wait_tensor_59 = None
        permute_547 = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        mm_276 = torch.ops.aten.mm.default(view_1441, permute_547);  view_1441 = permute_547 = None
        view_1442 = torch.ops.aten.view.default(mm_276, [2, 8192, 512]);  mm_276 = None
        convert_element_type_1168 = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None
        reduce_scatter_tensor_162 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1168, 'avg', 8, '0');  convert_element_type_1168 = None
        wait_tensor_390 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_162);  reduce_scatter_tensor_162 = None
        view_1443 = torch.ops.aten.view.default(view_1442, [2, 8192, 4, 128]);  view_1442 = None
        permute_549 = torch.ops.aten.permute.default(view_1443, [0, 2, 1, 3]);  view_1443 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        all_gather_into_tensor_45 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_133, 8, '0');  convert_element_type_133 = None
        wait_tensor_54 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_45);  all_gather_into_tensor_45 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [2], True);  pow_9 = None
        add_16 = torch.ops.aten.add.Scalar(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_134, rsqrt_8);  convert_element_type_134 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, wait_tensor_54)
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
        view_303 = torch.ops.aten.view.default(cat_17, [16384, 4096]);  cat_17 = None
        view_304 = torch.ops.aten.view.default(mm_28, [2, 8192, 512]);  mm_28 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        all_gather_into_tensor_48 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_139, 8, '0');  convert_element_type_139 = None
        wait_tensor_57 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_48);  all_gather_into_tensor_48 = None
        permute_45 = torch.ops.aten.permute.default(wait_tensor_57, [1, 0]);  wait_tensor_57 = None
        mm_29 = torch.ops.aten.mm.default(view_303, permute_45)
        view_311 = torch.ops.aten.view.default(mm_29, [2, 8192, 128]);  mm_29 = None
        view_318 = torch.ops.aten.view.default(mm_30, [2, 8192, 128]);  mm_30 = None
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
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_549, permute_47, permute_48, permute_49, getitem_244, getitem_245, getitem_250, getitem_251, None, None, None, 8192, 8192, 0.0, True);  permute_549 = permute_47 = permute_48 = permute_49 = getitem_244 = getitem_245 = getitem_250 = getitem_251 = None
        getitem_1153 = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_1154 = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_1155 = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None
        permute_550 = torch.ops.aten.permute.default(getitem_1155, [0, 2, 1, 3]);  getitem_1155 = None
        permute_551 = torch.ops.aten.permute.default(getitem_1154, [0, 2, 1, 3]);  getitem_1154 = None
        permute_552 = torch.ops.aten.permute.default(getitem_1153, [0, 2, 1, 3]);  getitem_1153 = None
        view_1444 = torch.ops.aten.view.default(permute_550, [2, 8192, 1, 4, 128]);  permute_550 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(view_1444, [3], True);  view_1444 = None
        squeeze_22 = torch.ops.aten.squeeze.dim(sum_71, 3);  sum_71 = None
        view_1445 = torch.ops.aten.view.default(permute_551, [2, 8192, 1, 4, 128]);  permute_551 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(view_1445, [3], True);  view_1445 = None
        squeeze_23 = torch.ops.aten.squeeze.dim(sum_72, 3);  sum_72 = None
        convert_element_type_1169 = torch.ops.prims.convert_element_type.default(squeeze_23, torch.float32);  squeeze_23 = None
        convert_element_type_1170 = torch.ops.prims.convert_element_type.default(permute_552, torch.float32);  permute_552 = None
        view_1446 = torch.ops.aten.view.default(convert_element_type_1169, [2, 8192, 1, 64, 2]);  convert_element_type_1169 = None
        view_as_complex_54 = torch.ops.aten.view_as_complex.default(view_1446);  view_1446 = None
        mul_368 = torch.ops.aten.mul.Tensor(view_as_complex_54, _conj);  view_as_complex_54 = None
        view_1447 = torch.ops.aten.view.default(convert_element_type_1170, [2, 8192, 4, 64, 2]);  convert_element_type_1170 = None
        view_as_complex_55 = torch.ops.aten.view_as_complex.default(view_1447);  view_1447 = None
        mul_369 = torch.ops.aten.mul.Tensor(view_as_complex_55, _conj);  view_as_complex_55 = None
        view_as_real_54 = torch.ops.aten.view_as_real.default(mul_368);  mul_368 = None
        view_1448 = torch.ops.aten.view.default(view_as_real_54, [2, 8192, 1, 128]);  view_as_real_54 = None
        convert_element_type_1171 = torch.ops.prims.convert_element_type.default(view_1448, torch.bfloat16);  view_1448 = None
        view_as_real_55 = torch.ops.aten.view_as_real.default(mul_369);  mul_369 = None
        view_1449 = torch.ops.aten.view.default(view_as_real_55, [2, 8192, 4, 128]);  view_as_real_55 = None
        convert_element_type_1172 = torch.ops.prims.convert_element_type.default(view_1449, torch.bfloat16);  view_1449 = None
        view_1450 = torch.ops.aten.view.default(squeeze_22, [2, 8192, 128]);  squeeze_22 = None
        view_1451 = torch.ops.aten.view.default(convert_element_type_1171, [2, 8192, 128]);  convert_element_type_1171 = None
        view_1452 = torch.ops.aten.view.default(convert_element_type_1172, [2, 8192, 512]);  convert_element_type_1172 = None
        view_1453 = torch.ops.aten.view.default(view_1450, [16384, 128]);  view_1450 = None
        permute_553 = torch.ops.aten.permute.default(view_1453, [1, 0])
        mm_277 = torch.ops.aten.mm.default(permute_553, view_303);  permute_553 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        all_gather_into_tensor_49 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_142, 8, '0');  convert_element_type_142 = None
        wait_tensor_58 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_49);  all_gather_into_tensor_49 = None
        permute_46 = torch.ops.aten.permute.default(wait_tensor_58, [1, 0]);  wait_tensor_58 = None
        permute_555 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_278 = torch.ops.aten.mm.default(view_1453, permute_555);  view_1453 = permute_555 = None
        view_1454 = torch.ops.aten.view.default(mm_278, [2, 8192, 4096]);  mm_278 = None
        convert_element_type_1177 = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        reduce_scatter_tensor_163 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1177, 'avg', 8, '0');  convert_element_type_1177 = None
        wait_tensor_391 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_163);  reduce_scatter_tensor_163 = None
        view_1455 = torch.ops.aten.view.default(view_1451, [16384, 128]);  view_1451 = None
        permute_557 = torch.ops.aten.permute.default(view_1455, [1, 0])
        mm_279 = torch.ops.aten.mm.default(permute_557, view_303);  permute_557 = None
        permute_559 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_280 = torch.ops.aten.mm.default(view_1455, permute_559);  view_1455 = permute_559 = None
        view_1456 = torch.ops.aten.view.default(mm_280, [2, 8192, 4096]);  mm_280 = None
        add_146 = torch.ops.aten.add.Tensor(view_1454, view_1456);  view_1454 = view_1456 = None
        convert_element_type_1182 = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        reduce_scatter_tensor_164 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1182, 'avg', 8, '0');  convert_element_type_1182 = None
        wait_tensor_392 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_164);  reduce_scatter_tensor_164 = None
        view_1457 = torch.ops.aten.view.default(view_1452, [16384, 512]);  view_1452 = None
        permute_561 = torch.ops.aten.permute.default(view_1457, [1, 0])
        mm_281 = torch.ops.aten.mm.default(permute_561, view_303);  permute_561 = view_303 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        all_gather_into_tensor_47 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_136, 8, '0');  convert_element_type_136 = None
        wait_tensor_56 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_47);  all_gather_into_tensor_47 = None
        permute_44 = torch.ops.aten.permute.default(wait_tensor_56, [1, 0]);  wait_tensor_56 = None
        permute_563 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_282 = torch.ops.aten.mm.default(view_1457, permute_563);  view_1457 = permute_563 = None
        view_1458 = torch.ops.aten.view.default(mm_282, [2, 8192, 4096]);  mm_282 = None
        add_147 = torch.ops.aten.add.Tensor(add_146, view_1458);  add_146 = view_1458 = None
        convert_element_type_1187 = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        reduce_scatter_tensor_165 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1187, 'avg', 8, '0');  convert_element_type_1187 = None
        wait_tensor_393 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_165);  reduce_scatter_tensor_165 = None
        split_122 = torch.ops.aten.split.Tensor(add_147, 1024, 1);  add_147 = None
        getitem_1156 = split_122[0]
        getitem_1157 = split_122[1]
        getitem_1158 = split_122[2]
        getitem_1159 = split_122[3]
        getitem_1160 = split_122[4]
        getitem_1161 = split_122[5]
        getitem_1162 = split_122[6]
        getitem_1163 = split_122[7];  split_122 = None
        cat_114 = torch.ops.aten.cat.default([getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163]);  getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = None
        reduce_scatter_tensor_166 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_114, 'sum', 8, '1');  cat_114 = None
        wait_tensor_394 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_166);  reduce_scatter_tensor_166 = None
        convert_element_type_1188 = torch.ops.prims.convert_element_type.default(wait_tensor_394, torch.float32);  wait_tensor_394 = None
        convert_element_type_1190 = torch.ops.prims.convert_element_type.default(wait_tensor_54, torch.float32);  wait_tensor_54 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_1188, convert_element_type_1190);  convert_element_type_1190 = None
        mul_372 = torch.ops.aten.mul.Tensor(mul_32, mul_370)
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        div_24 = torch.ops.aten.div.Tensor(mul_32, 4096)
        mul_373 = torch.ops.aten.mul.Tensor(div_24, sum_73);  div_24 = sum_73 = None
        sub_37 = torch.ops.aten.sub.Tensor(mul_370, mul_373);  mul_370 = mul_373 = None
        mul_374 = torch.ops.aten.mul.Tensor(sub_37, rsqrt_8);  sub_37 = rsqrt_8 = None
        mul_375 = torch.ops.aten.mul.Tensor(convert_element_type_1188, mul_32);  convert_element_type_1188 = mul_32 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        convert_element_type_1191 = torch.ops.prims.convert_element_type.default(mul_374, torch.bfloat16);  mul_374 = None
        convert_element_type_1192 = torch.ops.prims.convert_element_type.default(sum_74, torch.bfloat16);  sum_74 = None
        all_reduce_24 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1192, 'sum', '1');  convert_element_type_1192 = None
        wait_tensor_395 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_24);  all_reduce_24 = None
        convert_element_type_1193 = torch.ops.prims.convert_element_type.default(wait_tensor_395, torch.float32);  wait_tensor_395 = None
        reduce_scatter_tensor_167 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1193, 'avg', 8, '0');  convert_element_type_1193 = None
        wait_tensor_396 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_167);  reduce_scatter_tensor_167 = None
        add_148 = torch.ops.aten.add.Tensor(add_145, convert_element_type_1191);  add_145 = convert_element_type_1191 = None
        all_gather_into_tensor_204 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_148, 8, '1')
        wait_tensor_397 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_204);  all_gather_into_tensor_204 = None
        split_123 = torch.ops.aten.split.Tensor(wait_tensor_397, 2);  wait_tensor_397 = None
        getitem_1164 = split_123[0]
        getitem_1165 = split_123[1]
        getitem_1166 = split_123[2]
        getitem_1167 = split_123[3]
        getitem_1168 = split_123[4]
        getitem_1169 = split_123[5]
        getitem_1170 = split_123[6]
        getitem_1171 = split_123[7];  split_123 = None
        cat_115 = torch.ops.aten.cat.default([getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171], 1);  getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = None
        view_1459 = torch.ops.aten.view.default(cat_115, [16384, 4096]);  cat_115 = None
        permute_565 = torch.ops.aten.permute.default(view_1459, [1, 0])
        wait_tensor_47 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7);  reduce_scatter_tensor_7 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, wait_tensor_47);  wait_tensor_47 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        all_gather_into_tensor_40 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_119, 8, '0');  convert_element_type_119 = None
        wait_tensor_48 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_40);  all_gather_into_tensor_40 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_120, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [2], True);  pow_8 = None
        add_14 = torch.ops.aten.add.Scalar(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_120, rsqrt_7);  convert_element_type_120 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, wait_tensor_48)
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
        view_276 = torch.ops.aten.view.default(cat_15, [16384, 4096]);  cat_15 = None
        view_277 = torch.ops.aten.view.default(mm_25, [2, 8192, 1792]);  mm_25 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_125, sigmoid_3);  sigmoid_3 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        all_gather_into_tensor_43 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_127, 8, '0');  convert_element_type_127 = None
        wait_tensor_51 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_43);  all_gather_into_tensor_43 = None
        permute_42 = torch.ops.aten.permute.default(wait_tensor_51, [1, 0]);  wait_tensor_51 = None
        mm_26 = torch.ops.aten.mm.default(view_276, permute_42)
        view_284 = torch.ops.aten.view.default(mm_26, [2, 8192, 1792]);  mm_26 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_126, view_284)
        view_291 = torch.ops.aten.view.default(mul_31, [16384, 1792]);  mul_31 = None
        mm_283 = torch.ops.aten.mm.default(permute_565, view_291);  permute_565 = view_291 = None
        convert_element_type_130 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        all_gather_into_tensor_44 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_130, 8, '0');  convert_element_type_130 = None
        wait_tensor_52 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_44);  all_gather_into_tensor_44 = None
        permute_43 = torch.ops.aten.permute.default(wait_tensor_52, [1, 0]);  wait_tensor_52 = None
        permute_567 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_284 = torch.ops.aten.mm.default(view_1459, permute_567);  view_1459 = permute_567 = None
        view_1460 = torch.ops.aten.view.default(mm_284, [2, 8192, 1792]);  mm_284 = None
        convert_element_type_1198 = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        reduce_scatter_tensor_168 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1198, 'avg', 8, '0');  convert_element_type_1198 = None
        wait_tensor_398 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_168);  reduce_scatter_tensor_168 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_1460, convert_element_type_126);  convert_element_type_126 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_1460, view_284);  view_1460 = view_284 = None
        view_1461 = torch.ops.aten.view.default(mul_376, [16384, 1792]);  mul_376 = None
        permute_569 = torch.ops.aten.permute.default(view_1461, [1, 0])
        mm_285 = torch.ops.aten.mm.default(permute_569, view_276);  permute_569 = None
        permute_571 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_286 = torch.ops.aten.mm.default(view_1461, permute_571);  view_1461 = permute_571 = None
        view_1462 = torch.ops.aten.view.default(mm_286, [2, 8192, 4096]);  mm_286 = None
        convert_element_type_1203 = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        reduce_scatter_tensor_169 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1203, 'avg', 8, '0');  convert_element_type_1203 = None
        wait_tensor_399 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_169);  reduce_scatter_tensor_169 = None
        convert_element_type_1204 = torch.ops.prims.convert_element_type.default(mul_377, torch.float32);  mul_377 = None
        neg_12 = torch.ops.aten.neg.default(convert_element_type_125)
        exp_12 = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_149 = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(add_149);  add_149 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_379 = torch.ops.aten.mul.Tensor(convert_element_type_1204, mul_378);  convert_element_type_1204 = None
        sub_38 = torch.ops.aten.sub.Tensor(1, mul_378);  mul_378 = None
        mul_380 = torch.ops.aten.mul.Tensor(convert_element_type_125, sub_38);  convert_element_type_125 = sub_38 = None
        add_150 = torch.ops.aten.add.Tensor(mul_380, 1);  mul_380 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_379, add_150);  mul_379 = add_150 = None
        convert_element_type_1206 = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16);  mul_381 = None
        view_1463 = torch.ops.aten.view.default(convert_element_type_1206, [16384, 1792]);  convert_element_type_1206 = None
        permute_573 = torch.ops.aten.permute.default(view_1463, [1, 0])
        mm_287 = torch.ops.aten.mm.default(permute_573, view_276);  permute_573 = view_276 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        all_gather_into_tensor_42 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_122, 8, '0');  convert_element_type_122 = None
        wait_tensor_50 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_42);  all_gather_into_tensor_42 = None
        permute_41 = torch.ops.aten.permute.default(wait_tensor_50, [1, 0]);  wait_tensor_50 = None
        permute_575 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_288 = torch.ops.aten.mm.default(view_1463, permute_575);  view_1463 = permute_575 = None
        view_1464 = torch.ops.aten.view.default(mm_288, [2, 8192, 4096]);  mm_288 = None
        add_151 = torch.ops.aten.add.Tensor(view_1462, view_1464);  view_1462 = view_1464 = None
        convert_element_type_1211 = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None
        reduce_scatter_tensor_170 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1211, 'avg', 8, '0');  convert_element_type_1211 = None
        wait_tensor_400 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_170);  reduce_scatter_tensor_170 = None
        split_124 = torch.ops.aten.split.Tensor(add_151, 1024, 1);  add_151 = None
        getitem_1172 = split_124[0]
        getitem_1173 = split_124[1]
        getitem_1174 = split_124[2]
        getitem_1175 = split_124[3]
        getitem_1176 = split_124[4]
        getitem_1177 = split_124[5]
        getitem_1178 = split_124[6]
        getitem_1179 = split_124[7];  split_124 = None
        cat_116 = torch.ops.aten.cat.default([getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179]);  getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = getitem_1179 = None
        reduce_scatter_tensor_171 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_116, 'sum', 8, '1');  cat_116 = None
        wait_tensor_401 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_171);  reduce_scatter_tensor_171 = None
        convert_element_type_1212 = torch.ops.prims.convert_element_type.default(wait_tensor_401, torch.float32);  wait_tensor_401 = None
        convert_element_type_1214 = torch.ops.prims.convert_element_type.default(wait_tensor_48, torch.float32);  wait_tensor_48 = None
        mul_382 = torch.ops.aten.mul.Tensor(convert_element_type_1212, convert_element_type_1214);  convert_element_type_1214 = None
        mul_384 = torch.ops.aten.mul.Tensor(mul_28, mul_382)
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
        div_25 = torch.ops.aten.div.Tensor(mul_28, 4096)
        mul_385 = torch.ops.aten.mul.Tensor(div_25, sum_75);  div_25 = sum_75 = None
        sub_39 = torch.ops.aten.sub.Tensor(mul_382, mul_385);  mul_382 = mul_385 = None
        mul_386 = torch.ops.aten.mul.Tensor(sub_39, rsqrt_7);  sub_39 = rsqrt_7 = None
        mul_387 = torch.ops.aten.mul.Tensor(convert_element_type_1212, mul_28);  convert_element_type_1212 = mul_28 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        convert_element_type_1215 = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None
        convert_element_type_1216 = torch.ops.prims.convert_element_type.default(sum_76, torch.bfloat16);  sum_76 = None
        all_reduce_25 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1216, 'sum', '1');  convert_element_type_1216 = None
        wait_tensor_402 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_25);  all_reduce_25 = None
        convert_element_type_1217 = torch.ops.prims.convert_element_type.default(wait_tensor_402, torch.float32);  wait_tensor_402 = None
        reduce_scatter_tensor_172 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1217, 'avg', 8, '0');  convert_element_type_1217 = None
        wait_tensor_403 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_172);  reduce_scatter_tensor_172 = None
        add_152 = torch.ops.aten.add.Tensor(add_148, convert_element_type_1215);  add_148 = convert_element_type_1215 = None
        all_gather_into_tensor_205 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_152, 8, '1')
        wait_tensor_404 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_205);  all_gather_into_tensor_205 = None
        split_125 = torch.ops.aten.split.Tensor(wait_tensor_404, 2);  wait_tensor_404 = None
        getitem_1180 = split_125[0]
        getitem_1181 = split_125[1]
        getitem_1182 = split_125[2]
        getitem_1183 = split_125[3]
        getitem_1184 = split_125[4]
        getitem_1185 = split_125[5]
        getitem_1186 = split_125[6]
        getitem_1187 = split_125[7];  split_125 = None
        cat_117 = torch.ops.aten.cat.default([getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187], 1);  getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = None
        view_1465 = torch.ops.aten.view.default(cat_117, [16384, 4096]);  cat_117 = None
        permute_577 = torch.ops.aten.permute.default(view_1465, [1, 0])
        permute_39 = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3])
        view_258 = torch.ops.aten.view.default(permute_39, [2, 8192, -1]);  permute_39 = None
        view_264 = torch.ops.aten.view.default(view_258, [16384, 512]);  view_258 = None
        mm_289 = torch.ops.aten.mm.default(permute_577, view_264);  permute_577 = view_264 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        all_gather_into_tensor_39 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_116, 8, '0');  convert_element_type_116 = None
        wait_tensor_46 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_39);  all_gather_into_tensor_39 = None
        permute_40 = torch.ops.aten.permute.default(wait_tensor_46, [1, 0]);  wait_tensor_46 = None
        permute_579 = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_290 = torch.ops.aten.mm.default(view_1465, permute_579);  view_1465 = permute_579 = None
        view_1466 = torch.ops.aten.view.default(mm_290, [2, 8192, 512]);  mm_290 = None
        convert_element_type_1222 = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None
        reduce_scatter_tensor_173 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1222, 'avg', 8, '0');  convert_element_type_1222 = None
        wait_tensor_405 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_173);  reduce_scatter_tensor_173 = None
        view_1467 = torch.ops.aten.view.default(view_1466, [2, 8192, 4, 128]);  view_1466 = None
        permute_581 = torch.ops.aten.permute.default(view_1467, [0, 2, 1, 3]);  view_1467 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        all_gather_into_tensor_34 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_100, 8, '0');  convert_element_type_100 = None
        wait_tensor_41 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_34);  all_gather_into_tensor_34 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [2], True);  pow_7 = None
        add_12 = torch.ops.aten.add.Scalar(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_101, rsqrt_6);  convert_element_type_101 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, wait_tensor_41)
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
        view_231 = torch.ops.aten.view.default(cat_13, [16384, 4096]);  cat_13 = None
        view_232 = torch.ops.aten.view.default(mm_21, [2, 8192, 512]);  mm_21 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        all_gather_into_tensor_37 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_106, 8, '0');  convert_element_type_106 = None
        wait_tensor_44 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_37);  all_gather_into_tensor_37 = None
        permute_34 = torch.ops.aten.permute.default(wait_tensor_44, [1, 0]);  wait_tensor_44 = None
        mm_22 = torch.ops.aten.mm.default(view_231, permute_34)
        view_239 = torch.ops.aten.view.default(mm_22, [2, 8192, 128]);  mm_22 = None
        view_246 = torch.ops.aten.view.default(mm_23, [2, 8192, 128]);  mm_23 = None
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
        _scaled_dot_product_cudnn_attention_backward_12 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_581, permute_36, permute_37, permute_38, getitem_203, getitem_204, getitem_209, getitem_210, None, None, None, 8192, 8192, 0.0, True);  permute_581 = permute_36 = permute_37 = permute_38 = getitem_203 = getitem_204 = getitem_209 = getitem_210 = None
        getitem_1188 = _scaled_dot_product_cudnn_attention_backward_12[0]
        getitem_1189 = _scaled_dot_product_cudnn_attention_backward_12[1]
        getitem_1190 = _scaled_dot_product_cudnn_attention_backward_12[2];  _scaled_dot_product_cudnn_attention_backward_12 = None
        permute_582 = torch.ops.aten.permute.default(getitem_1190, [0, 2, 1, 3]);  getitem_1190 = None
        permute_583 = torch.ops.aten.permute.default(getitem_1189, [0, 2, 1, 3]);  getitem_1189 = None
        permute_584 = torch.ops.aten.permute.default(getitem_1188, [0, 2, 1, 3]);  getitem_1188 = None
        view_1468 = torch.ops.aten.view.default(permute_582, [2, 8192, 1, 4, 128]);  permute_582 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(view_1468, [3], True);  view_1468 = None
        squeeze_24 = torch.ops.aten.squeeze.dim(sum_77, 3);  sum_77 = None
        view_1469 = torch.ops.aten.view.default(permute_583, [2, 8192, 1, 4, 128]);  permute_583 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(view_1469, [3], True);  view_1469 = None
        squeeze_25 = torch.ops.aten.squeeze.dim(sum_78, 3);  sum_78 = None
        convert_element_type_1223 = torch.ops.prims.convert_element_type.default(squeeze_25, torch.float32);  squeeze_25 = None
        convert_element_type_1224 = torch.ops.prims.convert_element_type.default(permute_584, torch.float32);  permute_584 = None
        view_1470 = torch.ops.aten.view.default(convert_element_type_1223, [2, 8192, 1, 64, 2]);  convert_element_type_1223 = None
        view_as_complex_56 = torch.ops.aten.view_as_complex.default(view_1470);  view_1470 = None
        mul_388 = torch.ops.aten.mul.Tensor(view_as_complex_56, _conj);  view_as_complex_56 = None
        view_1471 = torch.ops.aten.view.default(convert_element_type_1224, [2, 8192, 4, 64, 2]);  convert_element_type_1224 = None
        view_as_complex_57 = torch.ops.aten.view_as_complex.default(view_1471);  view_1471 = None
        mul_389 = torch.ops.aten.mul.Tensor(view_as_complex_57, _conj);  view_as_complex_57 = None
        view_as_real_56 = torch.ops.aten.view_as_real.default(mul_388);  mul_388 = None
        view_1472 = torch.ops.aten.view.default(view_as_real_56, [2, 8192, 1, 128]);  view_as_real_56 = None
        convert_element_type_1225 = torch.ops.prims.convert_element_type.default(view_1472, torch.bfloat16);  view_1472 = None
        view_as_real_57 = torch.ops.aten.view_as_real.default(mul_389);  mul_389 = None
        view_1473 = torch.ops.aten.view.default(view_as_real_57, [2, 8192, 4, 128]);  view_as_real_57 = None
        convert_element_type_1226 = torch.ops.prims.convert_element_type.default(view_1473, torch.bfloat16);  view_1473 = None
        view_1474 = torch.ops.aten.view.default(squeeze_24, [2, 8192, 128]);  squeeze_24 = None
        view_1475 = torch.ops.aten.view.default(convert_element_type_1225, [2, 8192, 128]);  convert_element_type_1225 = None
        view_1476 = torch.ops.aten.view.default(convert_element_type_1226, [2, 8192, 512]);  convert_element_type_1226 = None
        view_1477 = torch.ops.aten.view.default(view_1474, [16384, 128]);  view_1474 = None
        permute_585 = torch.ops.aten.permute.default(view_1477, [1, 0])
        mm_291 = torch.ops.aten.mm.default(permute_585, view_231);  permute_585 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        all_gather_into_tensor_38 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_109, 8, '0');  convert_element_type_109 = None
        wait_tensor_45 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_38);  all_gather_into_tensor_38 = None
        permute_35 = torch.ops.aten.permute.default(wait_tensor_45, [1, 0]);  wait_tensor_45 = None
        permute_587 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_292 = torch.ops.aten.mm.default(view_1477, permute_587);  view_1477 = permute_587 = None
        view_1478 = torch.ops.aten.view.default(mm_292, [2, 8192, 4096]);  mm_292 = None
        convert_element_type_1231 = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        reduce_scatter_tensor_174 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1231, 'avg', 8, '0');  convert_element_type_1231 = None
        wait_tensor_406 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_174);  reduce_scatter_tensor_174 = None
        view_1479 = torch.ops.aten.view.default(view_1475, [16384, 128]);  view_1475 = None
        permute_589 = torch.ops.aten.permute.default(view_1479, [1, 0])
        mm_293 = torch.ops.aten.mm.default(permute_589, view_231);  permute_589 = None
        permute_591 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_294 = torch.ops.aten.mm.default(view_1479, permute_591);  view_1479 = permute_591 = None
        view_1480 = torch.ops.aten.view.default(mm_294, [2, 8192, 4096]);  mm_294 = None
        add_153 = torch.ops.aten.add.Tensor(view_1478, view_1480);  view_1478 = view_1480 = None
        convert_element_type_1236 = torch.ops.prims.convert_element_type.default(mm_293, torch.float32);  mm_293 = None
        reduce_scatter_tensor_175 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1236, 'avg', 8, '0');  convert_element_type_1236 = None
        wait_tensor_407 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_175);  reduce_scatter_tensor_175 = None
        view_1481 = torch.ops.aten.view.default(view_1476, [16384, 512]);  view_1476 = None
        permute_593 = torch.ops.aten.permute.default(view_1481, [1, 0])
        mm_295 = torch.ops.aten.mm.default(permute_593, view_231);  permute_593 = view_231 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        all_gather_into_tensor_36 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_103, 8, '0');  convert_element_type_103 = None
        wait_tensor_43 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_36);  all_gather_into_tensor_36 = None
        permute_33 = torch.ops.aten.permute.default(wait_tensor_43, [1, 0]);  wait_tensor_43 = None
        permute_595 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_296 = torch.ops.aten.mm.default(view_1481, permute_595);  view_1481 = permute_595 = None
        view_1482 = torch.ops.aten.view.default(mm_296, [2, 8192, 4096]);  mm_296 = None
        add_154 = torch.ops.aten.add.Tensor(add_153, view_1482);  add_153 = view_1482 = None
        convert_element_type_1241 = torch.ops.prims.convert_element_type.default(mm_295, torch.float32);  mm_295 = None
        reduce_scatter_tensor_176 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1241, 'avg', 8, '0');  convert_element_type_1241 = None
        wait_tensor_408 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_176);  reduce_scatter_tensor_176 = None
        split_126 = torch.ops.aten.split.Tensor(add_154, 1024, 1);  add_154 = None
        getitem_1191 = split_126[0]
        getitem_1192 = split_126[1]
        getitem_1193 = split_126[2]
        getitem_1194 = split_126[3]
        getitem_1195 = split_126[4]
        getitem_1196 = split_126[5]
        getitem_1197 = split_126[6]
        getitem_1198 = split_126[7];  split_126 = None
        cat_118 = torch.ops.aten.cat.default([getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198]);  getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = None
        reduce_scatter_tensor_177 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_118, 'sum', 8, '1');  cat_118 = None
        wait_tensor_409 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_177);  reduce_scatter_tensor_177 = None
        convert_element_type_1242 = torch.ops.prims.convert_element_type.default(wait_tensor_409, torch.float32);  wait_tensor_409 = None
        convert_element_type_1244 = torch.ops.prims.convert_element_type.default(wait_tensor_41, torch.float32);  wait_tensor_41 = None
        mul_390 = torch.ops.aten.mul.Tensor(convert_element_type_1242, convert_element_type_1244);  convert_element_type_1244 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_24, mul_390)
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        div_26 = torch.ops.aten.div.Tensor(mul_24, 4096)
        mul_393 = torch.ops.aten.mul.Tensor(div_26, sum_79);  div_26 = sum_79 = None
        sub_40 = torch.ops.aten.sub.Tensor(mul_390, mul_393);  mul_390 = mul_393 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_40, rsqrt_6);  sub_40 = rsqrt_6 = None
        mul_395 = torch.ops.aten.mul.Tensor(convert_element_type_1242, mul_24);  convert_element_type_1242 = mul_24 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        convert_element_type_1245 = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None
        convert_element_type_1246 = torch.ops.prims.convert_element_type.default(sum_80, torch.bfloat16);  sum_80 = None
        all_reduce_26 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1246, 'sum', '1');  convert_element_type_1246 = None
        wait_tensor_410 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_26);  all_reduce_26 = None
        convert_element_type_1247 = torch.ops.prims.convert_element_type.default(wait_tensor_410, torch.float32);  wait_tensor_410 = None
        reduce_scatter_tensor_178 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1247, 'avg', 8, '0');  convert_element_type_1247 = None
        wait_tensor_411 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_178);  reduce_scatter_tensor_178 = None
        add_155 = torch.ops.aten.add.Tensor(add_152, convert_element_type_1245);  add_152 = convert_element_type_1245 = None
        all_gather_into_tensor_206 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_155, 8, '1')
        wait_tensor_412 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_206);  all_gather_into_tensor_206 = None
        split_127 = torch.ops.aten.split.Tensor(wait_tensor_412, 2);  wait_tensor_412 = None
        getitem_1199 = split_127[0]
        getitem_1200 = split_127[1]
        getitem_1201 = split_127[2]
        getitem_1202 = split_127[3]
        getitem_1203 = split_127[4]
        getitem_1204 = split_127[5]
        getitem_1205 = split_127[6]
        getitem_1206 = split_127[7];  split_127 = None
        cat_119 = torch.ops.aten.cat.default([getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206], 1);  getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = None
        view_1483 = torch.ops.aten.view.default(cat_119, [16384, 4096]);  cat_119 = None
        permute_597 = torch.ops.aten.permute.default(view_1483, [1, 0])
        wait_tensor_34 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_5);  reduce_scatter_tensor_5 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, wait_tensor_34);  wait_tensor_34 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        all_gather_into_tensor_29 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_86, 8, '0');  convert_element_type_86 = None
        wait_tensor_35 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_29);  all_gather_into_tensor_29 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [2], True);  pow_6 = None
        add_10 = torch.ops.aten.add.Scalar(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(convert_element_type_87, rsqrt_5);  convert_element_type_87 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, wait_tensor_35)
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
        view_204 = torch.ops.aten.view.default(cat_11, [16384, 4096]);  cat_11 = None
        view_205 = torch.ops.aten.view.default(mm_18, [2, 8192, 1792]);  mm_18 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_92)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_92, sigmoid_2);  sigmoid_2 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        all_gather_into_tensor_32 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_94, 8, '0');  convert_element_type_94 = None
        wait_tensor_38 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_32);  all_gather_into_tensor_32 = None
        permute_31 = torch.ops.aten.permute.default(wait_tensor_38, [1, 0]);  wait_tensor_38 = None
        mm_19 = torch.ops.aten.mm.default(view_204, permute_31)
        view_212 = torch.ops.aten.view.default(mm_19, [2, 8192, 1792]);  mm_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_93, view_212)
        view_219 = torch.ops.aten.view.default(mul_23, [16384, 1792]);  mul_23 = None
        mm_297 = torch.ops.aten.mm.default(permute_597, view_219);  permute_597 = view_219 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        all_gather_into_tensor_33 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_97, 8, '0');  convert_element_type_97 = None
        wait_tensor_39 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_33);  all_gather_into_tensor_33 = None
        permute_32 = torch.ops.aten.permute.default(wait_tensor_39, [1, 0]);  wait_tensor_39 = None
        permute_599 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_298 = torch.ops.aten.mm.default(view_1483, permute_599);  view_1483 = permute_599 = None
        view_1484 = torch.ops.aten.view.default(mm_298, [2, 8192, 1792]);  mm_298 = None
        convert_element_type_1252 = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None
        reduce_scatter_tensor_179 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1252, 'avg', 8, '0');  convert_element_type_1252 = None
        wait_tensor_413 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_179);  reduce_scatter_tensor_179 = None
        mul_396 = torch.ops.aten.mul.Tensor(view_1484, convert_element_type_93);  convert_element_type_93 = None
        mul_397 = torch.ops.aten.mul.Tensor(view_1484, view_212);  view_1484 = view_212 = None
        view_1485 = torch.ops.aten.view.default(mul_396, [16384, 1792]);  mul_396 = None
        permute_601 = torch.ops.aten.permute.default(view_1485, [1, 0])
        mm_299 = torch.ops.aten.mm.default(permute_601, view_204);  permute_601 = None
        permute_603 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_300 = torch.ops.aten.mm.default(view_1485, permute_603);  view_1485 = permute_603 = None
        view_1486 = torch.ops.aten.view.default(mm_300, [2, 8192, 4096]);  mm_300 = None
        convert_element_type_1257 = torch.ops.prims.convert_element_type.default(mm_299, torch.float32);  mm_299 = None
        reduce_scatter_tensor_180 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1257, 'avg', 8, '0');  convert_element_type_1257 = None
        wait_tensor_414 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_180);  reduce_scatter_tensor_180 = None
        convert_element_type_1258 = torch.ops.prims.convert_element_type.default(mul_397, torch.float32);  mul_397 = None
        neg_13 = torch.ops.aten.neg.default(convert_element_type_92)
        exp_13 = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_156 = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(add_156);  add_156 = None
        mul_398 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_399 = torch.ops.aten.mul.Tensor(convert_element_type_1258, mul_398);  convert_element_type_1258 = None
        sub_41 = torch.ops.aten.sub.Tensor(1, mul_398);  mul_398 = None
        mul_400 = torch.ops.aten.mul.Tensor(convert_element_type_92, sub_41);  convert_element_type_92 = sub_41 = None
        add_157 = torch.ops.aten.add.Tensor(mul_400, 1);  mul_400 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_399, add_157);  mul_399 = add_157 = None
        convert_element_type_1260 = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16);  mul_401 = None
        view_1487 = torch.ops.aten.view.default(convert_element_type_1260, [16384, 1792]);  convert_element_type_1260 = None
        permute_605 = torch.ops.aten.permute.default(view_1487, [1, 0])
        mm_301 = torch.ops.aten.mm.default(permute_605, view_204);  permute_605 = view_204 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        all_gather_into_tensor_31 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_89, 8, '0');  convert_element_type_89 = None
        wait_tensor_37 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_31);  all_gather_into_tensor_31 = None
        permute_30 = torch.ops.aten.permute.default(wait_tensor_37, [1, 0]);  wait_tensor_37 = None
        permute_607 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_302 = torch.ops.aten.mm.default(view_1487, permute_607);  view_1487 = permute_607 = None
        view_1488 = torch.ops.aten.view.default(mm_302, [2, 8192, 4096]);  mm_302 = None
        add_158 = torch.ops.aten.add.Tensor(view_1486, view_1488);  view_1486 = view_1488 = None
        convert_element_type_1265 = torch.ops.prims.convert_element_type.default(mm_301, torch.float32);  mm_301 = None
        reduce_scatter_tensor_181 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1265, 'avg', 8, '0');  convert_element_type_1265 = None
        wait_tensor_415 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_181);  reduce_scatter_tensor_181 = None
        split_128 = torch.ops.aten.split.Tensor(add_158, 1024, 1);  add_158 = None
        getitem_1207 = split_128[0]
        getitem_1208 = split_128[1]
        getitem_1209 = split_128[2]
        getitem_1210 = split_128[3]
        getitem_1211 = split_128[4]
        getitem_1212 = split_128[5]
        getitem_1213 = split_128[6]
        getitem_1214 = split_128[7];  split_128 = None
        cat_120 = torch.ops.aten.cat.default([getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214]);  getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = None
        reduce_scatter_tensor_182 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_120, 'sum', 8, '1');  cat_120 = None
        wait_tensor_416 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_182);  reduce_scatter_tensor_182 = None
        convert_element_type_1266 = torch.ops.prims.convert_element_type.default(wait_tensor_416, torch.float32);  wait_tensor_416 = None
        convert_element_type_1268 = torch.ops.prims.convert_element_type.default(wait_tensor_35, torch.float32);  wait_tensor_35 = None
        mul_402 = torch.ops.aten.mul.Tensor(convert_element_type_1266, convert_element_type_1268);  convert_element_type_1268 = None
        mul_404 = torch.ops.aten.mul.Tensor(mul_20, mul_402)
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        div_27 = torch.ops.aten.div.Tensor(mul_20, 4096)
        mul_405 = torch.ops.aten.mul.Tensor(div_27, sum_81);  div_27 = sum_81 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_402, mul_405);  mul_402 = mul_405 = None
        mul_406 = torch.ops.aten.mul.Tensor(sub_42, rsqrt_5);  sub_42 = rsqrt_5 = None
        mul_407 = torch.ops.aten.mul.Tensor(convert_element_type_1266, mul_20);  convert_element_type_1266 = mul_20 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        convert_element_type_1269 = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16);  mul_406 = None
        convert_element_type_1270 = torch.ops.prims.convert_element_type.default(sum_82, torch.bfloat16);  sum_82 = None
        all_reduce_27 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1270, 'sum', '1');  convert_element_type_1270 = None
        wait_tensor_417 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_27);  all_reduce_27 = None
        convert_element_type_1271 = torch.ops.prims.convert_element_type.default(wait_tensor_417, torch.float32);  wait_tensor_417 = None
        reduce_scatter_tensor_183 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1271, 'avg', 8, '0');  convert_element_type_1271 = None
        wait_tensor_418 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_183);  reduce_scatter_tensor_183 = None
        add_159 = torch.ops.aten.add.Tensor(add_155, convert_element_type_1269);  add_155 = convert_element_type_1269 = None
        all_gather_into_tensor_207 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_159, 8, '1')
        wait_tensor_419 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_207);  all_gather_into_tensor_207 = None
        split_129 = torch.ops.aten.split.Tensor(wait_tensor_419, 2);  wait_tensor_419 = None
        getitem_1215 = split_129[0]
        getitem_1216 = split_129[1]
        getitem_1217 = split_129[2]
        getitem_1218 = split_129[3]
        getitem_1219 = split_129[4]
        getitem_1220 = split_129[5]
        getitem_1221 = split_129[6]
        getitem_1222 = split_129[7];  split_129 = None
        cat_121 = torch.ops.aten.cat.default([getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222], 1);  getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = None
        view_1489 = torch.ops.aten.view.default(cat_121, [16384, 4096]);  cat_121 = None
        permute_609 = torch.ops.aten.permute.default(view_1489, [1, 0])
        permute_28 = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3])
        view_186 = torch.ops.aten.view.default(permute_28, [2, 8192, -1]);  permute_28 = None
        view_192 = torch.ops.aten.view.default(view_186, [16384, 512]);  view_186 = None
        mm_303 = torch.ops.aten.mm.default(permute_609, view_192);  permute_609 = view_192 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        all_gather_into_tensor_28 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_83, 8, '0');  convert_element_type_83 = None
        wait_tensor_33 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_28);  all_gather_into_tensor_28 = None
        permute_29 = torch.ops.aten.permute.default(wait_tensor_33, [1, 0]);  wait_tensor_33 = None
        permute_611 = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_304 = torch.ops.aten.mm.default(view_1489, permute_611);  view_1489 = permute_611 = None
        view_1490 = torch.ops.aten.view.default(mm_304, [2, 8192, 512]);  mm_304 = None
        convert_element_type_1276 = torch.ops.prims.convert_element_type.default(mm_303, torch.float32);  mm_303 = None
        reduce_scatter_tensor_184 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1276, 'avg', 8, '0');  convert_element_type_1276 = None
        wait_tensor_420 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_184);  reduce_scatter_tensor_184 = None
        view_1491 = torch.ops.aten.view.default(view_1490, [2, 8192, 4, 128]);  view_1490 = None
        permute_613 = torch.ops.aten.permute.default(view_1491, [0, 2, 1, 3]);  view_1491 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        all_gather_into_tensor_23 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_67, 8, '0');  convert_element_type_67 = None
        wait_tensor_28 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_23);  all_gather_into_tensor_23 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_68, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [2], True);  pow_5 = None
        add_8 = torch.ops.aten.add.Scalar(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_68, rsqrt_4);  convert_element_type_68 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, wait_tensor_28)
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
        view_159 = torch.ops.aten.view.default(cat_9, [16384, 4096]);  cat_9 = None
        view_160 = torch.ops.aten.view.default(mm_14, [2, 8192, 512]);  mm_14 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        all_gather_into_tensor_26 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_73, 8, '0');  convert_element_type_73 = None
        wait_tensor_31 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_26);  all_gather_into_tensor_26 = None
        permute_23 = torch.ops.aten.permute.default(wait_tensor_31, [1, 0]);  wait_tensor_31 = None
        mm_15 = torch.ops.aten.mm.default(view_159, permute_23)
        view_167 = torch.ops.aten.view.default(mm_15, [2, 8192, 128]);  mm_15 = None
        view_174 = torch.ops.aten.view.default(mm_16, [2, 8192, 128]);  mm_16 = None
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
        _scaled_dot_product_cudnn_attention_backward_13 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_613, permute_25, permute_26, permute_27, getitem_162, getitem_163, getitem_168, getitem_169, None, None, None, 8192, 8192, 0.0, True);  permute_613 = permute_25 = permute_26 = permute_27 = getitem_162 = getitem_163 = getitem_168 = getitem_169 = None
        getitem_1223 = _scaled_dot_product_cudnn_attention_backward_13[0]
        getitem_1224 = _scaled_dot_product_cudnn_attention_backward_13[1]
        getitem_1225 = _scaled_dot_product_cudnn_attention_backward_13[2];  _scaled_dot_product_cudnn_attention_backward_13 = None
        permute_614 = torch.ops.aten.permute.default(getitem_1225, [0, 2, 1, 3]);  getitem_1225 = None
        permute_615 = torch.ops.aten.permute.default(getitem_1224, [0, 2, 1, 3]);  getitem_1224 = None
        permute_616 = torch.ops.aten.permute.default(getitem_1223, [0, 2, 1, 3]);  getitem_1223 = None
        view_1492 = torch.ops.aten.view.default(permute_614, [2, 8192, 1, 4, 128]);  permute_614 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(view_1492, [3], True);  view_1492 = None
        squeeze_26 = torch.ops.aten.squeeze.dim(sum_83, 3);  sum_83 = None
        view_1493 = torch.ops.aten.view.default(permute_615, [2, 8192, 1, 4, 128]);  permute_615 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(view_1493, [3], True);  view_1493 = None
        squeeze_27 = torch.ops.aten.squeeze.dim(sum_84, 3);  sum_84 = None
        convert_element_type_1277 = torch.ops.prims.convert_element_type.default(squeeze_27, torch.float32);  squeeze_27 = None
        convert_element_type_1278 = torch.ops.prims.convert_element_type.default(permute_616, torch.float32);  permute_616 = None
        view_1494 = torch.ops.aten.view.default(convert_element_type_1277, [2, 8192, 1, 64, 2]);  convert_element_type_1277 = None
        view_as_complex_58 = torch.ops.aten.view_as_complex.default(view_1494);  view_1494 = None
        mul_408 = torch.ops.aten.mul.Tensor(view_as_complex_58, _conj);  view_as_complex_58 = None
        view_1495 = torch.ops.aten.view.default(convert_element_type_1278, [2, 8192, 4, 64, 2]);  convert_element_type_1278 = None
        view_as_complex_59 = torch.ops.aten.view_as_complex.default(view_1495);  view_1495 = None
        mul_409 = torch.ops.aten.mul.Tensor(view_as_complex_59, _conj);  view_as_complex_59 = None
        view_as_real_58 = torch.ops.aten.view_as_real.default(mul_408);  mul_408 = None
        view_1496 = torch.ops.aten.view.default(view_as_real_58, [2, 8192, 1, 128]);  view_as_real_58 = None
        convert_element_type_1279 = torch.ops.prims.convert_element_type.default(view_1496, torch.bfloat16);  view_1496 = None
        view_as_real_59 = torch.ops.aten.view_as_real.default(mul_409);  mul_409 = None
        view_1497 = torch.ops.aten.view.default(view_as_real_59, [2, 8192, 4, 128]);  view_as_real_59 = None
        convert_element_type_1280 = torch.ops.prims.convert_element_type.default(view_1497, torch.bfloat16);  view_1497 = None
        view_1498 = torch.ops.aten.view.default(squeeze_26, [2, 8192, 128]);  squeeze_26 = None
        view_1499 = torch.ops.aten.view.default(convert_element_type_1279, [2, 8192, 128]);  convert_element_type_1279 = None
        view_1500 = torch.ops.aten.view.default(convert_element_type_1280, [2, 8192, 512]);  convert_element_type_1280 = None
        view_1501 = torch.ops.aten.view.default(view_1498, [16384, 128]);  view_1498 = None
        permute_617 = torch.ops.aten.permute.default(view_1501, [1, 0])
        mm_305 = torch.ops.aten.mm.default(permute_617, view_159);  permute_617 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        all_gather_into_tensor_27 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_76, 8, '0');  convert_element_type_76 = None
        wait_tensor_32 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_27);  all_gather_into_tensor_27 = None
        permute_24 = torch.ops.aten.permute.default(wait_tensor_32, [1, 0]);  wait_tensor_32 = None
        permute_619 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_306 = torch.ops.aten.mm.default(view_1501, permute_619);  view_1501 = permute_619 = None
        view_1502 = torch.ops.aten.view.default(mm_306, [2, 8192, 4096]);  mm_306 = None
        convert_element_type_1285 = torch.ops.prims.convert_element_type.default(mm_305, torch.float32);  mm_305 = None
        reduce_scatter_tensor_185 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1285, 'avg', 8, '0');  convert_element_type_1285 = None
        wait_tensor_421 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_185);  reduce_scatter_tensor_185 = None
        view_1503 = torch.ops.aten.view.default(view_1499, [16384, 128]);  view_1499 = None
        permute_621 = torch.ops.aten.permute.default(view_1503, [1, 0])
        mm_307 = torch.ops.aten.mm.default(permute_621, view_159);  permute_621 = None
        permute_623 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_308 = torch.ops.aten.mm.default(view_1503, permute_623);  view_1503 = permute_623 = None
        view_1504 = torch.ops.aten.view.default(mm_308, [2, 8192, 4096]);  mm_308 = None
        add_160 = torch.ops.aten.add.Tensor(view_1502, view_1504);  view_1502 = view_1504 = None
        convert_element_type_1290 = torch.ops.prims.convert_element_type.default(mm_307, torch.float32);  mm_307 = None
        reduce_scatter_tensor_186 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1290, 'avg', 8, '0');  convert_element_type_1290 = None
        wait_tensor_422 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_186);  reduce_scatter_tensor_186 = None
        view_1505 = torch.ops.aten.view.default(view_1500, [16384, 512]);  view_1500 = None
        permute_625 = torch.ops.aten.permute.default(view_1505, [1, 0])
        mm_309 = torch.ops.aten.mm.default(permute_625, view_159);  permute_625 = view_159 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        all_gather_into_tensor_25 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_70, 8, '0');  convert_element_type_70 = None
        wait_tensor_30 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_25);  all_gather_into_tensor_25 = None
        permute_22 = torch.ops.aten.permute.default(wait_tensor_30, [1, 0]);  wait_tensor_30 = None
        permute_627 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_310 = torch.ops.aten.mm.default(view_1505, permute_627);  view_1505 = permute_627 = None
        view_1506 = torch.ops.aten.view.default(mm_310, [2, 8192, 4096]);  mm_310 = None
        add_161 = torch.ops.aten.add.Tensor(add_160, view_1506);  add_160 = view_1506 = None
        convert_element_type_1295 = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None
        reduce_scatter_tensor_187 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1295, 'avg', 8, '0');  convert_element_type_1295 = None
        wait_tensor_423 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_187);  reduce_scatter_tensor_187 = None
        split_130 = torch.ops.aten.split.Tensor(add_161, 1024, 1);  add_161 = None
        getitem_1226 = split_130[0]
        getitem_1227 = split_130[1]
        getitem_1228 = split_130[2]
        getitem_1229 = split_130[3]
        getitem_1230 = split_130[4]
        getitem_1231 = split_130[5]
        getitem_1232 = split_130[6]
        getitem_1233 = split_130[7];  split_130 = None
        cat_122 = torch.ops.aten.cat.default([getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233]);  getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = None
        reduce_scatter_tensor_188 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_122, 'sum', 8, '1');  cat_122 = None
        wait_tensor_424 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_188);  reduce_scatter_tensor_188 = None
        convert_element_type_1296 = torch.ops.prims.convert_element_type.default(wait_tensor_424, torch.float32);  wait_tensor_424 = None
        convert_element_type_1298 = torch.ops.prims.convert_element_type.default(wait_tensor_28, torch.float32);  wait_tensor_28 = None
        mul_410 = torch.ops.aten.mul.Tensor(convert_element_type_1296, convert_element_type_1298);  convert_element_type_1298 = None
        mul_412 = torch.ops.aten.mul.Tensor(mul_16, mul_410)
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_412, [2], True);  mul_412 = None
        div_28 = torch.ops.aten.div.Tensor(mul_16, 4096)
        mul_413 = torch.ops.aten.mul.Tensor(div_28, sum_85);  div_28 = sum_85 = None
        sub_43 = torch.ops.aten.sub.Tensor(mul_410, mul_413);  mul_410 = mul_413 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_43, rsqrt_4);  sub_43 = rsqrt_4 = None
        mul_415 = torch.ops.aten.mul.Tensor(convert_element_type_1296, mul_16);  convert_element_type_1296 = mul_16 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1]);  mul_415 = None
        convert_element_type_1299 = torch.ops.prims.convert_element_type.default(mul_414, torch.bfloat16);  mul_414 = None
        convert_element_type_1300 = torch.ops.prims.convert_element_type.default(sum_86, torch.bfloat16);  sum_86 = None
        all_reduce_28 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1300, 'sum', '1');  convert_element_type_1300 = None
        wait_tensor_425 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_28);  all_reduce_28 = None
        convert_element_type_1301 = torch.ops.prims.convert_element_type.default(wait_tensor_425, torch.float32);  wait_tensor_425 = None
        reduce_scatter_tensor_189 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1301, 'avg', 8, '0');  convert_element_type_1301 = None
        wait_tensor_426 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_189);  reduce_scatter_tensor_189 = None
        add_162 = torch.ops.aten.add.Tensor(add_159, convert_element_type_1299);  add_159 = convert_element_type_1299 = None
        all_gather_into_tensor_208 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_162, 8, '1')
        wait_tensor_427 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_208);  all_gather_into_tensor_208 = None
        split_131 = torch.ops.aten.split.Tensor(wait_tensor_427, 2);  wait_tensor_427 = None
        getitem_1234 = split_131[0]
        getitem_1235 = split_131[1]
        getitem_1236 = split_131[2]
        getitem_1237 = split_131[3]
        getitem_1238 = split_131[4]
        getitem_1239 = split_131[5]
        getitem_1240 = split_131[6]
        getitem_1241 = split_131[7];  split_131 = None
        cat_123 = torch.ops.aten.cat.default([getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241], 1);  getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = None
        view_1507 = torch.ops.aten.view.default(cat_123, [16384, 4096]);  cat_123 = None
        permute_629 = torch.ops.aten.permute.default(view_1507, [1, 0])
        wait_tensor_21 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3);  reduce_scatter_tensor_3 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, wait_tensor_21);  wait_tensor_21 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        all_gather_into_tensor_18 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_53, 8, '0');  convert_element_type_53 = None
        wait_tensor_22 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_18);  all_gather_into_tensor_18 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [2], True);  pow_4 = None
        add_6 = torch.ops.aten.add.Scalar(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_54, rsqrt_3);  convert_element_type_54 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, wait_tensor_22)
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
        view_132 = torch.ops.aten.view.default(cat_7, [16384, 4096]);  cat_7 = None
        view_133 = torch.ops.aten.view.default(mm_11, [2, 8192, 1792]);  mm_11 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_59, sigmoid_1);  sigmoid_1 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        all_gather_into_tensor_21 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_61, 8, '0');  convert_element_type_61 = None
        wait_tensor_25 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_21);  all_gather_into_tensor_21 = None
        permute_20 = torch.ops.aten.permute.default(wait_tensor_25, [1, 0]);  wait_tensor_25 = None
        mm_12 = torch.ops.aten.mm.default(view_132, permute_20)
        view_140 = torch.ops.aten.view.default(mm_12, [2, 8192, 1792]);  mm_12 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_60, view_140)
        view_147 = torch.ops.aten.view.default(mul_15, [16384, 1792]);  mul_15 = None
        mm_311 = torch.ops.aten.mm.default(permute_629, view_147);  permute_629 = view_147 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        all_gather_into_tensor_22 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_64, 8, '0');  convert_element_type_64 = None
        wait_tensor_26 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_22);  all_gather_into_tensor_22 = None
        permute_21 = torch.ops.aten.permute.default(wait_tensor_26, [1, 0]);  wait_tensor_26 = None
        permute_631 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_312 = torch.ops.aten.mm.default(view_1507, permute_631);  view_1507 = permute_631 = None
        view_1508 = torch.ops.aten.view.default(mm_312, [2, 8192, 1792]);  mm_312 = None
        convert_element_type_1306 = torch.ops.prims.convert_element_type.default(mm_311, torch.float32);  mm_311 = None
        reduce_scatter_tensor_190 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1306, 'avg', 8, '0');  convert_element_type_1306 = None
        wait_tensor_428 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_190);  reduce_scatter_tensor_190 = None
        mul_416 = torch.ops.aten.mul.Tensor(view_1508, convert_element_type_60);  convert_element_type_60 = None
        mul_417 = torch.ops.aten.mul.Tensor(view_1508, view_140);  view_1508 = view_140 = None
        view_1509 = torch.ops.aten.view.default(mul_416, [16384, 1792]);  mul_416 = None
        permute_633 = torch.ops.aten.permute.default(view_1509, [1, 0])
        mm_313 = torch.ops.aten.mm.default(permute_633, view_132);  permute_633 = None
        permute_635 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_314 = torch.ops.aten.mm.default(view_1509, permute_635);  view_1509 = permute_635 = None
        view_1510 = torch.ops.aten.view.default(mm_314, [2, 8192, 4096]);  mm_314 = None
        convert_element_type_1311 = torch.ops.prims.convert_element_type.default(mm_313, torch.float32);  mm_313 = None
        reduce_scatter_tensor_191 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1311, 'avg', 8, '0');  convert_element_type_1311 = None
        wait_tensor_429 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_191);  reduce_scatter_tensor_191 = None
        convert_element_type_1312 = torch.ops.prims.convert_element_type.default(mul_417, torch.float32);  mul_417 = None
        neg_14 = torch.ops.aten.neg.default(convert_element_type_59)
        exp_14 = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_163 = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_163);  add_163 = None
        mul_418 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_419 = torch.ops.aten.mul.Tensor(convert_element_type_1312, mul_418);  convert_element_type_1312 = None
        sub_44 = torch.ops.aten.sub.Tensor(1, mul_418);  mul_418 = None
        mul_420 = torch.ops.aten.mul.Tensor(convert_element_type_59, sub_44);  convert_element_type_59 = sub_44 = None
        add_164 = torch.ops.aten.add.Tensor(mul_420, 1);  mul_420 = None
        mul_421 = torch.ops.aten.mul.Tensor(mul_419, add_164);  mul_419 = add_164 = None
        convert_element_type_1314 = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16);  mul_421 = None
        view_1511 = torch.ops.aten.view.default(convert_element_type_1314, [16384, 1792]);  convert_element_type_1314 = None
        permute_637 = torch.ops.aten.permute.default(view_1511, [1, 0])
        mm_315 = torch.ops.aten.mm.default(permute_637, view_132);  permute_637 = view_132 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        all_gather_into_tensor_20 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_56, 8, '0');  convert_element_type_56 = None
        wait_tensor_24 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_20);  all_gather_into_tensor_20 = None
        permute_19 = torch.ops.aten.permute.default(wait_tensor_24, [1, 0]);  wait_tensor_24 = None
        permute_639 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_316 = torch.ops.aten.mm.default(view_1511, permute_639);  view_1511 = permute_639 = None
        view_1512 = torch.ops.aten.view.default(mm_316, [2, 8192, 4096]);  mm_316 = None
        add_165 = torch.ops.aten.add.Tensor(view_1510, view_1512);  view_1510 = view_1512 = None
        convert_element_type_1319 = torch.ops.prims.convert_element_type.default(mm_315, torch.float32);  mm_315 = None
        reduce_scatter_tensor_192 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1319, 'avg', 8, '0');  convert_element_type_1319 = None
        wait_tensor_430 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_192);  reduce_scatter_tensor_192 = None
        split_132 = torch.ops.aten.split.Tensor(add_165, 1024, 1);  add_165 = None
        getitem_1242 = split_132[0]
        getitem_1243 = split_132[1]
        getitem_1244 = split_132[2]
        getitem_1245 = split_132[3]
        getitem_1246 = split_132[4]
        getitem_1247 = split_132[5]
        getitem_1248 = split_132[6]
        getitem_1249 = split_132[7];  split_132 = None
        cat_124 = torch.ops.aten.cat.default([getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249]);  getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = None
        reduce_scatter_tensor_193 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_124, 'sum', 8, '1');  cat_124 = None
        wait_tensor_431 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_193);  reduce_scatter_tensor_193 = None
        convert_element_type_1320 = torch.ops.prims.convert_element_type.default(wait_tensor_431, torch.float32);  wait_tensor_431 = None
        convert_element_type_1322 = torch.ops.prims.convert_element_type.default(wait_tensor_22, torch.float32);  wait_tensor_22 = None
        mul_422 = torch.ops.aten.mul.Tensor(convert_element_type_1320, convert_element_type_1322);  convert_element_type_1322 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_12, mul_422)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
        div_29 = torch.ops.aten.div.Tensor(mul_12, 4096)
        mul_425 = torch.ops.aten.mul.Tensor(div_29, sum_87);  div_29 = sum_87 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_422, mul_425);  mul_422 = mul_425 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_45, rsqrt_3);  sub_45 = rsqrt_3 = None
        mul_427 = torch.ops.aten.mul.Tensor(convert_element_type_1320, mul_12);  convert_element_type_1320 = mul_12 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
        convert_element_type_1323 = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        convert_element_type_1324 = torch.ops.prims.convert_element_type.default(sum_88, torch.bfloat16);  sum_88 = None
        all_reduce_29 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1324, 'sum', '1');  convert_element_type_1324 = None
        wait_tensor_432 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_29);  all_reduce_29 = None
        convert_element_type_1325 = torch.ops.prims.convert_element_type.default(wait_tensor_432, torch.float32);  wait_tensor_432 = None
        reduce_scatter_tensor_194 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1325, 'avg', 8, '0');  convert_element_type_1325 = None
        wait_tensor_433 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_194);  reduce_scatter_tensor_194 = None
        add_166 = torch.ops.aten.add.Tensor(add_162, convert_element_type_1323);  add_162 = convert_element_type_1323 = None
        all_gather_into_tensor_209 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_166, 8, '1')
        wait_tensor_434 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_209);  all_gather_into_tensor_209 = None
        split_133 = torch.ops.aten.split.Tensor(wait_tensor_434, 2);  wait_tensor_434 = None
        getitem_1250 = split_133[0]
        getitem_1251 = split_133[1]
        getitem_1252 = split_133[2]
        getitem_1253 = split_133[3]
        getitem_1254 = split_133[4]
        getitem_1255 = split_133[5]
        getitem_1256 = split_133[6]
        getitem_1257 = split_133[7];  split_133 = None
        cat_125 = torch.ops.aten.cat.default([getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257], 1);  getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = None
        view_1513 = torch.ops.aten.view.default(cat_125, [16384, 4096]);  cat_125 = None
        permute_641 = torch.ops.aten.permute.default(view_1513, [1, 0])
        permute_17 = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3])
        view_114 = torch.ops.aten.view.default(permute_17, [2, 8192, -1]);  permute_17 = None
        view_120 = torch.ops.aten.view.default(view_114, [16384, 512]);  view_114 = None
        mm_317 = torch.ops.aten.mm.default(permute_641, view_120);  permute_641 = view_120 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        all_gather_into_tensor_17 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_50, 8, '0');  convert_element_type_50 = None
        wait_tensor_20 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_17);  all_gather_into_tensor_17 = None
        permute_18 = torch.ops.aten.permute.default(wait_tensor_20, [1, 0]);  wait_tensor_20 = None
        permute_643 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_318 = torch.ops.aten.mm.default(view_1513, permute_643);  view_1513 = permute_643 = None
        view_1514 = torch.ops.aten.view.default(mm_318, [2, 8192, 512]);  mm_318 = None
        convert_element_type_1330 = torch.ops.prims.convert_element_type.default(mm_317, torch.float32);  mm_317 = None
        reduce_scatter_tensor_195 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1330, 'avg', 8, '0');  convert_element_type_1330 = None
        wait_tensor_435 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_195);  reduce_scatter_tensor_195 = None
        view_1515 = torch.ops.aten.view.default(view_1514, [2, 8192, 4, 128]);  view_1514 = None
        permute_645 = torch.ops.aten.permute.default(view_1515, [0, 2, 1, 3]);  view_1515 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        all_gather_into_tensor_12 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_34, 8, '0');  convert_element_type_34 = None
        wait_tensor_15 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [2], True);  pow_3 = None
        add_4 = torch.ops.aten.add.Scalar(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_35, rsqrt_2);  convert_element_type_35 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, wait_tensor_15)
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
        view_87 = torch.ops.aten.view.default(cat_5, [16384, 4096]);  cat_5 = None
        view_88 = torch.ops.aten.view.default(mm_7, [2, 8192, 512]);  mm_7 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        all_gather_into_tensor_15 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_40, 8, '0');  convert_element_type_40 = None
        wait_tensor_18 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        permute_12 = torch.ops.aten.permute.default(wait_tensor_18, [1, 0]);  wait_tensor_18 = None
        mm_8 = torch.ops.aten.mm.default(view_87, permute_12)
        view_95 = torch.ops.aten.view.default(mm_8, [2, 8192, 128]);  mm_8 = None
        view_102 = torch.ops.aten.view.default(mm_9, [2, 8192, 128]);  mm_9 = None
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
        _scaled_dot_product_cudnn_attention_backward_14 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_645, permute_14, permute_15, permute_16, getitem_121, getitem_122, getitem_127, getitem_128, None, None, None, 8192, 8192, 0.0, True);  permute_645 = permute_14 = permute_15 = permute_16 = getitem_121 = getitem_122 = getitem_127 = getitem_128 = None
        getitem_1258 = _scaled_dot_product_cudnn_attention_backward_14[0]
        getitem_1259 = _scaled_dot_product_cudnn_attention_backward_14[1]
        getitem_1260 = _scaled_dot_product_cudnn_attention_backward_14[2];  _scaled_dot_product_cudnn_attention_backward_14 = None
        permute_646 = torch.ops.aten.permute.default(getitem_1260, [0, 2, 1, 3]);  getitem_1260 = None
        permute_647 = torch.ops.aten.permute.default(getitem_1259, [0, 2, 1, 3]);  getitem_1259 = None
        permute_648 = torch.ops.aten.permute.default(getitem_1258, [0, 2, 1, 3]);  getitem_1258 = None
        view_1516 = torch.ops.aten.view.default(permute_646, [2, 8192, 1, 4, 128]);  permute_646 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(view_1516, [3], True);  view_1516 = None
        squeeze_28 = torch.ops.aten.squeeze.dim(sum_89, 3);  sum_89 = None
        view_1517 = torch.ops.aten.view.default(permute_647, [2, 8192, 1, 4, 128]);  permute_647 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(view_1517, [3], True);  view_1517 = None
        squeeze_29 = torch.ops.aten.squeeze.dim(sum_90, 3);  sum_90 = None
        convert_element_type_1331 = torch.ops.prims.convert_element_type.default(squeeze_29, torch.float32);  squeeze_29 = None
        convert_element_type_1332 = torch.ops.prims.convert_element_type.default(permute_648, torch.float32);  permute_648 = None
        view_1518 = torch.ops.aten.view.default(convert_element_type_1331, [2, 8192, 1, 64, 2]);  convert_element_type_1331 = None
        view_as_complex_60 = torch.ops.aten.view_as_complex.default(view_1518);  view_1518 = None
        mul_428 = torch.ops.aten.mul.Tensor(view_as_complex_60, _conj);  view_as_complex_60 = None
        view_1519 = torch.ops.aten.view.default(convert_element_type_1332, [2, 8192, 4, 64, 2]);  convert_element_type_1332 = None
        view_as_complex_61 = torch.ops.aten.view_as_complex.default(view_1519);  view_1519 = None
        mul_429 = torch.ops.aten.mul.Tensor(view_as_complex_61, _conj);  view_as_complex_61 = None
        view_as_real_60 = torch.ops.aten.view_as_real.default(mul_428);  mul_428 = None
        view_1520 = torch.ops.aten.view.default(view_as_real_60, [2, 8192, 1, 128]);  view_as_real_60 = None
        convert_element_type_1333 = torch.ops.prims.convert_element_type.default(view_1520, torch.bfloat16);  view_1520 = None
        view_as_real_61 = torch.ops.aten.view_as_real.default(mul_429);  mul_429 = None
        view_1521 = torch.ops.aten.view.default(view_as_real_61, [2, 8192, 4, 128]);  view_as_real_61 = None
        convert_element_type_1334 = torch.ops.prims.convert_element_type.default(view_1521, torch.bfloat16);  view_1521 = None
        view_1522 = torch.ops.aten.view.default(squeeze_28, [2, 8192, 128]);  squeeze_28 = None
        view_1523 = torch.ops.aten.view.default(convert_element_type_1333, [2, 8192, 128]);  convert_element_type_1333 = None
        view_1524 = torch.ops.aten.view.default(convert_element_type_1334, [2, 8192, 512]);  convert_element_type_1334 = None
        view_1525 = torch.ops.aten.view.default(view_1522, [16384, 128]);  view_1522 = None
        permute_649 = torch.ops.aten.permute.default(view_1525, [1, 0])
        mm_319 = torch.ops.aten.mm.default(permute_649, view_87);  permute_649 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        all_gather_into_tensor_16 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_43, 8, '0');  convert_element_type_43 = None
        wait_tensor_19 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        permute_13 = torch.ops.aten.permute.default(wait_tensor_19, [1, 0]);  wait_tensor_19 = None
        permute_651 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_320 = torch.ops.aten.mm.default(view_1525, permute_651);  view_1525 = permute_651 = None
        view_1526 = torch.ops.aten.view.default(mm_320, [2, 8192, 4096]);  mm_320 = None
        convert_element_type_1339 = torch.ops.prims.convert_element_type.default(mm_319, torch.float32);  mm_319 = None
        reduce_scatter_tensor_196 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1339, 'avg', 8, '0');  convert_element_type_1339 = None
        wait_tensor_436 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_196);  reduce_scatter_tensor_196 = None
        view_1527 = torch.ops.aten.view.default(view_1523, [16384, 128]);  view_1523 = None
        permute_653 = torch.ops.aten.permute.default(view_1527, [1, 0])
        mm_321 = torch.ops.aten.mm.default(permute_653, view_87);  permute_653 = None
        permute_655 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_322 = torch.ops.aten.mm.default(view_1527, permute_655);  view_1527 = permute_655 = None
        view_1528 = torch.ops.aten.view.default(mm_322, [2, 8192, 4096]);  mm_322 = None
        add_167 = torch.ops.aten.add.Tensor(view_1526, view_1528);  view_1526 = view_1528 = None
        convert_element_type_1344 = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None
        reduce_scatter_tensor_197 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1344, 'avg', 8, '0');  convert_element_type_1344 = None
        wait_tensor_437 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_197);  reduce_scatter_tensor_197 = None
        view_1529 = torch.ops.aten.view.default(view_1524, [16384, 512]);  view_1524 = None
        permute_657 = torch.ops.aten.permute.default(view_1529, [1, 0])
        mm_323 = torch.ops.aten.mm.default(permute_657, view_87);  permute_657 = view_87 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        all_gather_into_tensor_14 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_37, 8, '0');  convert_element_type_37 = None
        wait_tensor_17 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        permute_11 = torch.ops.aten.permute.default(wait_tensor_17, [1, 0]);  wait_tensor_17 = None
        permute_659 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_324 = torch.ops.aten.mm.default(view_1529, permute_659);  view_1529 = permute_659 = None
        view_1530 = torch.ops.aten.view.default(mm_324, [2, 8192, 4096]);  mm_324 = None
        add_168 = torch.ops.aten.add.Tensor(add_167, view_1530);  add_167 = view_1530 = None
        convert_element_type_1349 = torch.ops.prims.convert_element_type.default(mm_323, torch.float32);  mm_323 = None
        reduce_scatter_tensor_198 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1349, 'avg', 8, '0');  convert_element_type_1349 = None
        wait_tensor_438 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_198);  reduce_scatter_tensor_198 = None
        split_134 = torch.ops.aten.split.Tensor(add_168, 1024, 1);  add_168 = None
        getitem_1261 = split_134[0]
        getitem_1262 = split_134[1]
        getitem_1263 = split_134[2]
        getitem_1264 = split_134[3]
        getitem_1265 = split_134[4]
        getitem_1266 = split_134[5]
        getitem_1267 = split_134[6]
        getitem_1268 = split_134[7];  split_134 = None
        cat_126 = torch.ops.aten.cat.default([getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268]);  getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = None
        reduce_scatter_tensor_199 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_126, 'sum', 8, '1');  cat_126 = None
        wait_tensor_439 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_199);  reduce_scatter_tensor_199 = None
        convert_element_type_1350 = torch.ops.prims.convert_element_type.default(wait_tensor_439, torch.float32);  wait_tensor_439 = None
        convert_element_type_1352 = torch.ops.prims.convert_element_type.default(wait_tensor_15, torch.float32);  wait_tensor_15 = None
        mul_430 = torch.ops.aten.mul.Tensor(convert_element_type_1350, convert_element_type_1352);  convert_element_type_1352 = None
        mul_432 = torch.ops.aten.mul.Tensor(mul_8, mul_430)
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
        div_30 = torch.ops.aten.div.Tensor(mul_8, 4096)
        mul_433 = torch.ops.aten.mul.Tensor(div_30, sum_91);  div_30 = sum_91 = None
        sub_46 = torch.ops.aten.sub.Tensor(mul_430, mul_433);  mul_430 = mul_433 = None
        mul_434 = torch.ops.aten.mul.Tensor(sub_46, rsqrt_2);  sub_46 = rsqrt_2 = None
        mul_435 = torch.ops.aten.mul.Tensor(convert_element_type_1350, mul_8);  convert_element_type_1350 = mul_8 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
        convert_element_type_1353 = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        convert_element_type_1354 = torch.ops.prims.convert_element_type.default(sum_92, torch.bfloat16);  sum_92 = None
        all_reduce_30 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1354, 'sum', '1');  convert_element_type_1354 = None
        wait_tensor_440 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_30);  all_reduce_30 = None
        convert_element_type_1355 = torch.ops.prims.convert_element_type.default(wait_tensor_440, torch.float32);  wait_tensor_440 = None
        reduce_scatter_tensor_200 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1355, 'avg', 8, '0');  convert_element_type_1355 = None
        wait_tensor_441 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_200);  reduce_scatter_tensor_200 = None
        add_169 = torch.ops.aten.add.Tensor(add_166, convert_element_type_1353);  add_166 = convert_element_type_1353 = None
        all_gather_into_tensor_210 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_169, 8, '1')
        wait_tensor_442 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_210);  all_gather_into_tensor_210 = None
        split_135 = torch.ops.aten.split.Tensor(wait_tensor_442, 2);  wait_tensor_442 = None
        getitem_1269 = split_135[0]
        getitem_1270 = split_135[1]
        getitem_1271 = split_135[2]
        getitem_1272 = split_135[3]
        getitem_1273 = split_135[4]
        getitem_1274 = split_135[5]
        getitem_1275 = split_135[6]
        getitem_1276 = split_135[7];  split_135 = None
        cat_127 = torch.ops.aten.cat.default([getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276], 1);  getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = None
        view_1531 = torch.ops.aten.view.default(cat_127, [16384, 4096]);  cat_127 = None
        permute_661 = torch.ops.aten.permute.default(view_1531, [1, 0])
        wait_tensor_8 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        add_1 = torch.ops.aten.add.Tensor(wait_tensor_1, wait_tensor_8);  wait_tensor_8 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        all_gather_into_tensor_7 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_20, 8, '0');  convert_element_type_20 = None
        wait_tensor_9 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [2], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Scalar(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_1);  convert_element_type_21 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, wait_tensor_9)
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
        view_60 = torch.ops.aten.view.default(cat_3, [16384, 4096]);  cat_3 = None
        view_61 = torch.ops.aten.view.default(mm_4, [2, 8192, 1792]);  mm_4 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_26)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_26, sigmoid);  sigmoid = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        all_gather_into_tensor_10 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_28, 8, '0');  convert_element_type_28 = None
        wait_tensor_12 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        permute_9 = torch.ops.aten.permute.default(wait_tensor_12, [1, 0]);  wait_tensor_12 = None
        mm_5 = torch.ops.aten.mm.default(view_60, permute_9)
        view_68 = torch.ops.aten.view.default(mm_5, [2, 8192, 1792]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_27, view_68)
        view_75 = torch.ops.aten.view.default(mul_7, [16384, 1792]);  mul_7 = None
        mm_325 = torch.ops.aten.mm.default(permute_661, view_75);  permute_661 = view_75 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        all_gather_into_tensor_11 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_31, 8, '0');  convert_element_type_31 = None
        wait_tensor_13 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        permute_10 = torch.ops.aten.permute.default(wait_tensor_13, [1, 0]);  wait_tensor_13 = None
        permute_663 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_326 = torch.ops.aten.mm.default(view_1531, permute_663);  view_1531 = permute_663 = None
        view_1532 = torch.ops.aten.view.default(mm_326, [2, 8192, 1792]);  mm_326 = None
        convert_element_type_1360 = torch.ops.prims.convert_element_type.default(mm_325, torch.float32);  mm_325 = None
        reduce_scatter_tensor_201 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1360, 'avg', 8, '0');  convert_element_type_1360 = None
        wait_tensor_443 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_201);  reduce_scatter_tensor_201 = None
        mul_436 = torch.ops.aten.mul.Tensor(view_1532, convert_element_type_27);  convert_element_type_27 = None
        mul_437 = torch.ops.aten.mul.Tensor(view_1532, view_68);  view_1532 = view_68 = None
        view_1533 = torch.ops.aten.view.default(mul_436, [16384, 1792]);  mul_436 = None
        permute_665 = torch.ops.aten.permute.default(view_1533, [1, 0])
        mm_327 = torch.ops.aten.mm.default(permute_665, view_60);  permute_665 = None
        permute_667 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_328 = torch.ops.aten.mm.default(view_1533, permute_667);  view_1533 = permute_667 = None
        view_1534 = torch.ops.aten.view.default(mm_328, [2, 8192, 4096]);  mm_328 = None
        convert_element_type_1365 = torch.ops.prims.convert_element_type.default(mm_327, torch.float32);  mm_327 = None
        reduce_scatter_tensor_202 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1365, 'avg', 8, '0');  convert_element_type_1365 = None
        wait_tensor_444 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_202);  reduce_scatter_tensor_202 = None
        convert_element_type_1366 = torch.ops.prims.convert_element_type.default(mul_437, torch.float32);  mul_437 = None
        neg_15 = torch.ops.aten.neg.default(convert_element_type_26)
        exp_15 = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_170 = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(add_170);  add_170 = None
        mul_438 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_439 = torch.ops.aten.mul.Tensor(convert_element_type_1366, mul_438);  convert_element_type_1366 = None
        sub_47 = torch.ops.aten.sub.Tensor(1, mul_438);  mul_438 = None
        mul_440 = torch.ops.aten.mul.Tensor(convert_element_type_26, sub_47);  convert_element_type_26 = sub_47 = None
        add_171 = torch.ops.aten.add.Tensor(mul_440, 1);  mul_440 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_171);  mul_439 = add_171 = None
        convert_element_type_1368 = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None
        view_1535 = torch.ops.aten.view.default(convert_element_type_1368, [16384, 1792]);  convert_element_type_1368 = None
        permute_669 = torch.ops.aten.permute.default(view_1535, [1, 0])
        mm_329 = torch.ops.aten.mm.default(permute_669, view_60);  permute_669 = view_60 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        all_gather_into_tensor_9 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_23, 8, '0');  convert_element_type_23 = None
        wait_tensor_11 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        permute_8 = torch.ops.aten.permute.default(wait_tensor_11, [1, 0]);  wait_tensor_11 = None
        permute_671 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_330 = torch.ops.aten.mm.default(view_1535, permute_671);  view_1535 = permute_671 = None
        view_1536 = torch.ops.aten.view.default(mm_330, [2, 8192, 4096]);  mm_330 = None
        add_172 = torch.ops.aten.add.Tensor(view_1534, view_1536);  view_1534 = view_1536 = None
        convert_element_type_1373 = torch.ops.prims.convert_element_type.default(mm_329, torch.float32);  mm_329 = None
        reduce_scatter_tensor_203 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1373, 'avg', 8, '0');  convert_element_type_1373 = None
        wait_tensor_445 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_203);  reduce_scatter_tensor_203 = None
        split_136 = torch.ops.aten.split.Tensor(add_172, 1024, 1);  add_172 = None
        getitem_1277 = split_136[0]
        getitem_1278 = split_136[1]
        getitem_1279 = split_136[2]
        getitem_1280 = split_136[3]
        getitem_1281 = split_136[4]
        getitem_1282 = split_136[5]
        getitem_1283 = split_136[6]
        getitem_1284 = split_136[7];  split_136 = None
        cat_128 = torch.ops.aten.cat.default([getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284]);  getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = None
        reduce_scatter_tensor_204 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_128, 'sum', 8, '1');  cat_128 = None
        wait_tensor_446 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_204);  reduce_scatter_tensor_204 = None
        convert_element_type_1374 = torch.ops.prims.convert_element_type.default(wait_tensor_446, torch.float32);  wait_tensor_446 = None
        convert_element_type_1376 = torch.ops.prims.convert_element_type.default(wait_tensor_9, torch.float32);  wait_tensor_9 = None
        mul_442 = torch.ops.aten.mul.Tensor(convert_element_type_1374, convert_element_type_1376);  convert_element_type_1376 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_4, mul_442)
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
        div_31 = torch.ops.aten.div.Tensor(mul_4, 4096)
        mul_445 = torch.ops.aten.mul.Tensor(div_31, sum_93);  div_31 = sum_93 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_442, mul_445);  mul_442 = mul_445 = None
        mul_446 = torch.ops.aten.mul.Tensor(sub_48, rsqrt_1);  sub_48 = rsqrt_1 = None
        mul_447 = torch.ops.aten.mul.Tensor(convert_element_type_1374, mul_4);  convert_element_type_1374 = mul_4 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
        convert_element_type_1377 = torch.ops.prims.convert_element_type.default(mul_446, torch.bfloat16);  mul_446 = None
        convert_element_type_1378 = torch.ops.prims.convert_element_type.default(sum_94, torch.bfloat16);  sum_94 = None
        all_reduce_31 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1378, 'sum', '1');  convert_element_type_1378 = None
        wait_tensor_447 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_31);  all_reduce_31 = None
        convert_element_type_1379 = torch.ops.prims.convert_element_type.default(wait_tensor_447, torch.float32);  wait_tensor_447 = None
        reduce_scatter_tensor_205 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1379, 'avg', 8, '0');  convert_element_type_1379 = None
        wait_tensor_448 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_205);  reduce_scatter_tensor_205 = None
        add_173 = torch.ops.aten.add.Tensor(add_169, convert_element_type_1377);  add_169 = convert_element_type_1377 = None
        all_gather_into_tensor_211 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_173, 8, '1')
        wait_tensor_449 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_211);  all_gather_into_tensor_211 = None
        split_137 = torch.ops.aten.split.Tensor(wait_tensor_449, 2);  wait_tensor_449 = None
        getitem_1285 = split_137[0]
        getitem_1286 = split_137[1]
        getitem_1287 = split_137[2]
        getitem_1288 = split_137[3]
        getitem_1289 = split_137[4]
        getitem_1290 = split_137[5]
        getitem_1291 = split_137[6]
        getitem_1292 = split_137[7];  split_137 = None
        cat_129 = torch.ops.aten.cat.default([getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292], 1);  getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = None
        view_1537 = torch.ops.aten.view.default(cat_129, [16384, 4096]);  cat_129 = None
        permute_673 = torch.ops.aten.permute.default(view_1537, [1, 0])
        permute_6 = torch.ops.aten.permute.default(getitem_80, [0, 2, 1, 3])
        view_42 = torch.ops.aten.view.default(permute_6, [2, 8192, -1]);  permute_6 = None
        view_48 = torch.ops.aten.view.default(view_42, [16384, 512]);  view_42 = None
        mm_331 = torch.ops.aten.mm.default(permute_673, view_48);  permute_673 = view_48 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        all_gather_into_tensor_6 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_17, 8, '0');  convert_element_type_17 = None
        wait_tensor_7 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        permute_7 = torch.ops.aten.permute.default(wait_tensor_7, [1, 0]);  wait_tensor_7 = None
        permute_675 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_332 = torch.ops.aten.mm.default(view_1537, permute_675);  view_1537 = permute_675 = None
        view_1538 = torch.ops.aten.view.default(mm_332, [2, 8192, 512]);  mm_332 = None
        convert_element_type_1384 = torch.ops.prims.convert_element_type.default(mm_331, torch.float32);  mm_331 = None
        reduce_scatter_tensor_206 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1384, 'avg', 8, '0');  convert_element_type_1384 = None
        wait_tensor_450 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_206);  reduce_scatter_tensor_206 = None
        view_1539 = torch.ops.aten.view.default(view_1538, [2, 8192, 4, 128]);  view_1538 = None
        permute_677 = torch.ops.aten.permute.default(view_1539, [0, 2, 1, 3]);  view_1539 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_1, 8, '0');  convert_element_type_1 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(wait_tensor_1, torch.float32);  wait_tensor_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [2], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, wait_tensor_2)
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
        view_15 = torch.ops.aten.view.default(cat_1, [16384, 4096]);  cat_1 = None
        view_16 = torch.ops.aten.view.default(mm, [2, 8192, 512]);  mm = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        all_gather_into_tensor_4 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_7, 8, '0');  convert_element_type_7 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        permute_1 = torch.ops.aten.permute.default(wait_tensor_5, [1, 0]);  wait_tensor_5 = None
        mm_1 = torch.ops.aten.mm.default(view_15, permute_1)
        view_23 = torch.ops.aten.view.default(mm_1, [2, 8192, 128]);  mm_1 = None
        view_30 = torch.ops.aten.view.default(mm_2, [2, 8192, 128]);  mm_2 = None
        view_32 = torch.ops.aten.view.default(view_16, [2, 8192, -1, 128]);  view_16 = None
        view_33 = torch.ops.aten.view.default(view_23, [2, 8192, -1, 128]);  view_23 = None
        view_34 = torch.ops.aten.view.default(view_30, [2, 8192, -1, 128]);  view_30 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        view_35 = torch.ops.aten.view.default(convert_element_type_13, [2, 8192, 4, -1, 2]);  convert_element_type_13 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_35);  view_35 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_14, [2, 8192, 1, -1, 2]);  convert_element_type_14 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_36);  view_36 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_37);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_38 = torch.ops.aten.view.default(view_as_real, [2, 8192, 4, 128]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_37);  view_as_complex_1 = view_37 = None
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
        _scaled_dot_product_cudnn_attention_backward_15 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_677, permute_3, permute_4, permute_5, getitem_80, getitem_81, getitem_86, getitem_87, None, None, None, 8192, 8192, 0.0, True);  permute_677 = permute_3 = permute_4 = permute_5 = getitem_80 = getitem_81 = getitem_86 = getitem_87 = None
        getitem_1293 = _scaled_dot_product_cudnn_attention_backward_15[0]
        getitem_1294 = _scaled_dot_product_cudnn_attention_backward_15[1]
        getitem_1295 = _scaled_dot_product_cudnn_attention_backward_15[2];  _scaled_dot_product_cudnn_attention_backward_15 = None
        permute_678 = torch.ops.aten.permute.default(getitem_1295, [0, 2, 1, 3]);  getitem_1295 = None
        permute_679 = torch.ops.aten.permute.default(getitem_1294, [0, 2, 1, 3]);  getitem_1294 = None
        permute_680 = torch.ops.aten.permute.default(getitem_1293, [0, 2, 1, 3]);  getitem_1293 = None
        view_1540 = torch.ops.aten.view.default(permute_678, [2, 8192, 1, 4, 128]);  permute_678 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(view_1540, [3], True);  view_1540 = None
        squeeze_30 = torch.ops.aten.squeeze.dim(sum_95, 3);  sum_95 = None
        view_1541 = torch.ops.aten.view.default(permute_679, [2, 8192, 1, 4, 128]);  permute_679 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(view_1541, [3], True);  view_1541 = None
        squeeze_31 = torch.ops.aten.squeeze.dim(sum_96, 3);  sum_96 = None
        convert_element_type_1385 = torch.ops.prims.convert_element_type.default(squeeze_31, torch.float32);  squeeze_31 = None
        convert_element_type_1386 = torch.ops.prims.convert_element_type.default(permute_680, torch.float32);  permute_680 = None
        view_1542 = torch.ops.aten.view.default(convert_element_type_1385, [2, 8192, 1, 64, 2]);  convert_element_type_1385 = None
        view_as_complex_62 = torch.ops.aten.view_as_complex.default(view_1542);  view_1542 = None
        mul_448 = torch.ops.aten.mul.Tensor(view_as_complex_62, _conj);  view_as_complex_62 = None
        view_1543 = torch.ops.aten.view.default(convert_element_type_1386, [2, 8192, 4, 64, 2]);  convert_element_type_1386 = None
        view_as_complex_63 = torch.ops.aten.view_as_complex.default(view_1543);  view_1543 = None
        mul_449 = torch.ops.aten.mul.Tensor(view_as_complex_63, _conj);  view_as_complex_63 = _conj = None
        view_as_real_62 = torch.ops.aten.view_as_real.default(mul_448);  mul_448 = None
        view_1544 = torch.ops.aten.view.default(view_as_real_62, [2, 8192, 1, 128]);  view_as_real_62 = None
        convert_element_type_1387 = torch.ops.prims.convert_element_type.default(view_1544, torch.bfloat16);  view_1544 = None
        view_as_real_63 = torch.ops.aten.view_as_real.default(mul_449);  mul_449 = None
        view_1545 = torch.ops.aten.view.default(view_as_real_63, [2, 8192, 4, 128]);  view_as_real_63 = None
        convert_element_type_1388 = torch.ops.prims.convert_element_type.default(view_1545, torch.bfloat16);  view_1545 = None
        view_1546 = torch.ops.aten.view.default(squeeze_30, [2, 8192, 128]);  squeeze_30 = None
        view_1547 = torch.ops.aten.view.default(convert_element_type_1387, [2, 8192, 128]);  convert_element_type_1387 = None
        view_1548 = torch.ops.aten.view.default(convert_element_type_1388, [2, 8192, 512]);  convert_element_type_1388 = None
        view_1549 = torch.ops.aten.view.default(view_1546, [16384, 128]);  view_1546 = None
        permute_681 = torch.ops.aten.permute.default(view_1549, [1, 0])
        mm_333 = torch.ops.aten.mm.default(permute_681, view_15);  permute_681 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        all_gather_into_tensor_5 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_10, 8, '0');  convert_element_type_10 = None
        wait_tensor_6 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        permute_2 = torch.ops.aten.permute.default(wait_tensor_6, [1, 0]);  wait_tensor_6 = None
        permute_683 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_334 = torch.ops.aten.mm.default(view_1549, permute_683);  view_1549 = permute_683 = None
        view_1550 = torch.ops.aten.view.default(mm_334, [2, 8192, 4096]);  mm_334 = None
        convert_element_type_1393 = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None
        reduce_scatter_tensor_207 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1393, 'avg', 8, '0');  convert_element_type_1393 = None
        wait_tensor_451 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_207);  reduce_scatter_tensor_207 = None
        view_1551 = torch.ops.aten.view.default(view_1547, [16384, 128]);  view_1547 = None
        permute_685 = torch.ops.aten.permute.default(view_1551, [1, 0])
        mm_335 = torch.ops.aten.mm.default(permute_685, view_15);  permute_685 = None
        permute_687 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_336 = torch.ops.aten.mm.default(view_1551, permute_687);  view_1551 = permute_687 = None
        view_1552 = torch.ops.aten.view.default(mm_336, [2, 8192, 4096]);  mm_336 = None
        add_174 = torch.ops.aten.add.Tensor(view_1550, view_1552);  view_1550 = view_1552 = None
        convert_element_type_1398 = torch.ops.prims.convert_element_type.default(mm_335, torch.float32);  mm_335 = None
        reduce_scatter_tensor_208 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1398, 'avg', 8, '0');  convert_element_type_1398 = None
        wait_tensor_452 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_208);  reduce_scatter_tensor_208 = None
        view_1553 = torch.ops.aten.view.default(view_1548, [16384, 512]);  view_1548 = None
        permute_689 = torch.ops.aten.permute.default(view_1553, [1, 0])
        mm_337 = torch.ops.aten.mm.default(permute_689, view_15);  permute_689 = view_15 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        all_gather_into_tensor_3 = torch.ops._c10d_functional.all_gather_into_tensor.default(convert_element_type_4, 8, '0');  convert_element_type_4 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        permute = torch.ops.aten.permute.default(wait_tensor_4, [1, 0]);  wait_tensor_4 = None
        permute_691 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_338 = torch.ops.aten.mm.default(view_1553, permute_691);  view_1553 = permute_691 = None
        view_1554 = torch.ops.aten.view.default(mm_338, [2, 8192, 4096]);  mm_338 = None
        add_175 = torch.ops.aten.add.Tensor(add_174, view_1554);  add_174 = view_1554 = None
        convert_element_type_1403 = torch.ops.prims.convert_element_type.default(mm_337, torch.float32);  mm_337 = None
        reduce_scatter_tensor_209 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1403, 'avg', 8, '0');  convert_element_type_1403 = None
        wait_tensor_453 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_209);  reduce_scatter_tensor_209 = None
        split_138 = torch.ops.aten.split.Tensor(add_175, 1024, 1);  add_175 = None
        getitem_1296 = split_138[0]
        getitem_1297 = split_138[1]
        getitem_1298 = split_138[2]
        getitem_1299 = split_138[3]
        getitem_1300 = split_138[4]
        getitem_1301 = split_138[5]
        getitem_1302 = split_138[6]
        getitem_1303 = split_138[7];  split_138 = None
        cat_130 = torch.ops.aten.cat.default([getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303]);  getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = None
        reduce_scatter_tensor_210 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_130, 'sum', 8, '1');  cat_130 = None
        wait_tensor_454 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_210);  reduce_scatter_tensor_210 = None
        convert_element_type_1404 = torch.ops.prims.convert_element_type.default(wait_tensor_454, torch.float32);  wait_tensor_454 = None
        convert_element_type_1406 = torch.ops.prims.convert_element_type.default(wait_tensor_2, torch.float32);  wait_tensor_2 = None
        mul_450 = torch.ops.aten.mul.Tensor(convert_element_type_1404, convert_element_type_1406);  convert_element_type_1406 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul, mul_450)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_452, [2], True);  mul_452 = None
        div_32 = torch.ops.aten.div.Tensor(mul, 4096)
        mul_453 = torch.ops.aten.mul.Tensor(div_32, sum_97);  div_32 = sum_97 = None
        sub_49 = torch.ops.aten.sub.Tensor(mul_450, mul_453);  mul_450 = mul_453 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_49, rsqrt);  sub_49 = rsqrt = None
        mul_455 = torch.ops.aten.mul.Tensor(convert_element_type_1404, mul);  convert_element_type_1404 = mul = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1]);  mul_455 = None
        convert_element_type_1407 = torch.ops.prims.convert_element_type.default(mul_454, torch.bfloat16);  mul_454 = None
        convert_element_type_1408 = torch.ops.prims.convert_element_type.default(sum_98, torch.bfloat16);  sum_98 = None
        all_reduce_32 = torch.ops._c10d_functional.all_reduce.default(convert_element_type_1408, 'sum', '1');  convert_element_type_1408 = None
        wait_tensor_455 = torch.ops._c10d_functional.wait_tensor.default(all_reduce_32);  all_reduce_32 = None
        convert_element_type_1409 = torch.ops.prims.convert_element_type.default(wait_tensor_455, torch.float32);  wait_tensor_455 = None
        reduce_scatter_tensor_211 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1409, 'avg', 8, '0');  convert_element_type_1409 = None
        wait_tensor_456 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_211);  reduce_scatter_tensor_211 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, convert_element_type_1407);  add_173 = convert_element_type_1407 = None
        all_gather_into_tensor_212 = torch.ops._c10d_functional.all_gather_into_tensor.default(add_176, 8, '1');  add_176 = None
        wait_tensor_457 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_212);  all_gather_into_tensor_212 = None
        split_139 = torch.ops.aten.split.Tensor(wait_tensor_457, 2);  wait_tensor_457 = None
        getitem_1304 = split_139[0]
        getitem_1305 = split_139[1]
        getitem_1306 = split_139[2]
        getitem_1307 = split_139[3]
        getitem_1308 = split_139[4]
        getitem_1309 = split_139[5]
        getitem_1310 = split_139[6]
        getitem_1311 = split_139[7];  split_139 = None
        cat_131 = torch.ops.aten.cat.default([getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311], 1);  getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = getitem_1310 = getitem_1311 = None
        convert_element_type_1410 = torch.ops.prims.convert_element_type.default(cat_131, torch.float32);  cat_131 = None
        eq = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(unsqueeze_32, full_default_2, convert_element_type_1410);  unsqueeze_32 = full_default_2 = convert_element_type_1410 = None
        full_default_3 = torch.ops.aten.full.default([128256, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2 = torch.ops.aten.index_put.default(full_default_3, [primals_1], where, True);  full_default_3 = primals_1 = where = None
        convert_element_type_1411 = torch.ops.prims.convert_element_type.default(index_put_2, torch.bfloat16);  index_put_2 = None
        split_140 = torch.ops.aten.split.Tensor(convert_element_type_1411, 16032);  convert_element_type_1411 = None
        getitem_1312 = split_140[0];  split_140 = None
        convert_element_type_1412 = torch.ops.prims.convert_element_type.default(getitem_1312, torch.float32);  getitem_1312 = None
        reduce_scatter_tensor_212 = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_1412, 'avg', 8, '0');  convert_element_type_1412 = None
        wait_tensor_458 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_212);  reduce_scatter_tensor_212 = None
        return (None, wait_tensor_458, None, wait_tensor_456, wait_tensor_453, wait_tensor_452, wait_tensor_451, wait_tensor_450, wait_tensor_448, wait_tensor_445, wait_tensor_444, wait_tensor_443, wait_tensor_441, wait_tensor_438, wait_tensor_437, wait_tensor_436, wait_tensor_435, wait_tensor_433, wait_tensor_430, wait_tensor_429, wait_tensor_428, wait_tensor_426, wait_tensor_423, wait_tensor_422, wait_tensor_421, wait_tensor_420, wait_tensor_418, wait_tensor_415, wait_tensor_414, wait_tensor_413, wait_tensor_411, wait_tensor_408, wait_tensor_407, wait_tensor_406, wait_tensor_405, wait_tensor_403, wait_tensor_400, wait_tensor_399, wait_tensor_398, wait_tensor_396, wait_tensor_393, wait_tensor_392, wait_tensor_391, wait_tensor_390, wait_tensor_388, wait_tensor_385, wait_tensor_384, wait_tensor_383, wait_tensor_381, wait_tensor_378, wait_tensor_377, wait_tensor_376, wait_tensor_375, wait_tensor_373, wait_tensor_370, wait_tensor_369, wait_tensor_368, wait_tensor_366, wait_tensor_363, wait_tensor_362, wait_tensor_361, wait_tensor_360, wait_tensor_358, wait_tensor_355, wait_tensor_354, wait_tensor_353, wait_tensor_351, wait_tensor_348, wait_tensor_347, wait_tensor_346, wait_tensor_345, wait_tensor_343, wait_tensor_340, wait_tensor_339, wait_tensor_338, wait_tensor_336, wait_tensor_333, wait_tensor_332, wait_tensor_331, wait_tensor_330, wait_tensor_328, wait_tensor_325, wait_tensor_324, wait_tensor_323, wait_tensor_321, wait_tensor_318, wait_tensor_317, wait_tensor_316, wait_tensor_315, wait_tensor_313, wait_tensor_310, wait_tensor_309, wait_tensor_308, wait_tensor_306, wait_tensor_303, wait_tensor_302, wait_tensor_301, wait_tensor_300, wait_tensor_298, wait_tensor_295, wait_tensor_294, wait_tensor_293, wait_tensor_291, wait_tensor_288, wait_tensor_287, wait_tensor_286, wait_tensor_285, wait_tensor_283, wait_tensor_280, wait_tensor_279, wait_tensor_278, wait_tensor_276, wait_tensor_273, wait_tensor_272, wait_tensor_271, wait_tensor_270, wait_tensor_268, wait_tensor_265, wait_tensor_264, wait_tensor_263, wait_tensor_261, wait_tensor_258, wait_tensor_257, wait_tensor_256, wait_tensor_255, wait_tensor_253, wait_tensor_250, wait_tensor_249, wait_tensor_248, wait_tensor_246, wait_tensor_243, wait_tensor_242, wait_tensor_241, wait_tensor_240, wait_tensor_238, wait_tensor_235, wait_tensor_234, wait_tensor_233, wait_tensor_231, wait_tensor_228, wait_tensor_227, wait_tensor_226, wait_tensor_225, wait_tensor_223, wait_tensor_220, wait_tensor_219, wait_tensor_218, wait_tensor_216, wait_tensor_213)

def load_args(reader):
    buf0 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 8192), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 32833536, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2004, 4096), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf2, (8192, 64), dtype=torch.complex64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf3, (512,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64, 4096), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16, 4096), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf6, (16, 4096), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512, 512), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf8, (512,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf9, (224, 4096), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf10, (224, 4096), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf11, (512, 1792), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf12, (512,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64, 4096), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16, 4096), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16, 4096), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512, 512), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf18, (224, 4096), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf19, (224, 4096), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512, 1792), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf21, (512,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64, 4096), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf23, (16, 4096), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (16, 4096), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf25, (512, 512), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (512,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf27, (224, 4096), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf28, (224, 4096), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf29, (512, 1792), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 4096), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf32, (16, 4096), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf33, (16, 4096), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512, 512), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf36, (224, 4096), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf37, (224, 4096), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512, 1792), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64, 4096), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (16, 4096), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf42, (16, 4096), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512, 512), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf45, (224, 4096), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf46, (224, 4096), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512, 1792), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64, 4096), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf50, (16, 4096), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf51, (16, 4096), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512, 512), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf54, (224, 4096), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf55, (224, 4096), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512, 1792), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512,), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf58, (64, 4096), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf59, (16, 4096), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf60, (16, 4096), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512, 512), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf63, (224, 4096), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf64, (224, 4096), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512, 1792), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (64, 4096), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf68, (16, 4096), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (16, 4096), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512, 512), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf72, (224, 4096), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf73, (224, 4096), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512, 1792), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf76, (64, 4096), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf77, (16, 4096), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf78, (16, 4096), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512, 512), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf81, (224, 4096), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf82, (224, 4096), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512, 1792), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (64, 4096), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf86, (16, 4096), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf87, (16, 4096), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512, 512), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf90, (224, 4096), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf91, (224, 4096), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf92, (512, 1792), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf93, (512,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf94, (64, 4096), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf95, (16, 4096), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf96, (16, 4096), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512, 512), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf99, (224, 4096), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf100, (224, 4096), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512, 1792), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64, 4096), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf104, (16, 4096), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf105, (16, 4096), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf106, (512, 512), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf107, (512,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf108, (224, 4096), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf109, (224, 4096), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf110, (512, 1792), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512,), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf112, (64, 4096), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf113, (16, 4096), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf114, (16, 4096), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512, 512), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf117, (224, 4096), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf118, (224, 4096), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512, 1792), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf121, (64, 4096), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf122, (16, 4096), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf123, (16, 4096), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf124, (512, 512), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf125, (512,), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf126, (224, 4096), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf127, (224, 4096), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf128, (512, 1792), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf130, (64, 4096), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf131, (16, 4096), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf132, (16, 4096), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512, 512), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf134, (512,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf135, (224, 4096), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf136, (224, 4096), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf137, (512, 1792), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf139, (64, 4096), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf140, (16, 4096), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (16, 4096), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf142, (512, 512), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf143, (512,), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf144, (224, 4096), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf145, (224, 4096), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 3670016, device=device(type='cuda', index=0))
    reader.tensor(buf146, (512, 1792), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf147, (512,), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 32833536, device=device(type='cuda', index=0))
    reader.tensor(buf148, (2004, 4096), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf149, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # wait_tensor_1
    buf150 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf150, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf151 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf151, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf152 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf152, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_80
    buf153 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (2, 4, 8192, 1), is_leaf=True)  # getitem_81
    buf154 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf154, (), dtype=torch.int64, is_leaf=True)  # getitem_86
    buf155 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf155, (), dtype=torch.int64, is_leaf=True)  # getitem_87
    buf156 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf156, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_1
    buf157 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf157, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf158 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf158, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_3
    buf159 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf159, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_7
    buf160 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf160, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_9
    buf161 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf161, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_121
    buf162 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf162, (2, 4, 8192, 1), is_leaf=True)  # getitem_122
    buf163 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf163, (), dtype=torch.int64, is_leaf=True)  # getitem_127
    buf164 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf164, (), dtype=torch.int64, is_leaf=True)  # getitem_128
    buf165 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf165, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_3
    buf166 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf166, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_11
    buf167 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf167, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_7
    buf168 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf168, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_14
    buf169 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf169, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_16
    buf170 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf170, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_162
    buf171 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf171, (2, 4, 8192, 1), is_leaf=True)  # getitem_163
    buf172 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf172, (), dtype=torch.int64, is_leaf=True)  # getitem_168
    buf173 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf173, (), dtype=torch.int64, is_leaf=True)  # getitem_169
    buf174 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf174, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_5
    buf175 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf175, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_18
    buf176 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf176, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_11
    buf177 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf177, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_21
    buf178 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf178, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_23
    buf179 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf179, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_203
    buf180 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf180, (2, 4, 8192, 1), is_leaf=True)  # getitem_204
    buf181 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf181, (), dtype=torch.int64, is_leaf=True)  # getitem_209
    buf182 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf182, (), dtype=torch.int64, is_leaf=True)  # getitem_210
    buf183 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf183, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_7
    buf184 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf184, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_25
    buf185 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf185, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_15
    buf186 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf186, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_28
    buf187 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf187, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_30
    buf188 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf188, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_244
    buf189 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf189, (2, 4, 8192, 1), is_leaf=True)  # getitem_245
    buf190 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf190, (), dtype=torch.int64, is_leaf=True)  # getitem_250
    buf191 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf191, (), dtype=torch.int64, is_leaf=True)  # getitem_251
    buf192 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf192, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_9
    buf193 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf193, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_32
    buf194 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf194, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_19
    buf195 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf195, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_35
    buf196 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf196, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_37
    buf197 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf197, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_285
    buf198 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf198, (2, 4, 8192, 1), is_leaf=True)  # getitem_286
    buf199 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf199, (), dtype=torch.int64, is_leaf=True)  # getitem_291
    buf200 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf200, (), dtype=torch.int64, is_leaf=True)  # getitem_292
    buf201 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf201, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_11
    buf202 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf202, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_39
    buf203 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf203, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_23
    buf204 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf204, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_42
    buf205 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf205, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_44
    buf206 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf206, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_326
    buf207 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf207, (2, 4, 8192, 1), is_leaf=True)  # getitem_327
    buf208 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf208, (), dtype=torch.int64, is_leaf=True)  # getitem_332
    buf209 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf209, (), dtype=torch.int64, is_leaf=True)  # getitem_333
    buf210 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf210, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_13
    buf211 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf211, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_46
    buf212 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf212, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_27
    buf213 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf213, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_49
    buf214 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf214, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_51
    buf215 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf215, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_367
    buf216 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf216, (2, 4, 8192, 1), is_leaf=True)  # getitem_368
    buf217 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf217, (), dtype=torch.int64, is_leaf=True)  # getitem_373
    buf218 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf218, (), dtype=torch.int64, is_leaf=True)  # getitem_374
    buf219 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf219, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_15
    buf220 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf220, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_53
    buf221 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf221, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_31
    buf222 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf222, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_56
    buf223 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf223, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_58
    buf224 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf224, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_408
    buf225 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf225, (2, 4, 8192, 1), is_leaf=True)  # getitem_409
    buf226 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf226, (), dtype=torch.int64, is_leaf=True)  # getitem_414
    buf227 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf227, (), dtype=torch.int64, is_leaf=True)  # getitem_415
    buf228 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf228, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_17
    buf229 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf229, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_60
    buf230 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf230, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_35
    buf231 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf231, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_63
    buf232 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf232, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_65
    buf233 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf233, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_449
    buf234 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf234, (2, 4, 8192, 1), is_leaf=True)  # getitem_450
    buf235 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf235, (), dtype=torch.int64, is_leaf=True)  # getitem_455
    buf236 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf236, (), dtype=torch.int64, is_leaf=True)  # getitem_456
    buf237 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf237, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_19
    buf238 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf238, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_67
    buf239 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf239, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_39
    buf240 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf240, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_70
    buf241 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf241, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_72
    buf242 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf242, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_490
    buf243 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf243, (2, 4, 8192, 1), is_leaf=True)  # getitem_491
    buf244 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf244, (), dtype=torch.int64, is_leaf=True)  # getitem_496
    buf245 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf245, (), dtype=torch.int64, is_leaf=True)  # getitem_497
    buf246 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf246, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_21
    buf247 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf247, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_74
    buf248 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf248, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_43
    buf249 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf249, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_77
    buf250 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf250, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_79
    buf251 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf251, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_531
    buf252 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf252, (2, 4, 8192, 1), is_leaf=True)  # getitem_532
    buf253 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf253, (), dtype=torch.int64, is_leaf=True)  # getitem_537
    buf254 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf254, (), dtype=torch.int64, is_leaf=True)  # getitem_538
    buf255 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf255, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_23
    buf256 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf256, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_81
    buf257 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf257, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_47
    buf258 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf258, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_84
    buf259 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf259, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_86
    buf260 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf260, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_572
    buf261 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf261, (2, 4, 8192, 1), is_leaf=True)  # getitem_573
    buf262 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf262, (), dtype=torch.int64, is_leaf=True)  # getitem_578
    buf263 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf263, (), dtype=torch.int64, is_leaf=True)  # getitem_579
    buf264 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf264, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_25
    buf265 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf265, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_88
    buf266 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf266, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_51
    buf267 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf267, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_91
    buf268 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf268, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_93
    buf269 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf269, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_613
    buf270 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf270, (2, 4, 8192, 1), is_leaf=True)  # getitem_614
    buf271 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf271, (), dtype=torch.int64, is_leaf=True)  # getitem_619
    buf272 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf272, (), dtype=torch.int64, is_leaf=True)  # getitem_620
    buf273 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf273, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_27
    buf274 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf274, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_95
    buf275 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf275, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_55
    buf276 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf276, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_98
    buf277 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf277, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_100
    buf278 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf278, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_654
    buf279 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf279, (2, 4, 8192, 1), is_leaf=True)  # getitem_655
    buf280 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf280, (), dtype=torch.int64, is_leaf=True)  # getitem_660
    buf281 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf281, (), dtype=torch.int64, is_leaf=True)  # getitem_661
    buf282 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf282, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_29
    buf283 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf283, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_102
    buf284 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf284, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # add_59
    buf285 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf285, (16384, 512), dtype=torch.bfloat16, is_leaf=True)  # mm_105
    buf286 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf286, (16384, 128), dtype=torch.bfloat16, is_leaf=True)  # mm_107
    buf287 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf287, (2, 4, 8192, 128), (4194304, 128, 512, 1), dtype=torch.bfloat16, is_leaf=True)  # getitem_695
    buf288 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf288, (2, 4, 8192, 1), is_leaf=True)  # getitem_696
    buf289 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf289, (), dtype=torch.int64, is_leaf=True)  # getitem_701
    buf290 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf290, (), dtype=torch.int64, is_leaf=True)  # getitem_702
    buf291 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf291, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_31
    buf292 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf292, (16384, 1792), dtype=torch.bfloat16, is_leaf=True)  # mm_109
    buf293 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf293, (2, 1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # reduce_scatter_tensor_32
    buf294 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf294, (2, 1024, 1), is_leaf=True)  # rsqrt_32
    buf295 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf295, (16384, 4096), dtype=torch.bfloat16, is_leaf=True)  # view_1167
    buf296 = reader.storage(None, 525336576, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf296, (2, 8192, 16032), dtype=torch.bfloat16, is_leaf=True)  # tangents_1

load_args._version = 0

def get_pg_config():
    return {'0': {'size': 8, 'rank': 0}, '1': {'size': 8, 'rank': 0}}

def get_colls_estimations_file():
    return "colls8_8.table"
