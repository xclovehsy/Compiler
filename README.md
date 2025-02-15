CompilerGym编译优化结果

benchmark://user-v0/20250219T194127-dc9f
优化前IR指令数量: 146
-adce 1
None None False {'action_had_no_effect': True, 'new_action_space': False}
-instcombine 53
None None False {'action_had_no_effect': False, 'new_action_space': False}
-simplifycfg 10
None None False {'action_had_no_effect': False, 'new_action_space': False}
-mem2reg 103
None None False {'action_had_no_effect': False, 'new_action_space': False}
优化后IR指令数量: 56
opt -adce -instcombine -simplifycfg -mem2reg input.bc -o output.bc