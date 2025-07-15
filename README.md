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

安装clang以及工具链
sudo apt update
sudo apt install llvm clang

macos
brew install llvm


ProGraML依赖库
安装 nlohmann/json.hpp
sudo apt update
sudo apt install nlohmann-json3-dev




g++ -o compute_autophase compute_autophase.cc InstCount.cc \
    $(llvm-config --cxxflags --ldflags --system-libs --libs core irreader support analysis transformutils) \
    -fno-rtti -std=c++14

g++ -o compute_ir_instruction_count_mac compute_ir_instruction_count.cc InstCount.cc \
    $(llvm-config --cxxflags --ldflags --system-libs --libs core irreader support analysis transformutils) \
    -fno-rtti -std=c++17

sudo apt update
sudo apt install aria2
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh bert-base-uncased

modernbert
https://huggingface.co/answerdotai/ModernBERT-base
