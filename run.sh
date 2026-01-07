# test inst2vec tokenizer
# python -m src.tests.test_inst2vec_tokenizer

# python -m src.llvm_opt_gen_train --config yaml/llvm_opt_gen_train.yaml
# python -m src.inst2vec_bert_mlm_train --config /home/xucong24/Compiler/yaml/inst2vec_poj104_modernbert_train.yaml

# inst2vec_modernbert poj104 classify 
# python -m src.experiments.modernbert_classifyapp_inst2vec

# inst2vec_modernbert poj104 mlm train
# python -m src.training.instbert_mlm_trainer --config /home/xucong24/Compiler/configs/instbert_poj104_mlm.yaml

python -m src.training.optseq_gen_train --config /home/xucong24/Compiler/configs/optseq_gen_train.yaml