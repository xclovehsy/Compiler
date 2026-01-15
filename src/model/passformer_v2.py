from transformers import (
    EncoderDecoderModel,
    GPT2Decoder,
    GPT2Config,
    BertEncoder,
    BertConfig,
    BertTokenizer,
    GPT2Tokenizer,
    Seq2SeqModelOutput,
    BaseModelOutput
)
import torch
import torch.nn as nn


class GPT2DecoderWithAutophase(GPT2Decoder):
    def __init__(self, config):
        super().__init__(config)
        
        # 1. 定义静态特征投影层
        self.autophase_input_dim = 16
        self.autophase_proj = nn.Linear(
            in_features=self.autophase_input_dim,
            out_features=config.n_embd  # GPT-2中隐藏层维度为n_embd（对应BERT的hidden_size）
        )
        
        # 2. 定义dropout层（与GPT-2配置对齐，防止过拟合）
        self.autophase_dropout = nn.Dropout(config.embd_pdrop)
    
    def forward(
        self,
        hidden_states,  # GPT-2解码器嵌入层输出的词嵌入特征 (batch_size, dec_seq_len, n_embd)
        autophase=None,  # 新增：静态特征输入 (batch_size, static_feature_input_dim)
        attention_mask=None,
        encoder_hidden_states=None,  # 编码器输出的上下文特征（交叉注意力层使用）
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 3. 静态特征融合逻辑（嵌入层后、第一个Transformer层之前，与BERT解码器逻辑一致）
        if autophase is not None:
            autophase_proj = self.autophase_proj(autophase)
            autophase_proj = self.autophase_dropout(autophase_proj)
            # c. 广播匹配解码序列长度：(batch_size, n_embd) -> (batch_size, dec_seq_len, n_embd)
            # 确保每个解码token都能获得静态特征的约束（GPT自回归生成时，每一步seq_len=1，逻辑依然有效）
            dec_seq_len = hidden_states.shape[1]
            autophase_proj_broadcast = autophase_proj.unsqueeze(1).repeat(1, dec_seq_len, 1)
            
            # d. 特征融合（优先选择「逐元素相加」，稳定高效，适配GPT的特征分布）
            hidden_states = hidden_states + autophase_proj_broadcast

        # 4. 调用父类forward方法，继续GPT-2解码器后续的计算（自注意力、交叉注意力等）
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class Passformer(EncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):
        super().__init__(config, encoder, decoder)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        autophase=None,  # 新增：传递给GPT-2解码器的静态特征
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 统一配置默认参数（与原生模型对齐）
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # 第一步：编码器前向传播（BERT编码器，生成上下文特征，与原生逻辑一致）
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # 第二步：GPT-2解码器输入处理（提取解码器词嵌入）
        if decoder_input_ids is not None:
            decoder_embeds = self.decoder.embeddings(input_ids=decoder_input_ids)
        else:
            decoder_embeds = None
        
        # 第三步：自定义GPT-2解码器前向传播（传递静态特征）
        decoder_outputs = self.decoder(
            hidden_states=decoder_embeds,  # GPT-2词嵌入作为输入
            static_features=autophase,  # 传入静态特征
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,  # 接收BERT编码器输出
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 第四步：处理输出（与原生Seq2Seq模型对齐，生成最终logits）
        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        return Seq2SeqModelOutput(
            logits=decoder_outputs.logits if hasattr(decoder_outputs, 'logits') else None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

if __name__ == "__main__":
    # 测试Passformer模型的前向传播
    bert_encoder = BertEncoder(BertConfig())
    gpt2_decoder = GPT2DecoderWithAutophase(GPT2Config())
    model = Passformer(encoder=bert_encoder, decoder=gpt2_decoder)
    
    input_ids = torch.randint(0, 1000, (2, 10))  # 示例输入
    decoder_input_ids = torch.randint(0, 1000, (2, 5))  # 示例解码器输入
    autophase = torch.randn(2, 16)  # 示例静态特征输入
    
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        autophase=autophase
    )
    
    print("Logits shape:", outputs.logits.shape)

    
    # 1. 加载配置与分词器
    # bert_config = BertConfig.from_pretrained("bert-base-uncased")
    # gpt2_config = GPT2Config.from_pretrained("gpt2")
    # # 关键：启用GPT2解码器的交叉注意力（使其能接收编码器输出）
    # gpt2_config.add_cross_attention = True
    # gpt2_config.is_decoder = True

    # bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # # GPT2Tokenizer默认无pad_token，手动设置（避免批量处理报错）
    # gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # # 2. 初始化BERT编码器和自定义GPT-2解码器
    # bert_encoder = BertEncoder(bert_config)
    # custom_gpt2_decoder = GPT2DecoderWithStaticFeature(gpt2_config)

    # # 3. 初始化自定义EncoderDecoderModel
    # model = EncoderDecoderWithGPT2StaticDecoder(
    #     encoder=bert_encoder,
    #     decoder=custom_gpt2_decoder
    # )
    # # 加载预训练权重（BERT编码器+GPT-2解码器）
    # model = model.from_encoder_decoder_pretrained(
    #     "bert-base-uncased",
    #     "gpt2",
    #     encoder=bert_encoder,
    #     decoder=custom_gpt2_decoder
    # )

    # # 4. 准备输入数据和静态特征
    # text = "This is a sample text for GPT2 decoder static feature fusion."
    # # 编码器输入（BERT分词）
    # encoder_inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # # 解码器输入（GPT-2分词）
    # decoder_inputs = gpt2_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # # 构造静态特征（batch_size=1, static_feature_input_dim=16，与自定义解码器配置一致）
    # static_features = torch.randn(1, 16)

    # # 5. 模型前向传播（传入解码器静态特征）
    # outputs = model(
    #     input_ids=encoder_inputs["input_ids"],
    #     attention_mask=encoder_inputs["attention_mask"],
    #     decoder_input_ids=decoder_inputs["input_ids"],
    #     decoder_attention_mask=decoder_inputs["attention_mask"],
    #     static_features_dec=static_features
    # )

    # # 6. 验证输出
    # print("GPT-2解码器输出logits形状：", outputs.logits.shape)
    # print("模型前向传播成功，静态特征已融入GPT-2解码器！")