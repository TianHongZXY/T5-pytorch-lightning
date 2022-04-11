# T5 pytorch lightning
基于 pytorch lightning 和 🤗 transformers 预训练与 fine-tune T5模型，整体代码框架初始化于我的另一个仓库[Pytorch-Lightning-Template](https://github.com/TianHongZXY/Pytorch-Lightning-Template)，并在此基于上做了众多的修改和优化，后续会将本仓库的代码合并进该模板。

经过测试已支持fine-tune的模型有：[google/t5](https://huggingface.co/t5-base)，[google/mt5](https://huggingface.co/google/mt5-base)，[LangBoat/Mengzi](https://huggingface.co/Langboat/mengzi-t5-base)，[IDEA-CCNL/Randeng-770M](https://huggingface.co/IDEA-CCNL/Randeng-770M)，使用Randeng需要将[Fengshen框架](https://github.com/IDEA-CCNL/Fengshenbang-LM)中的`fengshen`文件夹放到根目录。