# backdoor-attacks
Backdoor Attacks Research Project
项目简介
本项目专注于大语言模型和多模态模型的后门攻击研究，结合了ModelScope Swift框架进行高效的模型微调和实验。项目旨在探索和分析深度学习模型中的安全漏洞，特别是后门攻击的机制和防御方法。
主要特性
🔍 后门攻击研究

多样化数据生成：支持并行和序列化数据处理流程
触发器生成：自动化触发器模式生成和优化
样本选择：智能样本筛选和数据集构建
模型预训练：支持多种预训练模型的后门植入

🚀 Swift框架集成

广泛模型支持：支持350+大语言模型和80+多模态模型
高效微调：基于LoRA、AdaLoRA等先进适配器技术
端到端流程：集成训练、推理、评估、量化全流程
多模态能力：首个系统性支持多模态大语言模型的框架

项目结构
backdoor-attacks/
├── utils/                      # 工具函数和辅助模块
├── data_produce_parral0.py     # 并行数据生成 - 配置0
├── data_produce_parral1.py     # 并行数据生成 - 配置1  
├── data_produce_seq0.py        # 序列数据生成 - 配置0
├── data_produce_seq1.py        # 序列数据生成 - 配置1
├── data_producesingle.py       # 单一数据生成
├── dataproduce_parral2.py      # 并行数据生成 - 配置2
├── dataproduce_seq2.py         # 序列数据生成 - 配置2
├── ncfm_dataset_handler.py     # NCFM数据集处理器
├── pretrain_model.py           # 预训练模型处理
├── sample_select.py            # 样本选择算法
├── select300.py                # 300样本选择
├── trigger_generator.py        # 触发器生成器
└── README.md                   # 项目文档

快速开始
环境要求
# Python 3.8+
pip install torch transformers
pip install ms-swift  # ModelScope Swift框架

基础使用

数据生成

# 并行数据生成
python data_produce_parral0.py

# 序列数据生成  
python data_produce_seq0.py


触发器生成

python trigger_generator.py


模型预训练

python pretrain_model.py

核心功能
数据处理模块

并行处理：多进程数据生成，提高处理效率
序列处理：有序数据生成，保证数据一致性
样本筛选：基于多种策略的智能样本选择

后门攻击实现

触发器设计：支持多种触发器模式和优化策略
攻击植入：在预训练阶段植入后门机制
效果评估：全面的攻击成功率和隐蔽性评估

Swift框架优势
基于ModelScope Swift框架，本项目具备以下优势：

支持最新的大语言模型和多模态模型
高效的参数微调技术
完整的模型生命周期管理
丰富的评估和基准测试工具

实验配置
数据集支持

NCFM数据集
自定义数据集
多模态数据集

模型支持
通过Swift框架支持：

LLaMA系列模型
ChatGLM系列模型
Qwen系列模型
多模态视觉-语言模型

研究应用
本项目的研究成果可应用于：

模型安全评估：评估大语言模型的安全漏洞
防御机制研究：开发针对后门攻击的防御策略
鲁棒性测试：测试模型在对抗性环境下的表现
安全基准建设：构建模型安全评估标准

技术特点
高效性

利用Swift框架的轻量级基础设施
支持大规模模型的高效微调
优化的数据处理流程

可扩展性

模块化设计，易于扩展新功能
支持多种攻击策略和防御方法
兼容多种模型架构

实用性

完整的实验流程和工具链
详细的评估指标和可视化
易于复现的实验设置

贡献指南
欢迎提交Issue和Pull Request来改进项目：

Fork本仓库
创建特性分支 (git checkout -b feature/AmazingFeature)
提交更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
开启Pull Request

许可证
本项目采用MIT许可证 - 查看 LICENSE 文件了解详情。
致谢

感谢ModelScope团队开发的Swift框架
@misc{zhao2024swiftascalablelightweightinfrastructure,
      title={SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning},
      author={Yuze Zhao and Jintao Huang and Jinghan Hu and Xingjun Wang and Yunlin Mao and Daoze Zhang and Zeyinzi Jiang and Zhikai Wu and Baole Ai and Ang Wang and Wenmeng Zhou and Yingda Chen},
      year={2024},
      eprint={2408.05517},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.05517},
}

联系方式
如有问题或建议，请通过以下方式联系：

GitHub Issues: 项目Issues页面
Email: [您的邮箱]


注意：本项目仅用于学术研究目的，请勿用于恶意攻击或其他非法用途。使用者需要遵守相关法律法规和伦理准则。
