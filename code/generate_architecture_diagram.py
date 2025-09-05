import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_memeclip_cot_architecture():
    """创建基于实际代码的MemeCLIP-CoT架构图"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#E3F2FD',      # 浅蓝色 - 输入
        'clip': '#FFF3E0',      # 浅橙色 - CLIP编码
        'mapping': '#F3E5F5',   # 浅紫色 - 特征映射
        'adapter': '#E8F5E8',   # 浅绿色 - 适配器
        'fusion': '#FFF8E1',    # 浅黄色 - 融合
        'cot': '#FFEBEE',       # 浅红色 - CoT推理
        'classifier': '#E0F2F1', # 浅青色 - 分类器
        'output': '#FCE4EC',    # 浅粉色 - 输出
        'border': '#424242'     # 深灰色 - 边框
    }
    
    # 标题
    ax.text(6, 13.5, 'MemeCLIP-CoT: 基于实际代码的架构图', 
            fontsize=22, fontweight='bold', ha='center')
    ax.text(6, 13, 'Chain of Thought Enhanced Multimodal Meme Detection Framework', 
            fontsize=14, ha='center', style='italic')
    
    # === 输入层 ===
    # 图像输入
    img_input = FancyBboxPatch((0.5, 11.5), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(img_input)
    ax.text(1.75, 12, '图像输入\n(Image Input)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 文本输入
    text_input = FancyBboxPatch((9, 11.5), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_input)
    ax.text(10.25, 12, '文本输入\n(Text Input)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # === CLIP编码层 ===
    # CLIP视觉编码器
    clip_vis = FancyBboxPatch((0.5, 10), 2.5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['clip'], 
                             edgecolor=colors['border'], linewidth=2)
    ax.add_patch(clip_vis)
    ax.text(1.75, 10.5, 'CLIP视觉编码器\n(CLIP Visual Encoder)', ha='center', va='center', fontsize=10)
    
    # CLIP文本编码器
    clip_text = FancyBboxPatch((9, 10), 2.5, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['clip'], 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(clip_text)
    ax.text(10.25, 10.5, 'CLIP文本编码器\n(CLIP Text Encoder)', ha='center', va='center', fontsize=10)
    
    # === 特征映射层 ===
    # 图像特征映射
    img_map = FancyBboxPatch((0.5, 8.5), 2.5, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['mapping'], 
                            edgecolor=colors['border'], linewidth=2)
    ax.add_patch(img_map)
    ax.text(1.75, 9, 'LinearProjection\n(图像特征映射)', ha='center', va='center', fontsize=10)
    
    # 文本特征映射
    text_map = FancyBboxPatch((9, 8.5), 2.5, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['mapping'], 
                             edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_map)
    ax.text(10.25, 9, 'LinearProjection\n(文本特征映射)', ha='center', va='center', fontsize=10)
    
    # === 适配器层 ===
    # 图像适配器
    img_adapter = FancyBboxPatch((0.5, 7), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['adapter'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(img_adapter)
    ax.text(1.75, 7.5, 'Adapter\n(图像适配器)', ha='center', va='center', fontsize=10)
    
    # 文本适配器
    text_adapter = FancyBboxPatch((9, 7), 2.5, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['adapter'], 
                                 edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_adapter)
    ax.text(10.25, 7.5, 'Adapter\n(文本适配器)', ha='center', va='center', fontsize=10)
    
    # === 特征融合与归一化 ===
    fusion_box = FancyBboxPatch((3.5, 6), 5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(6, 6.5, '特征融合 + 归一化\n(Feature Fusion & Normalization)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # === CoT推理模块 ===
    cot_main = FancyBboxPatch((2, 4), 8, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['cot'], 
                             edgecolor=colors['border'], linewidth=3)
    ax.add_patch(cot_main)
    ax.text(6, 4.75, 'Chain of Thought 推理模块\n(CoTClassifier)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # CoT内部组件
    # MultiModalFusionBlock
    mfb_box = FancyBboxPatch((2.2, 3.8), 2.2, 0.6, 
                             boxstyle="round,pad=0.05", 
                             facecolor='white', 
                             edgecolor=colors['border'], linewidth=1)
    ax.add_patch(mfb_box)
    ax.text(3.3, 4.1, 'MultiModalFusionBlock\n(MFB)', ha='center', va='center', fontsize=9)
    
    # ReasoningStep (多个)
    reasoning_box = FancyBboxPatch((4.9, 3.8), 2.2, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', 
                                   edgecolor=colors['border'], linewidth=1)
    ax.add_patch(reasoning_box)
    ax.text(6, 4.1, 'ReasoningStep\n(×3 steps)', ha='center', va='center', fontsize=9)
    
    # Consistency Loss
    consistency_box = FancyBboxPatch((7.6, 3.8), 2.2, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='white', 
                                    edgecolor=colors['border'], linewidth=1)
    ax.add_patch(consistency_box)
    ax.text(8.7, 4.1, 'Consistency Loss\n(一致性约束)', ha='center', va='center', fontsize=9)
    
    # === 分类器层 ===
    classifier_box = FancyBboxPatch((4, 2), 4, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['classifier'], 
                                   edgecolor=colors['border'], linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(6, 2.5, 'CoT分类器\n(CoT Classifier Head)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # === 输出层 ===
    output_box = FancyBboxPatch((5, 0.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 1, '预测结果\n(Prediction)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # === 添加箭头连接 ===
    arrows = [
        # 输入到CLIP
        ((1.75, 11.5), (1.75, 10)),
        ((10.25, 11.5), (10.25, 10)),
        # CLIP到映射
        ((1.75, 10), (1.75, 8.5)),
        ((10.25, 10), (10.25, 8.5)),
        # 映射到适配器
        ((1.75, 8.5), (1.75, 7)),
        ((10.25, 8.5), (10.25, 7)),
        # 适配器到融合
        ((1.75, 7), (3.5, 6.5)),
        ((10.25, 7), (8.5, 6.5)),
        # 融合到CoT
        ((6, 6), (6, 4)),
        # CoT内部流程
        ((3.3, 3.8), (6, 3.8)),
        ((6, 3.8), (8.7, 3.8)),
        # CoT到分类器
        ((6, 4), (6, 2.5)),
        # 分类器到输出
        ((6, 2), (6, 1)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['border'], ec=colors['border'])
        ax.add_patch(arrow)
    
    # === 添加技术说明 ===
    ax.text(0.5, 12.5, '技术特点:', fontsize=14, fontweight='bold')
    ax.text(0.5, 12.2, '• 基于CLIP的多模态特征提取', fontsize=11)
    ax.text(0.5, 11.9, '• 双线性池化融合 (MFB)', fontsize=11)
    ax.text(0.5, 11.6, '• 多步推理机制 (3步)', fontsize=11)
    ax.text(0.5, 11.3, '• 一致性约束保证推理质量', fontsize=11)
    
    # === 添加代码对应说明 ===
    ax.text(0.5, 1.5, '代码文件对应:', fontsize=12, fontweight='bold')
    ax.text(0.5, 1.2, '• MemeCLIP_CoT.py: 主模型类', fontsize=10)
    ax.text(0.5, 0.9, '• cot_modules.py: CoT核心模块', fontsize=10)
    ax.text(0.5, 0.6, '• models.py: 基础组件', fontsize=10)
    ax.text(0.5, 0.3, '• main_cot.py: 训练脚本', fontsize=10)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('MemeCLIP_CoT_Architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("架构图已保存为 'MemeCLIP_CoT_Architecture.png'")

def create_detailed_cot_flow():
    """创建详细的CoT推理流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E3F2FD',
        'fusion': '#FFF3E0',
        'reasoning': '#F3E5F5',
        'consistency': '#E8F5E8',
        'output': '#FFEBEE',
        'border': '#424242'
    }
    
    # 标题
    ax.text(6, 11.5, 'CoT推理模块详细流程图', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 输入特征
    input_box1 = FancyBboxPatch((1, 10), 2.5, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box1)
    ax.text(2.25, 10.4, '图像特征\n(Image Features)', ha='center', va='center', fontsize=10)
    
    input_box2 = FancyBboxPatch((8.5, 10), 2.5, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box2)
    ax.text(9.75, 10.4, '文本特征\n(Text Features)', ha='center', va='center', fontsize=10)
    
    # MultiModalFusionBlock
    mfb_box = FancyBboxPatch((4.5, 9), 3, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['fusion'], 
                            edgecolor=colors['border'], linewidth=2)
    ax.add_patch(mfb_box)
    ax.text(6, 9.4, 'MultiModalFusionBlock\n(双线性池化融合)', ha='center', va='center', fontsize=11)
    
    # 推理步骤
    steps = [
        (2, 7.5, '推理步骤 1\n(ReasoningStep 1)'),
        (6, 7.5, '推理步骤 2\n(ReasoningStep 2)'),
        (10, 7.5, '推理步骤 3\n(ReasoningStep 3)')
    ]
    
    for i, (x, y, text) in enumerate(steps):
        step_box = FancyBboxPatch((x-1, y), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['reasoning'], 
                                 edgecolor=colors['border'], linewidth=2)
        ax.add_patch(step_box)
        ax.text(x, y+0.4, text, ha='center', va='center', fontsize=10)
        
        # 添加注意力机制说明
        ax.text(x, y-0.2, 'MultiheadAttention\n+ FeedForward', ha='center', va='center', 
                fontsize=9, style='italic')
    
    # 一致性约束
    consistency_box = FancyBboxPatch((2, 5.5), 8, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['consistency'], 
                                   edgecolor=colors['border'], linewidth=2)
    ax.add_patch(consistency_box)
    ax.text(6, 5.9, '一致性约束 (Consistency Loss)\n确保推理步骤间的逻辑连贯性', 
            ha='center', va='center', fontsize=11)
    
    # 输出投影
    output_proj = FancyBboxPatch((4.5, 4), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_proj)
    ax.text(6, 4.4, '输出投影\n(Output Projection)', ha='center', va='center', fontsize=11)
    
    # 分类器
    classifier_box = FancyBboxPatch((4.5, 2.5), 3, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#E0F2F1', 
                                   edgecolor=colors['border'], linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(6, 2.9, '分类器\n(Classifier Head)', ha='center', va='center', fontsize=11)
    
    # 最终输出
    final_box = FancyBboxPatch((5.5, 1), 1, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FCE4EC', 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(6, 1.4, '预测\n(Prediction)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    # 添加箭头
    arrows = [
        ((2.25, 10), (4.5, 9.4)),
        ((9.75, 10), (7.5, 9.4)),
        ((6, 9), (2, 7.5)),
        ((2, 7.5), (6, 7.5)),
        ((6, 7.5), (10, 7.5)),
        ((10, 7.5), (6, 5.9)),
        ((6, 5.5), (6, 4.4)),
        ((6, 4), (6, 2.9)),
        ((6, 2.5), (6, 1.4)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['border'], ec=colors['border'])
        ax.add_patch(arrow)
    
    # 添加说明
    ax.text(0.5, 10.5, 'CoT推理优势:', fontsize=12, fontweight='bold')
    ax.text(0.5, 10.2, '• 模拟人类逐步推理', fontsize=10)
    ax.text(0.5, 9.9, '• 增强模型可解释性', fontsize=10)
    ax.text(0.5, 9.6, '• 提高复杂任务性能', fontsize=10)
    ax.text(0.5, 9.3, '• 保证推理一致性', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('CoT_Detailed_Flow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("详细CoT流程图已保存为 'CoT_Detailed_Flow.png'")

if __name__ == "__main__":
    create_memeclip_cot_architecture()
    create_detailed_cot_flow()
