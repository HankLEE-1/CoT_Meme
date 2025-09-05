import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_framework_diagram():
    """创建MemeCLIP-CoT框架图"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4FD',
        'feature': '#FFF2CC',
        'fusion': '#D5E8D4',
        'cot': '#E1D5E7',
        'output': '#F8CECC',
        'border': '#666666'
    }
    
    # 标题
    ax.text(5, 11.5, 'MemeCLIP-CoT: Chain of Thought 增强的多模态迷因检测框架', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 输入层
    input_box1 = FancyBboxPatch((0.5, 9.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box1)
    ax.text(2, 10, '图像输入\n(Image Input)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    input_box2 = FancyBboxPatch((6.5, 9.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box2)
    ax.text(8, 10, '文本输入\n(Text Input)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # CLIP特征提取
    clip_box1 = FancyBboxPatch((0.5, 8), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['feature'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(clip_box1)
    ax.text(2, 8.5, 'CLIP视觉编码器\n(CLIP Visual Encoder)', ha='center', va='center', fontsize=11)
    
    clip_box2 = FancyBboxPatch((6.5, 8), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['feature'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(clip_box2)
    ax.text(8, 8.5, 'CLIP文本编码器\n(CLIP Text Encoder)', ha='center', va='center', fontsize=11)
    
    # 特征映射
    map_box1 = FancyBboxPatch((0.5, 6.5), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['feature'], 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(map_box1)
    ax.text(2, 7, '图像特征映射\n(Image Feature Mapping)', ha='center', va='center', fontsize=11)
    
    map_box2 = FancyBboxPatch((6.5, 6.5), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['feature'], 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(map_box2)
    ax.text(8, 7, '文本特征映射\n(Text Feature Mapping)', ha='center', va='center', fontsize=11)
    
    # 适配器
    adapter_box1 = FancyBboxPatch((0.5, 5), 3, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['feature'], 
                                 edgecolor=colors['border'], linewidth=2)
    ax.add_patch(adapter_box1)
    ax.text(2, 5.5, '图像适配器\n(Image Adapter)', ha='center', va='center', fontsize=11)
    
    adapter_box2 = FancyBboxPatch((6.5, 5), 3, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['feature'], 
                                 edgecolor=colors['border'], linewidth=2)
    ax.add_patch(adapter_box2)
    ax.text(8, 5.5, '文本适配器\n(Text Adapter)', ha='center', va='center', fontsize=11)
    
    # 传统融合路径
    fusion_box = FancyBboxPatch((2, 3.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 4, '传统特征融合\n(Traditional Feature Fusion)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # CoT推理模块
    cot_box = FancyBboxPatch((1, 2), 8, 1.2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['cot'], 
                            edgecolor=colors['border'], linewidth=2)
    ax.add_patch(cot_box)
    ax.text(5, 2.6, 'Chain of Thought 推理模块\n(CoT Reasoning Module)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # CoT内部结构
    cot_inner1 = FancyBboxPatch((1.2, 1.5), 2.2, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=colors['border'], linewidth=1)
    ax.add_patch(cot_inner1)
    ax.text(2.3, 1.8, '多模态融合块\n(Multi-modal Fusion)', ha='center', va='center', fontsize=9)
    
    cot_inner2 = FancyBboxPatch((3.9, 1.5), 2.2, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=colors['border'], linewidth=1)
    ax.add_patch(cot_inner2)
    ax.text(5, 1.8, '推理步骤\n(Reasoning Steps)', ha='center', va='center', fontsize=9)
    
    cot_inner3 = FancyBboxPatch((6.6, 1.5), 2.2, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=colors['border'], linewidth=1)
    ax.add_patch(cot_inner3)
    ax.text(7.7, 1.8, '一致性约束\n(Consistency Loss)', ha='center', va='center', fontsize=9)
    
    # 输出层
    output_box1 = FancyBboxPatch((2, 0.5), 2.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box1)
    ax.text(3.25, 0.9, '传统分类器\n(Traditional Classifier)', ha='center', va='center', fontsize=10)
    
    output_box2 = FancyBboxPatch((5.5, 0.5), 2.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box2)
    ax.text(6.75, 0.9, 'CoT分类器\n(CoT Classifier)', ha='center', va='center', fontsize=10)
    
    # 最终输出
    final_box = FancyBboxPatch((3.5, -0.5), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#FFE6CC', 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(5, -0.1, '最终预测\n(Final Prediction)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 添加箭头连接
    arrows = [
        # 输入到CLIP
        ((2, 9.5), (2, 8)),
        ((8, 9.5), (8, 8)),
        # CLIP到映射
        ((2, 8), (2, 6.5)),
        ((8, 8), (8, 6.5)),
        # 映射到适配器
        ((2, 6.5), (2, 5)),
        ((8, 6.5), (8, 5)),
        # 适配器到传统融合
        ((2, 5), (2, 4)),
        ((8, 5), (8, 4)),
        # 适配器到CoT
        ((2, 5), (2, 2.6)),
        ((8, 5), (8, 2.6)),
        # 传统融合到传统分类器
        ((5, 3.5), (3.25, 0.5)),
        # CoT到CoT分类器
        ((5, 2), (6.75, 0.5)),
        # 分类器到最终输出
        ((3.25, 0.5), (4.25, -0.1)),
        ((6.75, 0.5), (5.75, -0.1)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['border'], ec=colors['border'])
        ax.add_patch(arrow)
    
    # 添加说明文字
    ax.text(0.5, 11, '技术特点:', fontsize=14, fontweight='bold')
    ax.text(0.5, 10.7, '• 多模态特征提取与融合', fontsize=11)
    ax.text(0.5, 10.4, '• Chain of Thought 逐步推理', fontsize=11)
    ax.text(0.5, 10.1, '• 一致性约束保证推理质量', fontsize=11)
    ax.text(0.5, 9.8, '• 双路径分类提升性能', fontsize=11)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('MemeCLIP_CoT_Framework.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("框架图已保存为 'MemeCLIP_CoT_Framework.png'")

def create_detailed_cot_diagram():
    """创建详细的CoT推理流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4FD',
        'fusion': '#FFF2CC',
        'reasoning': '#D5E8D4',
        'output': '#E1D5E7',
        'border': '#666666'
    }
    
    # 标题
    ax.text(5, 9.5, 'Chain of Thought 推理过程详解', 
            fontsize=18, fontweight='bold', ha='center')
    
    # 输入特征
    input_box1 = FancyBboxPatch((0.5, 8), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box1)
    ax.text(2, 8.4, '图像特征\n(Image Features)', ha='center', va='center', fontsize=11)
    
    input_box2 = FancyBboxPatch((6.5, 8), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(input_box2)
    ax.text(8, 8.4, '文本特征\n(Text Features)', ha='center', va='center', fontsize=11)
    
    # 初始融合
    fusion_box = FancyBboxPatch((3, 7), 4, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 7.4, '多模态融合块 (MFB)\nMulti-modal Fusion Block', ha='center', va='center', fontsize=11)
    
    # 推理步骤
    steps = [
        (2, 5.5, '推理步骤 1\nReasoning Step 1'),
        (5, 5.5, '推理步骤 2\nReasoning Step 2'),
        (8, 5.5, '推理步骤 3\nReasoning Step 3')
    ]
    
    for i, (x, y, text) in enumerate(steps):
        step_box = FancyBboxPatch((x-1, y), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['reasoning'], 
                                 edgecolor=colors['border'], linewidth=2)
        ax.add_patch(step_box)
        ax.text(x, y+0.4, text, ha='center', va='center', fontsize=10)
        
        # 添加注意力机制说明
        attn_text = f'注意力机制\n(Attention)'
        ax.text(x, y-0.3, attn_text, ha='center', va='center', fontsize=9, style='italic')
    
    # 一致性约束
    consistency_box = FancyBboxPatch((1, 3.5), 8, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFE6CC', 
                                   edgecolor=colors['border'], linewidth=2)
    ax.add_patch(consistency_box)
    ax.text(5, 3.9, '一致性约束 (Consistency Loss)\n确保推理步骤间的逻辑连贯性', ha='center', va='center', fontsize=11)
    
    # 输出投影
    output_box = FancyBboxPatch((3, 2.5), 4, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.9, '输出投影\n(Output Projection)', ha='center', va='center', fontsize=11)
    
    # 分类器
    classifier_box = FancyBboxPatch((3, 1.5), 4, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#F8CECC', 
                                   edgecolor=colors['border'], linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(5, 1.9, 'CoT分类器\n(CoT Classifier)', ha='center', va='center', fontsize=11)
    
    # 最终输出
    final_box = FancyBboxPatch((4, 0.5), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#D5E8D4', 
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(final_box)
    ax.text(5, 0.9, '预测结果\n(Prediction)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 添加箭头
    arrows = [
        ((2, 8), (3, 7.4)),
        ((8, 8), (7, 7.4)),
        ((5, 7), (2, 5.5)),
        ((2, 5.5), (5, 5.5)),
        ((5, 5.5), (8, 5.5)),
        ((8, 5.5), (5, 3.9)),
        ((5, 3.5), (5, 2.9)),
        ((5, 2.5), (5, 1.9)),
        ((5, 1.5), (5, 0.9)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=colors['border'], ec=colors['border'])
        ax.add_patch(arrow)
    
    # 添加说明
    ax.text(0.5, 9, 'CoT推理优势:', fontsize=12, fontweight='bold')
    ax.text(0.5, 8.7, '• 模拟人类逐步推理过程', fontsize=10)
    ax.text(0.5, 8.4, '• 增强模型可解释性', fontsize=10)
    ax.text(0.5, 8.1, '• 提高复杂任务性能', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('CoT_Detailed_Process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("详细CoT流程图已保存为 'CoT_Detailed_Process.png'")

if __name__ == "__main__":
    create_framework_diagram()
    create_detailed_cot_diagram() 