#!/usr/bin/env python3
"""
Tesla FSD PPT Generation - 3 Slides Version
快速生成 3 页演示文稿
"""

import json
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 80)
    print("Tesla FSD PPT Generation - 3 Slides")
    print("=" * 80)
    print()
    
    # 创建工作目录
    workspace = Path("/tmp/fsd_ppt_3slides/workspace")
    outputs = Path("/tmp/fsd_ppt_3slides/outputs")
    workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Workspace: {workspace}")
    print(f"✓ Outputs:   {outputs}")
    print()
    
    # 生成 3 页 PPT 的 Prompt
    prompt = """请使用 ppt-generation skill 为我生成一份专业的 Tesla FSD 演示文稿。

【演示文稿标题】
Tesla Full Self-Driving (FSD) - 自动驾驶革命

【风格设置】
- 名称: gradient-modern
- 主色调: Tesla 红色 (#E82127) 渐变到 电蓝色 (#0071E3)
- 字体: 现代几何无衬线字体，标题 80pt+，正文 20pt
- 效果: 流畅渐变、运动模糊、深度感、科技感

【幻灯片结构 (3张)】

第1张 - 标题幻灯片
标题: "Tesla Full Self-Driving"
副标题: "Autonomous Driving Revolution"
视觉: 红蓝渐变背景，FSD 炫光文本，神经网络可视化，Tesla Model 3 侧影，数字高速路

第2张 - 核心技术
标题: "Key Technologies"
要点:
- 8 cameras + 12 ultrasonic sensors (8摄像头+12超声波传感器)
- Real-time neural network processing (实时神经网络处理)
- Fleet Learning: 4+ billion miles (车队学习: 40亿英里)
- Safety-first redundancy architecture (安全第一的冗余架构)

第3张 - 结论
标题: "The Future of Mobility"
副标题: "Join the autonomous revolution"
视觉: 多辆 Tesla 在日落高速路，技术叠加层，自动驾驶的未来，
      Tesla Logo 和关键数据 (8M+ 车辆, 4B+ 英里, 99.9% 安全率)

【输出要求】
1. 生成位置: /tmp/fsd_ppt_3slides/outputs/Tesla_FSD_3Slides.pptx
2. 分辨率: 1024×576 (16:9)
3. 图像质量: 高质量 (num_steps=15 for faster generation)
4. 格式: 标准 PPTX 文件

【工作流】
1. 使用 ppt-generation skill 创建演示计划
2. 调用 image-generation skill 逐张生成图像 (3张)
3. 使用 python-pptx 组合成最终 PPTX 文件
4. 保存到输出目录

请立即开始生成演示文稿。"""
    
    # 保存 prompt
    prompt_file = workspace / "tesla_fsd_3slides.txt"
    prompt_file.write_text(prompt)
    
    print("[1/3] Prompt prepared")
    print(f"      File: {prompt_file}")
    print(f"      Size: {len(prompt)} characters")
    print()
    
    # 创建 PPT 计划
    ppt_plan = {
        "title": "Tesla Full Self-Driving (FSD)",
        "style": "gradient-modern",
        "slides_count": 3,
        "output_path": str(outputs / "Tesla_FSD_3Slides.pptx")
    }
    
    plan_file = workspace / "ppt_plan.json"
    with open(plan_file, "w") as f:
        json.dump(ppt_plan, f, indent=2)
    
    print("[2/3] PPT plan created")
    print(f"      File: {plan_file}")
    print(f"      Slides: {ppt_plan['slides_count']}")
    print()
    
    # 显示执行指令
    print("[3/3] Ready to generate")
    print()
    print("=" * 80)
    print("HOW TO GENERATE - Copy & Paste")
    print("=" * 80)
    print()
    print("1. Start DeerFlow:")
    print("   cd /Users/timchef/Bosch_DeepResearch/deer-flow-2-copy && make dev")
    print()
    print("2. Open browser: http://localhost:2026")
    print()
    print("3. Copy and paste this prompt to chat:")
    print()
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print()
    print("4. Wait for generation (5-10 min on CPU, 1-2 min on GPU)")
    print()
    print("5. Download file:")
    print(f"   {outputs / 'Tesla_FSD_3Slides.pptx'}")
    print()
    print("=" * 80)
    print("✓ Setup Complete - Ready to generate!")
    print("=" * 80)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
