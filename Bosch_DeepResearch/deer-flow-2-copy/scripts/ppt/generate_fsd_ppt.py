#!/usr/bin/env python3
"""
Tesla FSD PPT Generation Setup
为 agent 准备完整的 PPT 生成 prompt
"""

import json
from pathlib import Path
from datetime import datetime


def main():
    print("=" * 80)
    print("Tesla FSD PPT Generation - Standalone Agent Setup")
    print("=" * 80)
    print()
    
    # 创建工作目录
    workspace = Path("/tmp/fsd_agent_ppt/workspace")
    outputs = Path("/tmp/fsd_agent_ppt/outputs")
    workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Workspace: {workspace}")
    print(f"✓ Outputs:   {outputs}")
    print()
    
    # 生成完整 prompt
    prompt = """请使用 ppt-generation skill 为我生成一份专业的 Tesla FSD 演示文稿。

【演示文稿标题】
Tesla Full Self-Driving (FSD) - 自动驾驶革命

【风格设置】
- 名称: gradient-modern
- 主色调: Tesla 红色 (#E82127) 渐变到 电蓝色 (#0071E3)
- 字体: 现代几何无衬线字体，标题 80pt+，正文 20pt
- 效果: 流畅渐变、运动模糊、深度感、科技感

【幻灯片结构 (7张)】

第1张 - 标题幻灯片
标题: "Tesla Full Self-Driving"
副标题: "Autonomous Driving Revolution"
视觉: 红蓝渐变背景，FSD 炫光文本，神经网络可视化，Tesla Model 3 侧影，数字高速路

第2张 - 核心能力
标题: "What is Full Self-Driving?"
要点:
- Navigate on autopilot without intervention
- Auto park, summon, and lane change
- Real-time environmental awareness
- Continuous learning from fleet data

第3张 - 技术栈
标题: "Key Technologies"
要点:
- 8 cameras + 12 ultrasonic sensors
- Real-time neural network processing
- Fleet Learning: 4+ billion miles
- Safety-first redundancy architecture

第4张 - 真实能力
标题: "Real-World Capabilities"
要点:
- Urban navigation in complex traffic
- Automatic unprotected left turns
- Roundabout handling and intersections
- Adaptive behavior to road conditions

第5张 - 安全冗余
标题: "Safety & Redundancy"
要点:
- Multiple independent computation paths
- Continuous self-test and validation
- Safer than average human driver
- Transparent safety metrics reporting

第6张 - 路线图
标题: "The Path Forward"
要点:
- Expanding to more regions globally
- Improved urban and complex scenarios
- Integration with Tesla Optimus robots
- Robotaxi network deployment

第7张 - 结论
标题: "The Future of Mobility"
副标题: "Join the autonomous revolution"
视觉: 多辆 Tesla 在日落高速路，技术叠加层，自动驾驶的未来，
      Tesla Logo 和关键数据

【输出要求】
1. 生成位置: /tmp/fsd_agent_ppt/outputs/Tesla_FSD_Presentation.pptx
2. 分辨率: 1024×576 (16:9)
3. 图像质量: 高质量 (num_steps=20)
4. 格式: 标准 PPTX 文件

请立即开始生成演示文稿。"""
    
    # 保存 prompt 到文件
    prompt_file = workspace / "tesla_fsd_ppt_prompt.txt"
    prompt_file.write_text(prompt)
    
    print("[1/3] Prompt prepared")
    print(f"      Location: {prompt_file}")
    print(f"      Size: {len(prompt)} characters")
    print()
    
    # 创建 PPT 计划 JSON
    ppt_plan = {
        "title": "Tesla Full Self-Driving (FSD)",
        "style": "gradient-modern",
        "slides_count": 7,
        "output_path": str(outputs / "Tesla_FSD_Presentation.pptx")
    }
    
    plan_file = workspace / "ppt_plan.json"
    with open(plan_file, "w") as f:
        json.dump(ppt_plan, f, indent=2)
    
    print("[2/3] PPT plan created")
    print(f"      Location: {plan_file}")
    print(f"      Slides: {ppt_plan['slides_count']}")
    print()
    
    # 创建执行说明
    print("[3/3] Execution instructions")
    print()
    
    print("=" * 80)
    print("HOW TO GENERATE PPT - 3 Options")
    print("=" * 80)
    print()
    
    print("【OPTION A】Web UI (推荐)")
    print("-" * 80)
    print("1. Start DeerFlow:")
    print("   cd /Users/timchef/Bosch_DeepResearch/deer-flow-2-copy")
    print("   make dev")
    print()
    print("2. Open browser: http://localhost:2026")
    print()
    print("3. Copy this prompt to chat:")
    print()
    print(prompt)
    print()
    print("4. Wait for agent to generate PPT (10-20 min on CPU, 2-5 min on GPU)")
    print()
    
    print("【OPTION B】Direct File Execution")
    print("-" * 80)
    print(f"cat {prompt_file} | xclip")
    print("# Then paste into DeerFlow Web UI")
    print()
    
    print("【OPTION C】Python Client (Advanced)")
    print("-" * 80)
    print("from deerflow.client import DeerFlowClient")
    print(f"client = DeerFlowClient()")
    print(f"result = client.stream('{prompt}', thread_id='tesla_fsd')")
    print()
    
    print("=" * 80)
    print("EXPECTED OUTPUT")
    print("=" * 80)
    print()
    print(f"File: {outputs / 'Tesla_FSD_Presentation.pptx'}")
    print("Contains:")
    print("  ✓ 7 slides with AI-generated images")
    print("  ✓ gradient-modern style (red to blue)")
    print("  ✓ 1024x576 resolution (16:9)")
    print("  ✓ High-quality images (SDXL)")
    print()
    
    print("=" * 80)
    print("✓ Setup Complete!")
    print("=" * 80)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Next: 使用以上任一方式执行 PPT 生成")


if __name__ == "__main__":
    main()
