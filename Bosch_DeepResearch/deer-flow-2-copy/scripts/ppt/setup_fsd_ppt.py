"""
Standalone Tesla FSD PPT Generation Agent
简化版 - 直接使用 DeerFlow client
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_fsd_ppt_prompt():
    """生成 Tesla FSD PPT 的完整 prompt"""
    
    return """
请使用 ppt-generation skill 为我生成一份专业的 Tesla FSD 演示文稿。

需求详情:

【演示文稿标题】
Tesla Full Self-Driving (FSD) - 自动驾驶革命

【风格设置】
- 名称: gradient-modern
- 主色调: Tesla 红色 (#E82127) → 电蓝色 (#0071E3) 渐变
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
- Navigate on autopilot without intervention (自动导航无需干预)
- Auto park, summon, and lane change (自动泊车、召唤、变道)
- Real-time environmental awareness (实时环境感知)
- Continuous learning from fleet data (持续学习车队数据)

第3张 - 技术栈
标题: "Key Technologies"
要点:
- 8 cameras + 12 ultrasonic sensors (8摄像头+12超声波传感器)
- Real-time neural network processing (实时神经网络处理)
- Fleet Learning: 4+ billion miles (车队学习: 40亿英里)
- Safety-first redundancy architecture (安全第一的冗余架构)

第4张 - 真实能力
标题: "Real-World Capabilities"
要点:
- Urban navigation in complex traffic (复杂交通城市导航)
- Automatic unprotected left turns (自动无保护左转)
- Roundabout handling and intersections (环岛和十字路口)
- Adaptive behavior to road conditions (适应路况的行为)

第5张 - 安全冗余
标题: "Safety & Redundancy"
要点:
- Multiple independent computation paths (多个独立计算路径)
- Continuous self-test and validation (持续自检和验证)
- Safer than average human driver (比平均人类驾驶员更安全)
- Transparent safety metrics reporting (透明的安全指标报告)

第6张 - 路线图
标题: "The Path Forward"
要点:
- Expanding to more regions globally (全球区域扩展)
- Improved urban and complex scenarios (改进城市和复杂场景)
- Integration with Tesla Optimus robots (与Tesla Optimus机器人集成)
- Robotaxi network deployment (机器人出租车网络部署)

第7张 - 结论
标题: "The Future of Mobility"
副标题: "Join the autonomous revolution"
视觉: 多辆 Tesla 在日落高速路，技术叠加层，"自动驾驶的未来" 大文本，
      Tesla Logo 和关键数据 (8M+ 车辆, 4B+ 英里, 99.9% 安全率)

【输出要求】
1. 生成位置: /tmp/fsd_agent_ppt/outputs/Tesla_FSD_Presentation.pptx
2. 分辨率: 1024×576 (16:9)
3. 图像质量: 高质量 (num_steps=20)
4. 格式: 标准 PPTX 文件

【工作流】
1. 使用 ppt-generation skill 创建演示计划 (JSON)
2. 调用 image-generation skill 逐张生成高质量图像
3. 使用 python-pptx 组合成最终 PPTX 文件
4. 保存到输出目录

请立即开始生成演示文稿。
"""


def create_standalone_script():
    """创建一个完整的独立脚本版本"""
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Tesla FSD PPT Generation - Standalone Script
无前端，直接调用 DeerFlow agent 生成 PPT
\"\"\"

import json
import sys
from pathlib import Path
from datetime import datetime

# 配置
WORKSPACE = Path("/tmp/fsd_agent_ppt/workspace")
OUTPUTS = Path("/tmp/fsd_agent_ppt/outputs")
PROMPT_FILE = WORKSPACE / "fsd_ppt_prompt.txt"
PLAN_FILE = WORKSPACE / "fsd_ppt_plan.json"

def main():
    print("=" * 80)
    print("Tesla FSD PPT Generation - Standalone Agent")
    print("=" * 80)
    print()
    
    # 创建目录
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    
    print(f"Workspace: {WORKSPACE}")
    print(f"Outputs:   {OUTPUTS}")
    print()
    
    # 生成 prompt
    print("[1/3] Preparing generation prompt...")
"""
    
    # 添加 prompt
    fsd_prompt = generate_fsd_ppt_prompt()
    script_content += f'''
    prompt = """{fsd_prompt}"""
    
    prompt_file = WORKSPACE / "fsd_generation.txt"
    prompt_file.write_text(prompt)
    print(f"✓ Prompt ready: {{prompt_file}}")
    print()
    
    # 显示即将执行的步骤
    print("[2/3] Agent execution plan:")
    print("  Step 1: Create presentation plan (JSON)")
    print("  Step 2: Generate slide images (SDXL local)")
    print("  Step 3: Compose into PPTX file")
    print()
    
    # 提示用户后续步骤
    print("[3/3] To complete generation:")
    print()
    print("  Option A - Use DeerFlow Web UI:")
    print("    1. Start DeerFlow: make dev")
    print("    2. Paste the prompt into chat")
    print("    3. System will auto-generate and download PPT")
    print()
    print("  Option B - Use Python Client:")
    print("    from deerflow.client import DeerFlowClient")
    print("    client = DeerFlowClient()")
    print("    result = client.chat(prompt, thread_id='fsd_ppt')")
    print()
    print("  Option C - Use LangGraph Server:")
    print("    Start server and invoke agent directly")
    print()
    print("=" * 80)
    print("Prompt saved to: {{prompt_file}}")
    print("=" * 80)
    print()
    print("Generated at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
"""
    
    script_path = Path("/Users/timchef/Bosch_DeepResearch/deer-flow-2-copy/fsd_ppt_generator.py")
    script_path.write_text(script_content)
    
    print(f"✓ Created: {script_path}")
    return script_path


def main():
    """主函数"""
    
    print("=" * 80)
    print("Tesla FSD PPT Generation - Setup")
    print("=" * 80)
    print()
    
    # 创建工作目录
    workspace = Path("/tmp/fsd_agent_ppt/workspace")
    outputs = Path("/tmp/fsd_agent_ppt/outputs")
    workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    
    print(f"Created workspace: {workspace}")
    print(f"Created outputs:   {outputs}")
    print()
    
    # 保存 prompt
    print("[1/2] Saving PPT generation prompt...")
    prompt = generate_fsd_ppt_prompt()
    prompt_file = workspace / "tesla_fsd_ppt.txt"
    prompt_file.write_text(prompt)
    print(f"✓ Prompt saved: {prompt_file}")
    print(f"  Prompt length: {len(prompt)} characters")
    print()
    
    # 创建独立脚本
    print("[2/2] Creating standalone script...")
    script_path = create_standalone_script()
    print()
    
    # 显示使用说明
    print("=" * 80)
    print("NEXT STEPS - 执行选项")
    print("=" * 80)
    print()
    
    print("【选项 A】使用 Web UI (推荐)")
    print("  1. 启动 DeerFlow:")
    print("     cd /Users/timchef/Bosch_DeepResearch/deer-flow-2-copy")
    print("     make dev")
    print()
    print("  2. 打开浏览器访问 http://localhost:2026")
    print()
    print("  3. 复制以下 prompt 到聊天框:")
    print("     " + "-" * 70)
    print("     " + prompt[:200] + "...")
    print("     " + "-" * 70)
    print()
    print("  4. Agent 会自动:")
    print("     - 调用 ppt-generation skill")
    print("     - 生成演示计划")
    print("     - 逐张生成高质量图像 (SDXL)")
    print("     - 组合成 PPTX 文件")
    print("     - 保存到 /tmp/fsd_agent_ppt/outputs/")
    print()
    
    print("【选项 B】使用 Python 脚本")
    print(f"  python {script_path}")
    print()
    
    print("【选项 C】直接复制 Prompt")
    print(f"  cat {prompt_file}")
    print()
    
    # 显示 prompt 信息
    print("=" * 80)
    print("PROMPT INFORMATION")
    print("=" * 80)
    print()
    print(f"Location: {prompt_file}")
    print(f"Size: {len(prompt)} chars")
    print(f"Lines: {len(prompt.split(chr(10)))}")
    print()
    print("Preview (first 300 chars):")
    print("-" * 80)
    print(prompt[:300])
    print("-" * 80)
    print()
    
    print("=" * 80)
    print("✓ Setup Complete!")
    print("=" * 80)
    print()
    print("Ready to generate Tesla FSD PPT using DeerFlow agent skills!")
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
