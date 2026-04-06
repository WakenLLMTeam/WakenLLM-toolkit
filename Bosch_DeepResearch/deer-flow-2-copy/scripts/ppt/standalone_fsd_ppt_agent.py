"""
Standalone Tesla FSD PPT Generation Agent
无需前端，直接用 DeerFlow agent 生成 PPT
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加 DeerFlow 到路径
sys.path.insert(0, str(Path(__file__).parent / "backend" / "packages" / "harness"))

from deerflow.agents.lead_agent.agent import make_lead_agent
from deerflow.config.app_config import AppConfig
from langgraph.graph import RunnableConfig


async def generate_fsd_ppt():
    """
    使用 DeerFlow agent 生成 Tesla FSD PPT
    
    Workflow:
    1. Agent 读取用户 prompt
    2. 调用 ppt-generation skill
    3. 生成演示文稿规划
    4. 调用 image-generation skill 逐张生成图像
    5. 组合成 PPTX 文件
    """
    
    print("=" * 80)
    print("Tesla FSD PPT Generation - Standalone Agent")
    print("=" * 80)
    print()
    
    # 创建工作目录
    workspace = Path("/tmp/fsd_agent_ppt/workspace")
    outputs = Path("/tmp/fsd_agent_ppt/outputs")
    workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    
    print(f"Workspace: {workspace}")
    print(f"Outputs:   {outputs}")
    print()
    
    # 加载配置
    print("[1/4] Loading DeerFlow configuration...")
    try:
        config = AppConfig.from_file()
        print(f"✓ Config loaded")
    except Exception as e:
        print(f"✗ Config error: {e}")
        print("  Using default config...")
        config = AppConfig()
    
    print()
    
    # 创建 agent
    print("[2/4] Creating agent...")
    agent = make_lead_agent(
        thinking_enabled=False,
        model_name="claude-opus-4-1-20250805",
        is_plan_mode=False,
        subagent_enabled=True,
        rag_resources=None
    )
    print("✓ Agent created")
    print()
    
    # 准备 prompt
    user_prompt = """
请使用 ppt-generation skill 为我生成一份关于 Tesla Full Self-Driving (FSD) 的专业演示文稿。

要求:
1. 标题: "Tesla Full Self-Driving (FSD) - Autonomous Driving Revolution"
2. 风格: gradient-modern (红色到蓝色渐变，现代科技感)
3. 幻灯片数: 5-7 张
4. 内容要点:
   - 第1张: 标题幻灯片 (Tesla FSD 的未来)
   - 第2张: 核心功能 (自动导航、自动泊车、自动变道等)
   - 第3张: 技术架构 (8摄像头、12超声波传感器、实时神经网络)
   - 第4张: 安全冗余 (独立计算路径、自我检测验证)
   - 第5张: 性能数据 (4+ 十亿英里、4000万用户)
   - 第6张: 路线图 (全球扩展、Robotaxi网络)
   - 第7张: 结论 (未来出行革命)

5. 视觉风格:
   - 配色: Tesla红色 (#E82127) 到 电蓝色 (#0071E3)
   - 排版: 大胆现代字体，80pt+ 标题
   - 图像: 未来感、科技感、自动驾驶相关
   - 效果: 流畅渐变、运动模糊、深度感

6. 输出位置: /tmp/fsd_agent_ppt/outputs/Tesla_FSD_Presentation.pptx

请开始生成演示文稿。使用 skills 系统自动处理图像生成和 PPTX 组合。
"""
    
    print("[3/4] Generating PPT with agent...")
    print("-" * 80)
    print()
    
    # 创建 config
    config_obj = RunnableConfig(
        configurable={
            "model_name": "claude-opus-4-1-20250805",
            "thinking_enabled": False,
            "is_plan_mode": False,
            "subagent_enabled": True,
        }
    )
    
    try:
        # 调用 agent（同步方式）
        result = agent.invoke(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": user_prompt
                    }
                ]
            },
            config=config_obj
        )
        
        print()
        print("-" * 80)
        print("[4/4] Generation complete")
        print()
        
        # 显示结果
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                print("Agent Response:")
                print(last_message.content[:500] + "..." if len(last_message.content) > 500 else last_message.content)
        
        # 检查输出文件
        pptx_file = outputs / "Tesla_FSD_Presentation.pptx"
        if pptx_file.exists():
            file_size = pptx_file.stat().st_size / (1024 * 1024)  # MB
            print()
            print(f"✓ PPT file generated: {pptx_file}")
            print(f"  File size: {file_size:.2f} MB")
        else:
            print()
            print(f"⚠ PPT file not found at {pptx_file}")
            print(f"  Check for files in {outputs}:")
            for f in sorted(outputs.glob("*")):
                print(f"    - {f.name}")
        
        print()
        print("=" * 80)
        print("✓ PPT Generation Completed Successfully!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Main entry point"""
    try:
        success = await generate_fsd_ppt()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Tesla FSD PPT Agent...\n")
    asyncio.run(main())
