"""
ToolChain + ReActAgent 测试脚本

演示内容：
  1. ToolChain 基本用法（单步 & 多步链式调用）
  2. ToolChainManager 管理多条链
  3. 将 ToolChain 包装为工具，接入 ReActAgent 使用
"""

import os
import sys

# 确保能导入 hello_agents 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hello_agents import (
    HelloAgentsLLM,
    ReActAgent,
    ToolRegistry,
    ToolChain,
    ToolChainManager,
    calculate,
)


# ============================================================
# 第一部分：ToolChain 基础用法
# ============================================================
def test_basic_chain():
    """单步工具链：直接把用户输入当作表达式计算"""
    print("\n" + "=" * 60)
    print("测试 1：单步工具链")
    print("=" * 60)

    registry = ToolRegistry()
    registry.register_function("calculate", "数学计算器", calculate)

    chain = ToolChain("single_calc", "单步计算链")
    # {input} 会被替换为 execute() 传入的 input_data
    chain.add_step("calculate", "{input}", "result")

    result = chain.execute(registry, "2 + 3 * 4")
    print(f"最终结果: {result}")  # 预期: 14


def test_multi_step_chain():
    """多步工具链：前一步的结果作为后续步骤的输入"""
    print("\n" + "=" * 60)
    print("测试 2：多步工具链（步骤间变量传递）")
    print("=" * 60)

    registry = ToolRegistry()
    registry.register_function("calculate", "数学计算器", calculate)

    chain = ToolChain("multi_calc", "多步计算链")
    # 第 1 步：计算 10 + 20 => 结果存入 step1
    chain.add_step("calculate", "10 + 20", "step1")
    # 第 2 步：用第 1 步结果继续计算 => {step1} * 3
    chain.add_step("calculate", "{step1} * 3", "step2")
    # 第 3 步：最终计算 => {step2} + 100
    chain.add_step("calculate", "{step2} + 100", "final")

    result = chain.execute(registry, "开始计算")
    print(f"最终结果: {result}")  # 预期: (10+20)*3+100 = 190


# ============================================================
# 第二部分：ToolChainManager 管理多条链
# ============================================================
def test_chain_manager():
    """使用 ToolChainManager 注册和执行多条工具链"""
    print("\n" + "=" * 60)
    print("测试 3：ToolChainManager 管理多条链")
    print("=" * 60)

    registry = ToolRegistry()
    registry.register_function("calculate", "数学计算器", calculate)

    manager = ToolChainManager(registry)

    # 链 A：求圆的面积 (pi * r^2, r=5)
    chain_a = ToolChain("circle_area", "计算圆的面积")
    chain_a.add_step("calculate", "3.14159 * 5 ** 2", "area")
    manager.register_chain(chain_a)

    # 链 B：两步运算
    chain_b = ToolChain("two_step", "两步运算链")
    chain_b.add_step("calculate", "100 / 4", "half")
    chain_b.add_step("calculate", "{half} + 50", "total")
    manager.register_chain(chain_b)

    # 查看已注册的链
    print(f"已注册的链: {manager.list_chains()}")
    print(f"circle_area 详情: {manager.get_chain_info('circle_area')}")

    # 分别执行
    result_a = manager.execute_chain("circle_area", "")
    print(f"圆的面积: {result_a}")  # 预期 ≈ 78.54

    result_b = manager.execute_chain("two_step", "")
    print(f"两步运算结果: {result_b}")  # 预期: 100/4+50 = 75.0


# ============================================================
# 第三部分：将 ToolChain 包装为工具，接入 ReActAgent
# ============================================================
def test_chain_with_react_agent():
    """
    核心演示：把 ToolChain 封装成一个函数工具注册到 ToolRegistry，
    这样 ReActAgent 就能在推理循环中调用整条链。
    """
    print("\n" + "=" * 60)
    print("测试 4：ReActAgent + ToolChain 联合使用")
    print("=" * 60)

    # --- 1. 创建 LLM ---
    # 请确保环境变量中已配置好 API Key 和 Base URL
    # 例如: DEEPSEEK_API_KEY, OPENAI_API_KEY 等
    llm = HelloAgentsLLM()

    # --- 2. 创建内部 registry 供 chain 使用 ---
    inner_registry = ToolRegistry()
    inner_registry.register_function("calculate", "数学计算器", calculate)

    # --- 3. 构建工具链 ---
    # 链：先算加法，再将结果乘以 2
    double_chain = ToolChain("double_calc", "先计算表达式，再将结果乘以2")
    double_chain.add_step("calculate", "{input}", "first_result")
    double_chain.add_step("calculate", "{first_result} * 2", "final_result")

    chain_manager = ToolChainManager(inner_registry)
    chain_manager.register_chain(double_chain)

    # --- 4. 把链包装成一个普通函数工具 ---
    def run_double_chain(expression: str) -> str:
        """执行 double_calc 链：计算表达式后再乘以2"""
        return chain_manager.execute_chain("double_calc", expression)

    # --- 5. 构建 ReActAgent 使用的 registry ---
    agent_registry = ToolRegistry()
    # 注册普通计算器（单次计算）
    agent_registry.register_function(
        "calculate",
        "数学计算器，计算单个数学表达式，如 2+3、sqrt(16)",
        calculate,
    )
    # 注册链工具（计算后自动翻倍）
    agent_registry.register_function(
        "double_calculate",
        "翻倍计算器：先计算表达式的值，然后自动将结果乘以2。例如输入'3+4'，得到14",
        run_double_chain,
    )

    # --- 6. 创建 ReActAgent ---
    agent = ReActAgent(
        name="链式工具测试助手",
        llm=llm,
        tool_registry=agent_registry,
        max_steps=5,
    )

    # --- 7. 运行测试 ---
    questions = [
        "请用翻倍计算器计算 15 + 25 的翻倍结果",
        "先用普通计算器算 sqrt(144)，再用翻倍计算器算 10 * 5 的翻倍结果",
    ]

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"问题: {q}")
        print("─" * 50)
        response = agent.run(q)
        print(f"\n最终回答: {response}")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 不需要 LLM 的纯工具链测试
    test_basic_chain()
    test_multi_step_chain()
    test_chain_manager()

    # 需要 LLM 的 ReActAgent + ToolChain 测试
    # 如果没有配置 API Key，可以注释掉下面这行
    test_chain_with_react_agent()
