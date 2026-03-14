import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()  # 必须在 import hello_agents 之前调用，确保环境变量已加载

# 刷新数据库配置（用已加载的环境变量覆盖模块导入时的空配置）
from hello_agents.core.database_config import update_database_config
update_database_config(
    qdrant={
        "url": os.getenv("QDRANT_URL"),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
    }
)
from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import MCPTool

agent = SimpleAgent(name="助手", llm=HelloAgentsLLM())

# 无需任何配置，自动使用内置演示服务器
playwright_tool = MCPTool(
    name="playwright",
    server_command=["npx", "-y", "@playwright/mcp"]
)
agent.add_tool(playwright_tool)
# ✅ MCP工具 'calculator' 已展开为 6 个独立工具

# 智能体可以直接使用展开后的工具
response = agent.run("打开Chrome浏览器进行www.baidu.com网页的测试")
print(response)
