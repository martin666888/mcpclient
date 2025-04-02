import asyncio
import sys
import json
import signal
from typing import Optional, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# 全局 API 配置
API_CONFIG = {
    "api_key": "your-api-key-here",
    "api_base": "API base url",     # 或其他支持OpenAI兼容API的服务
    "model": "模型名称"              # 根据您使用的API提供商进行调整
}

class MCPClient:
    def __init__(self):
        # 初始化会话和客户端对象
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.pending_tasks: List[asyncio.Task] = []

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=API_CONFIG["api_key"],
            base_url=API_CONFIG["api_base"]
        )

    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是.py或.js文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，可用工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """使用OpenAI兼容 API和可用工具处理查询"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 初始API调用
        try:
            response = self.client.chat.completions.create(
                model=API_CONFIG["model"],
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

            # 处理响应
            final_text = []

            # 获取模型的响应
            message = response.choices[0].message
            if message.content:
                final_text.append(message.content)

            # 处理工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # 打印调试信息
                    print(f"工具调用: {tool_name}, 参数: {tool_args}")
                    
                    # 执行工具调用
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[调用工具 {tool_name}，参数 {tool_args}]")
                    
                    # 返回工具调用结果
                    tool_result = str(result.content)
                    
                    # 添加助手消息和工具结果到上下文
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })
                    
                    # 确保工具响应格式正确
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result  # 使用字符串
                    })
                    
                    # 打印调试信息
                    print("发送到API的消息结构:")
                    print(json.dumps(messages, indent=2))
                    
                    # 获取下一个响应 - 使用相同的模型
                    try:
                        second_response = self.client.chat.completions.create(
                            model=API_CONFIG["model"],  # 使用全局配置的模型
                            messages=messages,
                            tools=available_tools,
                            tool_choice="auto"
                        )
                        
                        next_message = second_response.choices[0].message
                        if next_message.content:
                            final_text.append(next_message.content)
                    except Exception as e:
                        final_text.append(f"获取最终响应时出错: {str(e)}")
                        print(f"获取最终响应时出错: {str(e)}")
            
            return "\n".join(final_text)
        
        except Exception as e:
            return f"API调用出错: {str(e)}"
    
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP客户端已启动！")
        print("输入您的查询或'quit'退出。")
        
        while True:
            try:
                query = input("\n查询: ").strip()
                if query.lower() == 'quit':
                    print("正在清理并退出...")
                    break
                
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\n错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    async def cleanup(self):
        """清理资源，简化版本，避免与anyio冲突"""
        try:
            # 简单地关闭退出栈，不使用wait_for
            await self.exit_stack.aclose()
        except Exception as e:
            # 捕获但不重新抛出异常，只是记录它们
            print(f"清理过程中出现异常 (可以忽略): {str(e)}")

async def main():
    if len(sys.argv) < 2:
        print("用法: python mcpclient.py <服务器脚本路径>")
        sys.exit(1)
    
    client = MCPClient()
    
    # Windows平台不支持add_signal_handler，省略信号处理
    
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except asyncio.CancelledError:
        # 捕获并忽略取消错误
        print("程序被取消，正常退出")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的异常堆栈，便于调试