"""
Main Agent Logic for Kitchen Inventory Assistant

This is the brain of the agent. It:
1. Receives user queries
2. Decides which tools to use
3. Calls the LLM with appropriate context
4. Returns formatted responses
"""

import re
import requests
import json
from src.prompts import (
    get_system_prompt,
    get_tool_selection_prompt,
    format_context,
    get_anti_hallucination_instructions,
    get_tool_descriptions
)
from src.tools import InventoryTools


class KitchenInventoryAgent:
    """
    The main AI agent class.
    
    Think of this as the agent's "brain" that coordinates:
    - Understanding user requests
    - Calling tools when needed
    - Generating responses via LLM
    """
    
    def __init__(self, vector_store):
        """
        Initialize the agent.
        
        Args:
            vector_store: VectorStore instance for inventory access
        """
        self.vector_store = vector_store
        self.tools = InventoryTools(vector_store)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5:3b"
        
        # Store conversation history for context
        self.conversation_history = []
    
    def call_llm(self, prompt, system_prompt=None):
        """
        Call the Ollama LLM with a prompt.
        
        Args:
            prompt: The main question/instruction for the LLM
            system_prompt: Optional system-level instructions
            
        Returns:
            LLM's text response
            
        How it works:
        1. Build request payload with prompt and system message
        2. Send to Ollama API
        3. Stream response tokens
        4. Return complete response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def parse_tool_call(self, llm_response):
        """
        Extract tool calls from LLM response.
        
        The LLM indicates tool usage with special format:
        TOOL: tool_name
        PARAMETERS: {"param": "value"}
        
        Args:
            llm_response: Text response from LLM
            
        Returns:
            Dictionary with tool_name and parameters, or None
        """
        # Look for TOOL: pattern
        tool_match = re.search(r'TOOL:\s*(\w+)', llm_response)
        params_match = re.search(r'PARAMETERS:\s*({.*?})', llm_response, re.DOTALL)
        
        if tool_match:
            tool_name = tool_match.group(1)
            
            # Parse parameters if present
            params = {}
            if params_match:
                try:
                    params = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    # If JSON parsing fails, try simple key:value format
                    params_text = params_match.group(1).strip('{}')
                    if ':' in params_text:
                        key, value = params_text.split(':', 1)
                        params = {key.strip().strip('"'): value.strip().strip('"')}
            
            return {"tool": tool_name, "parameters": params}
        
        return None
    
    def execute_tool(self, tool_name, parameters):
        """
        Execute a tool and return its result.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Dictionary of parameters for the tool
            
        Returns:
            Tool execution result as string
        """
        # Map tool names to actual tool methods
        tool_map = {
            "search_inventory": self.tools.search_inventory,
            "calculate": self.tools.calculate,
            "web_search": self.tools.web_search,
            "generate_monthly_report": self.tools.generate_monthly_report
        }
        
        if tool_name not in tool_map:
            return f"Unknown tool: {tool_name}"
        
        try:
            tool_func = tool_map[tool_name]
            
            # Different tools expect different parameters
            if tool_name == "generate_monthly_report":
                # This tool takes no parameters
                return tool_func()
            elif tool_name == "search_inventory":
                query = parameters.get("query", "")
                return tool_func(query)
            elif tool_name == "calculate":
                expression = parameters.get("expression", "")
                return tool_func(expression)
            elif tool_name == "web_search":
                query = parameters.get("query", "")
                return tool_func(query)
            else:
                return "Tool execution error: Unknown parameter format"
        
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def process_query(self, user_query):
        """
        Main processing function: handles the entire query → response flow.
        
        This is the core logic that ties everything together.
        
        Args:
            user_query: User's question/request
            
        Returns:
            Final response string
            
        Process:
        1. Analyze query to determine if tools are needed
        2. If tools needed, execute them
        3. Build context with tool results
        4. Generate final response via LLM
        5. Return formatted answer
        """
        print(f"\n{'='*60}")
        print(f"Processing query: {user_query}")
        print(f"{'='*60}\n")
        
        # Step 1: Determine if we need tools
        tool_descriptions = get_tool_descriptions()
        tool_selection_prompt = get_tool_selection_prompt(user_query, tool_descriptions)
        
        # Ask LLM which tool(s) to use
        system_prompt = get_system_prompt()
        tool_decision = self.call_llm(tool_selection_prompt, system_prompt)
        
        print(f"[Agent Decision]\n{tool_decision}\n")
        
        # Step 2: Parse and execute tool if needed
        tool_result = ""
        tool_call = self.parse_tool_call(tool_decision)
        
        if tool_call:
            print(f"[Executing Tool: {tool_call['tool']}]")
            tool_result = self.execute_tool(tool_call['tool'], tool_call['parameters'])
            print(f"[Tool Result]\n{tool_result[:200]}...\n")
        else:
            print("[No tool needed - generating direct response]\n")
        
        # Step 3: Build final prompt with all context
        final_prompt = f"""User Query: {user_query}

{get_anti_hallucination_instructions()}

"""
        
        if tool_result:
            final_prompt += f"""Tool Results:
{tool_result}

Using the tool results above, answer the user's query accurately and concisely.
"""
        else:
            final_prompt += """Answer the user's query based on your knowledge of the inventory system.
"""
        
        # Step 4: Generate final response
        final_response = self.call_llm(final_prompt, system_prompt)
        
        # Step 5: Store in conversation history (for potential multi-turn conversations)
        self.conversation_history.append({
            "user": user_query,
            "agent": final_response
        })
        
        return final_response
    
    def chat(self, user_input):
        """
        Simple chat interface wrapper.
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response
        """
        return self.process_query(user_input)