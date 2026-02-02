"""
Tools for the Kitchen Inventory Agent

Each tool is a Python function the agent can call.
The agent decides when to use each tool based on the user's question.
"""

import re
import requests


class InventoryTools:
    """
    Collection of tools the agent can use.
    
    Tool Types:
    1. Knowledge Search - Search vector database for inventory info
    2. Calculator - Perform mathematical calculations
    3. Web Search - Search internet for external information (simulated)
    """
    
    def __init__(self, vector_store):
        """
        Initialize tools with access to vector store.
        
        Args:
            vector_store: VectorStore instance for inventory searches
        """
        self.vector_store = vector_store
    
    def search_inventory(self, query):
        """
        TOOL: Search inventory database for specific items.
        
        Use when: User asks about stock levels, locations, or item details
        
        Args:
            query: What to search for (e.g., "olive oil", "flour")
            
        Returns:
            Relevant inventory information as text
            
        Examples:
        - "How much olive oil do we have?" → searches for "olive oil"
        - "Where is the flour stored?" → searches for "flour"
        """
        print(f"[TOOL CALL: search_inventory] Query: {query}")
        
        results = self.vector_store.search(query, n_results=2)
        
        if results:
            # Combine search results
            return "\n\n".join(results)
        else:
            return "No inventory information found for that query."
    
    def calculate(self, expression):
        """
        TOOL: Perform mathematical calculations.
        
        Use when: User asks to calculate quantities, conversions, or totals
        
        Args:
            expression: Math expression as string (e.g., "3.5 * 1000 / 15")
            
        Returns:
            Calculation result as string
            
        Examples:
        - "How many 15ml servings in 3.5L?" → "3.5 * 1000 / 15"
        - "What's 25kg - 5kg?" → "25 - 5"
        
        Security Note: Uses eval() which can be dangerous in production.
        In real systems, use a safe math parser library instead.
        """
        print(f"[TOOL CALL: calculate] Expression: {expression}")
        
        try:
            # Clean the expression (remove any non-math characters)
            # Allow only numbers, operators, parentheses, and decimal points
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            
            # Evaluate the mathematical expression
            # WARNING: eval() is unsafe for user input in production!
            # Use a library like 'simpleeval' for real applications
            result = eval(safe_expr)
            
            return f"Result: {result}"
        
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    def web_search(self, query):
        """
        TOOL: Search the web for information (simulated).
        
        Use when: User asks about information not in inventory database
        
        Args:
            query: What to search for online
            
        Returns:
            Simulated search results
            
        Examples:
        - "What are alternatives to olive oil?"
        - "Current market price for flour"
        - "Best suppliers for organic vegetables"
        
        Note: This is a simulation. In production, you'd integrate with:
        - Google Custom Search API
        - Bing Search API
        - SerpAPI
        - Or use the actual Ollama web search capability
        """
        print(f"[TOOL CALL: web_search] Query: {query}")
        
        # Simulated responses for common queries
        # In production, this would call a real search API
        simulated_responses = {
            "olive oil": "Common olive oil substitutes:\n- Canola oil (neutral flavor, good for high heat)\n- Avocado oil (similar health benefits, high smoke point)\n- Grapeseed oil (light flavor, versatile)\n- Sunflower oil (economical alternative)",
            
            "flour": "Types of flour and uses:\n- All-purpose: General baking and cooking\n- Bread flour: High protein, best for yeast breads\n- Cake flour: Low protein, tender baked goods\n- Whole wheat: Higher fiber, denser texture",
            
            "suppliers": "Finding reliable food suppliers:\n1. Check local wholesaler directories\n2. Join restaurant industry associations\n3. Attend food trade shows\n4. Get recommendations from other restaurants\n5. Compare pricing and delivery terms",
        }
        
        # Find best match
        query_lower = query.lower()
        for key, response in simulated_responses.items():
            if key in query_lower:
                return f"Web Search Results:\n{response}"
        
        # Default response
        return f"Web search for '{query}' would be performed here. In production, this would use a real search API."
    
    def generate_monthly_report(self):
        """
        TOOL: Generate end-of-month inventory report.
        
        Use when: User asks for inventory report, summary, or full stock list
        
        Returns:
            Formatted inventory report with all items
        """
        print(f"[TOOL CALL: generate_monthly_report]")
        
        # Get all inventory documents
        all_inventory = self.vector_store.get_all_inventory()
        
        if not all_inventory:
            return "No inventory data available."
        
        # Format as report
        report = "=" * 50 + "\n"
        report += "MONTHLY INVENTORY REPORT\n"
        report += "Date: January 31, 2026\n"
        report += "=" * 50 + "\n\n"
        
        for doc in all_inventory:
            report += doc + "\n\n" + "-" * 50 + "\n\n"
        
        return report