"""
Command-Line Interface for Kitchen Inventory Agent

This is the entry point for the application.
It provides a simple interactive chat interface.
"""

import sys
from src.vector_store import VectorStore
from src.agent import KitchenInventoryAgent


def print_welcome():
    """Display welcome message and instructions."""
    print("\n" + "="*60)
    print("   🍽️  KITCHEN INVENTORY MANAGEMENT ASSISTANT")
    print("="*60)
    print("\nI can help you with:")
    print("  • Check inventory levels ('How much olive oil do we have?')")
    print("  • Perform calculations ('How many servings in 3.5L?')")
    print("  • Search for information ('Olive oil substitutes?')")
    print("  • Generate reports ('Generate monthly inventory report')")
    print("\nType 'quit' or 'exit' to end the session.")
    print("="*60 + "\n")


def main():
    """
    Main function to run the agent.
    
    Process:
    1. Initialize vector store
    2. Load knowledge base
    3. Initialize agent
    4. Start interactive loop
    """
    print("Initializing Kitchen Inventory Agent...")
    
    # Initialize vector store
    try:
        vector_store = VectorStore()
        vector_store.load_knowledge_base()
        print("✓ Vector store initialized\n")
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = KitchenInventoryAgent(vector_store)
        print("✓ Agent initialized\n")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Display welcome message
    print_welcome()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for using Kitchen Inventory Assistant!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process query
            response = agent.chat(user_input)
            
            # Display response
            print(f"\nAgent: {response}\n")
            print("-"*60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nSession ended by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()