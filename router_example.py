#!/usr/bin/env python3
"""
Example usage of the Clean Inference Router

This script demonstrates how to use the refactored InferenceRouter that leverages
LangChain's built-in routing and executor patterns.
"""

from inference_router import CleanInferenceRouter

def main():
    # Initialize the clean router
    print("🚀 Initializing Clean Inference Router...")
    router = CleanInferenceRouter()
    print("✅ Router initialized successfully!")
    
    # Example questions that will be automatically routed using LangChain patterns
    example_questions = [
        # These should go to employee.db
        "How many employees are in the engineering department?",
        "What's the average salary in the company?",
        "Who has taken the most leave days this year?",
        "Show me all managers and their departments",
        
        # These should go to budget.db  
        "What's our total spending on office supplies?",
        "How much budget is allocated for marketing?",
        "Which vendor received the most payments?",
        "What are our top 5 expenses this quarter?",
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING LANGCHAIN-BASED ROUTING")
    print("="*80)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n📋 Test {i}/8")
        print(f"❓ Question: {question}")
        
        # The router will automatically classify and route using LangChain patterns
        result = router.process_question(question)
        
        print(f"🎯 Final Route: {result['route'].upper()}")
        print(f"✅ Success: {result['success']}")
        print("-" * 80)
    
    print("\n🎉 Automated testing complete!")
    print("💡 You can also run inference_router.py directly for interactive mode")
    print("🔧 This version uses LangChain's MultiRouteChain for cleaner routing!")

if __name__ == "__main__":
    main() 