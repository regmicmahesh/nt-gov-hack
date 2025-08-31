import getpass
import os
import dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import Dict, Any
import re

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="embeddings",
    embedding_function=embeddings,
)

class WebApiAgent:
    ...




class DatabaseRouter:
    """Clean router using LangChain's MultiPromptChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize database connections
        self.budget_db = SQLDatabase(engine=create_engine("sqlite:///budget.db"))
        self.employee_db = SQLDatabase(engine=create_engine("sqlite:///employee.db"))
        
        # Create database agents
        self.budget_agent = create_sql_agent(
            llm=self.llm, 
            db=self.budget_db, 
            agent_type="openai-tools", 
            verbose=False
        )
        
        self.employee_agent = create_sql_agent(
            llm=self.llm, 
            db=self.employee_db, 
            agent_type="openai-tools", 
            verbose=False
        )
        
        # Define routing destinations with prompts
        self.destinations = [
            {
                "name": "budget",
                "description": "Useful for questions about budgets, finances, costs, expenses, procurement, vendors, contracts, billing, and financial data",
                "prompt": PromptTemplate(
                    template="You are a budget and financial database expert. Answer questions about budgets, costs, expenses, procurement, vendors, and financial data.\n\nQuestion: {input}\n\nAnswer:",
                    input_variables=["input"]
                ),
                "chain": self.budget_agent
            },
            {
                "name": "employee", 
                "description": "Useful for questions about employees, staff, HR, salaries, performance, leave, departments, personnel, and human resources data",
                "prompt": PromptTemplate(
                    template="You are an employee and HR database expert. Answer questions about employees, staff, HR, salaries, performance, leave, departments, and human resources data.\n\nQuestion: {input}\n\nAnswer:",
                    input_variables=["input"]
                ),
                "chain": self.employee_agent
            }
        ]
        
        # Create the router chain using MultiPromptChain
        self.router_chain = MultiPromptChain.from_prompts(
            self.llm,
            self.destinations,
            MULTI_PROMPT_ROUTER_TEMPLATE,
            verbose=False
        )
    
    def route_and_answer(self, question: str) -> Dict[str, Any]:
        """Route question to appropriate database and get answer"""
        try:
            # Use LangChain's built-in routing
            result = self.router_chain.run(question)
            
            # Determine which route was taken based on the result content
            route = self._determine_route_from_result(result, question)
            
            return {
                "route": route,
                "question": question,
                "answer": result,
                "success": True
            }
            
        except Exception as e:
            # Simple fallback - default to budget if routing fails
            return {
                "route": "budget",
                "question": question,
                "answer": f"Error in routing: {str(e)}",
                "success": False
            }
    
    def _determine_route_from_result(self, result: str, question: str) -> str:
        """Determine which route was taken based on result analysis"""
        # Simple heuristic: if result contains budget-related terms, it's budget
        budget_indicators = ['budget', 'cost', 'expense', 'vendor', 'procurement', 'payment']
        employee_indicators = ['employee', 'staff', 'department', 'salary', 'performance', 'leave']
        
        result_lower = result.lower()
        question_lower = question.lower()
        
        budget_score = sum(1 for term in budget_indicators if term in result_lower or term in question_lower)
        employee_score = sum(1 for term in employee_indicators if term in result_lower or term in question_lower)
        
        return 'budget' if budget_score >= employee_score else 'employee'

class CleanInferenceRouter:
    """Main router class with clean LangChain patterns"""
    
    def __init__(self):
        self.router = DatabaseRouter()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question through the routing system"""
        print(f"🔀 Processing: {question}")
        print("-" * 50)
        
        result = self.router.route_and_answer(question)
        
        print(f"🎯 Routed to: {result['route'].upper()} database")
        print(f"✅ Answer: {result['answer'][:300]}...")
        print("=" * 60)
        
        return result
    
    def interactive_mode(self):
        """Interactive mode using LangChain patterns"""
        print("🤖 Clean Inference Router Started!")
        print("Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n💬 Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                self.process_question(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    router = CleanInferenceRouter()
    
    # # Test routing
    # test_questions = [
    #     "How many employees work in the IT department?",
    #     "What's the total budget for this year?",
    #     "Who has the highest performance score?",
    #     "What are our procurement expenses?",
    #     "How many people are on leave this month?",
    #     "What's the cost of office supplies?"
    # ]
    
    # print("🧪 Testing routing on sample questions:")
    # print("=" * 60)
    
    # for question in test_questions:
    #     router.process_question(question)
    
    print("\n" + "=" * 60)
    print("Starting interactive mode...")
    router.interactive_mode() 