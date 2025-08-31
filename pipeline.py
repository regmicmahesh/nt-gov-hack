from typing import Annotated, Optional
import dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
import os
import json
import yaml
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools import QuerySQLDataBaseTool, InfoSQLDatabaseTool
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from json_repair import repair_json

dotenv.load_dotenv()

# Load DATABASE configuration from YAML
# Get the directory where pipeline.py is located
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file_path = os.path.join(pipeline_dir, "database_with_details.yaml")

with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)
    DATABASE = config["databases"]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: Optional[str]  # Track the original user question
    thread_id: Optional[str]       # Unique identifier for this conversation
    retry_needed: Optional[bool]   # Flag for relevance check results
    needs_relevance_check: Optional[bool]  # Whether relevance checking is needed
    accuracy_retry_needed: Optional[bool]  # Flag for accuracy check results
    relevance_evaluation: Optional[dict]   # Results from relevance checking
    accuracy_evaluation: Optional[dict]    # Results from accuracy checking
    reasoning_analysis: Optional[dict]     # Results from reasoning analysis
    citation_analysis: Optional[dict]      # Results from citation analysis
    relevance_retry_count: Optional[int]   # Number of relevance retries attempted
    accuracy_retry_count: Optional[int]    # Number of accuracy retries attempted
    max_retries_exceeded: Optional[bool]   # Whether max retries were exceeded
    retry_warnings: Optional[list]         # List of retry warnings for final response
    executed_queries: Optional[list]       # Track executed database queries and results
    query_execution_log: Optional[dict]    # Detailed log of query executions


graph_builder = StateGraph(State)

# Connect to both databases
# Use absolute paths for database files
employee_db_path = os.path.join(pipeline_dir, "employee.db")
budget_db_path = os.path.join(pipeline_dir, "budget.db")
employee_db_uri = os.environ.get("EMPLOYEE_DATABASE_URI", f"sqlite:///{employee_db_path}")
budget_db_uri = os.environ.get("BUDGET_DATABASE_URI", f"sqlite:///{budget_db_path}")

employee_db = SQLDatabase.from_uri(employee_db_uri)
budget_db = SQLDatabase.from_uri(budget_db_uri)

# Define SQL tools for employee database
employee_query_tool = QuerySQLDataBaseTool(db=employee_db)
employee_info_tool = InfoSQLDatabaseTool(db=employee_db)
employee_sql_tools = [employee_query_tool, employee_info_tool]

# Define SQL tools for budget database
budget_query_tool = QuerySQLDataBaseTool(db=budget_db)
budget_info_tool = InfoSQLDatabaseTool(db=budget_db)
budget_sql_tools = [budget_query_tool, budget_info_tool]
import yaml

llm = init_chat_model("openai:gpt-4o-mini")
llm_high_quality = init_chat_model("openai:gpt-4o")  # Higher quality model for accuracy checking


@tool
def database_query_tool(query: str, database_name: str) -> str:
    """Executes a natural language query against the database. Use this for questions that are already contained in any of the available datasets."""
    
    database_info = DATABASE[database_name]
    if database_info["type"] == "database":
        file_name = database_info["file_name"]
        
        # Handle relative SQLite paths by making them absolute
        if file_name.startswith("sqlite:///") and not file_name.startswith("sqlite:////"):
            # It's a relative SQLite path, make it absolute
            db_file = file_name.replace("sqlite:///", "")
            absolute_db_path = os.path.join(pipeline_dir, db_file)
            file_name = f"sqlite:///{absolute_db_path}"
        
        db = SQLDatabase.from_uri(file_name)
    else:
        raise ValueError(f"Invalid database type: {database_info['type']}")
    
    agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True, top_k=1_000)
    return agent.invoke({"input": query})["output"]

llm_with_tools = llm.bind_tools([database_query_tool])

# Configuration for thresholds
RELEVANCE_THRESHOLD = 0.7  # User-configurable threshold (0.0 to 1.0)
ACCURACY_THRESHOLD = 0.8   # User-configurable threshold for accuracy check (0.0 to 1.0)
MAX_RETRIES = 2           # Maximum number of retries before proceeding with warnings

def get_message_content(message):
    """Helper function to extract content from both dict and LangChain message objects."""
    if hasattr(message, 'content'):
        return message.content
    elif isinstance(message, dict):
        return message.get('content', '')
    else:
        return str(message)

def get_message_role(message):
    """Helper function to extract role from both dict and LangChain message objects."""
    if hasattr(message, 'role'):
        return message.role
    elif isinstance(message, dict):
        return message.get('role', '')
    else:
        return 'unknown'

def chatbot(state: State):
    # First, generate the regular response with tools
    system_message = {
        "role": "system", 
        "content": ""
    }

    for database_name, database_info in DATABASE.items():
        system_message["content"] += f"You have access to the following database: {database_name}. {database_info['description']}\n"

    system_message["content"] += "Until you are confident about the answer, you must use the database_query_tool to answer the question. If you are not confident about the answer, just give your best guess. If the question is not related to the databases, you must politely decline and ask for a database-related question instead."
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Now ask the LLM to evaluate if this response needs relevance checking
    evaluation_prompt = f"""You just provided a response to a user query. Now evaluate whether your response needs relevance checking.

Your response was: {get_message_content(response)}

Relevance checking should be SKIPPED (false) when:
- You properly declined/rejected the question as irrelevant to databases
- The question is completely unrelated to available databases  
- You provided a polite refusal with no database information

Relevance checking is NEEDED (true) when:
- You used database tools or provided database information
- You attempted to answer a database-related question
- You provided any substantive response about the databases

Respond in JSON format:
{{
    "needs_relevance_check": true/false,
    "reasoning": "brief explanation for your decision"
}}"""

    try:
        # Use a separate LLM call to get the structured evaluation
        eval_thread_id = f"eval_{uuid.uuid4().hex[:8]}"
        evaluation_response = llm.invoke([{"role": "user", "content": evaluation_prompt}])
        eval_data = repair_json(evaluation_response.content, return_objects=True)
        
        needs_relevance_check = eval_data.get("needs_relevance_check", True)  # Default to True for safety
        reasoning = eval_data.get("reasoning", "No reasoning provided")
        
        print(f"🎯 LLM decision - Relevance checking needed: {needs_relevance_check}")
        print(f"💭 Reasoning: {reasoning}")
        
    except Exception as e:
        print(f"⚠️  Error in relevance evaluation: {e}, defaulting to needs_relevance_check=True")
        needs_relevance_check = True  # Default to checking for safety
    
    # Pass through state fields and set relevance check flag
    return {
        "messages": [response],
        "original_query": state.get("original_query"),
        "thread_id": state.get("thread_id"),
        "needs_relevance_check": needs_relevance_check
    }


class RelevanceChecker:
    """A node that checks the relevance of the LLM's response and handles low-quality answers."""
    
    def __init__(self, threshold: float = RELEVANCE_THRESHOLD):
        self.threshold = threshold
        print(f"🎯 RelevanceChecker initialized with threshold: {threshold}")
    
    def __call__(self, state: State):
        print("🔍 RelevanceChecker: Evaluating response quality...")
        
        # Initialize retry tracking
        relevance_retry_count = state.get("relevance_retry_count", 0)
        retry_warnings = state.get("retry_warnings", [])
        
        # Check if relevance checking is needed
        needs_relevance_check = state.get("needs_relevance_check", True)
        if not needs_relevance_check:
            print("🚫 LLM determined relevance check not needed - skipping")
            # Create a basic evaluation for responses that didn't need checking
            skip_evaluation = {
                "relevance_score": "N/A",
                "reasoning": "Response was determined to not require relevance checking (likely a decline/refusal)",
                "issues": [],
                "passed_threshold": True
            }
            return {
                "messages": [], 
                "retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": needs_relevance_check,
                "relevance_evaluation": skip_evaluation,
                "relevance_retry_count": relevance_retry_count,
                "retry_warnings": retry_warnings
            }
        
        if not (messages := state.get("messages", [])):
            print("❌ No messages to evaluate")
            no_message_evaluation = {
                "relevance_score": "N/A",
                "reasoning": "No messages available for evaluation",
                "issues": ["No messages to evaluate"],
                "passed_threshold": False
            }
            return {
                "messages": [], 
                "retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": no_message_evaluation
            }
        
        # Get the original query, AI response, and query execution info from state
        original_query = state.get("original_query", "")
        last_message = messages[-1]
        executed_queries = state.get("executed_queries", [])
        query_execution_log = state.get("query_execution_log", {})
        
        if not original_query:
            print("⚠️  No original query in state, skipping relevance check")
            no_query_evaluation = {
                "relevance_score": "N/A",
                "reasoning": "No original query available for relevance evaluation",
                "issues": ["No original query in state"],
                "passed_threshold": True  # Allow through since we can't evaluate
            }
            return {
                "messages": [], 
                "retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": no_query_evaluation,
                "relevance_retry_count": relevance_retry_count,
                "retry_warnings": retry_warnings,
                "executed_queries": executed_queries,
                "query_execution_log": query_execution_log
            }
        
        # Build query execution summary for context
        query_summary = ""
        if executed_queries:
            query_summary = "\n\nExecuted Database Queries:"
            for i, query_info in enumerate(executed_queries, 1):
                status = "✅ Success" if query_info.get("success", False) else "❌ Failed"
                query_summary += f"""
                Query {i}: {query_info.get('query', 'Unknown')}
                Database: {query_info.get('database_name', 'Unknown')}
                Status: {status}
                Result Preview: {query_info.get('result_preview', query_info.get('error', 'No result'))[:200]}...
                """
        else:
            query_summary = "\n\nNo database queries were executed."
        
        # Create relevance evaluation prompt
        relevance_prompt = f"""
        You are a quality control system. Evaluate the relevance and accuracy of an AI response to a user's question about databases.
        
        Original Question: {original_query}
        AI Response: {get_message_content(last_message)}
        {query_summary}
        
        Evaluate the response based on:
        1. Direct relevance to the question
        2. Factual accuracy (no hallucinations) 
        3. Completeness of the answer
        4. Use of appropriate database information
        5. Consistency between the response and the executed query results
        6. Whether the queries executed were appropriate for answering the question
        
        Consider:
        - Does the response directly address what the user asked?
        - Are the database queries relevant to the question?
        - Does the AI response accurately reflect the query results?
        - Are there any contradictions between the response and the actual data retrieved?
        
        Provide your evaluation in JSON format:
        {{
            "relevance_score": <score from 0.0 to 1.0>,
            "reasoning": "<detailed explanation of the score, referencing query execution where relevant>",
            "issues": ["<list of specific issues if any>"],
            "suggested_improvement": "<how to improve the response if score < 0.7>"
        }}
        """
        
        try:
            # Generate unique thread_id for this relevance check to avoid contamination
            relevance_thread_id = f"relevance_check_{uuid.uuid4().hex[:8]}"
            print(f"🧵 Using thread_id: {relevance_thread_id}")
            
            # Use LLM to evaluate relevance with unique thread
            eval_messages = [{"role": "user", "content": relevance_prompt}]
            evaluation_response = llm.invoke(eval_messages)
            eval_content = evaluation_response.content
            
            print(f"🤖 Raw evaluation: {eval_content[:200]}...")
            
            # Parse the evaluation
            eval_data = repair_json(eval_content, return_objects=True)
            relevance_score = float(eval_data.get("relevance_score", 0.0))
            reasoning = eval_data.get("reasoning", "No reasoning provided")
            issues = eval_data.get("issues", [])
            improvement = eval_data.get("suggested_improvement", "")
            
            print(f"📊 Relevance Score: {relevance_score:.2f} (Threshold: {self.threshold})")
            print(f"💭 Reasoning: {reasoning}")
            
            # Store evaluation in state for tracking
            evaluation_result = {
                "relevance_score": relevance_score,
                "reasoning": reasoning,
                "issues": issues,
                "suggested_improvement": improvement,
                "passed_threshold": relevance_score >= self.threshold
            }
            
            # Add evaluation metadata to the last message
            if hasattr(last_message, 'additional_kwargs'):
                last_message.additional_kwargs['relevance_eval'] = evaluation_result
            
            if relevance_score < self.threshold:
                print(f"🔴 Low relevance score! Issues found: {issues}")
                
                # Check if we've exceeded max retries
                if relevance_retry_count >= MAX_RETRIES:
                    print(f"⚠️  Maximum relevance retries ({MAX_RETRIES}) exceeded. Proceeding with warnings.")
                    warning_msg = f"⚠️ Response failed relevance check (score: {relevance_score:.2f}/{self.threshold}) after {MAX_RETRIES} attempts. Issues: {', '.join(issues)}"
                    retry_warnings.append(warning_msg)
                    
                    return {
                        "messages": [],
                        "retry_needed": False,  # Don't retry anymore
                        "original_query": state.get("original_query"),
                        "thread_id": state.get("thread_id"),
                        "needs_relevance_check": state.get("needs_relevance_check"),
                        "relevance_evaluation": evaluation_result,
                        "relevance_retry_count": relevance_retry_count,
                        "max_retries_exceeded": True,
                        "retry_warnings": retry_warnings,
                        "executed_queries": executed_queries,
                        "query_execution_log": query_execution_log
                    }
                
                # Increment retry count and proceed with retry
                relevance_retry_count += 1
                print(f"🔄 Attempting relevance retry {relevance_retry_count}/{MAX_RETRIES}")
                
                # Create a retry message with improved query guidance
                retry_content = f"""The previous response had a low relevance score ({relevance_score:.2f}).
Issues identified: {', '.join(issues)}

Please provide a better response to the original question: "{original_query}"

Suggested improvement: {improvement}

Make sure to:
1. Use the database_query_tool if you haven't already
2. Provide specific, factual information
3. Address the question directly
4. Avoid speculation or hallucination"""
                
                retry_message = SystemMessage(content=retry_content)
                
                # Return the retry message to trigger a new response
                return {
                    "messages": [retry_message], 
                    "retry_needed": True,
                    "original_query": state.get("original_query"),
                    "thread_id": state.get("thread_id"),
                    "needs_relevance_check": state.get("needs_relevance_check"),
                    "relevance_evaluation": evaluation_result,
                    "relevance_retry_count": relevance_retry_count,
                    "retry_warnings": retry_warnings,
                    "executed_queries": executed_queries,
                    "query_execution_log": query_execution_log
                }
            
            else:
                print("✅ Response passed relevance check!")
                return {
                    "messages": [], 
                    "retry_needed": False,
                    "original_query": state.get("original_query"),
                    "thread_id": state.get("thread_id"),
                    "needs_relevance_check": state.get("needs_relevance_check"),
                    "relevance_evaluation": evaluation_result,
                    "relevance_retry_count": relevance_retry_count,
                    "retry_warnings": retry_warnings,
                    "executed_queries": executed_queries,
                    "query_execution_log": query_execution_log
                }
                
        except Exception as e:
            print(f"❌ Error in relevance checking: {e}")
            # On error, allow the response to pass through
            error_evaluation = {
                "relevance_score": "Error",
                "reasoning": f"Error during relevance evaluation: {str(e)}",
                "issues": ["Evaluation failed"],
                "passed_threshold": True  # Pass through on error
            }
            return {
                "messages": [], 
                "retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": error_evaluation,
                "relevance_retry_count": relevance_retry_count,
                "retry_warnings": retry_warnings,
                "executed_queries": executed_queries,
                "query_execution_log": query_execution_log
            }


class AccuracyChecker:
    """A node that uses a high-quality model to check the factual accuracy of responses."""
    
    def __init__(self, threshold: float = ACCURACY_THRESHOLD):
        self.threshold = threshold
        print(f"🎯 AccuracyChecker initialized with threshold: {threshold} using high-quality model")
    
    def __call__(self, state: State):
        print("🔍 AccuracyChecker: Evaluating response accuracy with high-quality model...")
        
        # Initialize retry tracking
        accuracy_retry_count = state.get("accuracy_retry_count", 0)
        retry_warnings = state.get("retry_warnings", [])
        
        if not (messages := state.get("messages", [])):
            print("❌ No messages to evaluate for accuracy")
            no_message_evaluation = {
                "accuracy_score": "N/A",
                "reasoning": "No messages available for accuracy evaluation",
                "issues": ["No messages to evaluate"],
                "passed_threshold": False
            }
            return {
                "messages": [], 
                "accuracy_retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": no_message_evaluation,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": accuracy_retry_count,
                "retry_warnings": retry_warnings
            }
        
        # Get the original query, AI response, and query execution info from state
        original_query = state.get("original_query", "")
        last_message = messages[-1]
        executed_queries = state.get("executed_queries", [])
        query_execution_log = state.get("query_execution_log", {})
        
        if not original_query:
            print("⚠️  No original query in state, skipping accuracy check")
            no_query_evaluation = {
                "accuracy_score": "N/A",
                "reasoning": "No original query available for accuracy evaluation",
                "issues": ["No original query in state"],
                "passed_threshold": True  # Allow through since we can't evaluate
            }
            return {
                "messages": [], 
                "accuracy_retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": no_query_evaluation
            }
        
        # Skip accuracy check if relevance check indicated it wasn't needed
        needs_relevance_check = state.get("needs_relevance_check", True)
        if not needs_relevance_check:
            print("🚫 Relevance check was skipped, skipping accuracy check as well")
            skip_evaluation = {
                "accuracy_score": "N/A",
                "reasoning": "Accuracy check skipped because relevance check was not needed (likely a decline/refusal)",
                "issues": [],
                "passed_threshold": True
            }
            return {
                "messages": [], 
                "accuracy_retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": skip_evaluation
            }
        
        # Build query execution summary for accuracy evaluation
        query_summary = ""
        if executed_queries:
            query_summary = "\n\nExecuted Database Queries and Results:"
            for i, query_info in enumerate(executed_queries, 1):
                status = "✅ Success" if query_info.get("success", False) else "❌ Failed"
                query_summary += f"""
                Query {i}: {query_info.get('query', 'Unknown')}
                Database: {query_info.get('database_name', 'Unknown')}
                Status: {status}
                """
                if query_info.get("success", False):
                    query_summary += f"Full Result: {query_info.get('result', 'No result available')}\n"
                else:
                    query_summary += f"Error: {query_info.get('error', 'Unknown error')}\n"
        else:
            query_summary = "\n\nNo database queries were executed."
        
        # Create accuracy evaluation prompt focused on factual correctness
        accuracy_prompt = f"""
        You are a high-quality accuracy assessment system. Your job is to evaluate the factual accuracy and correctness of an AI response to a database query.
        
        Original Question: {original_query}
        AI Response: {get_message_content(last_message)}
        {query_summary}
        
        Evaluate the response based on:
        1. Factual accuracy of data presented
        2. Logical consistency of the information
        3. Absence of hallucinations or made-up data
        4. Correct interpretation of database results
        5. Mathematical accuracy in calculations or aggregations
        6. Proper handling of NULL values, edge cases, or data limitations
        7. CRITICAL: Verification against the actual database query results provided above
        8. Consistency between the AI response and the executed query results
        
        Be especially critical of:
        - Numbers that seem unrealistic or impossible
        - Claims that cannot be verified from the actual database results shown above
        - Logical inconsistencies in the data presentation
        - Overgeneralization from limited data
        - Discrepancies between the response and the actual query results
        - Missing information that should be available from the executed queries
        - Incorrect summarization or interpretation of the database results
        
        Cross-reference every claim in the AI response against the actual database results provided above.
        
        Provide your evaluation in JSON format:
        {{
            "accuracy_score": <score from 0.0 to 1.0>,
            "reasoning": "<detailed explanation of accuracy assessment, comparing response claims to actual database results>",
            "issues": ["<list of specific accuracy issues if any>"],
            "data_quality_concerns": ["<specific concerns about data integrity>"],
            "suggested_improvement": "<how to improve accuracy if score < {self.threshold}>"
        }}
        """
        
        try:
            # Generate unique thread_id for this accuracy check
            accuracy_thread_id = f"accuracy_check_{uuid.uuid4().hex[:8]}"
            print(f"🧵 Using thread_id for accuracy check: {accuracy_thread_id}")
            
            # Use high-quality LLM for accuracy evaluation
            eval_messages = [{"role": "user", "content": accuracy_prompt}]
            evaluation_response = llm_high_quality.invoke(eval_messages)
            eval_content = evaluation_response.content
            
            print(f"🤖 Raw accuracy evaluation: {eval_content[:200]}...")
            
            # Parse the evaluation
            eval_data = repair_json(eval_content, return_objects=True)
            accuracy_score = float(eval_data.get("accuracy_score", 0.0))
            reasoning = eval_data.get("reasoning", "No reasoning provided")
            issues = eval_data.get("issues", [])
            data_concerns = eval_data.get("data_quality_concerns", [])
            improvement = eval_data.get("suggested_improvement", "")
            
            print(f"📊 Accuracy Score: {accuracy_score:.2f} (Threshold: {self.threshold})")
            print(f"💭 Reasoning: {reasoning}")
            
            # Store evaluation in state for tracking
            evaluation_result = {
                "accuracy_score": accuracy_score,
                "reasoning": reasoning,
                "issues": issues,
                "data_quality_concerns": data_concerns,
                "suggested_improvement": improvement,
                "passed_threshold": accuracy_score >= self.threshold
            }
            
            # Add evaluation metadata to the last message
            if hasattr(last_message, 'additional_kwargs'):
                last_message.additional_kwargs['accuracy_eval'] = evaluation_result
            
            if accuracy_score < self.threshold:
                print(f"🔴 Low accuracy score! Issues found: {issues}")
                print(f"🔴 Data quality concerns: {data_concerns}")
                
                # Check if we've exceeded max retries
                if accuracy_retry_count >= MAX_RETRIES:
                    print(f"⚠️  Maximum accuracy retries ({MAX_RETRIES}) exceeded. Proceeding with warnings.")
                    warning_msg = f"⚠️ Response failed accuracy check (score: {accuracy_score:.2f}/{self.threshold}) after {MAX_RETRIES} attempts. Issues: {', '.join(issues)}"
                    retry_warnings.append(warning_msg)
                    if data_concerns:
                        warning_msg += f" Data quality concerns: {', '.join(data_concerns)}"
                    
                    return {
                        "messages": [],
                        "accuracy_retry_needed": False,  # Don't retry anymore
                        "original_query": state.get("original_query"),
                        "thread_id": state.get("thread_id"),
                        "needs_relevance_check": state.get("needs_relevance_check"),
                        "relevance_evaluation": state.get("relevance_evaluation"),
                        "accuracy_evaluation": evaluation_result,
                        "relevance_retry_count": state.get("relevance_retry_count", 0),
                        "accuracy_retry_count": accuracy_retry_count,
                        "max_retries_exceeded": True,
                        "retry_warnings": retry_warnings,
                        "executed_queries": executed_queries,
                        "query_execution_log": query_execution_log
                    }
                
                # Increment retry count and proceed with retry
                accuracy_retry_count += 1
                print(f"🔄 Attempting accuracy retry {accuracy_retry_count}/{MAX_RETRIES}")
                
                # Create a retry message with improved accuracy guidance
                retry_content = f"""The previous response had a low accuracy score ({accuracy_score:.2f}).

Accuracy Issues Identified: {', '.join(issues)}
Data Quality Concerns: {', '.join(data_concerns)}

Please provide a more accurate response to the original question: "{original_query}"

Suggested improvement: {improvement}

Make sure to:
1. Double-check all numerical data and calculations
2. Verify facts against database results carefully
3. Avoid speculation or assumptions about data
4. Be precise about data limitations or uncertainties
5. Use conservative language when data might be incomplete
6. Re-query the database if needed to verify information"""
                
                retry_message = SystemMessage(content=retry_content)
                
                # Return the retry message to trigger a new response
                return {
                    "messages": [retry_message], 
                    "accuracy_retry_needed": True,
                    "original_query": state.get("original_query"),
                    "thread_id": state.get("thread_id"),
                    "needs_relevance_check": state.get("needs_relevance_check"),
                    "relevance_evaluation": state.get("relevance_evaluation"),
                    "accuracy_evaluation": evaluation_result,
                    "relevance_retry_count": state.get("relevance_retry_count", 0),
                    "accuracy_retry_count": accuracy_retry_count,
                    "retry_warnings": retry_warnings,
                    "executed_queries": executed_queries,
                    "query_execution_log": query_execution_log
                }
            
            else:
                print("✅ Response passed accuracy check!")
                return {
                    "messages": [], 
                    "accuracy_retry_needed": False,
                    "original_query": state.get("original_query"),
                    "thread_id": state.get("thread_id"),
                    "needs_relevance_check": state.get("needs_relevance_check"),
                    "relevance_evaluation": state.get("relevance_evaluation"),
                    "accuracy_evaluation": evaluation_result,
                    "relevance_retry_count": state.get("relevance_retry_count", 0),
                    "accuracy_retry_count": accuracy_retry_count,
                    "retry_warnings": retry_warnings,
                    "executed_queries": executed_queries,
                    "query_execution_log": query_execution_log
                }
                
        except Exception as e:
            print(f"❌ Error in accuracy checking: {e}")
            # On error, allow the response to pass through
            error_evaluation = {
                "accuracy_score": "Error",
                "reasoning": f"Error during accuracy evaluation: {str(e)}",
                "issues": ["Evaluation failed"],
                "data_quality_concerns": [],
                "passed_threshold": True  # Pass through on error
            }
            return {
                "messages": [], 
                "accuracy_retry_needed": False,
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": error_evaluation,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": accuracy_retry_count,
                "retry_warnings": retry_warnings,
                "executed_queries": executed_queries,
                "query_execution_log": query_execution_log
            }


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        print(f"🔧 ToolNode initialized with {len(tools)} tools: {list(self.tools_by_name.keys())}")

    def __call__(self, inputs: dict):
        print(f"🛠️ ToolNode executing with inputs: {list(inputs.keys())}")
        
        if messages := inputs.get("messages", []):
            message = messages[-1]
            print(f"📨 Processing message: {message.content[:100]}...")
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        executed_queries = inputs.get("executed_queries", [])
        query_execution_log = inputs.get("query_execution_log", {})
        
        print(f"🔍 Found {len(message.tool_calls)} tool calls to execute")
        
        for i, tool_call in enumerate(message.tool_calls):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"⚙️ Executing tool {i+1}/{len(message.tool_calls)}: {tool_name}")
            print(f"📋 Tool arguments: {tool_args}")
            
            try:
                tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                print(f"✅ Tool {tool_name} executed successfully")
                print(f"📊 Result preview: {str(tool_result)[:200]}...")
                
                # Track query execution for database_query_tool
                if tool_name == "database_query_tool":
                    query_info = {
                        "tool_call_id": tool_id,
                        "query": tool_args.get("query", "Unknown query"),
                        "database_name": tool_args.get("database_name", "Unknown database"),
                        "result": tool_result,
                        "result_preview": str(tool_result)[:500],  # First 500 chars for preview
                        "success": True,
                        "timestamp": f"call_{i+1}"
                    }
                    executed_queries.append(query_info)
                    query_execution_log[f"call_{tool_id}"] = query_info
                    print(f"📝 Logged query execution: {tool_args.get('query', 'Unknown')[:50]}...")
                
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
            except Exception as e:
                print(f"❌ Error executing tool {tool_name}: {str(e)}")
                error_result = f"Error executing {tool_name}: {str(e)}"
                
                # Track failed query execution for database_query_tool
                if tool_name == "database_query_tool":
                    query_info = {
                        "tool_call_id": tool_id,
                        "query": tool_args.get("query", "Unknown query"),
                        "database_name": tool_args.get("database_name", "Unknown database"),
                        "result": None,
                        "error": str(e),
                        "success": False,
                        "timestamp": f"call_{i+1}"
                    }
                    executed_queries.append(query_info)
                    query_execution_log[f"call_{tool_id}"] = query_info
                    print(f"📝 Logged failed query execution: {tool_args.get('query', 'Unknown')[:50]}...")
                
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": error_result}),
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
        
        print(f"🎯 ToolNode completed, returning {len(outputs)} tool messages with {len(executed_queries)} queries logged")
        return {
            "messages": outputs,
            "original_query": inputs.get("original_query"),
            "thread_id": inputs.get("thread_id"),
            "needs_relevance_check": inputs.get("needs_relevance_check"),
            "relevance_retry_count": inputs.get("relevance_retry_count", 0),
            "accuracy_retry_count": inputs.get("accuracy_retry_count", 0),
            "max_retries_exceeded": inputs.get("max_retries_exceeded", False),
            "retry_warnings": inputs.get("retry_warnings", []),
            "executed_queries": executed_queries,
            "query_execution_log": query_execution_log
        }


class ReasoningAndCitationNode:
    """A node that generates reasoning for the response and identifies dataset citations."""
    
    def __call__(self, state: State):
        print("🧠 ReasoningAndCitationNode: Generating reasoning and citations...")
        
        if not (messages := state.get("messages", [])):
            print("❌ No messages to analyze for reasoning and citations")
            return {
                "messages": [], 
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": None,
                "citation_analysis": None,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }
        
        # Get the original query and conversation context
        original_query = state.get("original_query", "")
        
        # Find the latest AI response (non-system message)
        latest_ai_response = None
        tool_calls_made = []
        
        for message in reversed(messages):
            if hasattr(message, 'content') and message.content:
                role = getattr(message, 'role', 'unknown')
                if role == 'assistant' or hasattr(message, 'tool_calls'):
                    latest_ai_response = message.content
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_made.extend([call.get('name', 'unknown') for call in message.tool_calls])
                    break
                elif role == 'tool':
                    # Extract tool information
                    tool_name = getattr(message, 'name', 'unknown')
                    if tool_name not in tool_calls_made:
                        tool_calls_made.append(tool_name)
        
        if not latest_ai_response:
            print("⚠️  No AI response found for reasoning analysis")
            return {
                "messages": [], 
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": None,
                "citation_analysis": None,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }
        
        # Generate reasoning analysis
        reasoning_prompt = f"""
        You are an AI reasoning analyst. Analyze the following response and provide clear reasoning for why this response is correct and reliable.

        Original Question: {original_query}
        AI Response: {latest_ai_response}
        Tools Used: {', '.join(tool_calls_made) if tool_calls_made else 'None'}
        Available Databases: {', '.join(DATABASE.keys())}
        

        Generate a comprehensive reasoning analysis that includes:
        1. Why the response accurately addresses the user's question
        2. How the data sources support the conclusions
        3. The logical steps taken to arrive at the answer
        4. Any assumptions made and their validity
        5. Confidence level and any limitations

        Provide your analysis in JSON format:
        {{
            "reasoning_summary": "<concise summary of why this response is correct>",
            "data_validation": "<explanation of how data sources validate the response>",
            "logical_steps": ["<step 1>", "<step 2>", "<step 3>"],
            "assumptions": ["<assumption 1>", "<assumption 2>"],
            "confidence_factors": ["<factor 1>", "<factor 2>"],
            "limitations": ["<limitation 1>", "<limitation 2>"],
            "overall_confidence": "<High/Medium/Low with explanation>"
        }}
        """

        # Generate citation analysis
        citation_prompt = f"""
        You are a dataset citation analyst. Identify and format citations for all datasets used in generating this response.

        Original Question: {original_query}
        AI Response: {latest_ai_response}
        Tools Used: {', '.join(tool_calls_made) if tool_calls_made else 'None'}
        Available Databases: {list(DATABASE.keys())}
        Database Details: {DATABASE}

        Generate comprehensive citations that include:
        1. Which specific databases/datasets were consulted
        2. What type of data was extracted from each
        3. The relevance of each dataset to the query
        4. Any data transformations or calculations performed

        Provide your analysis in JSON format:
        {{
            "primary_datasets": ["<dataset 1>", "<dataset 2>"],
            "dataset_usage": {{
                "<dataset_name>": {{
                    "description": "<what this dataset contains>",
                    "data_extracted": "<what specific data was used>",
                    "relevance": "<why this dataset is relevant to the query>",
                    "confidence": "<High/Medium/Low>"
                }}
            }},
            "data_sources_summary": "<overall summary of data sources used>",
            "methodology": "<description of how data was processed/analyzed>"
        }}
        """

        try:
            # Generate unique thread IDs for analysis
            reasoning_thread_id = f"reasoning_{uuid.uuid4().hex[:8]}"
            citation_thread_id = f"citation_{uuid.uuid4().hex[:8]}"
            
            print(f"🧵 Generating reasoning analysis...")
            reasoning_response = llm.invoke([{"role": "user", "content": reasoning_prompt}])
            reasoning_data = repair_json(reasoning_response.content, return_objects=True)
            
            print(f"🧵 Generating citation analysis...")
            citation_response = llm.invoke([{"role": "user", "content": citation_prompt}])
            citation_data = repair_json(citation_response.content, return_objects=True)
            
            print(f"✅ Generated reasoning and citation analysis")
            
            return {
                "messages": [], 
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": reasoning_data,
                "citation_analysis": citation_data,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }
            
        except Exception as e:
            print(f"❌ Error in reasoning and citation analysis: {e}")
            
            # Provide fallback analysis
            fallback_reasoning = {
                "reasoning_summary": "Response generated using available database tools and information",
                "data_validation": "Data validated through database queries",
                "logical_steps": ["Query received", "Database consulted", "Response generated"],
                "assumptions": ["Database data is current and accurate"],
                "confidence_factors": ["Direct database access"],
                "limitations": ["Limited by available data"],
                "overall_confidence": "Medium - Standard database query process"
            }
            
            fallback_citation = {
                "primary_datasets": list(DATABASE.keys()),
                "dataset_usage": {db: {"description": info.get("description", "Database"), 
                                     "data_extracted": "Query-relevant data", 
                                     "relevance": "Consulted for query response",
                                     "confidence": "Medium"} 
                                 for db, info in DATABASE.items()},
                "data_sources_summary": "Standard database consultation",
                "methodology": "Direct database querying"
            }
            
            return {
                "messages": [], 
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "needs_relevance_check": state.get("needs_relevance_check"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": fallback_reasoning,
                "citation_analysis": fallback_citation,
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }


class SynthesisNode:
    """A node that uses LLM to synthesize and clean up the final response for user consumption."""
    
    def __call__(self, state: State):
        print("🎨 SynthesisNode: Using LLM to synthesize clean user response...")
        
        if not (messages := state.get("messages", [])):
            return {
                "messages": [{"role": "assistant", "content": "I apologize, but I wasn't able to generate a response to your query."}],
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id")
            }
        
        # Get the original query and the conversation history
        original_query = state.get("original_query", "")
        
        # Build context from the conversation
        conversation_context = []
        relevance_info = None
        accuracy_info = None
        reasoning_info = None
        citation_info = None
        
        # Try to get evaluation information from state first
        relevance_info = state.get("relevance_evaluation")
        accuracy_info = state.get("accuracy_evaluation")
        reasoning_info = state.get("reasoning_analysis")
        citation_info = state.get("citation_analysis")
        retry_warnings = state.get("retry_warnings", [])
        max_retries_exceeded = state.get("max_retries_exceeded", False)
        executed_queries = state.get("executed_queries", [])
        query_execution_log = state.get("query_execution_log", {})
        
        for message in messages:
            if hasattr(message, 'content') and message.content:
                role = getattr(message, 'role', 'unknown')
                content = message.content
                
                # Skip system retry messages
                if role == 'system' and ('retry' in content.lower() or 'relevance' in content.lower()):
                    continue
                
                # Extract evaluation info if available (fallback from message metadata)
                if not relevance_info and hasattr(message, 'additional_kwargs') and 'relevance_eval' in message.additional_kwargs:
                    relevance_info = message.additional_kwargs['relevance_eval']
                
                if not accuracy_info and hasattr(message, 'additional_kwargs') and 'accuracy_eval' in message.additional_kwargs:
                    accuracy_info = message.additional_kwargs['accuracy_eval']
                
                conversation_context.append(f"{role}: {content}")
        
        # Build comprehensive quality assessment section
        quality_section = ""
        reasoning_section = ""
        citation_section = ""
        retry_warnings_section = ""
        
        # Add retry warnings section if warnings exist
        if retry_warnings:
            retry_warnings_section = f"""

Retry Warnings:
{chr(10).join([f'- {warning}' for warning in retry_warnings])}"""
        
        if relevance_info or accuracy_info or retry_warnings:
            quality_section = "\n\nQuality Assessment Information:"
            
            if retry_warnings:
                quality_section += retry_warnings_section
            
            if relevance_info:
                quality_section += f"""
- Relevance Score: {relevance_info.get('relevance_score', 'N/A')}/1.0
- Relevance Check: {'✅ Passed' if relevance_info.get('passed_threshold', False) else '❌ Failed'}
- Relevance Reasoning: {relevance_info.get('reasoning', 'No reasoning provided')}
- Relevance Issues: {', '.join(relevance_info.get('issues', [])) if relevance_info.get('issues') else 'None identified'}"""
            
            if accuracy_info:
                quality_section += f"""
- Accuracy Score: {accuracy_info.get('accuracy_score', 'N/A')}/1.0
- Accuracy Check: {'✅ Passed' if accuracy_info.get('passed_threshold', False) else '❌ Failed'}
- Accuracy Reasoning: {accuracy_info.get('reasoning', 'No reasoning provided')}
- Accuracy Issues: {', '.join(accuracy_info.get('issues', [])) if accuracy_info.get('issues') else 'None identified'}
- Data Quality Concerns: {', '.join(accuracy_info.get('data_quality_concerns', [])) if accuracy_info.get('data_quality_concerns') else 'None identified'}"""

        # Build reasoning section
        if reasoning_info:
            reasoning_section = f"""

Reasoning Analysis:
- Response Summary: {reasoning_info.get('reasoning_summary', 'Standard database query response')}
- Data Validation: {reasoning_info.get('data_validation', 'Data validated through database queries')}
- Logical Steps: {', '.join(reasoning_info.get('logical_steps', [])) if reasoning_info.get('logical_steps') else 'Standard query process'}
- Key Assumptions: {', '.join(reasoning_info.get('assumptions', [])) if reasoning_info.get('assumptions') else 'Database accuracy assumed'}
- Confidence Factors: {', '.join(reasoning_info.get('confidence_factors', [])) if reasoning_info.get('confidence_factors') else 'Direct database access'}
- Limitations: {', '.join(reasoning_info.get('limitations', [])) if reasoning_info.get('limitations') else 'Limited by available data'}
- Overall Confidence: {reasoning_info.get('overall_confidence', 'Medium - Standard database process')}"""

        # Build citation section  
        if citation_info:
            citation_section = f"""

Citation Information:
- Primary Datasets: {', '.join(citation_info.get('primary_datasets', [])) if citation_info.get('primary_datasets') else 'Available databases'}
- Data Sources Summary: {citation_info.get('data_sources_summary', 'Standard database consultation')}
- Methodology: {citation_info.get('methodology', 'Direct database querying')}"""
            
            # Add detailed dataset usage if available
            if citation_info.get('dataset_usage'):
                citation_section += "\n- Detailed Dataset Usage:"
                for dataset, details in citation_info['dataset_usage'].items():
                    citation_section += f"""
  * {dataset}: {details.get('description', 'Database')} 
    - Data Used: {details.get('data_extracted', 'Query-relevant data')}
    - Relevance: {details.get('relevance', 'Consulted for response')}
    - Confidence: {details.get('confidence', 'Medium')}"""

        # Create synthesis prompt
        synthesis_prompt = f"""You are a helpful AI assistant. A user asked a question about databases, and there was a conversation with tools and processing. Your job is to synthesize the conversation into a clean, helpful response for the user.

Original User Question: {original_query}

Conversation History:
{chr(10).join(conversation_context[-10:])}  # Last 10 messages to avoid token limits

{quality_section}
{reasoning_section}
{citation_section}

Please provide a clean, concise, and helpful response to the user's original question. Focus on:
1. Direct answer to their question
2. Key findings or data from the database queries
3. Clear, user-friendly language
4. Include quality assessment details (relevance and accuracy scores, reasoning) in a professional way
5. Remove any technical processing details, debug information, or system messages
6. Use the reasoning_analysis and citation_analysis data provided above to populate the accordion sections with detailed, specific information
7. Make the accordion content comprehensive but user-friendly

IMPORTANT RETRY WARNINGS:
{'- MAX RETRIES EXCEEDED: This response failed quality checks multiple times. Please treat the information with extra caution.' if max_retries_exceeded else '- All quality checks passed successfully.'}

If there are retry warnings above, you MUST include a clear warning at the beginning of your response that the answer may not be fully reliable due to failed quality checks.

IMPORTANT: Format your response using proper markdown for better readability:
- Use **bold** for emphasis
- Use bullet points or numbered lists for multiple items
- Use headers (##) for sections if needed
- Use tables if presenting tabular data
- Use code blocks for SQL queries or technical details

REQUIRED SECTIONS:
Your response MUST include these sections at the end as HTML accordions:

<div style="margin-top: 30px;">
<details class="reasoning-accordion" style="margin-bottom: 15px; border: 1px solid #e1e5e9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<summary style="padding: 12px 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; cursor: pointer; font-weight: bold; border-radius: 7px 7px 0 0; user-select: none;">
🧠 Reasoning & Analysis
</summary>
<div style="padding: 20px; background-color: #ffffff; border-radius: 0 0 7px 7px;">

**Why this response is correct and reliable:**

[Use the reasoning_analysis data to provide detailed explanations including:]
- Data validation methodology
- Logical reasoning steps  
- Confidence factors and limitations
- Overall confidence assessment

</div>
</details>

<details class="citations-accordion" style="margin-bottom: 15px; border: 1px solid #e1e5e9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<summary style="padding: 12px 16px; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; cursor: pointer; font-weight: bold; border-radius: 7px 7px 0 0; user-select: none;">
📚 Data Sources & Citations
</summary>
<div style="padding: 20px; background-color: #ffffff; border-radius: 0 0 7px 7px;">

**Datasets and sources consulted:**

[Use the citation_analysis data to provide comprehensive source information including:]
- Primary databases accessed
- Specific data extracted from each source
- Dataset relevance to the query
- Data processing methodology
- Confidence levels for each source

</div>
</details>

<details class="queries-accordion" style="border: 1px solid #e1e5e9; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<summary style="padding: 12px 16px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; cursor: pointer; font-weight: bold; border-radius: 7px 7px 0 0; user-select: none;">
🔍 Database Queries Executed
</summary>
<div style="padding: 20px; background-color: #ffffff; border-radius: 0 0 7px 7px;">

**Database queries that were executed to generate this response:**

[Use the executed_queries data to provide comprehensive query information including:]
- SQL queries executed
- Target databases accessed
- Query execution status (success/failure)
- Query results summary
- Query relevance to answering the question

</div>
</details>
</div>

Your response should be conversational and helpful, as if you're directly answering the user."""

        try:
            # Use a fresh LLM call for synthesis with unique thread ID
            synthesis_thread_id = f"synthesis_{uuid.uuid4().hex[:8]}"
            synthesis_response = llm.invoke([{"role": "user", "content": synthesis_prompt}])
            clean_response = synthesis_response.content.strip()
            
            print(f"✨ Synthesized clean response: {clean_response[:100]}...")
            
            # Create a clean message for the final response
            clean_message = {"role": "assistant", "content": clean_response}
            
            return {
                "messages": [clean_message],
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": state.get("reasoning_analysis"),
                "citation_analysis": state.get("citation_analysis"),
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {}),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }
            
        except Exception as e:
            print(f"❌ Error in synthesis: {e}")
            # Fallback to a simple response
            fallback_response = "I processed your query, but encountered an issue with response formatting. Could you please try rephrasing your question?"
            return {
                "messages": [{"role": "assistant", "content": fallback_response}],
                "original_query": state.get("original_query"),
                "thread_id": state.get("thread_id"),
                "relevance_evaluation": state.get("relevance_evaluation"),
                "accuracy_evaluation": state.get("accuracy_evaluation"),
                "reasoning_analysis": state.get("reasoning_analysis"),
                "citation_analysis": state.get("citation_analysis"),
                "relevance_retry_count": state.get("relevance_retry_count", 0),
                "accuracy_retry_count": state.get("accuracy_retry_count", 0),
                "max_retries_exceeded": state.get("max_retries_exceeded", False),
                "retry_warnings": state.get("retry_warnings", []),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {}),
                "executed_queries": state.get("executed_queries", []),
                "query_execution_log": state.get("query_execution_log", {})
            }


# Initialize nodes
tool_node = BasicToolNode(tools=[database_query_tool])
relevance_checker = RelevanceChecker(threshold=RELEVANCE_THRESHOLD)
accuracy_checker = AccuracyChecker(threshold=ACCURACY_THRESHOLD)
reasoning_and_citation_node = ReasoningAndCitationNode()
synthesis_node = SynthesisNode()

# Add nodes to graph
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("relevance_checker", relevance_checker)
graph_builder.add_node("accuracy_checker", accuracy_checker)
graph_builder.add_node("reasoning_and_citation", reasoning_and_citation_node)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("synthesis", synthesis_node)

def route_tools(state: State):
    """Route to tools if the message has tool calls, otherwise to relevance checker."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "relevance_checker"

def route_relevance_check(state: State):
    """Route based on relevance check results."""
    # Check if retry is needed
    if state.get("retry_needed", False):
        print("🔄 Relevance check indicates retry needed, routing back to chatbot")
        return "chatbot"
    else:
        print("✅ Relevance check passed, routing to accuracy checker")
        return "accuracy_checker"

def route_accuracy_check(state: State):
    """Route based on accuracy check results."""
    # Check if retry is needed
    if state.get("accuracy_retry_needed", False):
        print("🔄 Accuracy check indicates retry needed, routing back to chatbot")
        return "chatbot"
    else:
        print("✅ Accuracy check passed, routing to reasoning and citation analysis")
        return "reasoning_and_citation"

# Set up graph routing
graph_builder.add_edge(START, "chatbot")

# From chatbot: route to tools or relevance checker
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "relevance_checker": "relevance_checker"},
)

# From tools: always go back to chatbot
graph_builder.add_edge("tools", "chatbot")

# From relevance checker: either retry (back to chatbot) or go to accuracy checker
graph_builder.add_conditional_edges(
    "relevance_checker",
    route_relevance_check,
    {"chatbot": "chatbot", "accuracy_checker": "accuracy_checker"},
)

# From accuracy checker: either retry (back to chatbot) or go to reasoning and citation
graph_builder.add_conditional_edges(
    "accuracy_checker",
    route_accuracy_check,
    {"chatbot": "chatbot", "reasoning_and_citation": "reasoning_and_citation"},
)

# From reasoning and citation: always go to synthesis
graph_builder.add_edge("reasoning_and_citation", "synthesis")

# From synthesis: always end
graph_builder.add_edge("synthesis", END)

graph = graph_builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

def set_relevance_threshold(threshold: float):
    """Set the relevance threshold for quality control (0.0 to 1.0)."""
    global RELEVANCE_THRESHOLD
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    RELEVANCE_THRESHOLD = threshold
    print(f"🎯 Relevance threshold updated to: {threshold}")
    
    # Update the existing relevance checker
    global relevance_checker
    relevance_checker.threshold = threshold
    print(f"✅ RelevanceChecker updated with new threshold")

def set_accuracy_threshold(threshold: float):
    """Set the accuracy threshold for quality control (0.0 to 1.0)."""
    global ACCURACY_THRESHOLD
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    ACCURACY_THRESHOLD = threshold
    print(f"🎯 Accuracy threshold updated to: {threshold}")
    
    # Update the existing accuracy checker
    global accuracy_checker
    accuracy_checker.threshold = threshold
    print(f"✅ AccuracyChecker updated with new threshold")

def set_both_thresholds(relevance_threshold: float, accuracy_threshold: float):
    """Set both relevance and accuracy thresholds (0.0 to 1.0)."""
    set_relevance_threshold(relevance_threshold)
    set_accuracy_threshold(accuracy_threshold)

def stream_graph_updates(user_input: str):
    """Stream graph updates with relevance and accuracy checking enabled and return clean response."""
    print(f"🚀 Processing query with relevance threshold: {RELEVANCE_THRESHOLD}, accuracy threshold: {ACCURACY_THRESHOLD}")
    
    # Generate unique thread_id for this conversation
    conversation_thread_id = f"conv_{uuid.uuid4().hex[:8]}"
    
    # Initialize state with original query and unique thread_id
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "original_query": user_input,
        "thread_id": conversation_thread_id,
        "needs_relevance_check": True,  # Will be determined by chatbot
        "retry_needed": False,
        "accuracy_retry_needed": False,
        "relevance_evaluation": None,
        "accuracy_evaluation": None,
        "reasoning_analysis": None,
        "citation_analysis": None,
        "relevance_retry_count": 0,
        "accuracy_retry_count": 0,
        "max_retries_exceeded": False,
        "retry_warnings": [],
        "executed_queries": [],
        "query_execution_log": {}
    }
    
    # Use conversation-specific config
    conv_config = {"configurable": {"thread_id": conversation_thread_id}}
    print(f"🧵 Using conversation thread_id: {conversation_thread_id}")
    
    synthesized_response = None
    
    for event in graph.stream(initial_state, conv_config):
        for node_name, node_output in event.items():
            if node_name == "synthesis":
                # Extract the clean response from synthesis node
                if "messages" in node_output and node_output["messages"]:
                    synthesized_response = get_message_content(node_output["messages"][-1])
                    print("Assistant:", synthesized_response)
                break  # We have our final response
            elif node_name == "relevance_checker":
                # Don't print relevance checker output as it's for internal processing
                continue
            elif node_name == "accuracy_checker":
                # Don't print accuracy checker output as it's for internal processing
                continue
            elif node_name == "reasoning_and_citation":
                # Don't print reasoning and citation output as it's for internal processing
                continue
            elif node_name == "chatbot" and "messages" in node_output:
                # Only print final chatbot responses for debugging, not for user
                last_message = node_output["messages"][-1]
                
                if get_message_role(last_message) != "system":
                    # Don't print here anymore since synthesis will handle final output
                    pass
            elif node_name == "tools":
                # Tool execution is already logged by BasicToolNode
                pass
        
        # If we got the synthesized response, we can break early
        if synthesized_response:
            break
    
    return synthesized_response or "I apologize, but I wasn't able to generate a response to your query."


# Evaluation functions have been moved to eval.py module
# Import evaluation functions for backward compatibility
try:
    from eval import (
        demo_relevance_checking, 
        evaluate_flow, 
        test_improved_relevance_system
    )
except ImportError as e:
    print(f"⚠️  Warning: Could not import evaluation functions from eval.py: {e}")
    print("💡 Evaluation functions have been refactored to eval.py. Use 'python eval.py [command]' to run evaluations.")



# if __name__ == "__main__":
#     user_input = "Based on the available data, give me some questions about the available data. For example, how many employees are there? What is the average salary of the employees? What is the total budget for the company?"
#     stream_graph_updates(user_input)
#     while True:
#         try:
#             user_input = input("User: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 print("Goodbye!")
#                 break
#             stream_graph_updates(user_input)
#         except Exception as e:
#             print(f"❌ Error: {str(e)}")
#             break

