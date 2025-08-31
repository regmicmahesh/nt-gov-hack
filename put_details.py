from typing import Annotated
import dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
import os
import yaml
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools import QuerySQLDataBaseTool, InfoSQLDatabaseTool
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from json_repair import repair_json

dotenv.load_dotenv()

# Load DATABASE configuration from YAML
with open("database.yaml", 'r') as file:
    config = yaml.safe_load(file)
    DATABASE = config["databases"]

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Connect to both databases
employee_db_uri = os.environ.get("EMPLOYEE_DATABASE_URI", "sqlite:///employee.db")
budget_db_uri = os.environ.get("BUDGET_DATABASE_URI", "sqlite:///budget.db")

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


@tool
def database_query_tool(query: str, database_name: str) -> str:
    """Executes a natural language query against the database. Use this for questions that are already contained in any of the available datasets."""
    
    database_info = DATABASE[database_name]
    if database_info["type"] == "database":
        db = SQLDatabase.from_uri(database_info["file_name"])
    else:
        raise ValueError(f"Invalid database type: {database_info['type']}")
    
    agent = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=True)
    return agent.invoke({"input": query})["output"]

llm_with_tools = llm.bind_tools([database_query_tool])

def chatbot(state: State):
    # Add system prompt to guide the LLM
    system_message = {
        "role": "system", 
        "content": ""
    }

    for database_name, database_info in DATABASE.items():
        system_message["content"] += f"You have access to the following database: {database_name} {database_info['description']}\n"
    
    messages = [system_message] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}




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
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": error_result}),
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )
        
        print(f"🎯 ToolNode completed, returning {len(outputs)} tool messages")
        return {"messages": outputs}



tool_node = BasicToolNode(tools=[database_query_tool])
graph_builder.add_node("tools", tool_node)

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "chatbot")

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# if __name__ == "__main__":
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


def display_database_schema(db, db_name):
    """Display comprehensive database schema information like PostgreSQL extended mode."""
    
    print(f"🗃️  DATABASE SCHEMA: {db_name.upper()}")
    print("=" * 60)
    
    # Get database info
    try:
        db_info = db.get_context()
        
        # Display table information
        tables = db.get_usable_table_names()
        print(f"📋 Tables ({len(tables)}): {', '.join(tables)}")
        print("-" * 60)
        
        # Display detailed schema for each table
        for table_name in tables:
            print(f"📊 TABLE: {table_name}")
            print("-" * 30)
            
            # Get table schema
            try:
                # Use direct SQL to get detailed column information
                schema_query = f"""
                PRAGMA table_info({table_name});
                """
                
                schema_result = db.run(schema_query)
                
                if schema_result:
                    lines = schema_result.strip().split('\n')
                    print("   Column Details:")
                    print("   " + "-" * 50)
                    
                    for line in lines:
                        if line.strip():
                            # Parse the column info (cid|name|type|notnull|dflt_value|pk)
                            parts = line.split('|')
                            if len(parts) >= 6:
                                col_name = parts[1]
                                col_type = parts[2]
                                not_null = "NOT NULL" if parts[3] == '1' else "NULLABLE"
                                default = f"DEFAULT {parts[4]}" if parts[4] != 'None' else ""
                                primary_key = "PRIMARY KEY" if parts[5] == '1' else ""
                                
                                print(f"   • {col_name:15} {col_type:12} {not_null:10} {default:15} {primary_key}")
                
                # Get sample data (first 3 rows)
                sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
                sample_result = db.run(sample_query)
                
                if sample_result and sample_result.strip():
                    print(f"\n   Sample Data (first 3 rows):")
                    print("   " + "-" * 40)
                    sample_lines = sample_result.strip().split('\n')
                    for line in sample_lines:
                        print(f"   {line}")
                
                # Get row count
                count_query = f"SELECT COUNT(*) as total_rows FROM {table_name};"
                count_result = db.run(count_query)
                if count_result:
                    print(f"\n   📈 Total Rows: {count_result.strip()}")
                
            except Exception as e:
                print(f"   ⚠️  Error getting schema for {table_name}: {e}")
            
            print("\n")
        
    except Exception as e:
        print(f"❌ Error getting database info: {e}")
    
    print("=" * 60)

def put_details():
    yaml_file = "database.yaml"
    output_file = "database_with_details.yaml"
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    for database_name, database_info in yaml_data["databases"].items():
        description = graph.invoke({"messages": [{"role": "user", "content": f"Please give me the detailed schema of the {database_name} database. Include details about the tables, columns, data types, primary keys, foreign keys, indexes, constraints, etc. Also try to infer some high level correlations among the columns."}]}, config=config)
        yaml_data["databases"][database_name]["description"] = description["messages"][-1].content

    with open(output_file, 'w') as file:
        yaml.dump(yaml_data, file)
    
if __name__ == "__main__":
    put_details()
    
    
        