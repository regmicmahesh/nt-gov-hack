import getpass
import os
import dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase



dotenv.load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_community.agent_toolkits import create_sql_agent




engine = create_engine("sqlite:///budget.db")
db = SQLDatabase(engine=engine)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

