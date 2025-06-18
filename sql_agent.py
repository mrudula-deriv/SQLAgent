import psycopg2
from dotenv import load_dotenv
import os
from psycopg2.extras import RealDictCursor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tabulate import tabulate
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import logging
from logging.handlers import RotatingFileHandler
import datetime
import re

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging configuration
def setup_logger():
    """Set up a logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/sql_agent.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger('sql_agent')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create logger
logger = setup_logger()

# Load environment variables from .env
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_BASE_URL = os.getenv('API_BASE_URL')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'gpt-4o')  # Default to gpt-4o for custom endpoint
llm_config = {
        'temperature': 0, 
        'model_name': OPENAI_MODEL_NAME,
        'api_key': OPENAI_API_KEY
    }
    
if API_BASE_URL:
    llm_config['base_url'] = API_BASE_URL

llm = ChatOpenAI(**llm_config)


db_params = {
    'host': os.getenv('host'),
    'port': os.getenv('port'),
    'database': os.getenv('dbname'),
    'user': os.getenv('user'),
    'password': os.getenv('password')
}

conn = psycopg2.connect(**db_params)
cursor = conn.cursor(cursor_factory=RealDictCursor)


def get_multi_schema_metadata(schemas: list[str]):
    query = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = ANY(%s)
    ORDER BY table_schema, table_name, ordinal_position;
    """
    cursor.execute(query, (schemas,))
    rows = cursor.fetchall()

    schema_dict = {}
    for row in rows:
        schema = row['table_schema']
        table = row['table_name']
        key = f"{schema}.{table}"
        if key not in schema_dict:
            schema_dict[key] = []
        schema_dict[key].append((row['column_name'], row['data_type']))
    return schema_dict

schema = get_multi_schema_metadata(['partner', 'client'])
table_info = "\n".join(
    [f"{table}: {', '.join([f'{col} ({dtype})' for col, dtype in cols])}" for table, cols in schema.items()]
)

# Define the state type
class AgentState(TypedDict):
    prompt: str
    sql_query: str
    verification_result: str
    matches_intent: bool
    results: str
    error: str
    attempt: int

def generate_sql_node(state: AgentState) -> AgentState:
    """Generate SQL query from natural language input."""
    logger.info("\n=== Generating SQL Query ===")
    logger.info(f"Input prompt: {state['prompt']}")
    
    # Configure ChatOpenAI with environment variables
    
 
    
    system_prompt = f"""You are a master SQL query generator specialized in Business Intelligence and KPI reporting. Follow these general guidelines when converting questions to SQL:

1. Data Quality Guidelines:
   - Filter out internal data (is_internal = FALSE)
   - Use appropriate date ranges for analysis 
   - Handle NULL values explicitly
   - Use DISTINCT for unique entity counts
   - Consider data completeness in calculations

2. Business Understanding:
   - Identify the core business metric being requested
   - Consider the complete business process flow
   - Account for all relevant business states
   - Apply appropriate business rules and filters

3. Data Quality:
   - Use DISTINCT when counting unique entities
   - Handle NULL values appropriately
   - Ensure date comparisons are logical
   - Consider data completeness in calculations

Available Schema:
{table_info}

IMPORTANT: Generate clean, executable SQL without any markdown or formatting."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["prompt"])
    ])

    # Clean any potential markdown formatting from the response
    sql_query = re.sub(r'^```sql\s*|^```\s*|```$', '', response.content, flags=re.MULTILINE).strip()
    
    logger.info(f"Generated SQL:\n{sql_query}")
    return {"sql_query": sql_query}

def verify_intent_node(state: AgentState) -> AgentState:
    """Verify if the SQL query matches the original intent."""
    logger.info("\n=== Verifying SQL Intent ===")
    logger.info(f"SQL to verify:\n{state['sql_query']}")
    
    
    
    system_prompt = f"""You are a SQL query interpreter and validator. Your task is to:
    1. Translate the SQL query into natural language
    2. Compare it with the original question and make sure the meaning is EXACTLY the same
    3. SHOULD STRICTLY check if the query returns exactly what is asked for in the original question
    4. STRICTLY verify the query against the available schema
    5. Check for potential issues in:
       - Table and column selection
       - Data type compatibility
       - Join conditions
       - Aggregation methods
       - Time period handling
       - Business logic interpretation

   

Available Schema:
{table_info}

Format your response as:
    SQL MEANING: [natural language translation of the SQL]
    INTENT MATCH: [True/False]
    SCHEMA VERIFICATION: [List any schema-related issues]
    BUSINESS LOGIC: [Check if the query correctly implements the business requirements]
    SUGGESTIONS: [Specific improvements if needed]"""
    
    user_prompt = f"""Original Question: {state["prompt"]}

SQL Query: {state["sql_query"]}

Please:
1. Translate the SQL query into clear business language
2. Verify every column and table exists in the schema
3. Confirm the logic matches the original question
4. Check all necessary filters and conditions"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    matches_intent = "INTENT MATCH: True" in response.content
    logger.info(f"Verification result:\n{response.content}")
    logger.info(f"Intent match: {matches_intent}")
    
    return {
        "verification_result": response.content,
        "matches_intent": matches_intent
    }

def correct_sql_node(state: AgentState) -> AgentState:
    """Correct the SQL query based on verification results."""
    logger.info("\n=== Correcting SQL Query ===")
    logger.info(f"Current SQL:\n{state['sql_query']}")
    logger.info(f"Verification feedback:\n{state['verification_result']}")
    
    # Increment attempt counter
    current_attempt = state.get("attempt", 0) + 1
    logger.info(f"Correction attempt: {current_attempt}")
    
    # Configure ChatOpenAI with environment variables
    llm_config = {
        'temperature': 0, 
        'model_name': 'gpt-4'
    }
    if API_BASE_URL:
        llm_config['base_url'] = API_BASE_URL

    system_prompt = f"""You are a SQL query corrector. Your task is to:
    1. Analyze the verification results
    2. Identify the issues that need to be fixed
    3. Generate a corrected SQL query that addresses all issues
    
    Available Schema:
    {table_info}
    
    IMPORTANT: Return ONLY the corrected SQL query. Do not include any explanations, labels, or additional text.
    The response should be a single SQL query that can be executed directly."""
    
    user_prompt = f"""Original Question: {state["prompt"]}

Original SQL Query: {state["sql_query"]}

Verification Results:
{state["verification_result"]}

Please provide ONLY the corrected SQL query without any additional text."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    logger.info(f"Corrected SQL:\n{response.content}")
    return {
        "sql_query": response.content,
        "attempt": current_attempt
    }

def execute_query_node(state: AgentState) -> AgentState:
    """Execute the SQL query and return results."""
    logger.info("\n=== Executing SQL Query ===")
    logger.info(f"Executing:\n{state['sql_query']}")
    
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(state["sql_query"])
                results = cursor.fetchall()

                if not results:
                    logger.info("Query executed successfully but returned no results")
                    return {"results": "✅ Query ran successfully, but no results were found."}

                columns = list(results[0].keys())
                rows = [list(row.values()) for row in results]

                summary = f"✅ Query successful. Retrieved {len(results)} row(s).\n"
                table = tabulate(rows, headers=columns, tablefmt="pretty")
                
                logger.info(f"Query executed successfully. Retrieved {len(results)} rows")
                return {"results": summary + "\n" + table}

    except Exception as e:
        error_msg = f"❌ Query failed:\n{str(e)}"
        logger.error(f"Query execution failed: {str(e)}")
        return {"error": error_msg}

def format_response_node(state: AgentState) -> AgentState:
    """Format the final response."""
    logger.info("\n=== Formatting Final Response ===")
    if state.get("error"):
        logger.info("Formatting error response")
        return {"results": state["error"]}
    logger.info("Formatting successful response")
    return state

def should_retry(state: AgentState) -> bool:
    """Determine if we should retry the query generation."""
    current_attempt = state.get("attempt", 0)
    matches_intent = state.get("matches_intent", False)
    should_retry = not matches_intent and current_attempt < 3
    
    logger.info(f"\n=== Retry Decision ===")
    logger.info(f"Current attempt: {current_attempt}")
    logger.info(f"Intent match: {matches_intent}")
    logger.info(f"Should retry: {should_retry}")
    
    if current_attempt >= 3:
        logger.warning("Maximum retry attempts reached. Proceeding to execution.")
    
    return should_retry

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("verify_intent", verify_intent_node)
workflow.add_node("correct_sql", correct_sql_node)
workflow.add_node("execute_query", execute_query_node)
workflow.add_node("format_response", format_response_node)

# Define the flow
workflow.add_edge("generate_sql", "verify_intent")
workflow.add_conditional_edges(
    "verify_intent",
    should_retry,
    {
        True: "correct_sql",
        False: "execute_query"
    }
)
workflow.add_edge("correct_sql", "verify_intent")
workflow.add_edge("execute_query", "format_response")
workflow.add_edge("format_response", END)

# Set the entry point
workflow.set_entry_point("generate_sql")

# Compile the graph
app = workflow.compile()

# Main execution
if __name__ == "__main__":
    logger.info("\n=== Starting SQL Agent Workflow ===")
    prompt = "What is the top 10 countries by new partner signups for April 2025"
    logger.info(f"Input prompt: {prompt}")
    
    # Initialize state
    initial_state = {
        "prompt": prompt,
        "sql_query": "",
        "verification_result": "",
        "matches_intent": False,
        "results": "",
        "error": "",
        "attempt": 0
    }
    
    # Run the workflow with recursion limit as safety measure
    logger.info("\n=== Executing Workflow ===")
    result = app.invoke(initial_state, config={"recursion_limit": 50})
    
    # Print and log results
    logger.info("\n=== Final Results ===")
    logger.info(f"Query Results:\n{result['results']}")
    print("\nFinal Results:")
    print("=============")
    print(result["results"])