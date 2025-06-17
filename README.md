# SQLAgent

A powerful SQL query generator that converts natural language questions into precise SQL queries for business analytics.

## Setup

1. Create a `.env` file with your configuration:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-4  # optional, defaults to gpt-4
API_BASE_URL=your_api_base_url  # optional

# Database configuration
host=your_db_host
port=your_db_port
dbname=your_database_name
user=your_db_user
password=your_db_password
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Customizing the Query

To use the SQL agent with your own query, modify the prompt in the main function of `sql_agent.py`:

```python
if __name__ == "__main__":
    # Change this prompt to your desired business question
    prompt = "What is the top 10 countries by new partner signups for April 2025"
```

### Running the Agent

Execute the agent with:
```bash
python sql_agent.py
```

## Features

- Converts natural language to SQL queries
- Verifies query intent matches the original question
- Automatically corrects queries if needed
- Handles complex business logic and data relationships
- Provides detailed logging and error handling

## Workflow

1. **Query Generation**: Converts your question into SQL
2. **Intent Verification**: Ensures the SQL matches your intent
3. **Query Correction**: Fixes any issues if needed
4. **Execution**: Runs the query and returns formatted results

## Notes

- The agent supports queries across 'partner' and 'client' schemas
- Includes automatic retry logic for query improvement
- Provides both console output and detailed logging
- Maximum 3 attempts for query correction