import sqlite3
import os
import json
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any, Union
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from datetime import datetime
import psutil
import re
import numpy as np
import sqlparse
import os
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

from tqdm import tqdm
model_name ='meta-llama/Llama-3.2-1B-Instruct'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sft_config = {
    "train_data_path": "spider_data/train_spider.json",
    "train_split": 0.7,
    "val_split": 0.3,
    "random_seed": 42,
}

config = {
    # Data configuration
    "train_data_path": "spider_data/train_spider.json",
    "train_split": 0.7,
    "val_split": 0.3,
    "random_seed": 42,

    
    # Training configuration
    "learning_rate": 1e-6,
    "batch_size": 4, 
    "mini_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_generations": 8,
    "num_epochs": 3,
    "max_length": 1024,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "kl_penalty": 0.1,
    "max_completion_length": 1000,
    "max_prompt_length": 10000,
    # Checkpointing and logging
    "checkpoint_steps": 1000,
    "hub_push_steps": 5000,
    "eval_steps": 500,
    "logging_steps": 10,
    
    # Reward function weights
    "reward_weights": {
        "format": 0.3,      # SQL format validity
        "execution": 0.7  # Execution accuracy
    },
    
    # Model and hub configuration
    "push_to_hub": True,
    "hub_model_id": f"aman313/llama-text2sql-grpo",
    "wandb_project": "text2sql-grpo",
    "wandb_run_name": f"grpo-{model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
}
    


def compute_result_accuracy(gold_queries, pred_queries, db_ids, db_path, column_order_insensitive=False, verbose=False):
    '''
       gold_queries: list of ground truth queries
       pred_queries: list of predicted queries
       db_ids: list of database ids on which the queries are executed
       db_path: path to the database files
       column_order_insensitive: if True, ignore column order when comparing results

       Group all the queries by db_id and compute the accuracy for each db.
       load the sqllite db corresponding to each db_id and get the results for gold and predicted queries.
       for each pair of gold and predicted queries, check if the results are the same.
       compute the accuracy for each db and return the average accuracy
       return the average accuracy
    '''
    
    # Group queries by database ID
    db_query_groups = defaultdict(list)
    for gold_query, pred_query, db_id in zip(gold_queries, pred_queries, db_ids):
        db_query_groups[db_id].append((gold_query, pred_query))
    
    db_accuracies = []
    
    # Process each database
    for db_id, query_pairs in db_query_groups.items():
        if verbose:
            for gold_query, pred_query in query_pairs:
                print(f"Gold: {gold_query}, Pred: {pred_query}")
                print('-'*100)

        # Construct database path
        db_file_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_file_path):
            print(f"Warning: Database file not found: {db_file_path}")
            continue
        
        # Calculate accuracy for this database
        db_accuracy = _calculate_db_accuracy(query_pairs, db_file_path, column_order_insensitive)
        db_accuracies.append(db_accuracy)
    
    # Return average accuracy across all databases
    return sum(db_accuracies) / len(db_accuracies) if db_accuracies else 0.0


def _calculate_db_accuracy(query_pairs: List[Tuple[str, str]], db_file_path: str, column_order_insensitive: bool = False) -> float:
    """Calculate accuracy for a single database."""
    if not query_pairs:
        return 0.0
    
    correct_count = 0
    total_count = len(query_pairs)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        
        for gold_query, pred_query in query_pairs:
            if pred_query == '':
                continue
            correct, gold_set, pred_set = _compare_query_results(cursor, gold_query, pred_query, column_order_insensitive)
            if correct:
                correct_count += 1
                # print(f"Gold: {gold_query}, Pred: {pred_query}")
                # print('-'*100)
                # print(f"Gold: {gold_set}, Pred: {pred_set}")
                # print('-'*100)
        
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to database {db_file_path}: {e}")
        return 0.0
    
    return correct_count / total_count


def _compare_query_results(cursor, gold_query: str, pred_query: str, column_order_insensitive: bool = False) -> Tuple[bool, Set[Tuple], Set[Tuple]]:
    """Compare results of gold and predicted queries."""
    try:
        # Execute gold query
        cursor.execute(gold_query)
        gold_results = cursor.fetchall()
        gold_columns = [desc[0] for desc in cursor.description]
        
        # Execute predicted query
        cursor.execute(pred_query)
        pred_results = cursor.fetchall()
        pred_columns = [desc[0] for desc in cursor.description]
        
        # Convert results to sets for comparison
        if column_order_insensitive:
            gold_set = _normalize_results_to_set_with_columns(gold_results, gold_columns)
            pred_set = _normalize_results_to_set_with_columns(pred_results, pred_columns)
        else:
            gold_set = _normalize_results_to_set(gold_results)
            pred_set = _normalize_results_to_set(pred_results)

        ### pretty print the results
        # for gold_row, pred_row in zip(gold_results, pred_results):
        #     print(f"Gold: {gold_row}, Pred: {pred_row}")
        #     print('-'*100)

        
        # Compare sets
        return (gold_set == pred_set, gold_set, pred_set)
        
    except Exception as e:
        # If predicted query fails, count as incorrect
        # But we still need to check if gold query works for validation
        try:
            cursor.execute(gold_query)
            return (False, set(), set())  # Gold query works, predicted failed -> incorrect
        except Exception:
            # Both queries failed - this shouldn't happen with gold queries
            print(f"Warning: Gold query also failed: {e}")
            return (False, set(), set())


def _normalize_results_to_set(results: List[Tuple]) -> Set[Tuple]:
    """Normalize query results to a set for comparison."""
    if not results:
        return set()
    
    # Convert each row to tuple and create set
    normalized_results = set()
    for row in results:
        # Convert each value to string to handle different data types consistently
        normalized_row = tuple(str(val) if val is not None else None for val in row)
        normalized_results.add(normalized_row)
    
    return normalized_results


def _normalize_results_to_set_with_columns(results: List[Tuple], columns: List[str]) -> Set[frozenset]:
    """Normalize query results to a set for column-order-insensitive comparison."""
    if not results:
        return set()
    
    # Convert each row to a frozenset of (column, value) pairs
    normalized_results = set()
    for row in results:
        # Create a frozenset of (column_name, value) pairs
        row_dict = {}
        for col_name, val in zip(columns, row):
            row_dict[col_name] = str(val) if val is not None else None
        
        # Convert to frozenset for hashability
        normalized_row = frozenset(row_dict.items())
        normalized_results.add(normalized_row)
    
    return normalized_results


class Text2SQLDataset:
    def __init__(self, data_path: str,  tokenizer: AutoTokenizer,tables_path: str = "spider_data/tables.json", 
                 max_examples: int = -1, db_path: str = "spider_data/database", max_prompt_length: int = 4096):
        """
        Initialize Text2SQL dataset for instruction fine-tuning.
        
        Args:
            data_path: Path to train_spider.json, train_others.json, dev.json, etc.
            tables_path: Path to tables.json for schema information
            max_examples: Maximum number of examples to load (for testing)
            db_path: Path to database directory for schema files
        """
        self.data_path = data_path
        self.tables_path = tables_path
        self.max_examples = max_examples
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        # Load data
        self.examples = self._load_data()
        self.tables = self._load_tables()
        
        # Process examples
        self.processed_examples = [self._format_example(ex) for ex in self.examples]
        self.processed_examples = [ex for ex in self.processed_examples if ex is not None]
    
    def _load_data(self) -> List[Dict]:
        """Load and parse the JSON data file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if self.max_examples>0:
            data = data[:self.max_examples]
            
        return data
    
    def _load_tables(self) -> Dict[str, Dict]:
        """Load table schema information."""
        with open(self.tables_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)
        
        # Convert to dict with db_id as key
        tables_dict = {}
        for table_info in tables_data:
            tables_dict[table_info['db_id']] = table_info
            
        return tables_dict
    
    def _get_schema_from_file(self, db_id: str) -> Union[str, None]:
        """Get schema from schema.sql file if available."""
        schema_path = os.path.join(self.db_path, db_id, "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                # only read create table statements
                return '\n'.join([line for line in f.readlines() if line.strip().startswith('CREATE TABLE')])
        return None
    
    def _create_schema_from_tables_json(self, table_info: Dict) -> str:
        """Create CREATE TABLE statements from tables.json information."""
        schema_parts = []
        
        table_names = table_info['table_names_original']
        column_names = table_info['column_names_original']
        column_types = table_info['column_types']
        primary_keys = table_info['primary_keys']
        foreign_keys = table_info['foreign_keys']
        
        # Group columns by table
        table_columns = {}
        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:  # Skip the * column
                continue
            if table_idx not in table_columns:
                table_columns[table_idx] = []
            table_columns[table_idx].append((col_idx, col_name, column_types[col_idx]))
        
        # Create CREATE TABLE statements
        for table_idx, table_name in enumerate(table_names):
            if table_idx not in table_columns:
                continue
                
            create_stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n'
            
            # Add columns
            column_stmts = []
            for col_idx, col_name, col_type in table_columns[table_idx]:
                # Map column types
                sql_type = self._map_column_type(col_type)
                column_stmts.append(f'"{col_name}" {sql_type}')
            
            create_stmt += ',\n'.join(column_stmts)
            
            # Add primary key
            pk_cols = []
            for col_idx in primary_keys:
                for c_idx, c_name, _ in table_columns[table_idx]:
                    if c_idx == col_idx:
                        pk_cols.append(f'"{c_name}"')
            
            if pk_cols:
                create_stmt += f',\nPRIMARY KEY ({",".join(pk_cols)})'
            
            # Add foreign keys
            for fk_col, ref_col in foreign_keys:
                # Find the column names and referenced table
                fk_name = None
                ref_table = None
                ref_col_name = None
                
                for c_idx, c_name, _ in table_columns[table_idx]:
                    if c_idx == fk_col:
                        fk_name = c_name
                        break
                
                for ref_table_idx, ref_table_cols in table_columns.items():
                    for c_idx, c_name, _ in ref_table_cols:
                        if c_idx == ref_col:
                            ref_table = table_names[ref_table_idx]
                            ref_col_name = c_name
                            break
                    if ref_table:
                        break
                
                if fk_name and ref_table and ref_col_name:
                    create_stmt += f',\nFOREIGN KEY ("{fk_name}") REFERENCES `{ref_table}`("{ref_col_name}")'
            
            create_stmt += '\n);'
            schema_parts.append(create_stmt)
        
        return '\n'.join(schema_parts)
    
    def _map_column_type(self, col_type: str) -> str:
        """Map column type to SQL type."""
        type_mapping = {
            'text': 'text',
            'number': 'int',
            'time': 'text',
            'boolean': 'boolean',
            'others': 'text'
        }
        return type_mapping.get(col_type, 'text')
    
    def _format_example(self, example: Dict) -> Union[Dict, None]:
        """Format single example for instruction fine-tuning."""
        db_id = example['db_id']
        question = example['question']
        query = example['query']
        
        # Get schema
        #schema = self._get_schema_from_file(db_id)
        #if not schema and db_id in self.tables:
        schema = self._create_schema_from_tables_json(self.tables[db_id])
        
        if not schema:
            schema = f"-- Schema for {db_id} not available"
        
        # Format input prompt
        input_text = f"""Given this schema:
            {schema}

            Write an SQL query for the below requirement. Only return the query, no other text.          
            {question}"""
        #tokenized_input = self.tokenizer.encode(input_text, return_tensors="pt")
        if len(input_text) > 0.75*self.max_prompt_length:
            logger.warning(f"Prompt length exceeded {self.max_prompt_length} tokens. Skipping example {len(input_text)}.")
            return None
        
        return {
            "input": input_text,
            "output": query,
            "db_id": db_id,
            "question": question,
            "query": query
        }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.processed_examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single example."""
        return self.processed_examples[idx]
    
def generate_predictions_vllm(model_or_model_name, dataset, batch_size=64):
    '''
    Generate predictions using vLLM for efficient inference
    
    Args:
        model_name: the model to use for prediction
        dataset: the dataset to use for prediction
        
    Returns:
        List of predicted SQL queries
    '''
    try:
        from vllm import LLM, SamplingParams
        logger.info(f"Loading model {model_name} with vLLM")
        
        if isinstance(model_or_model_name, str):
            llm = LLM(
                model=model_name,
                max_model_len=10000 ,          # Keep model's full capacity
                max_num_batched_tokens=10000, # Increase to match model capacity
                tensor_parallel_size=1,      # Single process for macOS
                gpu_memory_utilization=0.3
            )
        else:
            llm = model_or_model_name
        # Configure sampling parameters with defaults
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop=None
        )
        
        logger.info("Model loaded successfully")
        
    except ImportError:
        error_msg = "vLLM is not installed. Please install vLLM to use this function."
        logger.error(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Failed to load model {model_name} with vLLM: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _extract_query(response):
        '''
        Extract SQL query from model response
        
        Args:
            response: the response from the model
            
        Returns:
            Extracted SQL query or empty string if extraction fails
        '''
        try:
            return response.split("```sql")[1].split("```")[0].strip()
        except:
            return ''
    
    # Collect all prompts from the dataset
    prompts = []
    logger.info("Preparing prompts for batch processing")
    
    for example in dataset:
        prompts.append(example["input"])
    
    logger.info(f"Processing {len(prompts)} examples with batch size {batch_size}")
    
    # Process in batches for efficiency
    predictions = []
    
    # Create batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        batch_prompts = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        
        try:
            # Generate responses for the batch
            logger.debug(f"Generating responses for batch {i//batch_size + 1}")
            responses = llm.chat(batch_prompts, sampling_params)
            
            # Extract queries from responses
            for response in responses:
                #generated_text = response.outputs[0].text
                #query = _extract_query(generated_text)
                predictions.append(response.outputs[0].text)
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Add empty predictions for failed batch
            for _ in range(len(batch_prompts)):
                predictions.append('')
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Non-empty predictions: {sum(1 for p in predictions if p != '')}/{len(predictions)}")
    
    return predictions

def generate_predictions(model_name, dataset    ):

    '''
    model: the model to use for prediction
    dataset: the dataset to use for prediction
    '''
    # Use a pipeline as a high-level helper

    def _extract_query(response):
        '''
        response: the response from the model
        '''
        try:
            return response.split("```sql")[1].split("```")[0].strip()
        except:
            return ''

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    predictions = []

    for example in dataset:
        messages = [
            {"role": "user", "content": example["input"]}
        ]
        response = model.generate(tokenizer.encode(example["input"], return_tensors="pt"), max_length=1000)
        predictions.append(_extract_query(response[0]['generated_text'][1]['content']))

    return predictions

def evaluate_predictions(predictions, dataset):
    '''
    predictions: list of predictions
    dataset: the dataset to use for evaluation
    '''

    gold_queries = [example['query'] for example in dataset]
    pred_queries = predictions
    db_ids = [example['db_id'] for example in dataset]
    db_path = dataset.db_path
    column_order_insensitive = True
    accuracy = compute_result_accuracy(gold_queries, pred_queries, db_ids, db_path, column_order_insensitive, verbose=False)
    return accuracy



# Enhanced reward function with weighted components
def compute_reward(completions, query, db_id, **kwargs):
    """
    Compute rewards for generated queries using weighted sum of:
    1. SQL format validity (can be parsed/extracted)
    2. Execution accuracy (compute_result_accuracy)
    
    Args:
        queries (List[str]): Generated SQL queries
        examples (List[Dict]): Corresponding dataset examples
        
    Returns:
        List[float]: Reward scores
    """
    rewards = []
    format_weight = config["reward_weights"]["format"]
    execution_weight = config["reward_weights"]["execution"]
    completions = [x[0]['content'] for x in completions]
    
    for c, q, dbid in zip(completions, query, db_id):
        try:
            # Extract SQL query from response
            extracted_query = _extract_query_for_reward(c)
            
            # Component 1: SQL format validity
            format_score = _validate_sql_format(extracted_query)
            
            # Component 2: Execution accuracy
            execution_score = 0.0
            if extracted_query and format_score > 0:  # Only test execution if format is valid
                try:
                    accuracy = compute_result_accuracy(
                        [q], [extracted_query], [dbid], 
                        config["train_dataset_full"].db_path, column_order_insensitive=True
                    )
                    execution_score = float(accuracy)
                except Exception as e:
                    logger.debug(f"Execution accuracy failed: {e}")
                    execution_score = 0.0
            
            # Weighted sum
            total_reward = format_weight * format_score + execution_weight * execution_score
            rewards.append(total_reward)
            
        except Exception as e:
            logger.warning(f"Error computing reward: {e}")
            rewards.append(0.0)
    
    return rewards

def _extract_query_for_reward(response):
    """Extract SQL query from model response."""
    try:
        extracted_query = response.split("```sql")[1].split("```")[0].strip()
        return extracted_query
    except:
        try:
            extracted_query = response.split("```")[1].split("```")[0].strip()
            return extracted_query
        except:
            #print("Failed to extract query from response: ", response)
            return response

def _validate_sql_format(query):
    """
    Validate SQL format and return score between 0.0 and 1.0
    
    Args:
        query (str): SQL query to validate
        
    Returns:
        float: Format validity score
    """
    if not query or not query.strip():
        return 0.0
    
    try:
        # Basic SQL keyword check
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN', 'GROUP BY', 'ORDER BY']
        query_upper = query.upper()
        
        # Must contain at least one SQL keyword
        has_sql_keyword = any(keyword in query_upper for keyword in sql_keywords)
        if not has_sql_keyword:
            return 0.0
        
        # Try to parse with sqlparse
        try:
            parsed = sqlparse.parse(query)
            if parsed and len(parsed) > 0:
                # Successfully parsed
                return 1.0
            else:
                return 0.5  # Partial credit for containing SQL keywords
        except Exception:
            return 0.5  # Partial credit for containing SQL keywords
            
    except Exception:
        return 0.0

    # Custom evaluation function
def evaluate_model(eval_dataset, model_name, config, max_eval_samples=50):
    """Evaluate model on validation set with detailed logging."""
    logger.info("Running detailed evaluation...")
    
    # Generate predictions
    predictions = []
    examples = []
    format_scores = []
    execution_scores = []

    reduced_dataset = Dataset.from_list([x for x in eval_dataset][:max_eval_samples])
    predictions = generate_predictions_vllm(model_or_model_name=model_name, dataset=reduced_dataset, batch_size=config["batch_size"])
    examples = [x for x in reduced_dataset]

    try:
        for prediction, example in zip(predictions, examples):
            
        # Compute individual reward components for logging
            extracted_query = _extract_query_for_reward(prediction)
            format_score = _validate_sql_format(extracted_query)
            format_scores.append(format_score)
            
            execution_score = 0.0
            if extracted_query and format_score > 0:
                    try:
                        accuracy = compute_result_accuracy(
                            [example["query"]], [extracted_query], [example["db_id"]], 
                            config["train_dataset_full"].db_path, column_order_insensitive=True
                        )
                        execution_score = float(accuracy)
                    except Exception:
                        execution_score = 0.0
            execution_scores.append(execution_score)
        
    except Exception as e:
            logger.warning(f"Error in evaluation: {e}")
            predictions.append("")
            examples.append(example)
            format_scores.append(0.0)
            execution_scores.append(0.0)
    
    # Compute overall metrics
    rewards = compute_reward(predictions, [example["query"] for example in examples], [example["db_id"] for example in examples])
    accuracy = np.mean(rewards)
    
    # Log detailed metrics
    eval_metrics = {
        "eval/accuracy": accuracy,
        "eval/mean_reward": np.mean(rewards),
        "eval/std_reward": np.std(rewards),
        "eval/format_score_mean": np.mean(format_scores),
        "eval/execution_score_mean": np.mean(execution_scores),
        "eval/format_score_std": np.std(format_scores),
        "eval/execution_score_std": np.std(execution_scores),
        "eval/perfect_format_fraction": np.mean(np.array(format_scores) == 1.0),
        "eval/perfect_execution_fraction": np.mean(np.array(execution_scores) == 1.0),
    }
    
    wandb.log(eval_metrics)
    
    # Log sample predictions
    sample_logs = []
    for i, (pred, ex, reward, fmt_score, exec_score) in enumerate(zip(
        predictions[:10], examples[:10], rewards[:10], format_scores[:10], execution_scores[:10]
    )):
        sample_logs.append({
            "question": ex["question"],
            "gold_query": ex["label"],
            "predicted_query": _extract_query_for_reward(pred),
            "total_reward": reward,
            "format_score": fmt_score,
            "execution_score": exec_score,
            "full_response": pred[:200] + "..." if len(pred) > 200 else pred
        })
    
    wandb.log({
        "eval/samples": wandb.Table(
            columns=["question", "gold_query", "predicted_query", "total_reward", 
                    "format_score", "execution_score", "full_response"],
            data=[[s["question"], s["gold_query"], s["predicted_query"], s["total_reward"], 
                    s["format_score"], s["execution_score"], s["full_response"]] for s in sample_logs]
        )
    })
    
    return accuracy, eval_metrics



def tune_sft(model_name, dataset, override_config=None):
    """
    Tune the model using SFT (Supervised Fine-Tuning) for Text2SQL task.
    """
    global sft_config
    sft_config.update(override_config)
    sft_config["train_dataset_full"] = dataset




def format_chat_sample(example, tokenizer):
    messages = [
        {"role": "user", "content": example["prompt"]},
    ]
    example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example


def tune_rl(model_name, dataset, override_config=None):
    """
    Tune the model using GRPO (Group Relative Policy Optimization) for Text2SQL task.
    
    Args:
        model_name (str): Name of the model to tune
        dataset (Text2SQLDataset): Dataset for training (will be split)
        config (dict): Configuration dictionary with hyperparameters
    """

    global config
    # Merge with provided config
    if override_config is not None:
        config.update(override_config)
    config["train_dataset_full"] = dataset
    
    # Set random seed for reproducibility
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    
    # Initialize wandb
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config
    )
    
    logger.info(f"Starting GRPO training for {model_name}")
    logger.info(f"Configuration: {config}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for A6000
        device_map="cuda",
        
    )
    
    # Load training dataset if different from provided dataset
    if config["train_data_path"] != dataset.data_path:
        logger.info(f"Loading training data from: {config['train_data_path']}")
        train_dataset_full = Text2SQLDataset(
            data_path=config["train_data_path"],
            tokenizer=tokenizer,
            tables_path=dataset.tables_path,
            db_path=dataset.db_path,
            max_prompt_length=config["max_length"]
        )
    else:
        train_dataset_full = dataset
    
    # Split dataset
    logger.info("Splitting dataset into train/val")
    train_indices, val_indices = train_test_split(
        range(len(train_dataset_full)),
        train_size=config["train_split"],
        random_state=config["random_seed"],
        shuffle=True
    )
    
    train_dataset = [{"text":train_dataset_full[i]["input"], "prompt":[{"role": "user", "content":train_dataset_full[i]["input"]}], "completion":[{"role": "assistant", "content":train_dataset_full[i]["query"]}], "db_id":train_dataset_full[i]["db_id"], "question":train_dataset_full[i]["question"], "query":train_dataset_full[i]["query"], "input":train_dataset_full[i]["input"], "output":train_dataset_full[i]["output"]}  for i in train_indices]

    val_dataset = [{"text":train_dataset_full[i]["input"], "prompt":[{"role": "user", "content":train_dataset_full[i]["input"]}], "completion":[{"role": "assistant", "content":train_dataset_full[i]["query"]}], "db_id":train_dataset_full[i]["db_id"], "question":train_dataset_full[i]["question"], "query":train_dataset_full[i]["query"], "input":train_dataset_full[i]["input"], "output":train_dataset_full[i]["output"]}  for i in val_indices]

    train_dataset = Dataset.from_list(train_dataset)
    val_dataset = Dataset.from_list(val_dataset)
    
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create GRPO configuration
    grpo_config = GRPOConfig(
        learning_rate=config["learning_rate"],
        num_generations=config["num_generations"],
        per_device_train_batch_size=config["batch_size"],
        max_completion_length=config["max_completion_length"],
        max_prompt_length=config["max_prompt_length"],
        loss_type="grpo",
        per_device_eval_batch_size=config["batch_size"],
        use_vllm=True,
        vllm_mode="colocate",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        save_steps=config["checkpoint_steps"],
        eval_steps=config["eval_steps"],
        logging_steps=config["logging_steps"],
        warmup_steps=config["warmup_steps"],
        max_grad_norm=config["max_grad_norm"],
        output_dir=f"./checkpoints/{config['wandb_run_name']}",
        run_name=config["wandb_run_name"],
        report_to="wandb",
        push_to_hub=config["push_to_hub"],
        hub_model_id=config["hub_model_id"],
        dataloader_num_workers=0,  # Set to 0 for single GPU
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable for memory efficiency
        vllm_gpu_memory_utilization = 0.5,
        log_completions=True,
        num_completions_to_print=5,
        temperature=0.7,
        #top_p=0.8,
        #top_k=100,
        #repetition_penalty=1.5,
        #min_p=0.1,
    )
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=[compute_reward],
        args=grpo_config,
    )
    
    # Custom metrics logging callback
    from transformers.trainer_callback import TrainerCallback
    
    class DetailedMetricsCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.step = 0
            self.total_train_examples = 0
            self.last_gpu_memory = 0
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            self.step += 1
            if logs is None:
                logs = {}
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                gpu_memory_percent = (gpu_memory / 48) * 100  # A6000 has 48GB
                
                wandb.log({
                    "gpu_memory_allocated_gb": gpu_memory,
                    "gpu_memory_cached_gb": gpu_memory_cached,
                    "gpu_memory_percent": gpu_memory_percent,
                    "step": self.step
                })
                
                # Log memory usage changes
                if self.last_gpu_memory > 0:
                    memory_change = gpu_memory - self.last_gpu_memory
                    wandb.log({
                        "gpu_memory_delta_gb": memory_change,
                        "step": self.step
                    })
                self.last_gpu_memory = gpu_memory
            
            # Log system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            wandb.log({
                "system_cpu_percent": cpu_percent,
                "system_memory_percent": memory_percent,
                "step": self.step
            })
            
            # Log training metrics with enhanced reward breakdown
            if logs:
                # Add custom metrics if available
                enhanced_logs = logs.copy()
                if "rewards" in logs:
                    rewards = logs["rewards"]
                    if isinstance(rewards, list) and len(rewards) > 0:
                        enhanced_logs.update({
                            "reward_mean": np.mean(rewards),
                            "reward_std": np.std(rewards),
                            "reward_min": np.min(rewards),
                            "reward_max": np.max(rewards),
                            "reward_positive_fraction": np.mean(np.array(rewards) > 0),
                        })
                
                wandb.log(enhanced_logs)
    
    # Add callback to trainer
    metrics_callback = DetailedMetricsCallback()
    trainer.add_callback(metrics_callback)
    
    
    # Training loop with custom evaluation and checkpointing
    logger.info("Starting GRPO training...")
    
    try:
        # model.eval()
        # # Initial evaluation
        # with torch.no_grad():
        #     initial_accuracy, initial_metrics = evaluate_model(val_dataset, model_name=model_name, config=config)
        # logger.info(f"Initial accuracy: {initial_accuracy:.4f}")
        
        #model.train()
        # Start training
        trainer.train()
        
        # model.eval()
        # # Final evaluation
        # with torch.no_grad():
        #     final_accuracy, final_metrics = evaluate_model(val_dataset, model_name=model_name, config=config)
        # logger.info(f"Final accuracy: {final_accuracy:.4f}")
        
        # Save final model
        trainer.save_model()
        
        # Push to hub if configured
        if config["push_to_hub"]:
            logger.info("Pushing final model to Hub...")
            trainer.push_to_hub()
        
        # Log final metrics
        # wandb.log({
        #     "final/accuracy": final_accuracy,
        #     "final/improvement": final_accuracy - initial_accuracy,
        #     "final/initial_accuracy": initial_accuracy,
        # })
        
        logger.info("GRPO training completed successfully!")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    dataset_path = "spider_data/train_spider.json"
    tables_path = "spider_data/tables.json"
    # login to huggingface and read the token from .env file
    from huggingface_hub import login
    from dotenv import load_dotenv
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = Text2SQLDataset(data_path=dataset_path, tokenizer=tokenizer, tables_path=tables_path, db_path="spider_data/test_database", max_prompt_length=10000)
    #predictions = generate_predictions(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", dataset=test_dataset)
    predictions = generate_predictions_vllm(model_name=model_name, dataset=test_dataset, batch_size=50)
    print('Fraction of non empty predictions: ', sum(1 for p in predictions if p != '') / len(predictions))
    accuracy = evaluate_predictions(predictions, test_dataset)
    print('Accuracy: ', accuracy)
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Text2SQLDataset(data_path=dataset_path, tokenizer=tokenizer, tables_path=tables_path, db_path="spider_data/database", max_prompt_length=10000)
    tune_rl(model_name=model_name, dataset=train_dataset)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # test_dataset = Text2SQLDataset(data_path=dataset_path, tokenizer=tokenizer, tables_path=tables_path, db_path="spider_data/test_database", max_prompt_length=10000, max_examples=10)
    # predictions = generate_predictions_vllm(model_or_model_name=model_name, dataset=train_dataset, batch_size=50)
    # predictions = [_extract_query_for_reward(p) for p in predictions]
    # print('Fraction of non empty predictions: ', sum(1 for p in predictions if p != '') / len(predictions))
    # accuracy = evaluate_predictions(predictions, train_dataset)
    # print('Accuracy: ', accuracy)
