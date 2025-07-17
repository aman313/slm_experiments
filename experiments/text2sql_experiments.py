import sqlite3
import os
import json
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_result_accuracy(gold_queries, pred_queries, db_ids, db_path, column_order_insensitive=False):
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
            if _compare_query_results(cursor, gold_query, pred_query, column_order_insensitive):
                correct_count += 1
        
        conn.close()
        
    except Exception as e:
        print(f"Error connecting to database {db_file_path}: {e}")
        return 0.0
    
    return correct_count / total_count


def _compare_query_results(cursor, gold_query: str, pred_query: str, column_order_insensitive: bool = False) -> bool:
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
        return gold_set == pred_set
        
    except Exception as e:
        # If predicted query fails, count as incorrect
        # But we still need to check if gold query works for validation
        try:
            cursor.execute(gold_query)
            return False  # Gold query works, predicted failed -> incorrect
        except Exception:
            # Both queries failed - this shouldn't happen with gold queries
            print(f"Warning: Gold query also failed: {e}")
            return False


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
                 max_examples: int = None, db_path: str = "spider_data/database", max_prompt_length: int = 4096):
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
        
        if self.max_examples:
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
    
    def _get_schema_from_file(self, db_id: str) -> str:
        """Get schema from schema.sql file if available."""
        schema_path = os.path.join(self.db_path, db_id, "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                return f.read()
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
    
    def _format_example(self, example: Dict) -> Dict:
        """Format single example for instruction fine-tuning."""
        db_id = example['db_id']
        question = example['question']
        query = example['query']
        
        # Get schema
        schema = self._get_schema_from_file(db_id)
        if not schema and db_id in self.tables:
            schema = self._create_schema_from_tables_json(self.tables[db_id])
        
        if not schema:
            schema = f"-- Schema for {db_id} not available"
        
        # Format input prompt
        input_text = f"""Given this schema:
            {schema}

            Write an SQL query for the below requirement.
            The query should be in the following format:
            ```sql
            <query>
            ```
            {question}"""
        tokenized_input = self.tokenizer.encode(input_text, return_tensors="pt")
        if tokenized_input.shape[1] > self.max_prompt_length:
            logger.warning(f"Prompt length exceeded {self.max_prompt_length} tokens. Skipping example.")
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
    
def generate_predictions_vllm(model_name, dataset):
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
        
        # Initialize vLLM model with macOS-friendly configuration
        # macOS doesn't have NVIDIA GPUs, so this will run on CPU
        llm = LLM(
            model=model_name,
            max_model_len=8192,          # Keep model's full capacity
            max_num_batched_tokens=8192, # Increase to match model capacity
            tensor_parallel_size=1,      # Single process for macOS
        )
        
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
    
    logger.info(f"Processing {len(prompts)} examples with batch size 2")
    
    # Process in batches for efficiency
    predictions = []
    batch_size = 2
    
    # Create batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        try:
            # Generate responses for the batch
            logger.debug(f"Generating responses for batch {i//batch_size + 1}")
            responses = llm.generate(batch_prompts, sampling_params)
            
            # Extract queries from responses
            for response in responses:
                generated_text = response.outputs[0].text
                query = _extract_query(generated_text)
                predictions.append(query)
                
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
    accuracy = compute_result_accuracy(gold_queries, pred_queries, db_ids, db_path, column_order_insensitive)
    return accuracy

if __name__ == "__main__":
    dataset_path = "spider_data/test.json"
    tables_path = "spider_data/test_tables.json"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    test_dataset = Text2SQLDataset(data_path=dataset_path, tokenizer=tokenizer, tables_path=tables_path, max_examples=10, db_path="spider_data/test_database")
    #predictions = generate_predictions(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", dataset=test_dataset)
    predictions = generate_predictions_vllm(model_name="HuggingFaceTB/SmolLM2-135M-Instruct", dataset=test_dataset)
    print('Fraction of non empty predictions: ', sum(1 for p in predictions if p != '') / len(predictions))
    accuracy = evaluate_predictions(predictions, test_dataset)
    print('Accuracy: ', accuracy)