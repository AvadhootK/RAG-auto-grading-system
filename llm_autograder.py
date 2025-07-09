import json
import nbformat
from typing import Dict, List, Any, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
from pathlib import Path
import re
import ast
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class NotebookGradingSystem:
    """
    Automated grading system for Python notebooks using RAG + Chain of Thought
    
    This system processes Jupyter notebooks, retrieves relevant rubric criteria,
    and uses LLM with structured reasoning to provide detailed grades and feedback.
    """
    
    def __init__(self, hf_model_name: str = "microsoft/DialoGPT-medium", model_name: str = "gpt-4o-mini"):
        """
        Initialize the grading system with Hugging Face model / OpenAI model and embedding model.
        
        Args:
            hf_model_name: OpenAI model name / Hugging Face model name (default: microsoft/DialoGPT-medium)
                          Alternative small models: "distilgpt2", "gpt2", "microsoft/DialoGPT-small"
        """

        # ---------- Huggingface model initialization ------------- #
        # self.hf_model_name = hf_model_name
        
        # # Initialize Hugging Face pipeline for text generation
        # print(f"Loading Hugging Face model: {hf_model_name}")
        # self.llm_pipeline = pipeline(
        #     "text-generation",
        #     model=hf_model_name,
        #     tokenizer=hf_model_name,
        #     max_length=2048,
        #     do_sample=True,
        #     temperature=0.3,
        #     pad_token_id=50256, 
        #     device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        # )
        
        # ---------- OpenAI model initialization --------------- #
        self.model_name = model_name
    
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI client with model: {model_name}")

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index (will be populated when rubric is loaded)
        self.faiss_index = None
        self.rubric_chunks = []
        
    def load_rubric_database(self, rubric_file_path: str):
        """
        STEP 1: Load and process rubric criteria for RAG retrieval
        
        This function:
        1. Loads rubric criteria from a JSON file
        2. Creates text chunks from each rubric criterion
        3. Generates embeddings for each chunk
        4. Builds FAISS index for fast similarity search
        
        Args:
            rubric_file_path: Path to JSON file containing rubric criteria
            
        Expected rubric format:
        {
            "question_1": {
                "criteria": [
                    {"aspect": "correctness", "description": "Code solves problem correctly", "weight": 60},
                    {"aspect": "style", "description": "Code follows PEP 8 standards", "weight": 20},
                    {"aspect": "explanation", "description": "Markdown explains approach clearly", "weight": 20}
                ]
            }
        }
        """
        print("Loading rubric database...")
        
        # Load rubric from JSON file
        with open(rubric_file_path, 'r') as f:
            rubric_data = json.load(f)
        
        # Create text chunks from rubric criteria
        self.rubric_chunks = []
        self.question_to_chunk_indices = {}  # Maps question_id to list of chunk indices

        chunk_index = 0
        for question_id, question_rubric in rubric_data.items():
            self.question_to_chunk_indices[question_id] = []

            for criterion in question_rubric['criteria']:
                chunk_text = f"Question: {question_id}\n"
                chunk_text += f"Aspect: {criterion['aspect']}\n"
                chunk_text += f"Description: {criterion['description']}\n"
                chunk_text += f"Weight: {criterion['weight']}%"
                
                self.rubric_chunks.append({
                    'text': chunk_text,
                    'question_id': question_id,
                    'aspect': criterion['aspect'],
                    'description': criterion['description'],
                    'weight': criterion['weight']
                })

                # Map question to chunk index
                self.question_to_chunk_indices[question_id].append(chunk_index)
                chunk_index += 1
        
        # Generate embeddings for all rubric chunks
        chunk_texts = [chunk['text'] for chunk in self.rubric_chunks]

        # print(json.dumps(self.rubric_chunks, indent=2))
        # self.rubric_chunks is structured like this:
        """
              {
    "text": "Question: question_3\nAspect: analysis\nDescription: Written analysis interprets the visualization correctly\nWeight: 30%",
    "question_id": "question_3",
    "aspect": "analysis",
    "description": "Written analysis interprets the visualization correctly",
    "weight": 30
  }
        """

        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Build FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"Loaded {len(self.rubric_chunks)} rubric criteria chunks")
        print(f"Question mapping: {dict([(q, len(indices)) for q, indices in self.question_to_chunk_indices.items()])}")
    
    def extract_notebook_content(self, notebook_path: str) -> List[Dict[str, Any]]:
        """
        STEP 2: Extract and structure content from Jupyter notebook
        
        This function:
        1. Reads the .ipynb file using nbformat
        2. Extracts each cell's content (markdown, code, output)
        3. Groups cells into task-level segments
        4. Performs basic static analysis on code cells
        
        Args:
            notebook_path: Path to the Jupyter notebook file
            
        Returns:
            List of dictionaries containing structured cell content
        """
        print(f"Extracting content from {notebook_path}...")
        
        # Read notebook file
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        extracted_content = []
        current_question = None
        
        for i, cell in enumerate(notebook.cells):
            cell_data = {
                'cell_index': i,
                'cell_type': cell.cell_type,
                'source': cell.source,
                'question_id': current_question
            }
            
            if cell.cell_type == 'markdown':
                # Look for question markers (e.g., "## Question 1", "# Problem 2")
                question_match = re.search(r'(?:question|problem|task)\s*(\d+)', 
                                         cell.source.lower())
                if question_match:
                    current_question = f"question_{question_match.group(1)}"
                    cell_data['question_id'] = current_question
                
                cell_data['content_type'] = 'explanation'
                
            elif cell.cell_type == 'code':
                # Perform static analysis on code
                cell_data['content_type'] = 'code'
                cell_data['static_analysis'] = self._analyze_code_static(cell.source)
                
                # Extract outputs if they exist
                outputs = []
                if hasattr(cell, 'outputs') and cell.outputs:
                    for output in cell.outputs:
                        if output.output_type == 'execute_result':
                            outputs.append(output.data.get('text/plain', ''))
                        elif output.output_type == 'stream':
                            outputs.append(output.text)
                
                cell_data['outputs'] = outputs
            
            extracted_content.append(cell_data)
        
        print(f"Extracted {len(extracted_content)} cells")
        return extracted_content
    
    def _analyze_code_static(self, code: str) -> Dict[str, Any]:
        """
        Perform static analysis on code cell content
        
        This function:
        1. Checks if code is syntactically valid
        2. Extracts imports, functions, and variables
        3. Identifies potential style issues
        4. Counts lines of code and comments
        
        Args:
            code: Python code string
            
        Returns:
            Dictionary containing static analysis results
        """
        analysis = {
            'is_valid_syntax': True,
            'syntax_error': None,
            'imports': [],
            'functions': [],
            'variables': [],
            'line_count': len(code.splitlines()),
            'comment_count': 0,
            'style_issues': []
        }
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Extract imports, functions, and variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    analysis['imports'].extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(f"from {node.module}")
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append(target.id)
            
            # Count comments
            analysis['comment_count'] = len([line for line in code.splitlines() 
                                           if line.strip().startswith('#')])
            
            # Basic style checks
            if not code.strip():
                analysis['style_issues'].append("Empty code cell")
            if analysis['comment_count'] == 0 and analysis['line_count'] > 5:
                analysis['style_issues'].append("No comments in complex code")
            
        except SyntaxError as e:
            analysis['is_valid_syntax'] = False
            analysis['syntax_error'] = str(e)
        
        return analysis
    
    def retrieve_rubric_criteria(self, query_text: str, question_id: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        STEP 3: Retrieve relevant rubric criteria using RAG with question-specific filtering
        
        This function:
        1. Encodes the query text (question + code + markdown)
        2. Performs similarity search against rubric database
        3. Filters results to only include criteria for the specified question
        4. Returns top-k most relevant rubric criteria
        5. Includes similarity scores for relevance filtering
        
        Args:
            query_text: Combined text from notebook cell(s)
            question_id: Specific question to retrieve criteria for (if None, searches all)
            k: Number of top results to return
            
        Returns:
            List of relevant rubric criteria with similarity scores
        """
        if self.faiss_index is None:
            raise ValueError("Rubric database not loaded. Call load_rubric_database() first.")
        
        # Encode query text
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        if question_id:
            if question_id not in self.question_to_chunk_indices:
                print(f"Warning: No rubric criteria found for question_id: {question_id}")
                return []

            # Search MORE results than needed to account for filtering
            search_k = min(len(self.rubric_chunks), k * 3)  # Search 3x more to account for filtering
            similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)

            # Get indices for this question
            question_chunk_indices = self.question_to_chunk_indices[question_id]

            # Filter results to only include the specified question's criteria
            relevant_criteria = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx in question_chunk_indices:  # Only include if it belongs to our question
                    if similarity > 0.0:  # Filter out very low similarity matches (default: 0.1)
                        criterion = self.rubric_chunks[idx].copy()
                        criterion['similarity'] = float(similarity)
                        relevant_criteria.append(criterion)
                    
                        # Stop when we have enough results
                        if len(relevant_criteria) >= k:
                            break
            
            # print(json.dumps(relevant_criteria, indent=2))

        else:

            # Search for top-k similar rubric criteria
            similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), k)
        
            # Retrieve matching rubric criteria
            relevant_criteria = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity > 0.1:  # Filter out very low similarity matches
                    criterion = self.rubric_chunks[idx].copy()
                    criterion['similarity'] = float(similarity)
                    relevant_criteria.append(criterion)
        
        return relevant_criteria

    def generate_grade_with_cot(self, question_content: Dict[str, str], 
                               rubric_criteria: List[Dict[str, Any]],
                               question_id: str) -> Dict[str, Any]:
        """
        STEP 4: Generate grade using Chain of Thought reasoning
        
        This function:
        1. Constructs a structured prompt with retrieved rubric criteria
        2. Implements Chain of Thought reasoning with 6 specific phases
        3. Calls OpenAI API with the structured prompt
        4. Parses the response to extract grades, reasoning, and confidence
        
        Args:
            question_content: Consolidated content from notebook question (from consolidate_question_content)
            rubric_criteria: Retrieved rubric criteria from RAG
            question_id: The question identifier
            
        Returns:
            Dictionary containing detailed grading results
        """
        # Prepare rubric criteria text
        rubric_text = "\n".join([
            f"- {criterion['aspect'].title()}: {criterion['description']} (Weight: {criterion['weight']}%)"
            for criterion in rubric_criteria
        ])
        
        # Extract consolidated content
        markdown_content = question_content.get('markdown_content', '')
        code_content = question_content.get('code_content', '')
        outputs = question_content.get('outputs', '')
        static_analysis_summary = question_content.get('static_analysis_summary', {})

        # Construct Chain of Thought prompt with consistent scoring
        prompt = f"""You are an auto-grader for a data science course. Below is a student's complete answer for {question_id} and the rubric criteria. Grade this student's answer:

        Rubric Criteria: 
        {rubric_text}

        Student's Complete Answer:
        Markdown Explanation: 
        {markdown_content}

        Code Implementation:
        {code_content}

        Code Output:
        {outputs}

        Static Analysis Summary:
        - Functions defined: {static_analysis_summary.get('functions', [])}
        - Imports used: {static_analysis_summary.get('imports', [])}
        - Total lines of code: {static_analysis_summary.get('total_lines', 0)}
        - Syntax errors: {static_analysis_summary.get('syntax_errors', [])}
        - Has syntax errors: {static_analysis_summary.get('has_syntax_errors', False)}

        Please follow these reasoning steps:

        1. **Understanding Phase**: What is this question asking for? What's the expected approach?
        2. **Correctness Phase**: Does the code solve the problem correctly? Are there logical errors?
        3. **Quality Phase**: Is the code well-written? Are there style issues?
        4. **Explanation Phase**: Does the markdown explanation demonstrate understanding?
        5. **Output Phase**: Are the results correct and properly interpreted?
        6. **Scoring Phase**: Score each rubric aspect (0-10), then calculate weighted final score

        IMPORTANT SCORING RULES:
        - All individual scores should be on a 0-10 scale
        - Final score MUST be calculated as weighted average using rubric weights
        - Example: If aspect1=6/10 (60% weight), aspect2=8/10 (40% weight), then Final=(6*0.6 + 8*0.4)=6.8/10

        Respond in this format:
        - Understanding Check: ...
        - Correctness Check: ...
        - Quality Check: ...
        - Explanation Check: ...
        - Output Check: ...
        - Rubric Breakdown: [aspect1: X/10 points, aspect2: Y/10 points, etc.]
        - Weighted Calculation: Show step-by-step: (score1*weight1 + score2*weight2 + ...) / total_weight
        - Confidence Level: X% (how confident are you in this grade?)
        - Final Score: X/10 (must match your Weighted Calculation)
        - Feedback: ...
        """
        
        try:
            # Call Hugging Face model
            # response = self.llm_pipeline(
            #     prompt,
            #     max_length=len(prompt.split()) + 300,  # Add space for response
            #     max_new_tokens=512, 
            #     num_return_sequences=1,
            #     truncation=True
            # )
            
            # # Extract generated text (remove the input prompt)
            # generated_text = response[0]['generated_text']
            # response_text = generated_text[len(prompt):].strip()
            
            # # Parse response
            # parsed_result = self._parse_grading_response(response_text)
            
            # # Add metadata
            # parsed_result['raw_response'] = response_text
            # parsed_result['rubric_criteria_used'] = rubric_criteria
            
            # return parsed_result

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert auto-grader for data science courses. Follow the specified reasoning format exactly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
        
            response_text = response.choices[0].message.content
        
            # Parse response
            parsed_result = self._parse_grading_response(response_text, rubric_criteria)
        
            # Add metadata
            parsed_result['question_id'] = question_id
            parsed_result['raw_response'] = response_text
            parsed_result['rubric_criteria_used'] = rubric_criteria
        
            return parsed_result
            
        except Exception as e:
            print(f"Error grading {question_id}: {str(e)}")
            return {
                'error': str(e),
                'question_id': question_id,
                'final_score': 0,
                'confidence': 0,
                'feedback': f"Error occurred during grading: {str(e)}",
                'reasoning_steps': {},
                'rubric_breakdown': {}
            }
    
    # def _parse_grading_response(self, response_text: str, rubric_criteria: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """
    #     Parse the structured response from the LLM with consistent 0-10 scoring
    #     """
    #     result = {
    #         'reasoning_steps': {},
    #         'rubric_breakdown': {},
    #         'confidence': 0,
    #         'final_score': 0,
    #         'feedback': ''
    #     }
        
    #     # Extract reasoning steps
    #     steps = ['Understanding Check', 'Correctness Check', 'Quality Check', 
    #             'Explanation Check', 'Output Check']
        
    #     for step in steps:
    #         pattern = rf'{step}:\s*(.+?)(?=\n-|\n\n|$)'
    #         match = re.search(pattern, response_text, re.DOTALL)
    #         if match:
    #             result['reasoning_steps'][step.lower().replace(' check', '')] = match.group(1).strip()
        
    #     # Extract rubric breakdown - FIXED TO HANDLE BOTH FORMATS
    #     breakdown_match = re.search(r'Rubric Breakdown:\s*\[(.*?)\]', response_text, re.DOTALL)
    #     if breakdown_match:
    #         breakdown_text = breakdown_match.group(1)
    #         # Parse scores in format "aspect: X/Y points" or "aspect: X/Y"
    #         score_matches = re.findall(r'(\w+):\s*(\d+)/(\d+)', breakdown_text)
    #         for aspect, score, max_score in score_matches:
    #             # Convert all scores to 0-10 scale
    #             normalized_score = (int(score) / int(max_score)) * 10
    #             result['rubric_breakdown'][aspect.lower()] = {
    #                 'score': min(10, max(0, normalized_score)),  # Clamp between 0-10
    #                 'max': 10
    #             }
        
    #     # Extract confidence level
    #     confidence_match = re.search(r'Confidence Level:\s*(\d+)%', response_text)
    #     if confidence_match:
    #         result['confidence'] = int(confidence_match.group(1))
        
    #     # Extract final score and normalize to 0-10
    #     score_match = re.search(r'Final Score:\s*(\d+(?:\.\d+)?)/(\d+)', response_text)
    #     if score_match:
    #         score = float(score_match.group(1))
    #         max_score = float(score_match.group(2))
    #         # Normalize to 0-10 scale
    #         result['final_score'] = min(10, max(0, (score / max_score) * 10))
    #     # elif result['rubric_breakdown']:
    #     #     # Calculate weighted average from rubric breakdown if final score not found
    #     #     total_score = sum(item['score'] for item in result['rubric_breakdown'].values())
    #     #     num_aspects = len(result['rubric_breakdown'])
    #     #     result['final_score'] = total_score / num_aspects if num_aspects > 0 else 0
    #     elif result['rubric_breakdown']:
    #         # Calculate WEIGHTED average from rubric breakdown if final score not found
    #         total_weighted_score = 0
    #         total_weight = 0
            
    #         # Get weights from rubric_criteria (passed as parameter)
    #         aspect_weights = {}
    #         for criterion in rubric_criteria:
    #             aspect_weights[criterion['aspect'].lower()] = criterion['weight']
            
    #         for aspect, score_info in result['rubric_breakdown'].items():
    #             weight = aspect_weights.get(aspect, 100 / len(result['rubric_breakdown']))  # Default equal weight
    #             total_weighted_score += score_info['score'] * weight
    #             total_weight += weight
            
    #         result['final_score'] = (total_weighted_score / total_weight) if total_weight > 0 else 0

        
    #     # Extract feedback
    #     feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n-|\n\n|$)', response_text, re.DOTALL)
    #     if feedback_match:
    #         result['feedback'] = feedback_match.group(1).strip()
        
    #     return result

    def _parse_grading_response(self, response_text: str, rubric_criteria: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse the structured response from the LLM with consistent 0-10 scoring
        """
        result = {
            'reasoning_steps': {},
            'rubric_breakdown': {},
            'confidence': 0,
            'final_score': 0,
            'feedback': ''
        }
        
        # print(f"DEBUG: Raw response text:\n{response_text}")
        # print(f"DEBUG: Rubric criteria aspects: {[c['aspect'] for c in rubric_criteria] if rubric_criteria else 'None'}")
        
        # Extract reasoning steps
        steps = ['Understanding Check', 'Correctness Check', 'Quality Check', 
                'Explanation Check', 'Output Check']
        
        for step in steps:
            pattern = rf'{step}:\s*(.+?)(?=\n-|\n\n|$)'
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                result['reasoning_steps'][step.lower().replace(' check', '')] = match.group(1).strip()
        
        # Extract rubric breakdown with multiple fallback patterns
        rubric_breakdown_patterns = [
            # Pattern 1: Original format with brackets
            r'Rubric Breakdown:\s*\[(.*?)\]',
            # Pattern 2: Without brackets, until next section
            r'Rubric Breakdown:\s*(.*?)(?=\n(?:Weighted Calculation|Confidence Level|Final Score|Feedback):|$)',
            # Pattern 3: Look for any line with "breakdown" (case insensitive)
            r'(?i)breakdown:\s*(.*?)(?=\n(?:Weighted|Confidence|Final|Feedback):|$)',
            # Pattern 4: Look for scoring pattern anywhere in text
            r'(?i)(?:scores?|scoring|breakdown):\s*(.*?)(?=\n(?:Weighted|Confidence|Final|Feedback):|$)'
        ]
        
        breakdown_text = None
        for pattern in rubric_breakdown_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                breakdown_text = match.group(1).strip()
                # print(f"DEBUG: Found breakdown with pattern: {pattern}")
                # print(f"DEBUG: Breakdown text: {breakdown_text}")
                break
        
        if breakdown_text:
            # Multiple score parsing patterns
            score_patterns = [
                # Pattern 1: "aspect: X/Y points"
                r'(\w+):\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*points?',
                # Pattern 2: "aspect: X/Y"
                r'(\w+):\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
                # Pattern 3: "aspect - X/Y"
                r'(\w+)\s*-\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
                # Pattern 4: Handle multi-word aspects (e.g., "code quality: 8/10")
                r'([a-zA-Z][a-zA-Z\s]+?):\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
                # Pattern 5: Just scores without labels, use rubric order
                r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)'
            ]
            
            for pattern in score_patterns:
                score_matches = re.findall(pattern, breakdown_text)
                if score_matches:
                    # print(f"DEBUG: Found scores with pattern: {pattern}")
                    # print(f"DEBUG: Score matches: {score_matches}")
                    
                    # Handle patterns with aspect names
                    if len(score_matches[0]) == 3:  # (aspect, score, max_score)
                        for aspect, score, max_score in score_matches:
                            # Clean aspect name
                            aspect_clean = aspect.lower().strip()
                            # Convert all scores to 0-10 scale
                            normalized_score = (float(score) / float(max_score)) * 10
                            result['rubric_breakdown'][aspect_clean] = {
                                'score': min(10, max(0, normalized_score)),
                                'max': 10
                            }
                    # Handle patterns with just scores (use rubric order)
                    elif len(score_matches[0]) == 2 and rubric_criteria:  # (score, max_score)
                        for i, (score, max_score) in enumerate(score_matches):
                            if i < len(rubric_criteria):
                                aspect = rubric_criteria[i]['aspect'].lower()
                                normalized_score = (float(score) / float(max_score)) * 10
                                result['rubric_breakdown'][aspect] = {
                                    'score': min(10, max(0, normalized_score)),
                                    'max': 10
                                }
                    break
        
        # FALLBACK: If no breakdown found, try to extract from anywhere in response
        if not result['rubric_breakdown'] and rubric_criteria:
            print("DEBUG: No rubric breakdown found, trying fallback extraction...")
            
            # Try to find scores for each rubric aspect individually
            for criterion in rubric_criteria:
                aspect = criterion['aspect'].lower()
                # Look for aspect name followed by score anywhere in text
                aspect_patterns = [
                    rf'(?i){aspect}[:\s-]*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
                    rf'(?i){aspect}[:\s-]*(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(\d+(?:\.\d+)?)',
                    rf'(?i){aspect}[:\s-]*(\d+(?:\.\d+)?)\s*points?'
                ]
                
                for pattern in aspect_patterns:
                    match = re.search(pattern, response_text)
                    if match:
                        if len(match.groups()) == 2:
                            score, max_score = match.groups()
                            normalized_score = (float(score) / float(max_score)) * 10
                        else:
                            score = match.group(1)
                            normalized_score = float(score)  # Assume already on 0-10 scale
                        
                        result['rubric_breakdown'][aspect] = {
                            'score': min(10, max(0, normalized_score)),
                            'max': 10
                        }
                        # print(f"DEBUG: Found fallback score for {aspect}: {normalized_score}")
                        break
        
        # Extract confidence level
        confidence_patterns = [
            r'Confidence Level:\s*(\d+)%',
            r'Confidence:\s*(\d+)%',
            r'(?i)confidence[:\s]*(\d+)%'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text)
            if match:
                result['confidence'] = int(match.group(1))
                break
        
        # Extract final score with multiple patterns
        score_patterns = [
            r'Final Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Overall Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)',
            r'Final:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response_text)
            if match:
                score = float(match.group(1))
                max_score = float(match.group(2))
                result['final_score'] = min(10, max(0, (score / max_score) * 10))
                break
        
        # Calculate weighted average if final score not found but breakdown exists
        if result['final_score'] == 0 and result['rubric_breakdown'] and rubric_criteria:
            total_weighted_score = 0
            total_weight = 0
            
            # Get weights from rubric_criteria
            aspect_weights = {}
            for criterion in rubric_criteria:
                aspect_weights[criterion['aspect'].lower()] = criterion['weight']
            
            for aspect, score_info in result['rubric_breakdown'].items():
                weight = aspect_weights.get(aspect, 100 / len(result['rubric_breakdown']))
                total_weighted_score += score_info['score'] * weight
                total_weight += weight
            
            if total_weight > 0:
                result['final_score'] = total_weighted_score / total_weight
                # print(f"DEBUG: Calculated weighted final score: {result['final_score']}")
        
        # Extract feedback
        feedback_patterns = [
            r'Feedback:\s*(.+?)(?=\n-|\n\n|$)',
            r'Comments:\s*(.+?)(?=\n-|\n\n|$)',
            r'Suggestions:\s*(.+?)(?=\n-|\n\n|$)'
        ]
        
        for pattern in feedback_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                result['feedback'] = match.group(1).strip()
                break
        
        # print(f"DEBUG: Final parsed result: {result}")
        return result
    
    def grade_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """
        STEP 5: Main grading pipeline - orchestrates all steps
        
        This function:
        1. Extracts content from the notebook
        2. Groups cells by question/task
        3. For each question, retrieves relevant rubric criteria
        4. Generates grades using Chain of Thought reasoning
        5. Compiles overall results and statistics
        
        Args:
            notebook_path: Path to the notebook file to grade
            
        Returns:
            Complete grading results with scores, feedback, and metadata
        """
        print(f"Starting grading process for {notebook_path}...")
        
        # Extract notebook content
        extracted_content = self.extract_notebook_content(notebook_path)

        # print(json.dumps(extracted_content, indent=2))
        
        # Group cells by question
        questions = {}
        for cell in extracted_content:
            question_id = cell.get('question_id', 'general')
            # check for valid question_id
            if not question_id:
                continue
            if question_id not in questions:
                questions[question_id] = []
            questions[question_id].append(cell)
        
        # print(json.dumps(questions, indent=2))

        # Grade each question
        # grading_results = {}
        # total_score = 0
        # total_confidence = 0
        
        # for question_id, cells in questions.items():
        #     print(f"Grading {question_id}...")
            
        #     # Combine cell content for RAG query
        #     query_parts = []
        #     for cell in cells:
        #         query_parts.append(f"Question: {question_id}")
        #         query_parts.append(f"Content: {cell.get('source', '')}")
        #         if cell.get('static_analysis'):
        #             query_parts.append(f"Code Analysis: {cell['static_analysis']}")
            
        #     # print(json.dumps(query_parts, indent=2))

        #     query_text = '\n'.join(query_parts)
            
        #     # Retrieve relevant rubric criteria
        #     relevant_criteria = self.retrieve_rubric_criteria(query_text, question_id=question_id, k=5)
        #     # print(json.dumps(relevant_criteria, indent=2))
            
        #     # Grade each cell in the question
        #     question_results = []
        #     for cell in cells:
        #         if cell.get('source', '').strip():  # Skip empty cells
        #             # print(json.dumps(cell, indent=2))
        #             grade_result = self.generate_grade_with_cot(cell, relevant_criteria)
        #             question_results.append(grade_result)
        #             # print(json.dumps(question_results, indent=2))

            
        #     # Aggregate question results
        #     if question_results:
        #         avg_score = sum(r.get('final_score', 0) for r in question_results) / len(question_results)
        #         avg_confidence = sum(r.get('confidence', 0) for r in question_results) / len(question_results)
                
        #         grading_results[question_id] = {
        #             'average_score': avg_score,
        #             'average_confidence': avg_confidence,
        #             'cell_results': question_results,
        #             'relevant_criteria': relevant_criteria
        #         }
                
        #         total_score += avg_score
        #         total_confidence += avg_confidence
        
        # Grade each question holistically
        grading_results = {}
        total_score = 0
        total_confidence = 0
        
        for question_id, cells in questions.items():
            print(f"Grading {question_id}...")
            
            # Consolidate all cells for this question
            question_content = self.consolidate_question_content(cells)
            
            # print(json.dumps(question_content, indent=2))
            # print(y)

            # Retrieve relevant rubric criteria for this question
            relevant_criteria = self.retrieve_rubric_criteria(
                question_content['all_content'], 
                question_id=question_id, 
                k=10
            )
            
            # Grade the entire question at once
            grade_result = self.generate_grade_with_cot(
                question_content, 
                relevant_criteria, 
                question_id
            )
            
            grading_results[question_id] = grade_result
            
            total_score += grade_result.get('final_score', 0)
            total_confidence += grade_result.get('confidence', 0)

        # Calculate overall statistics
        num_questions = len(grading_results)
        overall_results = {
            'notebook_path': notebook_path,
            'overall_score': total_score / num_questions if num_questions > 0 else 0,
            'overall_confidence': total_confidence / num_questions if num_questions > 0 else 0,
            'questions_graded': num_questions,
            'question_results': grading_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"Grading complete. Overall score: {overall_results['overall_score']:.2f}/10")
        return overall_results

    # ----------------------- Start of Fixing Question Level grading Logic ----------------- # 

    def generate_question_grade_with_simplified_cot(self, question_content: Dict[str, str], 
                                                   rubric_criteria: List[Dict[str, Any]],
                                                   question_id: str) -> Dict[str, Any]:
        """
        NEW: Grade entire question with simplified reasoning suitable for smaller models
        """
        # Prepare rubric criteria text
        rubric_text = "\n".join([
            f"- {criterion['aspect'].title()}: {criterion['description']} (Weight: {criterion['weight']}%)"
            for criterion in rubric_criteria
        ])
        
        # Create a simplified prompt that smaller models can handle
        prompt = f"""Grade this student's answer for {question_id}:

RUBRIC:
{rubric_text}

STUDENT ANSWER:
{question_content['all_content'][:1000]}  # Truncate to prevent context overflow

ANALYSIS:
Code functions: {question_content['static_analysis_summary']['functions']}
Syntax errors: {question_content['static_analysis_summary']['syntax_errors']}

Please provide:
1. Score for each rubric aspect (0-10)
2. Brief feedback
3. Overall score (0-10)

Format your response as:
SCORES: [aspect1: X/10, aspect2: Y/10, ...]
FEEDBACK: [your feedback here]
OVERALL: Z/10
"""
        
        try:
            # response = self.llm_pipeline(
            #     prompt,
            #     max_new_tokens=200,
            #     num_return_sequences=1,
            #     truncation=True,
            #     do_sample=True,
            #     temperature=0.1  # Lower temperature for more consistent grading
            # )
            
            # generated_text = response[0]['generated_text']
            # response_text = generated_text[len(prompt):].strip()

            # # print(response_text)
            
            # # Parse the simplified response
            # parsed_result = self._parse_simplified_response(response_text, rubric_criteria)
            
            # # Add metadata
            # parsed_result['question_id'] = question_id
            # parsed_result['raw_response'] = response_text
            # parsed_result['rubric_criteria_used'] = rubric_criteria
            
            # return parsed_result

            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert auto-grader. Provide concise, structured feedback in the specified format."},
                {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
        
            response_text = response.choices[0].message.content
        
            # Parse the simplified response
            parsed_result = self._parse_simplified_response(response_text, rubric_criteria)
        
            # Add metadata
            parsed_result['question_id'] = question_id
            parsed_result['raw_response'] = response_text
            parsed_result['rubric_criteria_used'] = rubric_criteria
        
            return parsed_result
            
        except Exception as e:
            print(f"Error grading {question_id}: {str(e)}")
            return {
                'error': str(e),
                'question_id': question_id,
                'final_score': 0,
                'confidence': 0,
                'feedback': f"Error occurred during grading: {str(e)}",
                'aspect_scores': {}
            }
    
    def _parse_simplified_response(self, response_text: str, rubric_criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        NEW: Parse simplified response format
        """
        result = {
            'aspect_scores': {},
            'final_score': 0,
            'confidence': 70,  # Default confidence for simplified approach
            'feedback': '',
            'reasoning_summary': response_text
        }
        
        # Extract aspect scores
        scores_match = re.search(r'SCORES:\s*\[(.*?)\]', response_text, re.DOTALL)
        if scores_match:
            scores_text = scores_match.group(1)
            # Parse individual scores
            for criterion in rubric_criteria:
                aspect = criterion['aspect']
                pattern = rf'{aspect}:\s*(\d+(?:\.\d+)?)/10'
                score_match = re.search(pattern, scores_text, re.IGNORECASE)
                if score_match:
                    result['aspect_scores'][aspect] = {
                        'score': float(score_match.group(1)),
                        'max_score': 10,
                        'weight': criterion['weight']
                    }
        
        # Extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?=OVERALL:|$)', response_text, re.DOTALL)
        if feedback_match:
            result['feedback'] = feedback_match.group(1).strip()
        
        # Extract overall score
        overall_match = re.search(r'OVERALL:\s*(\d+(?:\.\d+)?)/10', response_text)
        if overall_match:
            result['final_score'] = float(overall_match.group(1))
        elif result['aspect_scores']:
            # Calculate weighted average if overall score not found
            total_weighted_score = 0
            total_weight = 0
            for aspect, score_info in result['aspect_scores'].items():
                total_weighted_score += score_info['score'] * score_info['weight']
                total_weight += score_info['weight']
            
            if total_weight > 0:
                result['final_score'] = total_weighted_score / total_weight
        
        return result

    def consolidate_question_content(self, cells: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        NEW: Consolidate all cells for a question into structured content
        """
        consolidated = {
            'markdown_content': '',
            'code_content': '',
            'outputs': '',
            'static_analysis_summary': {},
            'all_content': ''
        }
        
        markdown_parts = []
        code_parts = []
        output_parts = []
        all_functions = []
        all_imports = []
        total_lines = 0
        syntax_errors = []
        
        for cell in cells:
            if cell.get('cell_type') == 'markdown' and cell.get('source', '').strip():
                markdown_parts.append(cell['source'])
            elif cell.get('cell_type') == 'code' and cell.get('source', '').strip():
                code_parts.append(cell['source'])
                
                # Aggregate outputs
                if cell.get('outputs'):
                    output_parts.extend(cell['outputs'])
                
                # Aggregate static analysis
                if cell.get('static_analysis'):
                    analysis = cell['static_analysis']
                    all_functions.extend(analysis.get('functions', []))
                    all_imports.extend(analysis.get('imports', []))
                    total_lines += analysis.get('line_count', 0)
                    
                    if not analysis.get('is_valid_syntax', True):
                        syntax_errors.append(analysis.get('syntax_error', 'Unknown syntax error'))
        
        # Combine content
        consolidated['markdown_content'] = '\n\n'.join(markdown_parts)
        consolidated['code_content'] = '\n\n'.join(code_parts)
        consolidated['outputs'] = '\n'.join(output_parts)
        consolidated['all_content'] = f"Explanation: {consolidated['markdown_content']}\n\nCode: {consolidated['code_content']}\n\nOutputs: {consolidated['outputs']}"
        
        # Summarize static analysis
        consolidated['static_analysis_summary'] = {
            'functions': list(set(all_functions)),
            'imports': list(set(all_imports)),
            'total_lines': total_lines,
            'syntax_errors': syntax_errors,
            'has_syntax_errors': len(syntax_errors) > 0
        }
        
        return consolidated

    def grade_notebook_simplified(self, notebook_path: str) -> Dict[str, Any]:
        """
        FIXED: Grade questions holistically instead of cell-by-cell
        """
        print(f"Starting grading process for {notebook_path}...")
        
        # Extract notebook content
        extracted_content = self.extract_notebook_content(notebook_path)
        
        # Group cells by question
        questions = {}
        for cell in extracted_content:
            question_id = cell.get('question_id')
            if question_id:
                if question_id not in questions:
                    questions[question_id] = []
                questions[question_id].append(cell)
        
        # Grade each question holistically
        grading_results = {}
        total_score = 0
        total_confidence = 0
        
        for question_id, cells in questions.items():
            print(f"Grading {question_id}...")
            
            # Consolidate all cells for this question
            question_content = self.consolidate_question_content(cells)
            
            # print(json.dumps(question_content, indent=2))
            # print(y)

            # Retrieve relevant rubric criteria for this question
            relevant_criteria = self.retrieve_rubric_criteria(
                question_content['all_content'], 
                question_id=question_id, 
                k=10
            )
            
            # Grade the entire question at once
            grade_result = self.generate_question_grade_with_simplified_cot(
                question_content, 
                relevant_criteria, 
                question_id
            )
            
            grading_results[question_id] = grade_result
            
            total_score += grade_result.get('final_score', 0)
            total_confidence += grade_result.get('confidence', 0)
        
        # Calculate overall statistics
        num_questions = len(grading_results)
        overall_results = {
            'notebook_path': notebook_path,
            'overall_score': total_score / num_questions if num_questions > 0 else 0,
            'overall_confidence': total_confidence / num_questions if num_questions > 0 else 0,
            'questions_graded': num_questions,
            'question_results': grading_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"Grading complete. Overall score: {overall_results['overall_score']:.2f}/10")
        return overall_results
    
    # ----------------------- End of Fixing Question Level grading Logic ----------------- # 
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        STEP 6: Save grading results to JSON file
        
        This function saves the complete grading results in a structured format
        that can be loaded later for analysis or display in the UI.
        
        Args:
            results: Grading results from grade_notebook()
            output_path: Path to save the JSON results file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

def create_sample_rubric():
    """
    Create a sample rubric file for testing
    """
    sample_rubric = {
        "question_1": {
            "criteria": [
                {
                    "aspect": "correctness",
                    "description": "Code produces correct output and solves the problem as specified",
                    "weight": 60
                },
                {
                    "aspect": "style",
                    "description": "Code follows Python best practices and PEP 8 style guidelines",
                    "weight": 20
                },
                {
                    "aspect": "explanation",
                    "description": "Markdown explanation clearly describes the approach and reasoning",
                    "weight": 20
                }
            ]
        },
        "question_2": {
            "criteria": [
                {
                    "aspect": "correctness",
                    "description": "Implementation correctly handles edge cases and produces expected results",
                    "weight": 50
                },
                {
                    "aspect": "efficiency",
                    "description": "Algorithm is efficient and uses appropriate data structures",
                    "weight": 30
                },
                {
                    "aspect": "documentation",
                    "description": "Code includes clear comments and docstrings",
                    "weight": 20
                }
            ]
        }
    }
    
    with open('sample_rubric.json', 'w') as f:
        json.dump(sample_rubric, f, indent=2)
    print("Sample rubric created as 'sample_rubric.json'")

# Example usage
if __name__ == "__main__":
    # Create sample rubric
    # create_sample_rubric()
    
    # Initialize grading system
    # grader = NotebookGradingSystem(
    #     hf_model_name="microsoft/DialoGPT-small"  # You can also try "distilgpt2" or "gpt2"
    # )
    # Initialize grading system with OpenAI
    grader = NotebookGradingSystem(
        model_name="gpt-4o-mini"  # Cost-effective choice, can also use "gpt-4o" or "gpt-4-turbo"
    )
    
    # Load rubric database
    grader.load_rubric_database('Test/sample_rubric.json')
    
    # Grade a notebook
    # results = grader.grade_notebook('Test/sample_notebook.ipynb')
    results = grader.grade_notebook('Test/incorrect_sample_notebook.ipynb')
    
    # Save results
    grader.save_results(results, 'Test/incorrect_grading_results.json')
    
    # print("Grading system initialized successfully!")
    # print("Next steps:")
    # print("1. Create your rubric JSON file or use the sample one")
    # print("2. Run grader.grade_notebook('path_to_notebook.ipynb')")
    # print("3. The system will use the free Hugging Face model for grading")