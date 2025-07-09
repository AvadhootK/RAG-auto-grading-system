# LLM-Based Auto-Grading Framework Using RAG

An automated, scalable grading system for Python-based data science notebooks, powered by **Large Language Models (LLMs)**, **Chain-of-Thought (CoT)** reasoning, and **Retrieval-Augmented Generation (RAG)**. This tool streamlines evaluation using rubric-based semantic analysis, static code checks, and interpretable scoring.

Designed to handle real-world course notebooks (e.g., data science, ML assignments), the system uses:
- **Vector similarity search (FAISS)** to retrieve relevant rubric components.
- **Prompt-engineered LLMs** (OpenAI / Hugging Face) to reason step-by-step through each submission.
- A **Streamlit UI** for easy interaction, upload, and result download.


## Features

1. **Notebook & Rubric Upload**  
   - Upload and grade multiple Jupyter notebooks  
   - Use custom or sample rubrics (JSON format)
2. **RAG-Based Retrieval**  
   - RAG-based rubric retrieval for modular and topic-aligned grading
3. **Prompt Engineering**  
   - Prompts built with structured fields (markdown, code, output, static analysis) and rubric data.
4.  **LLM Evaluation with Chain-of-Thought**  
    - Chain-of-Thought grading with LLMs (OpenAI / Hugging Face)
    - Static analysis of code for syntax and functional issues
    - Each question graded through detailed reasoning steps: Understanding → Correctness → Style → Output → Scoring.
5. **Structured Output Parsing**      
      - Structured output with scores, feedback, and confidence.
6. **Interactive Streamlit UI with Exportable Summary Reports**
  
#### Sample Rubric Format

```json
{
  "question_1": {
    "criteria": [
      {
        "aspect": "correctness",
        "description": "Code produces correct output",
        "weight": 60
      },
      {
        "aspect": "style",
        "description": "Code follows best practices",
        "weight": 40
      }
    ]
  }
}
```

## Demo

<img width="1465" alt="image" src="https://github.com/user-attachments/assets/05a1a81b-1433-4ced-a8ab-1c2b1c5a41ca" />  
 <br><br>
<img width="1461" alt="image" src="https://github.com/user-attachments/assets/924083d7-442b-4747-a42d-48a1cb888a0d" />




