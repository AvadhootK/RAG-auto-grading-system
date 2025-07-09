# import streamlit as st
# import json
# import pandas as pd
# from pathlib import Path
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import os
# import tempfile
# import traceback

# # Import the grading system
# from llm_autograder import NotebookGradingSystem, create_sample_rubric

# st.set_page_config(
#     page_title="Automated Notebook Grading System",
#     page_icon="📝",
#     layout="wide"
# )

# def load_grading_results(results_file):
#     """Load grading results from JSON file"""
#     try:
#         with open(results_file, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         st.error(f"Error loading results: {e}")
#         return None

# def display_confidence_indicator(confidence):
#     """Display confidence level with color coding"""
#     if confidence >= 90:
#         color = "green"
#         label = "High"
#     elif confidence >= 70:
#         color = "orange"
#         label = "Medium"
#     else:
#         color = "red"
#         label = "Low"
    
#     st.markdown(f"""
#     <div style="display: inline-block; padding: 4px 8px; border-radius: 4px; 
#                 background-color: {color}; color: white; font-weight: bold;">
#         {label} ({confidence}%)
#     </div>
#     """, unsafe_allow_html=True)

# def display_rubric_breakdown(aspect_scores):
#     """Display rubric breakdown as a progress bar chart"""
#     if not aspect_scores:
#         st.warning("No aspect scores available")
#         return
    
#     aspects = list(aspect_scores.keys())
#     scores = [aspect_scores[aspect]['score'] for aspect in aspects]
#     max_scores = [aspect_scores[aspect].get('max_score', 10) for aspect in aspects]
#     percentages = [score/max_score * 100 for score, max_score in zip(scores, max_scores)]
    
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=aspects,
#         y=percentages,
#         text=[f"{score:.1f}/{max_score}" for score, max_score in zip(scores, max_scores)],
#         textposition='auto',
#         marker_color=['green' if p >= 80 else 'orange' if p >= 60 else 'red' for p in percentages]
#     ))
    
#     fig.update_layout(
#         title="Rubric Breakdown",
#         xaxis_title="Assessment Aspects",
#         yaxis_title="Score Percentage",
#         yaxis=dict(range=[0, 100]),
#         height=400
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

# def display_reasoning_steps(reasoning_steps):
#     """Display Chain of Thought reasoning steps"""
#     if not reasoning_steps:
#         st.warning("No reasoning steps available")
#         return
    
#     step_titles = {
#         'understanding': '🎯 Understanding Phase',
#         'correctness': '✅ Correctness Phase', 
#         'quality': '⭐ Quality Phase',
#         'explanation': '📝 Explanation Phase',
#         'output': '📊 Output Phase'
#     }
    
#     for step, content in reasoning_steps.items():
#         if content:
#             with st.expander(step_titles.get(step, step.title())):
#                 st.write(content)

# def create_temp_file(uploaded_file, suffix):
#     """Create a temporary file from uploaded content"""
#     temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False)
#     temp_file.write(uploaded_file.read())
#     temp_file.close()
#     return temp_file.name

# def main():
#     st.title("📝 Automated Notebook Grading System")
#     st.markdown("### RAG + Chain of Thought Grading with LLM")
    
#     # Initialize session state
#     if 'grading_results' not in st.session_state:
#         st.session_state.grading_results = None
    
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Select Page", ["Grade New Notebook", "View Results", "System Settings"])
    
#     if page == "Grade New Notebook":
#         st.header("Grade a New Notebook")
        
#         # File upload
#         uploaded_file = st.file_uploader("Upload Jupyter Notebook", type=['ipynb'])
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Model provider selection
#             model_provider = st.radio("Select Model Provider", 
#                                     ["OpenAI (API Key Required)", "Hugging Face (Free)"])
            
#             if model_provider == "OpenAI (API Key Required)":
#                 # OpenAI API Key input
#                 api_key = st.text_input("OpenAI API Key", type="password", 
#                                        help="Enter your OpenAI API key")
                
#                 # Model selection
#                 model_name = st.selectbox("Select Model", 
#                                         ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
#             else:
#                 st.info("Using free Hugging Face model")
#                 api_key = None
#                 model_name = st.selectbox("Select HF Model", 
#                                         ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium", 
#                                          "distilgpt2", "gpt2"])
        
#         with col2:
#             # Rubric file upload
#             rubric_file = st.file_uploader("Upload Rubric JSON", type=['json'])
            
#             # Option to create sample rubric
#             if st.button("Create Sample Rubric"):
#                 create_sample_rubric()
#                 st.success("Sample rubric created as 'sample_rubric.json'")
#                 st.info("You can download this file and modify it for your needs")
            
#             # Grading options
#             confidence_threshold = st.slider("Confidence Threshold for Review", 
#                                             min_value=0, max_value=100, value=70,
#                                             help="Grades below this confidence level will be flagged for review")
        
#         # Display rubric format helper
#         with st.expander("📋 Rubric Format Helper"):
#             st.json({
#                 "question_1": {
#                     "criteria": [
#                         {
#                             "aspect": "correctness",
#                             "description": "Code produces correct output and solves the problem as specified",
#                             "weight": 60
#                         },
#                         {
#                             "aspect": "style",
#                             "description": "Code follows Python best practices and PEP 8 style guidelines",
#                             "weight": 20
#                         },
#                         {
#                             "aspect": "explanation",
#                             "description": "Markdown explanation clearly describes the approach and reasoning",
#                             "weight": 20
#                         }
#                     ]
#                 }
#             })
        
#         if st.button("Start Grading", type="primary"):
#             if uploaded_file and rubric_file:
#                 if model_provider == "OpenAI (API Key Required)" and not api_key:
#                     st.error("Please provide OpenAI API key")
#                     return
                
#                 try:
#                     # Create temporary files
#                     notebook_path = create_temp_file(uploaded_file, '.ipynb')
#                     rubric_path = create_temp_file(rubric_file, '.json')
                    
#                     # Initialize grading system
#                     with st.spinner("Initializing grading system..."):
#                         if model_provider == "OpenAI (API Key Required)":
#                             # Set OpenAI API key in environment
#                             os.environ['OPENAI_API_KEY'] = api_key
#                             grader = NotebookGradingSystem(model_name=model_name)
#                         else:
#                             grader = NotebookGradingSystem(hf_model_name=model_name)
                        
#                         grader.load_rubric_database(rubric_path)
                    
#                     # Grade the notebook
#                     with st.spinner("Grading notebook... This may take a few minutes."):
#                         results = grader.grade_notebook_simplified(notebook_path)
                        
#                         # Save results
#                         results_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#                         grader.save_results(results, results_file)
                        
#                         # Store in session state
#                         st.session_state.grading_results = results
                    
#                     st.success("Grading completed!")
                    
#                     # Display results
#                     display_grading_results(results, confidence_threshold)
                    
#                 except Exception as e:
#                     st.error(f"Error during grading: {str(e)}")
#                     st.error(traceback.format_exc())
                
#                 finally:
#                     # Clean up temporary files
#                     try:
#                         if 'notebook_path' in locals():
#                             os.unlink(notebook_path)
#                         if 'rubric_path' in locals():
#                             os.unlink(rubric_path)
#                     except:
#                         pass
                
#             else:
#                 st.error("Please upload both a notebook file and a rubric file")
    
#     elif page == "View Results":
#         st.header("View Grading Results")
        
#         # Show current session results
#         if st.session_state.grading_results:
#             st.subheader("Current Session Results")
#             display_grading_results(st.session_state.grading_results, confidence_threshold=70)
#             st.divider()
        
#         # Load existing results from files
#         st.subheader("Load Previous Results")
#         results_files = list(Path(".").glob("results_*.json"))
        
#         if results_files:
#             selected_file = st.selectbox("Select Results File", results_files)
            
#             if selected_file and st.button("Load Results"):
#                 results = load_grading_results(selected_file)
#                 if results:
#                     display_grading_results(results, confidence_threshold=70)
#         else:
#             st.info("No previous grading results found. Grade some notebooks first!")
    
#     elif page == "System Settings":
#         st.header("System Settings")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Model Settings")
            
#             # Model provider selection
#             model_provider = st.radio("Select Model Provider", 
#                                     ["OpenAI (API Key Required)", "Hugging Face (Free)"])
            
#             if model_provider == "OpenAI (API Key Required)":
#                 openai_model = st.selectbox("Select OpenAI Model", [
#                     "gpt-4o-mini",
#                     "gpt-4o",
#                     "gpt-4-turbo",
#                     "gpt-3.5-turbo"
#                 ])
#                 st.warning("OpenAI models require API key and usage fees")
#                 hf_model = None
#             else:
#                 hf_model = st.selectbox("Select Hugging Face Model", [
#                     "microsoft/DialoGPT-small",
#                     "microsoft/DialoGPT-medium", 
#                     "distilgpt2",
#                     "gpt2"
#                 ])
#                 st.info("These models are free and run locally after download")
#                 openai_model = None
        
#         with col2:
#             st.subheader("Grading Settings")
            
#             confidence_threshold = st.slider("Default Confidence Threshold", 
#                                             min_value=0, max_value=100, value=70)
            
#             auto_save = st.checkbox("Auto-save results", value=True)
            
#             detailed_feedback = st.checkbox("Generate detailed feedback", value=True)
            
#             enable_batch_processing = st.checkbox("Enable batch processing", value=False)
        
#         if st.button("Save Settings"):
#             settings = {
#                 "model_provider": model_provider,
#                 "hf_model": hf_model,
#                 "openai_model": openai_model,
#                 "confidence_threshold": confidence_threshold,
#                 "auto_save": auto_save,
#                 "detailed_feedback": detailed_feedback,
#                 "enable_batch_processing": enable_batch_processing
#             }
            
#             with open("grading_settings.json", "w") as f:
#                 json.dump(settings, f, indent=2)
            
#             st.success("Settings saved successfully!")

# def display_grading_results(results, confidence_threshold=70):
#     """Display comprehensive grading results"""
    
#     # Overall statistics
#     st.subheader("Overall Results")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Overall Score", f"{results['overall_score']:.1f}/10")
    
#     with col2:
#         st.metric("Overall Confidence", f"{results['overall_confidence']:.0f}%")
    
#     with col3:
#         st.metric("Questions Graded", results['questions_graded'])
    
#     with col4:
#         # Flag for review if confidence is low
#         needs_review = results['overall_confidence'] < confidence_threshold
#         st.metric("Review Required", "Yes" if needs_review else "No")
    
#     # Confidence indicator
#     st.write("**Confidence Level:**")
#     display_confidence_indicator(results['overall_confidence'])
    
#     # Question-by-question breakdown
#     st.subheader("Question-by-Question Results")
    
#     for question_id, question_result in results['question_results'].items():
#         with st.expander(f"📋 {question_id.replace('_', ' ').title()}", expanded=True):
            
#             # Question metrics
#             q_col1, q_col2 = st.columns(2)
            
#             with q_col1:
#                 st.metric("Question Score", f"{question_result.get('final_score', 0):.1f}/10")
            
#             with q_col2:
#                 st.metric("Question Confidence", f"{question_result.get('confidence', 0):.0f}%")
            
#             # Display aspect scores if available
#             if question_result.get('aspect_scores'):
#                 st.write("**Aspect Breakdown:**")
#                 display_rubric_breakdown(question_result['aspect_scores'])
            
#             # Display reasoning steps or summary
#             if question_result.get('reasoning_steps'):
#                 st.write("**Chain of Thought Reasoning:**")
#                 display_reasoning_steps(question_result['reasoning_steps'])
#             elif question_result.get('reasoning_summary'):
#                 st.write("**Reasoning Summary:**")
#                 st.text_area("", question_result['reasoning_summary'], height=200, disabled=True)
            
#             # Feedback
#             if question_result.get('feedback'):
#                 st.write("**Feedback:**")
#                 st.info(question_result['feedback'])
            
#             # Show raw response in expandable section
#             if question_result.get('raw_response'):
#                 with st.expander("View Raw Grading Response"):
#                     st.text_area("", question_result['raw_response'], height=300, disabled=True)
            
#             # Manual override option
#             st.write("**Manual Override:**")
#             override_col1, override_col2 = st.columns(2)
            
#             with override_col1:
#                 new_score = st.number_input(
#                     f"Override Score for {question_id}",
#                     min_value=0.0,
#                     max_value=10.0,
#                     value=float(question_result.get('final_score', 0)),
#                     step=0.1,
#                     key=f"override_{question_id}"
#                 )
            
#             with override_col2:
#                 override_reason = st.text_input(
#                     f"Override Reason for {question_id}",
#                     key=f"reason_{question_id}"
#                 )
            
#             if st.button(f"Apply Override for {question_id}", key=f"apply_{question_id}"):
#                 if override_reason:
#                     st.success(f"Override applied: {new_score}/10 - {override_reason}")
#                     # You could save this override to the results here
#                 else:
#                     st.warning("Please provide a reason for the override")
            
#             st.divider()
    
#     # Export options
#     st.subheader("Export Results")
    
#     export_col1, export_col2, export_col3 = st.columns(3)
    
#     with export_col1:
#         if st.button("Export as JSON"):
#             json_str = json.dumps(results, indent=2)
#             st.download_button(
#                 label="Download JSON",
#                 data=json_str,
#                 file_name=f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                 mime="application/json"
#             )
    
#     with export_col2:
#         if st.button("Export as CSV"):
#             # Convert to CSV format
#             csv_data = []
#             for question_id, question_result in results['question_results'].items():
#                 csv_data.append({
#                     'Question': question_id,
#                     'Score': question_result.get('final_score', 0),
#                     'Confidence': question_result.get('confidence', 0),
#                     'Feedback': question_result.get('feedback', '')
#                 })
            
#             df = pd.DataFrame(csv_data)
#             csv_str = df.to_csv(index=False)
#             st.download_button(
#                 label="Download CSV",
#                 data=csv_str,
#                 file_name=f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv"
#             )
    
#     with export_col3:
#         if st.button("Generate Report"):
#             # Generate a summary report
#             report = f"""
# # Grading Report

# **Notebook:** {results['notebook_path']}
# **Overall Score:** {results['overall_score']:.1f}/10
# **Overall Confidence:** {results['overall_confidence']:.0f}%
# **Questions Graded:** {results['questions_graded']}
# **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ## Question Breakdown

# """
#             for question_id, question_result in results['question_results'].items():
#                 report += f"### {question_id.replace('_', ' ').title()}\n"
#                 report += f"- Score: {question_result.get('final_score', 0):.1f}/10\n"
#                 report += f"- Confidence: {question_result.get('confidence', 0):.0f}%\n"
#                 if question_result.get('feedback'):
#                     report += f"- Feedback: {question_result['feedback']}\n"
#                 report += "\n"
            
#             st.download_button(
#                 label="Download Report",
#                 data=report,
#                 file_name=f"grading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
#                 mime="text/markdown"
#             )

# if __name__ == "__main__":
#     main()



# --------------- simplified ---------------- #
import streamlit as st
import json
import os
import tempfile
import traceback
from datetime import datetime

# Import the grading system
from llm_autograder import NotebookGradingSystem, create_sample_rubric

st.set_page_config(
    page_title="Automated Notebook Grading System",
    page_icon="📝",
    layout="wide"
)

def create_temp_file(uploaded_file, suffix):
    """Create a temporary file from uploaded content"""
    temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name

def main():
    st.title("📝 Automated Notebook Grading System")
    st.markdown("### RAG + Chain of Thought Grading with LLM")
    
    # File uploads
    st.header("Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Jupyter Notebook", type=['ipynb'])
    
    with col2:
        rubric_file = st.file_uploader("Upload Rubric JSON", type=['json'])
    
    # Model selection
    st.header("Model Settings")
    
    model_provider = st.radio("Select Model Provider", 
                            ["OpenAI", "Hugging Face"])
    
    if model_provider == "OpenAI":
        # Check if API key exists in environment
        env_api_key = os.getenv('OPENAI_API_KEY')
        
        if env_api_key:
            st.success("✅ OpenAI API key found in environment variables")
            api_key = env_api_key
            # Option to override with manual input
            if st.checkbox("Override with manual API key"):
                api_key = st.text_input("OpenAI API Key Override", type="password")
                if not api_key:
                    api_key = env_api_key
        else:
            st.warning("⚠️ No OpenAI API key found in environment")
            api_key = st.text_input("OpenAI API Key", type="password")
        
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        api_key = None
        model_name = st.selectbox("Model", ["microsoft/DialoGPT-small", "distilgpt2", "gpt2"])
    
    # Helper buttons
    st.header("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Sample Rubric"):
            create_sample_rubric()
            st.success("Sample rubric created as 'sample_rubric.json'")
    
    with col2:
        if st.button("Show Rubric Format"):
            st.json({
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
            })
    
    # Main grading button
    st.header("Grade Notebook")
    
    if st.button("Start Grading", type="primary"):
        if not uploaded_file or not rubric_file:
            st.error("Please upload both notebook and rubric files")
            return
        
        if model_provider == "OpenAI" and not api_key:
            st.error("Please provide OpenAI API key")
            return
        
        try:
            # Create temporary files
            notebook_path = create_temp_file(uploaded_file, '.ipynb')
            rubric_path = create_temp_file(rubric_file, '.json')
            
            # Initialize grading system
            with st.spinner("Setting up grader..."):
                if model_provider == "OpenAI":
                    # Set API key in environment if provided via UI
                    if api_key and api_key != os.getenv('OPENAI_API_KEY'):
                        os.environ['OPENAI_API_KEY'] = api_key
                    grader = NotebookGradingSystem(model_name=model_name)
                else:
                    grader = NotebookGradingSystem(hf_model_name=model_name)
                
                grader.load_rubric_database(rubric_path)
            
            # Grade the notebook
            with st.spinner("Grading notebook..."):
                results = grader.grade_notebook(notebook_path)
            
            st.success("Grading completed!")
            
            # Display results
            display_results(results)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            with st.expander("See error details"):
                st.text(traceback.format_exc())
        
        finally:
            # Clean up temporary files
            try:
                if 'notebook_path' in locals():
                    os.unlink(notebook_path)
                if 'rubric_path' in locals():
                    os.unlink(rubric_path)
            except:
                pass

def display_results(results):
    """Display grading results with improved formatting"""
    
    st.header("📊 Grading Results")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = results.get('overall_score', 0)
        st.metric(
            "Overall Score", 
            f"{score:.1f}/10",
            delta=f"{((score/10)*100):.1f}%" if score > 0 else None
        )
    
    with col2:
        confidence = results.get('overall_confidence', 0)
        st.metric("Confidence", f"{confidence:.0f}%")
    
    with col3:
        st.metric("Questions Graded", results.get('questions_graded', 0))
    
    # Question-by-question results
    st.subheader("📝 Question Details")
    
    question_results = results.get('question_results', {})
    
    for question_id, question_result in question_results.items():
        with st.expander(f"🔍 {question_id.replace('_', ' ').title()}", expanded=True):
            
            # Check if there's an error
            if question_result.get('error'):
                st.error(f"Error grading this question: {question_result['error']}")
                continue
            
            # Score and confidence in columns
            col1, col2 = st.columns(2)
            
            with col1:
                final_score = question_result.get('final_score', 0)
                st.metric("Score", f"{final_score:.1f}/10")
            
            with col2:
                confidence = question_result.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.0f}%")
            
            # Rubric breakdown if available
            if question_result.get('rubric_breakdown') or question_result.get('aspect_scores'):
                st.write("**📋 Rubric Breakdown:**")
                
                # Handle both old and new formats
                breakdown = question_result.get('rubric_breakdown', question_result.get('aspect_scores', {}))
                
                if breakdown:
                    breakdown_cols = st.columns(len(breakdown))
                    for i, (aspect, score_info) in enumerate(breakdown.items()):
                        with breakdown_cols[i]:
                            if isinstance(score_info, dict):
                                score = score_info.get('score', 0)
                                max_score = score_info.get('max', 10)
                                weight = score_info.get('weight', 0)
                                st.metric(
                                    f"{aspect.title()}", 
                                    f"{score:.1f}/{max_score}",
                                    delta=f"Weight: {weight}%" if weight > 0 else None
                                )
                            else:
                                st.metric(f"{aspect.title()}", f"{score_info:.1f}/10")
            
            # Feedback
            feedback = question_result.get('feedback', '')
            if feedback:
                st.write("**💬 Feedback:**")
                st.info(feedback)
            
            # Reasoning details
            reasoning = question_result.get('reasoning_summary', '')
            if reasoning:
                st.write("**🧠 Reasoning:**")
                with st.expander("View detailed reasoning", expanded=False):
                    st.text_area(
                        "Reasoning Details", 
                        reasoning, 
                        height=200, 
                        disabled=True, 
                        key=f"reasoning_{question_id}",
                        label_visibility="collapsed"
                    )
            
            # Reasoning steps (if available)
            reasoning_steps = question_result.get('reasoning_steps', {})
            if reasoning_steps:
                st.write("**🔄 Reasoning Steps:**")
                for step_name, step_content in reasoning_steps.items():
                    with st.expander(f"{step_name.replace('_', ' ').title()}", expanded=False):
                        st.write(step_content)
            
            # Rubric criteria used
            criteria_used = question_result.get('rubric_criteria_used', [])
            if criteria_used:
                st.write("**📚 Rubric Criteria Applied:**")
                for criterion in criteria_used:
                    st.write(f"- **{criterion.get('aspect', 'Unknown').title()}**: {criterion.get('description', 'No description')} (Weight: {criterion.get('weight', 0)}%)")
    
    # Export section
    st.subheader("📥 Export Results")
    
    # Create downloadable JSON
    json_str = json.dumps(results, indent=2)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download Results (JSON)",
            data=json_str,
            file_name=f"grading_results_{timestamp}.json",
            mime="application/json"
        )
    
    with col2:
        # Create a summary report
        summary_report = create_summary_report(results)
        st.download_button(
            label="📊 Download Summary Report",
            data=summary_report,
            file_name=f"summary_report_{timestamp}.txt",
            mime="text/plain"
        )

def create_summary_report(results):
    """Create a text summary report"""
    report = f"""GRADING SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL RESULTS:
- Overall Score: {results.get('overall_score', 0):.1f}/10
- Overall Confidence: {results.get('overall_confidence', 0):.0f}%
- Questions Graded: {results.get('questions_graded', 0)}

QUESTION BREAKDOWN:
"""
    
    for question_id, question_result in results.get('question_results', {}).items():
        report += f"\n{question_id.replace('_', ' ').title()}:\n"
        report += f"  Score: {question_result.get('final_score', 0):.1f}/10\n"
        report += f"  Confidence: {question_result.get('confidence', 0):.0f}%\n"
        
        feedback = question_result.get('feedback', '')
        if feedback:
            report += f"  Feedback: {feedback}\n"
        
        report += "\n"
    
    return report

if __name__ == "__main__":
    main()