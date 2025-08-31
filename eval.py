"""
Evaluation module for the OpenMind project.
Contains functions for evaluating the database query pipeline, displaying schemas, and testing relevance checking.
"""

import os
import yaml
from langchain_community.utilities.sql_database import SQLDatabase

from json_repair import repair_json


def demo_relevance_checking(stream_graph_updates, relevance_threshold):
    """Demonstrate the relevance checking functionality."""
    print("=" * 80)
    print("🎯 RELEVANCE CHECKING DEMO")
    print("=" * 80)
    
    # Example queries to test
    test_queries = [
        "How many employees are in the company?",  # Good query
        "What's the weather like today?",          # Irrelevant query
        "Tell me about the budget for defense.",   # Good query
    ]
    
    print(f"Testing all queries with relevance threshold: {relevance_threshold}")
    print("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}/{len(test_queries)}: {query}")
        try:
            stream_graph_updates(query)
        except Exception as e:
            print(f"❌ Error during testing: {e}")
        print("-" * 60)


def test_improved_relevance_system(stream_graph_updates):
    """Test the improved relevance system with proper state management."""
    print("🧪 Testing improved relevance checking system...")
    
    # Test query
    test_query = "How many employees are there?"
    
    try:
        print(f"Testing query: '{test_query}'")
        stream_graph_updates(test_query)
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def evaluate_flow(pipeline_dir, graph):
    """Evaluate the flow by running ground truth SQL queries with formatted output."""

    PROMPT = """You will be given a question and a ground-truth answer. Then you will be given a response from the LLM. You will need to provide confidence score for the answer based on the ground-truth answer. The confidence score should be between 0 and 100. 100 is the highest confidence score. Give answer in the JSON format.
    
    {
        "confidence_score": <confidence_score>,
        "reasoning": "<brief reasoning of why this score>"
    }

    """

    try:
        # Load ground truth data
        ground_truth_path = os.path.join(pipeline_dir, "ground_truth.yaml")
        with open(ground_truth_path, 'r') as file:
            ground_truth = yaml.safe_load(file)
    
        
        # Process all evaluation cases
        evaluation_results = []
        
        if "evals" in ground_truth:
            for db_name, test_cases in ground_truth["evals"].items():
                print(f"🎯 EVALUATING {db_name.upper().replace('_', ' ')}")
                print("=" * 80)
                
                # Select the appropriate database
                current_db = SQLDatabase.from_uri(f"sqlite:///{db_name}")
                
                for i, test_case in enumerate(test_cases, 1):
                    query = test_case["query"]
                    ground_truth_sql = test_case["ground_truth"]["sql"]
                    
                    print(f"\n📝 Test Case #{i}")
                    print(f"   Natural Language: {query}")
                    print(f"   SQL Query: {ground_truth_sql}")
                    print("-" * 60)
                    
                    try:
                        # Execute the SQL query and get results
                        result = current_db.run(ground_truth_sql)
                        
                        # Format and display the results
                        print("📊 Ground Truth Results:")
                        
                        if isinstance(result, str):
                            # Handle simple string results
                            lines = result.strip().split('\n')
                            if len(lines) == 1:
                                # Single result (like COUNT)
                                print(f"   ✅ {result}")
                            else:
                                # Multiple lines - format as table
                                for j, line in enumerate(lines):
                                    if j == 0:
                                        print(f"   Header: {line}")
                                        print("   " + "-" * (len(line) + 5))
                                    else:
                                        print(f"   Data:   {line}")
                        else:
                            # Handle other result types
                            print(f"   ✅ {result}")
                        
                        # Now test the LLM with the same query
                        print("\n🤖 Testing LLM Response:")
                        print("   " + "-" * 40)
                        
                        llm_config = {
                            "configurable": {
                                "thread_id": f"llm_test_{i}"
                            }
                        }
                        
                        try:
                            llm_response = graph.invoke({"messages": [{"role": "user", "content": query}]}, config=llm_config)
                            llm_answer = llm_response["messages"][-1].content
                            print(f"   🗣️  LLM Answer: {llm_answer}")
                            
                            # Evaluate the LLM response against ground truth
                            print("\n🎯 Confidence Evaluation:")
                            eval_prompt = f"""You will be given a question and a ground-truth answer. Then you will be given a response from the LLM. You will need to provide confidence score for the answer based on the ground-truth answer. The confidence score should be between 0 and 100. 100 is the highest confidence score. Give answer in the JSON format.
                            
                            {{
                                "confidence_score": <confidence_score>,
                                "explanation": "<brief explanation of why this score>"
                            }}
                            
                            Question: {query}
                            Ground-truth answer: {result}
                            LLM answer: {llm_answer}
                            """
                            
                            eval_config = {
                                "configurable": {
                                    "thread_id": f"eval_test_{i}"
                                }
                            }
                            
                            eval_response = graph.invoke({"messages": [{"role": "user", "content": eval_prompt}]}, config=eval_config)
                            eval_result = eval_response["messages"][-1].content
                            
                            try:
                                # Try to parse the JSON response
                                confidence_data = repair_json(eval_result, return_objects=True)
                                confidence_score = confidence_data.get("confidence_score", "Unknown")
                                explanation = confidence_data.get("explanation", "No explanation provided")
                                print(f"   📊 Confidence Score: {confidence_score}/100")
                                print(f"   💭 Explanation: {explanation}")
                            except Exception as json_error:
                                print(f"   ⚠️  Could not parse evaluation JSON: {json_error}")
                                print(f"   📝 Raw evaluation: {eval_result[:200]}...")
                                confidence_score = "Unknown"
                                explanation = "JSON parsing failed"
                            
                            evaluation_results.append({
                                "test_case": i,
                                "database": db_name,
                                "query": query,
                                "sql": ground_truth_sql,
                                "ground_truth_result": result,
                                "llm_answer": llm_answer,
                                "confidence_score": confidence_score,
                                "explanation": explanation,
                                "status": "success"
                            })
                            
                        except Exception as llm_error:
                            print(f"   ❌ Error getting LLM response: {llm_error}")
                            evaluation_results.append({
                                "test_case": i,
                                "database": db_name,
                                "query": query,
                                "sql": ground_truth_sql,
                                "ground_truth_result": result,
                                "llm_answer": None,
                                "confidence_score": 0,
                                "explanation": f"LLM error: {str(llm_error)}",
                                "status": "llm_error"
                            })
                        
                    except Exception as e:
                        print(f"   ❌ Error executing ground truth query: {e}")
                        evaluation_results.append({
                            "test_case": i,
                            "database": db_name,
                            "query": query,
                            "sql": ground_truth_sql,
                            "ground_truth_result": None,
                            "llm_answer": None,
                            "confidence_score": 0,
                            "explanation": f"Ground truth query error: {str(e)}",
                            "status": "ground_truth_error"
                        })
                    
                    print("-" * 60)
        
        # Enhanced Summary with confidence scores
        print("\n📈 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        
        successful_tests = sum(1 for r in evaluation_results if r["status"] == "success")
        llm_errors = sum(1 for r in evaluation_results if r["status"] == "llm_error")
        ground_truth_errors = sum(1 for r in evaluation_results if r["status"] == "ground_truth_error")
        total_tests = len(evaluation_results)
        
        print(f"📊 Test Results:")
        print(f"   ✅ Successful Evaluations: {successful_tests}/{total_tests}")
        print(f"   🤖 LLM Response Errors: {llm_errors}/{total_tests}")
        print(f"   🗃️  Ground Truth Errors: {ground_truth_errors}/{total_tests}")
        print(f"   📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "📈 Success Rate: N/A")
        
        # Calculate average confidence score for successful tests
        if successful_tests > 0:
            confidence_scores = []
            for result in evaluation_results:
                if result["status"] == "success" and isinstance(result.get("confidence_score"), (int, float)):
                    confidence_scores.append(result["confidence_score"])
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"\n🎯 Confidence Analysis:")
                print(f"   📊 Average Confidence: {avg_confidence:.1f}/100")
                print(f"   🔝 Highest Confidence: {max(confidence_scores)}/100")
                print(f"   🔻 Lowest Confidence: {min(confidence_scores)}/100")
                
                # Confidence distribution
                high_conf = sum(1 for score in confidence_scores if score >= 80)
                medium_conf = sum(1 for score in confidence_scores if 50 <= score < 80)
                low_conf = sum(1 for score in confidence_scores if score < 50)
                
                print(f"   🟢 High Confidence (≥80): {high_conf}")
                print(f"   🟡 Medium Confidence (50-79): {medium_conf}")
                print(f"   🔴 Low Confidence (<50): {low_conf}")
        
        print("=" * 100)
        
        # Return comprehensive evaluation results
        confidence_scores = [r.get("confidence_score") for r in evaluation_results 
                           if r["status"] == "success" and isinstance(r.get("confidence_score"), (int, float))]
        
        return {
            "evaluation_results": evaluation_results,
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "llm_errors": llm_errors,
                "ground_truth_errors": ground_truth_errors,
                "success_rate": (successful_tests/total_tests)*100 if total_tests > 0 else 0,
                "confidence_scores": {
                    "average": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                    "highest": max(confidence_scores) if confidence_scores else 0,
                    "lowest": min(confidence_scores) if confidence_scores else 0,
                    "distribution": {
                        "high_confidence": sum(1 for score in confidence_scores if score >= 80),
                        "medium_confidence": sum(1 for score in confidence_scores if 50 <= score < 80),
                        "low_confidence": sum(1 for score in confidence_scores if score < 50)
                    }
                }
            }
        }
    except FileNotFoundError:
        print(f"❌ Error: ground_truth.yaml file not found at {ground_truth_path}!")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error executing SQL query: {e}")
        return None


# Convenience function to run evaluation with proper imports
def run_evaluation():
    """Run the evaluation with all necessary imports and setup."""
    try:
        from pipeline import pipeline_dir, graph
        return evaluate_flow(pipeline_dir, graph)
    except ImportError as e:
        print(f"❌ Error importing from pipeline: {e}")
        return None


# Convenience function for relevance demo
def run_relevance_demo():
    """Run relevance checking demo with proper imports."""
    try:
        from pipeline import stream_graph_updates, RELEVANCE_THRESHOLD
        demo_relevance_checking(stream_graph_updates, RELEVANCE_THRESHOLD)
    except ImportError as e:
        print(f"❌ Error importing from pipeline: {e}")


# Convenience function for relevance system test
def run_relevance_test():
    """Run relevance system test with proper imports."""
    try:
        from pipeline import stream_graph_updates
        test_improved_relevance_system(stream_graph_updates)
    except ImportError as e:
        print(f"❌ Error importing from pipeline: {e}")


if __name__ == "__main__":
    """Allow direct execution of evaluation functions."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "evaluate":
            print("🚀 Running full evaluation...")
            run_evaluation()
        elif command == "demo":
            print("🚀 Running relevance demo...")
            run_relevance_demo()
        elif command == "test":
            print("🚀 Running relevance test...")
            run_relevance_test()
        else:
            print("Usage: python eval.py [evaluate|demo|test]")
    else:
        print("🔍 OpenMind - Evaluation Module")
        print("Available commands:")
        print("  python eval.py evaluate  - Run full evaluation")
        print("  python eval.py demo      - Run relevance checking demo")
        print("  python eval.py test      - Run relevance system test")
