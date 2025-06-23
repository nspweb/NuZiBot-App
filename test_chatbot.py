#!/usr/bin/env python3
"""
NuZiBot Evaluation System
Comprehensive testing and performance analysis for the Nutrition Chatbot
Author: Your Name
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import chatbot model
from chatbot_model import NutritionChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatbotEvaluator:
    """
    Comprehensive evaluation system for NuZiBot nutrition chatbot
    """
    
    def __init__(self, food_csv_path: str, nutrition_excel_path: str, llm_type: str = "groq"):
        """
        Initialize the evaluator
        
        Args:
            food_csv_path: Path to food database CSV
            nutrition_excel_path: Path to nutrition requirements Excel
            llm_type: Type of LLM to use ("groq" or "gemini")
        """
        self.food_csv_path = food_csv_path
        self.nutrition_excel_path = nutrition_excel_path
        self.llm_type = llm_type
        
        # Load chatbot
        self.chatbot = self._load_chatbot()
        
        # Define test cases
        self.test_questions = self._initialize_test_cases()
        
        # Results storage
        self.results = []
        self.metrics = {}
        
        # Create output directory
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"ChatbotEvaluator initialized with {len(self.test_questions)} test cases")
    
    def _load_chatbot(self) -> NutritionChatbot:
        """Load the nutrition chatbot"""
        try:
            logger.info("Loading NutritionChatbot...")
            chatbot = NutritionChatbot(
                food_csv_path=self.food_csv_path,
                nutrition_excel_path=self.nutrition_excel_path,
                llm_type=self.llm_type
            )
            # Set sample user profile
            sample_profile = {
                "age": 15,
                "gender": "Male",
                "height": 160.0,
                "weight": 55.0,
                "bmi": 21.5,
                "bmr": 1400,
                "tdee": 2000,
                "target_energy_intake": 2000,  # <-- Tambahkan baris ini
                "activity_level": "Moderately Active",
                "special_conditions": ["None"],
                "dietary_preferences": []
            }
            chatbot.update_user_profile(sample_profile)
            logger.info("✅ Chatbot loaded successfully")
            return chatbot
        except Exception as e:
            logger.error(f"❌ Failed to load chatbot: {str(e)}")
            raise
    
    def _initialize_test_cases(self) -> List[Dict]:
        """Initialize comprehensive test cases"""
        return [
            # === NUTRITION KNOWLEDGE TESTS ===
            {
                "id": "NK001",
                "question": "What foods are high in protein?",
                "category": "nutrition_knowledge",
                "subcategory": "macronutrients",
                "expected_keywords": ["protein", "meat", "fish", "eggs", "beans", "nuts", "dairy"],
                "is_nutrition_related": True,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "NK002",
                "question": "What are good sources of vitamin C?",
                "category": "nutrition_knowledge",
                "subcategory": "vitamins",
                "expected_keywords": ["vitamin c", "orange", "citrus", "fruits", "vegetables", "berries"],
                "is_nutrition_related": True,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "NK003",
                "question": "How much water should a teenager drink daily?",
                "category": "nutrition_knowledge",
                "subcategory": "hydration",
                "expected_keywords": ["water", "daily", "liters", "glasses", "hydration", "teenager"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "teenager"
            },
            {
                "id": "NK004",
                "question": "What nutrients are important for bone health?",
                "category": "nutrition_knowledge",
                "subcategory": "minerals",
                "expected_keywords": ["calcium", "vitamin d", "magnesium", "phosphorus", "bone"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "general"
            },
            {
                "id": "NK005",
                "question": "Explain the difference between simple and complex carbohydrates",
                "category": "nutrition_knowledge",
                "subcategory": "macronutrients",
                "expected_keywords": ["simple", "complex", "carbohydrates", "sugar", "fiber", "starch"],
                "is_nutrition_related": True,
                "difficulty": "hard",
                "target_audience": "advanced"
            },
            
            # === MEAL PLANNING TESTS ===
            {
                "id": "MP001",
                "question": "Can you suggest a healthy breakfast for a 15-year-old?",
                "category": "meal_planning",
                "subcategory": "breakfast",
                "expected_keywords": ["breakfast", "healthy", "teenager", "nutrition", "balanced"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "teenager"
            },
            {
                "id": "MP002",
                "question": "What's a good lunch for someone with diabetes?",
                "category": "meal_planning",
                "subcategory": "lunch",
                "expected_keywords": ["lunch", "diabetes", "blood sugar", "low glycemic", "balanced"],
                "is_nutrition_related": True,
                "difficulty": "hard",
                "target_audience": "medical"
            },
            {
                "id": "MP003",
                "question": "Suggest a post-workout snack for teenagers",
                "category": "meal_planning",
                "subcategory": "snacks",
                "expected_keywords": ["post-workout", "snack", "protein", "recovery", "teenager"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "teenager"
            },
            {
                "id": "MP004",
                "question": "Plan a balanced dinner for a family with children",
                "category": "meal_planning",
                "subcategory": "dinner",
                "expected_keywords": ["dinner", "balanced", "family", "children", "nutrition"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "family"
            },
            
            # === DIETARY RESTRICTIONS TESTS ===
            {
                "id": "DR001",
                "question": "What vegetarian foods are high in iron?",
                "category": "dietary_restrictions",
                "subcategory": "vegetarian",
                "expected_keywords": ["vegetarian", "iron", "plant-based", "spinach", "legumes"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "vegetarian"
            },
            {
                "id": "DR002",
                "question": "Are there lactose-free calcium sources?",
                "category": "dietary_restrictions",
                "subcategory": "lactose_intolerant",
                "expected_keywords": ["lactose-free", "calcium", "dairy alternatives", "fortified"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "lactose_intolerant"
            },
            {
                "id": "DR003",
                "question": "What can a child with nut allergies eat for protein?",
                "category": "dietary_restrictions",
                "subcategory": "allergies",
                "expected_keywords": ["nut allergies", "protein", "safe", "alternatives", "children"],
                "is_nutrition_related": True,
                "difficulty": "hard",
                "target_audience": "allergic"
            },
            {
                "id": "DR004",
                "question": "Vegan protein sources for growing teenagers",
                "category": "dietary_restrictions",
                "subcategory": "vegan",
                "expected_keywords": ["vegan", "protein", "plant-based", "teenagers", "growth"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "vegan"
            },
            
            # === HEALTH CONDITIONS TESTS ===
            {
                "id": "HC001",
                "question": "What foods help lower cholesterol?",
                "category": "health_conditions",
                "subcategory": "cardiovascular",
                "expected_keywords": ["cholesterol", "heart healthy", "oats", "fiber", "omega-3"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "medical"
            },
            {
                "id": "HC002",
                "question": "Diet recommendations for teenagers with acne",
                "category": "health_conditions",
                "subcategory": "skin_health",
                "expected_keywords": ["acne", "diet", "teenagers", "skin", "anti-inflammatory"],
                "is_nutrition_related": True,
                "difficulty": "medium",
                "target_audience": "teenager"
            },
            
            # === OFF-TOPIC TESTS (Should be rejected) ===
            {
                "id": "OT001",
                "question": "What's the weather like today?",
                "category": "off_topic",
                "subcategory": "weather",
                "expected_keywords": ["sorry", "nutrition", "topic", "outside", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "OT002",
                "question": "How do I fix my computer?",
                "category": "off_topic",
                "subcategory": "technology",
                "expected_keywords": ["sorry", "nutrition", "topic", "outside", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "OT003",
                "question": "Tell me a joke about cats",
                "category": "off_topic",
                "subcategory": "entertainment",
                "expected_keywords": ["sorry", "nutrition", "topic", "outside", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "OT004",
                "question": "What's the capital of France?",
                "category": "off_topic",
                "subcategory": "geography",
                "expected_keywords": ["sorry", "nutrition", "topic", "outside", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            },
            
            # === EDGE CASES ===
            {
                "id": "EC001",
                "question": "",  # Empty question
                "category": "edge_cases",
                "subcategory": "empty_input",
                "expected_keywords": ["sorry", "question", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            },
            {
                "id": "EC002",
                "question": "????????",  # Non-sense input
                "category": "edge_cases",
                "subcategory": "invalid_input",
                "expected_keywords": ["sorry", "understand", "help"],
                "is_nutrition_related": False,
                "difficulty": "easy",
                "target_audience": "general"
            }
        ]
    
    def evaluate_response_quality(self, question: str, response: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate response quality based on multiple criteria
        
        Args:
            question: Original question
            response: Chatbot response
            expected_keywords: Keywords expected in response
            
        Returns:
            Dictionary with quality metrics
        """
        if not response:
            return {
                "quality_score": 0,
                "keyword_coverage": 0,
                "matched_keywords": [],
                "response_length": 0,
                "contains_nutrition_info": False,
                "readability_score": 0,
                "helpfulness_score": 0
            }
        
        response_lower = response.lower()
        
        # Keyword matching
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
        keyword_coverage = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
        
        # Nutrition information detection
        nutrition_terms = [
            "nutrition", "healthy", "vitamin", "protein", "carbohydrate", "fat", "mineral",
            "calories", "fiber", "antioxidant", "nutrient", "diet", "food", "eat", "meal"
        ]
        contains_nutrition_info = any(term in response_lower for term in nutrition_terms)
        
        # Response length score (optimal range: 100-500 characters)
        length = len(response)
        if 100 <= length <= 500:
            length_score = 1.0
        elif 50 <= length < 100 or 500 < length <= 800:
            length_score = 0.7
        elif length < 50:
            length_score = 0.3
        else:
            length_score = 0.5
        
        # Readability score (simple heuristic)
        sentences = response.count('.') + response.count('!') + response.count('?')
        words = len(response.split())
        avg_sentence_length = words / max(sentences, 1)
        readability_score = 1.0 if 10 <= avg_sentence_length <= 20 else 0.7
        
        # Helpfulness score (based on actionable content)
        helpful_terms = ["suggest", "recommend", "try", "include", "should", "can", "good", "best"]
        helpfulness_score = min(1.0, sum(1 for term in helpful_terms if term in response_lower) / 3)
        
        # Overall quality score
        quality_score = (
            keyword_coverage * 0.4 +
            length_score * 0.2 +
            readability_score * 0.2 +
            helpfulness_score * 0.2
        )
        
        return {
            "quality_score": quality_score,
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "response_length": length,
            "contains_nutrition_info": contains_nutrition_info,
            "readability_score": readability_score,
            "helpfulness_score": helpfulness_score
        }
    
    def evaluate_response_time(self, question: str) -> Tuple[str, float, str]:
        """
        Measure response time and handle errors
        
        Args:
            question: Question to ask
            
        Returns:
            Tuple of (response, response_time, error)
        """
        start_time = time.time()
        try:
            response = self.chatbot.get_response(question)
            end_time = time.time()
            response_time = end_time - start_time
            return response, response_time, None
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return None, response_time, str(e)
    
    def check_topic_handling(self, response: str, should_be_nutrition_related: bool) -> bool:
        """
        Check if chatbot correctly handles on/off topic questions
        
        Args:
            response: Chatbot response
            should_be_nutrition_related: Whether question should be answered
            
        Returns:
            True if topic handled correctly
        """
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Check for rejection phrases
        rejection_phrases = [
            "sorry", "outside", "topic", "nutrition", "can't help", "cannot help",
            "not able", "unable", "outside my scope", "beyond my knowledge"
        ]
        contains_rejection = any(phrase in response_lower for phrase in rejection_phrases)
        
        if should_be_nutrition_related:
            # Nutrition questions should NOT be rejected
            return not contains_rejection
        else:
            # Off-topic questions SHOULD be rejected
            return contains_rejection
    
    def run_single_test(self, test_case: Dict) -> Dict[str, Any]:
        """
        Run a single test case
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Test result dictionary
        """
        logger.info(f"Testing {test_case['id']}: {test_case['question'][:50]}...")
        
        # Get response and measure time
        response, response_time, error = self.evaluate_response_time(test_case["question"])
        
        if error:
            return {
                "test_id": test_case["id"],
                "question": test_case["question"],
                "category": test_case["category"],
                "subcategory": test_case["subcategory"],
                "difficulty": test_case["difficulty"],
                "target_audience": test_case["target_audience"],
                "response": f"ERROR: {error}",
                "response_time": response_time,
                "quality_score": 0,
                "keyword_coverage": 0,
                "matched_keywords": [],
                "is_correct_topic": False,
                "error": True,
                "passed": False
            }
        
        # Evaluate quality
        quality_eval = self.evaluate_response_quality(
            test_case["question"], 
            response, 
            test_case["expected_keywords"]
        )
        
        # Check topic handling
        is_correct_topic = self.check_topic_handling(response, test_case["is_nutrition_related"])
        
        # Determine if test passed (composite criteria)
        passed = (
            quality_eval["quality_score"] >= 0.6 and
            is_correct_topic and
            response_time <= 30.0  # 30 second timeout
        )
        
        return {
            "test_id": test_case["id"],
            "question": test_case["question"],
            "category": test_case["category"],
            "subcategory": test_case["subcategory"],
            "difficulty": test_case["difficulty"],
            "target_audience": test_case["target_audience"],
            "response": response,
            "response_time": response_time,
            "quality_score": quality_eval["quality_score"],
            "keyword_coverage": quality_eval["keyword_coverage"],
            "matched_keywords": quality_eval["matched_keywords"],
            "is_correct_topic": is_correct_topic,
            "contains_nutrition_info": quality_eval["contains_nutrition_info"],
            "response_length": quality_eval["response_length"],
            "readability_score": quality_eval["readability_score"],
            "helpfulness_score": quality_eval["helpfulness_score"],
            "error": False,
            "passed": passed
        }
    
    def run_comprehensive_evaluation(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive evaluation on all test cases
        
        Returns:
            List of test results
        """
        logger.info(f"🚀 Starting comprehensive evaluation with {len(self.test_questions)} test cases")
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"Progress: {i}/{len(self.test_questions)} ({i/len(self.test_questions)*100:.1f}%)")
            
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {total_time:.2f} seconds")
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with calculated metrics
        """
        total_tests = len(results)
        successful_tests = len([r for r in results if not r["error"]])
        passed_tests = len([r for r in results if r["passed"]])
        
        # Basic metrics
        error_rate = (total_tests - successful_tests) / total_tests if total_tests > 0 else 0
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Response time metrics
        response_times = [r["response_time"] for r in results if not r["error"]]
        time_stats = {
            "avg": np.mean(response_times) if response_times else 0,
            "median": np.median(response_times) if response_times else 0,
            "min": np.min(response_times) if response_times else 0,
            "max": np.max(response_times) if response_times else 0,
            "std": np.std(response_times) if response_times else 0
        }
        
        # Quality metrics
        quality_scores = [r["quality_score"] for r in results if not r["error"]]
        quality_stats = {
            "avg": np.mean(quality_scores) if quality_scores else 0,
            "median": np.median(quality_scores) if quality_scores else 0,
            "min": np.min(quality_scores) if quality_scores else 0,
            "max": np.max(quality_scores) if quality_scores else 0,
            "std": np.std(quality_scores) if quality_scores else 0
        }
        
        # Topic handling accuracy
        topic_accuracy = len([r for r in results if r.get("is_correct_topic", False)]) / total_tests
        
        # Category-wise performance
        category_performance = {}
        for category in set(r["category"] for r in results):
            category_results = [r for r in results if r["category"] == category]
            category_performance[category] = {
                "total": len(category_results),
                "passed": len([r for r in category_results if r["passed"]]),
                "pass_rate": len([r for r in category_results if r["passed"]]) / len(category_results) if category_results else 0,
                "avg_quality": np.mean([r["quality_score"] for r in category_results if not r["error"]]) if category_results else 0,
                "avg_response_time": np.mean([r["response_time"] for r in category_results if not r["error"]]) if category_results else 0
            }
        
        # Difficulty-wise performance
        difficulty_performance = {}
        for difficulty in set(r["difficulty"] for r in results):
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            difficulty_performance[difficulty] = {
                "total": len(difficulty_results),
                "passed": len([r for r in difficulty_results if r["passed"]]),
                "pass_rate": len([r for r in difficulty_results if r["passed"]]) / len(difficulty_results) if difficulty_results else 0,
                "avg_quality": np.mean([r["quality_score"] for r in difficulty_results if not r["error"]]) if difficulty_results else 0
            }
        
        # Target audience performance
        audience_performance = {}
        for audience in set(r["target_audience"] for r in results):
            audience_results = [r for r in results if r["target_audience"] == audience]
            audience_performance[audience] = {
                "total": len(audience_results),
                "passed": len([r for r in audience_results if r["passed"]]),
                "pass_rate": len([r for r in audience_results if r["passed"]]) / len(audience_results) if audience_results else 0,
                "avg_quality": np.mean([r["quality_score"] for r in audience_results if not r["error"]]) if audience_results else 0
            }
        
        self.metrics = {
            "overview": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "passed_tests": passed_tests,
                "error_rate": error_rate,
                "pass_rate": pass_rate,
                "topic_accuracy": topic_accuracy
            },
            "response_time": time_stats,
            "quality": quality_stats,
            "category_performance": category_performance,
            "difficulty_performance": difficulty_performance,
            "audience_performance": audience_performance,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.metrics
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("📊 Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Overall Performance Overview (Pie Chart)
        ax1 = plt.subplot(4, 2, 1)
        overview = self.metrics["overview"]
        labels = ['Passed', 'Failed', 'Errors']
        sizes = [
            overview['passed_tests'],
            overview['successful_tests'] - overview['passed_tests'],
            overview['total_tests'] - overview['successful_tests']
        ]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Overall Test Results', fontsize=14, fontweight='bold')
        
        # 2. Response Time Distribution (Histogram)
        ax2 = plt.subplot(4, 2, 2)
        response_times = [r["response_time"] for r in self.results if not r["error"]]
        plt.hist(response_times, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(response_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(response_times):.2f}s')
        plt.legend()
        
        # 3. Quality Score Distribution (Histogram)
        ax3 = plt.subplot(4, 2, 3)
        quality_scores = [r["quality_score"] for r in self.results if not r["error"]]
        plt.hist(quality_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        plt.title('Quality Score Distribution', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(quality_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(quality_scores):.3f}')
        plt.legend()
        
        # 4. Category Performance (Bar Plot)
        ax4 = plt.subplot(4, 2, 4)
        categories = list(self.metrics["category_performance"].keys())
        pass_rates = [self.metrics["category_performance"][cat]["pass_rate"] * 100 for cat in categories]
        bars = plt.bar(categories, pass_rates, color='lightcoral', alpha=0.7)
        plt.xlabel('Category')
        plt.ylabel('Pass Rate (%)')
        plt.title('Pass Rate by Category', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 5. Difficulty Performance (Bar Plot)
        ax5 = plt.subplot(4, 2, 5)
        difficulties = list(self.metrics["difficulty_performance"].keys())
        diff_pass_rates = [self.metrics["difficulty_performance"][diff]["pass_rate"] * 100 for diff in difficulties]
        bars = plt.bar(difficulties, diff_pass_rates, color='gold', alpha=0.7)
        plt.xlabel('Difficulty Level')
        plt.ylabel('Pass Rate (%)')
        plt.title('Pass Rate by Difficulty', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, rate in zip(bars, diff_pass_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 6. Target Audience Performance (Bar Plot)
        ax6 = plt.subplot(4, 2, 6)
        audiences = list(self.metrics["audience_performance"].keys())
        aud_pass_rates = [self.metrics["audience_performance"][aud]["pass_rate"] * 100 for aud in audiences]
        bars = plt.bar(audiences, aud_pass_rates, color='mediumpurple', alpha=0.7)
        plt.xlabel('Target Audience')
        plt.ylabel('Pass Rate (%)')
        plt.title('Pass Rate by Target Audience', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rate in zip(bars, aud_pass_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 7. Quality vs Response Time Scatter
        ax7 = plt.subplot(4, 2, 7)
        response_times = [r["response_time"] for r in self.results if not r["error"]]
        quality_scores = [r["quality_score"] for r in self.results if not r["error"]]
        categories = [r["category"] for r in self.results if not r["error"]]
        
        # Color by category
        unique_cats = list(set(categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
        for i, cat in enumerate(unique_cats):
            cat_times = [rt for rt, c in zip(response_times, categories) if c == cat]
            cat_qualities = [qs for qs, c in zip(quality_scores, categories) if c == cat]
            plt.scatter(cat_times, cat_qualities, c=[colors[i]], label=cat, alpha=0.7)
        
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Quality Score')
        plt.title('Quality vs Response Time', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 8. Category Quality Heatmap
        ax8 = plt.subplot(4, 2, 8)
        categories = list(self.metrics["category_performance"].keys())
        metrics_names = ['Pass Rate', 'Avg Quality', 'Avg Response Time (norm)']
        
        heatmap_data = []
        for cat in categories:
            perf = self.metrics["category_performance"][cat]
            # Normalize response time to 0-1 scale
            norm_time = 1 - min(perf["avg_response_time"] / 10, 1)  # Assume 10s is max acceptable
            heatmap_data.append([
                perf["pass_rate"],
                perf["avg_quality"],
                norm_time
            ])
        
        heatmap_df = pd.DataFrame(heatmap_data, index=categories, columns=metrics_names)
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
        plt.title('Category Performance Metrics', fontsize=14, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = self.output_dir / "evaluation_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualizations saved to {output_path}")
        
        # 9. Box Plot for Quality Scores by Category
        plt.figure(figsize=(12, 6))
        quality_data = [r["quality_score"] for r in self.results if not r["error"]]
        categories = [r["category"] for r in self.results if not r["error"]]
        sns.boxplot(x=categories, y=quality_data, palette="husl")
        plt.xlabel('Category')
        plt.ylabel('Quality Score')
        plt.title('Quality Score Distribution by Category', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        boxplot_path = self.output_dir / "quality_boxplot.png"
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Box plot saved to {boxplot_path}")
    
    def save_results(self):
        """Save evaluation results and metrics to files"""
        # Save detailed results to CSV
        results_df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")
        
        # Save metrics to JSON
        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate summary report
        summary_path = self.output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("NuZiBot Evaluation Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {self.metrics['timestamp']}\n")
            f.write(f"Total Tests: {self.metrics['overview']['total_tests']}\n")
            f.write(f"Successful Tests: {self.metrics['overview']['successful_tests']}\n")
            f.write(f"Passed Tests: {self.metrics['overview']['passed_tests']}\n")
            f.write(f"Pass Rate: {self.metrics['overview']['pass_rate']:.2%}\n")
            f.write(f"Error Rate: {self.metrics['overview']['error_rate']:.2%}\n")
            f.write(f"Topic Accuracy: {self.metrics['overview']['topic_accuracy']:.2%}\n")
            f.write("\nResponse Time Metrics:\n")
            f.write(f"  Average: {self.metrics['response_time']['avg']:.2f}s\n")
            f.write(f"  Median: {self.metrics['response_time']['median']:.2f}s\n")
            f.write(f"  Min: {self.metrics['response_time']['min']:.2f}s\n")
            f.write(f"  Max: {self.metrics['response_time']['max']:.2f}s\n")
            f.write("\nQuality Metrics:\n")
            f.write(f"  Average: {self.metrics['quality']['avg']:.3f}\n")
            f.write(f"  Median: {self.metrics['quality']['median']:.3f}\n")
            f.write(f"  Min: {self.metrics['quality']['min']:.3f}\n")
            f.write(f"  Max: {self.metrics['quality']['max']:.3f}\n")
            f.write("\nCategory Performance:\n")
            for cat, perf in self.metrics["category_performance"].items():
                f.write(f"  {cat}:\n")
                f.write(f"    Pass Rate: {perf['pass_rate']:.2%}\n")
                f.write(f"    Avg Quality: {perf['avg_quality']:.3f}\n")
                f.write(f"    Avg Response Time: {perf['avg_response_time']:.2f}s\n")
        logger.info(f"Summary report saved to {summary_path}")
    
    def run_evaluation(self):
        """Run full evaluation pipeline"""
        try:
            # Run tests
            results = self.run_comprehensive_evaluation()
            
            # Calculate metrics
            metrics = self.calculate_metrics(results)
            
            # Generate visualizations
            self.generate_visualizations()
            
            # Save results
            self.save_results()
            
            # Print summary
            print("\n=== Evaluation Summary ===")
            print(f"Total Tests: {metrics['overview']['total_tests']}")
            print(f"Passed Tests: {metrics['overview']['passed_tests']}")
            print(f"Pass Rate: {metrics['overview']['pass_rate']:.2%}")
            print(f"Error Rate: {metrics['overview']['error_rate']:.2%}")
            print(f"Topic Accuracy: {metrics['overview']['topic_accuracy']:.2%}")
            print(f"Average Response Time: {metrics['response_time']['avg']:.2f}s")
            print(f"Average Quality Score: {metrics['quality']['avg']:.3f}")
            print(f"Results saved in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the evaluation"""
    try:
        # Check file existence
        food_csv_path = "merged_food_with_ingredients.csv"
        nutrition_excel_path = "Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx"
        
        if not Path(food_csv_path).exists():
            raise FileNotFoundError(f"Food CSV file not found: {food_csv_path}")
        if not Path(nutrition_excel_path).exists():
            raise FileNotFoundError(f"Nutrition Excel file not found: {nutrition_excel_path}")
        
        # Initialize evaluator
        evaluator = ChatbotEvaluator(
            food_csv_path=food_csv_path,
            nutrition_excel_path=nutrition_excel_path,
            llm_type="groq"
        )
        
        # Run evaluation
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()