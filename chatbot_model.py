import ast
from dotenv import load_dotenv
from tenacity import retry
load_dotenv()
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
import json
import os
from dataclasses import dataclass
import random
from datetime import datetime
import requests
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from langdetect import detect  # Impor fungsi detect

@dataclass
class UserProfile:
    weight: float
    height: float
    age: int
    gender: str
    bmi: float
    bmr: float
    target_energy_intake: float
    activity_level: str
    special_conditions: List[str]
    dietary_preferences: str

class NutritionDatabase:
    def __init__(self, food_csv_path: str = None, nutrition_excel_path: str = None):
        """
        Initialize database with custom dataset paths
        
        Args:
            food_csv_path: Path to 'merged_food_with_ingredients.csv'
            nutrition_excel_path: Path to 'Recommended Dietary Allowances...' Excel file
        """
        self.food_csv_path = food_csv_path
        self.nutrition_excel_path = nutrition_excel_path
        
        self.food_data = self._load_food_data()
        self.nutrition_requirements = self._load_nutrition_requirements()
    
    def _load_food_data(self) -> pd.DataFrame:
        """Load food data from CSV file or create sample data"""
        if self.food_csv_path and os.path.exists(self.food_csv_path):
            try:
                print(f"Loading food data from: {self.food_csv_path}")
                df = pd.read_csv(self.food_csv_path)
                
                # Validate required columns
                required_columns = ['id', 'cuisine', 'ingredients', 'matched_food', 'calories', 'proteins', 'fat', 'carbohydrate']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"Warning: Missing columns in CSV: {missing_columns}")
                    print(f"Available columns: {list(df.columns)}")
                
                # Add image column if not exists
                if 'image' not in df.columns:
                    df['image'] = 'https://via.placeholder.com/300x200?text=Food+Image'
                
                # Clean data
                df = df.dropna(subset=['matched_food', 'calories'])
                
                # Ensure numeric columns are numeric
                numeric_columns = ['calories', 'proteins', 'fat', 'carbohydrate']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                print(f"Successfully loaded {len(df)} food items from CSV")
                return df
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                print("Using sample data instead...")
                return self._create_sample_food_data()
        else:
            if self.food_csv_path:
                print(f"CSV file not found at: {self.food_csv_path}")
                print("Using sample data instead...")
            return self._create_sample_food_data()
    
    def _create_sample_food_data(self) -> pd.DataFrame:
        """Create comprehensive sample food data"""
        sample_data = [
            {
                'id': 25693,
                'cuisine': 'indonesian',
                'ingredients': "['nasi', 'ayam', 'sayuran', 'minyak goreng', 'bumbu rempah']",
                'matched_food': 'Nasi Ayam Sayur',
                'calories': 450.0,
                'proteins': 25.0,
                'fat': 15.0,
                'carbohydrate': 55.0,
                'image': 'https://via.placeholder.com/300x200?text=Nasi+Ayam'
            },
            {
                'id': 25694, 
                'cuisine': 'indonesian',
                'ingredients': "['ikan', 'nasi', 'sayuran hijau', 'tomat', 'bumbu bakar']",
                'matched_food': 'Ikan Bakar dengan Nasi',
                'calories': 380.0,
                'proteins': 30.0,
                'fat': 12.0,
                'carbohydrate': 45.0,
                'image': 'https://via.placeholder.com/300x200?text=Ikan+Bakar'
            },
            {
                'id': 25695,
                'cuisine': 'indonesian', 
                'ingredients': "['tempe', 'tahu', 'sayuran', 'nasi', 'kecap']",
                'matched_food': 'Tempe Tahu Sayur',
                'calories': 320.0,
                'proteins': 18.0,
                'fat': 10.0,
                'carbohydrate': 42.0,
                'image': 'https://via.placeholder.com/300x200?text=Tempe+Tahu'
            },
            {
                'id': 25696,
                'cuisine': 'indonesian',
                'ingredients': "['telur', 'nasi', 'sayuran', 'minyak', 'bawang']", 
                'matched_food': 'Telur Dadar Sayur',
                'calories': 350.0,
                'proteins': 15.0,
                'fat': 18.0,
                'carbohydrate': 35.0,
                'image': 'https://via.placeholder.com/300x200?text=Telur+Dadar'
            },
            {
                'id': 25697,
                'cuisine': 'international',
                'ingredients': "['buah-buahan', 'yogurt', 'madu', 'granola']",
                'matched_food': 'Fruit Yogurt Bowl',
                'calories': 180.0,
                'proteins': 8.0,
                'fat': 5.0,
                'carbohydrate': 28.0,
                'image': 'https://via.placeholder.com/300x200?text=Fruit+Bowl'
            },
            {
                'id': 25698,
                'cuisine': 'international',
                'ingredients': "['oatmeal', 'banana', 'almond', 'susu', 'madu']",
                'matched_food': 'Oatmeal Banana Almond',
                'calories': 320.0,
                'proteins': 12.0,
                'fat': 8.0,
                'carbohydrate': 54.0,
                'image': 'https://via.placeholder.com/300x200?text=Oatmeal'
            },
            {
                'id': 25699,
                'cuisine': 'international',
                'ingredients': "['telur', 'roti gandum', 'alpukat', 'tomat']",
                'matched_food': 'Scrambled Eggs Avocado Toast',
                'calories': 290.0,
                'proteins': 18.0,
                'fat': 14.0,
                'carbohydrate': 24.0,
                'image': 'https://via.placeholder.com/300x200?text=Eggs+Toast'
            },
            {
                'id': 25700,
                'cuisine': 'indonesian',
                'ingredients': "['kacang tanah', 'garam', 'cabai']",
                'matched_food': 'Kacang Tanah Rebus',
                'calories': 150.0,
                'proteins': 7.0,
                'fat': 12.0,
                'carbohydrate': 6.0,
                'image': 'https://via.placeholder.com/300x200?text=Kacang+Tanah'
            },
            {
                'id': 25701,
                'cuisine': 'indonesian',
                'ingredients': "['nasi', 'rendang', 'sayuran', 'sambal']",
                'matched_food': 'Nasi Rendang',
                'calories': 520.0,
                'proteins': 28.0,
                'fat': 22.0,
                'carbohydrate': 58.0,
                'image': 'https://via.placeholder.com/300x200?text=Nasi+Rendang'
            },
            {
                'id': 25702,
                'cuisine': 'indonesian',
                'ingredients': "['gado-gado', 'sayuran', 'tahu', 'tempe', 'bumbu kacang']",
                'matched_food': 'Gado-Gado Lengkap',
                'calories': 380.0,
                'proteins': 16.0,
                'fat': 18.0,
                'carbohydrate': 42.0,
                'image': 'https://via.placeholder.com/300x200?text=Gado+Gado'
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def _load_nutrition_requirements(self) -> pd.DataFrame:
        """Load nutrition requirements from Excel file or create sample data"""
        if self.nutrition_excel_path and os.path.exists(self.nutrition_excel_path):
            try:
                print(f"Loading nutrition requirements from: {self.nutrition_excel_path}")
                
                xl_file = pd.ExcelFile(self.nutrition_excel_path)
                print(f"Available sheets: {xl_file.sheet_names}")
                
                df = pd.read_excel(self.nutrition_excel_path, sheet_name=0)
                
                print(f"Excel columns: {list(df.columns)}")
                print(f"Successfully loaded {len(df)} nutrition requirement records")
                
                numeric_columns = ['Total Water (L/d)', 'Carbohydrate (g/d)', 'Total Fiber (g/d)', 
                                  'Fat (g/d)', 'Protein (g/d)']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
            except Exception as e:
                print(f"Error loading Excel file: {e}")
                print("Using sample nutrition data instead...")
                return self._create_sample_nutrition_data()
        else:
            if self.nutrition_excel_path:
                print(f"Excel file not found at: {self.nutrition_excel_path}")
                print("Using sample nutrition data instead...")
            return self._create_sample_nutrition_data()
    
    def _create_sample_nutrition_data(self) -> pd.DataFrame:
        """Create sample nutrition requirements data"""
        data = [
            {'Life Stage Group': 'Children', 'Age Group': '1–3 y', 'Total Water (L/d)': 1.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 19, 'Fat (g/d)': 30, 'Protein (g/d)': 13},
            {'Life Stage Group': 'Children', 'Age Group': '4–8 y', 'Total Water (L/d)': 1.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 35, 'Protein (g/d)': 19},
            {'Life Stage Group': 'Males', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.4, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 31, 'Fat (g/d)': 40, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Males', 'Age Group': '14–18 y', 'Total Water (L/d)': 3.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 55, 'Protein (g/d)': 52},
            {'Life Stage Group': 'Males', 'Age Group': '19–30 y', 'Total Water (L/d)': 3.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 65, 'Protein (g/d)': 56},
            {'Life Stage Group': 'Males', 'Age Group': '31–50 y', 'Total Water (L/d)': 3.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 65, 'Protein (g/d)': 56},
            {'Life Stage Group': 'Females', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.1, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 35, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Females', 'Age Group': '14–18 y', 'Total Water (L/d)': 2.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 45, 'Protein (g/d)': 46},
            {'Life Stage Group': 'Females', 'Age Group': '19–30 y', 'Total Water (L/d)': 2.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 55, 'Protein (g/d)': 46},
            {'Life Stage Group': 'Females', 'Age Group': '31–50 y', 'Total Water (L/d)': 2.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 55, 'Protein (g/d)': 46},
        ]
        
        df = pd.DataFrame(data)
        numeric_columns = ['Total Water (L/d)', 'Carbohydrate (g/d)', 'Total Fiber (g/d)', 
                          'Fat (g/d)', 'Protein (g/d)']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_nutrition_requirements(self, age: int, gender: str) -> Dict[str, float]:
        """Get nutrition requirements based on age and gender"""
        if age <= 3:
            age_group = '1–3 y'
            life_stage = 'Children'
        elif age <= 8:
            age_group = '4–8 y'
            life_stage = 'Children'
        elif age <= 13:
            age_group = '9–13 y'
            life_stage = gender.title() + 's'
        elif age <= 18:
            age_group = '14–18 y'
            life_stage = gender.title() + 's'
        elif age <= 30:
            age_group = '19–30 y'
            life_stage = gender.title() + 's'
        else:
            age_group = '31–50 y'
            life_stage = gender.title() + 's'
        
        mask = (self.nutrition_requirements['Age Group'] == age_group) & \
               (self.nutrition_requirements['Life Stage Group'] == life_stage)
        
        if not mask.any():
            return {
                'Total Water (L/d)': 2.5,
                'Carbohydrate (g/d)': 130.0,
                'Total Fiber (g/d)': 25.0,
                'Fat (g/d)': 65.0,
                'Protein (g/d)': 50.0
            }
        
        req = self.nutrition_requirements[mask].iloc[0]
        return {
            'Total Water (L/d)': float(req['Total Water (L/d)']),
            'Carbohydrate (g/d)': float(req['Carbohydrate (g/d)']),
            'Total Fiber (g/d)': float(req['Total Fiber (g/d)']),
            'Fat (g/d)': max(float(req['Fat (g/d)']), 20.0),
            'Protein (g/d)': float(req['Protein (g/d)'])
        }
    
    def search_foods(self, query: str, limit: int = 5) -> List[Dict]:
        """Search foods based on query"""
        query_lower = query.lower()
        
        filtered_foods = []
        for _, food in self.food_data.iterrows():
            try:
                ingredients = ast.literal_eval(food['ingredients'])
                if (query_lower in food['matched_food'].lower() or 
                    query_lower in food['ingredients'].lower() or
                    any(query_lower in ingredient.lower() for ingredient in ingredients)):
                    filtered_foods.append(food.to_dict())
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse ingredients for {food['matched_food']}")
                continue
        
        return filtered_foods[:limit]

class DetailedNutritionKnowledge:
    """Comprehensive nutrition knowledge base for detailed responses"""
    
    @staticmethod
    def get_bmi_interpretation(bmi: float) -> Dict[str, str]:
        """Get detailed BMI interpretation"""
        if bmi < 18.5:
            category = "Underweight"
            description = "BMI di bawah normal. Disarankan untuk meningkatkan asupan kalori dengan makanan bergizi."
            recommendations = [
                "Tingkatkan porsi makan secara bertahap",
                "Konsumsi makanan padat nutrisi seperti kacang-kacangan, alpukat",
                "Makan lebih sering dengan porsi kecil (5-6 kali sehari)",
                "Konsultasi dengan ahli gizi untuk program penambahan berat badan"
            ]
        elif 18.5 <= bmi < 25:
            category = "Normal"
            description = "BMI dalam rentang normal. Pertahankan pola makan sehat dan aktivitas fisik."
            recommendations = [
                "Pertahankan pola makan seimbang",
                "Lakukan olahraga teratur 150 menit/minggu",
                "Konsumsi buah dan sayur minimal 5 porsi/hari",
                "Minum air putih minimal 8 gelas/hari"
            ]
        elif 25 <= bmi < 30:
            category = "Overweight"
            description = "BMI di atas normal. Disarankan menurunkan berat badan untuk kesehatan optimal."
            recommendations = [
                "Kurangi asupan kalori 300-500 kcal/hari",
                "Tingkatkan aktivitas fisik menjadi 300 menit/minggu",
                "Batasi makanan olahan dan tinggi gula",
                "Fokus pada makanan tinggi serat dan protein"
            ]
        else:
            category = "Obese"
            description = "BMI menunjukkan obesitas. Perlu program penurunan berat badan yang terstruktur."
            recommendations = [
                "Konsultasi dengan dokter atau ahli gizi",
                "Buat rencana penurunan berat badan bertahap (0.5-1 kg/minggu)",
                "Gabungkan diet rendah kalori dengan olahraga",
                "Pertimbangkan dukungan kelompok atau program khusus"
            ]
        
        return {
            "category": category,
            "description": description,
            "recommendations": recommendations
        }
    
    @staticmethod
    def get_nutrient_functions() -> Dict[str, Dict[str, str]]:
        """Get detailed information about nutrient functions"""
        return {
            "protein": {
                "function": "Membangun dan memperbaiki jaringan tubuh, produksi enzim dan hormon",
                "sources": "Daging, ikan, telur, kacang-kacangan, tahu, tempe",
                "deficiency": "Kehilangan massa otot, penyembuhan lambat, sistem imun lemah",
                "excess": "Beban berlebih pada ginjal, dehidrasi"
            },
            "carbohydrate": {
                "function": "Sumber energi utama untuk otak dan otot",
                "sources": "Nasi, roti, pasta, buah-buahan, sayuran bertepung",
                "deficiency": "Kelelahan, kesulitan konsentrasi, hipoglikemia",
                "excess": "Penambahan berat badan, diabetes tipe 2"
            },
            "fat": {
                "function": "Penyimpanan energi, penyerapan vitamin larut lemak, produksi hormon",
                "sources": "Minyak, kacang-kacangan, alpukat, ikan berlemak",
                "deficiency": "Kulit kering, gangguan hormon, defisiensi vitamin A,D,E,K",
                "excess": "Obesitas, penyakit jantung, kolesterol tinggi"
            },
            "fiber": {
                "function": "Melancarkan pencernaan, mengontrol gula darah, menurunkan kolesterol",
                "sources": "Sayuran, buah-buahan, kacang-kacangan, biji-bijian utuh",
                "deficiency": "Sembelit, kolesterol tinggi, gula darah tidak stabil",
                "excess": "Kembung, gangguan penyerapan mineral"
            },
            "water": {
                "function": "Mengatur suhu tubuh, transportasi nutrisi, detoksifikasi",
                "sources": "Air putih, buah-buahan, sayuran, sup",
                "deficiency": "Dehidrasi, kelelahan, batu ginjal, sembelit",
                "excess": "Hiponatremia (jarang terjadi)"
            }
        }
        
from retrying import retry

class FastLLMProcessor:
    def __init__(self, llm_type: str = "groq"):
        """
        Initialize with LLM type: 'groq' only
        
        Args:
            llm_type: Must be 'groq' for Groq API
        """
        self.llm_type = llm_type
        self.api_key = None
        
        if llm_type != "groq":
            raise ValueError("Only 'groq' LLM type is supported.")
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_name = "llama3-8b-8192"
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please set the environment variable.")
        
        print(f"Initialized {self.llm_type} LLM processor")
    
    def _make_api_request(self, data: Dict, headers: Dict = None, timeout: int = 15) -> Optional[Dict]:
        """Generic API request handler with retry logic"""
        @retry(stop_max_attempt_number=3, wait_fixed=1000, wait_exponential_multiplier=2000)
        def request_with_retry():
            try:
                response = requests.post(self.base_url, headers=headers, json=data, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise Exception(f"API request failed: {str(e)}")
        
        try:
            return request_with_retry()
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return None

    def generate_response_groq(self, prompt: str, context: str = "") -> str:
        """Generate response using Groq API with language detection"""
        if not self.api_key:
            return "Error: GROQ_API_KEY not found. Please contact the administrator."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Deteksi bahasa input dengan error handling yang lebih baik
        language = "en"  # Default
        try:
            language = detect(prompt)
        except Exception as e:
            print(f"Language detection failed: {e}")
        
        # Sesuaikan bahasa prompt sistem
        language_instruction = "Jawab dalam bahasa Indonesia dengan nada ramah dan sesuai untuk anak-anak/remaja." if language == "id" else "Respond in English with a friendly tone suitable for children/teens."

        system_prompt = f"""You are NuZiBot, an expert nutrition consultant for children and adolescents, supporting SDG 2 (Zero Hunger) and SDG 3 (Good Health and Well-Being). Your purpose is to provide educational and practical nutritional advice tailored to the needs of young users (ages 1-18) in Indonesia and globally. ONLY respond to questions related to nutrition, diet, healthy eating, or health conditions (e.g., allergies, diabetes, hypertension, veganism). For any unrelated questions (e.g., geography, mechanics, or other non-nutrition topics), politely redirect the user to ask about nutrition or diet, and provide a relevant suggestion based on their profile (age, gender, dietary needs).

        {language_instruction}

        If no specific user profile (age, gender, activity level, dietary preferences) is provided in the context, assume a default profile of a 15-year-old, moderately active, with no special dietary restrictions, and provide general advice suitable for this profile.

        Your responses must include:
        - Specific nutritional values (calories, protein, fat, carbs)
        - Practical meal suggestions with exact portions, suitable for children/adolescents
        - Health benefits and risks, explained in simple terms
        - Alternative options for dietary needs (e.g., vegan, allergies, lactose intolerance)
        - Cultural context for Indonesian foods (e.g., nasi uduk, gado-gado)
        - Step-by-step guidance or tips for healthy eating
        Use clear sections, emojis (e.g., 🥗, 🍽️), and actionable advice written in an engaging tone suitable for young users."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nUser question: {prompt}\n\nProvide a detailed, comprehensive response related to nutrition or redirect to a nutrition topic if the question is unrelated."}
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": False
        }
        
        response = self._make_api_request(data, headers)
        if response and "choices" in response and response["choices"]:
            try:
                response_text = response["choices"][0]["message"]["content"].strip()
                return response_text
            except (KeyError, IndexError) as e:
                print(f"Error parsing response: {e}")
                return f"{'Maaf, saya tidak bisa memproses permintaan Anda saat ini.' if language == 'id' else 'Sorry, I couldn\'t process your request right now.'}"
        return f"{'Maaf, saya tidak bisa memproses permintaan Anda saat ini. Silakan coba lagi nanti.' if language == 'id' else 'Sorry, I couldn\'t process your request right now. Please try again later.'}"

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Main method to generate response using Groq"""
        start_time = time.time()
        
        try:
            response = self.generate_response_groq(prompt, context)
            elapsed_time = time.time() - start_time
            print(f"Response generated in {elapsed_time:.2f} seconds using {self.llm_type}")
            return response
        except Exception as e:
            print(f"Error generating response with {self.llm_type}: {str(e)}")
            try:
                lang = detect(prompt)
                error_msg = 'Maaf, terjadi kesalahan saat memproses permintaan Anda.' if lang == 'id' else 'Sorry, an error occurred while processing your request.'
            except:
                error_msg = 'Sorry, an error occurred while processing your request.'
            return error_msg

class NutritionChatbot:
    def __init__(self, food_csv_path: str = None, nutrition_excel_path: str = None, llm_type: str = "groq"):
        """
        Initialize chatbot with specified LLM type and data paths.
        
        Args:
            food_csv_path: Path to food CSV file
            nutrition_excel_path: Path to nutrition Excel file
            llm_type: 'groq' or 'rule_based'
        """
        try:
            self.database = NutritionDatabase(food_csv_path, nutrition_excel_path)
            self.llm_processor = FastLLMProcessor(llm_type=llm_type)
            self.user_profile: Optional[UserProfile] = None
            self.conversation_history: List[tuple] = []
            print(f"Nutrition Chatbot initialized with {llm_type} LLM")
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            raise

    def update_user_profile(self, profile_data: Dict[str, Any]) -> None:
        """Update user profile with validation"""
        required_fields = ['weight', 'height', 'age', 'gender', 'bmi', 'bmr', 
                          'target_energy_intake', 'activity_level', 'special_conditions', 'dietary_preferences']
        
        # Validate required fields
        for field in required_fields:
            if field not in profile_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert and validate numeric fields
        try:
            weight = float(profile_data['weight'])
            height = float(profile_data['height'])
            age = int(profile_data['age'])
            bmi = float(profile_data['bmi'])
            bmr = float(profile_data['bmr'])
            target_energy = float(profile_data['target_energy_intake'])
        except (ValueError, TypeError):
            raise ValueError("Numeric fields must be valid numbers")

        if weight <= 0:
            raise ValueError("Weight must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")
        if age < 0:
            raise ValueError("Age cannot be negative")
        if profile_data['gender'].lower() not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        if bmi <= 0:
            raise ValueError("BMI must be positive")
        if bmr <= 0:
            raise ValueError("BMR must be positive")
        if target_energy <= 0:
            raise ValueError("Target energy intake must be positive")
        
        # Create user profile
        self.user_profile = UserProfile(
            weight=weight,
            height=height,
            age=age,
            gender=profile_data['gender'],
            bmi=bmi,
            bmr=bmr,
            target_energy_intake=target_energy,
            activity_level=profile_data['activity_level'],
            special_conditions=profile_data['special_conditions'] if profile_data['special_conditions'] else [],
            dietary_preferences=profile_data['dietary_preferences']
        )
        print("User profile updated successfully")
    
    def get_nutrition_recommendations(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Get nutrition recommendations based on profile"""
        return self.database.get_nutrition_requirements(profile['age'], profile['gender'])
    
    def _create_context(self) -> str:
        """Create context string from user profile"""
        if not self.user_profile:
            return "Context: User has not completed their profile."
        
        context = f"""User Profile Context:
        - Age: {self.user_profile.age} years
        - Gender: {self.user_profile.gender}
        - Weight: {self.user_profile.weight} kg
        - Height: {self.user_profile.height} cm
        - BMI: {self.user_profile.bmi:.1f}
        - Activity Level: {self.user_profile.activity_level}
        - Caloric Needs: {self.user_profile.target_energy_intake:.0f} kcal/day
        - Special Conditions: {', '.join(self.user_profile.special_conditions) if self.user_profile.special_conditions else 'None'}
        - Dietary Preferences: {self.user_profile.dietary_preferences if self.user_profile.dietary_preferences else 'None'}"""
        
        return context
    
    def get_response(self, user_input: str) -> str:
        """Get chatbot response to user input"""
        try:
            # Validate input
            if not user_input or not user_input.strip():
                return "Please provide a valid question about nutrition."
            
            # Add to conversation history
            self.conversation_history.append(("user", user_input))
            
            # Create context
            context = self._create_context()
            
            # Generate response
            response = self.llm_processor.generate_response(user_input, context)
            
            # Add response to history
            self.conversation_history.append(("bot", response))
            
            # Limit conversation history
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            return "Sorry, I encountered an error while processing your request. Please try again."
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
    
    def get_conversation_history(self) -> List[tuple]:
        """Get current conversation history"""
        return self.conversation_history.copy()

if __name__ == "__main__":
    print("=== Fast Nutrition Chatbot Demo ===")
    print("\nAvailable LLM options:")
    print("1. 'groq' - Fast API (requires GROQ_API_KEY)")
    print("2. 'rule_based' - Fastest, no API needed")
    
    chatbot = NutritionChatbot(
        food_csv_path="merged_food_with_ingredients.csv",
        nutrition_excel_path="Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx",
        llm_type="groq"
    )
    
    sample_profile = {
        'weight': 55.0,
        'height': 165.0,
        'age': 15,
        'gender': 'Female',
        'bmi': 20.2,
        'bmr': 1350,
        'target_energy_intake': 2000,
        'activity_level': 'Moderately Active',
        'special_conditions': [],
        'dietary_preferences': ''
    }
    
    chatbot.update_user_profile(sample_profile)
    
    test_queries = [
        "What is a suitable breakfast menu for me?",
        "What is my daily protein requirement?", 
        "I am allergic to nuts, what foods are safe?",
        "Tips to gain weight healthily?",
        "How many calories should I eat per day?",
        "Saya sudah makan nasi kuning dan makan es krim sore tadi, apakah tidak apa apa?",
        "How to fix a bicycle?"  # Test off-topic query
    ]
    
    print("\n=== Testing Responses ===")
    for query in test_queries:
        print(f"\n👤 User: {query}")
        start_time = time.time()
        response = chatbot.get_response(query)
        elapsed_time = time.time() - start_time
        print(f"🤖 NuZiBot ({elapsed_time:.2f}s): {response}")
        print("-" * 70)
    
    print(f"\n📊 Performance Summary:")
    print(f"Using {chatbot.llm_processor.llm_type} LLM")
    print(f"\n🚀 To use Groq, get a free key from https://console.groq.com/")