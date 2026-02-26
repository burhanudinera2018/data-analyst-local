import requests
import json
import time
from typing import Optional, Dict, Any, List

class LocalLLM:
    """
    Kelas untuk berinteraksi dengan Local LLM (Ollama)
    dengan berbagai optimasi dan fitur tambahan
    """
    
    def __init__(self, 
                 model: str = "gemma2:2b",  # Ganti dengan model yang Anda punya
                 base_url: str = "http://localhost:11434",
                 timeout: int = 60):
        """
        Inisialisasi koneksi ke Ollama
        
        Args:
            model: Nama model yang digunakan (mistral, gemma2:2b, llama2, dll)
            base_url: URL dasar Ollama API
            timeout: Timeout untuk request (detik)
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = timeout
        
        # System prompt default untuk data analyst
        self.default_system_prompt = """
        You are an expert data analyst assistant for an e-commerce company.
        
        DATABASE SCHEMA:
        - users (id, first_name, last_name, email, created_at)
        - products (id, name, category, brand, price, cost)
        - order_items (id, order_id, user_id, product_id, quantity, sale_price, status, created_at)
        
        GUIDELINES:
        1. Provide accurate, concise answers based on the data
        2. When generating SQL, ensure it's valid PostgreSQL syntax
        3. Explain complex concepts in simple terms
        4. If user asks in Indonesian, respond in Indonesian
        5. Always include business insights when relevant
        6. If you're unsure about something, be honest about limitations
        
        For SQL generation:
        - Use appropriate JOINs, GROUP BY, and aggregations
        - Include error handling where relevant
        - Optimize queries for performance
        """
        
        # Cache untuk menyimpan responses (jika digunakan)
        self.response_cache = {}
    
    def ask(self, 
            prompt: str, 
            system_prompt: Optional[str] = None, 
            temperature: float = 0.3,
            max_tokens: int = 2000,
            context: Optional[str] = None,
            use_cache: bool = False) -> str:
        """
        Kirim prompt ke local LLM dengan berbagai opsi
        
        Args:
            prompt: Pertanyaan atau perintah user
            system_prompt: Instruksi sistem (opsional)
            temperature: Kreativitas respons (0.0 - 1.0)
            max_tokens: Maksimum token yang dihasilkan
            context: Konteks tambahan (misal: ringkasan data)
            use_cache: Gunakan cache untuk menghindari request berulang
            
        Returns:
            String respons dari LLM
        """
        
        # Buat full prompt dengan konteks
        full_prompt = ""
        if system_prompt:
            full_prompt += system_prompt + "\n\n"
        else:
            full_prompt += self.default_system_prompt + "\n\n"
        
        if context:
            full_prompt += f"CONTEXT:\n{context}\n\n"
        
        full_prompt += f"USER QUERY:\n{prompt}\n\n"
        full_prompt += "RESPONSE:"
        
        # Cek cache jika diaktifkan
        if use_cache:
            cache_key = f"{self.model}_{full_prompt}_{temperature}"
            if cache_key in self.response_cache:
                print("📦 Using cached response")
                return self.response_cache[cache_key]
        
        # Prepare payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "stop": ["USER:", "CONTEXT:", "HUMAN:", "AI:"]
            }
        }
        
        # Kirim request
        try:
            start_time = time.time()
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()['response']
                
                # Bersihkan respons dari kemungkinan artifacts
                result = self._clean_response(result)
                
                # Simpan ke cache jika diaktifkan
                if use_cache:
                    self.response_cache[cache_key] = result
                
                print(f"✅ LLM response received in {elapsed:.2f}s")
                return result
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return f"Maaf, terjadi kesalahan pada server LLM. {error_msg}"
                
        except requests.exceptions.Timeout:
            return f"⏱️ Timeout: LLM tidak merespon dalam {self.timeout} detik. Coba lagi nanti."
        except requests.exceptions.ConnectionError:
            return "🔌 Gagal terhubung ke Ollama. Pastikan Ollama sudah berjalan dengan 'ollama serve'"
        except Exception as e:
            return f"❌ Error tidak terduga: {str(e)}"
    
    def ask_with_context(self, 
                        prompt: str, 
                        data_context: Dict[str, Any],
                        temperature: float = 0.3) -> str:
        """
        Tanya LLM dengan konteks data spesifik
        
        Args:
            prompt: Pertanyaan user
            data_context: Dictionary berisi konteks data (ringkasan statistik)
            temperature: Kreativitas respons
        """
        # Format konteks data menjadi string
        context_str = "Current Data Summary:\n"
        for key, value in data_context.items():
            context_str += f"- {key}: {value}\n"
        
        return self.ask(prompt, context=context_str, temperature=temperature)
    
    def explain_sql(self, 
                   sql_query: str, 
                   detailed: bool = False,
                   language: str = "english") -> str:
        """
        Jelaskan SQL query dalam bahasa sederhana
        
        Args:
            sql_query: Query SQL yang akan dijelaskan
            detailed: True untuk penjelasan detail, False untuk ringkas
            language: 'english' atau 'indonesian'
        """
        system = """You are a SQL expert teacher. Explain SQL queries in a way that 
        even non-technical stakeholders can understand."""
        
        detail_level = "detailed step-by-step" if detailed else "simple"
        lang_instruction = "Respond in Indonesian." if language == "indonesian" else "Respond in English."
        
        prompt = f"""
        {lang_instruction}
        
        Explain this SQL query in {detail_level} terms:
        
        ```sql
        {sql_query}
        ```
        
        Include:
        1. What the query does (overall purpose)
        2. Which tables are involved
        3. What each part (SELECT, FROM, WHERE, GROUP BY, etc.) means
        4. What the expected output represents
        """
        
        return self.ask(prompt, system, temperature=0.2)
    
    def generate_sql(self, 
                    natural_language: str, 
                    schema_info: Optional[str] = None,
                    examples: Optional[List[Dict]] = None) -> str:
        """
        Generate SQL dari natural language dengan few-shot examples
        
        Args:
            natural_language: Deskripsi dalam bahasa alami
            schema_info: Informasi schema database (opsional)
            examples: Contoh-contoh few-shot learning (opsional)
        """
        system = """You are a SQL expert. Generate accurate, efficient PostgreSQL queries 
        based on user requests. Only output the SQL query, no explanations."""
        
        # Schema default jika tidak disediakan
        if not schema_info:
            schema_info = """
            Table: users (id, first_name, last_name, email, created_at)
            Table: products (id, name, category, brand, price, cost)
            Table: order_items (id, order_id, user_id, product_id, quantity, sale_price, status, created_at)
            """
        
        prompt = f"""
        Database schema:
        {schema_info}
        """
        
        # Tambahkan few-shot examples jika ada
        if examples:
            prompt += "\n\nExamples:\n"
            for i, ex in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Request: {ex['request']}\n"
                prompt += f"SQL: {ex['sql']}\n\n"
        
        prompt += f"""
        User request: {natural_language}
        
        Generate only the SQL query (no explanations, no markdown formatting):
        """
        
        # Generate dengan temperature rendah untuk konsistensi
        sql = self.ask(prompt, system, temperature=0.1)
        
        # Bersihkan hasil SQL
        sql = self._clean_sql(sql)
        
        return sql
    
    def analyze_data(self, 
                    question: str, 
                    df_summary: Dict[str, Any],
                    include_recommendations: bool = True) -> str:
        """
        Analisis data berdasarkan pertanyaan dan ringkasan dataframe
        
        Args:
            question: Pertanyaan user tentang data
            df_summary: Ringkasan statistik dataframe
            include_recommendations: Sertakan rekomendasi bisnis
        """
        system = "You are a senior data analyst providing insights from e-commerce data."
        
        context = "DATA SUMMARY:\n"
        for key, value in df_summary.items():
            context += f"{key}: {value}\n"
        
        prompt = f"""
        Question: {question}
        
        Based on the data summary above, please provide:
        1. Key insights answering the question
        2. Supporting data points
        """
        
        if include_recommendations:
            prompt += "\n3. Business recommendations based on these insights"
        
        return self.ask(prompt, system, context=context, temperature=0.4)
    
    def check_model_availability(self) -> bool:
        """
        Cek apakah model tersedia di Ollama
        """
        try:
            # Cek daftar model
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                # Cek apakah model yang diminta ada
                model_found = any(self.model in m for m in available_models)
                
                if not model_found:
                    print(f"⚠️ Model '{self.model}' tidak ditemukan. Model tersedia: {available_models}")
                    return False
                return True
            return False
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """
        Dapatkan daftar model yang tersedia di Ollama
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except:
            return []
    
    def _clean_response(self, response: str) -> str:
        """
        Bersihkan respons dari artifacts yang tidak diinginkan
        """
        # Hapus prompt yang mungkin terulang
        unwanted_phrases = [
            "USER:", "CONTEXT:", "HUMAN:", "AI:", 
            "USER QUERY:", "RESPONSE:", "Assistant:"
        ]
        
        cleaned = response
        for phrase in unwanted_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        # Hapus spasi berlebih
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _clean_sql(self, sql: str) -> str:
        """
        Bersihkan SQL dari formatting yang tidak diinginkan
        """
        # Hapus markdown SQL jika ada
        sql = sql.replace('```sql', '').replace('```', '')
        
        # Hapus leading/trailing whitespace
        sql = sql.strip()
        
        # Pastikan tidak ada teks tambahan setelah SQL
        # (ambil hanya sampai titik koma terakhir jika ada)
        if ';' in sql:
            sql = sql.split(';')[0] + ';'
        
        return sql


# ==================== FUNGSI UTILITAS ====================

def get_llm_with_fallback(preferred_model: str = "gemma2:2b") -> LocalLLM:
    """
    Inisialisasi LLM dengan fallback ke model lain jika model preferensi tidak tersedia
    """
    llm = LocalLLM(model=preferred_model)
    
    # Cek ketersediaan model
    if not llm.check_model_availability():
        available = llm.list_available_models()
        if available:
            # Gunakan model pertama yang tersedia
            fallback_model = available[0]
            print(f"⚠️ Model '{preferred_model}' tidak tersedia. Menggunakan '{fallback_model}' sebagai fallback.")
            return LocalLLM(model=fallback_model)
        else:
            print("❌ Tidak ada model yang tersedia. Jalankan 'ollama pull gemma2:2b' terlebih dahulu.")
            return None
    return llm


def test_ollama_connection() -> Dict[str, Any]:
    """
    Test koneksi ke Ollama dan kembalikan status
    """
    result = {
        "connected": False,
        "models": [],
        "error": None
    }
    
    try:
        # Test API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            result["connected"] = True
            models = response.json().get('models', [])
            result["models"] = [m['name'] for m in models]
        else:
            result["error"] = f"Error {response.status_code}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Ollama tidak berjalan. Jalankan 'ollama serve'"
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ==================== TESTING ====================

if __name__ == "__main__":
    print("🔍 Testing Local LLM Helper...")
    
    # Test koneksi
    status = test_ollama_connection()
    if status["connected"]:
        print(f"✅ Connected to Ollama")
        print(f"📦 Available models: {status['models']}")
        
        # Inisialisasi LLM
        llm = get_llm_with_fallback()
        
        if llm:
            # Test basic ask
            print("\n🤖 Testing basic ask...")
            response = llm.ask("Hello, who are you?", temperature=0.3)
            print(f"Response: {response[:100]}...")
            
            # Test SQL generation
            print("\n🔧 Testing SQL generation...")
            sql = llm.generate_sql("Show top 5 products by sales")
            print(f"Generated SQL: {sql}")
            
            # Test SQL explanation
            print("\n📖 Testing SQL explanation...")
            test_query = "SELECT category, SUM(sale_price) as total FROM products JOIN order_items ON products.id = order_items.product_id GROUP BY category"
            explanation = llm.explain_sql(test_query, language="indonesian")
            print(f"Explanation: {explanation[:150]}...")
    else:
        print(f"❌ {status['error']}")