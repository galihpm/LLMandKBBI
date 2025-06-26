# -*- coding: utf-8 -*-
"""Generate Indonesian Dictionary Definitions with Ollama Llama 3.2:3b

Local version adapted from Google Colab OpenAI script
"""

import pandas as pd
import time
import os
import requests
import json
from tqdm import tqdm

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"

def check_ollama_connection():
    """Check if Ollama is running and the model is available"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version")
        if response.status_code == 200:
            print("✓ Ollama is running")
            
            # Check if model is available
            models_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if models_response.status_code == 200:
                models = models_response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if MODEL_NAME in model_names:
                    print(f"✓ Model {MODEL_NAME} is available")
                    return True
                else:
                    print(f"✗ Model {MODEL_NAME} not found. Available models: {model_names}")
                    print(f"Please run: ollama pull {MODEL_NAME}")
                    return False
            else:
                print("✗ Could not fetch model list")
                return False
        else:
            print("✗ Ollama is not running. Please start Ollama first.")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def generate_definition_ollama(word, max_retries=3):
    """Generate definition using Ollama API"""
    
    kbbi_style_guidelines = """You are an expert Indonesian linguist specializing in creating dictionary definitions in the style of Kamus Besar Bahasa Indonesia (KBBI).

INTERNAL INSTRUCTIONS (Do not include in output):
1. For each word, ANALYZE its part of speech
2. Choose ONE definition type UNLESS the word naturally requires more than one (e.g., analytical definition plus a synonym definition).
   - Analytical Definition (genus + differentia)
   - Encyclopedic Definition (can be detailed, but avoid to be too encyclopedic)
   - Synonym Definition (use another word with the closest or the same meaning)
   - Antonym Definition (use negation in the beginning of definition)
   - Ostensive Definition (define something as you are pointing at the object directly)
3. If needed, COMBINE multiple definitions within a single sentence using a semicolon `;`. Example: komputer: alat untuk mengolah data secara elektronik; laptop
4. CREATE the definition following KBBI principles:
   - The definition must be self-explanatory
   - Avoid using words more complicated than the words being defined
   - Match the part of speech in the first word of the definition
   - Omit copula words like "adalah" and "merupakan"
   - Use simpler terms than the word being defined
   - Avoid circular definitions
   - Be specific but not too specific

OUTPUT FORMAT:
Return ONLY the definition with no explanations, headers, or additional text.

REFERENCE EXAMPLES:
Analytical Definition — pohon: tumbuhan yang berbatang keras dan besar; pokok kayu
Encyclopedic Definition — matahari: benda angkasa, titik pusat tata surya berupa bola berisi gas yang mendatangkan terang dan panas pada bumi pada siang hari
Synonym Definition — kudus: suci; murni
Antonym Definition — nirkabel: tanpa menggunakan kabel
Ostensive Definition — biru: warna dasar yg serupa dng warna langit yg terang (tidak berawan dan sebagainya) serta merupakan warna asli (bukan hasil campuran beberapa warna)"""

    prompt = f"Kata: {word}\nDefinisi:"
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": f"{kbbi_style_guidelines}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200,  # Limit response length
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                definition = result.get('response', '').strip()
                
                # Clean up the definition
                definition = clean_definition(word, definition)
                return definition
            else:
                print(f"Error: HTTP {response.status_code} for word '{word}'")
                
        except requests.exceptions.Timeout:
            print(f"Timeout for word '{word}' (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            print(f"Error generating definition for '{word}' (attempt {attempt+1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)  # Wait before retry
    
    return f"Error: Failed to generate definition after {max_retries} attempts"

def clean_definition(word, definition):
    """Clean the generated definition"""
    import re
    
    # Remove any format where the word is followed by colon
    pattern = r'^' + re.escape(word) + r'\s*:\s*'
    definition = re.sub(pattern, '', definition, flags=re.IGNORECASE).strip()
    
    # Handle cases where only a colon appears at the beginning
    definition = re.sub(r'^:\s*', '', definition).strip()
    
    # Remove the word if it still appears at the beginning (without colon)
    if definition.lower().startswith(word.lower()):
        word_pattern = r'^' + re.escape(word) + r'\s*\d*\.?\s*'
        definition = re.sub(word_pattern, '', definition, flags=re.IGNORECASE).strip()
    
    # Handle any remaining numbering patterns at the beginning
    definition = re.sub(r'^(\d+\.?\s*)', '', definition).strip()
    
    # Remove common unwanted prefixes
    unwanted_prefixes = ['definisi:', 'arti:', 'makna:', 'pengertian:']
    for prefix in unwanted_prefixes:
        if definition.lower().startswith(prefix):
            definition = definition[len(prefix):].strip()
    
    return definition

def load_data():
    """Load word list and reference data"""
    try:
        # Load word list
        words_df = pd.read_csv('word.csv')
        print(f"✓ Word list loaded successfully: {len(words_df)} words found")
        
        word_column = "word"
        words_list = words_df[word_column].tolist()
        
        # Load reference definitions
        ref_df = pd.read_csv('reference.csv')
        print(f"✓ Reference definitions loaded successfully: {len(ref_df)} entries found")
        
        return words_list, ref_df
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("Make sure 'word.csv' and 'reference.csv' are in the current directory")
        return None, None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None

def generate_definitions(words_list, ref_df):
    """Generate definitions for all words"""
    results = []
    ref_word_column = "word"
    ref_def_column = "definition"
    
    print(f"Generating definitions for {len(words_list)} words...")
    
    for word in tqdm(words_list, desc="Generating definitions"):
        definition = generate_definition_ollama(word)
        
        # Find the reference definition
        reference_def = None
        if ref_df is not None:
            match = ref_df[ref_df[ref_word_column] == word]
            if not match.empty:
                reference_def = match.iloc[0][ref_def_column]
        
        result = {
            "word": word,
            "generated_definition": definition,
        }
        
        if reference_def:
            result["reference_definition"] = reference_def
        
        results.append(result)
        
        # Small delay to be gentle with the API
        time.sleep(0.5)
    
    return pd.DataFrame(results)



def main():
    """Main function to run the definition generation and evaluation"""
    print("Indonesian Dictionary Definition Generator with Ollama")
    print("="*60)
    
    # Check Ollama connection
    if not check_ollama_connection():
        return
    
    # Load data
    words_list, ref_df = load_data()
    if words_list is None:
        return
    
    # Generate definitions
    results_df = generate_definitions(words_list, ref_df)
    
    # Save results
    output_filename = "generated_definitions_ollama.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"✓ Generated definitions saved to {output_filename}")
    
    print("\n✓ Generation complete!")

if __name__ == "__main__":
    main()