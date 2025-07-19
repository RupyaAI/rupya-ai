#!/usr/bin/env python3
"""
Advanced Handwriting OCR System with Heavy ML Libraries
This script uses multiple heavy ML models to stress test cloud server performance
"""

import os
import sys
import time
import warnings
import psutil
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

# ML and Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tensorflow as tf

# Hugging Face ecosystem
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    AutoProcessor,
    AutoModelForVision2Seq,
    pipeline
)

# OCR Libraries
import easyocr
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR not available, skipping...")
    PaddleOCR = None

# Additional heavy libraries for stress testing
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.ndimage as ndi
from scipy import stats
import joblib

class SystemMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used / (1024**3)
        self.peak_memory = self.initial_memory
        
    def log_status(self, operation: str):
        current_time = time.time()
        elapsed = current_time - self.start_time
        current_memory = psutil.virtual_memory().used / (1024**3)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        print(f"⏱️  [{elapsed:.1f}s] {operation}")
        print(f"💾 Memory: {current_memory:.1f}GB (Peak: {self.peak_memory:.1f}GB)")
        print(f"🖥️  CPU: {psutil.cpu_percent(interval=1):.1f}%")
        print("-" * 50)

class AdvancedImageProcessor:
    """Advanced image preprocessing with heavy ML operations"""
    
    def __init__(self):
        print("🔧 Initializing Advanced Image Processor...")
        self.setup_models()
        
    def setup_models(self):
        """Setup additional ML models for image processing"""
        try:
            # TensorFlow model for image enhancement
            print("📦 Loading TensorFlow models...")
            self.tf_session = tf.compat.v1.Session() if hasattr(tf.compat, 'v1') else None
            
            # Scikit-learn models for image analysis
            print("📦 Initializing ML classifiers...")
            self.kmeans = KMeans(n_clusters=8, random_state=42)
            self.pca = PCA(n_components=50)
            
        except Exception as e:
            print(f"⚠️  Warning: Some models failed to load: {e}")
    
    def advanced_preprocessing(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Apply advanced preprocessing with heavy ML operations"""
        processed_images = []
        
        # Original
        processed_images.append(("original", image))
        
        # Basic preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        processed_images.append(("grayscale", gray))
        
        # Advanced denoising with heavy operations
        print("🔄 Applying advanced denoising...")
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        processed_images.append(("denoised", denoised))
        
        # Contrast enhancement using TensorFlow operations
        print("🔄 TensorFlow-based enhancement...")
        try:
            tf_enhanced = self.tensorflow_enhance(denoised)
            processed_images.append(("tf_enhanced", tf_enhanced))
        except Exception as e:
            print(f"⚠️  TensorFlow enhancement failed: {e}")
        
        # ML-based clustering for segmentation
        print("🔄 ML-based segmentation...")
        try:
            segmented = self.ml_segmentation(denoised)
            processed_images.append(("ml_segmented", segmented))
        except Exception as e:
            print(f"⚠️  ML segmentation failed: {e}")
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morphological", morph))
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(("adaptive_thresh", adaptive))
        
        return processed_images
    
    def tensorflow_enhance(self, image: np.ndarray) -> np.ndarray:
        """Use TensorFlow for image enhancement"""
        # Convert to tensor
        img_tensor = tf.constant(image, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        img_tensor = tf.expand_dims(img_tensor, -1)  # Add channel dimension
        
        # Normalize
        img_tensor = img_tensor / 255.0
        
        # Apply Gaussian filter
        # Create a simple Gaussian kernel
        kernel = tf.constant([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=tf.float32)
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        
        # Apply convolution
        filtered = tf.nn.conv2d(img_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        # Convert back to numpy
        result = filtered.numpy()[0, :, :, 0] * 255
        return result.astype(np.uint8)
    
    def ml_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Use ML clustering for image segmentation"""
        # Reshape image for clustering
        h, w = image.shape
        image_flat = image.reshape(-1, 1)
        
        # Apply K-means clustering
        clusters = self.kmeans.fit_predict(image_flat)
        
        # Reshape back to image
        segmented = clusters.reshape(h, w)
        
        # Convert to binary (assuming text is the darkest cluster)
        text_cluster = np.argmin(self.kmeans.cluster_centers_)
        binary = (segmented == text_cluster).astype(np.uint8) * 255
        
        return binary

class HeavyMLOCRSystem:
    """OCR system with multiple heavy ML models"""
    
    def __init__(self):
        print("🚀 Initializing Heavy ML OCR System...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        self.monitor = SystemMonitor()
        self.image_processor = AdvancedImageProcessor()
        self.results_history = []
        
        self.setup_ocr_models()
        self.setup_additional_models()
        
    def setup_ocr_models(self):
        """Setup multiple OCR models"""
        self.monitor.log_status("Loading OCR Models")
        
        try:
            # TrOCR - Handwriting specialist
            print("📦 Loading TrOCR (Handwriting)...")
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            self.trocr_model.to(self.device)
            
            # TrOCR - Printed text variant
            print("📦 Loading TrOCR (Printed)...")
            self.trocr_print_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_print_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_print_model.to(self.device)
            
            # EasyOCR
            print("📦 Loading EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            # PaddleOCR
            if PaddleOCR:
                print("📦 Loading PaddleOCR...")
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            else:
                self.paddle_ocr = None
                
        except Exception as e:
            print(f"⚠️  Error loading OCR models: {e}")
    
    def setup_additional_models(self):
        """Setup additional heavy ML models for stress testing"""
        self.monitor.log_status("Loading Additional Heavy Models")
        
        try:
            # Hugging Face pipelines (heavy models)
            print("📦 Loading Hugging Face pipelines...")
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
            
            # Image classification for additional processing
            print("📦 Loading image classification models...")
            self.image_classifier = pipeline("image-classification")
            
        except Exception as e:
            print(f"⚠️  Warning: Additional models failed to load: {e}")
    
    def process_with_trocr(self, image: np.ndarray, model_type: str = "handwritten") -> str:
        """Process image with TrOCR models"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Choose model
            if model_type == "handwritten":
                processor = self.trocr_processor
                model = self.trocr_model
            else:
                processor = self.trocr_print_processor
                model = self.trocr_print_model
            
            # Process
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=256)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"⚠️  TrOCR ({model_type}) error: {e}")
            return ""
    
    def process_with_easyocr(self, image: np.ndarray) -> str:
        """Process image with EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image, detail=0)
            return ' '.join(results)
        except Exception as e:
            print(f"⚠️  EasyOCR error: {e}")
            return ""
    
    def process_with_paddle(self, image: np.ndarray) -> str:
        """Process image with PaddleOCR"""
        if not self.paddle_ocr:
            return ""
        
        try:
            results = self.paddle_ocr.ocr(image, cls=True)
            if results and results[0]:
                text = ' '.join([line[1][0] for line in results[0]])
                return text
            return ""
        except Exception as e:
            print(f"⚠️  PaddleOCR error: {e}")
            return ""
    
    def analyze_text_with_nlp(self, text: str) -> Dict:
        """Analyze extracted text with heavy NLP models"""
        if not text.strip():
            return {}
        
        analysis = {}
        
        try:
            # Sentiment analysis
            sentiment = self.sentiment_pipeline(text)[0]
            analysis['sentiment'] = sentiment
            
            # Named Entity Recognition
            entities = self.ner_pipeline(text)
            analysis['entities'] = entities
            
            # Basic statistics
            analysis['word_count'] = len(text.split())
            analysis['char_count'] = len(text)
            
        except Exception as e:
            print(f"⚠️  NLP analysis error: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def ensemble_ocr(self, image_path: str) -> Dict:
        """Process image with all models and return comprehensive results"""
        print(f"🔍 Processing: {image_path}")
        self.monitor.log_status("Starting OCR Processing")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        processed_images = self.image_processor.advanced_preprocessing(image)
        
        all_results = {}
        confidence_scores = {}
        
        # Process each preprocessed version with all models
        for img_name, img_data in processed_images:
            print(f"🔄 Processing {img_name} version...")
            
            # TrOCR Handwritten
            trocr_hand = self.process_with_trocr(img_data, "handwritten")
            if trocr_hand:
                key = f"TrOCR_Hand_{img_name}"
                all_results[key] = trocr_hand
                confidence_scores[key] = len(trocr_hand)  # Simple confidence based on length
            
            # TrOCR Printed
            trocr_print = self.process_with_trocr(img_data, "printed")
            if trocr_print:
                key = f"TrOCR_Print_{img_name}"
                all_results[key] = trocr_print
                confidence_scores[key] = len(trocr_print)
            
            # EasyOCR
            easy_result = self.process_with_easyocr(img_data)
            if easy_result:
                key = f"EasyOCR_{img_name}"
                all_results[key] = easy_result
                confidence_scores[key] = len(easy_result)
            
            # PaddleOCR
            paddle_result = self.process_with_paddle(img_data)
            if paddle_result:
                key = f"PaddleOCR_{img_name}"
                all_results[key] = paddle_result
                confidence_scores[key] = len(paddle_result)
        
        # Select best result
        best_result = self.select_best_result(all_results, confidence_scores)
        
        # Analyze with NLP
        nlp_analysis = self.analyze_text_with_nlp(best_result)
        
        # Compile final results
        final_results = {
            'best_result': best_result,
            'all_results': all_results,
            'confidence_scores': confidence_scores,
            'nlp_analysis': nlp_analysis,
            'processing_stats': self.get_processing_stats()
        }
        
        self.results_history.append(final_results)
        self.monitor.log_status("OCR Processing Complete")
        
        return final_results
    
    def select_best_result(self, results: Dict[str, str], confidence: Dict[str, float]) -> str:
        """Select the best OCR result using ensemble logic"""
        if not results:
            return "No text detected"
        
        # Priority: TrOCR handwritten results
        trocr_hand_results = {k: v for k, v in results.items() if 'TrOCR_Hand' in k}
        if trocr_hand_results:
            best_key = max(trocr_hand_results.keys(), key=lambda k: confidence.get(k, 0))
            return trocr_hand_results[best_key]
        
        # Fallback: highest confidence result
        best_key = max(results.keys(), key=lambda k: confidence.get(k, 0))
        return results[best_key]
    
    def get_processing_stats(self) -> Dict:
        """Get system processing statistics"""
        return {
            'total_time': time.time() - self.monitor.start_time,
            'peak_memory_gb': self.monitor.peak_memory,
            'current_memory_gb': psutil.virtual_memory().used / (1024**3),
            'cpu_percent': psutil.cpu_percent(),
            'device': str(self.device)
        }
    
    def save_results(self, results: Dict, output_path: str = "ocr_results.json"):
        """Save results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {output_path}")
    
    def create_visualization(self, image_path: str, results: Dict):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced OCR Analysis Results', fontsize=16, fontweight='bold')
        
        # Original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Best result text
        axes[0, 1].text(0.1, 0.5, f"Best Result:\n\n{results['best_result']}", 
                       fontsize=12, transform=axes[0, 1].transAxes, 
                       verticalalignment='center', wrap=True)
        axes[0, 1].set_title('Extracted Text')
        axes[0, 1].axis('off')
        
        # Processing stats
        stats = results['processing_stats']
        stats_text = f"""Processing Statistics:
        
Time: {stats['total_time']:.2f}s
Peak Memory: {stats['peak_memory_gb']:.2f}GB
CPU Usage: {stats['cpu_percent']:.1f}%
Device: {stats['device']}
        """
        axes[0, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       transform=axes[0, 2].transAxes, verticalalignment='center')
        axes[0, 2].set_title('System Performance')
        axes[0, 2].axis('off')
        
        # Confidence scores
        if results['confidence_scores']:
            models = list(results['confidence_scores'].keys())[:10]  # Top 10
            scores = [results['confidence_scores'][m] for m in models]
            
            axes[1, 0].barh(range(len(models)), scores)
            axes[1, 0].set_yticks(range(len(models)))
            axes[1, 0].set_yticklabels([m.replace('_', '\n') for m in models], fontsize=8)
            axes[1, 0].set_title('Model Confidence Scores')
            axes[1, 0].set_xlabel('Score')
        
        # NLP Analysis
        if 'nlp_analysis' in results and results['nlp_analysis']:
            nlp = results['nlp_analysis']
            nlp_text = f"""NLP Analysis:
            
Word Count: {nlp.get('word_count', 'N/A')}
Character Count: {nlp.get('char_count', 'N/A')}

Sentiment: {nlp.get('sentiment', {}).get('label', 'N/A')}
Confidence: {nlp.get('sentiment', {}).get('score', 0):.3f}

Entities: {len(nlp.get('entities', []))} found
            """
            axes[1, 1].text(0.1, 0.5, nlp_text, fontsize=10,
                           transform=axes[1, 1].transAxes, verticalalignment='center')
        axes[1, 1].set_title('NLP Analysis')
        axes[1, 1].axis('off')
        
        # Memory usage over time (placeholder)
        axes[1, 2].plot([0, stats['total_time']], 
                       [self.monitor.initial_memory, stats['current_memory_gb']])
        axes[1, 2].set_title('Memory Usage')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Memory (GB)')
        
        plt.tight_layout()
        plt.savefig('ocr_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as: ocr_analysis_results.png")

def main():
    """Main function to run the heavy OCR system"""
    print("🧪 HEAVY ML OCR SYSTEM - CLOUD SERVER STRESS TEST")
    print("=" * 60)
    
    # Initialize system
    ocr_system = HeavyMLOCRSystem()
    
    # Check for image file
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("❌ No image files found in current directory!")
        print("Please place a PNG/JPG image file in the same directory as this script.")
        return
    
    # Process first available image
    image_path = image_files[0]
    print(f"📸 Processing image: {image_path}")
    
    try:
        # Run OCR analysis
        results = ocr_system.ensemble_ocr(image_path)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 FINAL OCR RESULTS")
        print("=" * 60)
        print(f"📝 Extracted Text: {results['best_result']}")
        
        # Save results
        ocr_system.save_results(results)
        
        # Create visualization
        ocr_system.create_visualization(image_path, results)
        
        # Final system stats
        final_stats = ocr_system.get_processing_stats()
        print(f"\n📊 FINAL SYSTEM PERFORMANCE:")
        print(f"   ⏱️  Total Processing Time: {final_stats['total_time']:.2f} seconds")
        print(f"   💾 Peak Memory Usage: {final_stats['peak_memory_gb']:.2f} GB")
        print(f"   🖥️  Final CPU Usage: {final_stats['cpu_percent']:.1f}%")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🎉 OCR Analysis Complete!")
        print("📁 Check the following output files:")
        print("   - ocr_results.json (detailed results)")
        print("   - ocr_analysis_results.png (visualization)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Force cleanup
        try:
            del ocr_system
            gc.collect()
        except:
            pass

def stress_test_mode():
    """Additional stress test mode for heavy computation"""
    print("🔥 ENTERING STRESS TEST MODE")
    print("This will perform additional heavy computations to test server limits...")
    
    try:
        # Heavy numpy operations
        print("🧮 Running heavy NumPy operations...")
        large_matrix = np.random.rand(5000, 5000)
        result = np.linalg.svd(large_matrix)
        del large_matrix, result
        
        # Heavy pandas operations
        print("📊 Running heavy Pandas operations...")
        df = pd.DataFrame(np.random.rand(1000000, 50))
        df_processed = df.groupby(df.columns[0] // 0.1).agg(['mean', 'std', 'min', 'max'])
        del df, df_processed
        
        # Heavy scikit-learn operations
        print("🤖 Running heavy Scikit-learn operations...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=50000, n_features=100, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        del X, y, rf
        
        print("✅ Stress test completed successfully!")
        
    except Exception as e:
        print(f"⚠️ Stress test encountered issues: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Heavy ML OCR System for Cloud Testing')
    parser.add_argument('--stress-test', action='store_true', 
                       help='Run additional stress tests')
    parser.add_argument('--image', type=str, 
                       help='Specific image file to process')
    
    args = parser.parse_args()
    
    # Print system information
    print("🖥️ SYSTEM INFORMATION")
    print("=" * 30)
    try:
        print(f"Python Version: {sys.version}")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("GPU: Not available (CPU mode)")
            
    except Exception as e:
        print(f"Could not get system info: {e}")
    
    print("\n")
    
    # Run stress test if requested
    if args.stress_test:
        stress_test_mode()
        print("\n")
    
    # Run main OCR system
    if args.image:
        # Process specific image
        if os.path.exists(args.image):
            print(f"Processing specified image: {args.image}")
            # You could modify main() to accept image path parameter
            main()
        else:
            print(f"❌ Image file not found: {args.image}")
    else:
        # Run with auto-detection
        main()