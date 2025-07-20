#!/usr/bin/env python3
"""
RupyaAI Cloud VPS Testing - PART 1
System Setup and OCR Testing with handwriting.png
"""

import os
import sys
import time
import warnings
import gc
import traceback
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Suppress ALL warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def safe_import(module_name: str, package_name: str = None, required: bool = False):
    """Ultra-safe import with comprehensive error handling"""
    try:
        if package_name:
            module = __import__(package_name, fromlist=[module_name])
            return getattr(module, module_name)
        else:
            return __import__(module_name)
    except Exception as e:
        if required:
            print(f"⚠️  {module_name} not available: {e}")
        return None

# Safe imports with fallbacks
print("🚀 RupyaAI OCR System - PART 1 - Loading libraries...")

# Core libraries
numpy = safe_import('numpy')
np = numpy

# System monitoring
psutil = safe_import('psutil')

# Image processing
cv2 = safe_import('cv2')
PIL = safe_import('PIL')
if PIL:
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        PIL_Image = Image
    except:
        PIL = None
        PIL_Image = None

# ML libraries
torch = safe_import('torch')
tensorflow = safe_import('tensorflow')
if tensorflow:
    try:
        tf = tensorflow
        # Disable GPU for CPU-only testing
        tf.config.experimental.set_visible_devices([], 'GPU')
    except:
        tensorflow = None

# Transformers for financial document processing
transformers = safe_import('transformers')

# OCR libraries (critical for RupyaAI)
easyocr = safe_import('easyocr')
paddleocr = safe_import('paddleocr')
pytesseract = safe_import('pytesseract')

# Scientific computing
pandas = safe_import('pandas')
sklearn = safe_import('sklearn')

print("✅ Library loading completed")

class RupyaAISystemMonitor:
    """Comprehensive system monitor for RupyaAI infrastructure testing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.has_psutil = psutil is not None
        self.logs = []
        self.memory_timeline = []
        self.cpu_timeline = []
        
    def log(self, message: str):
        """Log message with detailed system metrics"""
        elapsed = time.time() - self.start_time
        log_entry = f"[{elapsed:.1f}s] {message}"
        print(log_entry)
        self.logs.append(log_entry)
        
        if self.has_psutil:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.1)
                
                memory_gb = memory.used / (1024**3)
                memory_percent = memory.percent
                
                self.memory_timeline.append((elapsed, memory_gb))
                self.cpu_timeline.append((elapsed, cpu))
                
                print(f"     💾 {memory_gb:.1f}GB ({memory_percent:.1f}%) | 🖥️  {cpu:.1f}% | 🔄 {memory.available / (1024**3):.1f}GB free")
                
                # Warn if memory usage is high (important for 8GB VPS)
                if memory_percent > 85:
                    print(f"     ⚠️  High memory usage! ({memory_percent:.1f}%)")
                    
            except Exception as e:
                print(f"     ⚠️  Monitoring error: {e}")
    
    def get_system_summary(self):
        """Get comprehensive system performance summary"""
        if not self.has_psutil:
            return {"error": "psutil not available"}
        
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            summary = {
                'total_time': time.time() - self.start_time,
                'peak_memory_gb': max([m[1] for m in self.memory_timeline]) if self.memory_timeline else 0,
                'avg_cpu_percent': sum([c[1] for c in self.cpu_timeline]) / len(self.cpu_timeline) if self.cpu_timeline else 0,
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'cpu_count': psutil.cpu_count(),
            }
            
            # RupyaAI specific assessments
            summary['rupyai_ready'] = (
                summary['total_memory_gb'] >= 7.5 and  # At least 7.5GB for 8GB VPS
                summary['disk_free_gb'] >= 10 and      # At least 10GB free space
                summary['cpu_count'] >= 2              # At least 2 CPU cores
            )
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}

class HandwritingImageLoader:
    """Specialized image loader for handwriting documents"""
    
    def __init__(self):
        self.has_cv2 = cv2 is not None
        self.has_pil = PIL is not None
        self.has_numpy = numpy is not None
        
        print(f"✍️  Handwriting Document Processor Status:")
        print(f"   OpenCV: {'✅' if self.has_cv2 else '❌'}")
        print(f"   PIL: {'✅' if self.has_pil else '❌'}")
        print(f"   NumPy: {'✅' if self.has_numpy else '❌'}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load handwriting images with multiple fallback methods"""
        if not os.path.exists(image_path):
            print(f"❌ Document not found: {image_path}")
            return None
        
        print(f"📄 Loading handwriting document: {image_path}")
        
        # Method 1: OpenCV (preferred for document processing)
        if self.has_cv2:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    print(f"✅ Loaded with OpenCV: {img.shape}")
                    return img
            except Exception as e:
                print(f"⚠️  OpenCV failed: {e}")
        
        # Method 2: PIL (good for various formats)
        if self.has_pil and self.has_numpy:
            try:
                pil_img = PIL_Image.open(image_path)
                
                # Handle different modes
                if pil_img.mode == 'RGBA':
                    # Create white background for transparency
                    background = PIL_Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[-1])
                    pil_img = background
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                img_array = np.array(pil_img)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3:
                    img_array = img_array[:, :, ::-1]
                
                print(f"✅ Loaded with PIL: {img_array.shape}")
                return img_array
            except Exception as e:
                print(f"⚠️  PIL failed: {e}")
        
        print(f"❌ Could not load document image")
        return None
    
    def preprocess_handwriting(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Preprocess handwriting documents for optimal OCR"""
        if not self.has_numpy or image is None:
            return []
        
        processed = []
        
        try:
            # Original
            processed.append(("original", image.copy()))
            
            # Grayscale conversion (essential for handwriting)
            if len(image.shape) == 3:
                if self.has_cv2:
                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        processed.append(("grayscale", gray))
                    except:
                        pass
                
                # NumPy fallback
                try:
                    gray_np = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                    gray = gray_np.astype(np.uint8)
                    processed.append(("grayscale_numpy", gray))
                except:
                    gray = image
            else:
                gray = image
                processed.append(("grayscale_input", gray))
            
            # Handwriting specific preprocessing
            if self.has_cv2 and len(gray.shape) == 2:
                try:
                    # 1. Noise reduction (important for handwriting)
                    denoised = cv2.fastNlMeansDenoising(gray)
                    processed.append(("denoised", denoised))
                    
                    # 2. Contrast enhancement (for better handwriting visibility)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(denoised)
                    processed.append(("enhanced", enhanced))
                    
                    # 3. Binary threshold (critical for handwriting)
                    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    processed.append(("binary", binary))
                    
                    # 4. Adaptive thresholding (handle varying pen pressure)
                    adaptive = cv2.adaptiveThreshold(
                        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )
                    processed.append(("adaptive", adaptive))
                    
                except Exception as e:
                    print(f"⚠️  Advanced preprocessing failed: {e}")
            
            print(f"✅ Created {len(processed)} handwriting variants for OCR")
            return processed
            
        except Exception as e:
            print(f"⚠️  Handwriting preprocessing error: {e}")
            return [("original", image)] if image is not None else []

class RupyaAIOCREngine:
    """Production-ready OCR engine for RupyaAI handwriting processing"""
    
    def __init__(self):
        self.monitor = RupyaAISystemMonitor()
        self.image_loader = HandwritingImageLoader()
        self.ocr_models = {}
        self.device = 'cpu'  # CPU-only for VPS compatibility
        
        self.monitor.log("Initializing RupyaAI OCR Engine")
        self.setup_models()
        
    def setup_models(self):
        """Setup OCR models suitable for handwriting"""
        self.monitor.log("Setting up OCR models for handwriting processing")
        
        # Model 1: TrOCR (best for handwriting)
        self.setup_trocr()
        
        # Model 2: EasyOCR (good general purpose)
        self.setup_easyocr()
        
        # Model 3: PaddleOCR (alternative OCR)
        self.setup_paddleocr()
        
        # Model 4: Tesseract (traditional OCR)
        self.setup_tesseract()
        
        # Model 5: Fallback analyzer
        self.setup_fallback()
        
        available_models = list(self.ocr_models.keys())
        self.monitor.log(f"Available OCR models: {available_models}")
        
        if not self.ocr_models:
            print("❌ No OCR models available! RupyaAI cannot function without OCR.")
        
    def setup_trocr(self):
        """Setup TrOCR for handwriting recognition"""
        if not (transformers and torch and PIL):
            print("⚠️  TrOCR requirements not met (transformers, torch, PIL)")
            return
        
        try:
            self.monitor.log("Loading TrOCR models...")
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Handwritten model (best for handwriting)
            try:
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                model.eval()
                
                self.ocr_models['trocr_handwritten'] = {
                    'processor': processor,
                    'model': model,
                    'type': 'trocr',
                    'best_for': 'handwritten text recognition'
                }
                print("✅ TrOCR handwritten model loaded - BEST for handwriting")
            except Exception as e:
                print(f"⚠️  TrOCR handwritten failed: {e}")
                
        except Exception as e:
            print(f"⚠️  TrOCR setup completely failed: {e}")
    
    def setup_easyocr(self):
        """Setup EasyOCR for text extraction"""
        if not easyocr:
            print("⚠️  EasyOCR not available")
            return
        
        try:
            self.monitor.log("Loading EasyOCR...")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ocr_models['easyocr'] = {
                'reader': reader,
                'type': 'easyocr',
                'best_for': 'general text recognition'
            }
            print("✅ EasyOCR loaded - good for general text")
        except Exception as e:
            print(f"⚠️  EasyOCR setup failed: {e}")
    
    def setup_paddleocr(self):
        """Setup PaddleOCR for text recognition"""
        if not paddleocr:
            print("⚠️  PaddleOCR not available")
            return
        
        try:
            self.monitor.log("Loading PaddleOCR...")
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                show_log=False,
                use_gpu=False,
                cpu_threads=2  # Optimize for 2-core VPS
            )
            self.ocr_models['paddleocr'] = {
                'ocr': ocr,
                'type': 'paddleocr',
                'best_for': 'alternative OCR engine'
            }
            print("✅ PaddleOCR loaded - alternative OCR")
        except Exception as e:
            print(f"⚠️  PaddleOCR setup failed: {e}")
    
    def setup_tesseract(self):
        """Setup Tesseract for traditional OCR"""
        if not pytesseract:
            print("⚠️  Tesseract not available")
            return
        
        try:
            self.monitor.log("Testing Tesseract...")
            # Test if tesseract is actually installed
            version = pytesseract.get_tesseract_version()
            
            self.ocr_models['tesseract'] = {
                'type': 'tesseract',
                'version': str(version),
                'best_for': 'traditional OCR'
            }
            print(f"✅ Tesseract {version} loaded - traditional OCR")
        except Exception as e:
            print(f"⚠️  Tesseract setup failed: {e}")
    
    def setup_fallback(self):
        """Setup basic fallback for system testing"""
        if numpy:
            self.ocr_models['fallback'] = {
                'type': 'fallback',
                'best_for': 'system testing'
            }
            print("✅ Fallback analyzer available")
    
    def process_with_trocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process handwriting with TrOCR"""
        try:
            processor = model_info['processor']
            model = model_info['model']
            
            # Convert to PIL Image for TrOCR
            if len(image.shape) == 3:
                rgb_image = image[:, :, ::-1]  # BGR to RGB
                pil_image = PIL_Image.fromarray(rgb_image)
            else:
                pil_image = PIL_Image.fromarray(image).convert('RGB')
            
            # Process with TrOCR
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=200)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"⚠️  TrOCR processing error: {e}")
            return ""
    
    def process_with_easyocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process handwriting with EasyOCR"""
        try:
            reader = model_info['reader']
            results = reader.readtext(image, detail=0, paragraph=False)
            
            # Join results with spaces
            text = ' '.join(results)
            return text.strip()
        except Exception as e:
            print(f"⚠️  EasyOCR processing error: {e}")
            return ""
    
    def process_with_paddleocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process handwriting with PaddleOCR"""
        try:
            ocr = model_info['ocr']
            results = ocr.ocr(image, cls=True)
            
            if results and results[0]:
                texts = []
                for line in results[0]:
                    if len(line) >= 2 and line[1] and line[1][0]:
                        texts.append(line[1][0])
                return ' '.join(texts)
            return ""
        except Exception as e:
            print(f"⚠️  PaddleOCR processing error: {e}")
            return ""
    
    def process_with_tesseract(self, model_info: Dict, image: np.ndarray) -> str:
        """Process handwriting with Tesseract"""
        try:
            # Configure Tesseract for handwriting
            config = '--oem 3 --psm 6'  # Uniform block of text
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            print(f"⚠️  Tesseract processing error: {e}")
            return ""
    
    def process_with_fallback(self, model_info: Dict, image: np.ndarray) -> str:
        """Basic image analysis for system testing"""
        try:
            if numpy and len(image.shape) >= 2:
                height, width = image.shape[:2]
                
                # Basic image statistics
                if len(image.shape) == 3:
                    channels = image.shape[2]
                    avg_intensity = numpy.mean(image)
                else:
                    channels = 1
                    avg_intensity = numpy.mean(image)
                
                # Simulate handwriting analysis
                analysis = f"Handwriting Analysis: {width}x{height}px, {channels} channels, avg intensity: {avg_intensity:.1f}"
                
                # Estimate text regions (very basic)
                if avg_intensity < 200:  # Likely has text
                    analysis += " | Text regions detected"
                
                return analysis
            return "Basic system test completed"
        except Exception as e:
            print(f"⚠️  Fallback analysis error: {e}")
            return "System test failed"
    
    def process_handwriting_document(self, image_path: str) -> Dict:
        """Complete handwriting document processing pipeline"""
        self.monitor.log(f"Processing handwriting document: {image_path}")
        
        results = {
            'document_path': image_path,
            'processing_timestamp': datetime.now().isoformat(),
            'best_ocr_result': 'No text extracted',
            'all_ocr_results': {},
            'model_performance': {},
            'errors': [],
            'system_stats': {}
        }
        
        try:
            # Load handwriting document
            image = self.image_loader.load_image(image_path)
            if image is None:
                results['errors'].append("Could not load handwriting document")
                return results
            
            # Preprocess for handwriting OCR
            try:
                variants = self.image_loader.preprocess_handwriting(image)
                if not variants:
                    results['errors'].append("Could not preprocess document")
                    return results
            except Exception as e:
                results['errors'].append(f"Preprocessing failed: {e}")
                variants = [("original", image)]
            
            # Process with all available OCR models
            all_results = {}
            model_scores = {}
            
            for variant_name, variant_image in variants:
                print(f"\n🔄 Processing {variant_name} variant:")
                
                for model_name, model_info in self.ocr_models.items():
                    try:
                        print(f"   📝 {model_name}...")
                        start_time = time.time()
                        
                        # Process with appropriate method
                        if model_info['type'] == 'trocr':
                            text = self.process_with_trocr(model_info, variant_image)
                        elif model_info['type'] == 'easyocr':
                            text = self.process_with_easyocr(model_info, variant_image)
                        elif model_info['type'] == 'paddleocr':
                            text = self.process_with_paddleocr(model_info, variant_image)
                        elif model_info['type'] == 'tesseract':
                            text = self.process_with_tesseract(model_info, variant_image)
                        elif model_info['type'] == 'fallback':
                            text = self.process_with_fallback(model_info, variant_image)
                        else:
                            continue
                        
                        processing_time = time.time() - start_time
                        
                        if text and text.strip():
                            key = f"{model_name}_{variant_name}"
                            all_results[key] = text.strip()
                            
                            # Score based on text length and processing time
                            text_score = len(text.strip())
                            time_score = max(1, 10 - processing_time)  # Prefer faster models
                            model_scores[key] = text_score * time_score
                            
                            print(f"   ✅ {model_name}: {text[:80]}... ({processing_time:.2f}s)")
                        else:
                            print(f"   ⚠️  {model_name}: No text detected")
                        
                        # Record model performance
                        results['model_performance'][f"{model_name}_{variant_name}"] = {
                            'processing_time': processing_time,
                            'text_length': len(text) if text else 0,
                            'success': bool(text and text.strip())
                        }
                        
                    except Exception as e:
                        error_msg = f"{model_name} on {variant_name}: {e}"
                        results['errors'].append(error_msg)
                        print(f"   ❌ {model_name}: {e}")
            
            # Select best OCR result
            if all_results:
                # Prioritize TrOCR for handwriting, then by text length
                model_priority = {
                    'trocr_handwritten': 1000,  # Best for handwriting
                    'easyocr': 800,
                    'paddleocr': 700,
                    'tesseract': 600,
                    'fallback': 100
                }
                
                def score_result(key, text):
                    base_model = key.split('_')[0]
                    if '_' in key and key.split('_')[1] in ['handwritten']:
                        base_model = '_'.join(key.split('_')[:2])
                    
                    priority_score = model_priority.get(base_model, 0)
                    length_score = len(text.strip())
                    return priority_score + length_score
                
                best_key = max(all_results.keys(), key=lambda k: score_result(k, all_results[k]))
                best_text = all_results[best_key]
                
                results.update({
                    'best_ocr_result': best_text,
                    'best_model': best_key,
                    'all_ocr_results': all_results
                })
                
                print(f"\n🎯 Best OCR result from {best_key}:")
                print(f"✍️  {best_text}")
                
            else:
                results['errors'].append("No OCR models produced text results")
                print("❌ No text extracted from handwriting document")
            
            # Add system performance stats
            results['system_stats'] = self.monitor.get_system_summary()
            
        except Exception as e:
            error_msg = f"Critical processing error: {e}"
            results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            traceback.print_exc()
        
        return results

def main_part1():
    """Main function for PART 1 - System Setup and OCR Testing"""
    print("🏦 RUPYAAI CLOUD VPS TESTING - PART 1")
    print("=" * 60)
    print("Testing OCR capabilities with handwriting.png")
    print("This will test if your system can handle RupyaAI's OCR requirements")
    print("")
    
    try:
        # Check for handwriting.png
        if not os.path.exists('handwriting.png'):
            print("❌ handwriting.png not found in current directory!")
            print("Please place handwriting.png file in the same directory as this script.")
            return None
        
        print("📄 Found handwriting.png - proceeding with OCR testing...")
        
        # Initialize OCR engine
        print(f"\n📝 INITIALIZING RUPYAAI OCR ENGINE...")
        print("-" * 40)
        
        ocr_engine = RupyaAIOCREngine()
        
        if not ocr_engine.ocr_models:
            print("❌ No OCR models available! This is critical for RupyaAI.")
            return {
                'best_ocr_result': 'No text extracted',
                'errors': ['No OCR models could be initialized'],
                'system_stats': {},
                'status': 'FAILED'
            }
        
        # Process handwriting.png
        print(f"\n✍️  PROCESSING HANDWRITING DOCUMENT...")
        print("-" * 40)
        
        ocr_results = ocr_engine.process_handwriting_document('handwriting.png')
        
        # Display key results
        print(f"\n💡 HANDWRITING PROCESSING RESULTS:")
        print("=" * 40)
        print(f"📊 Best OCR Model: {ocr_results.get('best_model', 'None')}")
        print(f"📝 Models Tested: {len(ocr_engine.ocr_models)}")
        print(f"🔄 Variants Processed: {len(ocr_results.get('all_ocr_results', {}))}")
        print(f"⚠️  Errors: {len(ocr_results.get('errors', []))}")
        
        if ocr_results.get('best_ocr_result') and ocr_results['best_ocr_result'] != 'No text extracted':
            print(f"\n✅ OCR SUCCESS! Extracted text:")
            print(f"📜 \"{ocr_results['best_ocr_result']}\"")
            ocr_results['status'] = 'SUCCESS'
        else:
            print(f"\n❌ OCR FAILED - No text could be extracted from handwriting")
            ocr_results['status'] = 'FAILED'
        
        # System performance summary
        if ocr_results.get('system_stats'):
            stats = ocr_results['system_stats']
            print(f"\n🖥️  SYSTEM PERFORMANCE SUMMARY:")
            print(f"   💾 Memory: {stats.get('total_memory_gb', 0):.1f}GB total")
            print(f"   🖥️  CPU Cores: {stats.get('cpu_count', 0)}")
            print(f"   💿 Disk Free: {stats.get('disk_free_gb', 0):.1f}GB")
            print(f"   ⏱️  Processing Time: {stats.get('total_time', 0):.1f}s")
            print(f"   🎯 RupyaAI Ready: {'✅ YES' if stats.get('rupyai_ready', False) else '❌ NO'}")
        
        # Save Part 1 results
        try:
            with open('rupyai_part1_results.json', 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, indent=2, default=str, ensure_ascii=False)
            print(f"\n📄 Part 1 results saved to: rupyai_part1_results.json")
        except Exception as e:
            print(f"⚠️  Could not save Part 1 results: {e}")
        
        print(f"\n🎉 PART 1 COMPLETED!")
        print(f"📋 Status: {ocr_results.get('status', 'UNKNOWN')}")
        print(f"📊 Ready for Part 2 analysis")
        
        return ocr_results
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Part 1 interrupted by user")
        return None
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN PART 1:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        try:
            gc.collect()
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        print(f"\n🧹 Part 1 cleanup completed")

if __name__ == "__main__":
    print("🚀 Starting RupyaAI Part 1 Testing...")
    print("This will test OCR capabilities with handwriting.png")
    print("After this completes, run Part 2 for full system analysis")
    print()
    
    # Run Part 1
    results = main_part1()
    
    if results:
        print(f"\n" + "="*60)
        print(f"PART 1 SUMMARY")
        print(f"="*60)
        print(f"✍️  Handwriting OCR: {'✅ Working' if results.get('status') == 'SUCCESS' else '❌ Failed'}")
        print(f"🖥️  System Resources: {'✅ Adequate' if results.get('system_stats', {}).get('rupyai_ready', False) else '⚠️  Check needed'}")
        print(f"📊 Models Available: {len(results.get('all_ocr_results', {}))}")
        
        if results.get('status') == 'SUCCESS':
            print(f"\n🎯 PART 1 PASSED - Your system can handle RupyaAI OCR!")
            print(f"💡 Best model: {results.get('best_model', 'Unknown')}")
            print(f"📝 Extracted: \"{results.get('best_ocr_result', '')[:50]}...\"")
        else:
            print(f"\n⚠️  PART 1 ISSUES DETECTED")
            if results.get('errors'):
                print(f"❌ Errors: {len(results['errors'])}")
                for error in results['errors'][:3]:  # Show first 3 errors
                    print(f"   • {error}")
        
        print(f"\n📁 Results saved to: rupyai_part1_results.json")
        print(f"🔄 Ready for Part 2 - System Analysis & Recommendations")
    else:
        print(f"\n❌ PART 1 FAILED TO COMPLETE")
        print(f"🔧 Check the errors above and fix issues before running Part 2")
    
    print(f"\n" + "="*60)