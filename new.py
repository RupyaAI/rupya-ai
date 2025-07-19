#!/usr/bin/env python3
"""
Ultra Fault-Tolerant OCR System
This system handles ALL possible failures gracefully and continues processing
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

# Suppress ALL warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
print("📦 Loading libraries safely...")

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
        from PIL import Image
        PIL_Image = Image
    except:
        PIL = None
        PIL_Image = None

# Plotting (non-essential)
matplotlib = safe_import('matplotlib')
if matplotlib:
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')  # Non-interactive backend
    except:
        matplotlib = None

# ML libraries
torch = safe_import('torch')
tensorflow = safe_import('tensorflow')
if tensorflow:
    try:
        tf = tensorflow
        # Disable GPU for stability
        tf.config.experimental.set_visible_devices([], 'GPU')
    except:
        tensorflow = None

# Transformers
transformers = safe_import('transformers')

# OCR libraries
easyocr = safe_import('easyocr')
paddleocr = safe_import('paddleocr')

print("✅ Library loading completed")

class BulletproofSystemMonitor:
    """System monitor that works in any environment"""
    
    def __init__(self):
        self.start_time = time.time()
        self.has_psutil = psutil is not None
        self.logs = []
        
    def log(self, message: str):
        """Log message with timestamp"""
        elapsed = time.time() - self.start_time
        log_entry = f"[{elapsed:.1f}s] {message}"
        print(log_entry)
        self.logs.append(log_entry)
        
        if self.has_psutil:
            try:
                memory = psutil.virtual_memory().used / (1024**3)
                cpu = psutil.cpu_percent(interval=0.1)
                print(f"     💾 {memory:.1f}GB | 🖥️  {cpu:.1f}%")
            except:
                pass

class UltraRobustImageLoader:
    """Image loader that tries every possible method"""
    
    def __init__(self):
        self.has_cv2 = cv2 is not None
        self.has_pil = PIL is not None
        self.has_numpy = numpy is not None
        
        print(f"🔧 Image Loader Status:")
        print(f"   OpenCV: {'✅' if self.has_cv2 else '❌'}")
        print(f"   PIL: {'✅' if self.has_pil else '❌'}")
        print(f"   NumPy: {'✅' if self.has_numpy else '❌'}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Try every possible way to load an image"""
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return None
        
        print(f"📸 Loading image: {image_path}")
        
        # Method 1: OpenCV
        if self.has_cv2:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    print(f"✅ Loaded with OpenCV: {img.shape}")
                    return img
            except Exception as e:
                print(f"⚠️  OpenCV failed: {e}")
        
        # Method 2: PIL
        if self.has_pil and self.has_numpy:
            try:
                pil_img = PIL_Image.open(image_path)
                img_array = np.array(pil_img)
                
                # Ensure 3 channels
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]  # Remove alpha
                
                # Convert RGB to BGR for consistency
                if len(img_array.shape) == 3:
                    img_array = img_array[:, :, ::-1]
                
                print(f"✅ Loaded with PIL: {img_array.shape}")
                return img_array
            except Exception as e:
                print(f"⚠️  PIL failed: {e}")
        
        # Method 3: Raw binary reading (last resort)
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
            print(f"⚠️  Loaded as raw binary: {len(data)} bytes")
            print("❌ Cannot process raw binary - need image processing library")
        except Exception as e:
            print(f"❌ Even raw reading failed: {e}")
        
        return None
    
    def preprocess_image(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Safely preprocess image with available tools"""
        if not self.has_numpy or image is None:
            return []
        
        processed = []
        
        try:
            # Original
            processed.append(("original", image.copy()))
            
            # Grayscale conversion
            if len(image.shape) == 3:
                # OpenCV method
                if self.has_cv2:
                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        processed.append(("grayscale_cv2", gray))
                    except:
                        pass
                
                # NumPy method (fallback)
                try:
                    gray_np = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                    processed.append(("grayscale_numpy", gray_np.astype(np.uint8)))
                except:
                    pass
            
            # Advanced preprocessing (only if OpenCV available)
            if self.has_cv2 and len(processed) > 1:
                try:
                    gray = processed[-1][1]  # Last grayscale version
                    
                    # Simple blur
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    processed.append(("blurred", blurred))
                    
                    # Threshold
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    processed.append(("threshold", thresh))
                    
                except Exception as e:
                    print(f"⚠️  Advanced preprocessing failed: {e}")
            
            print(f"✅ Created {len(processed)} image variants")
            return processed
            
        except Exception as e:
            print(f"⚠️  Preprocessing error: {e}")
            return [("original", image)] if image is not None else []

class BulletproofOCREngine:
    """OCR engine that gracefully handles any failure"""
    
    def __init__(self):
        self.monitor = BulletproofSystemMonitor()
        self.image_loader = UltraRobustImageLoader()
        self.ocr_models = {}
        self.device = 'cpu'  # Always use CPU for maximum compatibility
        
        self.monitor.log("Initializing OCR Engine")
        self.setup_models()
        
    def setup_models(self):
        """Setup whatever OCR models are available"""
        self.monitor.log("Setting up OCR models")
        
        # Model 1: TrOCR (if available)
        self.setup_trocr()
        
        # Model 2: EasyOCR (if available)
        self.setup_easyocr()
        
        # Model 3: PaddleOCR (if available)
        self.setup_paddleocr()
        
        # Model 4: Basic fallback (always available if NumPy exists)
        self.setup_fallback()
        
        available_models = list(self.ocr_models.keys())
        self.monitor.log(f"Available models: {available_models}")
        
        if not self.ocr_models:
            print("❌ No OCR models available!")
        
    def setup_trocr(self):
        """Setup TrOCR if possible"""
        if not (transformers and torch):
            return
        
        try:
            self.monitor.log("Loading TrOCR...")
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Try handwritten model
            try:
                processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                model.eval()  # Set to evaluation mode
                
                self.ocr_models['trocr_handwritten'] = {
                    'processor': processor,
                    'model': model,
                    'type': 'trocr'
                }
                print("✅ TrOCR handwritten model loaded")
            except Exception as e:
                print(f"⚠️  TrOCR handwritten failed: {e}")
            
            # Try printed model
            try:
                processor_print = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                model_print = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
                model_print.eval()
                
                self.ocr_models['trocr_printed'] = {
                    'processor': processor_print,
                    'model': model_print,
                    'type': 'trocr'
                }
                print("✅ TrOCR printed model loaded")
            except Exception as e:
                print(f"⚠️  TrOCR printed failed: {e}")
                
        except Exception as e:
            print(f"⚠️  TrOCR setup completely failed: {e}")
    
    def setup_easyocr(self):
        """Setup EasyOCR if possible"""
        if not easyocr:
            return
        
        try:
            self.monitor.log("Loading EasyOCR...")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)  # Force CPU
            self.ocr_models['easyocr'] = {
                'reader': reader,
                'type': 'easyocr'
            }
            print("✅ EasyOCR loaded")
        except Exception as e:
            print(f"⚠️  EasyOCR setup failed: {e}")
    
    def setup_paddleocr(self):
        """Setup PaddleOCR if possible"""
        if not paddleocr:
            return
        
        try:
            self.monitor.log("Loading PaddleOCR...")
            from paddleocr import PaddleOCR
            
            # Force CPU and disable logging
            ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                show_log=False,
                use_gpu=False,
                cpu_threads=1
            )
            self.ocr_models['paddleocr'] = {
                'ocr': ocr,
                'type': 'paddleocr'
            }
            print("✅ PaddleOCR loaded")
        except Exception as e:
            print(f"⚠️  PaddleOCR setup failed: {e}")
    
    def setup_fallback(self):
        """Setup basic fallback OCR"""
        if numpy:
            self.ocr_models['fallback'] = {
                'type': 'fallback'
            }
            print("✅ Fallback OCR available")
    
    def process_with_trocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process with TrOCR"""
        try:
            processor = model_info['processor']
            model = model_info['model']
            
            # Convert to PIL Image
            if len(image.shape) == 3:
                # BGR to RGB
                rgb_image = image[:, :, ::-1]
                pil_image = PIL_Image.fromarray(rgb_image)
            else:
                pil_image = PIL_Image.fromarray(image).convert('RGB')
            
            # Process
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=128)
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"⚠️  TrOCR processing error: {e}")
            return ""
    
    def process_with_easyocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process with EasyOCR"""
        try:
            reader = model_info['reader']
            results = reader.readtext(image, detail=0, paragraph=False)
            return ' '.join(results)
        except Exception as e:
            print(f"⚠️  EasyOCR processing error: {e}")
            return ""
    
    def process_with_paddleocr(self, model_info: Dict, image: np.ndarray) -> str:
        """Process with PaddleOCR"""
        try:
            ocr = model_info['ocr']
            results = ocr.ocr(image, cls=True)
            
            if results and results[0]:
                texts = []
                for line in results[0]:
                    if len(line) >= 2 and line[1]:
                        texts.append(line[1][0])
                return ' '.join(texts)
            return ""
        except Exception as e:
            print(f"⚠️  PaddleOCR processing error: {e}")
            return ""
    
    def process_with_fallback(self, model_info: Dict, image: np.ndarray) -> str:
        """Basic fallback processing"""
        try:
            # Very basic character recognition attempt
            # This is just a placeholder - real implementation would be complex
            if numpy and len(image.shape) >= 2:
                height, width = image.shape[:2]
                pixel_count = numpy.prod(image.shape)
                avg_intensity = numpy.mean(image) if hasattr(numpy, 'mean') else 128
                
                # Generate a basic report
                return f"Image analyzed: {width}x{height} pixels, avg intensity: {avg_intensity:.1f}"
            return "Basic image analysis completed"
        except Exception as e:
            print(f"⚠️  Fallback processing error: {e}")
            return "Fallback analysis failed"
    
    def process_image_safely(self, image: np.ndarray, variant_name: str) -> Dict[str, str]:
        """Process image with all available models safely"""
        results = {}
        
        for model_name, model_info in self.ocr_models.items():
            try:
                print(f"   🔄 {model_name}...")
                
                model_type = model_info['type']
                
                if model_type == 'trocr':
                    result = self.process_with_trocr(model_info, image)
                elif model_type == 'easyocr':
                    result = self.process_with_easyocr(model_info, image)
                elif model_type == 'paddleocr':
                    result = self.process_with_paddleocr(model_info, image)
                elif model_type == 'fallback':
                    result = self.process_with_fallback(model_info, image)
                else:
                    continue
                
                if result and result.strip():
                    key = f"{model_name}_{variant_name}"
                    results[key] = result.strip()
                    print(f"   ✅ {model_name}: {result[:50]}...")
                
            except Exception as e:
                print(f"   ❌ {model_name}: {e}")
                continue
        
        return results
    
    def process_image_file(self, image_path: str) -> Dict:
        """Process an image file with comprehensive error handling"""
        self.monitor.log(f"Processing {image_path}")
        
        results = {
            'image_path': image_path,
            'best_result': 'No text detected',
            'all_results': {},
            'processing_log': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            # Load image
            image = self.image_loader.load_image(image_path)
            if image is None:
                results['errors'].append("Could not load image")
                return results
            
            # Preprocess image
            try:
                variants = self.image_loader.preprocess_image(image)
                if not variants:
                    results['errors'].append("Could not preprocess image")
                    return results
            except Exception as e:
                results['errors'].append(f"Preprocessing failed: {e}")
                variants = [("original", image)]
            
            # Process each variant with all models
            all_results = {}
            
            for variant_name, variant_image in variants:
                print(f"\n🔄 Processing {variant_name} variant:")
                
                try:
                    variant_results = self.process_image_safely(variant_image, variant_name)
                    all_results.update(variant_results)
                except Exception as e:
                    error_msg = f"Error processing {variant_name}: {e}"
                    results['errors'].append(error_msg)
                    print(f"❌ {error_msg}")
            
            # Select best result
            if all_results:
                # Simple scoring: prefer longer results from better models
                model_priority = {
                    'trocr_handwritten': 4,
                    'trocr_printed': 3,
                    'easyocr': 2,
                    'paddleocr': 1,
                    'fallback': 0
                }
                
                def score_result(key, text):
                    model_name = key.split('_')[0] + '_' + key.split('_')[1] if '_' in key else key.split('_')[0]
                    model_score = model_priority.get(model_name, 0)
                    length_score = len(text.strip())
                    return model_score * 1000 + length_score
                
                best_key = max(all_results.keys(), key=lambda k: score_result(k, all_results[k]))
                best_result = all_results[best_key]
                
                results.update({
                    'best_result': best_result,
                    'best_model': best_key,
                    'all_results': all_results
                })
                
                print(f"\n🎯 Best result from