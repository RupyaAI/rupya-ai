#!/usr/bin/env python3
"""
Heavy ML Libraries Installation Script for Cloud Server Testing
This script installs all the heavy machine learning libraries to test server capacity
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {command}: {e}")
        print(f"Output: {e.output}")
        return False

def install_packages():
    """Install all required packages"""
    print("🚀 Starting Heavy ML Libraries Installation for Server Load Testing...")
    print("="*70)
    
    # Core ML and Deep Learning Libraries
    packages = [
        # PyTorch ecosystem (CPU version for compatibility)
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        
        # TensorFlow ecosystem
        "tensorflow>=2.13.0",
        "tensorflow-hub",
        "tensorflow-datasets",
        
        # Hugging Face ecosystem
        "transformers[torch]",
        "datasets",
        "tokenizers",
        "accelerate",
        "diffusers",
        "sentence-transformers",
        
        # Computer Vision and OCR
        "easyocr",
        "paddlepaddle",
        "paddleocr",
        "opencv-python",
        "opencv-contrib-python",
        "pytesseract",
        
        # Image Processing
        "pillow",
        "scikit-image",
        "imageio",
        "albumentations",
        
        # Scientific Computing
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        
        # Deep Learning utilities
        "onnx",
        "onnxruntime",
        "torchmetrics",
        "lightning",
        
        # Audio processing (additional load testing)
        "librosa",
        "soundfile",
        
        # NLP Libraries
        "spacy",
        "nltk",
        "gensim",
        
        # Additional heavy libraries for stress testing
        "xgboost",
        "lightgbm",
        "catboost",
        "dask[complete]",
        "joblib",
        
        # Utility libraries
        "tqdm",
        "requests",
        "psutil",
        "GPUtil"
    ]
    
    failed_packages = []
    successful_packages = []
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        command = f"{sys.executable} -m pip install {package}"
        
        if run_command(command):
            successful_packages.append(package)
        else:
            failed_packages.append(package)
    
    # Install additional spacy models
    print("\n📚 Installing Spacy language models...")
    spacy_models = ["en_core_web_sm", "en_core_web_md"]
    for model in spacy_models:
        command = f"{sys.executable} -m spacy download {model}"
        run_command(command)
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 INSTALLATION SUMMARY")
    print("="*70)
    print(f"✅ Successfully installed: {len(successful_packages)} packages")
    print(f"❌ Failed to install: {len(failed_packages)} packages")
    
    if successful_packages:
        print("\n✅ Successful installations:")
        for pkg in successful_packages:
            print(f"  - {pkg}")
    
    if failed_packages:
        print("\n❌ Failed installations:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
    
    # System info
    print("\n🖥️  SYSTEM INFORMATION")
    print("="*30)
    try:
        import psutil
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    except:
        print("Could not get system information")
    
    # Test GPU availability
    print(f"\n🔥 GPU TESTING")
    print("="*20)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA not available (CPU mode)")
    except:
        print("Could not test CUDA")
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU devices: {len(gpus)}")
        else:
            print("❌ No TensorFlow GPU devices found")
    except:
        print("Could not test TensorFlow GPU")

if __name__ == "__main__":
    print("🧪 CLOUD SERVER LOAD TESTING - ML LIBRARIES INSTALLATION")
    print("This script will install heavy ML libraries to test your server capacity")
    print("Estimated installation time: 10-30 minutes depending on server specs")
    
    user_input = input("\nProceed with installation? (y/n): ")
    if user_input.lower() in ['y', 'yes']:
        install_packages()
        print("\n🎉 Installation complete! Ready to run advanced_ocr.py")
    else:
        print("Installation cancelled.")