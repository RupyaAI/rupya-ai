#!/usr/bin/env python3
"""
RupyaAI Cloud VPS Testing - PART 2
Comprehensive System Analysis and Final Deployment Recommendation
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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def safe_import(module_name: str, required: bool = False):
    """Safe import with error handling"""
    try:
        return __import__(module_name)
    except Exception as e:
        if required:
            print(f"⚠️  {module_name} not available: {e}")
        return None

# Load libraries for system analysis
numpy = safe_import('numpy')
psutil = safe_import('psutil')

print("🔬 RupyaAI System Analysis - PART 2")

def load_part1_results():
    """Load Part 1 results for analysis"""
    try:
        if not os.path.exists('rupyai_part1_results.json'):
            print("❌ Part 1 results not found! Please run Part 1 first.")
            return None
            
        with open('rupyai_part1_results.json', 'r') as f:
            results = json.load(f)
            
        print("✅ Part 1 results loaded successfully")
        return results
        
    except Exception as e:
        print(f"❌ Could not load Part 1 results: {e}")
        return None

def run_comprehensive_system_tests():
    """Run comprehensive system tests for RupyaAI deployment"""
    print("\n🧪 COMPREHENSIVE RUPYAAI SYSTEM TESTS")
    print("=" * 60)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'overall_status': 'UNKNOWN',
        'recommendations': [],
        'hostinger_verdict': 'PENDING'
    }
    
    # Test 1: Memory Analysis
    try:
        if psutil:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent
            
            # RupyaAI specific memory requirements
            min_required = 7.5  # For 8GB VPS
            optimal_required = 6.0  # For smooth operation
            
            if total_gb >= min_required:
                if available_gb >= optimal_required:
                    memory_status = 'EXCELLENT'
                    memory_score = 100
                else:
                    memory_status = 'GOOD'
                    memory_score = 85
            else:
                memory_status = 'INSUFFICIENT'
                memory_score = 40
                test_results['recommendations'].append("Upgrade to 8GB+ RAM VPS")
            
            test_results['tests']['memory'] = {
                'total_gb': round(total_gb, 2),
                'available_gb': round(available_gb, 2),
                'used_percent': round(used_percent, 1),
                'status': memory_status,
                'score': memory_score,
                'requirement': 'Hostinger KVM 2: 8GB RAM',
                'verdict': '✅ PERFECT' if memory_status == 'EXCELLENT' else '✅ GOOD' if memory_status == 'GOOD' else '❌ UPGRADE NEEDED'
            }
            
            print(f"💾 Memory Analysis: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
            print(f"   Status: {test_results['tests']['memory']['verdict']}")
            
        else:
            test_results['tests']['memory'] = {'status': 'SKIP', 'score': 0}
            print("💾 Memory Test: ⏭️  SKIPPED (psutil not available)")
            
    except Exception as e:
        test_results['tests']['memory'] = {'status': 'ERROR', 'score': 0, 'error': str(e)}
        print(f"💾 Memory Test: ❌ ERROR - {e}")
    
    # Test 2: CPU Performance Analysis
    try:
        if psutil:
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Test CPU under load
            print("🖥️  Testing CPU performance...")
            start_time = time.time()
            
            # CPU stress test
            if numpy:
                # Matrix multiplication test
                size = 800
                a = numpy.random.rand(size, size)
                b = numpy.random.rand(size, size)
                c = numpy.dot(a, b)
                del a, b, c
            else:
                # Basic CPU test without numpy
                result = sum(i * i for i in range(100000))
            
            cpu_test_time = time.time() - start_time
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # RupyaAI CPU scoring
            if cpu_count >= 2:
                if cpu_test_time < 3.0:  # Fast processing
                    cpu_status = 'EXCELLENT'
                    cpu_score = 100
                elif cpu_test_time < 6.0:  # Adequate processing
                    cpu_status = 'GOOD'
                    cpu_score = 85
                else:  # Slow but usable
                    cpu_status = 'ADEQUATE'
                    cpu_score = 70
            else:
                cpu_status = 'INSUFFICIENT'
                cpu_score = 40
                test_results['recommendations'].append("Upgrade to 2+ vCPU cores")
            
            test_results['tests']['cpu'] = {
                'core_count': cpu_count,
                'test_time_seconds': round(cpu_test_time, 2),
                'current_usage_percent': round(cpu_usage, 1),
                'frequency_mhz': round(cpu_freq.current, 0) if cpu_freq else 'Unknown',
                'status': cpu_status,
                'score': cpu_score,
                'requirement': 'Hostinger KVM 2: 2 shared vCPU',
                'verdict': '✅ EXCELLENT' if cpu_status == 'EXCELLENT' else '✅ GOOD' if cpu_status == 'GOOD' else '⚠️  ADEQUATE' if cpu_status == 'ADEQUATE' else '❌ UPGRADE NEEDED'
            }
            
            print(f"🖥️  CPU Analysis: {cpu_count} cores, {cpu_test_time:.2f}s test time")
            print(f"   Status: {test_results['tests']['cpu']['verdict']}")
            
        else:
            test_results['tests']['cpu'] = {'status': 'SKIP', 'score': 0}
            print("🖥️  CPU Test: ⏭️  SKIPPED (psutil not available)")
            
    except Exception as e:
        test_results['tests']['cpu'] = {'status': 'ERROR', 'score': 0, 'error': str(e)}
        print(f"🖥️  CPU Test: ❌ ERROR - {e}")
    
    # Test 3: Storage Analysis
    try:
        if psutil:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            used_gb = disk.used / (1024**3)
            
            # RupyaAI storage requirements
            min_free = 10.0  # Minimum for operation
            recommended_free = 25.0  # Recommended for production
            
            if free_gb >= recommended_free:
                storage_status = 'EXCELLENT'
                storage_score = 100
            elif free_gb >= min_free:
                storage_status = 'GOOD'
                storage_score = 80
            else:
                storage_status = 'LOW'
                storage_score = 50
                test_results['recommendations'].append("Free up disk space or upgrade storage")
            
            test_results['tests']['storage'] = {
                'total_gb': round(total_gb, 1),
                'free_gb': round(free_gb, 1),
                'used_gb': round(used_gb, 1),
                'free_percent': round((free_gb / total_gb) * 100, 1),
                'status': storage_status,
                'score': storage_score,
                'requirement': 'Hostinger KVM 2: 100GB NVMe SSD',
                'verdict': '✅ EXCELLENT' if storage_status == 'EXCELLENT' else '✅ GOOD' if storage_status == 'GOOD' else '⚠️  LOW SPACE'
            }
            
            print(f"💿 Storage Analysis: {free_gb:.1f}GB free of {total_gb:.1f}GB total")
            print(f"   Status: {test_results['tests']['storage']['verdict']}")
            
        else:
            test_results['tests']['storage'] = {'status': 'SKIP', 'score': 0}
            print("💿 Storage Test: ⏭️  SKIPPED (psutil not available)")
            
    except Exception as e:
        test_results['tests']['storage'] = {'status': 'ERROR', 'score': 0, 'error': str(e)}
        print(f"💿 Storage Test: ❌ ERROR - {e}")
    
    # Test 4: Network Connectivity
    try:
        import urllib.request
        import socket
        
        print("🌐 Testing network connectivity...")
        
        # Test multiple endpoints
        endpoints = [
            ('https://httpbin.org/status/200', 'General Internet'),
            ('https://huggingface.co', 'Hugging Face Models'),
            ('https://github.com', 'GitHub'),
        ]
        
        network_results = []
        total_response_time = 0
        
        for url, name in endpoints:
            try:
                start_time = time.time()
                response = urllib.request.urlopen(url, timeout=10)
                response_time = time.time() - start_time
                total_response_time += response_time
                
                network_results.append({
                    'endpoint': name,
                    'status': 'SUCCESS',
                    'response_time': round(response_time, 2)
                })
                
            except Exception as e:
                network_results.append({
                    'endpoint': name,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        successful_tests = len([r for r in network_results if r['status'] == 'SUCCESS'])
        avg_response_time = total_response_time / max(successful_tests, 1)
        
        if successful_tests == len(endpoints):
            if avg_response_time < 2.0:
                network_status = 'EXCELLENT'
                network_score = 100
            else:
                network_status = 'GOOD'
                network_score = 85
        else:
            network_status = 'POOR'
            network_score = 40
            test_results['recommendations'].append("Check network connectivity")
        
        test_results['tests']['network'] = {
            'successful_connections': successful_tests,
            'total_tests': len(endpoints),
            'average_response_time': round(avg_response_time, 2),
            'results': network_results,
            'status': network_status,
            'score': network_score,
            'requirement': 'Stable internet for model downloads',
            'verdict': '✅ EXCELLENT' if network_status == 'EXCELLENT' else '✅ GOOD' if network_status == 'GOOD' else '❌ ISSUES'
        }
        
        print(f"🌐 Network Analysis: {successful_tests}/{len(endpoints)} connections successful")
        print(f"   Status: {test_results['tests']['network']['verdict']}")
        
    except Exception as e:
        test_results['tests']['network'] = {'status': 'ERROR', 'score': 0, 'error': str(e)}
        print(f"🌐 Network Test: ❌ ERROR - {e}")
    
    # Test 5: Python Environment
    python_version = sys.version_info
    
    if python_version >= (3, 9):
        python_status = 'EXCELLENT'
        python_score = 100
    elif python_version >= (3, 8):
        python_status = 'GOOD'
        python_score = 90
    else:
        python_status = 'OUTDATED'
        python_score = 50
        test_results['recommendations'].append("Upgrade to Python 3.8+")
    
    test_results['tests']['python'] = {
        'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        'status': python_status,
        'score': python_score,
        'requirement': 'Python 3.8+ recommended',
        'verdict': '✅ EXCELLENT' if python_status == 'EXCELLENT' else '✅ GOOD' if python_status == 'GOOD' else '⚠️  OUTDATED'
    }
    
    print(f"🐍 Python Analysis: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"   Status: {test_results['tests']['python']['verdict']}")
    
    return test_results

def analyze_part1_ocr_performance(part1_results):
    """Analyze OCR performance from Part 1"""
    print(f"\n📊 OCR PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    ocr_analysis = {
        'models_tested': 0,
        'successful_extractions': 0,
        'best_model': 'None',
        'performance_score': 0,
        'financial_readiness': 'UNKNOWN'
    }
    
    if not part1_results:
        print("❌ No Part 1 results to analyze")
        return ocr_analysis
    
    # Analyze OCR results
    all_results = part1_results.get('all_ocr_results', {})
    model_performance = part1_results.get('model_performance', {})
    
    # Count successful OCR operations
    successful_ocr = [k for k, v in all_results.items() if v and v.strip() and 'Handwriting Analysis' not in v]
    ocr_analysis['models_tested'] = len(model_performance)
    ocr_analysis['successful_extractions'] = len(successful_ocr)
    ocr_analysis['best_model'] = part1_results.get('best_model', 'None')
    
    # Calculate performance score
    if part1_results.get('status') == 'SUCCESS':
        if 'trocr' in ocr_analysis['best_model'].lower():
            ocr_analysis['performance_score'] = 95  # TrOCR is best for handwriting
        elif len(successful_ocr) > 5:
            ocr_analysis['performance_score'] = 85  # Good alternative performance
        else:
            ocr_analysis['performance_score'] = 70  # Basic functionality
    else:
        ocr_analysis['performance_score'] = 30
    
    # Financial document readiness assessment
    if ocr_analysis['performance_score'] >= 90:
        ocr_analysis['financial_readiness'] = 'EXCELLENT'
    elif ocr_analysis['performance_score'] >= 75:
        ocr_analysis['financial_readiness'] = 'GOOD'
    elif ocr_analysis['performance_score'] >= 60:
        ocr_analysis['financial_readiness'] = 'ADEQUATE'
    else:
        ocr_analysis['financial_readiness'] = 'INSUFFICIENT'
    
    print(f"✍️  Models Tested: {ocr_analysis['models_tested']}")
    print(f"✅ Successful Extractions: {ocr_analysis['successful_extractions']}")
    print(f"🏆 Best Model: {ocr_analysis['best_model']}")
    print(f"📊 Performance Score: {ocr_analysis['performance_score']}/100")
    print(f"🏦 Financial Doc Readiness: {ocr_analysis['financial_readiness']}")
    
    return ocr_analysis

def calculate_rupyai_deployment_score(system_tests, ocr_analysis, part1_results):
    """Calculate comprehensive RupyaAI deployment readiness score"""
    print(f"\n🎯 RUPYAAI DEPLOYMENT READINESS CALCULATION")
    print("=" * 50)
    
    scores = {
        'memory': system_tests['tests'].get('memory', {}).get('score', 0),
        'cpu': system_tests['tests'].get('cpu', {}).get('score', 0),
        'storage': system_tests['tests'].get('storage', {}).get('score', 0),
        'network': system_tests['tests'].get('network', {}).get('score', 0),
        'python': system_tests['tests'].get('python', {}).get('score', 0),
        'ocr_performance': ocr_analysis.get('performance_score', 0)
    }
    
    # Weighted scoring (OCR is most important for RupyaAI)
    weights = {
        'memory': 0.25,      # Critical for ML models
        'cpu': 0.20,        # Important for processing
        'storage': 0.15,    # Needed for models and data
        'network': 0.10,    # For model downloads
        'python': 0.10,     # Environment requirement
        'ocr_performance': 0.20  # Core RupyaAI functionality
    }
    
    # Calculate weighted score
    total_score = 0
    for component, score in scores.items():
        weighted_score = score * weights[component]
        total_score += weighted_score
        print(f"   {component.upper()}: {score}/100 (weight: {weights[component]:.0%}) = {weighted_score:.1f}")
    
    final_score = round(total_score, 1)
    
    # Determine deployment readiness
    if final_score >= 90:
        readiness = 'READY_FOR_PRODUCTION'
        confidence = 'HIGH'
    elif final_score >= 80:
        readiness = 'READY_WITH_MINOR_ISSUES'
        confidence = 'GOOD'
    elif final_score >= 70:
        readiness = 'READY_WITH_WARNINGS'
        confidence = 'MODERATE'
    else:
        readiness = 'NOT_READY'
        confidence = 'LOW'
    
    print(f"\n🎯 FINAL SCORE: {final_score}/100")
    print(f"📊 READINESS: {readiness}")
    print(f"🎪 CONFIDENCE: {confidence}")
    
    return {
        'final_score': final_score,
        'component_scores': scores,
        'readiness': readiness,
        'confidence': confidence
    }

def generate_hostinger_recommendation(deployment_score, system_tests, ocr_analysis, part1_results):
    """Generate final Hostinger KVM 2 purchase recommendation"""
    print(f"\n💰 HOSTINGER KVM 2 PURCHASE RECOMMENDATION")
    print("=" * 50)
    
    recommendation = {
        'verdict': 'PENDING',
        'confidence': 'UNKNOWN',
        'reasons': [],
        'risks': [],
        'benefits': [],
        'action_plan': []
    }
    
    final_score = deployment_score['final_score']
    
    # Analyze key requirements for Hostinger KVM 2
    memory_status = system_tests['tests'].get('memory', {}).get('status', 'UNKNOWN')
    cpu_status = system_tests['tests'].get('cpu', {}).get('status', 'UNKNOWN')
    ocr_readiness = ocr_analysis.get('financial_readiness', 'UNKNOWN')
    
    # Decision logic
    if final_score >= 85 and ocr_readiness in ['EXCELLENT', 'GOOD']:
        recommendation['verdict'] = 'STRONG_BUY'
        recommendation['confidence'] = 'HIGH'
        recommendation['reasons'] = [
            f"Excellent deployment score: {final_score}/100",
            f"OCR performance: {ocr_readiness}",
            f"Memory: {memory_status}",
            f"CPU: {cpu_status}",
            "Perfect match for RupyaAI requirements"
        ]
        
    elif final_score >= 75 and ocr_readiness in ['EXCELLENT', 'GOOD', 'ADEQUATE']:
        recommendation['verdict'] = 'BUY'
        recommendation['confidence'] = 'GOOD'
        recommendation['reasons'] = [
            f"Good deployment score: {final_score}/100",
            f"OCR functionality working: {ocr_readiness}",
            "Meets minimum RupyaAI requirements",
            "Suitable for financial document processing"
        ]
        
    elif final_score >= 65:
        recommendation['verdict'] = 'BUY_WITH_CAUTION'
        recommendation['confidence'] = 'MODERATE'
        recommendation['reasons'] = [
            f"Moderate deployment score: {final_score}/100",
            "Basic requirements met",
            "May need optimization for production"
        ]
        
    else:
        recommendation['verdict'] = 'DO_NOT_BUY'
        recommendation['confidence'] = 'HIGH'
        recommendation['reasons'] = [
            f"Low deployment score: {final_score}/100",
            "Does not meet RupyaAI requirements",
            "Consider higher specification VPS"
        ]
    
    # Add benefits and risks
    if recommendation['verdict'] in ['STRONG_BUY', 'BUY']:
        recommendation['benefits'] = [
            "8GB RAM perfect for ML models",
            "2 vCPU adequate for OCR processing", 
            "100GB NVMe SSD for fast model loading",
            "Mumbai datacenter for low latency",
            "Cost-effective at ₹500/month",
            "Can handle financial document processing",
            "Room for growth and optimization"
        ]
        
        recommendation['risks'] = [
            "Shared CPU may have performance variations",
            "No GPU for advanced ML training",
            "Limited to CPU-based model inference"
        ]
        
        recommendation['action_plan'] = [
            "✅ PROCEED with Hostinger KVM 2 annual plan",
            "🔧 Install system libraries (build-essential, libgl1, etc.)",
            "💾 Add 4GB swap file for compilation",
            "🚀 Deploy RupyaAI OCR pipeline",
            "📊 Monitor performance in production",
            "🔄 Consider upgrade path if needed"
        ]
        
    elif recommendation['verdict'] == 'BUY_WITH_CAUTION':
        recommendation['benefits'] = [
            "Meets basic requirements",
            "Good starting point for MVP",
            "Can be upgraded later"
        ]
        
        recommendation['risks'] = [
            "May face performance limitations",
            "Possible memory constraints under load",
            "CPU bottlenecks during peak usage"
        ]
        
        recommendation['action_plan'] = [
            "⚠️  PROCEED with caution",
            "📊 Monitor performance closely",
            "🎯 Plan for early upgrade if needed",
            "🔧 Optimize code for efficiency"
        ]
        
    else:
        recommendation['risks'] = [
            "Insufficient resources for RupyaAI",
            "Poor performance expected",
            "May not handle production workloads"
        ]
        
        recommendation['action_plan'] = [
            "❌ DO NOT purchase Hostinger KVM 2",
            "🔄 Consider higher specification VPS",
            "💰 Look into Hostinger KVM 4 (16GB RAM)",
            "☁️  Alternative: AWS/GCP with GPU"
        ]
    
    return recommendation

def create_final_report(part1_results, system_tests, ocr_analysis, deployment_score, hostinger_rec):
    """Create comprehensive final report"""
    print(f"\n📄 GENERATING COMPREHENSIVE FINAL REPORT")
    print("=" * 45)
    
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'RupyaAI Hostinger KVM 2 Deployment Assessment',
            'version': '2.0',
            'test_duration_minutes': round((time.time() - start_time) / 60, 1) if 'start_time' in globals() else 'Unknown'
        },
        'executive_summary': {
            'deployment_score': deployment_score['final_score'],
            'readiness_status': deployment_score['readiness'],
            'hostinger_verdict': hostinger_rec['verdict'],
            'confidence_level': hostinger_rec['confidence'],
            'ocr_performance': ocr_analysis['financial_readiness']
        },
        'detailed_analysis': {
            'system_tests': system_tests,
            'ocr_performance': ocr_analysis,
            'part1_results_summary': {
                'status': part1_results.get('status') if part1_results else 'MISSING',
                'best_model': part1_results.get('best_model') if part1_results else 'None',
                'processing_time': part1_results.get('system_stats', {}).get('total_time') if part1_results else 0
            }
        },
        'deployment_readiness': deployment_score,
        'hostinger_recommendation': hostinger_rec,
        'next_steps': hostinger_rec['action_plan']
    }
    
    # Save comprehensive report
    try:
        with open('rupyai_final_assessment.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        print("✅ Comprehensive report saved: rupyai_final_assessment.json")
    except Exception as e:
        print(f"⚠️  Could not save comprehensive report: {e}")
    
    return report

def display_final_verdict(hostinger_rec, deployment_score):
    """Display the final verdict in a clear, actionable format"""
    print(f"\n" + "=" * 70)
    print(f"🎯 FINAL RUPYAAI HOSTINGER KVM 2 VERDICT")
    print(f"=" * 70)
    
    verdict = hostinger_rec['verdict']
    score = deployment_score['final_score']
    confidence = hostinger_rec['confidence']
    
    # Main verdict
    if verdict == 'STRONG_BUY':
        print(f"🟢 VERDICT: ✅ STRONG BUY - PROCEED IMMEDIATELY")
        print(f"🎪 Confidence: {confidence} ({score}/100)")
        print(f"💰 Recommendation: Purchase Hostinger KVM 2 annual plan NOW")
        
    elif verdict == 'BUY':
        print(f"🟡 VERDICT: ✅ BUY - GOOD CHOICE")
        print(f"🎪 Confidence: {confidence} ({score}/100)")
        print(f"💰 Recommendation: Purchase Hostinger KVM 2 annual plan")
        
    elif verdict == 'BUY_WITH_CAUTION':
        print(f"🟠 VERDICT: ⚠️  BUY WITH CAUTION")
        print(f"🎪 Confidence: {confidence} ({score}/100)")
        print(f"💰 Recommendation: Consider purchase but monitor closely")
        
    else:
        print(f"🔴 VERDICT: ❌ DO NOT BUY")
        print(f"🎪 Confidence: {confidence} ({score}/100)")
        print(f"💰 Recommendation: Look for higher specification VPS")
    
    print(f"\n🏦 RUPYAAI SUITABILITY:")
    for reason in hostinger_rec['reasons']:
        print(f"   • {reason}")
    
    if hostinger_rec['benefits']:
        print(f"\n✅ BENEFITS:")
        for benefit in hostinger_rec['benefits']:
            print(f"   • {benefit}")
    
    if hostinger_rec['risks']:
        print(f"\n⚠️  RISKS TO CONSIDER:")
        for risk in hostinger_rec['risks']:
            print(f"   • {risk}")
    
    print(f"\n🚀 RECOMMENDED ACTION PLAN:")
    for i, action in enumerate(hostinger_rec['action_plan'], 1):
        print(f"   {i}. {action}")
    
    print(f"\n" + "=" * 70)

def main_part2():
    """Main function for Part 2 analysis"""
    global start_time
    start_time = time.time()
    
    print("🏦 RUPYAAI CLOUD VPS TESTING - PART 2")
    print("=" * 60)
    print("Comprehensive System Analysis & Final Hostinger Recommendation")
    print("")
    
    try:
        # Load Part 1 results
        print("📊 Loading Part 1 OCR test results...")
        part1_results = load_part1_results()
        
        if not part1_results:
            print("❌ Cannot proceed without Part 1 results")
            return None
        
        # Run comprehensive system tests
        system_tests = run_comprehensive_system_tests()
        
        # Analyze OCR performance from Part 1
        ocr_analysis = analyze_part1_ocr_performance(part1_results)
        
        # Calculate deployment readiness score
        deployment_score = calculate_rupyai_deployment_score(system_tests, ocr_analysis, part1_results)
        
        # Generate Hostinger recommendation
        hostinger_rec = generate_hostinger_recommendation(deployment_score, system_tests, ocr_analysis, part1_results)
        
        # Create final comprehensive report
        final_report = create_final_report(part1_results, system_tests, ocr_analysis, deployment_score, hostinger_rec)
        
        # Display final verdict
        display_final_verdict(hostinger_rec, deployment_score)
        
        print(f"\n📁 All reports saved:")
        print(f"   • rupyai_final_assessment.json (comprehensive)")
        print(f"   • rupyai_part1_results.json (OCR tests)")
        
        return final_report
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Part 2 interrupted by user")
        return None
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN PART 2:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        try:
            gc.collect()
        except:
            pass
        
        print(f"\n🧹 Part 2 analysis completed")

if __name__ == "__main__":
    print("🔬 Starting RupyaAI Part 2 System Analysis...")
    result = main_part2()
    
    if result:
        print("\n🎉 PART 2 ANALYSIS COMPLETE!")
        print("📊 Check rupyai_final_assessment.json for detailed report")
        
        # Quick summary
        exec_summary = result.get('executive_summary', {})
        print(f"\n📋 QUICK SUMMARY:")
        print(f"   🎯 Deployment Score: {exec_summary.get('deployment_score', 'N/A')}/100")
        print(f"   🎪 System Status: {exec_summary.get('readiness_status', 'UNKNOWN')}")
        print(f"   💰 Hostinger Verdict: {exec_summary.get('hostinger_verdict', 'PENDING')}")
        print(f"   🏦 OCR Performance: {exec_summary.get('ocr_performance', 'UNKNOWN')}")
        
        # Final recommendation based on Part 1 results
        if exec_summary.get('hostinger_verdict') == 'STRONG_BUY':
            print(f"\n🚀 FINAL RECOMMENDATION: GO AHEAD WITH HOSTINGER KVM 2!")
            print(f"   Your system is perfectly suited for RupyaAI deployment")
            print(f"   Expected performance: Excellent financial document processing")
            
        elif exec_summary.get('hostinger_verdict') == 'BUY':
            print(f"\n✅ FINAL RECOMMENDATION: HOSTINGER KVM 2 IS A GOOD CHOICE")
            print(f"   Your system meets RupyaAI requirements well")
            print(f"   Expected performance: Good financial document processing")
            
        else:
            print(f"\n⚠️  FINAL RECOMMENDATION: REVIEW BEFORE PURCHASE")
            print(f"   Check the detailed report for specific concerns")
    
    else:
        print("\n❌ Part 2 analysis failed - check error messages above")