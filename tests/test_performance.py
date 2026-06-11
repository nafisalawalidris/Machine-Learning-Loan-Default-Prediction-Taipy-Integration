# tests/test_performance.py
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import preprocess_input


def test_prediction_speed():
    """Test that preprocessing completes within acceptable time limits"""
    
    test_input = {
        "Loan Amount": 15000000.0,
        "Funded Amount": 15000000.0,
        "Funded Amount Investor": 14500000.0,
        "Interest Rate": 12.5,
        "Term": 36,
        "Annual Income": 60000000.0,
        "Grade": "B",
        "Sub Grade": "B3",
        "Home Ownership": "MORTGAGE",
        "Employment Duration": "5 years",
        "Verification Status": "Verified",
        "Purpose": "debt_consolidation",
        "Batch Enrolled": "BAT901476",
        "Delinquency - two years": 0,
        "Inquires - six months": 1,
        "Open Accounts": 10,
        "Public Record": 0,
        "Revolving Balance": 8000000.0,
        "Revolving Utilities": 55.0,
        "Total Accounts": 25,
        "Total Revolving Credit Limit": 20000000.0,
        "Total Current Balance": 50000000.0,
        "Total Collection Amount": 0.0,
    }
    
    # Warm-up
    for _ in range(10):
        preprocess_input(test_input, "Logistic Regression")
    
    # Measure time for 100 iterations
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        X = preprocess_input(test_input, "Logistic Regression")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    print(f"\nPerformance Test Results:")
    print(f"  Total time for {iterations} iterations: {total_time:.3f} seconds")
    print(f"  Average preprocessing time: {avg_time_ms:.2f} ms")
    
    # Assert that average preprocessing time is under 100ms
    assert avg_time_ms < 100, f"Preprocessing too slow: {avg_time_ms:.2f}ms"
    
    print("Performance test passed!")


if __name__ == "__main__":
    test_prediction_speed()