# tests/test_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

# Import from your actual app
from app import (
    validate_inputs, 
    preprocess_input, 
    encode_employment_duration,
    GRADE_ORDER,
    AIInsightsEngine
)


class TestValidation:
    """Test input validation functions"""
    
    def test_valid_inputs(self):
        """Test that valid inputs pass validation"""
        class MockState:
            loan_amount = 15000000.0
            funded_amount = 15000000.0
            funded_amount_inv = 14500000.0
            interest_rate = 12.5
            term = 36
            annual_income = 60000000.0
            revolving_util = 55.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is True
        assert message == ""
    
    def test_invalid_loan_amount_zero(self):
        """Test that zero loan amount is rejected"""
        class MockState:
            loan_amount = 0
            interest_rate = 12.5
            annual_income = 60000000.0
            revolving_util = 55.0
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
        assert "greater than zero" in message.lower()
    
    def test_excessive_loan_amount(self):
        """Test that excessive loan amount is rejected"""
        class MockState:
            loan_amount = 200_000_000_000
            interest_rate = 12.5
            annual_income = 60000000.0
            revolving_util = 55.0
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
        assert "exceeds" in message.lower() or "100 billion" in message.lower()
    
    def test_invalid_interest_rate_high(self):
        """Test that out-of-range interest rates are rejected"""
        class MockState:
            loan_amount = 15000000.0
            interest_rate = 75
            annual_income = 60000000.0
            revolving_util = 55.0
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
        assert "between" in message.lower()
    
    def test_negative_interest_rate(self):
        """Test that negative interest rates are rejected"""
        class MockState:
            loan_amount = 15000000.0
            interest_rate = -5
            annual_income = 60000000.0
            revolving_util = 55.0
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
    
    def test_invalid_annual_income_zero(self):
        """Test that zero income is rejected"""
        class MockState:
            loan_amount = 15000000.0
            interest_rate = 12.5
            annual_income = 0
            revolving_util = 55.0
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
        assert "greater than zero" in message.lower()
    
    def test_invalid_revolving_utilisation_high(self):
        """Test that revolving utilisation above 100 per cent is rejected"""
        class MockState:
            loan_amount = 15000000.0
            interest_rate = 12.5
            annual_income = 60000000.0
            revolving_util = 150
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False
    
    def test_negative_revolving_utilisation(self):
        """Test that negative revolving utilisation is rejected"""
        class MockState:
            loan_amount = 15000000.0
            interest_rate = 12.5
            annual_income = 60000000.0
            revolving_util = -10
            funded_amount = 15000000.0
        
        state = MockState()
        is_valid, message = validate_inputs(state)
        assert is_valid is False


class TestEmploymentDurationEncoding:
    """Test employment duration encoding function"""
    
    def test_years_format(self):
        """Test '5 years' format conversion"""
        result = encode_employment_duration("5 years")
        assert result == 5
    
    def test_year_format(self):
        """Test '1 year' singular format"""
        result = encode_employment_duration("1 year")
        assert result == 1
    
    def test_less_than_one_year(self):
        """Test 'less than 1 year' format - returns the number found"""
        result = encode_employment_duration("< 1 year")
        assert result == 1
    
    def test_ten_plus_years(self):
        """Test '10+ years' format converts to 15"""
        result = encode_employment_duration("10+ years")
        assert result == 15
    
    def test_month_format(self):
        """Test months conversion to years"""
        result = encode_employment_duration("6 months")
        assert result == 0.5
    
    def test_empty_string(self):
        """Test empty string handling"""
        result = encode_employment_duration("")
        assert result == 0
    
    def test_nan_value(self):
        """Test NaN value handling"""
        result = encode_employment_duration(pd.NA)
        assert result == 0
    
    def test_no_numbers(self):
        """Test string with no numbers"""
        result = encode_employment_duration("unknown")
        assert result == 0


class TestGradeOrdering:
    """Test grade ordinal encoding"""
    
    def test_grade_a(self):
        """Test Grade A maps to 1"""
        assert GRADE_ORDER["A"] == 1
    
    def test_grade_b(self):
        """Test Grade B maps to 2"""
        assert GRADE_ORDER["B"] == 2
    
    def test_grade_g(self):
        """Test Grade G maps to 7"""
        assert GRADE_ORDER["G"] == 7
    
    def test_all_grades_present(self):
        """Test all expected grades are present"""
        expected_grades = ["A", "B", "C", "D", "E", "F", "G"]
        for grade in expected_grades:
            assert grade in GRADE_ORDER


class TestAIInsightsEngine:
    """Test AI insights generation - Matching actual app implementation"""
    
    def test_approval_decision_exists(self):
        """Test that approval_decision method exists and returns tuple"""
        result = AIInsightsEngine.approval_decision(15)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (suggestion, message, css_class)
    
    def test_low_risk_approval(self):
        """Test low risk approval suggestion"""
        suggestion, message, css_class = AIInsightsEngine.approval_decision(15)
        assert suggestion == "STRONG APPROVE"
        assert "expedited" in message.lower() or "low risk" in message.lower()
        assert css_class == "decision-green"
    
    def test_medium_risk_approval(self):
        """Test medium risk approval suggestion"""
        suggestion, message, css_class = AIInsightsEngine.approval_decision(40)
        assert suggestion == "CONSIDER"
        assert "review" in message.lower() or "documentation" in message.lower()
        assert css_class == "decision-amber"
    
    def test_high_risk_approval(self):
        """Test high risk approval suggestion"""
        suggestion, message, css_class = AIInsightsEngine.approval_decision(60)
        assert suggestion == "CAUTION"
        assert "high risk" in message.lower() or "reduced amount" in message.lower()
        assert css_class == "decision-orange"
    
    def test_critical_risk_approval(self):
        """Test critical risk rejection"""
        suggestion, message, css_class = AIInsightsEngine.approval_decision(85)
        assert suggestion == "REJECT"
        assert "exceeds" in message.lower() or "decline" in message.lower()
        assert css_class == "decision-red"
    
    def test_risk_band_exists(self):
        """Test that risk_band method returns appropriate risk level"""
        risk_level, css_class = AIInsightsEngine.risk_band(15)
        assert risk_level in ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]
        assert css_class in ["risk-low", "risk-moderate", "risk-elevated", "risk-high", "risk-critical"]
    
    def test_analyse_method_exists(self):
        """Test that analyse method returns three values"""
        result = AIInsightsEngine.analyse(
            loan_amount=15000000.0,
            annual_income=60000000.0,
            revolving_util=55.0,
            delinquency_2yr=0,
            inquiries_6mo=1,
            interest_rate=12.5,
            open_accounts=10,
            public_records=0,
            term=36
        )
        assert isinstance(result, tuple)
        assert len(result) == 3  # (factors_text, recommendations_text, credit_score)
    
    def test_analyse_high_risk_factors(self):
        """Test analyse identifies high risk factors correctly"""
        factors_text, recommendations_text, credit_score = AIInsightsEngine.analyse(
            loan_amount=40000000.0,      # High loan
            annual_income=20000000.0,     # Low income
            revolving_util=85.0,          # High utilisation
            delinquency_2yr=2,            # Multiple delinquencies
            inquiries_6mo=5,              # Many inquiries
            interest_rate=22.0,           # High rate
            open_accounts=1,              # Few accounts
            public_records=1,             # Public record
            term=60                       # Long term
        )
        # Should have many risk factors
        assert credit_score < 50  # Low credit score for high risk
        assert len(factors_text) > 20  # Non-empty factors
    
    def test_analyse_low_risk_factors(self):
        """Test analyse identifies low risk profile correctly"""
        factors_text, recommendations_text, credit_score = AIInsightsEngine.analyse(
            loan_amount=5000000.0,       # Small loan
            annual_income=100000000.0,    # High income
            revolving_util=15.0,          # Low utilisation
            delinquency_2yr=0,            # No delinquencies
            inquiries_6mo=0,              # No inquiries
            interest_rate=8.0,            # Low rate
            open_accounts=5,              # Several accounts
            public_records=0,             # No public records
            term=36                       # Standard term
        )
        # Should have few risk factors
        assert credit_score > 70  # High credit score for low risk


class TestPreprocessing:
    """Test preprocessing pipeline"""
    
    def test_preprocessing_returns_dataframe(self):
        """Test that preprocessing returns a DataFrame"""
        raw_input = {
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
        
        result = preprocess_input(raw_input, "Logistic Regression")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_preprocessing_handles_missing_fields(self):
        """Test preprocessing handles missing fields gracefully"""
        raw_input = {
            "Loan Amount": 15000000.0,
            "Interest Rate": 12.5,
            "Annual Income": 60000000.0,
            "Grade": "B",
            "Sub Grade": "B3",
        }
        
        # Should not crash
        result = preprocess_input(raw_input, "Logistic Regression")
        assert isinstance(result, pd.DataFrame)