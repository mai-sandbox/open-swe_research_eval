#!/usr/bin/env python3
"""
Test script to verify the research assistant structure and imports.
"""

def test_imports():
    """Test that all imports work correctly."""
    try:
        import research_assistant
        print("✅ SUCCESS: All imports work correctly")
        return True
    except Exception as e:
        print(f"❌ FAILED: Import error - {e}")
        return False

def test_structure():
    """Test that the code structure is valid."""
    try:
        import research_assistant
        # Check if key components exist
        assert hasattr(research_assistant, 'app'), "Missing compiled graph 'app'"
        assert hasattr(research_assistant, 'main'), "Missing main function"
        assert hasattr(research_assistant, 'web_search'), "Missing web_search tool"
        assert hasattr(research_assistant, 'document_lookup'), "Missing document_lookup tool"
        assert hasattr(research_assistant, 'calculate_stats'), "Missing calculate_stats tool"

