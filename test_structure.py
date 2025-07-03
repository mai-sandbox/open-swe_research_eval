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
        assert hasattr(research_assistant, 'request_human_approval'), "Missing request_human_approval tool"
        print("✅ SUCCESS: Code structure is valid")
        return True
    except Exception as e:
        print(f"❌ FAILED: Structure error - {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Research Assistant Structure...")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    # Test structure
    structure_success = test_structure()
    
    print("=" * 50)
    if import_success and structure_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The research assistant code is structurally sound")
        print("✅ All required components are present")
        print("✅ LangGraph compilation works correctly")
    else:
        print("❌ SOME TESTS FAILED!")

if __name__ == "__main__":
    main()



