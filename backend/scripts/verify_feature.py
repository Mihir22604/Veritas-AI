import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(backend_dir))

try:
    from app.services.ml_service import classifier
    print("Successfully imported classifier.")

    # Sample news for testing
    test_cases = [
        {
            "label": "Genuine",
            "text": "The government has announced a new initiative to improve public transportation by investing in electric buses and expanding subway lines."
        },
        {
            "label": "Fake",
            "text": "Scientists have discovered that eating chocolate every day can make you live forever and cure all diseases instantly."
        }
    ]

    for case in test_cases:
        print(f"\nTesting {case['label']} News:")
        print(f"Input Text: {case['text'][:100]}...")
        result = classifier.predict(case['text'])
        print(f"Result: {result}")
        
    print("\nVerification successful!")

except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
