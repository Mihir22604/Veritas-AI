import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(backend_dir))

try:
    from app.services.ml_service import classifier
    print("Successfully imported classifier.")

    # Domain-specific test cases (Political statements similar to LIAR dataset)
    test_cases = [
        {
            "label": "Genuine (Political)",
            "text": "The United States has a higher incarceration rate than any other country in the world."
        },
        {
            "label": "Fake (Political)",
            "text": "Barack Obama was not born in the United States and his birth certificate is a complete forgery."
        }
    ]

    for case in test_cases:
        print(f"\nTesting {case['label']} News:")
        print(f"Input Text: {case['text']}")
        result = classifier.predict(case['text'])
        print(f"Result: {result}")
        
    print("\nVerification successful!")

except Exception as e:
    print(f"Verification failed: {e}")
    sys.exit(1)
