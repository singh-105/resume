import os
from src.model_training import train_model

def load_jds():
    jd_dir = 'data/job_descriptions'
    jds = {}
    for filename in os.listdir(jd_dir):
        if filename.endswith('.txt'):
            domain = filename.replace('.txt', '')
            with open(os.path.join(jd_dir, filename), 'r', encoding='utf-8') as f:
                jds[domain] = f.read()
    return jds

def main():
    print("Loading Job Descriptions...")
    jds = load_jds()
    
    if not jds:
        print("No JDs found in data/job_descriptions!")
        return

    print(f"Found {len(jds)} domains: {list(jds.keys())}")
    
    # Train models for all domains
    for domain, text in jds.items():
        print(f"\n--- Training for {domain} ---")
        train_model(domain, text, jds)
    
    print("\nAll models trained and saved!")

if __name__ == "__main__":
    main()
