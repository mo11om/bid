import subprocess
import json
import datetime
import os

# Define the HCP situations we want to test
scenarios = [
    {"name": "Part Score (Marginal)", "n_hcp": "12 14", "s_hcp": "6 9"},
    {"name": "Game Forcing (Standard)", "n_hcp": "15 17", "s_hcp": "10 12"},
    {"name": "Small Slam (Strong)", "n_hcp": "18 20", "s_hcp": "13 15"},
    {"name": "Grand Slam (Monster)", "n_hcp": "22 24", "s_hcp": "15 17"},
]

def run_scenarios():
    report_lines = [
        "# Bridge AI: Detailed HCP Scenario Report",
        f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "This report evaluates the LLM's bidding accuracy and 1-IMP rollout quality across distinct point-count situations.",
        ""
    ]

    for scene in scenarios:
        print(f"[*] Running evaluation for {scene['name']}...")
        json_file = f"{scene['name'].replace(' ', '_').lower()}.json"
        
        # Execute the underlying eval harness
        subprocess.run([
            "python", "run_eval.py", 
            "--mode", "scenario_test", 
            "--count", "25",
            "--north-hcp", *scene['n_hcp'].split(),
            "--south-hcp", *scene['s_hcp'].split(),
            "--results-json", json_file
        ], check=True)

        # Parse the JSON results for the detailed report
        with open(json_file, 'r') as f:
            results = json.load(f)
            
        accuracy = results.get('exact_accuracy', 0) * 100
        rollout_quality = results.get('1_imp_quality', 0) * 100
        
        report_lines.extend([
            f"## {scene['name']}",
            f"- **North HCP Range:** {scene['n_hcp'].replace(' ', ' to ')}",
            f"- **South HCP Range:** {scene['s_hcp'].replace(' ', ' to ')}",
            f"- **Exact Match Accuracy:** {accuracy:.1f}%",
            f"- **1-IMP Rollout Quality (Acceptable Contracts):** {rollout_quality:.1f}%",
            ""
        ])

    # Generate the final report
    report_path = "hcp_scenario_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"[+] Complete! Detailed report generated at: {report_path}")

if __name__ == "__main__":
    run_scenarios()

