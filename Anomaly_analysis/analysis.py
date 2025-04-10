import csv
import pandas as pd
from time import sleep
from litellm import completion


API_KEY = " "  # Replace with your key
BASE_URL = " "
MODEL = "gpt-4o"


def get_completion(messages):
    response = completion(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
        custom_llm_provider="openai",
        temperature=0.7,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def analyze_anomalous_log(log_entry):
    messages = [
        {
            "role": "system",
            "content": "You are a cybersecurity analyst specializing in anomaly detection. "
                       "Your task is to analyze a given log entry that has been flagged as anomalous. "
                       "Identify the **top 2-3 most probable root causes** and provide **concise** recommendations for further investigation. "
                       "Keep your response **brief and to the point**, avoiding unnecessary details. "
                       "Limit the response to **a maximum of 100 words**."
        },
        {
            "role": "user",
            "content": f"Analyze the following anomalous log entry:\n\n{log_entry}\n\n"
                       "Provide:\n"
                       "1. The possible causes of the anomaly.\n"
                       "2. Recommendations for further action."
        }
    ]

    response = get_completion(messages) 
    return response



def process_anomalous_logs(input_file="representative_logs1.csv"):
    df = pd.read_csv(input_file)

    results = []
    analysis_reports = []

    print("\nStarting anomaly analysis...\n")

    for idx, row in df.iterrows():
        log_entry = row['remaining_log']
        anomaly_label = row['formatted_label']

        print(f"\nProcessing Log {idx+1}/{len(df)}")
        print(f"Log Entry: {log_entry}")
        print(f"Anomaly Status: {'Anomalous' if anomaly_label == 1 else 'Normal'}")

        if anomaly_label == 1: 
            explanation = analyze_anomalous_log(log_entry)
        else:
            explanation = "Log entry is normal. No further analysis required."

        results.append((log_entry, anomaly_label, explanation))
        analysis_reports.append(f"Log {idx+1}:\n{explanation}\n" + "="*50 + "\n")

        print("\nGenerated Analysis:")
        print(explanation)

        sleep(1)  


    output_file = "anomaly_analysis_results.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["log_entry", "anomaly_label", "analysis_report"])
        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to {output_file}")

    with open("anomaly_analysis_report.txt", "w") as f:
        f.write("ANOMALY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        for report in analysis_reports:
            f.write(report)

    print("\nDetailed anomaly report saved to anomaly_analysis_report.txt")

if __name__ == "__main__":
    process_anomalous_logs()
