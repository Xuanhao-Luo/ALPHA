
import csv
import pandas as pd
from time import sleep
from litellm import completion

 
API_KEY = " "   
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

def extract_decision(response):
    if response.endswith("1"):
        return "1"
    elif response.endswith("0"):
        return "0"
    else:
        for char in reversed(response):
            if char in ["0", "1"]:
                return char
    return "Unknown"


def read_csv(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
        return "\n".join([",".join(row) for row in data])


data_with_anomaly_1 = read_csv("error_anomaly.csv")
data_with_no_anomaly = read_csv("error_normal.csv")


def analyze_log_entry(log_entry, method="fewshot"):

    initial_messages = [
        {
            "role": "system",
            "content": """You are a data analyser which spots any anomaly in network logs. You have been trained on examples of both anomalous and normal logs and I will give you more examples. Your task is to determine whether a given log entry is anomalous or normal. Always conclude your analysis with a single digit response: '1' for Anomalous or '0' for Non-anomalous."""
        },
        {"role": "user", "content": "Here are examples of anomalous logs: " + data_with_anomaly_1},
        {"role": "assistant", "content": "These logs show characteristics of anomalous behavior. My decision: 1"},
        {"role": "user", "content": "Here are examples of normal logs: " + data_with_no_anomaly},
        {"role": "assistant", "content": "These logs appear to be normal network activity. My decision: 0"},
    ]

    if method == "fewshot":
        prompt = [
            {"role": "system", "content": "Analyze the log entry based on prior examples and patterns."},
            {"role": "user", "content": f"Analyze this log entry: {log_entry}"}
        ]
    elif method == "cot":
        prompt = [
            {"role": "system", "content": 
             "Use a step-by-step reasoning approach before making a final decision."},
            {"role": "user", "content": 
             "Analyze the following log entry:\n"
             "Step 1: Identify key components (timestamps, error messages, system events).\n"
             "Step 2: Compare against known anomalous and normal patterns.\n"
             "Step 3: Provide a logical explanation of why it is anomalous or normal.\n"
             "Final decision: [0/1]\n\n"
             f"Log Entry: {log_entry}"}
        ]
    else:
        raise ValueError("Invalid method. Choose 'fewshot' or 'cot'.")

    response = get_completion(initial_messages + prompt)
    decision = extract_decision(response)

    return log_entry, decision, response


def process_log_file(input_file="representative_logs.csv", output_file="log_analysis_results_1.csv", method="cot"):

    df = pd.read_csv(input_file)
    results = []

    print("\nStarting log analysis...\n")

    for idx, row in df.iterrows():
        log_entry = row['remaining_log']
        anomaly_label = row['formatted_label']  

        print(f"\nProcessing Log {idx+1}/{len(df)}")
        print(f"Anomaly Status: {'1' if anomaly_label == 1 else '0'}")

        analyzed_log, predicted_label, explanation = analyze_log_entry(log_entry, method)

        results.append((log_entry, anomaly_label, predicted_label, explanation))
        
        print(f"LLM Decision: {predicted_label}")

        sleep(1) 

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["log_entry", "true_label", "predicted_label", "analysis_report"])
        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    process_log_file()
