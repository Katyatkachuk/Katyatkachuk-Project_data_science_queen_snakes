import pandas as pd
import matplotlib.pyplot as plt

def plot_cancer_vs_noncancer(csv_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Define cancer and non-cancer diagnoses
    cancer_diagnoses = ['BCC', 'MEL', 'SCC']
    noncancer_diagnoses = ['ACK', 'NEV', 'SEK']

    # Calculate the count for each group
    cancer_counts = data['diagnostic'].isin(cancer_diagnoses).sum()
    noncancer_counts = data['diagnostic'].isin(noncancer_diagnoses).sum()

    # Print the count of cancer and non-cancer photos
    print(f"Number of cancer photos: {cancer_counts}")
    print(f"Number of non-cancer photos: {noncancer_counts}")

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Cancer', 'Non-Cancer'], [cancer_counts, noncancer_counts], color=['red', 'green'])

    # Set the title and labels
    plt.title('Cancer vs Non-Cancer Diagnosis')
    plt.ylabel('Count')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\katya\ITU\2 semester\Project in Data Science\Project_Data_Science_2_sem\Classifier\data\metadata.csv"  # Path to your CSV file
    plot_cancer_vs_noncancer(file_path)
