import pandas as pd
import os

def load_phishing_data(data_dir: str, filename: str, url_col: str, label_col: str) -> pd.DataFrame:
    """
    Loads a phishing URL dataset, standardizes column names, and recodes labels to 0 (benign) and 1 (phishing).

    Parameters:
        data_dir (str): Path to the data directory (e.g., "data/")
        filename (str): Name of the CSV file (e.g., "urlset.csv")
        url_col (str): Name of the column containing URLs
        label_col (str): Name of the column containing labels

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized columns: 'url', 'label'
    """
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', low_memory=False)

    # Drop rows with missing URL or label
    df = df.dropna(subset=[url_col, label_col])

    # Keep only the specified columns and rename
    df = df[[url_col, label_col]].rename(columns={url_col: 'url', label_col: 'label'})

    # Normalize labels to 0 (benign) and 1 (phishing)
    phishing_values = {'phishing', 'phish', 'malicious', '1', '1.0', 'yes', 'true'}
    benign_values   = {'benign', 'legit', '0', '0.0', 'no', 'false'}

    df['label'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() in phishing_values else
                                              0 if str(x).strip().lower() in benign_values else None)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    return df

def load_alexa_domains(data_dir: str, filename: str) -> pd.DataFrame:
    """
    Loads the Alexa Top 1M domains list from a .txt file (one domain per line).

    Parameters:
        data_dir (str): Directory containing the Alexa file
        filename (str): Name of the .txt file (e.g., 'alexa_domains_1M.txt')

    Returns:
        pd.DataFrame: DataFrame with columns ['alexa_domain', 'rank']
    """
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path, header=None, names=['alexa_domain'])
    df['rank'] = df.index + 1  # 1-based rank based on file order
    return df