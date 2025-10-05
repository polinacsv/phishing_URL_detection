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
    Loads Alexa Top domains from .txt (one per line) or .csv.
    Returns df with columns: ['alexa_domain','ranking'] (both lowercased domain).
    """
    import os
    import pandas as pd

    path = os.path.join(data_dir, filename)
    # try to detect delimiter/header automatically
    if filename.lower().endswith(".txt"):
        df = pd.read_csv(path, header=None, names=["alexa_domain"], dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    # normalize columns
    if "alexa_domain" not in df.columns:
        # try to infer a domain column
        domain_col = None
        for c in df.columns:
            if "domain" in c.lower() or "host" in c.lower():
                domain_col = c
                break
        if domain_col is None and df.shape[1] == 1:
            df.columns = ["alexa_domain"]
        elif domain_col:
            df = df.rename(columns={domain_col: "alexa_domain"})
        else:
            raise ValueError("Could not find an Alexa domain column.")

    df["alexa_domain"] = df["alexa_domain"].astype(str).str.strip().str.lower()

    if "rank" not in df.columns:
        df["ranking"] = df.index + 1
    else:
        df["ranking"] = (
            pd.to_numeric(df["rank"], errors="coerce")
            .fillna(df.index + 1)
            .astype(int)
        )

    return df[["alexa_domain", "ranking"]]