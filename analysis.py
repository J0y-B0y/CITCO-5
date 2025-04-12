import requests
import pandas as pd
import argparse
import urllib.parse
from bs4 import BeautifulSoup
from typing import Tuple, Dict, List, Optional
import logging
import os
from sklearn.feature_selection import r_regression
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
time.sleep(random.uniform(1, 3))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s – %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def exportNames(code,txt):
    fields = ['Name-Nom', 'AreaOfApplicationCode']

    df = pd.read_csv('database.csv', encoding='latin-1', usecols=fields)

    filterBySubject = df[(df['AreaOfApplicationCode'] == code)] # 800 is the information systems code

    df_Out = filterBySubject[['Name-Nom']]

    print (df_Out) #only returns names

    df_Out.to_csv('names.csv')
    
    if (txt):
        df_Out.to_csv('names.txt', sep='\t', index=False)

print("Welcome to NSERC DB Scraper")
print("=================================")

print("By default, this program scrapes the database and return names of award winners")

print("")

code = input("Please enter the area of application code for awards you would like to retrieve. Press enter to select default code. (Information systems is 800): ")
if code == '':
    code = 800
else:
    code = int(code)

print("")

year = input("Enter the range (in fiscal years) of databases that you want to pull from: ")

print("")

txt = input("Would you like a .txt file as well?: ")
print("")

print("Showing the first and last 5 entries:")

exportNames(code, txt)

print("names.csv has been exported.")

# ------------------------
# Scholar Search (2.1)
# ------------------------
def search_scholar_profile(name: str) -> Tuple[Optional[str], Optional[str]]:
    query = urllib.parse.quote_plus(name)
    url = f"https://scholar.google.com/scholar?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    logging.info(f"Searching for: {name}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        link = soup.find('a', href=True)
        if link and "/citations?" in link['href']:
            return "https://scholar.google.com" + link['href'], None
        return None, "Profile not found"
    except requests.RequestException as e:
        return None, f"Request error: {e}"

# ------------------------
# Scholar Scraping (2.2)
# ------------------------
def scrape_citation_data(profile_url: str) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(profile_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        years = [int(tag.text) for tag in soup.select('.gsc_g_t')]
        citations = [int(tag.text) for tag in soup.select('.gsc_g_al')]

        if not years or not citations:
            return None, "No citation data found"

        return dict(zip(years, citations)), None
    except Exception as e:
        return None, f"Failed to parse Scholar page: {e}"

# ------------------------
# Data Filtering (2.3)
# ------------------------
def filter_six_year_window(data: Dict[int, int], dg_year: int) -> Dict[int, int]:
    """
    Filters citation data for the 6 years preceding the DG award year.
    """
    return {year: data[year] for year in range(dg_year - 6, dg_year) if year in data}

# ------------------------
# Totals Calculation (2.4)
# ------------------------
def compute_totals(filtered_data: Dict[int, int]) -> Tuple[int, int]:
    """
    Computes total citations and publications.
    """
    total_citations = sum(filtered_data.values())
    total_publications = len(filtered_data)
    return total_citations, total_publications

# ------------------------
# Failure Handling (2.5)
# ------------------------
def handle_failure(name: str, reason: str, log_file: str = "error_log.txt") -> None:
    """
    Logs failed lookups or processing.
    """
    logging.warning(f"{name} – {reason}")
    with open(log_file, 'a') as f:
        f.write(f"{name}: {reason}\n")

# ------------------------
# Data Export (2.6)
# ------------------------
def export_dataset(data: List[dict], output_path: str, export_txt: bool = False) -> None:
    """
    Exports researcher data to CSV and optionally TXT.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logging.info(f"Exported CSV: {output_path}")

    if export_txt:
        txt_path = output_path.replace(".csv", ".txt")
        df.to_csv(txt_path, sep='\t', index=False)
        logging.info(f"Exported TXT: {txt_path}")

# ------------------------
# Main Pipeline
# ------------------------
def process_researchers(input_list: List[dict], export_txt: bool = True, output_file: str = "researcher_stats.csv") -> None:
    """
    Runs the full analysis pipeline for all researchers.
    """
    processed_data = []
    os.makedirs("exports", exist_ok=True)

    for r in input_list:
        name, dg_year = r.get('name'), r.get('dg_year')
        if not name or not dg_year:
            continue

        profile_url, err = search_scholar_profile(name)
        if err:
            handle_failure(name, err)
            continue

        citation_data, err = scrape_citation_data(profile_url)
        if err:
            handle_failure(name, err)
            continue

        filtered = filter_six_year_window(citation_data, dg_year)
        total_citations, total_pubs = compute_totals(filtered)

        logging.info(f"{name} | Citations: {total_citations}, Publications: {total_pubs}")
        processed_data.append({
            "Name": name,
            "DG Year": dg_year,
            "Total Citations": total_citations,
            "Total Publications": total_pubs
        })

    
    export_dataset(processed_data, os.path.join("exports", output_file), export_txt)


# Pearson Correlation Function
def compute_pearson(citations, grants):
    return pearsonr(citations, grants)

def compute_spearman(citations, grants):
    return spearmanr(citations, grants)

def coleman_index(citations, grants):
    if citations.mean() == 0:
        return 0.0
    return (citations.mean() - grants.mean()) / citations.mean()


# Coleman Index Calculation (Custom Formula)
def coleman_index(citations, grants):
    # Avoid division by zero by checking the mean of citations
    if citations.mean() == 0:
        return 0.0
    coleman = (citations.mean() - grants.mean()) / citations.mean()
    return coleman


def load_nserc_data(csv_path: str = "names.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print("Available columns in names.csv:", df.columns.tolist())
    if "Name-Nom" not in df.columns:
        raise ValueError("Required column 'Name-Nom' not found in names.csv")
    df = df.rename(columns={"Name-Nom": "name"})
    df = df.dropna(subset=["name"], how="any")
    return df[["name"]]

# 3.4 Validate correlation results
def validate_results_with_alternative(citations: pd.Series, grants: pd.Series):
    """
    Validates the correlation results using Pearson and Spearman correlation
    with the same dataset.
    """
    r, _ = compute_pearson(citations, grants)
    rho, _ = compute_spearman(citations, grants)
    print("Validation with same dataset:")
    print(f" Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")

# 3.5 Sensitivity analysis
def run_sensitivity_analysis(citations: pd.Series, grants: pd.Series):
    """
    Runs a sensitivity analysis by introducing artificial outliers and computing
    Pearson and Spearman correlations again to see how sensitive the correlation is to noise.
    """
    print("\nRunning sensitivity analysis with noise...")
    
    noisy_citations = citations.copy()
    noisy_grants = grants.copy()

    # Introduce artificial outliers
    noisy_citations.iloc[0] *= 2  # artificial outlier in citations
    noisy_grants.iloc[1] *= 1.5  # artificial outlier in grants

    # Compute the Pearson and Spearman correlation on the noisy data
    r, _ = compute_pearson(noisy_citations, noisy_grants)
    rho, _ = compute_spearman(noisy_citations, noisy_grants)

    print(f" Pearson r (noisy) = {r:.4f}, Spearman rho (noisy) = {rho:.4f}")

# 3.6 Visualization
def generate_visualizations(citations: pd.Series, grants: pd.Series, output_dir="exports"):
    """
    Generates visualizations including a scatter plot with regression line
    and a correlation matrix for the citation and grant data.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({"Citations": citations, "Grant Amount": grants})

    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    sns.regplot(x="Citations", y="Grant Amount", data=df, ci=None)
    plt.title("Citations vs Grant Amount")
    plt.savefig(os.path.join(output_dir, "scatter_regression.png"))
    plt.close()

    # Correlation matrix
    corr = df.corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    print("✅ Visualizations saved to 'exports/'")

def export_pdf_report(citations, grants, output_path="exports/report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="CITCO Correlation Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt="This report summarizes the results of statistical analysis run on data collected from NSERC's Discovery Grant program and Google Scholar citation data. The objective is to determine whether a correlation exists between funding amounts and academic influence, measured via citation counts.")

    pdf.ln(10)
    r, p_r = compute_pearson(citations, grants)
    rho, p_rho = compute_spearman(citations, grants)
    coleman = coleman_index(citations, grants)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Statistical Summary:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"Pearson Correlation: r = {r:.4f}, p = {p_r:.4e}\n"
                            f"Spearman Correlation: rho = {rho:.4f}, p = {p_rho:.4e}\n"
                            f"Coleman Index: {coleman:.4f}")

    pdf.ln(5)
    pdf.cell(0, 10, txt="Visualizations:", ln=True)
    pdf.image("exports/scatter_regression.png", w=180)
    pdf.add_page()
    pdf.image("exports/correlation_matrix.png", w=180)

    pdf.output(output_path)
    print(f"✅ PDF report saved to {output_path}")

def export_txt_report(citations, grants, output_path="exports/report.txt"):
    with open(output_path, 'w') as f:
        r, p_r = compute_pearson(citations, grants)
        rho, p_rho = compute_spearman(citations, grants)
        coleman = coleman_index(citations, grants)

        f.write("CITCO Correlation Analysis Report\n")
        f.write("===============================\n\n")
        f.write("Pearson Correlation: r = {:.4f}, p = {:.4e}\n".format(r, p_r))
        f.write("Spearman Correlation: rho = {:.4f}, p = {:.4e}\n".format(rho, p_rho))
        f.write("Coleman Index: {:.4f}\n\n".format(coleman))
        f.write("Note: Visualizations are saved separately in the exports directory.\n")

    print(f"✅ TXT report saved to {output_path}")


# ------------------------
# Analyze Correlation (3.6)
# ------------------------
def analyze_correlation(csv_path: str = "exports/researcher_stats.csv", export_format: str = "pdf"):
    df = pd.read_csv(csv_path)

    if 'Total Citations' not in df or 'Grant Amount' not in df:
        raise ValueError("CSV must include 'Total Citations' and 'Grant Amount' columns")

    citations = df['Total Citations']
    grants = df['Grant Amount']

    print("=== Pearson Correlation ===")
    r, p_r = compute_pearson(citations, grants)
    print(f"Pearson r = {r:.4f}, p = {p_r:.4e}")

    print("\n=== Spearman Correlation ===")
    rho, p_rho = compute_spearman(citations, grants)
    print(f"Spearman rho = {rho:.4f}, p = {p_rho:.4e}")

    print("\n=== Coleman Index ===")
    coleman = coleman_index(citations, grants)
    print(f"Coleman Index = {coleman:.4f}")

    print("\n=== 3.4 Validate Correlation Results ===")
    validate_results_with_alternative(citations, grants)

    print("\n=== 3.5 Sensitivity Analysis ===")
    run_sensitivity_analysis(citations, grants)

    print("\n=== 3.6 Generating Visualizations ===")
    generate_visualizations(citations, grants)

    if export_format == "pdf":
        export_pdf_report(citations, grants)
    elif export_format == "txt":
        export_txt_report(citations, grants)
    else:
        print("Invalid export format specified. Use 'pdf' or 'txt'.")

# ------------------------
# Process Researchers and Export Data
# ------------------------
def process_researchers(input_list: List[dict], export_txt: bool = True, output_file: str = "researcher_stats.csv") -> None:
    processed_data = []
    os.makedirs("exports", exist_ok=True)

    for r in input_list:
        name, dg_year = r.get('name'), r.get('dg_year')
        if not name or not dg_year:
            continue

        profile_url, err = search_scholar_profile(name)
        if err:
            handle_failure(name, err)
            continue

        citation_data, err = scrape_citation_data(profile_url)
        if err:
            handle_failure(name, err)
            continue

        filtered = filter_six_year_window(citation_data, dg_year)
        total_citations, total_pubs = compute_totals(filtered)

        processed_data.append({
            "Name": name,
            "DG Year": dg_year,
            "Total Citations": total_citations,
            "Total Publications": total_pubs
        })

    if processed_data:
        export_dataset(processed_data, os.path.join("exports", output_file), export_txt)

# ------------------------
# Main Pipeline
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="CITCO Statistical Analyzer")
    parser.add_argument('--input', type=str, help="CSV file with columns: name,dg_year,Grant Amount", default="names.csv")
    parser.add_argument('--output', type=str, default="researcher_stats.csv", help="Output CSV file name")
    parser.add_argument('--txt', action='store_true', help="Also export .txt version")
    parser.add_argument('--analyze', action='store_true', help="Run statistical correlation analysis")

    args = parser.parse_args()

    # Load NSERC data from names.csv
    df = load_nserc_data(args.input)
    researcher_list = df.to_dict('records')

    # Process and scrape Scholar data
    process_researchers(researcher_list, export_txt=args.txt, output_file=args.output)

    # Analyze correlations if requested
    analyze_correlation(os.path.join("exports", args.output))

if __name__ == "__main__":
    main()
