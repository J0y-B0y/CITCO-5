import requests
import pandas as pd
import argparse
import urllib.parse
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
from serpapi import GoogleSearch
from urllib.parse import urlencode
logging.info(f"SERPAPI_KEY from Python: {os.getenv('SERPAPI_KEY')}")
# Wait to simulate human-like behavior
time.sleep(random.uniform(1, 3))


def normalize_name(name: str) -> str:
  """
    If the name is 'Last, First' -> 'First Last'.
    Otherwise, just strip extra spaces.
    """
  return " ".join(reversed(
      name.split(", "))).strip() if "," in name else name.strip()


def extract_author_id(profile_url: str) -> str:
  parsed = urllib.parse.urlparse(profile_url)
  query = urllib.parse.parse_qs(parsed.query)
  return query.get("user", [""])[0]


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)s – %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# ------------------------
# Scholar Profile Search (attempt #1)
# ------------------------
def search_scholar_profile(name: str) -> Tuple[Optional[str], Optional[str]]:
  """
    1) Use google_scholar_profiles engine to find an author's profile link by name.
    2) If that fails, fallback to google_scholar (paper search) and parse the
       first /citations?user= link in 'organic_results' if any.
    Returns: (profile_url, error or None)
    """
  # 1) Try google_scholar_profiles
  profile_url, err = search_scholar_profiles_engine(name)
  if profile_url is not None:
    return profile_url, None

  logging.info(f"Falling back to google_scholar engine for {name}...")
  # 2) Fallback to google_scholar search
  fallback_url, fb_err = search_scholar_papers_engine(name)
  if fallback_url is not None:
    return fallback_url, None
  else:
    # If STILL not found, return error
    return None, fb_err


def search_scholar_profiles_engine(
    name: str) -> Tuple[Optional[str], Optional[str]]:
  """
    Attempts to find a google_scholar_profiles match. If found, return link.
    """
  formatted_name = normalize_name(name)
  logging.info(f"(1/2) google_scholar_profiles: {formatted_name}")

  params = {
      "engine": "google_scholar_profiles",
      "mauthors": formatted_name,
      "api_key": os.getenv("SERPAPI_KEY")
  }

  try:
    search = GoogleSearch(params)
    results = search.get_dict()

    profiles = results.get("profiles")
    if not profiles:
      return None, "Profile not found"

    # Prioritize profiles with verified email or higher cited_by
    sorted_profiles = sorted(
        profiles,
        key=lambda x:
        (x.get("email", "").startswith("Verified"), x.get("cited_by", 0)),
        reverse=True)

    best_profile = sorted_profiles[0]
    return best_profile.get("link"), None

  except Exception as e:
    return None, f"SerpAPI error (profiles): {e}"


def search_scholar_papers_engine(
    name: str) -> Tuple[Optional[str], Optional[str]]:
  """
    Fallback: use google_scholar (paper search) and see if any /citations?user= link is present
    in 'organic_results'.
    """
  formatted_name = normalize_name(name)
  logging.info(f"(2/2) google_scholar (papers) fallback: {formatted_name}")

  params = {
      "engine": "google_scholar",
      "q": formatted_name,
      "api_key": os.getenv("SERPAPI_KEY")
  }

  try:
    search = GoogleSearch(params)
    results = search.get_dict()

    # 'organic_results' might contain multiple hits
    org = results.get("organic_results", [])
    for item in org:
      link = item.get("link")
      # Check if link looks like: https://scholar.google.com/citations?user=xxxx
      if link and "/citations?user=" in link:
        return link, None

    return None, "Profile not found (fallback)."
  except Exception as e:
    return None, f"SerpAPI error (papers fallback): {e}"


# ------------------------
# Citation Data Fetcher
# ------------------------
def fetch_author_citations_via_serpapi(
    author_id: str) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
  """
    Grabs the citations by year from google_scholar_author endpoint
    """
  params = {
      "engine": "google_scholar_author",
      "author_id": author_id,
      "api_key": os.getenv("SERPAPI_KEY"),
      "hl": "en"
  }

  try:
    search = GoogleSearch(params)
    results = search.get_dict()
    citation_graph = results.get("cited_by", {}).get("graph", [])
    if not citation_graph:
      return None, "No citation data found"

    year_citation_map = {
        int(entry["year"]): int(entry["citations"])
        for entry in citation_graph if "year" in entry and "citations" in entry
    }
    return year_citation_map, None
  except Exception as e:
    return None, f"SerpAPI author fetch error: {e}"


# ------------------------
# Data Filtering (2.3)
# ------------------------
def filter_six_year_window(data: Dict[int, int],
                           dg_year: int) -> Dict[int, int]:
  return {
      year: data[year]
      for year in range(dg_year - 6, dg_year) if year in data
  }


# ------------------------
# Totals Calculation (2.4)
# ------------------------
def compute_totals(filtered_data: Dict[int, int]) -> Tuple[int, int]:
  total_citations = sum(filtered_data.values())
  total_publications = len(filtered_data)  # # of years with citations
  return total_citations, total_publications


# ------------------------
# Failure Handling (2.5)
# ------------------------
def handle_failure(name: str,
                   reason: str,
                   log_file: str = "error_log.txt") -> None:
  logging.warning(f"{name} – {reason}")
  with open(log_file, 'a') as f:
    f.write(f"{name}: {reason}\n")


# ------------------------
# Data Export (2.6)
# ------------------------
def export_dataset(data: List[dict],
                   output_path: str,
                   export_txt: bool = False) -> None:
  df = pd.DataFrame(data)
  print(f"Exporting data to {output_path}")
  df.to_csv(output_path, index=False)
  logging.info(f"Exported CSV: {output_path}")

  if export_txt:
    txt_path = output_path.replace(".csv", ".txt")
    df.to_csv(txt_path, sep='\t', index=False)
    logging.info(f"Exported TXT: {txt_path}")


# ------------------------
# Researcher Processing
# ------------------------
def process_researchers(input_list: List[dict],
                        export_txt: bool = True,
                        output_file: str = "researcher_stats.csv") -> None:
  processed_data = []
  os.makedirs("exports", exist_ok=True)

  for r in input_list:
    name, dg_year = r.get('name'), r.get('dg_year', 2021)
    if not name:
      continue

    # 1) Find the user's google scholar profile link (with fallback)
    profile_url, err = search_scholar_profile(name)
    if err:
      handle_failure(name, err)
      continue

    author_id = extract_author_id(profile_url)
    citation_data, err = fetch_author_citations_via_serpapi(author_id)
    if err:
      handle_failure(name, err)
      continue

    filtered = filter_six_year_window(citation_data, dg_year)
    total_citations, total_pubs = compute_totals(filtered)

    processed_data.append({
        "Name": name,
        "DG Year": dg_year,
        "Total Citations": total_citations,
        "Total Publications": total_pubs,
        "Grant Amount": r.get("Grant Amount", 10000)  # default
    })

  if processed_data:
    export_dataset(processed_data, os.path.join("exports", output_file),
                   export_txt)
  else:
    print("No data to export!")


# ------------------------
# Correlation / Analysis
# ------------------------
def compute_pearson(citations, grants):
  r, p = pearsonr(citations, grants)
  return r, p


def compute_spearman(citations, grants):
  rho, p_rho = spearmanr(citations, grants)
  return rho, p_rho


def coleman_index(citations, grants):
  if citations.mean() == 0:
    return 0.0
  return (citations.mean() - grants.mean()) / citations.mean()


def load_nserc_data(csv_path: str = "names.csv") -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  print("Available columns in names.csv:", df.columns.tolist())
  if "Name-Nom" not in df.columns:
    raise ValueError("Required column 'Name-Nom' not found in names.csv")
  df = df.rename(columns={"Name-Nom": "name"})
  df = df.dropna(subset=["name"], how="any")
  return df[["name"]]


def validate_results_with_alternative(citations: pd.Series, grants: pd.Series):
  r, _ = compute_pearson(citations, grants)
  rho, _ = compute_spearman(citations, grants)
  print("Validation with same dataset:")
  print(f" Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")


def run_sensitivity_analysis(citations: pd.Series, grants: pd.Series):
  print("\nRunning sensitivity analysis with noise...")
  noisy_citations = citations.copy()
  noisy_grants = grants.copy()
  if len(noisy_citations) > 1:
    noisy_citations.iloc[0] *= 2
    noisy_grants.iloc[1] *= 1.5
  r, _ = compute_pearson(noisy_citations, noisy_grants)
  rho, _ = compute_spearman(noisy_citations, noisy_grants)
  print(f" Pearson r (noisy) = {r:.4f}, Spearman rho (noisy) = {rho:.4f}")


def generate_visualizations(citations: pd.Series,
                            grants: pd.Series,
                            output_dir="exports"):
  os.makedirs(output_dir, exist_ok=True)
  df = pd.DataFrame({"Citations": citations, "Grant Amount": grants})
  plt.figure(figsize=(8, 6))
  sns.regplot(x="Citations", y="Grant Amount", data=df, ci=None)
  plt.title("Citations vs Grant Amount")
  plt.savefig(os.path.join(output_dir, "scatter_regression.png"))
  plt.close()

  corr = df.corr()
  plt.figure(figsize=(5, 4))
  sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
  plt.title("Correlation Matrix")
  plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
  plt.close()
  print("✅ Visualizations saved to 'exports/'")


def analyze_correlation(csv_path: str = "exports/researcher_stats.csv"):
  if not os.path.exists(csv_path):
    print(f"❌ Output file not found: {csv_path}")
    return
  df = pd.read_csv(csv_path)
  if 'Total Citations' not in df or 'Grant Amount' not in df:
    raise ValueError(
        "CSV must include 'Total Citations' and 'Grant Amount' columns")

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


def main():
  parser = argparse.ArgumentParser(description="CITCO Statistical Analyzer")
  parser.add_argument('--input',
                      type=str,
                      default="names.csv",
                      help="CSV file with columns: name,dg_year,Grant Amount")
  parser.add_argument('--output',
                      type=str,
                      default="researcher_stats.csv",
                      help="Output CSV file name")
  parser.add_argument('--txt',
                      action='store_true',
                      help="Also export .txt version")
  parser.add_argument('--analyze',
                      action='store_true',
                      help="Run statistical correlation analysis")
  args = parser.parse_args()

  # 1) Load from CSV
  df = load_nserc_data(args.input)
  researcher_list = df.to_dict('records')

  # 2) Process & scrape
  process_researchers(researcher_list,
                      export_txt=args.txt,
                      output_file=args.output)

  # 3) Check & Analyze
  output_path = os.path.join("exports", args.output)
  if not os.path.exists(output_path):
    print(f"❌ Output file not generated: {output_path}")
    return

  if args.analyze:
    analyze_correlation(output_path)


if __name__ == "__main__":
  main()
