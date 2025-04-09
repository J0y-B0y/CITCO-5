import requests
import pandas as pd
import argparse
import urllib.parse
from bs4 import BeautifulSoup
from typing import Tuple, Dict, List, Optional
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s – %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ------------------------
# Scholar Search (2.1)
# ------------------------
def search_scholar_profile(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Searches Google Scholar for a researcher's profile link.
    """
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
    """
    Scrapes citation history from a Google Scholar profile.
    """
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

# ------------------------
# CLI Interface
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="CITCO Statistical Analyzer (Sprint 2, Sections 2.1–2.6)")
    parser.add_argument('--input', type=str, help="CSV file with columns: name,dg_year")
    parser.add_argument('--output', type=str, default="researcher_stats.csv", help="Output CSV file name")
    parser.add_argument('--txt', action='store_true', help="Also export .txt version")
    args = parser.parse_args()

    # Load researcher input
    if args.input:
        df = pd.read_csv(args.input)
        researcher_list = df.to_dict('records')
    else:
        # Fallback example
        researcher_list = [
            {"name": "Alice Smith", "dg_year": 2019},
            {"name": "John Doe", "dg_year": 2020},
            {"name": "Jane Quantum", "dg_year": 2021}
        ]

    process_researchers(researcher_list, export_txt=args.txt, output_file=args.output)

if __name__ == "__main__":
    main()
