from playwright.sync_api import sync_playwright
import time

def test_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Wait for streamlit to load
        page.wait_for_selector('h1:has-text("GNN Symptom Checker")', timeout=10000)

        # Check title
        print("Page title:", page.title())

        # Find the multiselect input for symptoms
        # Streamlit multiselects have a specific structure. The input is inside a div with data-baseweb="select"
        page.click('div[data-baseweb="select"]')

        # We need to wait for the dropdown to appear and select a symptom, e.g., "Fever"
        page.fill('input[aria-autocomplete="list"]', "high_fever")
        page.keyboard.press("Enter")
        time.sleep(1) # wait for streamlit to process

        # Add another symptom
        page.click('div[data-baseweb="select"]')
        page.fill('input[aria-autocomplete="list"]', "cough")
        page.keyboard.press("Enter")
        time.sleep(1)
        page.keyboard.press("Escape") # Close the dropdown menu
        time.sleep(1)

        # Click predict button
        page.click('button:has-text("Predict Disease")')

        # Wait for the results to appear (we look for the "Top Predictions:" subheader)
        page.wait_for_selector('h3:has-text("Top Predictions")', timeout=10000)

        # Read the top predictions
        # The new structure uses st.metric for the diseases, which creates data-testid="stMetricLabel"
        predictions = page.query_selector_all('div[data-testid="stMetricLabel"]')
        print("Predictions:")
        for pred in predictions:
            print("-", pred.inner_text())

        browser.close()

if __name__ == "__main__":
    test_app()
