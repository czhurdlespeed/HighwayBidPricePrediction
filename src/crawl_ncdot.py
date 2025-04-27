import os
import time
from datetime import datetime

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def setup_driver():
    options = webdriver.ChromeOptions()

    # Set up download directory
    download_dir = os.path.join(os.getcwd(), "ExcelFiles")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Configure Chrome options for downloading
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            # Add these lines to handle Excel files
            "plugins.always_open_pdf_externally": True,  # Force download for PDFs
            "download.open_pdf_in_system_reader": False,
            # MIME types for Excel files
            "profile.default_content_settings.popups": 0,
            "download.default_directory": download_dir,
            "safebrowsing.enabled": True,
            "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "download.prompt_for_download": False,
            # Add MIME type handling
            "plugins.plugins_list": [
                {"enabled": False, "name": "Chrome PDF Viewer"}
            ],
            "download.extensions_to_open": "",
            "profile.default_content_settings.popups": 0,
            "download.default_directory": download_dir,
        },
    )

    # Add additional arguments
    options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-popup-blocking")

    return webdriver.Chrome(options=options)


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except:
        return None


def download_excel_file(driver):
    try:
        # Find the Excel link
        excel_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "a.ms-listlink.ms-draggable[app='ms-excel']")
            )
        )

        # Get the href (download URL)
        download_url = excel_link.get_attribute("href")
        file_name = excel_link.text
        print(f"Downloading: {file_name}")

        # Use JavaScript to trigger download
        js_script = f"""
        var link = document.createElement('a');
        link.href = '{download_url}';
        link.download = '{file_name}.xls';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        """
        driver.execute_script(js_script)

        time.sleep(3)  # Wait for download to start
        return True
    except Exception as e:
        print(f"Error downloading Excel file: {str(e)}")
        return False


def get_all_page_urls(driver, base_url):
    urls = [base_url]
    try:
        while True:
            # Try to find specifically the next button (not previous) by looking for right arrow image
            next_button = driver.find_element(
                By.CSS_SELECTOR,
                "a.ms-commandLink.ms-promlink-button.ms-promlink-button-enabled img.ms-promlink-button-right",
            ).find_element(By.XPATH, "..")

            if next_button.is_displayed() and next_button.is_enabled():
                next_button.click()
                time.sleep(2)
                # Get the URL of the new page
                current_url = driver.current_url
                urls.append(current_url)
                print(f"Found page {len(urls)}: {current_url}")
            else:
                break

    except NoSuchElementException:
        print(f"Total pages found: {len(urls)}")

    return urls


def navigate_letting_pages():
    driver = setup_driver()
    cutoff_date = datetime(2022, 1, 1)
    main_window = None
    base_url = (
        "https://connect.ncdot.gov/letting/Central%20Letting/Forms/BidTabs.aspx"
    )

    try:
        # First get all page URLs
        print("Discovering all pages...")
        driver.get(base_url)
        time.sleep(3)
        page_urls = get_all_page_urls(driver, base_url)
        print(f"Found {len(page_urls)} pages")

        # Now process each page
        for page_num, url in enumerate(page_urls, 1):
            print(f"\nProcessing page {page_num} of {len(page_urls)}")
            driver.get(url)
            time.sleep(3)

            if main_window is None:
                main_window = driver.current_window_handle

            # Get all rows on current page
            table_rows = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.CSS_SELECTOR, "tr.ms-itmhover")
                )
            )

            # Process each row
            for row in table_rows:
                try:
                    date_cell = row.find_element(
                        By.CSS_SELECTOR, "td[role='gridcell'] span.ms-noWrap"
                    )
                    date_text = date_cell.get_attribute("title")
                    letting_date = parse_date(date_text)

                    if letting_date and letting_date > cutoff_date:
                        link = row.find_element(
                            By.CSS_SELECTOR, "a.ms-listlink.ms-draggable"
                        )
                        print(f"Processing: {link.text} - Date: {date_text}")

                        # Open in new tab using JavaScript
                        href = link.get_attribute("href")
                        driver.execute_script(
                            "window.open(arguments[0]);", href
                        )

                        # Switch to new tab
                        driver.switch_to.window(driver.window_handles[-1])

                        if download_excel_file(driver):
                            print("Successfully downloaded Excel file")

                        # Close tab and switch back
                        driver.close()
                        driver.switch_to.window(main_window)
                        time.sleep(2)
                    else:
                        print(f"Skipping: Date {date_text} is before cutoff")

                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        driver.quit()


if __name__ == "__main__":
    navigate_letting_pages()
