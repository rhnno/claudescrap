from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
options = webdriver.ChromeOptions()
service = Service(ChromeDriverManager().install())
options.add_argument('--headless')
driver = webdriver.Chrome(service=service, options=options)
print("Success!")
driver.quit()