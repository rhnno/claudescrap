# Manual Login Profile Testing Guide

Since the automated tests are experiencing timeout issues, here's how to manually test the login profile system:

## 🔧 **Setup Status**

✅ **Enhanced browser.py** - Updated with profile and login functionality  
✅ **Chrome profile directory** - `chrome_profiles/research_profile/` exists  
✅ **Credentials template** - Need to create `config/login_credentials.json`  

## 📝 **Step 1: Create Credentials File**

Create `config/login_credentials.json` with your actual credentials:

```json
{
  "tokopedia": {
    "email": "your-email@example.com",
    "password": "your-password",
    "login_url": "https://accounts.tokopedia.com/otp/c/page?otp_type=116&msisdn=&ld=https%3A%2F%2Fwww.tokopedia.com%2F"
  },
  "shopee": {
    "email": "your-email@example.com",
    "password": "your-password",
    "login_url": "https://shopee.co.id/buyer/login"
  },
  "bukalapak": {
    "email": "your-email@example.com",
    "password": "your-password",
    "login_url": "https://accounts.bukalapak.com/login"
  }
}
```

## 🧪 **Step 2: Manual Testing**

### **Test 1: Basic Profile Setup**
```python
# Create a simple test file: test_profile_manual.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'func'))

from func.browser import BrowserManager

# Initialize with profile
browser = BrowserManager(use_profile=True, profile_name="research_profile")
print(f"Profile path: {browser.profile_path}")
print(f"Credentials file: {browser.credentials_file}")

# Setup driver (this will create the profile)
driver = browser.setup_driver()
print("✅ Browser setup successful!")

# Test navigation
browser.navigate_to("https://www.tokopedia.com")
print("✅ Navigation successful!")

# Keep browser open for manual inspection
input("Press Enter to close browser...")
browser.close()
```

### **Test 2: Login Functionality**
```python
# test_login_manual.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'func'))

from func.browser import BrowserManager

browser = BrowserManager(use_profile=True, profile_name="research_profile")
browser.setup_driver()

# Test login to Tokopedia
print("🔐 Testing Tokopedia login...")
success = browser.ensure_login("tokopedia")

if success:
    print("✅ Login successful!")
    # Test search functionality
    browser.navigate_to("https://www.tokopedia.com/search?st=product&q=laptop")
    print("🔍 Search page loaded")
else:
    print("❌ Login failed")

input("Press Enter to close...")
browser.close()
```

## 🔍 **Step 3: Integration with Main Scraper**

Update your `main.py` to use the enhanced browser:

```python
# In main.py, replace the browser initialization:

# OLD:
# browser = BrowserManager()

# NEW:
browser = BrowserManager(use_profile=True, profile_name="research_profile")
browser.setup_driver()

# Ensure login before scraping
browser.ensure_login("tokopedia")  # This will auto-login if needed

# Then proceed with normal scraping...
```

## 🎯 **Expected Results**

### **First Run:**
1. Chrome profile directory created
2. Browser opens with persistent profile
3. Auto-login attempts (may need manual 2FA/CAPTCHA)
4. Session saved in profile

### **Subsequent Runs:**
1. Browser loads existing profile
2. Already logged in (no need to re-login)
3. Can immediately start scraping
4. Faster startup time

## 🛠️ **Troubleshooting**

### **Common Issues:**

**1. Selenium Import Error:**
```bash
pip install selenium
```

**2. ChromeDriver Not Found:**
```bash
# Download ChromeDriver and add to PATH
# Or use webdriver-manager:
pip install webdriver-manager
```

**3. Login Fails:**
- Check credentials in `config/login_credentials.json`
- Handle 2FA manually on first login
- Complete any CAPTCHAs manually
- Check if site has changed login flow

**4. Profile Issues:**
- Delete `chrome_profiles/research_profile/` and recreate
- Check file permissions
- Ensure Chrome is fully closed before testing

## 🔒 **Security Notes**

- **Credentials Storage**: Consider using environment variables for sensitive data
- **Profile Security**: The profile contains your login session data
- **Backup**: Consider backing up your profile after successful login
- **Cleanup**: Clear profile if sharing the code

## 📊 **Testing Checklist**

- [ok] Credentials file created with real credentials
- [ok] Browser launches with profile
- [notyet] Auto-login works for at least one site
- [notyet] Session persists between browser restarts
- [notyet] Can navigate to search pages while logged in
- [no] Integration with main scraper works
- [didnttry] No rate limiting or detection issues

## 🎉 **Success Indicators**

✅ **Profile Created**: `chrome_profiles/research_profile/` contains Chrome data  
✅ **Auto-Login Works**: Can login without manual intervention  
✅ **Session Persists**: Stay logged in between runs  
✅ **Scraping Enhanced**: Can access more content while logged in  
✅ **No Detection**: No unusual CAPTCHAs or blocks  

## 💡 **Next Steps After Testing**

1. **Fine-tune Login Selectors**: Update selectors if sites change
2. **Add More Sites**: Extend to other Indonesian e-commerce sites
3. **Implement 2FA Handling**: Add support for SMS/email verification
4. **Monitor Success Rates**: Track login success over time
5. **Integrate with Dashboard**: Add login status to your planned frontend

---

*Use this guide to manually test the login profile system. Once working, you can automate the testing process.*