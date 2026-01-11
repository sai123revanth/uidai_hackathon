import requests
import time

# --- CONFIGURATION ---
# Replace these with your actual 15 Streamlit App URLs
urls = [
    "https://uidaihackathon-e2cehtuk9hddodamhb5kkv.streamlit.app/",
    "https://uidaihackathon-lnf6m8mb68ifhno7t9hxkz.streamlit.app/",
    "https://uidaihackathon-sfg9wxwtifyamf7moqrchv.streamlit.app/",
    "https://uidaihackathon-ychxvfn834rmkxedwxqqjx.streamlit.app/",
    "https://uidaihackathon-4mmhnktehvbisjrxxs3ne8.streamlit.app/",
    "https://uidaihackathon-5sml4e97xyorssavybuq52.streamlit.app/",
    "https://uidaihackathon-uovw8keu8zjsfm9sgqeqta.streamlit.app/",]

def ping_apps():
    print(f"Starting ping cycle for {len(urls)} apps...")
    
    success_count = 0
    
    for url in urls:
        try:
            # We use a timeout to prevent the script from hanging if an app is down
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Success: {url}")
                success_count += 1
            else:
                print(f"⚠️ Warning: {url} returned status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error pinging {url}: {e}")
        
        # Small delay to be polite to the server
        time.sleep(1)

    print(f"\nFinished. Successfully pinged {success_count}/{len(urls)} apps.")

if __name__ == "__main__":
    ping_apps()