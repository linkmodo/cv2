import streamlit as st
import time
import sys
import platform
import pyautogui

# ========== CONFIGURATION ==========
# Safety settings
try:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
except Exception as e:
    st.error(f"PyAutoGUI initialization failed: {str(e)}")
    st.stop()

# Platform detection
try:
    IS_MAC = 'mac' in platform.platform().lower()
    HOTKEY_MOD = 'command' if IS_MAC else 'ctrl'
except Exception as e:
    st.error(f"Platform detection failed: {str(e)}")
    st.stop()

# Timing parameters
DEFAULT_DELAY = 0.5
FOCUS_DELAY = 1.0
TYPE_INTERVAL = 0.01

def safe_automation(func):
    """Decorator for adding error handling to automation functions."""
    def wrapper(*args, **kwargs):
        try:
            # Verify screen resolution
            screen_width, screen_height = pyautogui.size()
            if screen_width < 1024 or screen_height < 768:
                raise ValueError("Screen resolution too small for reliable automation")
                
            return func(*args, **kwargs)
            
        except pyautogui.FailSafeException:
            st.error("Emergency stop triggered!")
            sys.exit(1)
            
        except Exception as e:
            st.error(f"Automation failed: {str(e)}")
            sys.exit(1)
            
    return wrapper

@safe_automation
def open_new_tab():
    """Open a new browser tab."""
    with st.spinner("Opening new tab..."):
        pyautogui.hotkey(HOTKEY_MOD, 't')
        time.sleep(FOCUS_DELAY)

@safe_automation
def navigate_to_url(url):
    """Navigate to specified URL in browser."""
    with st.spinner(f"Navigating to {url}..."):
        pyautogui.hotkey(HOTKEY_MOD, 'l')
        time.sleep(0.2)
        pyautogui.write(url, interval=TYPE_INTERVAL)
        time.sleep(0.1)
        pyautogui.press('enter')
        time.sleep(FOCUS_DELAY)

@safe_automation
def draw_spiral():
    """Draw a shrinking spiral."""
    with st.spinner("Drawing spiral (move mouse to top-left to abort)..."):
        time.sleep(DEFAULT_DELAY)
        distance = 200
        while distance > 0:
            pyautogui.drag(distance, 0, button='left', duration=0.2)
            distance -= 5
            pyautogui.drag(0, distance, button='left', duration=0.2)
            distance -= 5
            pyautogui.drag(-distance, 0, button='left', duration=0.2)
            distance -= 5
            pyautogui.drag(0, -distance, button='left', duration=0.2)
            distance -= 5

def main():
    st.title("HCI Automation Controller")
    st.warning("WARNING: This app controls your actual mouse and keyboard!")
    
    tab1, tab2 = st.tabs(["Browser Control", "Mouse Demo"])
    
    with tab1:
        st.header("Browser Automation")
        url = st.text_input("Enter URL:", "https://pyautogui.readthedocs.io")
        if st.button("Open URL"):
            open_new_tab()
            navigate_to_url(url)
            st.success("Done!")
    
    with tab2:
        st.header("Mouse Drawing")
        if st.button("Draw Spiral"):
            draw_spiral()
            st.success("Drawing complete!")

if __name__ == "__main__":
    main()
