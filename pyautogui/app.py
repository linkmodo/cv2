# app.py
import streamlit as st
import pyautogui
import time
import platform
import keyboard

# Safety configuration
pyautogui.PAUSE = 0.5
pyautogui.FAILSAFE = True
stop_drawing = False

# Platform detection
IS_MAC = 'mac' in platform.platform().lower()
HOTKEY_MOD = 'command' if IS_MAC else 'ctrl'

def main():
    st.title("PyAutoGUI Web Controller")
    st.warning("Use with caution! This controls your actual mouse/keyboard.")
    
    tab1, tab2, tab3 = st.tabs(["Browser Control", "Mouse Demo", "Keyboard Demo"])
    
    with tab1:
        st.header("Browser Automation")
        url = st.text_input("Enter URL:", "[https://pyautogui.readthedocs.io](https://pyautogui.readthedocs.io)")
        if st.button("Open in Browser"):
            with st.spinner("Opening browser..."):
                try:
                    pyautogui.hotkey(HOTKEY_MOD, 't')
                    time.sleep(1)
                    pyautogui.write(url)
                    pyautogui.press('enter')
                    st.success("Done!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Mouse Control")
        if st.button("Draw Spiral"):
            stop_drawing = False  # Reset flag
            with st.spinner("Drawing (Press ESC to stop)..."):
                try:
                    time.sleep(3)  # Give time to switch to drawing app
                    distance = 200
                    while distance > 0 and not stop_drawing:
                        if keyboard.is_pressed('esc'):
                            stop_drawing = True
                            break
                        pyautogui.drag(distance, 0, duration=0.2)
                        distance -= 5
                        pyautogui.drag(0, distance, duration=0.2)
                        distance -= 5
                    if stop_drawing:
                        st.warning("Drawing stopped by user!")
                    else:
                        st.success("Drawing complete!")
                except pyautogui.FailSafeException:
                    st.error("Emergency stop triggered!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("Keyboard Input")
        text = st.text_input("Text to type:")
        if st.button("Type Text"):
            with st.spinner("Typing..."):
                try:
                    time.sleep(3)  # Give time to focus window
                    pyautogui.write(text, interval=0.1)
                    st.success("Typing complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
