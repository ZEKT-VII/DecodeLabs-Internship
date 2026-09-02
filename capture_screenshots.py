"""Automated script to capture high-definition real screenshots of Project 1 UI."""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path(__file__).resolve().parent / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 960},
            device_scale_factor=2,  # 2x Retina sharpness
            color_scheme="dark",
        )
        page = context.new_page()

        print("[*] Navigating to Project 1...")
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        time.sleep(1.5)

        # 1. Select the rich architectural session
        print("[*] Selecting session 694f64dc-d452-4646-8a94-6b09ceed7855...")
        page.evaluate("selectSession('694f64dc-d452-4646-8a94-6b09ceed7855')")
        time.sleep(2.0)

        path_chat = SCREENSHOT_DIR / "project1_chat_architecture.png"
        page.screenshot(path=str(path_chat))
        print(f"[+] 1. Saved Chat Architecture Screenshot: {path_chat}")

        # 2. Open Local Session Memory Drawer
        print("[*] Opening Local Memory Drawer...")
        page.evaluate("toggleLocalMemoryDrawer()")
        time.sleep(1.2)
        path_drawer = SCREENSHOT_DIR / "project1_dual_layer_memory.png"
        page.screenshot(path=str(path_drawer))
        print(f"[+] 2. Saved Dual-Layer Memory Drawer Screenshot: {path_drawer}")

        # Close the drawer
        page.evaluate("toggleLocalMemoryDrawer()")
        time.sleep(0.5)

        # 3. Switch to Settings Tab & Test Connection Probe
        print("[*] Switching to Settings Tab...")
        page.evaluate("switchSidebarTab('settings')")
        time.sleep(1.0)

        print("[*] Triggering Test Connection probe...")
        try:
            page.evaluate("handleTestConnection()")
            time.sleep(3.5)
        except Exception as e:
            print(f"[!] Probe evaluation warning: {e}")

        path_settings = SCREENSHOT_DIR / "project1_frontier_settings_vault.png"
        page.screenshot(path=str(path_settings))
        print(f"[+] 3. Saved Frontier Settings & AES Vault Screenshot: {path_settings}")

        # 4. Global Memory Tab
        print("[*] Switching to Global Memory Tab...")
        page.evaluate("switchSidebarTab('global')")
        time.sleep(1.0)
        path_global = SCREENSHOT_DIR / "project1_global_memory.png"
        page.screenshot(path=str(path_global))
        print(f"[+] 4. Saved Global Memory Screenshot: {path_global}")

        browser.close()
        print("[*] All screenshots captured successfully!")


if __name__ == "__main__":
    capture_all()
