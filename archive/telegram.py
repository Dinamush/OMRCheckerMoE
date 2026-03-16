from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def setup_driver():
    chrome_options = Options()
    # Use an existing Chrome user profile (ensure this path exists and is not in use)
    chrome_options.add_argument("user-data-dir=C:/Users/Administrator/chrome_profile")
    # Specify a remote debugging port
    chrome_options.add_argument("--remote-debugging-port=9222")
    # Other recommended options
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Hide the "Chrome is being controlled by automated test software" infobar
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def inject_overlay(driver):
    overlay_script = """
    (function() {
        // Helper function to trigger a download via an anchor element.
        function downloadURI(uri, filename) {
            var a = document.createElement("a");
            a.href = uri;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        // Add a download button overlay for video elements.
        function addDownloadButtonToVideo(video, index) {
            if (video.parentElement.querySelector('.download-overlay')) return;
            let btn = document.createElement("button");
            btn.innerText = "Download";
            btn.className = "download-overlay";
            btn.style.position = "absolute";
            btn.style.top = "10px";
            btn.style.right = "10px";
            btn.style.zIndex = "1000";
            btn.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
            btn.style.color = "#fff";
            btn.style.border = "none";
            btn.style.padding = "5px 10px";
            btn.style.cursor = "pointer";
            video.parentElement.style.position = "relative";
            video.parentElement.appendChild(btn);

            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                let src = video.currentSrc || video.src;
                if(src) {
                    downloadURI(src, "video_" + index + ".mp4");
                } else {
                    alert("No video source found!");
                }
            });
        }

        // Add a download button overlay for image elements.
        function addDownloadButtonToImage(img, index) {
            if (img.parentElement.querySelector('.download-overlay')) return;
            let btn = document.createElement("button");
            btn.innerText = "Download";
            btn.className = "download-overlay";
            btn.style.position = "absolute";
            btn.style.top = "10px";
            btn.style.right = "10px";
            btn.style.zIndex = "1000";
            btn.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
            btn.style.color = "#fff";
            btn.style.border = "none";
            btn.style.padding = "5px 10px";
            btn.style.cursor = "pointer";
            img.parentElement.style.position = "relative";
            img.parentElement.appendChild(btn);

            btn.addEventListener('click', function(e) {
                e.stopPropagation();
                let src = img.src;
                if(src) {
                    downloadURI(src, "image_" + index + ".jpg");
                } else {
                    alert("No image source found!");
                }
            });
        }

        // Process already loaded media.
        function processMedia() {
            document.querySelectorAll("video").forEach(function(video, index) {
                addDownloadButtonToVideo(video, index);
            });
            document.querySelectorAll("img").forEach(function(img, index) {
                addDownloadButtonToImage(img, index);
            });
        }

        // Create a global "Download All Media" button.
        let globalBtn = document.createElement("button");
        globalBtn.innerText = "Download All Media";
        globalBtn.style.position = "fixed";
        globalBtn.style.bottom = "20px";
        globalBtn.style.right = "20px";
        globalBtn.style.zIndex = "2000";
        globalBtn.style.padding = "10px 20px";
        globalBtn.style.backgroundColor = "red";
        globalBtn.style.color = "#fff";
        globalBtn.style.border = "none";
        globalBtn.style.cursor = "pointer";
        document.body.appendChild(globalBtn);

        globalBtn.addEventListener("click", function() {
            // Download all videos.
            document.querySelectorAll("video").forEach(function(video, index) {
                let src = video.currentSrc || video.src;
                if(src) {
                    downloadURI(src, "video_" + index + ".mp4");
                }
            });
            // Download all images.
            document.querySelectorAll("img").forEach(function(img, index) {
                let src = img.src;
                if(src) {
                    downloadURI(src, "image_" + index + ".jpg");
                }
            });
        });

        // Process current media elements.
        processMedia();

        // Use a MutationObserver to handle dynamically loaded media.
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        if (node.tagName === "VIDEO") {
                            addDownloadButtonToVideo(node, Date.now());
                        } else if (node.tagName === "IMG") {
                            addDownloadButtonToImage(node, Date.now());
                        } else {
                            let videos = node.querySelectorAll ? node.querySelectorAll("video") : [];
                            videos.forEach(function(video) {
                                addDownloadButtonToVideo(video, Date.now());
                            });
                            let imgs = node.querySelectorAll ? node.querySelectorAll("img") : [];
                            imgs.forEach(function(img) {
                                addDownloadButtonToImage(img, Date.now());
                            });
                        }
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();
    """
    driver.execute_script(overlay_script)
    print("Overlay and global 'Download All Media' button injected.")

def main():
    driver = setup_driver()
    # Open Telegram Web (adjust URL if needed, e.g., https://web.telegram.org/)
    driver.get("https://web.telegram.org/")
    print("Telegram Web loaded. Please ensure you're logged in and navigate to your target group manually.")
    
    # Allow time for manual login/navigation.
    time.sleep(30)  # Adjust as needed
    
    # Inject the overlay into the page.
    inject_overlay(driver)
    print("Overlay injected. You can click individual download buttons or the 'Download All Media' button at the bottom right.")
    
    # Keep the browser open to interact with the page.
    time.sleep(300)
    driver.quit()

if __name__ == "__main__":
    main()
