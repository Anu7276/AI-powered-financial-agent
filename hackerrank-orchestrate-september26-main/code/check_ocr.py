for mod in ['pytesseract', 'easyocr', 'PIL', 'cv2']:
    try:
        __import__(mod)
        print(f"{mod}: INSTALLED")
    except ImportError:
        print(f"{mod}: NOT INSTALLED")
