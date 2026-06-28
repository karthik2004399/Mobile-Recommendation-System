import pandas as pd
import numpy as np
import re
import pickle
import os

def clean_price(p):
    if isinstance(p, (int, float)): return float(p)
    nums = re.findall(r'\d+', str(p).replace(',', ''))
    return float(nums[0]) if nums else np.nan

def extract_features(df):
    structured = []
    
    for idx, row in df.iterrows():
        name = str(row.get('name', '')).lower()
        corpus = str(row.get('corpus', '')).lower()
        full_text = name + " " + corpus
        
        # 1. Basic Info
        phone_id = f"MOB_{idx:04d}"
        
        # Brand extraction
        brands = ['samsung', 'apple', 'oneplus', 'xiaomi', 'redmi', 'poco', 'realme', 'oppo', 'vivo', 'iqoo', 'motorola', 'infinix', 'tecno', 'nokia', 'google', 'nothing']
        brand = next((b.capitalize() for b in brands if b in name), 'Unknown')
        
        model_name = row.get('name', 'Unknown')
        price = clean_price(row.get('price'))
        
        # 2. Performance
        processor = np.nan
        proc_match = re.search(r'(snapdragon\s*[\w\s]+|dimensity\s*\d+|exynos\s*\d+|a\d+\s*bionic|tensor\s*g\d|helio\s*g\d+|unisoc\s*t\d+)', full_text)
        if proc_match: processor = proc_match.group(1).title()
        
        ram_gb = np.nan
        ram_match = re.search(r'(\d+)\s*gb\s*ram', full_text)
        if ram_match: ram_gb = int(ram_match.group(1))
        
        storage_gb = np.nan
        rom_match = re.search(r'(\d+)\s*gb\s*(?:rom|storage)', full_text)
        if rom_match: storage_gb = int(rom_match.group(1))
        elif re.search(r'1\s*tb\s*(?:rom|storage)', full_text): storage_gb = 1024
        
        # 3. Camera
        rear_camera_mp = np.nan
        rear_match = re.search(r'(\d+)mp.*(?:rear|primary)', full_text)
        if rear_match: rear_camera_mp = int(rear_match.group(1))
        else:
            # Fallback grab the highest MP
            mps = re.findall(r'(\d+)mp', full_text)
            if mps: rear_camera_mp = max([int(m) for m in mps])
            
        front_camera_mp = np.nan
        front_match = re.search(r'(\d+)mp.*front', full_text)
        if front_match: front_camera_mp = int(front_match.group(1))
        
        camera_sensor = np.nan
        if 'imx' in full_text: camera_sensor = 'Sony IMX'
        elif 'gn' in full_text or 'isocell' in full_text: camera_sensor = 'Samsung ISOCELL'
        
        ois = 'Yes' if 'ois' in full_text else 'No'
        
        video_quality = np.nan
        if '8k' in full_text: video_quality = '8K'
        elif '4k' in full_text: video_quality = '4K'
        elif '1080p' in full_text: video_quality = '1080p'
        
        # 4. Battery
        battery_mah = np.nan
        bat_match = re.search(r'(\d+)\s*mah', full_text)
        if bat_match: battery_mah = int(bat_match.group(1))
        
        fast_charging_watt = np.nan
        charge_match = re.search(r'(\d+)\s*w\s*(?:fast|super|vooc|dart|flash|sonic)', full_text)
        if charge_match: fast_charging_watt = int(charge_match.group(1))
        
        # 5. Display
        display_type = 'LCD'
        if 'amoled' in full_text: display_type = 'AMOLED'
        elif 'oled' in full_text: display_type = 'OLED'
        elif 'poled' in full_text: display_type = 'pOLED'
        
        screen_size = np.nan
        screen_match = re.search(r'(\d+\.\d+)\s*inch', full_text)
        if screen_match: screen_size = float(screen_match.group(1))
        
        refresh_rate = np.nan
        rr_match = re.search(r'(\d+)\s*hz', full_text)
        if rr_match: refresh_rate = int(rr_match.group(1))
        
        resolution = np.nan
        if 'fhd+' in full_text or 'full hd+' in full_text: resolution = 'FHD+'
        elif 'qhd+' in full_text or 'quad hd+' in full_text: resolution = 'QHD+'
        elif 'hd+' in full_text: resolution = 'HD+'
        
        # 6. Connectivity
        is_5g = 'Yes' if '5g' in full_text else 'No'
        
        bluetooth_version = np.nan
        bt_match = re.search(r'bluetooth\s*(5\.\d)', full_text)
        if bt_match: bluetooth_version = f"v{bt_match.group(1)}"
        
        # 7. Build & Design
        weight = np.nan
        wt_match = re.search(r'(\d{3})\s*g\b', full_text)
        if wt_match: weight = int(wt_match.group(1))
        
        thickness = np.nan
        th_match = re.search(r'(\d+\.\d+)\s*mm', full_text)
        if th_match: thickness = float(th_match.group(1))
        
        ip_rating = np.nan
        ip_match = re.search(r'(ip\d{2})', full_text)
        if ip_match: ip_rating = ip_match.group(1).upper()
        
        # 8. Audio & Extras
        stereo_speakers = 'Yes' if 'stereo' in full_text or 'dual speaker' in full_text else 'No'
        headphone_jack = 'Yes' if '3.5mm' in full_text or 'audio jack' in full_text else 'No'
        
        fingerprint_type = np.nan
        if 'in-display' in full_text or 'under display' in full_text: fingerprint_type = 'In-Display'
        elif 'side' in full_text and 'fingerprint' in full_text: fingerprint_type = 'Side-mounted'
        elif 'rear' in full_text and 'fingerprint' in full_text: fingerprint_type = 'Rear-mounted'
        
        # 9. Software
        os_type = 'iOS' if 'apple' in brand.lower() else 'Android'
        
        ui_type = np.nan
        if 'oneui' in full_text or 'one ui' in full_text: ui_type = 'OneUI'
        elif 'oxygen' in full_text: ui_type = 'OxygenOS'
        elif 'miui' in full_text: ui_type = 'MIUI'
        elif 'hyperos' in full_text: ui_type = 'HyperOS'
        elif 'coloros' in full_text: ui_type = 'ColorOS'
        elif 'funtouch' in full_text: ui_type = 'Funtouch OS'
        
        android_version = np.nan
        and_match = re.search(r'android\s*(\d{2})', full_text)
        if and_match: android_version = f"Android {and_match.group(1)}"
        
        # Append as dictionary
        structured.append({
            'phone_id': phone_id,
            'brand': brand,
            'model_name': model_name,
            'price': price,
            'processor': processor,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'rear_camera_mp': rear_camera_mp,
            'front_camera_mp': front_camera_mp,
            'camera_sensor': camera_sensor,
            'ois': ois,
            'video_quality': video_quality,
            'battery_mah': battery_mah,
            'fast_charging_watt': fast_charging_watt,
            'display_type': display_type,
            'screen_size': screen_size,
            'refresh_rate': refresh_rate,
            'resolution': resolution,
            '5g': is_5g,
            'bluetooth_version': bluetooth_version,
            'weight': weight,
            'thickness': thickness,
            'ip_rating': ip_rating,
            'stereo_speakers': stereo_speakers,
            'headphone_jack': headphone_jack,
            'fingerprint_type': fingerprint_type,
            'os': os_type,
            'ui_type': ui_type,
            'android_version': android_version
        })
        
    return pd.DataFrame(structured)

if __name__ == "__main__":
    print("Loading raw dataframe...")
    df = pickle.load(open('src/model/dataframe.pkl', 'rb'))
    print(f"Original shape: {df.shape}")
    
    print("Extracting 26+ specific columns...")
    new_df = extract_features(df)
    
    # Create directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    out_path = 'data/processed/smartphones_structured_26cols.csv'
    new_df.to_csv(out_path, index=False)
    
    print(f"\nExtraction complete! Saved to {out_path}")
    print(f"New shape: {new_df.shape}")
    print("\nSample Data:")
    print(new_df.head(3).to_string())
    print("\nMissing values breakdown:")
    print(new_df.isna().sum())
