import pandas as pd
import numpy as np
import re

class RecommenderEngine:
    def __init__(self, df, similarity_matrix):
        self.df = df.copy()
        self.similarity_matrix = similarity_matrix
        self._preprocess_data()
        
    def _preprocess_data(self):
        # Extract RAM
        def extract_ram(text):
            text = str(text).lower()
            match = re.search(r'(\d+)\s*gb\s*ram|ram\s*(\d+)', text)
            if match:
                val = int(match.group(1) or match.group(2))
                return val if val <= 24 else 0
            return 0
        
        def extract_storage(text):
            text = str(text).lower()
            match = re.search(r'(\d+)\s*gb\s*(?:rom|storage)|storage\s*(\d+)', text)
            if match:
                val = int(match.group(1) or match.group(2))
                return val if val <= 1024 else 0
            return 0

        # Extract Battery
        def extract_battery(text):
            text = str(text).lower()
            val = 0
            match = re.search(r'capacity(\d+)', text)
            if match: val = int(match.group(1))
            else:
                match = re.search(r'(\d+)mah', text)
                if match: val = int(match.group(1))
            
            return val if val <= 10000 else 0 # Cap at 10,000mAh

        # Extract Camera (Max MP)
        def extract_camera(text):
            text = str(text).lower()
            mps = re.findall(r'(\d+)mp', text)
            if mps:
                val = max([int(m) for m in mps])
                return val if val <= 200 else 200 # Cap at 200MP
            return 0

        # Extract Screen Size
        def extract_screen(text):
            text = str(text).lower()
            match = re.search(r'(\d+\.?\d*)\s*inch', text)
            if match: return float(match.group(1))
            return 0.0

        # Clean Price
        def clean_price(p):
            if not p or str(p).lower() in ('nan', 'none', ''):
                return 0.0
            
            if isinstance(p, (int, float)):
                val = float(p)
            else:
                p_str = str(p).replace(',', '')
                clean_str = re.sub(r'[^\d.]', '', p_str)
                try:
                    val = float(clean_str)
                except:
                    return 0.0
            
            # Universal USD Fallback: If price is suspiciously low, convert it.
            if 0 < val < 5000:
                return val * 83.0
            return val

        # Improved Extraction Logic with Prioritization
        if 'ram_gb' in self.df.columns:
            self.df['ram'] = pd.to_numeric(self.df['ram_gb'], errors='coerce').fillna(self.df['corpus'].apply(extract_ram))
        else:
            self.df['ram'] = self.df['corpus'].apply(extract_ram)

        if 'storage_gb' in self.df.columns:
            self.df['storage'] = pd.to_numeric(self.df['storage_gb'], errors='coerce').fillna(self.df['corpus'].apply(extract_storage))
        else:
            self.df['storage'] = self.df['corpus'].apply(extract_storage)

        if 'battery_mah' in self.df.columns:
            self.df['battery'] = pd.to_numeric(self.df['battery_mah'], errors='coerce').fillna(self.df['corpus'].apply(extract_battery))
        else:
            self.df['battery'] = self.df['corpus'].apply(extract_battery)

        if 'rear_camera_mp' in self.df.columns:
            self.df['camera'] = pd.to_numeric(self.df['rear_camera_mp'], errors='coerce').fillna(self.df['corpus'].apply(extract_camera))
        else:
            self.df['camera'] = self.df['corpus'].apply(extract_camera)

        if 'screen_size' in self.df.columns:
            self.df['screen'] = pd.to_numeric(self.df['screen_size'], errors='coerce').fillna(self.df['corpus'].apply(extract_screen))
        else:
            self.df['screen'] = self.df['corpus'].apply(extract_screen)
        self.df['price_numeric'] = self.df['price'].apply(clean_price)
        self.df['ratings'] = pd.to_numeric(self.df['ratings'], errors='coerce').fillna(0.0)
        
        # Manual Scaling (Min-Max)
        features = ['ram', 'storage', 'price_numeric', 'ratings', 'battery', 'camera', 'screen']
        self.df_scaled = self.df[features].copy()
        for feature in features:
            f_min = self.df_scaled[feature].min()
            f_max = self.df_scaled[feature].max()
            if f_max > f_min:
                self.df_scaled[feature] = (self.df_scaled[feature] - f_min) / (f_max - f_min)
            else:
                self.df_scaled[feature] = 0.0

        # ── Compute 1–10 Feature Scores for Preference-Based Filtering ──────────
        self._compute_scores()

        # Remove duplicate phones by name to prevent repeated recommendations
        if 'name' in self.df.columns:
            self.df['base_name'] = self.df['name'].astype(str).str.replace(r'\s*[\(\[].*', '', regex=True).str.strip().str.lower()
            dup_mask = ~self.df['name'].duplicated(keep='first')
            self.df = self.df[dup_mask].reset_index(drop=True)
            self.df_scaled = self.df_scaled[dup_mask].reset_index(drop=True)
            # NOTE: We intentionally do NOT slice similarity_matrix here.
            # The matrix was built on the original pickle-loaded df and may have
            # a different shape after the merge in app.py. The individual
            # recommendation algorithms already call .drop_duplicates() on their
            # output, and bounds checks below keep index-based methods safe.

    def _compute_scores(self):
        """Derive performance/camera/battery/display scores (1–10) from raw specs."""
        df = self.df
        corpus = df['corpus'].astype(str).str.lower()

        def scale_to_10(series, low, high):
            """Min-max scale a series into 1–10 range given expected low/high."""
            clamped = series.clip(low, high)
            return 1 + 9 * (clamped - low) / (high - low)

        # --- Performance Score (RAM + processor tier) ---
        proc_score = pd.Series(4.0, index=df.index)  # Default low-mid
        
        # Elite Tier
        proc_score[corpus.str.contains(r'snapdragon\s*8|sd\s*8|dimensity\s*9[0-9]{3}|a1[6-9]\s*bionic|tensor\s*g[234]', na=False)] = 9.8
        # High Tier
        proc_score[corpus.str.contains(r'snapdragon\s*7|sd\s*7|dimensity\s*8[0-9]{3}|exynos\s*2[0-9]{3}|tensor\s*g1', na=False)] = 8.5
        # Mid Tier
        proc_score[corpus.str.contains(r'snapdragon\s*6|sd\s*6|dimensity\s*7[0-9]{3}|helio\s*g9[0-9]|g8[0-9]', na=False)] = 7.0
        # Budget Tier
        proc_score[corpus.str.contains(r'snapdragon\s*4|sd\s*4|dimensity\s*6[0-9]{3}|helio\s*g[0-7][0-9]', na=False)] = 5.0
        # Entry Tier
        proc_score[corpus.str.contains(r'helio\s*[ap]|unisoc|quad\s*core', na=False)] = 3.0

        ram_score = scale_to_10(df['ram'].fillna(0), 4, 16) # 4GB is baseline
        self.df['performance_score'] = ((proc_score * 0.65) + (ram_score * 0.35)).clip(1, 10).round(1)

        # --- Camera Score (MP + OIS + sensor quality) ---
        cam_mp = df['camera'].fillna(0)
        # Re-scale: 50MP should be a good score (~7.5), not a low one.
        cam_base = scale_to_10(cam_mp, 8, 108) # Use 108 as the "high" standard
        cam_bonus = pd.Series(0.0, index=df.index)
        if 'ois' in df.columns:
            cam_bonus += df['ois'].astype(str).str.contains('Yes', case=False, na=False).astype(float) * 1.5
        cam_bonus += corpus.str.contains(r'imx\d+|isocell|sony|periscope|telephoto|night mode|zeiss|leica|hasselblad', na=False).astype(float) * 1.2
        cam_bonus += corpus.str.contains(r'\b200mp|\b108mp', na=False).astype(float) * 1.0
        
        if 'front_camera_mp' in df.columns:
            front_score = scale_to_10(df['front_camera_mp'].fillna(0), 8, 32)
            self.df['camera_score'] = ((cam_base * 0.7) + (front_score * 0.1) + (cam_bonus * 0.2)).clip(1, 10).round(1)
        else:
            self.df['camera_score'] = ((cam_base * 0.8) + (cam_bonus * 0.2)).clip(1, 10).round(1)

        # --- Battery Score (mAh + fast charging) ---
        bat_mah = df['battery'].fillna(0)
        bat_base = scale_to_10(bat_mah, 3000, 6000) # 5000mAh is high standard
        fc_bonus = pd.Series(0.0, index=df.index)
        if 'fast_charging_watt' in df.columns:
            fc_bonus += scale_to_10(df['fast_charging_watt'].fillna(0), 0, 120) * 0.4
        else:
            fc_bonus += corpus.str.contains(r'\d{2,3}\s*w|fast charg|turbo charg', na=False).astype(float) * 2.0
        self.df['battery_score'] = ((bat_base * 0.7) + (fc_bonus * 0.3)).clip(1, 10).round(1)

        # --- Display Score (AMOLED + refresh rate + screen size) ---
        disp_base = pd.Series(5.0, index=df.index)
        if 'display_type' in df.columns:
            disp_base[df['display_type'].astype(str).str.contains('Super AMOLED', case=False, na=False)] = 9.0
            disp_base[df['display_type'].astype(str).str.contains('AMOLED|OLED', case=False, na=False)] = 8.5
            disp_base[df['display_type'].astype(str).str.contains('IPS|LCD', case=False, na=False)] = 6.0
        else:
            disp_base[corpus.str.contains('super amoled', na=False)] = 9.0
            disp_base[corpus.str.contains('amoled|oled', na=False)] = 8.5
        hz_bonus = pd.Series(0.0, index=df.index)
        if 'refresh_rate' in df.columns:
            hz_bonus = scale_to_10(df['refresh_rate'].fillna(60), 60, 165) * 0.5
        else:
            hz_bonus[corpus.str.contains(r'144\s*hz|165\s*hz|240\s*hz', na=False)] = 1.5
            hz_bonus[corpus.str.contains(r'120\s*hz', na=False)] = 1.0
            hz_bonus[corpus.str.contains(r'90\s*hz', na=False)] = 0.5
        screen_bonus = scale_to_10(df['screen'].fillna(6.0), 5.0, 7.2) * 0.3
        self.df['display_score'] = (disp_base + hz_bonus + screen_bonus).clip(1, 10).round(1)

    def rule_based(self, budget, min_ram=0, min_storage=0, min_battery=0,
                   min_camera=0, min_front_camera=0, brand='Any',
                   network='Any', display_type='Any',
                   fast_charging=False, min_refresh_rate=0):

        df = self.df.copy()

        # 1. Hard Constraints (Must Match)
        # Always respect Budget
        mask = (df['price_numeric'] >= 50) & (df['price_numeric'] <= budget)

        # Respect Brand strictly if selected and available
        if brand != 'Any':
            brand_mask = mask & df['name'].str.contains(brand, case=False, na=False)
            if len(df[brand_mask]) > 0:
                mask = brand_mask

        # Respect Network strictly if selected and available
        if network == '5G Only':
            if '5g' in df.columns:
                net_mask = mask & df['5g'].astype(str).str.contains('Yes|True|1', case=False, na=False)
            else:
                net_mask = mask & df['corpus'].str.contains(r'\b5g\b', case=False, na=False)
            if len(df[net_mask]) > 0: mask = net_mask
        elif network == '4G Only':
            if '5g' in df.columns:
                net_mask = mask & ~df['5g'].astype(str).str.contains('Yes|True|1', case=False, na=False)
            else:
                net_mask = mask & ~df['corpus'].str.contains(r'\b5g\b', case=False, na=False)
            if len(df[net_mask]) > 0: mask = net_mask

        # Respect Display Type strictly if selected and available
        if display_type != 'Any':
            if 'display_type' in df.columns:
                disp_mask = mask & df['display_type'].astype(str).str.contains(display_type, case=False, na=False)
            else:
                disp_mask = mask & df['corpus'].str.contains(display_type, case=False, na=False)
            if len(df[disp_mask]) > 0: mask = disp_mask

        pool = df[mask].copy()
        if pool.empty:
            return pd.DataFrame()

        # 2. Soft Constraints (Scoring System to guarantee top 10)
        pool['match_score'] = 0.0

        ram_col = pool['ram_gb'].fillna(pool['ram']) if 'ram_gb' in pool.columns else pool['ram']
        pool.loc[ram_col.fillna(0) >= min_ram, 'match_score'] += 1.0

        st_col = pool['storage_gb'].fillna(pool['storage']) if 'storage_gb' in pool.columns else pool['storage']
        pool.loc[st_col.fillna(0) >= min_storage, 'match_score'] += 1.0

        bat_col = pool['battery_mah'].fillna(pool['battery']) if 'battery_mah' in pool.columns else pool['battery']
        pool.loc[bat_col.fillna(0) >= min_battery, 'match_score'] += 1.0

        cam_col = pool['rear_camera_mp'].fillna(pool['camera']) if 'rear_camera_mp' in pool.columns else pool['camera']
        pool.loc[cam_col.fillna(0) >= min_camera, 'match_score'] += 1.0

        if min_front_camera > 0 and 'front_camera_mp' in pool.columns:
            pool.loc[pool['front_camera_mp'].fillna(0) >= min_front_camera, 'match_score'] += 1.0

        if fast_charging:
            if 'fast_charging_watt' in pool.columns:
                pool.loc[pool['fast_charging_watt'].fillna(0) > 18, 'match_score'] += 1.0
            else:
                pool.loc[pool['corpus'].str.contains(r'fast charg|quick charg|turbo charg|\d{2,3}\s*w', case=False, na=False), 'match_score'] += 1.0

        if min_refresh_rate > 60 and 'refresh_rate' in pool.columns:
            pool.loc[pool['refresh_rate'].fillna(0) >= min_refresh_rate, 'match_score'] += 1.0
        elif min_refresh_rate > 60:
            hz_pattern = '|'.join([f'{r}hz' for r in [90,120,144,165,240] if r >= min_refresh_rate])
            if hz_pattern:
                pool.loc[pool['corpus'].str.contains(hz_pattern, case=False, na=False), 'match_score'] += 1.0

        # 3. Compute an overall spec score to tie-break phones
        pool['spec_score'] = (
            pool['performance_score'].fillna(0) + 
            pool['camera_score'].fillna(0) + 
            pool['battery_score'].fillna(0) + 
            pool['display_score'].fillna(0)
        )

        # 4. Sort by match score -> spec score (highest specs) -> highest ratings
        return pool.sort_values(
            by=['match_score', 'spec_score', 'ratings'], 
            ascending=[False, False, False]
        ).drop_duplicates(subset=['base_name'], keep='first').head(10).drop(columns=['match_score', 'spec_score'])

    def rule_based_enhanced(self, min_budget=5000, max_budget=200000, min_ram=0, min_storage=0, 
                           min_battery=0, min_camera=0, min_front_camera=0, brands=None,
                           network='Any', display_types=None, fast_charging=False, 
                           min_refresh_rate=0, min_rating=0.0, sort_option='Relevance'):

        df = self.df.copy()
        
        # Default values
        if brands is None:
            brands = []
        if display_types is None:
            display_types = []

        # 1. Hard Constraints (Must Match) - Applied in specified order
        
        # 1.1 Price Range Filter (First priority)
        mask = (df['price_numeric'] >= min_budget) & (df['price_numeric'] <= max_budget)

        # 1.2 Multi-Brand Filter (OR condition) - Second priority
        if brands:
            brand_masks = []
            for brand in brands:
                brand_mask = df['name'].str.contains(brand, case=False, na=False)
                brand_masks.append(brand_mask)
            if brand_masks:
                combined_brand_mask = brand_masks[0]
                for m in brand_masks[1:]:
                    combined_brand_mask |= m
                mask &= combined_brand_mask

        # 1.3 Single Rating Filter (>= selected rating) - Third priority
        if min_rating > 0.0:
            mask &= df['ratings'] >= min_rating

        # 1.4 Multi-Display Type Filter (OR condition) - Fourth priority
        if display_types:
            display_masks = []
            for display_type in display_types:
                if 'display_type' in df.columns:
                    disp_mask = df['display_type'].astype(str).str.contains(display_type, case=False, na=False)
                else:
                    disp_mask = df['corpus'].str.contains(display_type, case=False, na=False)
                display_masks.append(disp_mask)
            if display_masks:
                combined_display_mask = display_masks[0]
                for m in display_masks[1:]:
                    combined_display_mask |= m
                mask &= combined_display_mask

        # 1.5 Network Filter
        if network == '5G Only':
            if '5g' in df.columns:
                net_mask = df['5g'].astype(str).str.contains('Yes|True|1', case=False, na=False)
            else:
                net_mask = df['corpus'].str.contains(r'\b5g\b', case=False, na=False)
            mask &= net_mask
        elif network == '4G Only':
            if '5g' in df.columns:
                net_mask = ~df['5g'].astype(str).str.contains('Yes|True|1', case=False, na=False)
            else:
                net_mask = ~df['corpus'].str.contains(r'\b5g\b', case=False, na=False)
            mask &= net_mask

        pool = df[mask].copy()
        if pool.empty:
            return pd.DataFrame()

        # 2. Soft Constraints (Scoring System) - For additional specifications
        pool['match_score'] = 0.0

        ram_col = pool['ram_gb'].fillna(pool['ram']) if 'ram_gb' in pool.columns else pool['ram']
        pool.loc[ram_col.fillna(0) >= min_ram, 'match_score'] += 1.0

        st_col = pool['storage_gb'].fillna(pool['storage']) if 'storage_gb' in pool.columns else pool['storage']
        pool.loc[st_col.fillna(0) >= min_storage, 'match_score'] += 1.0

        bat_col = pool['battery_mah'].fillna(pool['battery']) if 'battery_mah' in pool.columns else pool['battery']
        pool.loc[bat_col.fillna(0) >= min_battery, 'match_score'] += 1.0

        cam_col = pool['rear_camera_mp'].fillna(pool['camera']) if 'rear_camera_mp' in pool.columns else pool['camera']
        pool.loc[cam_col.fillna(0) >= min_camera, 'match_score'] += 1.0

        if min_front_camera > 0 and 'front_camera_mp' in pool.columns:
            pool.loc[pool['front_camera_mp'].fillna(0) >= min_front_camera, 'match_score'] += 1.0

        if fast_charging:
            if 'fast_charging_watt' in pool.columns:
                pool.loc[pool['fast_charging_watt'].fillna(0) > 18, 'match_score'] += 1.0
            else:
                pool.loc[pool['corpus'].str.contains(r'fast charg|quick charg|turbo charg|\d{2,3}\s*w', case=False, na=False), 'match_score'] += 1.0

        if min_refresh_rate > 60 and 'refresh_rate' in pool.columns:
            pool.loc[pool['refresh_rate'].fillna(0) >= min_refresh_rate, 'match_score'] += 1.0
        elif min_refresh_rate > 60:
            hz_pattern = '|'.join([f'{r}hz' for r in [90,120,144,165,240] if r >= min_refresh_rate])
            if hz_pattern:
                pool.loc[pool['corpus'].str.contains(hz_pattern, case=False, na=False), 'match_score'] += 1.0

        # 3. Compute spec score for tie-breaking
        pool['spec_score'] = (
            pool['performance_score'].fillna(0) + 
            pool['camera_score'].fillna(0) + 
            pool['battery_score'].fillna(0) + 
            pool['display_score'].fillna(0)
        )

        # 4. Apply Sorting based on user selection
        if sort_option == 'Price: Low to High':
            pool = pool.sort_values(by=['price_numeric'], ascending=True)
        elif sort_option == 'Price: High to Low':
            pool = pool.sort_values(by=['price_numeric'], ascending=False)
        elif sort_option == 'Rating: High to Low':
            pool = pool.sort_values(by=['ratings'], ascending=False)
        elif sort_option == 'Popularity (Rating Count)':
            # Use ratings as a proxy for popularity (higher rating count = more popular)
            pool = pool.sort_values(by=['ratings', 'price_numeric'], ascending=[False, True])
        else:  # Relevance (default)
            pool = pool.sort_values(
                by=['match_score', 'spec_score', 'ratings'], 
                ascending=[False, False, False]
            )

        return pool.drop_duplicates(subset=['base_name'], keep='first').head(10).drop(columns=['match_score', 'spec_score'])


    def persona_based(self, persona_idx, sort_option='Relevance'):
        # Base pool: exclude feature phones (price < 50)
        base = self.df[self.df['price_numeric'] >= 50].copy()

        # --- Helpers ---
        def col_num(col):
            """Return a numeric Series for col, falling back to 0."""
            if col in base.columns:
                return pd.to_numeric(base[col], errors='coerce').fillna(0)
            return pd.Series(0, index=base.index)

        def kw_match(col, pattern):
            """Return boolean mask: keyword search on a column."""
            if col in base.columns:
                return base[col].astype(str).str.contains(pattern, case=False, na=False)
            return pd.Series(False, index=base.index)

        # Prefer structured columns, fall back to corpus-extracted ones
        ram      = col_num('ram_gb').where(col_num('ram_gb') > 0, base['ram'])
        battery  = col_num('battery_mah').where(col_num('battery_mah') > 0, base['battery'])
        storage  = col_num('storage_gb').where(col_num('storage_gb') > 0, base['storage'])
        camera   = col_num('rear_camera_mp').where(col_num('rear_camera_mp') > 0, base['camera'])
        f_camera = col_num('front_camera_mp')
        screen   = col_num('screen_size').where(col_num('screen_size') > 0, base['screen'])
        price    = base['price_numeric']

        def try_filters(masks, min_results=5):
            """Apply masks one-by-one, relax until >= min_results remain."""
            result = base.copy()
            for m in masks:
                candidate = result[m]
                if len(candidate) >= min_results:
                    result = candidate
            return result

        # ── Persona Logic ──────────────────────────────────────────────────────
        if persona_idx == 1:  # 🎮 Gaming
            # Strict: 8GB+ RAM, Snapdragon 7/8 series or Dimensity, 120Hz+
            m_ram  = ram >= 8
            m_proc = kw_match('processor', r'Snapdragon\s*[78]|Dimensity\s*[89]\d{2}|Dimensity\s*\d{4}')
            m_proc |= kw_match('corpus', r'snapdragon\s*[78]\d{2}|dimensity\s*[89]\d{2}|dimensity\s*\d{4}')
            m_hz   = kw_match('corpus', r'120\s*hz|144\s*hz|165\s*hz|240\s*hz')
            m_cool = kw_match('corpus', r'gaming|cooling|vapor chamber|liquid cool|heat pipe')
            strict = base[m_ram & (m_proc | m_hz | m_cool)]
            if len(strict) < 5:
                strict = base[m_ram & (m_proc | m_cool | m_hz)]
            if len(strict) < 5:
                strict = base[m_ram]
            result = strict

        elif persona_idx == 2:  # 📱 Normal / Daily Use
            # 4–8GB RAM, 4000mAh+, smooth chipset, reasonable price
            m_ram  = (ram >= 4) & (ram <= 8)
            m_bat  = battery >= 4000
            m_chip = kw_match('corpus', r'snapdragon|dimensity|helio\s*g|exynos|tensor')
            strict = base[m_ram & m_bat & m_chip]
            if len(strict) < 5:
                strict = base[m_ram & m_bat]
            if len(strict) < 5:
                strict = base[m_ram]
            result = strict

        elif persona_idx == 3:  # 📸 Photography
            # 48MP+ rear, OIS preferred, night mode / sensor keywords
            m_cam  = camera >= 48
            m_ois  = kw_match('ois', r'Yes') | kw_match('corpus', r'\bois\b|optical image stab')
            m_kw   = kw_match('corpus', r'night mode|night vision|imx\d+|isocell|gn\d|lytia|periscope|telephoto')
            strict = base[m_cam & (m_ois | m_kw)]
            if len(strict) < 5:
                strict = base[m_cam]
            if len(strict) < 5:
                strict = base[camera >= 32]
            result = strict

        elif persona_idx == 4:  # 💼 Business / Productivity
            # 8GB+ RAM, 4500mAh+, security/clean UI, smooth performance
            m_ram  = ram >= 8
            m_bat  = battery >= 4500
            m_sec  = kw_match('corpus', r'security|knox|enterprise|biometric|fingerprint|face unlock')
            m_ui   = kw_match('corpus', r'clean|stock|one ui|miui|oxygen|coloros')
            strict = base[m_ram & m_bat & (m_sec | m_ui)]
            if len(strict) < 5:
                strict = base[m_ram & m_bat]
            if len(strict) < 5:
                strict = base[m_ram]
            result = strict

        elif persona_idx == 5:  # 🎬 Entertainment / Media
            # AMOLED, stereo speakers, 6.5"+ screen, 4500mAh+
            m_disp   = kw_match('display_type', r'AMOLED|Super AMOLED|OLED') | kw_match('corpus', r'amoled|oled|super amoled')
            m_stereo = kw_match('stereo_speakers', r'Yes') | kw_match('corpus', r'stereo|dolby atmos|dual speaker')
            m_screen = screen >= 6.5
            m_bat    = battery >= 4500
            strict = base[m_disp & m_stereo & m_screen & m_bat]
            if len(strict) < 5:
                strict = base[m_disp & (m_stereo | m_screen) & m_bat]
            if len(strict) < 5:
                strict = base[m_disp & m_bat]
            if len(strict) < 5:
                strict = base[m_disp]
            result = strict

        elif persona_idx == 6:  # 🔋 Battery-Focused
            # 4500–5500mAh, efficient chipset, fast charging
            m_bat  = battery >= 4500
            m_fc   = kw_match('corpus', r'fast charg|quick charg|turbo charg|\d{2,3}\s*w\s*charg|\d{2,3}w')
            m_eff  = kw_match('corpus', r'helio|snapdragon\s*[46]|dimensity\s*[67]|efficient')
            strict = base[m_bat & (m_fc | m_eff)]
            if len(strict) < 5:
                strict = base[m_bat]
            if len(strict) < 5:
                strict = base[battery >= 4000]
            result = strict

        elif persona_idx == 7:  # 📶 Social Media / Content Creators
            # 16MP+ front, 48MP+ rear, 128GB+ storage, fast processor
            m_front   = f_camera >= 16
            m_rear    = camera >= 48
            m_storage = storage >= 128
            m_perf    = kw_match('corpus', r'snapdragon\s*[78]\d{2}|dimensity\s*[89]\d{2}|tensor|bionic')
            strict = base[m_front & m_rear & m_storage]
            if len(strict) < 5:
                strict = base[(m_front | m_rear) & m_storage]
            if len(strict) < 5:
                strict = base[m_rear & m_storage]
            if len(strict) < 5:
                strict = base[m_rear]
            result = strict

        elif persona_idx == 8:  # 🎓 Students
            # Budget (≤25000), 6GB+ RAM, 4000mAh+, decent performance
            m_price = price <= 25000
            m_ram   = ram >= 6
            m_bat   = battery >= 4000
            strict = base[m_price & m_ram & m_bat]
            if len(strict) < 5:
                strict = base[m_price & m_ram]
            if len(strict) < 5:
                strict = base[m_price & (ram >= 4)]
            if len(strict) < 5:
                strict = base[m_price]
            result = strict

        elif persona_idx == 9:  # 👴 Senior Citizens / Simple Use
            # Simple UI, loud speakers, 4000mAh+, affordable (≤15000)
            m_price = price <= 15000
            m_bat   = battery >= 4000
            m_kw    = kw_match('corpus', r'simple|loud|speaker|easy|large font|big display|accessibility')
            strict = base[m_price & m_bat & m_kw]
            if len(strict) < 5:
                strict = base[m_price & m_bat]
            if len(strict) < 5:
                strict = base[m_price]
            result = strict

        elif persona_idx == 10:  # 📡 Travel / Outdoor
            # 5000mAh+, gorilla glass / rugged build, GPS, network stability
            m_bat   = battery >= 5000
            m_build = kw_match('corpus', r'gorilla\s*glass|corning|rugged|ip6[78]|ip\s*68|ip\s*67|military')
            m_gps   = kw_match('corpus', r'\bgps\b|glonass|beidou|navigation')
            m_5g    = kw_match('corpus', r'5g') | kw_match('5g', r'Yes|True|1')
            strict = base[m_bat & (m_build | m_gps)]
            if len(strict) < 5:
                strict = base[m_bat & (m_5g | m_gps)]
            if len(strict) < 5:
                strict = base[m_bat]
            result = strict

        elif persona_idx == 11:  # 🧑💻 Tech Enthusiasts
            # Latest chipset, 120Hz+ display, AI features, fast charging, 8GB+ RAM
            m_ram   = ram >= 8
            m_hz    = kw_match('corpus', r'120\s*hz|144\s*hz|165\s*hz|240\s*hz')
            m_ai    = kw_match('corpus', r'\bai\b|artificial intelligence|machine learning|on.device ai|generative')
            m_chip  = kw_match('processor', r'Gen\s*[234]|Bionic|Tensor\s*G[23]|Dimensity\s*9\d{2,3}')
            m_chip |= kw_match('corpus', r'gen\s*[234]|a1[567]\s*bionic|tensor\s*g[23]|dimensity\s*9\d{2,3}')
            m_fc    = kw_match('corpus', r'\d{2,3}\s*w\s*charg|\d{2,3}w\s*fast|wireless charg')
            strict = base[m_ram & (m_chip | m_hz) & (m_ai | m_fc)]
            if len(strict) < 5:
                strict = base[m_ram & (m_chip | m_hz)]
            if len(strict) < 5:
                strict = base[m_ram & m_hz]
            if len(strict) < 5:
                strict = base[m_ram]
            result = strict

        elif persona_idx == 12:  # 🎨 Design / Aesthetic Lovers
            # Premium design, slim, lightweight, unique colors, quality build
            m_kw    = kw_match('corpus', r'premium|slim|lightweight|thin|glass back|leather|vegan|color|gradient|ceramic|titanium|sleek|elegant|aesthetic')
            m_build = kw_match('corpus', r'gorilla\s*glass|corning|aluminium|aluminum|stainless')
            strict = base[m_kw | m_build]
            if len(strict) < 5:
                strict = base[m_kw]
            result = strict

        else:
            result = base

        # ── Final Fallback: if still too few, relax to keyword-only ────────────
        fallback_kw = {
            1: r'snapdragon|dimensity|120hz|gaming|cooling',
            2: r'4gb|6gb|balanced|everyday',
            3: r'camera|ois|night|sensor|mp',
            4: r'productivity|security|business|multitask',
            5: r'amoled|oled|stereo|dolby|entertainment',
            6: r'5000mah|battery|mah|fast.?charg',
            7: r'creator|vlog|reel|front.camera|selfie',
            8: r'budget|student|affordable|value',
            9: r'simple|loud|easy|senior|basic',
            10: r'gorilla|ip68|gps|travel|outdoor|5g',
            11: r'ai|flagship|gen\s*\d|144hz|next.?gen',
            12: r'slim|premium|design|color|aesthetic|lightweight',
        }

        if len(result) < 3:
            kw = fallback_kw.get(persona_idx, '')
            if kw:
                result = base[kw_match('corpus', kw)]
        if len(result) < 3:
            result = base  # absolute fallback: just return top-rated phones

        result = result.copy()
        result['spec_score'] = (
            result['performance_score'].fillna(0) + 
            result['camera_score'].fillna(0) + 
            result['battery_score'].fillna(0) + 
            result['display_score'].fillna(0)
        )

        # Apply sorting based on user selection
        if sort_option == 'Price: Low to High':
            result = result.sort_values(by=['price_numeric'], ascending=True)
        elif sort_option == 'Price: High to Low':
            result = result.sort_values(by=['price_numeric'], ascending=False)
        elif sort_option == 'Rating: High to Low':
            result = result.sort_values(by=['ratings'], ascending=False)
        elif sort_option == 'Popularity (Rating Count)':
            result = result.sort_values(by=['ratings', 'price_numeric'], ascending=[False, True])
        else:  # Relevance (default)
            result = result.sort_values(by=['spec_score', 'ratings'], ascending=[False, False])
        
        return result.drop_duplicates(subset=['base_name'], keep='first').head(10).drop(columns=['spec_score'])


    def content_based(self, mobile_name):
        try:
            mobile_index = self.df[self.df['name'] == mobile_name].index[0]
            # Guard against mismatch between deduplicated df and similarity matrix
            if (self.similarity_matrix is None or
                mobile_index >= self.similarity_matrix.shape[0]):
                return pd.DataFrame()
            similarity_array = self.similarity_matrix[mobile_index]
            similar_indices = sorted(list(enumerate(similarity_array)), reverse=True, key=lambda x: x[1])[1:30]
            results = self.df.iloc[[i[0] for i in similar_indices if i[0] < len(self.df)]]
            return results[results['price_numeric'] >= 50].drop_duplicates(subset=['base_name'], keep='first').head(10)
        except (IndexError, KeyError):
            return pd.DataFrame()

    def preference_based(self, min_budget=5000, max_budget=200000, usage_type='normal', perf_pref='medium', camera_pref='medium', battery_pref='medium', display_pref='medium', sort_option='Relevance', top_n=10):
        """
        Content-Based Filtering using user preference vector + cosine similarity.
        
        Params:
            budget        : max price in INR
            usage_type    : 'gaming' | 'photography' | 'normal' | 'battery' | 'entertainment'
            perf_pref     : 'high' | 'medium' | 'low'
            camera_pref   : 'high' | 'medium' | 'low'
            battery_pref  : 'high' | 'medium' | 'low'
            display_pref  : 'high' | 'medium' | 'low'
        """
        # REFINED WEIGHTS: High is now massively more important than medium
        PREF_MAP = {'high': 10.0, 'medium': 2.5, 'low': 0.2}

        # Build weights based on preferences
        weights = np.array([
            PREF_MAP.get(perf_pref, 2.5),
            PREF_MAP.get(camera_pref, 2.5),
            PREF_MAP.get(battery_pref, 2.5),
            PREF_MAP.get(display_pref, 2.5),
        ])

        # Usage type acts as a much STRONGER multiplier
        usage_multipliers = {
            'gaming':        np.array([2.5, 1.0, 1.5, 1.2]), # Massive performance boost
            'photography':   np.array([1.0, 2.5, 1.0, 1.5]), # Massive camera boost
            'battery':       np.array([1.2, 1.0, 2.5, 1.0]), # Massive battery boost
            'entertainment': np.array([1.0, 1.2, 1.2, 2.5]), # Massive display boost
            'normal':        np.array([1.0, 1.0, 1.0, 1.0])
        }
        weights *= usage_multipliers.get(usage_type, np.array([1.0, 1.0, 1.0, 1.0]))

        # Filter by price range
        pool = self.df[
            (self.df['price_numeric'] >= 50) &
            (self.df['price_numeric'] >= min_budget) &
            (self.df['price_numeric'] <= max_budget)
        ].copy()

        if pool.empty:
            return pd.DataFrame(), pd.Series(dtype=str)

        # Phone feature matrix
        phone_matrix = pool[['performance_score','camera_score','battery_score','display_score']].fillna(5.0).values

        # Calculate final preference score
        pref_scores = phone_matrix @ weights
        
        # Apply a 'Luxury Penalty' ONLY if user explicitly chose 'low'
        for i, (pref, label) in enumerate(zip([perf_pref, camera_pref, battery_pref, display_pref], ['Perf', 'Cam', 'Bat', 'Disp'])):
            if pref == 'low':
                overspec_mask = phone_matrix[:, i] > 7.5
                pref_scores[overspec_mask] -= (phone_matrix[overspec_mask, i] - 7.5) * 15.0 

        # Add quality boost
        pref_scores += pool['ratings'].fillna(0).values * 5.0
        
        # DYNAMIC Price Segment Boost: 
        # Favor phones that are in the "sweet spot" of the user's budget (top 40% of range)
        price_ratio = (pool['price_numeric'] / (max_budget + 1))
        pref_scores += (price_ratio ** 3) * 60.0 # Cubic boost for the premium segment

        pool = pool.copy()
        pool['_score'] = pref_scores
        
        # Apply sorting based on user selection
        if sort_option == 'Price: Low to High':
            pool = pool.sort_values(by=['price_numeric'], ascending=True)
        elif sort_option == 'Price: High to Low':
            pool = pool.sort_values(by=['price_numeric'], ascending=False)
        elif sort_option == 'Rating: High to Low':
            pool = pool.sort_values(by=['ratings'], ascending=False)
        elif sort_option == 'Popularity (Rating Count)':
            pool = pool.sort_values(by=['ratings', 'price_numeric'], ascending=[False, True])
        else:  # Relevance (default)
            pool = pool.sort_values(by=['_score'], ascending=False)
        
        pool = pool.drop_duplicates(subset=['base_name'], keep='first').head(top_n)

        # Return empty reasons as they were requested to be removed
        reasons = pd.Series([""] * len(pool), index=pool.index)
        return pool.drop(columns=['_score'], errors='ignore'), reasons


    def collaborative_popularity(self, budget_range=None):
        filtered = self.df[self.df['price_numeric'] >= 50]
        if budget_range:
            filtered = filtered[(filtered['price_numeric'] >= budget_range[0]) & (filtered['price_numeric'] <= budget_range[1])]
        return filtered.sort_values(by=['ratings', 'price_numeric'], ascending=[False, True]).drop_duplicates(subset=['base_name'], keep='first').head(10)

    def weighted_scoring(self, weights, max_budget=None, sort_option='Relevance'):
        # Start with a pool of actual smartphones (>= 5000 INR) to avoid feature phones dominating
        pool = self.df[self.df['price_numeric'] >= 5000].copy()
        
        # Enforce budget cap if provided
        if max_budget:
            pool = pool[pool['price_numeric'] <= max_budget]
            
        if pool.empty:
            return pd.DataFrame()

        # Exaggerate weights (exponential) to make sliders highly sensitive
        weights = {k: v**3 for k, v in weights.items()}

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            return pool.sort_values(by='ratings', ascending=False).drop_duplicates(subset=['base_name'], keep='first').head(10)

        score = pd.Series(0.0, index=pool.index)

        def scale_feature(series):
            series = pd.to_numeric(series, errors='coerce').fillna(0)
            s_min = series.min()
            s_max = series.quantile(0.95) # Cap at 95th percentile
            if s_max > s_min:
                return (series.clip(s_min, s_max) - s_min) / (s_max - s_min) * 10.0
            return pd.Series(10.0, index=series.index)

        if 'performance' in weights:
            score += scale_feature(pool['performance_score']) * weights['performance']
            
        if 'ram' in weights:
            score += scale_feature(pool['ram']) * weights['ram']
            
        if 'storage' in weights:
            score += scale_feature(pool['storage']) * weights['storage']
            
        if 'rating' in weights:
            score += scale_feature(pool['ratings']) * weights['rating']
            
        if 'battery' in weights:
            score += scale_feature(pool['battery_score']) * weights['battery']
            
        if 'camera' in weights:
            score += scale_feature(pool['camera_score']) * weights['camera']

        pool['score'] = score
        
        # Apply sorting based on user selection
        if sort_option == 'Price: Low to High':
            pool = pool.sort_values(by=['price_numeric'], ascending=True)
        elif sort_option == 'Price: High to Low':
            pool = pool.sort_values(by=['price_numeric'], ascending=False)
        elif sort_option == 'Rating: High to Low':
            pool = pool.sort_values(by=['ratings'], ascending=False)
        elif sort_option == 'Popularity (Rating Count)':
            pool = pool.sort_values(by=['ratings', 'price_numeric'], ascending=[False, True])
        else:  # Relevance (default)
            pool = pool.sort_values(by=['score', 'ratings'], ascending=[False, False])
        
        return pool.drop_duplicates(subset=['base_name'], keep='first').head(10)

    def knn_recommend(self, mobile_name, k=10):
        try:
            mobile_index = self.df[self.df['name'] == mobile_name].index[0]
            # Guard against mismatch between deduplicated df and scaled features
            if mobile_index >= len(self.df_scaled):
                return pd.DataFrame()
            target_features = self.df_scaled.iloc[mobile_index].values
            distances = np.sqrt(np.sum((self.df_scaled.values - target_features)**2, axis=1))
            nearest_indices = np.argsort(distances)[1:k+30] # Grab extra in case we filter duplicates
            # Ensure indices are within bounds of the deduplicated dataframe
            nearest_indices = [i for i in nearest_indices if i < len(self.df)]
            results = self.df.iloc[nearest_indices]
            return results[results['price_numeric'] >= 50].drop_duplicates(subset=['base_name'], keep='first').head(k)
        except (IndexError, KeyError):
            return pd.DataFrame()
