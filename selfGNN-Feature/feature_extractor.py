"""
Feature extractor for Yelp-Merchant dataset.

Reads raw Yelp JSON files, re-runs k-core filtering to reconstruct
user2id / merchant2id mappings, then extracts and saves:
  - user_features.npy     shape (num_users, 4)
  - merchant_features.npy shape (num_merchants, 6)
  - edge_weights.pkl      dict {(user_int, merchant_int): float_rating}
  - user2id.pkl           dict {user_str: int}
  - merchant2id.pkl       dict {merchant_str: int}

Run from project root or from selfGNN-Feature/:
    python selfGNN-Feature/feature_extractor.py
"""
import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
RAW_DIR = os.path.join(_root, 'Datasets', 'yelp_dataset')
OUT_DIR = os.path.join(_root, 'Datasets', 'yelp-merchant')

REVIEW_JSON = os.path.join(RAW_DIR, 'yelp_academic_dataset_review.json')
BUSINESS_JSON = os.path.join(RAW_DIR, 'yelp_academic_dataset_business.json')
MIN_INTERACTIONS = 5


# ------------------------------------------------------------------ #
#  Step 0.5: Load merchant_bids (businesses with at least 1 category) #
# ------------------------------------------------------------------ #
def load_merchant_bids(business_path):
    """Matches preprocessing notebook Cell 2: only businesses with categories."""
    print('Loading business IDs with categories...')
    merchant_bids = set()
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = b.get('categories') or ''
            cat_tokens = {c.strip() for c in cats.split(',') if c.strip()}
            if cat_tokens:
                merchant_bids.add(b['business_id'])
    print(f'  Merchant businesses with categories: {len(merchant_bids):,}')
    return merchant_bids


# ------------------------------------------------------------------ #
#  Step 1: Read review data and build interaction dict                 #
# ------------------------------------------------------------------ #
def read_reviews(review_path, merchant_bids):
    """Stream review JSON. Filters to businesses with categories (matching preprocessing).
    Returns raw_inter: {user_str: {business_str: [(stars, date_str)]}}
    """
    print('Reading reviews (streaming, filtered to merchant_bids)...')
    raw_inter = defaultdict(lambda: defaultdict(list))
    total = 0
    kept = 0
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = r.get('business_id', '')
            if bid not in merchant_bids:
                total += 1
                if total % 1_000_000 == 0:
                    print(f'  Read {total:,} reviews, kept {kept:,}...')
                continue
            uid = r.get('user_id', '')
            stars = float(r.get('stars', 3.0))
            date = r.get('date', '')
            if uid and bid and date:
                raw_inter[uid][bid].append((stars, date))
                kept += 1
            total += 1
            if total % 1_000_000 == 0:
                print(f'  Read {total:,} reviews, kept {kept:,}...')
    print(f'  Total reviews: {total:,}, kept: {kept:,}, unique users: {len(raw_inter):,}')
    return raw_inter


# ------------------------------------------------------------------ #
#  Step 2: K-core filtering (identical to preprocessing notebook)     #
# ------------------------------------------------------------------ #
def kcore_filter(raw_inter, min_count=MIN_INTERACTIONS):
    """Input: {user_str: {biz_str: [(stars, date)]}}
    Output: filtered dict (same structure, only entries passing k-core).
    """
    print(f'Running k-core filtering (min={min_count})...')
    data = {u: dict(bs) for u, bs in raw_inter.items()}
    iteration = 0
    while True:
        biz_cnt = defaultdict(int)
        for u, bs in data.items():
            for b in bs:
                biz_cnt[b] += 1
        valid_biz = {b for b, cnt in biz_cnt.items() if cnt >= min_count}

        data = {u: {b: v for b, v in bs.items() if b in valid_biz}
                for u, bs in data.items()}
        data = {u: bs for u, bs in data.items() if len(bs) >= min_count}

        biz_cnt2 = defaultdict(int)
        for u, bs in data.items():
            for b in bs:
                biz_cnt2[b] += 1
        still_valid = {b for b, cnt in biz_cnt2.items() if cnt >= min_count}

        iteration += 1
        print(f'  Iter {iteration}: {len(data):,} users, {len(still_valid):,} merchants')

        if still_valid == valid_biz and all(len(bs) >= min_count for bs in data.values()):
            break
        if len(data) == 0 or len(still_valid) == 0:
            break
    return data


# ------------------------------------------------------------------ #
#  Step 3: Build sorted ID mappings                                   #
# ------------------------------------------------------------------ #
def build_mappings(filtered):
    user_strs = sorted(filtered.keys())
    all_biz = {b for bs in filtered.values() for b in bs}
    merchant_strs = sorted(all_biz)
    user2id = {u: i for i, u in enumerate(user_strs)}
    merchant2id = {b: i for i, b in enumerate(merchant_strs)}
    print(f'Users: {len(user2id):,}, Merchants: {len(merchant2id):,}')
    return user2id, merchant2id


# ------------------------------------------------------------------ #
#  Step 4: Extract user features from reviews                         #
# ------------------------------------------------------------------ #
def extract_user_features(filtered, user2id):
    """4 features per user: review_count, avg_stars, review_span_days, unique_biz."""
    print('Extracting user features...')
    num_users = len(user2id)
    feats = np.zeros((num_users, 4), dtype=np.float32)
    coverage = 0

    for u_str, uid in user2id.items():
        if u_str not in filtered:
            continue
        reviews = []
        for b_str, entries in filtered[u_str].items():
            for stars, date_str in entries:
                try:
                    dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                except ValueError:
                    continue
                reviews.append((stars, dt, b_str))
        if not reviews:
            continue
        coverage += 1
        stars_list = [r[0] for r in reviews]
        dates = [r[1] for r in reviews]
        unique_biz = len(set(r[2] for r in reviews))
        span_days = (max(dates) - min(dates)).days

        feats[uid, 0] = len(reviews)              # total review count
        feats[uid, 1] = np.mean(stars_list)       # avg stars given
        feats[uid, 2] = span_days                 # review span in days
        feats[uid, 3] = unique_biz                # unique businesses

    print(f'  User feature coverage: {coverage:,}/{num_users:,} '
          f'({100*coverage/num_users:.1f}%)')
    return feats


# ------------------------------------------------------------------ #
#  Step 5: Extract merchant features from business JSON               #
# ------------------------------------------------------------------ #
def extract_merchant_features(business_path, merchant2id):
    """6 features per merchant: stars, log_review_count, num_categories,
    is_open, city_idx, category_idx."""
    print('Reading business JSON...')
    biz_data = {}
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError:
                continue
            biz_data[b['business_id']] = b

    print(f'  Total businesses in JSON: {len(biz_data):,}')

    # Build city and category encodings from merchants in our dataset
    cities = defaultdict(int)
    categories_cnt = defaultdict(int)
    for b_str in merchant2id:
        if b_str not in biz_data:
            continue
        b = biz_data[b_str]
        cities[b.get('city', '')] += 1
        cats = b.get('categories') or ''
        for cat in cats.split(','):
            cat = cat.strip()
            if cat:
                categories_cnt[cat] += 1

    top_cities = {c: i for i, (c, _) in
                  enumerate(sorted(cities.items(), key=lambda x: -x[1])[:50])}
    top_categories = {c: i for i, (c, _) in
                      enumerate(sorted(categories_cnt.items(), key=lambda x: -x[1])[:30])}

    num_merchants = len(merchant2id)
    feats = np.zeros((num_merchants, 6), dtype=np.float32)
    coverage = 0

    for b_str, bid in merchant2id.items():
        if b_str not in biz_data:
            continue
        coverage += 1
        b = biz_data[b_str]
        stars = float(b.get('stars', 3.0))
        review_count = int(b.get('review_count', 0))
        is_open = int(b.get('is_open', 0))
        city = b.get('city', '')
        cats_str = b.get('categories') or ''
        cats = [c.strip() for c in cats_str.split(',') if c.strip()]
        num_cats = len(cats)
        primary_cat = cats[0] if cats else ''

        city_idx = top_cities.get(city, 50)
        cat_idx = top_categories.get(primary_cat, 30)

        feats[bid, 0] = stars
        feats[bid, 1] = np.log1p(review_count)
        feats[bid, 2] = num_cats
        feats[bid, 3] = is_open
        feats[bid, 4] = city_idx
        feats[bid, 5] = cat_idx

    print(f'  Merchant feature coverage: {coverage:,}/{num_merchants:,} '
          f'({100*coverage/num_merchants:.1f}%)')
    return feats


# ------------------------------------------------------------------ #
#  Step 6: Extract edge weights (most recent star rating)             #
# ------------------------------------------------------------------ #
def extract_edge_weights(filtered, user2id, merchant2id):
    """For each (user, merchant) training pair, use the most recent rating."""
    print('Extracting edge weights...')
    edge_weights = {}
    for u_str, uid in user2id.items():
        if u_str not in filtered:
            continue
        for b_str, entries in filtered[u_str].items():
            if b_str not in merchant2id:
                continue
            bid = merchant2id[b_str]
            # Most recent rating
            latest = None
            latest_stars = None
            for stars, date_str in entries:
                try:
                    dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                except ValueError:
                    continue
                if latest is None or dt > latest:
                    latest = dt
                    latest_stars = stars
            if latest_stars is not None:
                edge_weights[(uid, bid)] = float(latest_stars)

    print(f'  Edge weights: {len(edge_weights):,} pairs')
    w_vals = np.array(list(edge_weights.values()))
    print(f'  Weight stats: min={w_vals.min():.2f}, max={w_vals.max():.2f}, '
          f'mean={w_vals.mean():.2f}, std={w_vals.std():.2f}')
    return edge_weights


# ------------------------------------------------------------------ #
#  Step 7: Normalize features (zero mean, unit variance)             #
# ------------------------------------------------------------------ #
def normalize_features(feats, name='features'):
    """Normalize only non-zero rows. Zero rows remain zero."""
    nonzero_mask = (feats != 0).any(axis=1)
    if nonzero_mask.sum() == 0:
        return feats
    mean = feats[nonzero_mask].mean(axis=0)
    std = feats[nonzero_mask].std(axis=0) + 1e-8
    result = feats.copy()
    result[nonzero_mask] = (feats[nonzero_mask] - mean) / std
    print(f'  {name} normalized. Shape: {result.shape}, '
          f'mean={result[nonzero_mask].mean():.4f}, std={result[nonzero_mask].std():.4f}')
    return result


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print('=' * 60)
    print('Yelp-Merchant Feature Extractor')
    print('=' * 60)
    print(f'Raw dir: {RAW_DIR}')
    print(f'Output dir: {OUT_DIR}')

    for p in [REVIEW_JSON, BUSINESS_JSON]:
        if not os.path.isfile(p):
            print(f'ERROR: {p} not found')
            sys.exit(1)

    # Step 0.5: Load merchant_bids (must match preprocessing notebook)
    merchant_bids = load_merchant_bids(BUSINESS_JSON)

    # Step 1: Read reviews (filtered to merchant_bids)
    raw_inter = read_reviews(REVIEW_JSON, merchant_bids)

    # Step 2: K-core filtering
    filtered = kcore_filter(raw_inter)

    # Step 3: Build mappings
    user2id, merchant2id = build_mappings(filtered)

    # Verify against preprocessed data
    processed_mat = os.path.join(OUT_DIR, 'trn_mat_time')
    if os.path.isfile(processed_mat):
        with open(processed_mat, 'rb') as f:
            trnMat = pickle.load(f)
        exp_users, exp_merchants = trnMat[0].shape
        if len(user2id) != exp_users or len(merchant2id) != exp_merchants:
            print(f'WARNING: Reconstructed {len(user2id)} users / {len(merchant2id)} merchants '
                  f'but preprocessed data has {exp_users} / {exp_merchants}')
            print('  Mappings may differ. Features will still be extracted.')
        else:
            print(f'Mapping verification: OK ({exp_users:,} users, {exp_merchants:,} merchants)')

    # Step 4: User features
    user_feats = extract_user_features(filtered, user2id)
    user_feats = normalize_features(user_feats, 'user_features')

    # Step 5: Merchant features
    merchant_feats = extract_merchant_features(BUSINESS_JSON, merchant2id)
    merchant_feats = normalize_features(merchant_feats, 'merchant_features')

    # Step 6: Edge weights
    edge_weights = extract_edge_weights(filtered, user2id, merchant2id)

    # Step 7: Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, 'user_features.npy'), user_feats)
    np.save(os.path.join(OUT_DIR, 'merchant_features.npy'), merchant_feats)
    with open(os.path.join(OUT_DIR, 'edge_weights.pkl'), 'wb') as f:
        pickle.dump(edge_weights, f)
    with open(os.path.join(OUT_DIR, 'user2id.pkl'), 'wb') as f:
        pickle.dump(user2id, f)
    with open(os.path.join(OUT_DIR, 'merchant2id.pkl'), 'wb') as f:
        pickle.dump(merchant2id, f)

    print()
    print('=' * 60)
    print('Saved:')
    print(f'  user_features.npy     shape={user_feats.shape}')
    print(f'  merchant_features.npy shape={merchant_feats.shape}')
    print(f'  edge_weights.pkl      {len(edge_weights):,} pairs')
    print(f'  user2id.pkl           {len(user2id):,} users')
    print(f'  merchant2id.pkl       {len(merchant2id):,} merchants')
    print('Feature dimensions:')
    print(f'  d_u = {user_feats.shape[1]} (review_count, avg_stars, span_days, unique_biz)')
    print(f'  d_v = {merchant_feats.shape[1]} (stars, log_review_count, num_cats, '
          f'is_open, city_idx, cat_idx)')
    print('=' * 60)


if __name__ == '__main__':
    main()
