# Step 1: Import Libraries

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from model_utils import CLASSES, DATA_PATH, PROTOCOL, RANDOM_STATE, RESULTS_DIR, TARGET


# Step 2: Reproduce the Original Stratified Assignment

def main():
    raw = pd.read_csv(DATA_PATH, encoding='cp1252')
    if raw.Incident_ID.isna().any() or not raw.Incident_ID.is_unique:
        raise ValueError('Dataset IDs must be nonmissing and unique.')
    target = raw[TARGET].astype('string').str.strip().str.lower().replace('', pd.NA)
    if not target.dropna().isin(CLASSES + ['aircraft missing']).all():
        raise ValueError('Unexpected target categories.')
    eligible = raw.loc[target.isin(CLASSES)].copy()
    eligible[TARGET] = target.loc[eligible.index]
    train, temporary = train_test_split(
        eligible, test_size=.3, stratify=eligible[TARGET], random_state=RANDOM_STATE
    )
    validation, test = train_test_split(
        temporary, test_size=.5, stratify=temporary[TARGET], random_state=RANDOM_STATE
    )
    split = pd.concat([
        train[['Incident_ID']].assign(split='train'),
        validation[['Incident_ID']].assign(split='validation'),
        test[['Incident_ID']].assign(split='test'),
    ], ignore_index=True)

    # Step 3: Keep One Representative per Exact-Record Group
    # Prefer the original held-out representative; preserve all other assignments.
    columns = [col for col in raw if col != 'Incident_ID']
    groups = eligible.assign(group=pd.util.hash_pandas_object(eligible[columns], index=False).to_numpy())
    groups = groups[['Incident_ID', 'group']].merge(split, on='Incident_ID', validate='one_to_one')
    groups['priority'] = groups['split'].map({'train': 0, 'validation': 1, 'test': 2})
    representatives = groups.sort_values('priority', ascending=False, kind='stable').drop_duplicates('group')
    kept = set(representatives.Incident_ID)
    split = split.loc[split.Incident_ID.isin(kept)].copy()
    excluded = groups.loc[~groups.Incident_ID.isin(kept), 'Incident_ID'].tolist()
    if len(split) != groups.group.nunique():
        raise RuntimeError('Exact-record group deduplication failed.')

    # Step 4: Save the Versioned Protocol and Split Manifest
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(RESULTS_DIR / 'split_manifest.csv', index=False)
    metadata = {
        'protocol': PROTOCOL,
        'seed': RANDOM_STATE,
        'duplicate_policy': 'one exact-record representative; prefer original test, then validation, then train',
        'excluded_duplicate_ids': excluded,
        'split_counts': split['split'].value_counts().to_dict(),
        'natural_missing_targets': int(target.isna().sum()),
        'scope': 'synthetic masking on originally labeled four-class records; raw CSV unchanged',
    }
    (RESULTS_DIR / 'split_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f"Split saved: {metadata['split_counts']} | Exact-record representatives removed: {len(excluded)}")


if __name__ == '__main__':
    main()
