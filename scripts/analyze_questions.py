#!/usr/bin/env python3
"""Analyze generated questions."""
import json
from collections import Counter

data = json.load(open('data/lab/questions.json'))

print("=" * 80)
print("QUESTION GENERATION ANALYSIS")
print("=" * 80)

# Count by paper
by_paper = Counter([q['paper_id'] for q in data])
by_field = Counter([q['paper_id'].split('_')[1] for q in data])
by_style = Counter([q['style'] for q in data])

# Styles by paper
styles_by_paper = {}
for q in data:
    p = q['paper_id']
    styles_by_paper[p] = styles_by_paper.get(p, Counter()) + Counter([q['style']])

print("\nBy Paper:")
print(f"{'Paper':<60} {'Final':>6} {'Factual':>8} {'Conceptual':>10} {'Paraphrased':>12}")
print("-" * 96)
for p in sorted(by_paper.keys()):
    s = styles_by_paper[p]
    print(f"{p:<60} {by_paper[p]:>6} {s['factual']:>8} {s['conceptual']:>10} {s['paraphrased']:>12}")

print("\nBy Field:")
print(f"{'Field':<20} {'Count':>10}")
print("-" * 31)
for f in sorted(by_field.keys()):
    print(f"{f:<20} {by_field[f]:>10}")

print("\nBy Style:")
print(f"{'Style':<20} {'Count':>10}")
print("-" * 31)
for s in sorted(by_style.keys()):
    print(f"{s:<20} {by_style[s]:>10}")

print("\nMissing Papers:")
all_papers = [
    '001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing',
    '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction',
    '003_NLP_Should_I_visit_this_place_Inclusion_and_Exclusion_Phrase_Mining_from_Reviews',
    '035_Computer_Vision_Leaf_Segmentation_and_Counting_with_Deep_Learning_on_Model_Certainty_Test-Time_Augmentation_Trade-Of',
    '041_Computer_Vision_Appearance-Invariant_6-DoF_Visual_Localization_using_Generative_Adversarial_Networks',
    '045_Computer_Vision_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection',
    '072_Machine_Learning_Ensemble_model_for_pre-discharge_icd10_coding_prediction',
    '074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs',
    '090_Machine_Learning_Adaptive_Precision_Training_for_Resource_Constrained_Devices'
]
missing = [p for p in all_papers if p not in by_paper]
for p in missing:
    print(f"  {p}")

print(f"\nTotal: {len(data)} questions")
print(f"Expected: 54 questions (9 papers x 6)")
print(f"Missing: {54 - len(data)} questions")
print(f"Papers with <5 questions: {[p for p, c in by_paper.items() if c < 5]}")
