import json
from pathlib import Path

# Quick config papers (6)
quick_papers = {
    '001_NLP_End-to-End_Speaker_Diarization_as_Post-Processing',
    '002_NLP_Regularized_Attentive_Capsule_Network_for_Overlapped_Relation_Extraction',
    '035_Computer_Vision_Leaf_Segmentation_and_Counting_with_Deep_Learning_on_Model_Certainty_Test-Time_Augmentation_Trade-Of',
    '041_Computer_Vision_Appearance-Invariant_6-DoF_Visual_Localization_using_Generative_Adversarial_Networks',
    '072_Machine_Learning_Ensemble_model_for_pre-discharge_icd10_coding_prediction',
    '074_Machine_Learning_Unifying_Homophily_and_Heterophily_Network_Transformation_via_Motifs'
}

# All 9 papers
all_chunks = []
with open('data/lab/chunks_512_51.jsonl', 'r') as f:
    for line in f:
        all_chunks.append(json.loads(line))

# Find chunks from new papers only
new_papers = set()
for chunk in all_chunks:
    if chunk['paper_id'] not in quick_papers:
        new_papers.add(chunk['paper_id'])

print(f'New papers to embed: {len(new_papers)}')
for p in sorted(new_papers):
    print(f'  {p}')

new_chunks = [c for c in all_chunks if c['paper_id'] in new_papers]
print(f'New chunks: {len(new_chunks)}')

# Save new chunks to separate file
with open('data/lab/chunks_512_51_new.jsonl', 'w') as f:
    for chunk in new_chunks:
        f.write(json.dumps(chunk) + '\n')
print('Saved to data/lab/chunks_512_51_new.jsonl')
