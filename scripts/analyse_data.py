import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import re

def analyze_dataset(csv_path):
    """Analyse détaillée du dataset d'entraînement"""
    
    print("=== ANALYSE DU DATASET D'ENTRAÎNEMENT ===\n")
    
    # Charger les données
    try:
        df = pd.read_csv(csv_path, sep="\\|\\|\\|", engine="python")
        df = df[['modern', 'old_french']].dropna()
        df['modern'] = df['modern'].str.strip()
        df['old_french'] = df['old_french'].str.strip()
        print(f"✓ Dataset chargé : {len(df)} paires de traduction")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return
    
    # 1. Statistiques générales
    print("\n1. STATISTIQUES GÉNÉRALES")
    print("-" * 40)
    print(f"Nombre total de paires : {len(df)}")
    print(f"Textes modernes uniques : {df['modern'].nunique()}")
    print(f"Textes anciens uniques : {df['old_french'].nunique()}")
    
    # Longueurs moyennes
    modern_lengths = df['modern'].str.len()
    old_lengths = df['old_french'].str.len()
    print(f"Longueur moyenne - français moderne : {modern_lengths.mean():.1f} caractères")
    print(f"Longueur moyenne - ancien français : {old_lengths.mean():.1f} caractères")
    
    # 2. Échantillons du dataset
    print("\n2. ÉCHANTILLONS DU DATASET")
    print("-" * 40)
    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
    for i, idx in enumerate(sample_indices):
        modern = df.iloc[idx]['modern']
        old = df.iloc[idx]['old_french']
        print(f"{i+1:2d}. Moderne : {modern}")
        print(f"    Ancien  : {old}")
        print()
    
    # 3. Analyse des similitudes
    print("3. ANALYSE DES SIMILITUDES")
    print("-" * 40)
    
    similarities = []
    identical_count = 0
    
    for _, row in df.iterrows():
        similarity = SequenceMatcher(None, row['modern'], row['old_french']).ratio()
        similarities.append(similarity)
        if row['modern'].lower() == row['old_french'].lower():
            identical_count += 1
    
    similarities = np.array(similarities)
    print(f"Similarité moyenne : {similarities.mean():.3f}")
    print(f"Similarité médiane : {np.median(similarities):.3f}")
    print(f"Textes identiques : {identical_count} ({identical_count/len(df)*100:.1f}%)")
    print(f"Similarité > 0.9 : {np.sum(similarities > 0.9)} ({np.sum(similarities > 0.9)/len(df)*100:.1f}%)")
    print(f"Similarité < 0.5 : {np.sum(similarities < 0.5)} ({np.sum(similarities < 0.5)/len(df)*100:.1f}%)")
    
    # 4. Analyse des différences linguistiques
    print("\n4. DIFFÉRENCES LINGUISTIQUES COURANTES")
    print("-" * 40)
    
    # Mots qui changent le plus souvent
    word_changes = Counter()
    
    for _, row in df.iterrows():
        modern_words = row['modern'].lower().split()
        old_words = row['old_french'].lower().split()
        
        if len(modern_words) == len(old_words):
            for m_word, o_word in zip(modern_words, old_words):
                if m_word != o_word:
                    word_changes[f"{m_word} → {o_word}"] += 1
    
    print("Transformations les plus fréquentes :")
    for change, count in word_changes.most_common(15):
        print(f"  {change} : {count} fois")
    
    # 5. Caractéristiques de l'ancien français
    print("\n5. CARACTÉRISTIQUES DE L'ANCIEN FRANÇAIS")
    print("-" * 40)
    
    # Mots typiques de l'ancien français
    all_old_text = ' '.join(df['old_french'].str.lower())
    all_modern_text = ' '.join(df['modern'].str.lower())
    
    old_words = Counter(re.findall(r'\b\w+\b', all_old_text))
    modern_words = Counter(re.findall(r'\b\w+\b', all_modern_text))
    
    # Mots qui apparaissent plus souvent en ancien français
    old_specific = []
    for word, old_count in old_words.most_common(50):
        modern_count = modern_words.get(word, 0)
        if old_count > modern_count * 1.5 and old_count > 5:
            ratio = old_count / max(modern_count, 1)
            old_specific.append((word, old_count, ratio))
    
    print("Mots plus fréquents en ancien français :")
    for word, count, ratio in sorted(old_specific, key=lambda x: x[2], reverse=True)[:15]:
        print(f"  '{word}' : {count} fois (ratio: {ratio:.1f}x)")
    
    # 6. Recommandations
    print("\n6. RECOMMANDATIONS")
    print("-" * 40)
    
    if similarities.mean() > 0.85:
        print("⚠️  PROBLÈME : Dataset trop similaire entre moderne et ancien français")
        print("   → Le modèle apprend peu de transformations authentiques")
        
    if identical_count / len(df) > 0.3:
        print("⚠️  PROBLÈME : Trop de textes identiques dans le dataset")
        print("   → Filtrer ou enrichir avec plus de vraies transformations")
        
    if len(df) < 1000:
        print("⚠️  PROBLÈME : Dataset trop petit")
        print("   → Augmenter la taille du dataset ou utiliser de la data augmentation")
        
    print("\n✅ SOLUTIONS RECOMMANDÉES :")
    print("1. Filtrer le dataset pour ne garder que les paires avec similarité < 0.8")
    print("2. Utiliser un modèle plus large (T5-base ou T5-large)")
    print("3. Augmenter le nombre d'époques d'entraînement")
    print("4. Ajouter plus de données d'ancien français authentique")
    print("5. Utiliser des techniques de data augmentation")

def create_filtered_dataset(csv_path, output_path, max_similarity=0.8):
    """Crée un dataset filtré avec moins de similarités"""
    
    df = pd.read_csv(csv_path, sep="\\|\\|\\|", engine="python")
    df = df[['modern', 'old_french']].dropna()
    df['modern'] = df['modern'].str.strip()
    df['old_french'] = df['old_french'].str.strip()
    
    # Calculer les similarités
    similarities = []
    for _, row in df.iterrows():
        sim = SequenceMatcher(None, row['modern'].lower(), row['old_french'].lower()).ratio()
        similarities.append(sim)
    
    df['similarity'] = similarities
    
    # Filtrer
    filtered_df = df[df['similarity'] < max_similarity].copy()
    filtered_df = filtered_df.drop('similarity', axis=1)
    
    print(f"\nDataset original : {len(df)} paires")
    print(f"Dataset filtré : {len(filtered_df)} paires")
    print(f"Réduction : {(1 - len(filtered_df)/len(df))*100:.1f}%")
    
    # Sauvegarder
    filtered_df.to_csv(output_path, sep="|", index=False)
    print(f"Dataset filtré sauvegardé : {output_path}")
    
    return filtered_df

if __name__ == "__main__":
    dataset_path = "../data/dataset.csv"  # Ajustez le chemin
    
    # Analyse complète
    analyze_dataset(dataset_path)
    
    # Créer un dataset filtré
    print("\n" + "="*60)
    create_filtered_dataset(dataset_path, "../data/dataset_filtered.csv", max_similarity=0.8)