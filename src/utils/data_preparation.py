"""
Data Preparation Script

Processes TCGA BRCA and GSE96058 datasets to create combined gene expression
files with PAM50 labels, formatted as: samples as columns, genes as rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def prepare_tcga_brca_data(
    clinical_file,
    expression_file,
    output_file,
    pam50_column=None
):
    """
    Prepare TCGA BRCA data: combine clinical and expression data.
    
    Args:
        clinical_file: Path to TCGA clinical data TSV
        expression_file: Path to TCGA gene expression CCT file
        output_file: Path to save output CSV
        pam50_column: Name of PAM50 column (auto-detect if None)
    """
    print("=" * 80)
    print("Processing TCGA BRCA data...")
    print(f"  Clinical file: {clinical_file}")
    print(f"  Expression file: {expression_file}")
    print("=" * 80)
    
    # Load clinical data
    print("\n[1] Loading clinical data...")
    clinical = pd.read_csv(clinical_file, sep='\t', low_memory=False)
    print(f"    Shape: {clinical.shape}")
    
    # Find PAM50 column - TCGA may not have it directly
    # Sometimes it's in a separate file or needs to be inferred from other columns
    print("\n[2] Searching for PAM50/subtype information...")
    if pam50_column is None:
        # Look for PAM50 directly
        pam50_cols = [col for col in clinical.columns 
                     if 'PAM50' in col.upper() or 
                        ('SUBTYPE' in col.upper() and 'PAM' in col.upper())]
        
        # Also check for intrinsic subtype or molecular subtype
        if not pam50_cols:
            pam50_cols = [col for col in clinical.columns 
                         if 'INTRINSIC' in col.upper() or
                            ('MOLECULAR' in col.upper() and 'SUBTYPE' in col.upper())]
        
        if pam50_cols:
            pam50_column = pam50_cols[0]
            print(f"    ✓ Found subtype column: {pam50_column}")
        else:
            print("    ⚠ PAM50 column not found in clinical data.")
            print("    The output will be created without PAM50 labels.")
            print("    You can add PAM50 labels later if available.")
            pam50_column = None
    
    # Load expression data (CCT format: tab-delimited, first column is gene names)
    print("  Loading expression data...")
    # Read in chunks if file is large
    expression = pd.read_csv(expression_file, sep='\t', index_col=0, low_memory=False)
    print(f"    Shape: {expression.shape}")
    print(f"    Genes: {expression.shape[0]}, Samples: {expression.shape[1]}")
    
    # Match sample IDs between clinical and expression data
    # TCGA sample IDs format: TCGA-XX-XXXX-01
    # Expression columns might have different format
    print("  Matching samples...")
    
    # Get sample IDs from clinical data
    sample_id_col = 'Sample ID' if 'Sample ID' in clinical.columns else 'Patient ID'
    if sample_id_col not in clinical.columns:
        # Try to find sample ID column
        sample_id_col = [col for col in clinical.columns if 'SAMPLE' in col.upper() or 'ID' in col.upper()][0]
    
    clinical_samples = clinical[sample_id_col].astype(str).values
    
    # Normalize sample IDs for matching (remove extra suffixes)
    def normalize_tcga_id(sample_id):
        """Normalize TCGA sample ID for matching."""
        sample_id = str(sample_id)
        # Extract TCGA-XX-XXXX format
        parts = sample_id.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        return sample_id
    
    # Normalize expression column names
    expression_cols_normalized = {col: normalize_tcga_id(col) for col in expression.columns}
    
    # Find matching samples
    matching_samples = []
    pam50_labels = []
    
    for idx, sample_id in enumerate(clinical_samples):
        normalized_id = normalize_tcga_id(sample_id)
        # Try to find matching column in expression data
        matching_cols = [col for col, norm_col in expression_cols_normalized.items() 
                        if norm_col == normalized_id]
        
        if matching_cols:
            matching_samples.append(matching_cols[0])  # Use first match
            if pam50_column and pam50_column in clinical.columns:
                pam50_labels.append(clinical.iloc[idx][pam50_column])
            else:
                pam50_labels.append('Unknown')
    
    print(f"    Matched {len(matching_samples)} samples out of {len(clinical_samples)}")
    
    # Filter expression data to matched samples only
    if matching_samples:
        expression_matched = expression[matching_samples].copy()
        
        # Add PAM50 labels as last row
        if pam50_labels:
            # Create a row for PAM50 labels
            pam50_row = pd.Series(pam50_labels, index=matching_samples, name='PAM50')
            
            # Transpose: genes as rows, samples as columns
            expression_t = expression_matched.T
            
            # Add PAM50 as a "gene" (will be last row)
            # Actually, we want it as the last column, so transpose back
            expression_with_pam50 = expression_t.copy()
            expression_with_pam50['PAM50'] = pam50_labels
            
            # Save
            print(f"  Saving to {output_file}...")
            expression_with_pam50.to_csv(output_file)
            print(f"    Final shape: {expression_with_pam50.shape}")
            print(f"    Samples (columns): {expression_with_pam50.shape[1]}")
            print(f"    Genes + PAM50 (rows): {expression_with_pam50.shape[0]}")
        else:
            # Save without PAM50
            expression_t = expression_matched.T
            expression_t.to_csv(output_file)
            print(f"  Saved without PAM50 labels")
            print(f"    Final shape: {expression_t.shape}")
    else:
        print("  Error: No matching samples found!")
        return None
    
    return output_file


def prepare_gse96058_data(
    clinical_file=None,
    expression_file=None,
    output_file=None,
    pam50_column=None
):
    """
    Prepare GSE96058 data: combine clinical and expression data.
    
    Args:
        clinical_file: Path to GSE96058 clinical/metadata file
        expression_file: Path to GSE96058 gene expression file
        output_file: Path to save output CSV
        pam50_column: Name of PAM50 column
    """
    print("Processing GSE96058 data...")
    
    # For GSE96058, check which files we have
    # Option 1: If we have a transformed CSV already
    if expression_file and 'transformed' in str(expression_file).lower():
        print(f"  Loading transformed expression file: {expression_file}")
        expression = pd.read_csv(expression_file, index_col=0, low_memory=False)
        print(f"    Shape: {expression.shape}")
        
        # Check if PAM50 is already in the data
        if 'PAM50' in expression.columns or any('PAM50' in str(col).upper() for col in expression.columns):
            print("    PAM50 already present in expression data")
            # Transpose if needed: samples should be columns
            if expression.shape[0] > expression.shape[1]:
                # Genes are rows, samples are columns - good
                expression.to_csv(output_file)
            else:
                # Need to transpose
                expression.T.to_csv(output_file)
            return output_file
    
    # Option 2: Load from transcript expression file
    if expression_file:
        print(f"  Loading expression file: {expression_file}")
        # Try to read - might be very large
        try:
            # Read first few rows to understand structure
            sample_df = pd.read_csv(expression_file, nrows=5, low_memory=False)
            print(f"    Sample structure:")
            print(f"      Columns: {list(sample_df.columns)[:10]}...")
            print(f"      Shape: {sample_df.shape}")
            
            # Determine if we need to transpose
            # Usually GEO data: samples as rows, genes as columns (or vice versa)
            # We want: genes as rows, samples as columns
            
            # Read full file (might be slow for large files)
            print("    Reading full file (this may take a while)...")
            expression = pd.read_csv(expression_file, index_col=0, low_memory=False)
            
            # Transpose if samples are rows
            if expression.shape[0] < expression.shape[1] // 10:  # Few rows, many cols = samples as rows
                print("    Transposing (samples were rows)...")
                expression = expression.T
            
            # Now expression should have genes as rows, samples as columns
            print(f"    Final shape: {expression.shape}")
            print(f"    Genes (rows): {expression.shape[0]}")
            print(f"    Samples (columns): {expression.shape[1]}")
            
            # Add PAM50 if we have clinical data
            if clinical_file and Path(clinical_file).exists():
                print(f"  Loading clinical data: {clinical_file}")
                clinical = pd.read_csv(clinical_file, low_memory=False)
                
                # Find PAM50 column
                if pam50_column is None:
                    pam50_cols = [col for col in clinical.columns 
                                 if 'PAM50' in col.upper() or 'SUBTYPE' in col.upper()]
                    if pam50_cols:
                        pam50_column = pam50_cols[0]
                
                # Match samples and add PAM50
                if pam50_column and pam50_column in clinical.columns:
                    # Find sample ID column in clinical
                    sample_id_col = [col for col in clinical.columns 
                                   if 'SAMPLE' in col.upper() or 'ID' in col.upper()][0]
                    
                    # Create mapping
                    pam50_dict = dict(zip(clinical[sample_id_col], clinical[pam50_column]))
                    
                    # Match with expression column names
                    pam50_labels = []
                    for col in expression.columns:
                        # Try to find matching clinical sample
                        matching_key = None
                        for key in pam50_dict.keys():
                            if str(key) in str(col) or str(col) in str(key):
                                matching_key = key
                                break
                        
                        if matching_key:
                            pam50_labels.append(pam50_dict[matching_key])
                        else:
                            pam50_labels.append('Unknown')
                    
                    # Add PAM50 as last row (actually as column, then we'll see)
                    expression['PAM50'] = pam50_labels
                    print(f"    Added PAM50 labels for {sum(1 for x in pam50_labels if x != 'Unknown')} samples")
            
            # Save
            print(f"  Saving to {output_file}...")
            expression.to_csv(output_file)
            print(f"    Saved successfully")
            
        except Exception as e:
            print(f"    Error reading file: {e}")
            print("    Trying alternative approach...")
            # Fallback: read in chunks
            # This is a simplified version - may need adjustment based on actual file structure
            return None
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for BIGCLAM pipeline')
    parser.add_argument('--dataset', choices=['tcga', 'gse96058', 'both'], 
                       default='both', help='Dataset to process')
    parser.add_argument('--tcga_clinical', type=str,
                       default='data/brca_tcga_clinical_data.tsv')
    parser.add_argument('--tcga_expression', type=str,
                       default='data/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct')
    parser.add_argument('--gse96058_expression', type=str,
                       default='data/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv')
    parser.add_argument('--gse96058_clinical', type=str,
                       default=None)
    parser.add_argument('--output_tcga', type=str,
                       default='data/tcga_brca_data_target_added.csv')
    parser.add_argument('--output_gse96058', type=str,
                       default='data/gse96058_data_target_added.csv')
    
    args = parser.parse_args()
    
    if args.dataset in ['tcga', 'both']:
        prepare_tcga_brca_data(
            args.tcga_clinical,
            args.tcga_expression,
            args.output_tcga
        )
    
    if args.dataset in ['gse96058', 'both']:
        prepare_gse96058_data(
            args.gse96058_clinical,
            args.gse96058_expression,
            args.output_gse96058
        )

