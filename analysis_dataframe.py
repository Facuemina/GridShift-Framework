import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
import os

def run_analysis(csv_path='all_rats_metrics.csv'):
    # =========================================================================
    # 1. LOAD DATA & PREPROCESSING
    # =========================================================================
    df = pd.read_csv(csv_path)
    
    # Create a Unique ID for repeated measures alignment (Rat + Cell_Type + Neuron_ID)
    df['UID'] = df['Rat'].astype(str) + '_' + df['Cell_Type'] + '_' + df['Neuron_ID'].astype(str)
    
    # Define session groupings and cell types
    all_sessions = ['0°', '30°', '60°', "0°'"]
    shift_sessions = ['30°', '60°', "0°'"]  # 0° excluded as it is the baseline for shifts
    cell_types = ['Omni', 'Conj1', 'Conj2']
    
    # --- NEW: Categorize Conjunctive cells into UP and DOWN based on Pref_Angle ---
    df['Cell_Subtype'] = df['Cell_Type'] # Default copy
    
    # Masks for strictly UP and strictly DOWN preferences
    up_mask = (df['Pref_Angle'] > 0) & (df['Pref_Angle'] < np.pi)
    down_mask = (df['Pref_Angle'] > np.pi) & (df['Pref_Angle'] < 2 * np.pi)
    
    # Apply to Conj1
    df.loc[(df['Cell_Type'] == 'Conj1') & up_mask, 'Cell_Subtype'] = 'Conj1 UP'
    df.loc[(df['Cell_Type'] == 'Conj1') & down_mask, 'Cell_Subtype'] = 'Conj1 DOWN'
    
    # Apply to Conj2
    df.loc[(df['Cell_Type'] == 'Conj2') & up_mask, 'Cell_Subtype'] = 'Conj2 UP'
    df.loc[(df['Cell_Type'] == 'Conj2') & down_mask, 'Cell_Subtype'] = 'Conj2 DOWN'

    # Note: Cells with Pref_Angle exactly == 0 or np.pi remain just 'Conj1' or 'Conj2' 
    # and will naturally be excluded from the subtype specific analysis below.
    
    # Isolate Omni cells for the targeted directional map analysis
    df_omni = df[df['Cell_Type'] == 'Omni'].copy()
    
    # Helper functions for stats
    def apply_friedman(data, val_col, sessions):
        pivot = data.pivot(index='UID', columns='Session_Angle', values=val_col).dropna(subset=sessions)
        arrays = [pivot[ses].values for ses in sessions]
        
        if len(arrays) == len(sessions) and all(len(a) > 0 for a in arrays):
            stat, p_val = friedmanchisquare(*arrays)
            return p_val
        return np.nan

    def fmt_p(p):
        return f"{p:.2e}" if pd.notnull(p) else "NaN"

    # =========================================================================
    # FIGURE 1: TOTAL SHIFTS (Separated by Cell Type)
    # =========================================================================
    fig1, axes1 = plt.subplots(3, 2, figsize=(10, 12), sharex=True)
    df_shifts = df[df['Session_Angle'].isin(shift_sessions)]
    
    for i, ctype in enumerate(cell_types):
        df_sub = df_shifts[df_shifts['Cell_Type'] == ctype]
        
        # Shift X
        pval_x = apply_friedman(df_sub, 'Shift_X_Total', shift_sessions)
        sns.boxplot(data=df_sub, x='Session_Angle', y='Shift_X_Total', fill=False, ax=axes1[i, 0])
        axes1[i, 0].set_title(f'{ctype} - Total Shift X\nFriedman p = {fmt_p(pval_x)}')
        axes1[i, 0].axhline(0, color='k', linestyle='--')
        axes1[i, 0].set_xlabel('')
        
        # Shift Y
        pval_y = apply_friedman(df_sub, 'Shift_Y_Total', shift_sessions)
        sns.boxplot(data=df_sub, x='Session_Angle', y='Shift_Y_Total', fill=False, ax=axes1[i, 1])
        axes1[i, 1].set_title(f'{ctype} - Total Shift Y\nFriedman p = {fmt_p(pval_y)}')
        axes1[i, 1].axhline(0, color='k', linestyle='--')
        axes1[i, 1].set_xlabel('')

    fig1.tight_layout()
    plt.show()

    # =========================================================================
    # FIGURE 2: OMNI SHIFTS UP VS DOWN TRAJECTORIES
    # =========================================================================
    shift_metrics = [('Shift_X_Up', 'Shift_X_Down', 'Shift X'), 
                     ('Shift_Y_Up', 'Shift_Y_Down', 'Shift Y')]
                     
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    df_omni_shifts = df_omni[df_omni['Session_Angle'].isin(shift_sessions)]
    
    for i, (col_up, col_down, title) in enumerate(shift_metrics):
        # Melt data for Seaborn 'hue' separation
        melted = df_omni_shifts.melt(id_vars=['UID', 'Session_Angle'], 
                                     value_vars=[col_up, col_down],
                                     var_name='Direction', value_name='Shift')
        melted['Direction'] = melted['Direction'].map({col_up: 'Up', col_down: 'Down'})
        
        sns.boxplot(data=melted, x='Session_Angle', y='Shift', hue='Direction', fill=False, ax=axes2[i])
        axes2[i].axhline(0, color='k', linestyle='--')
        
        # Compute Wilcoxon tests per session to include in the title
        pvals = []
        for ses in shift_sessions:
            ses_data = df_omni_shifts[df_omni_shifts['Session_Angle'] == ses].dropna(subset=[col_up, col_down])
            if len(ses_data) > 0:
                _, pval = wilcoxon(ses_data[col_up], ses_data[col_down])
                pvals.append(f"{ses}: {fmt_p(pval)}")
            else:
                pvals.append(f"{ses}: NaN")
                
        title_str = f'Omni {title} (Up vs Down)\nWilcoxon p: ' + " | ".join(pvals)
        axes2[i].set_title(title_str, fontsize=10)
                
    fig2.tight_layout()
    plt.show()

    # =========================================================================
    # FIGURE 3: MEAN FIRING RATES (Separated by Cell Subtype)
    # =========================================================================
    # We now loop over 5 specific groups for Firing Rate analysis
    fr_groups = ['Omni', 'Conj1 UP', 'Conj1 DOWN', 'Conj2 UP', 'Conj2 DOWN']
    
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 5))
    
    for i, grp in enumerate(fr_groups):
        df_sub = df[df['Cell_Subtype'] == grp]
        pval_fr = apply_friedman(df_sub, 'Mean_FR', all_sessions)
        
        sns.boxplot(data=df_sub, x='Session_Angle', y='Mean_FR', order=all_sessions, fill=False, ax=axes3[i])
        axes3[i].set_title(f'{grp} Mean FR\nFriedman p = {fmt_p(pval_fr)}')
        axes3[i].set_xlabel('Session Angle')
        if i == 0:
            axes3[i].set_ylabel('Mean Firing Rate')
        else:
            axes3[i].set_ylabel('')

    fig3.tight_layout()
    plt.show()

    # =========================================================================
    # FIGURE 4: OMNI FIRING RATES UP VS DOWN TRAJECTORIES
    # =========================================================================
    if 'Mean_FR_Up' in df_omni.columns and 'Mean_FR_Down' in df_omni.columns:
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        
        melted_fr = df_omni.melt(id_vars=['UID', 'Session_Angle'], 
                                 value_vars=['Mean_FR_Up', 'Mean_FR_Down'],
                                 var_name='Direction', value_name='Firing_Rate')
        melted_fr['Direction'] = melted_fr['Direction'].map({'Mean_FR_Up': 'Up', 'Mean_FR_Down': 'Down'})
        
        sns.boxplot(data=melted_fr, x='Session_Angle', y='Firing_Rate', hue='Direction', order=all_sessions, fill=False, ax=ax4)
        
        # Compute Wilcoxon tests per session to include in the title
        pvals = []
        for ses in all_sessions:
            ses_data = df_omni[df_omni['Session_Angle'] == ses].dropna(subset=['Mean_FR_Up', 'Mean_FR_Down'])
            if len(ses_data) > 0:
                _, pval = wilcoxon(ses_data['Mean_FR_Up'], ses_data['Mean_FR_Down'])
                pvals.append(f"{ses}: {fmt_p(pval)}")
            else:
                pvals.append(f"{ses}: NaN")
                
        # Split string for cleaner title rendering
        title_str = (f'Omni Mean Firing Rate (Up vs Down maps)\nWilcoxon p-vals:\n' 
                     f'{" | ".join(pvals[:2])}\n{" | ".join(pvals[2:])}')
        ax4.set_title(title_str, fontsize=10)
        
        fig4.tight_layout()
        plt.show()
    else:
        print("\n[Notice] 'Mean_FR_Up' and 'Mean_FR_Down' columns were not found in the CSV.")

if __name__ == "__main__":
    
    base_path = os.path.split(os.getcwd())[0] 
    NUMS = range(1,6) #define rats to analyze
    NUMS_LABEL = "".join('_'+str(i) for i in NUMS)
    output_path = os.path.join(base_path, f'all_rats_metrics_nums{NUMS_LABEL}.csv')

    run_analysis(output_path)