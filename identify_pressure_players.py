"""
================================================================================
SupeRR Selector Season 2 - Rajasthan Royals Hackathon
Identification of Most Reliable Players Under Pressure (Batting & Bowling)
Framework: T20 Cricket High-Pressure Conditions Framework (t20_high_pressure_conditions_framework.txt)
IPL 2027 Auction Eligible Cohort
With All 4 Advanced Strategic Suggestions Fully Implemented:
1. Phase-Adjusted Performance: Runs Saved Above Expected (RSAE) for Bowlers &
   Runs Above Expected (RAE) for Batters relative to Powerplay, Middle, and Death pars.
2. Bayesian Shrinkage (Empirical Bayes): Regressing small-sample rates toward
   population means (K=200 prior balls) + elevated volume protection.
3. Competition Caliber & Venue Dew Adjustments: 1.25x weight for IPL fixtures +
   1.10x dew multiplier for 2nd innings bowling defense in evening/night matches.
4. Dual-Tier Playoff Pressure: Base 1.25x on all playoff balls + 1.50x extra
   Crunch Leverage Multiplier (1.875x total) for high RRR, late death, and collapses.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore

# ------------------------------------------------------------------------------
# 1. CONFIGURATION & WEIGHTS
# ------------------------------------------------------------------------------
MIN_PRESSURE_BALLS_OVERALL = 150
MIN_PRESSURE_MATCHES_OVERALL = 10
K_BAYESIAN_PRIOR_BALLS = 200.0  # Empirical Bayes shrinkage prior strength

# Year-based calibrated recency decay weights (ALL non-zero, gradual decline)
YEAR_RECENCY_WEIGHTS = {
    2026: 1.00,  # Current season
    2025: 0.85,  # Immediate prior season
    2024: 0.65,  # Intermediate season
    2023: 0.50,  # Distant season
    2022: 0.35   # Baseline season (4 years ago, still non-zero weight)
}

# Elite international teams for competition-level normalization
TOP_TIER_TEAMS = {
    'India', 'Australia', 'England', 'South Africa', 'New Zealand', 'Pakistan', 
    'West Indies', 'Sri Lanka', 'Afghanistan',
    'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders', 'Rajasthan Royals', 
    'Gujarat Titans', 'Royal Challengers Bengaluru', 'Delhi Capitals', 
    'Lucknow Super Giants', 'Sunrisers Hyderabad', 'Punjab Kings'
}

# Calibrated Composite Pressure Reliability Score (CPRS) Factor Weights
# Fully accommodating Phase RAE/RSAE and Bayesian Shrinkage
BAT_WEIGHTS = {
    'w_sr': 0.18,        # Bayesian-regressed pressure Strike Rate
    'w_avg': 0.15,       # Bayesian-regressed pressure Average
    'rae': 0.14,         # Phase Runs Above Expected (RAE per 100 balls)
    'delta_sr': 0.11,    # Clutch Delta SR (Weighted Pressure SR - Career Base SR)
    'delta_avg': 0.08,   # Clutch Delta Avg (Weighted Pressure Avg - Career Base Avg)
    'p_bdry_pct': 0.08,  # Boundary % under pressure
    'p_dot_min': 0.08,   # Dot balls played under pressure minimization
    'w_cons': 0.09,      # Recency-weighted match consistency rate
    'w_vol': 0.09        # Volume confidence factor (log-balls exposure)
}

BOWL_WEIGHTS = {
    'w_econ': 0.15,      # Bayesian-regressed pressure Economy
    'rsae': 0.15,        # Phase Runs Saved Above Expected (RSAE per over)
    'w_wkts': 0.15,      # Wickets Taken Under Pressure (game-breaking strike volume)
    'w_sr': 0.13,        # Bayesian-regressed Wicket Strike Rate (balls per wicket)
    'p_bdry_supp': 0.08, # Boundaries against bowler under pressure suppression
    'delta_econ': 0.09,  # Clutch Delta Econ (Career Base Econ - Weighted Pressure Econ)
    'delta_sr': 0.05,    # Clutch Delta SR (Career Base SR - Weighted Pressure SR)
    'p_dot_pct': 0.07,   # Dot ball generation % under pressure
    'w_cons': 0.07,      # Recency-weighted match consistency rate
    'w_vol': 0.06        # Volume confidence factor (log-balls exposure)
}


def load_and_preprocess_data(data_dir="."):
    """Loads and preprocesses matches, competitions, mapping, and delivery-level data."""
    print("Loading data files...")
    bbb_path = os.path.join(data_dir, 'ball_by_ball.csv')
    matches_path = os.path.join(data_dir, 'matches.csv')
    comps_path = os.path.join(data_dir, 'competitions.csv')
    map_path = os.path.join(data_dir, 'competition_team_player_mapping.csv')

    df = pd.read_csv(bbb_path)
    matches = pd.read_csv(matches_path)
    comps = pd.read_csv(comps_path)
    mapping = pd.read_csv(map_path)

    excluded_players = set(mapping[mapping['team_name'].isin(['Pakistan', 'Pakistan A'])]['player_name'].unique())
    print(f"Loaded {len(df):,} deliveries across {len(matches):,} matches.")

    # Dates and seasons
    matches['match_date'] = pd.to_datetime(matches['match_date'])
    matches['season'] = matches['match_date'].dt.year
    matches['recency_w'] = matches['season'].map(YEAR_RECENCY_WEIGHTS).fillna(0.50)

    # Suggestion 3: Competition caliber (IPL 1.25x)
    m_merged = matches.merge(comps[['comp_id', 'comp_name']], on='comp_id', how='left')

    def get_comp_weight(row):
        comp_name = str(row.get('comp_name', ''))
        ht = str(row['home_team'])
        at = str(row['away_team'])
        if 'IPL' in comp_name or 'Indian Premier League' in comp_name or row['comp_id'] in [1331, 1404]:
            return 1.25  # IPL premier multiplier
        elif ht in TOP_TIER_TEAMS and at in TOP_TIER_TEAMS:
            return 1.00  # Elite international bilateral
        elif ht in TOP_TIER_TEAMS or at in TOP_TIER_TEAMS:
            return 0.85
        return 0.70

    m_merged['comp_w'] = m_merged.apply(get_comp_weight, axis=1)

    # Identify knockout/playoff matches (Final, Qualifier, Eliminator, Semi Final, or knock_out == 1)
    ko_mask = (m_merged['knock_out'] == 1) | (m_merged['stage'].isin(['Final', 'Qualifier', 'Eliminator', 'Semi Final']))
    ko_match_ids = set(m_merged.loc[ko_mask, 'match_id'])
    print(f"Identified {len(ko_match_ids)} Knockout / Playoff matches.")

    # Sort deterministically
    df = df.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

    # Clean runs and extras
    df['ball_runs'] = df['runs'] + df['extras1_runs'].fillna(0) + df['extras_2_runs'].fillna(0)
    
    # Bowler extras: wides and no-balls (Overstep) are charged to bowler; byes & legbyes are not
    df['bowler_extras'] = np.where(
        df['extras_type'].isin(['wide', 'Overstep']),
        df['extras1_runs'].fillna(0) + df['extras_2_runs'].fillna(0),
        0
    )
    df['bowler_runs_conceded'] = df['runs'] + df['bowler_extras']
    
    # Legal deliveries (exclude wide and no-ball for balls faced / balls bowled)
    df['is_legal'] = ~df['extras_type'].isin(['wide', 'Overstep'])
    
    # Bowler wickets (exclude run out, retired out, obstructing field)
    df['is_bowler_wkt'] = df['is_wicket'] & (
        ~df['dismissal_type'].isin(['RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'])
    )

    # First batting team per match
    first_bat = df[df['innings'] == 1].groupby('match_id')['batting_team_id'].first().rename('first_bat_team_id')
    m_merged = m_merged.merge(first_bat, on='match_id', how='left')

    # Innings 1 totals & legal balls for winning run rate calculation
    inn1_summary = df[df['innings'] == 1].groupby('match_id').agg(
        inn1_runs=('ball_runs', 'sum'),
        inn1_legal=('is_legal', 'sum')
    ).reset_index()
    inn1_summary['inn1_rpo'] = (inn1_summary['inn1_runs'] / np.maximum(1, inn1_summary['inn1_legal'])) * 6.0
    m_merged = m_merged.merge(inn1_summary, on='match_id', how='left')

    # Calculate Par RPO = Median 1st-Innings winning RPO grouped by (Competition, Season, Venue Country)
    won_bat1 = m_merged[m_merged['win_team_id'] == m_merged['first_bat_team_id']]
    par_c_s_v = won_bat1.groupby(['comp_id', 'season', 'venue_country'])['inn1_rpo'].median().to_dict()
    par_venue = won_bat1.groupby('venue_country')['inn1_rpo'].median().to_dict()
    global_par = won_bat1['inn1_rpo'].median()
    if pd.isna(global_par) or global_par <= 0:
        global_par = 8.50

    def get_par_rpo(row):
        key = (row['comp_id'], row['season'], row['venue_country'])
        if key in par_c_s_v and pd.notna(par_c_s_v[key]):
            return par_c_s_v[key]
        if row['venue_country'] in par_venue and pd.notna(par_venue[row['venue_country']]):
            return par_venue[row['venue_country']]
        return global_par

    m_merged['par_rpo'] = m_merged.apply(get_par_rpo, axis=1)
    m_merged['par_total'] = m_merged['par_rpo'] * 20.0

    # Targets & Initial Required Run Rate (IRRR) for Innings 2
    matches_inn1_runs = m_merged.set_index('match_id')['inn1_runs'].to_dict()
    revised_targets = m_merged.set_index('match_id')['revised_target'].to_dict()

    def get_target(mid):
        rt = revised_targets.get(mid, np.nan)
        if pd.notna(rt) and rt > 0:
            return rt
        tot = matches_inn1_runs.get(mid, np.nan)
        if pd.notna(tot):
            return tot + 1
        return np.nan

    m_merged['target'] = m_merged['match_id'].map(get_target)
    m_merged['irrr'] = np.where(
        m_merged['second_inn_revised_overs'] > 0,
        (m_merged['target'] / m_merged['second_inn_revised_overs']),
        m_merged['target'] / 20.0
    )

    # Map match-level metadata into delivery dataframe
    df['comp_name'] = df['match_id'].map(m_merged.set_index('match_id')['comp_name'])
    df['year'] = df['match_id'].map(m_merged.set_index('match_id')['season']).fillna(2024).astype(int)
    df['recency_w'] = df['match_id'].map(m_merged.set_index('match_id')['recency_w']).fillna(0.50)
    df['comp_w'] = df['match_id'].map(m_merged.set_index('match_id')['comp_w']).fillna(1.0)
    df['is_knockout'] = df['match_id'].isin(ko_match_ids)
    df['par_rpo'] = df['match_id'].map(m_merged.set_index('match_id')['par_rpo']).fillna(global_par)
    df['par_total'] = df['match_id'].map(m_merged.set_index('match_id')['par_total']).fillna(global_par * 20.0)
    df['target'] = df['match_id'].map(m_merged.set_index('match_id')['target'])
    df['irrr'] = df['match_id'].map(m_merged.set_index('match_id')['irrr']).fillna(9.0)

    # Cumulative match state
    print("Computing cumulative match states, partnerships, and run rates...")
    grouped_inn = df.groupby(['match_id', 'innings'])
    df['cum_runs'] = grouped_inn['ball_runs'].cumsum() - df['ball_runs']
    df['cum_wkts'] = grouped_inn['is_wicket'].cumsum() - df['is_wicket'].astype(int)
    df['legal_balls_bowled'] = grouped_inn['is_legal'].cumsum() - df['is_legal'].astype(int)
    df['balls_remaining'] = np.maximum(0, 120 - df['legal_balls_bowled'])
    df['crr'] = np.where(df['legal_balls_bowled'] > 0, (df['cum_runs'] / df['legal_balls_bowled']) * 6.0, 0.0)

    # Projected 20-over score
    df['proj_score'] = np.where(
        df['legal_balls_bowled'] >= 12,
        df['cum_runs'] + (df['cum_runs'] / np.maximum(1, df['legal_balls_bowled'])) * df['balls_remaining'],
        df['par_total']
    )

    # Chasing state
    df['runs_needed'] = np.maximum(0, df['target'] - df['cum_runs'])
    df['rrr'] = np.where(
        (df['innings'] == 2) & (df['balls_remaining'] > 0),
        (df['runs_needed'] / df['balls_remaining']) * 6.0,
        0.0
    )

    # Striker-level micro state (balls faced, runs, strike rate in current innings)
    grouped_striker = df.groupby(['match_id', 'innings', 'striker_name'])
    df['striker_balls'] = grouped_striker['is_legal'].cumsum() - df['is_legal'].astype(int)
    df['striker_runs'] = grouped_striker['runs'].cumsum() - df['runs']
    df['striker_sr'] = np.where(df['striker_balls'] > 0, (df['striker_runs'] / df['striker_balls']) * 100.0, 0.0)

    # Partnership micro state (runs, balls, strike rate of active partnership)
    grouped_part = df.groupby(['match_id', 'innings', 'cum_wkts'])
    df['part_runs'] = grouped_part['ball_runs'].cumsum() - df['ball_runs']
    df['part_balls'] = grouped_part['is_legal'].cumsum() - df['is_legal'].astype(int)
    df['part_sr'] = np.where(df['part_balls'] > 0, (df['part_runs'] / df['part_balls']) * 100.0, 0.0)

    # --------------------------------------------------------------------------
    # Suggestion 1: Phase Expected Economy and Strike Rate
    # --------------------------------------------------------------------------
    df['phase'] = np.where(df['overid'] <= 6, 'powerplay', np.where(df['overid'] <= 15, 'middle', 'death'))
    phase_stats = df.groupby('phase').agg(
        p_balls=('is_legal', 'sum'),
        p_runs=('ball_runs', 'sum'),
        p_bat_runs=('runs', 'sum')
    )
    phase_stats['par_econ'] = (phase_stats['p_runs'] / phase_stats['p_balls']) * 6.0
    phase_stats['par_sr'] = (phase_stats['p_bat_runs'] / phase_stats['p_balls']) * 100.0

    df['x_runs_conceded'] = (df['phase'].map(phase_stats['par_econ']) / 6.0) * df['is_legal']
    df['x_runs_scored'] = (df['phase'].map(phase_stats['par_sr']) / 100.0) * df['is_legal']

    # --------------------------------------------------------------------------
    # Suggestion 4: Dual-Tier Playoff Pressure Multiplier (Intra-Knockout Crunch)
    # --------------------------------------------------------------------------
    is_crunch_leverage = (
        ((df['innings'] == 2) & (df['rrr'] >= 10.0)) |
        (df['overid'] >= 18) |
        ((df['overid'] <= 6) & (df['cum_wkts'] >= 2)) |
        (df['cum_wkts'] >= 5)
    )
    df['ko_mult'] = np.where(
        df['is_knockout'],
        np.where(is_crunch_leverage, 1.875, 1.25),  # 1.50x extra crunch leverage inside playoffs
        1.00
    )

    # --------------------------------------------------------------------------
    # Suggestion 3: Evening/Night Dew Multiplier (2nd innings defending)
    # --------------------------------------------------------------------------
    df['ist_hour'] = pd.to_datetime(df['delivery_ist_time'], errors='coerce').dt.hour
    df['dew_mult'] = np.where((df['innings'] == 2) & (df['ist_hour'] >= 13), 1.10, 1.00)

    # Total Delivery Weight = Comp_Caliber * Recency * Playoff_Leverage * Dew_Factor
    df['weight'] = df['comp_w'] * df['recency_w'] * df['ko_mult'] * df['dew_mult']

    return df, excluded_players


def classify_pressure_situations(df):
    """
    Classifies each delivery into high-pressure conditions according to
    t20_high_pressure_conditions_framework.txt.
    
    Includes Playoff matches as high-pressure for BOTH sides on EVERY delivery.
    """
    print("Classifying pressure situations according to T20 High-Pressure Framework...")

    # Super Overs (Innings >= 3): Pressure for both sides
    c_super = df['innings'] >= 3

    # Playoff / Knockout Matches: Dual-side pressure on ALL deliveries
    c_knockout = df['is_knockout']

    # --------------------------------------------------------------------------
    # CASE 1: BATTER IN 1ST INNINGS (SETTING A TARGET)
    # --------------------------------------------------------------------------
    b_c1 = (df['innings'] == 1) & (
        # 1. Top-Order Collapse / Rebuilding Phase
        ((df['overid'] <= 6) & (df['cum_wkts'] >= 2)) |
        ((df['overid'] <= 10) & (df['cum_wkts'] >= 3)) |
        ((df['cum_wkts'] >= 3) & (df['cum_runs'] <= 45)) |
        ((df['cum_wkts'] >= 4) & (df['cum_runs'] <= 75)) |
        # 2. Run-Rate Deficit (Falling Behind Par) after Over 8.0
        ((df['overid'] > 8) & (df['crr'] <= df['par_rpo'] - 1.5)) |
        # 3. Death-Over Acceleration Under Deficit (Overs 16-20, proj score >= 15 runs below par)
        ((df['overid'] >= 16) & (df['proj_score'] <= df['par_total'] - 15)) |
        # 4. Micro-Pressure / Dot-Ball Accumulation
        ((df['striker_balls'] >= 10) & (df['striker_sr'] < 100))
    )

    # --------------------------------------------------------------------------
    # CASE 3: BATTER IN 2ND INNINGS (CHASING A TARGET)
    # --------------------------------------------------------------------------
    b_c3 = (df['innings'] == 2) & (
        # 1. Steep Chase from Ball One (IRRR >= 9.25 or >= 0.95 above Par RPO)
        (df['irrr'] >= 9.25) | (df['irrr'] >= df['par_rpo'] + 0.95) |
        # 2. Escalating Required Run Rate
        ((df['overid'] >= 7) & (df['overid'] <= 15) & (df['rrr'] >= 10.0)) |
        ((df['overid'] >= 16) & (df['rrr'] >= 12.0)) |
        # 3. Resource Depletion / Collapse in a Chase
        ((df['cum_wkts'] >= 4) & (df['cum_runs'] < 0.5 * df['target'])) |
        ((df['cum_wkts'] >= 6) & (df['runs_needed'] > 35)) |
        # 4. Climax / Last-Over Finish (Overs 19-20 with >= 10 RRR)
        ((df['overid'] >= 19) & (df['rrr'] >= 10.0) & (df['runs_needed'] > 0)) |
        # 5. Personal Stagnation Under Chase Pressure
        ((df['striker_balls'] >= 10) & (df['striker_sr'] < 100) & (df['rrr'] >= 9.0))
    )

    df['is_bat_pressure'] = c_knockout | c_super | b_c1 | b_c3

    # --------------------------------------------------------------------------
    # CASE 2: BOWLER IN 1ST INNINGS (CONTAINING & TAKING WICKETS)
    # --------------------------------------------------------------------------
    bw_c2 = (df['innings'] == 1) & (
        # 1. Powerplay Onslaught (Overs 1-6)
        (df['overid'] <= 6) |
        # 2. Unbroken Set-Batter Partnership
        ((df['part_runs'] >= 50) & (df['part_sr'] >= 130)) |
        ((df['striker_balls'] >= 25) & (df['striker_sr'] >= 150)) |
        ((df['overid'] > 8) & ((df['crr'] >= df['par_rpo'] + 1.0) | (df['cum_wkts'] <= 2))) |
        # 3. Death-Over Containment (Overs 16-20)
        ((df['overid'] >= 16) & (df['cum_wkts'] <= 4)) |  # >= 6 wickets in hand
        ((df['overid'] >= 16) & (df['proj_score'] >= df['par_total'] + 15))
    )

    # --------------------------------------------------------------------------
    # CASE 4: BOWLER IN 2ND INNINGS (DEFENDING A TARGET)
    # --------------------------------------------------------------------------
    bw_c4 = (df['innings'] == 2) & (
        # 1. Defending a Below-Par Total (IRRR <= 8.0 or >= 1.20 below Par RPO)
        (df['irrr'] <= 8.0) | (df['irrr'] <= df['par_rpo'] - 1.20) |
        # 2. The "Close Contest" Equilibrium Zone (Overs 12-20, RRR - CRR between -1.0 and +1.5)
        ((df['overid'] >= 12) & ((df['rrr'] - df['crr'] >= -1.0) & (df['rrr'] - df['crr'] <= 1.5))) |
        # 3. Vulnerable Death Overs (Overs 16-20, RRR <= 11.0, >= 5 wickets in hand)
        ((df['overid'] >= 16) & (df['rrr'] <= 11.0) & (df['cum_wkts'] <= 5) & (df['runs_needed'] > 0)) |
        # 4. Final-Over / Climax Defense
        ((df['overid'] >= 19) & (df['runs_needed'] <= 18) & (df['runs_needed'] > 0)) |
        ((df['runs_needed'] <= 24) & (df['balls_remaining'] <= 18) & (df['cum_wkts'] <= 6) & (df['runs_needed'] > 0)) |
        # 5. Unbroken Set-Batter Partnership
        ((df['part_runs'] >= 50) & (df['part_sr'] >= 130)) |
        ((df['striker_balls'] >= 25) & (df['striker_sr'] >= 150)) |
        ((df['overid'] > 8) & (df['cum_wkts'] <= 2)) |
        ((df['overid'] > 12) & (df['cum_wkts'] <= 3))
    )

    df['is_bowl_pressure'] = c_knockout | c_super | bw_c2 | bw_c4

    bat_p_count = df['is_bat_pressure'].sum()
    bowl_p_count = df['is_bowl_pressure'].sum()
    ko_p_count = df['is_knockout'].sum()

    print(f"Total Deliveries Analyzed: {len(df):,}")
    print(f"Batter Pressure Deliveries: {bat_p_count:,} ({bat_p_count/len(df)*100:.1f}%)")
    print(f"Bowler Pressure Deliveries: {bowl_p_count:,} ({bowl_p_count/len(df)*100:.1f}%)")
    print(f"Playoff Deliveries (High Pressure Both Sides): {ko_p_count:,} ({ko_p_count/len(df)*100:.1f}%)")

    return df


def evaluate_batters(df, excluded_players):
    """
    Computes recency-weighted baseline vs pressure performance metrics
    across all 5 years (2022-2026), incorporating Phase-Adjusted RAE and Bayesian Shrinkage.
    """
    print("\n--- Evaluating Batters (Phase RAE + Bayesian Shrinkage) ---")

    # 1. Career Overall Baseline (all balls 2022-2026)
    bat_all = df.groupby('striker_name').apply(lambda g: pd.Series({
        'tot_balls': len(g),
        'tot_runs': g['runs'].sum(),
        'tot_outs': g['is_wicket'].sum(),
        'base_sr': (g['runs'].sum() / max(1, len(g))) * 100,
        'base_avg': g['runs'].sum() / max(1, g['is_wicket'].sum()),
    }), include_groups=False)

    # 2. Batter Pressure Deliveries
    pdf = df[df['is_bat_pressure']]

    bat_p = pdf.groupby('striker_name').apply(lambda g: pd.Series({
        'p_balls': len(g),
        'p_runs': g['runs'].sum(),
        'p_outs': g['is_wicket'].sum(),
        'w_balls': g['weight'].sum(),
        'w_runs': (g['runs'] * g['weight']).sum(),
        'w_outs': (g['is_wicket'] * g['weight']).sum(),
        'w_x_runs': (g['x_runs_scored'] * g['weight']).sum(),
        'p_fours': g['is_four'].sum(),
        'p_sixes': g['is_six'].sum(),
        'p_dots': ((g['runs'] == 0) & g['is_legal']).sum(),
        'w_dots': (((g['runs'] == 0) & g['is_legal']) * g['weight']).sum(),
        'p_innings': g['match_id'].nunique(),
        'ko_balls': (g['is_knockout']).sum(),
        'balls_2022_2024': (g['year'] < 2025).sum(),
        'balls_2025_2026': (g['year'] >= 2025).sum(),
    }), include_groups=False)

    # Weighted metrics
    bat_p['w_sr'] = (bat_p['w_runs'] / np.maximum(1, bat_p['w_balls'])) * 100
    bat_p['w_avg'] = bat_p['w_runs'] / np.maximum(0.5, bat_p['w_outs'])
    bat_p['raw_p_sr'] = (bat_p['p_runs'] / np.maximum(1, bat_p['p_balls'])) * 100
    bat_p['raw_p_avg'] = bat_p['p_runs'] / np.maximum(1, bat_p['p_outs'])
    bat_p['p_bdry_pct'] = ((bat_p['p_fours'] + bat_p['p_sixes']) / np.maximum(1, bat_p['p_balls'])) * 100
    bat_p['p_dot_pct'] = (bat_p['p_dots'] / np.maximum(1, bat_p['p_balls'])) * 100
    bat_p['w_dot_pct'] = (bat_p['w_dots'] / np.maximum(1, bat_p['w_balls'])) * 100

    # Suggestion 1: Phase-Adjusted Runs Above Expected per 100 balls
    bat_p['rae_100'] = ((bat_p['w_runs'] - bat_p['w_x_runs']) / np.maximum(1, bat_p['w_balls'])) * 100.0

    bat = bat_p.join(bat_all, how='inner')
    bat['delta_sr'] = bat['w_sr'] - bat['base_sr']
    bat['delta_avg'] = bat['w_avg'] - bat['base_avg']

    # 3. Recency-Weighted Match Consistency Rate
    bat_m = pdf.groupby(['striker_name', 'match_id']).agg(
        m_runs=('runs', 'sum'),
        m_balls=('runs', 'count'),
        m_weight=('recency_w', 'first')
    ).reset_index()
    bat_m['m_sr'] = (bat_m['m_runs'] / np.maximum(1, bat_m['m_balls'])) * 100
    bat_m['is_success'] = (bat_m['m_sr'] >= 135) & (bat_m['m_runs'] >= 15)

    bat_cons = bat_m.groupby('striker_name').apply(
        lambda g: (g['is_success'] * g['m_weight']).sum() / g['m_weight'].sum(),
        include_groups=False
    ).rename('consistency_rate')
    bat = bat.join(bat_cons)

    # 4. Eligibility Filter: p_balls >= 150, p_innings >= 10
    bat_qual = bat[
        (bat['p_balls'] >= MIN_PRESSURE_BALLS_OVERALL) & 
        (bat['p_innings'] >= MIN_PRESSURE_MATCHES_OVERALL) & 
        (~bat.index.isin(excluded_players))
    ].copy()
    print(f"Eligible batters (overall >= {MIN_PRESSURE_BALLS_OVERALL} balls): {len(bat_qual)}")

    # --------------------------------------------------------------------------
    # Suggestion 2: Empirical Bayes Bayesian Shrinkage on Rates
    # --------------------------------------------------------------------------
    pop_sr = (bat_qual['w_runs'].sum() / bat_qual['w_balls'].sum()) * 100.0
    pop_avg = bat_qual['w_runs'].sum() / bat_qual['w_outs'].sum()
    bat_qual['reg_w_sr'] = (bat_qual['w_balls'] * bat_qual['w_sr'] + K_BAYESIAN_PRIOR_BALLS * pop_sr) / (bat_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)
    bat_qual['reg_w_avg'] = (bat_qual['w_balls'] * np.minimum(bat_qual['w_avg'], 60.0) + K_BAYESIAN_PRIOR_BALLS * pop_avg) / (bat_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)
    bat_qual['reg_rae'] = (bat_qual['w_balls'] * bat_qual['rae_100'] + K_BAYESIAN_PRIOR_BALLS * 0.0) / (bat_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)

    # Standardized Z-Scores
    bat_qual['z_w_sr'] = zscore(bat_qual['reg_w_sr'])
    bat_qual['z_w_avg'] = zscore(bat_qual['reg_w_avg'])
    bat_qual['z_rae'] = zscore(bat_qual['reg_rae'])
    bat_qual['z_delta_sr'] = zscore(bat_qual['delta_sr'])
    bat_qual['z_delta_avg'] = zscore(np.clip(bat_qual['delta_avg'], -20, 30))
    bat_qual['z_bdry'] = zscore(bat_qual['p_bdry_pct'])
    bat_qual['z_dot'] = -zscore(bat_qual['p_dot_pct'])
    bat_qual['z_cons'] = zscore(bat_qual['consistency_rate'])
    bat_qual['z_vol'] = zscore(np.log10(bat_qual['w_balls']))

    # Composite Pressure Reliability Score (CPRS)
    bat_qual['composite_pressure_score'] = (
        BAT_WEIGHTS['w_sr'] * bat_qual['z_w_sr'] +
        BAT_WEIGHTS['w_avg'] * bat_qual['z_w_avg'] +
        BAT_WEIGHTS['rae'] * bat_qual['z_rae'] +
        BAT_WEIGHTS['delta_sr'] * bat_qual['z_delta_sr'] +
        BAT_WEIGHTS['delta_avg'] * bat_qual['z_delta_avg'] +
        BAT_WEIGHTS['p_bdry_pct'] * bat_qual['z_bdry'] +
        BAT_WEIGHTS['p_dot_min'] * bat_qual['z_dot'] +
        BAT_WEIGHTS['w_cons'] * bat_qual['z_cons'] +
        BAT_WEIGHTS['w_vol'] * bat_qual['z_vol']
    )

    top_batters = bat_qual.sort_values('composite_pressure_score', ascending=False).reset_index()
    top_batters['rank'] = top_batters.index + 1
    return top_batters


def evaluate_bowlers(df, excluded_players):
    """
    Computes recency-weighted baseline vs pressure performance metrics
    incorporating Phase-Adjusted RSAE, Wickets Under Pressure, and Bayesian Shrinkage.
    """
    print("\n--- Evaluating Bowlers (Phase RSAE + Wickets Factor + Bayesian Shrinkage) ---")

    # 1. Career Overall Baseline (all balls 2022-2026)
    bowl_all = df.groupby('bowler_name').apply(lambda g: pd.Series({
        'tot_balls': g['is_legal'].sum(),
        'tot_runs': g['bowler_runs_conceded'].sum(),
        'tot_wkts': g['is_bowler_wkt'].sum(),
        'base_econ': (g['bowler_runs_conceded'].sum() / max(1, g['is_legal'].sum())) * 6,
        'base_sr': g['is_legal'].sum() / max(1, g['is_bowler_wkt'].sum()),
    }), include_groups=False)

    pdf = df[df['is_bowl_pressure']]

    bowl_p = pdf.groupby('bowler_name').apply(lambda g: pd.Series({
        'p_balls': g['is_legal'].sum(),
        'p_runs': g['bowler_runs_conceded'].sum(),
        'p_wkts': g['is_bowler_wkt'].sum(),
        'w_balls': (g['is_legal'] * g['weight']).sum(),
        'w_runs': (g['bowler_runs_conceded'] * g['weight']).sum(),
        'w_wkts': (g['is_bowler_wkt'] * g['weight']).sum(),
        'w_x_runs': (g['x_runs_conceded'] * g['weight']).sum(),
        'p_dots': ((g['bowler_runs_conceded'] == 0) & g['is_legal']).sum(),
        'p_fours': g['is_four'].sum(),
        'p_sixes': g['is_six'].sum(),
        'p_bdries': (g['is_four'] | g['is_six']).sum(),
        'w_bdries': ((g['is_four'] | g['is_six']) * g['weight']).sum(),
        'p_matches': g['match_id'].nunique(),
        'ko_balls': (g.loc[g['is_knockout'], 'is_legal']).sum(),
        'balls_2022_2024': (g.loc[g['year'] < 2025, 'is_legal']).sum(),
        'balls_2025_2026': (g.loc[g['year'] >= 2025, 'is_legal']).sum(),
    }), include_groups=False)

    # Weighted metrics
    bowl_p['w_econ'] = (bowl_p['w_runs'] / np.maximum(1, bowl_p['w_balls'])) * 6
    bowl_p['w_sr'] = bowl_p['w_balls'] / np.maximum(0.5, bowl_p['w_wkts'])
    bowl_p['raw_p_econ'] = (bowl_p['p_runs'] / np.maximum(1, bowl_p['p_balls'])) * 6
    bowl_p['raw_p_sr'] = bowl_p['p_balls'] / np.maximum(1, bowl_p['p_wkts'])
    bowl_p['p_dot_pct'] = (bowl_p['p_dots'] / np.maximum(1, bowl_p['p_balls'])) * 100
    bowl_p['p_bdry_pct'] = (bowl_p['p_bdries'] / np.maximum(1, bowl_p['p_balls'])) * 100
    bowl_p['w_bdry_pct'] = (bowl_p['w_bdries'] / np.maximum(1, bowl_p['w_balls'])) * 100

    # Suggestion 1: Phase Runs Saved Above Expected per Over
    bowl_p['rsae_ov'] = ((bowl_p['w_x_runs'] - bowl_p['w_runs']) / np.maximum(1, bowl_p['w_balls'])) * 6.0

    bowl = bowl_p.join(bowl_all, how='inner')
    bowl['delta_econ'] = bowl['base_econ'] - bowl['w_econ']
    bowl['delta_sr'] = bowl['base_sr'] - bowl['w_sr']

    # 3. Recency-Weighted Match Consistency Rate
    bowl_m = pdf.groupby(['bowler_name', 'match_id']).agg(
        m_runs=('bowler_runs_conceded', 'sum'),
        m_balls=('is_legal', 'sum'),
        m_wkts=('is_bowler_wkt', 'sum'),
        m_weight=('recency_w', 'first')
    ).reset_index()
    bowl_m['m_econ'] = (bowl_m['m_runs'] / np.maximum(1, bowl_m['m_balls'])) * 6
    bowl_m['is_success'] = (
        ((bowl_m['m_wkts'] >= 1) & (bowl_m['m_econ'] <= 8.5)) |
        (bowl_m['m_econ'] <= 7.0) |
        (bowl_m['m_wkts'] >= 2)
    )

    bowl_cons = bowl_m.groupby('bowler_name').apply(
        lambda g: (g['is_success'] * g['m_weight']).sum() / g['m_weight'].sum(),
        include_groups=False
    ).rename('consistency_rate')
    bowl = bowl.join(bowl_cons)

    # 4. Eligibility Filter: p_balls >= 150, p_matches >= 10
    bowl_qual = bowl[
        (bowl['p_balls'] >= MIN_PRESSURE_BALLS_OVERALL) & 
        (bowl['p_matches'] >= MIN_PRESSURE_MATCHES_OVERALL) & 
        (~bowl.index.isin(excluded_players))
    ].copy()
    print(f"Eligible bowlers (overall >= {MIN_PRESSURE_BALLS_OVERALL} balls): {len(bowl_qual)}")

    # --------------------------------------------------------------------------
    # Suggestion 2: Empirical Bayes Bayesian Shrinkage on Rates
    # --------------------------------------------------------------------------
    pop_econ = (bowl_qual['w_runs'].sum() / bowl_qual['w_balls'].sum()) * 6.0
    pop_sr = bowl_qual['w_balls'].sum() / bowl_qual['w_wkts'].sum()
    bowl_qual['reg_w_econ'] = (bowl_qual['w_balls'] * bowl_qual['w_econ'] + K_BAYESIAN_PRIOR_BALLS * pop_econ) / (bowl_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)
    bowl_qual['reg_w_sr'] = (bowl_qual['w_balls'] * bowl_qual['w_sr'] + K_BAYESIAN_PRIOR_BALLS * pop_sr) / (bowl_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)
    bowl_qual['reg_rsae'] = (bowl_qual['w_balls'] * bowl_qual['rsae_ov'] + K_BAYESIAN_PRIOR_BALLS * 0.0) / (bowl_qual['w_balls'] + K_BAYESIAN_PRIOR_BALLS)

    # Standardized Z-Scores
    bowl_qual['z_w_econ'] = -zscore(bowl_qual['reg_w_econ'])
    bowl_qual['z_rsae'] = zscore(bowl_qual['reg_rsae'])  # Higher runs saved is better!
    bowl_qual['z_w_sr'] = -zscore(bowl_qual['reg_w_sr'])
    bowl_qual['z_w_wkts'] = zscore(bowl_qual['w_wkts'])  # Wickets under pressure
    bowl_qual['z_delta_econ'] = zscore(np.clip(bowl_qual['delta_econ'], -3.0, 3.0))
    bowl_qual['z_delta_sr'] = zscore(np.clip(bowl_qual['delta_sr'], -15.0, 15.0))
    bowl_qual['z_dot'] = zscore(bowl_qual['p_dot_pct'])
    bowl_qual['z_bdry'] = -zscore(bowl_qual['p_bdry_pct'])
    bowl_qual['z_cons'] = zscore(bowl_qual['consistency_rate'])
    bowl_qual['z_vol'] = zscore(np.log10(bowl_qual['w_balls']))

    # Composite Pressure Reliability Score (CPRS)
    bowl_qual['composite_pressure_score'] = (
        BOWL_WEIGHTS['w_econ'] * bowl_qual['z_w_econ'] +
        BOWL_WEIGHTS['rsae'] * bowl_qual['z_rsae'] +
        BOWL_WEIGHTS['w_wkts'] * bowl_qual['z_w_wkts'] +
        BOWL_WEIGHTS['w_sr'] * bowl_qual['z_w_sr'] +
        BOWL_WEIGHTS['delta_econ'] * bowl_qual['z_delta_econ'] +
        BOWL_WEIGHTS['delta_sr'] * bowl_qual['z_delta_sr'] +
        BOWL_WEIGHTS['p_dot_pct'] * bowl_qual['z_dot'] +
        BOWL_WEIGHTS['p_bdry_supp'] * bowl_qual['z_bdry'] +
        BOWL_WEIGHTS['w_cons'] * bowl_qual['z_cons'] +
        BOWL_WEIGHTS['w_vol'] * bowl_qual['z_vol']
    )

    top_bowlers = bowl_qual.sort_values('composite_pressure_score', ascending=False).reset_index()
    top_bowlers['rank'] = top_bowlers.index + 1
    return top_bowlers


def export_and_display_results(top_batters, top_bowlers, output_dir="."):
    """Exports top 10 tables to CSV and prints comprehensive formatted summaries."""
    top10_bat = top_batters.head(10).copy()
    top10_bowl = top_bowlers.head(10).copy()

    # User-facing columns (explicitly including dot balls played under pressure & boundaries conceded)
    bat_cols = [
        'rank', 'striker_name', 'p_balls', 'ko_balls', 'balls_2022_2024', 'balls_2025_2026',
        'p_dots', 'p_dot_pct', 'w_sr', 'raw_p_sr', 'w_avg', 'base_sr', 'delta_sr', 'rae_100',
        'p_bdry_pct', 'consistency_rate', 'composite_pressure_score'
    ]
    bowl_cols = [
        'rank', 'bowler_name', 'p_balls', 'ko_balls', 'p_wkts', 'w_wkts', 'balls_2022_2024', 'balls_2025_2026',
        'p_bdries', 'p_bdry_pct', 'w_econ', 'raw_p_econ', 'base_econ', 'delta_econ', 'rsae_ov',
        'w_sr', 'p_dot_pct', 'consistency_rate', 'composite_pressure_score'
    ]

    out_bat_path = os.path.join(output_dir, 'top_10_batters_under_pressure.csv')
    out_bowl_path = os.path.join(output_dir, 'top_10_bowlers_under_pressure.csv')

    top10_bat[bat_cols].to_csv(out_bat_path, index=False)
    top10_bowl[bowl_cols].to_csv(out_bowl_path, index=False)

    print("\n" + "=" * 155)
    print("TOP 10 MOST RELIABLE BATTERS UNDER PRESSURE (DOT BALLS PLAYED UNDER PRESSURE INCLUDED)")
    print("=" * 155)
    print(f"{'Rank':<5}{'Player Name':<22}{'Balls':<7}{'KO':<5}{'22-24':<7}{'25-26':<7}{'Dots':<7}{'Dot%':<7}{'Wtd SR':<9}{'Wtd Avg':<9}{'Base SR':<9}{'Δ SR':<8}{'RAE/100':<9}{'Consist':<9}{'CPRS'}")
    print("-" * 155)
    for _, r in top10_bat.head(10)[bat_cols].iterrows():
        print(f"{int(r['rank']):<5}{r['striker_name']:<22}{int(r['p_balls']):<7}{int(r['ko_balls']):<5}{int(r['balls_2022_2024']):<7}{int(r['balls_2025_2026']):<7}{int(r['p_dots']):<7}{r['p_dot_pct']:<7.1f}{r['w_sr']:<9.1f}{r['w_avg']:<9.1f}{r['base_sr']:<9.1f}{r['delta_sr']:<+8.1f}{r['rae_100']:<+9.1f}{r['consistency_rate']*100:<8.1f}%{r['composite_pressure_score']:<+.3f}")

    print("\n" + "=" * 160)
    print("TOP 10 MOST RELIABLE BOWLERS UNDER PRESSURE (BOUNDARIES CONCEDED UNDER PRESSURE INCLUDED)")
    print("=" * 160)
    print(f"{'Rank':<5}{'Player Name':<20}{'Balls':<7}{'KO':<5}{'P Wkts':<8}{'Wtd Wkt':<9}{'Bdries':<8}{'Bdry%':<8}{'Wtd Econ':<10}{'Base Econ':<10}{'Δ Econ':<8}{'RSAE/ov':<9}{'Wtd SR':<8}{'Consist':<9}{'CPRS'}")
    print("-" * 160)
    for _, r in top10_bowl.head(10)[bowl_cols].iterrows():
        print(f"{int(r['rank']):<5}{r['bowler_name']:<20}{int(r['p_balls']):<7}{int(r['ko_balls']):<5}{int(r['p_wkts']):<8}{r['w_wkts']:<9.1f}{int(r['p_bdries']):<8}{r['p_bdry_pct']:<8.1f}{r['w_econ']:<10.2f}{r['base_econ']:<10.2f}{r['delta_econ']:<+8.2f}{r['rsae_ov']:<+9.2f}{r['w_sr']:<8.1f}{r['consistency_rate']*100:<8.1f}%{r['composite_pressure_score']:<+.3f}")
    print("=" * 160)
    print(f"\nCSV files saved successfully:\n1. {out_bat_path}\n2. {out_bowl_path}")


def main():
    df, excluded_players = load_and_preprocess_data(".")
    df = classify_pressure_situations(df)
    top_batters = evaluate_batters(df, excluded_players)
    top_bowlers = evaluate_bowlers(df, excluded_players)
    export_and_display_results(top_batters, top_bowlers, ".")


if __name__ == "__main__":
    main()