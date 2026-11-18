# SupeRR Selector Season 2: Identifying the Most Reliable Players Under Pressure
## Round 1 Official Methodology Report & Analytical Framework

**Competition:** SupeRR Selector Season 2 — Rajasthan Royals  
**Dataset Analyzed:** Historical IPL & T20 International (T20I) Records (2022–2026; 200,999 Deliveries across 883 Matches)  
**Deliverables:** Top 10 Batters Under Pressure, Top 10 Bowlers Under Pressure, Reproducible Pipeline (`identify_pressure_players_clean.py`)  

---

## Executive Summary & Final Unstop Rankings

In T20 cricket, aggregate career statistics often mask how players respond when match leverage peaks. Under acute scoreboard or resource pressure, ordinary batting baselines deteriorate—dot-ball rates climb and strike rates compress—while bowling discipline is tested by fielding restrictions, set partnerships, and dew.

To separate players who **consistently rise under pressure** from those whose output regresses, we engineered a delivery-level, Empirical Bayes–standardized rating system: the **Composite Pressure Reliability Score (CPRS)**. Every legal delivery across the 2022–2026 IPL and T20I dataset is evaluated against context-specific match-state triggers and compared directly against each player's own career baseline.

### Final Top 10 Batters Under Pressure (Ordered Rank 1–10)

| Rank | Player Name | Pressure Balls | Knockout Balls | 2022–24 Balls | 2025–26 Balls | Pressure Dots | Dot % | Wtd Pressure SR | Wtd Pressure Avg | Base SR | Clutch Δ SR | Phase RAE/100 | Consistency Rate | Composite Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Vaibhav Sooryavanshi** | 336 | 94 | 0 | 336 | 115 | 34.2% | **227.5** | **45.5** | 211.3 | **+16.1** | **+97.8** | **61.4%** | **+2.912** |
| **2** | **Virat Kohli** | 960 | 173 | 585 | 375 | **255** | **26.6%** | **158.0** | **55.9** | 140.2 | **+17.8** | **+24.9** | **48.9%** | **+1.909** |
| **3** | **Quinton de Kock** | 558 | 5 | 406 | 152 | 165 | 29.6% | **169.1** | **41.2** | 143.6 | **+25.5** | **+39.6** | **52.9%** | **+1.694** |
| **4** | **Sai Sudharsan** | 746 | 186 | 341 | 405 | **198** | **26.5%** | **158.0** | **52.3** | 142.9 | **+15.1** | **+27.0** | **49.3%** | **+1.694** |
| **5** | **Shreyas Iyer** | 1004 | 74 | 367 | 637 | **263** | **26.2%** | **160.8** | **45.8** | 147.6 | **+13.2** | **+26.9** | **48.4%** | **+1.672** |
| **6** | **Rajat Patidar** | 568 | 196 | 275 | 293 | 156 | 27.5% | **170.7** | **41.1** | 158.9 | **+11.9** | **+38.2** | **47.4%** | **+1.563** |
| **7** | **Ryan Rickelton** | 323 | 0 | 82 | 241 | 111 | 34.4% | **179.8** | **34.2** | 152.8 | **+27.0** | **+47.3** | **53.2%** | **+1.526** |
| **8** | **Phil Salt** | 416 | 37 | 193 | 223 | 111 | 26.7% | **173.9** | **38.3** | 163.6 | **+10.3** | **+40.7** | **47.2%** | **+1.333** |
| **9** | **Jonny Bairstow** | 286 | 49 | 237 | 49 | 90 | 31.5% | **166.5** | **42.0** | 145.5 | **+21.0** | **+34.5** | **55.2%** | **+1.265** |
| **10** | **Priyansh Arya** | 210 | 60 | 0 | 210 | 66 | 31.4% | **185.6** | **26.7** | 182.7 | **+2.9** | **+57.7** | **62.1%** | **+1.201** |

---

### Final Top 10 Bowlers Under Pressure (Ordered Rank 1–10)

| Rank | Player Name | Pressure Balls | Knockout Balls | Pressure Wickets | Wtd Wickets | Bdries Conceded | Bdry Conceded % | Wtd Pressure Econ | Base Econ | Clutch Δ Econ | Phase RSAE/ov | Wtd Wicket SR | Consistency Rate | Composite Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Sunil Narine** | 963 | 48 | **43** | **41.4** | **121** | **12.6%** | **7.40** | 6.86 | **-0.54** | **+1.27** | **21.1** | **66.4%** | **+1.287** |
| **2** | **Rashid Khan** | 1578 | 216 | **92** | **73.5** | **224** | **14.2%** | **8.28** | 7.43 | **-0.84** | **+0.47** | **17.0** | **61.9%** | **+1.286** |
| **3** | **Blessing Muzarabani** | 624 | 0 | 35 | 23.7 | 106 | 17.0% | **7.43** | 7.52 | **+0.09** | **+1.59** | **16.4** | **66.1%** | **+1.271** |
| **4** | **Liam Dawson** | 256 | 0 | 16 | 14.6 | **31** | **12.1%** | **6.96** | 7.92 | **+0.96** | **+1.76** | **14.1** | **63.0%** | **+1.266** |
| **5** | **Jasprit Bumrah** | 1180 | 67 | **52** | **45.9** | **180** | **15.3%** | **7.50** | 7.26 | **-0.24** | **+1.47** | **24.4** | **62.0%** | **+1.261** |
| **6** | **Bhuvneshwar Kumar** | 1457 | 132 | **73** | **67.7** | 272 | 18.7% | **8.41** | 8.28 | **-0.13** | **+0.70** | **18.7** | **51.2%** | **+1.253** |
| **7** | **Arshdeep Singh** | 1950 | 60 | **115** | **98.7** | 423 | 21.7% | **9.26** | 9.12 | **-0.14** | **-0.19** | **16.5** | **46.0%** | **+1.144** |
| **8** | **Krunal Pandya** | 801 | 123 | **40** | **43.0** | **118** | **14.7%** | **7.85** | 7.81 | **-0.04** | **+0.66** | **18.5** | **49.9%** | **+1.089** |
| **9** | **Ripon Mondol** | 181 | 54 | 17 | 12.0 | **23** | **12.7%** | **7.26** | 7.46 | **+0.19** | **+1.89** | **10.5** | **67.7%** | **+1.011** |
| **10** | **Axar Patel** | 1497 | 24 | **71** | **60.9** | 256 | 17.1% | **8.27** | 7.89 | **-0.38** | **+0.30** | **20.3** | **51.8%** | **+0.976** |

---

## 1. Pressure Definition: Match Situations & Indicators

Pressure in T20 cricket is asymmetric: a match state that favors the batting side (e.g., chasing 155 with 8 wickets in hand) places acute defensive pressure on the bowling unit, whereas a steep chase or early collapse places acute survival and acceleration pressure on the batters. 

To anchor our thresholds to scoring environments across eras and countries, we first computed a **Context Par Run Rate ($\text{Par RPO}$)** defined as the median 1st-innings winning run rate grouped by `(Competition, Season, Venue Country)`. Deliveries are then classified as **High-Pressure** separately for batters and bowlers across four distinct match regimes, plus universal high-pressure flags for **Super Overs** and **Knockout/Playoff fixtures** (Finals, Qualifiers, Eliminators, Semi-Finals).

### A. Batting High-Pressure Regime
1. **1st Innings (Setting a Target under Resource/Tempo Deficit):**
   * **Top-Order Collapse:** Team loses $\ge 2$ wickets in the Powerplay (Overs 1–6), $\ge 3$ wickets in Overs 1–10, or falls to $\le 45/3$ or $\le 75/4$.
   * **Run-Rate Stagnation:** Current Run Rate ($\text{CRR}$) falls $\ge 1.5\text{ RPO}$ below $\text{Par RPO}$ after Over 8.
   * **Death-Over Catch-Up:** Overs 16–20 when the projected 20-over score is $\ge 15$ runs below par ($\text{Par Total} - 15$).
   * **Individual Dot-Ball Pressure:** Striker has faced $\ge 10$ balls with an innings Strike Rate $< 100$.
2. **2nd Innings (Chasing a Target):**
   * **Steep Target from Ball One:** Initial Required Run Rate ($\text{IRRR}$) $\ge 9.25\text{ RPO}$ or $\ge 0.95\text{ RPO}$ above $\text{Par RPO}$.
   * **Escalating Ask:** Current Required Run Rate ($\text{RRR}$) reaches $\ge 10.0\text{ RPO}$ in Overs 7–15 or $\ge 12.0\text{ RPO}$ in Overs 16–20.
   * **Chase Collapse:** $\ge 4$ wickets lost before reaching $50\%$ of the target, or $\ge 6$ wickets lost with $> 35$ runs still required.
   * **Death Climax & Stagnation:** Overs 19–20 with $\text{RRR} \ge 10.0$, or striker SR $< 100$ after $10+$ balls while $\text{RRR} \ge 9.0$.

### B. Bowling High-Pressure Regime
1. **1st Innings (Containing Momentum & Breaking Stands):**
   * **Powerplay Fielding Restrictions:** All deliveries in Overs 1–6 (only 2 fielders outside the inner circle).
   * **Set-Batter & Partnership Dominance:** Active partnership $\ge 50$ runs at $\text{SR} \ge 130$, striker on $25+$ balls at $\text{SR} \ge 150$, or batting side scoring $\ge 1.0\text{ RPO}$ above par after Over 8 with $\le 2$ wickets down.
   * **Death-Over Assault:** Overs 16–20 when the batting team retains $\ge 6$ wickets in hand ($\le 4$ down) or projects $\ge 15$ runs above par.
2. **2nd Innings (Defending a Target):**
   * **Below-Par Cushion:** Defending a target where $\text{IRRR} \le 8.0\text{ RPO}$ or $\ge 1.20\text{ RPO}$ below $\text{Par RPO}$.
   * **Close-Contest Equilibrium:** Overs 12–20 where $\text{RRR} - \text{CRR} \in [-1.0, +1.5]\text{ RPO}$ (chase is neck-and-neck).
   * **Vulnerable Death Defense & Climax:** Overs 16–20 with $\text{RRR} \le 11.0$ and $\ge 5$ wickets in hand; Overs 19–20 defending $\le 18$ runs; or defending $\le 24$ runs off $\le 18$ balls with $\ge 4$ wickets in hand.

Across the 200,999 deliveries analyzed, this framework isolates high-leverage situations cleanly while preserving sufficient sample depth across all five seasons.

---

## 2. Performance Measurement & Scoring Logic

### A. Player Baselines & Clutch Delta ($\Delta$)
For every player, we compute a **Career Overall Baseline** across all deliveries faced/bowled from 2022–2026 (`Base SR` and `Base Avg` for batters; `Base Econ` and `Base Wicket SR` for bowlers). We then measure the **Clutch Delta**—the direct shift between weighted high-pressure output and overall career output:
$$\Delta \text{SR}_{\text{bat}} = w\text{SR}_{\text{pressure}} - \text{Base SR}, \qquad \Delta \text{Econ}_{\text{bowl}} = \text{Base Econ} - w\text{Econ}_{\text{pressure}}$$
Positive deltas indicate players who actively elevate their game when pressure spikes rather than merely coasting on flat-track averages.

### B. Phase-Adjusted Expected Runs ($\text{RAE}_{100}$ & $\text{RSAE}_{\text{ov}}$)
Raw economy and strike rate penalize death-over specialists and reward middle-over accumulators. Using dataset-wide phase baselines—**Powerplay (8.67 RPO)**, **Middle (8.34 RPO)**, and **Death (10.21 RPO)**—we calculate:
* **Batters ($\text{RAE}_{100}$):** Weighted runs scored above phase expectation per 100 pressure balls.
* **Bowlers ($\text{RSAE}_{\text{ov}}$):** Weighted runs saved below phase expectation per pressure over.

### C. Delivery Weighting & Empirical Bayes Shrinkage
Each pressure delivery is weighted by match context:
$$w_i = W_{\text{comp}} \times W_{\text{recency}} \times W_{\text{knockout}} \times W_{\text{dew}}$$
* **Recency Decay ($W_{\text{recency}}$):** $2026: 1.00,\; 2025: 0.85,\; 2024: 0.65,\; 2023: 0.50,\; 2022: 0.35$.
* **Dual-Tier Knockout Leverage ($W_{\text{knockout}}$):** $1.25\times$ base multiplier on all playoff/knockout balls, escalating to $1.875\times$ ($1.25 \times 1.50$) on intra-knockout crunch balls ($\text{RRR} \ge 10$, Overs 18–20, or $\ge 5$ wickets down).
* **Evening Dew Factor ($W_{\text{dew}}$):** $1.10\times$ multiplier for 2nd-innings bowlers defending targets in afternoon/night matches.

To prevent small-sample inflation, weighted rate metrics are shrunk toward population means using **Empirical Bayes regression** with a prior strength of $K = 200$ deliveries:
$$\hat{\theta}_{\text{reg}} = \frac{w\text{Balls} \cdot \hat{\theta}_{\text{player}} + K \cdot \mu_{\text{pop}}}{w\text{Balls} + K}$$

### D. Composite Pressure Reliability Score (CPRS) Formulation
All component metrics are standardized via Z-scores across the eligible cohort and combined into a single interpretable index:

**Batting CPRS ($CPRS_{\text{Bat}}$):**
* **18%** Bayesian-Regressed Pressure Strike Rate ($w\text{SR}$)
* **15%** Bayesian-Regressed Pressure Average ($w\text{Avg}$)
* **14%** Phase Runs Above Expected ($\text{RAE}_{100}$)
* **11%** Clutch Delta Strike Rate ($\Delta \text{SR}$)
* **9%** Volume Confidence Factor ($\log_{10} w\text{Balls}$)
* **9%** Recency-Weighted Match Consistency Rate
* **8%** Clutch Delta Average ($\Delta \text{Avg}$)
* **8%** Boundary Generation % ($p\text{Bdry\%}$)
* **8%** Dot-Ball Minimization ($-p\text{Dot\%}$)

**Bowling CPRS ($CPRS_{\text{Bowl}}$):**
* **15%** Bayesian-Regressed Pressure Economy ($w\text{Econ}$)
* **15%** Phase Runs Saved Above Expected ($\text{RSAE}_{\text{ov}}$)
* **15%** Wickets Taken Under Pressure ($w\text{Wkts}$)
* **13%** Bayesian-Regressed Wicket Strike Rate ($w\text{SR}$)
* **9%** Clutch Delta Economy ($\Delta \text{Econ}$)
* **8%** Boundary Concession Suppression ($-p\text{Bdry\%}$)
* **7%** Dot-Ball Generation % ($p\text{Dot\%}$)
* **7%** Recency-Weighted Match Consistency Rate
* **6%** Volume Confidence Factor ($\log_{10} w\text{Balls}$)
* **5%** Clutch Delta Wicket Strike Rate ($\Delta \text{SR}$)

---

## 3. Fair Comparisons: Sample Sizes & Competition Context

1. **Minimum Sample Eligibility:** Players must have faced or bowled at least **150 high-pressure deliveries** across at least **10 distinct high-pressure matches** between 2022 and 2026 to qualify for ranking.
2. **IPL vs. T20I Caliber Normalization:** Because the dataset combines IPL fixtures with bilateral and ICC T20Is (which vary in opposition depth), we apply a **Competition Caliber Multiplier ($W_{\text{comp}}$)**:
   * **IPL Matches:** $1.25\times$ (reflecting sustained squad depth and high-scoring pressure).
   * **Top-Tier Full-Member T20Is:** $1.00\times$.
   * **Mixed / Associate T20Is:** $0.85\times$ / $0.70\times$.
3. **Contextual Pitch & Era Par:** Grouping $\text{Par RPO}$ by `(Competition, Season, Venue Country)` ensures a 165-run chase in low-scoring conditions is evaluated fairly alongside a 205-run chase on high-scoring surfaces, while accounting for T20 scoring inflation between 2022 and 2026.

---

## 4. Evidence Supporting the Top 10 Selections

### Batting Insights
* **Explosive Top-End Impact vs. Longitudinal Anchors:** **Vaibhav Sooryavanshi** (#1, CPRS `+2.912`) leads the cohort with an extraordinary `227.5` weighted pressure SR (`+97.8` phase RAE/100) and `61.4%` match consistency across 336 pressure balls (94 in knockouts). Immediately behind him are high-volume multi-season pillars **Virat Kohli** (#2, `960` pressure balls, `55.9` weighted average, `+17.8` $\Delta\text{SR}$) and **Sai Sudharsan** (#4, `746` pressure balls, `186` knockout balls, `52.3` weighted average).
* **Dot-Ball Avoidance Under Pressure:** **Shreyas Iyer** (#5, `26.2%` dots across `1,004` pressure balls), **Sai Sudharsan** (`26.5%`), **Virat Kohli** (`26.6%`), and **Phil Salt** (#8, `26.7%`) rank among the lowest dot-ball percentages in the dataset, demonstrating superior strike rotation when boundaries are restricted.
* **Clutch Elevation:** **Ryan Rickelton** (#7, `+27.0` $\Delta\text{SR}$), **Quinton de Kock** (#3, `+25.5` $\Delta\text{SR}$), and **Jonny Bairstow** (#9, `+21.0` $\Delta\text{SR}$) substantially outperform their overall career strike rates when match pressure triggers activate.

### Bowling Insights
* **Boundary Suppression & Control:** **Sunil Narine** (#1, CPRS `+1.287`) concedes boundaries on just `12.6%` of pressure deliveries (`7.40` weighted economy, `+1.27` $\text{RSAE}_{\text{ov}}$) while delivering a `66.4%` match consistency rate across `963` pressure balls. **Liam Dawson** (#4, `12.1%` boundary rate, `+0.96` $\Delta\text{Econ}$) and **Ripon Mondol** (#9, `12.7%` boundary rate, `10.5` weighted wicket SR) excel in tight containment.
* **High-Volume Strike & Death Execution:** **Rashid Khan** (#2, `1,578` pressure balls, `92` pressure wickets, `216` knockout balls), **Jasprit Bumrah** (#5, `1,180` pressure balls, `+1.47` $\text{RSAE}_{\text{ov}}$, `7.50` weighted economy), **Bhuvneshwar Kumar** (#6, `1,457` pressure balls, `73` wickets), and **Arshdeep Singh** (#7, `1,950` pressure balls, `115` pressure wickets) anchor the rankings through unmatched workload and wicket-taking volume in the toughest overs.

---

## 5. Reliability Checks & Model Limitations

### Match-to-Match Consistency Verification
To ensure rankings are not driven by one or two outlier performances, every player is evaluated on a **Recency-Weighted Match Consistency Rate**:
* **Batters:** Percentage of high-pressure appearances scoring $\ge 15$ runs at $\text{SR} \ge 135$. Top-10 batters sustain consistency rates between **47.2% and 62.1%**.
* **Bowlers:** Percentage of high-pressure appearances taking $\ge 2$ wickets, conceding $\le 7.0\text{ RPO}$, or taking $\ge 1$ wicket at $\le 8.5\text{ RPO}$. Top-10 bowlers achieve consistency rates up to **67.7%**.

### Known Limitations
1. **Venue Granularity:** Because exact stadium names are abstracted to `venue_country` in the match metadata, stadium-specific dimensions (e.g., short square boundaries vs. large outfields within the same country) are captured indirectly via `(Competition, Season, Venue Country)` median par run rates rather than ground-level geometry.
2. **Emerging Sample Sizes:** Players who debuted in 2025–2026 (such as Vaibhav Sooryavanshi and Priyansh Arya) satisfy the 150-ball minimum and receive recency weighting, though Empirical Bayes shrinkage ($K=200$) intentionally moderates their raw rates relative to 1,000-ball veterans.
3. **Fielding & Dropped Catches:** Delivery records capture credited wickets and runs conceded but do not isolate dropped catches or misfields, which can slightly penalize a bowler's economy or flatter a batter's survival rate in isolated matches.

---

## 6. Reproducibility

All rankings, intermediate features, and summary tables can be reproduced deterministically from the supplied competition CSV files using the accompanying pipeline:
```bash
python identify_pressure_players_clean.py
```
This outputs `top_10_batters_under_pressure.csv` and `top_10_bowlers_under_pressure.csv`, matching the exact ordered lists submitted on Unstop.