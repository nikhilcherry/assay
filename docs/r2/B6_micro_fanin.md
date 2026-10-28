# Block B6 Results Summary -- Recipient Burst (`FanIn`) & Slow-Drip Cohort Burst (`MicroCharge`)

## 1. Design Refinements & Honest Negative Findings

### A. `FanIn` (`undocumented:recipient_fan_in`)
1. **Why `device_profile` Bursts Were Dropped**: In Round 2, `device_profile` burst detection triggered on `HHF-014` and `HHF-016` because browser/OS version strings (e.g., `SM-G950U ... chrome 66.0`) naturally experience sudden "first-time card" spikes across the bank whenever a browser update rolls out. Device sharing belongs to the graph ring detector (`B15`); `FanIn` now evaluates **only** non-freemail `R_emaildomain` values (keeping ISP domains such as `sc.rr.com`, `comcast.net`, `att.net` eligible while excluding `gmail`, `yahoo`, `hotmail`, `outlook`, `aol`, `anonymous.com`, `icloud`, `live`, `msn`, `me.com`, `ymail`, and country variants).
2. **Episode Linkage Rule (72-Hour Window)**: An investigator asks: *"Is THIS flagged loss part of the recipient burst?"* A recipient burst only counts as fraud evidence when the flagged transaction—or another transaction on the same card within **72 hours** of the flagged transaction—pays the bursting `R_emaildomain`.
   - **HHF-006 (`C01727-K2`, flagged txn `3660263`)**: Pays `sc.rr.com` directly (`$312.68` on `2017-01-28`, plus `$296.55` on `01-17` and `$366.36` on `01-18`). Emitted as **primary fraud evidence** (`kind='evidence'`).
   - **HHF-013 (`C06115-K1`, flagged txn `3626720`)**: Flagged txn (`$409.30` on `2017-01-26`) is an in-person `ProductCD == 'W'` purchase with no recipient email; the card's `sc.rr.com` charges occurred outside the 72h episode. Emitted strictly as a **context note** (`kind='context'`, `lr=1.0`, `affected_txn_ids=[]`) naming the connected cards and amounts without inflating fraud odds.
3. **Historical Backtest & LR Assumption Disclosure**: Across the 4,675 confirmed fraud cases in Jul–Dec 2016 and 5,000 seeded Jul–Oct legitimate card-days, non-freemail `R_emaildomain` bursts ($k_{\text{now}} \ge 5$, $\ge 3\times$ baseline max, Poisson $p < 10^{-4}$) occur **0 times** in Jul–Dec closed fraud and **0 times** in Jul–Oct legitimate card-days (empirical Laplace LR `1.0693`, 95% CI `[0.0669, 17.0945]`). Because Jul–Dec history has too few recipient bursts to estimate a statistical LR, `FanIn` sets `lr = 3.0` **explicitly flagged as `ASSUMPTION`** in both the finding attestation and this document.

---

### B. `MicroCharge` (`undocumented:slow_drip_micro_charges`)
1. **Policy R5 vs. R9 Justification**: Fraud Operations Policy v1.0 **Rule R5 (`card_testing`)** defines card testing as *"Three or more small online authorizations on one card within an hour, followed by a larger purchase."* The slow drips in **HHF-004** (`C12975-K2`) and **HHF-005** (`C00348-K1`) consist of 6 online `ProductCD == 'C'` charges under `$6.50` spread across **21.2 days** ($\ge 2$ weeks) with no large follow-on purchase. Under **Rule R9 (`Undocumented patterns`)**, they are classified as `undocumented:slow_drip_micro_charges`.
2. **Honest Negative on Per-Card Slow Drips**: Evaluating the per-card slow-drip condition alone ($\ge 4$ sub-`$6.50` `ProductCD == 'C'` charges over $\ge 12\text{--}14$ days with zero prior) across Jul–Dec historical cases yields **LR = 0.63** (firing on `1.62%` of legitimate card-days vs. `1.03%` of fraud episodes). Single-card low-value recurring digital charges are normal customer behaviour in Jul–Dec.
3. **Cohort-Burst Redesign**: What is genuinely novel in January 2017 is the **coordinated cohort burst**: **14 to 17 distinct cards** simultaneously starting a novel slow drip ($\ge 4$ charges $< \$6.50$ on `ProductCD == 'C'`, span $\ge 14$ days, zero prior sub-`$6.50` `C` charges) within a 30-day window.
   - We measure $k_{\text{now}}$, the count of cards whose novel slow drip **starts** in the 30-day window `[opened_at - 30d, opened_at]`, against the causal 30-day window counts across Jul–Dec 2016 (`baseline_mean` $\approx 1.0\text{--}1.5$ cards per 30d window).
   - `MicroCharge` fires only when: (1) the flagged card qualifies and `flagged_txn_id` is one of its drip charges, (2) cohort size $k_{\text{now}} \ge 5$, and (3) Poisson tail $p < 10^{-4}$ against the Jul–Dec 30-day window baseline.
   - Across Jul–Dec 2016 closed fraud and 5,000 Jul–Oct legitimate card-days, the cohort-burst detector fires **0 times on legitimate card-days** (`0.0000%` false-positive rate) and **0 times on Jul–Dec fraud** (empirical Laplace LR `1.0693`, 95% CI `[0.0669, 17.0945]`). Because this coordinated slow-drip cohort campaign is new in January 2017, `MicroCharge` uses fallback `lr = 3.0` **explicitly flagged as `ASSUMPTION`**.

---

## 2. Exam Pack Summary (HHF-001 .. HHF-018)

| Exam Case | Detector Outcome | Details |
|---|---|---|
| **HHF-004** (`C12975-K2`) | **`MicroCharge` FIRES** (`lr=3.0` [ASSUMPTION]) | 6 charges $<\$6.50$ (`ProductCD='C'`) over `21.2d`, `0` prior; 30d cohort burst $k_{\text{now}} = 13\text{--}14$ cards vs. Jul–Dec baseline mean $\approx 1.0$, $p < 10^{-10}$. |
| **HHF-005** (`C00348-K1`) | **`MicroCharge` FIRES** (`lr=3.0` [ASSUMPTION]) | 6 charges $<\$6.50$ (`ProductCD='C'`) over `21.2d`, `0` prior; 30d cohort burst $k_{\text{now}} = 14$ cards vs. Jul–Dec baseline mean $\approx 1.0$, $p < 10^{-10}$. |
| **HHF-006** (`C01727-K2`) | **`FanIn` FIRES** (`kind='evidence'`, `lr=3.0` [ASSUMPTION]) | Flagged txn `3660263` (`$312.68`) pays `R_emaildomain=sc.rr.com`; $k_{\text{now}} = 17$ first-time cards in 21d vs. Jul–Dec baseline mean `0.75`, max `2`, $p = 3.0 \times 10^{-21}$. |
| **HHF-013** (`C06115-K1`) | **`FanIn` CONTEXT NOTE** (`kind='context'`, `lr=1.0`) | Card has prior `sc.rr.com` activity, but flagged txn `3626720` (`$409.30`, `ProductCD='W'`) is outside the 72h episode. Emits context note only (`affected_txn_ids=[]`). |
| **HHF-014, HHF-016** | **No match** | Previously triggered on `device_profile` browser-rollout bursts; eliminated by restricting `FanIn` to `R_emaildomain`. |
| **HHF-017, HHF-018** | **No match** | Isolated `ProductCD == 'W'` charges (`$43.92`, `$34.22`) using freemail domains (`gmail.com`, `yahoo.com`). Connect to nothing. |
