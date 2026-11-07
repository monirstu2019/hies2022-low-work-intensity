# Analytical extract — review before public release

This extract was prepared from the author-constructed and merged HIES 2022 person-level file (`women.dta`, 62,407 rows / 324 variables) using the primary study's `TTT(7).py` Steps 1.1, 1.2, 1.7 and 1.11.

- **Analytical records:** 2,110 working women aged 15+ selected by `sample_earnings_women_15plus == 1`.
- **Outcome:** `low_work_intensity = 1` when `earn_annual_hours <= 1440`. Exactly 553 records meet this threshold.
- **Predictor inputs:** Necessary source values for the manuscript's 35 candidate predictor groups, including 14 wealth PCA inputs. The fold-specific PCA, imputation, rare-category grouping and Boruta selection are NOT pre-fitted into the file. This is deliberate to prevent cross-validation data leakage.
- **Structural missingness:** recoding, missingness indicator, education grouping and PCA room-count preparation match the main workflow.
- **Survey frame:** 720 PSU–stratum units across 16 strata; the selected women span 589 PSU–stratum units.
- **Record and PSU IDs:** `record_id` is a new sequential number; `psu_public` is a newly assigned cluster code consistently mapped across the two released files. Source household IDs, person IDs, names, direct contact details and the source PSU codes are not included.

## Important limitations

1. This is *derived respondent-level microdata*, not aggregate statistics. **Written permission or a data-use agreement authorizing redistribution of derived extracts must be confirmed before posting publicly.** This package alone does NOT establish permission.
2. Removing direct IDs and relabeling clusters does not guarantee complete anonymization. Review rare combinations of indirect identifiers and any data protection obligations before release.
3. `TTT(7).py` and supporting scripts expect the full 62,407-row merged dataset, original row indices and original PSU codes for some analyses. **These two CSVs are not a drop-in replacement and do not by themselves enable exact end-to-end reproduction of every figure or point estimate.** Reviewer with authorised original HIES files can reconstruct the 62,407-row merge and run the full scripts. To run exclusively from this extract, scripts and CV manifests need adaptation, validated against output.
4. The original HIES modules are not contained here; their lawful retrieval is via BBS. The public README/Data Availability must distinguish original raw files, merged derivation, released analytical subset and what remains restricted.
