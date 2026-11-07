# Data access and redistribution

The raw Household Income and Expenditure Survey (HIES) 2022 survey modules belong to the Bangladesh Bureau of Statistics and are not shared here. Authors created a 62,407-record merged person-level data source using `code/stata/HIES2022_Data_Preparation_and_Merging.do`. The released analytical subset contains 2,110 records of working women, with 59 columns necessary to understand the model inputs, cleaning, survey weights and outcome; the full merged file is not shared.

**Before publishing this derived data**, obtain confirmation that redistribution of this extract is allowed under the authors' BBS microdata agreement or a separate authorization. Removal of names, household/person IDs and relabeling PSUs is only risk reduction, not a promise of full anonymization. If the agreement forbids redistribution, remove the `data/` folder and state the restriction to the journal.

BBS contact/access routes: https://nsds.bbs.gov.bd/en/microdata and https://eservice.bbs.gov.bd/.

The supplied full analysis scripts expect the 62,407-record merged data and preserved historical row indices for several steps. The 2,110-record extract and 720-PSU frame are useful for review but do not alone reproduce every original model estimate. Exact reproduction requires authorized input data and rerunning the documented pipeline. Editors may require a separate restricted-access arrangement to review raw and merged inputs.
