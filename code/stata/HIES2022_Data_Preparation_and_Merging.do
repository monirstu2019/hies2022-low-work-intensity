clear all
set more off
version 17

* Repository-ready paths. Set HIES_RAW_DIR and HIES_DERIVED_DIR in the
* Stata environment, or edit the fallback paths below before running.
local hies_root : environment HIES_RAW_DIR
if `"`hies_root'"' == "" local hies_root "C:/HIES2022/raw_modules"
global ROOT `"`hies_root'"'

local hies_out : environment HIES_DERIVED_DIR
if `"`hies_out'"' == "" local hies_out "C:/HIES2022/derived"
global OUT `"`hies_out'"'
global STAGE "$OUT/stage_stepwise"

cap mkdir "$OUT"
cap mkdir "$STAGE"

use "$ROOT/HH_SEC_1A1.dta", clear

describe PSU HHID S1AQ00 S1AQ01 S1AQ02 S1AQ03 S1AQ05 S1AQ07 S1AQ10 S1AQ12 S1AQ13 S1AQ14

count if missing(PSU) | missing(HHID) | missing(S1AQ00)

rename PSU psu
rename HHID hhid
rename S1AQ00 pid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" & "`v'" != "pid" {
        rename `v' r_`v'
    }
}

isid psu hhid pid

save "$STAGE/roster.dta", replace

use "$STAGE/roster.dta", clear

describe psu hhid pid r_S1AQ01 r_S1AQ02 r_S1AQ03 r_S1AQ05 r_S1AQ07 r_S1AQ10 r_S1AQ12 r_S1AQ13 r_S1AQ14

count








* ============================================================
* STEP 2: CREATE CLEAN LABOUR-FORCE SCREENING FILE ONLY
* Source: HH_SEC_1B.dta
* Key: PSU HHID S1BQ00
* Output: $STAGE/lfs.dta
* ============================================================

use "$ROOT/HH_SEC_1B.dta", clear

describe PSU HHID S1BQ00

count if missing(PSU) | missing(HHID) | missing(S1BQ00)

rename PSU psu
rename HHID hhid
rename S1BQ00 pid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" & "`v'" != "pid" {
        rename `v' lf_`v'
    }
}

isid psu hhid pid

save "$STAGE/lfs.dta", replace

use "$STAGE/lfs.dta", clear

describe psu hhid pid lf_*

count


















* ============================================================
* STEP 3: MERGE ROSTER + LABOUR-FORCE SCREENING ONLY
* Output: $STAGE/person_core_step3.dta
* ============================================================

use "$STAGE/roster.dta", clear

isid psu hhid pid

merge 1:1 psu hhid pid using "$STAGE/lfs.dta", gen(_m_lfs)

tab _m_lfs

count

save "$STAGE/person_core_step3.dta", replace

describe psu hhid pid r_S1AQ01 r_S1AQ03 lf_S1BQ01 lf_S1BQ05 lf_S1BQ06 _m_lfs

use "$STAGE/person_core_step3.dta", clear

tab _m_lfs
count












* ============================================================
* STEP 4: CREATE BASIC DEMOGRAPHIC + LFP VARIABLES
* Input:  $STAGE/person_core_step3.dta
* Output: $STAGE/person_core_step4_lfp.dta
* ============================================================

use "$STAGE/person_core_step3.dta", clear

* ----------------------------
* Demographic variables
* ----------------------------

gen female = .
replace female = 1 if r_S1AQ01 == 2
replace female = 0 if r_S1AQ01 == 1
label variable female "Female person: 1=female, 0=male"

gen age = r_S1AQ03
label variable age "Age in completed years"

gen relation_head = r_S1AQ02
label variable relation_head "Relationship to household head"

gen marital_status = r_S1AQ05
label variable marital_status "Marital status"

gen earner = .
replace earner = 1 if r_S1AQ07 == 1
replace earner = 0 if r_S1AQ07 == 2
label variable earner "Earner status"

gen has_mobile = .
replace has_mobile = 1 if r_S1AQ10 == 1
replace has_mobile = 0 if r_S1AQ10 == 2
label variable has_mobile "Has mobile phone"

gen has_bank_account = .
replace has_bank_account = 1 if r_S1AQ12 == 1
replace has_bank_account = 0 if r_S1AQ12 == 2
label variable has_bank_account "Has bank/deposit/current/loan account"

gen has_mfs_account = .
replace has_mfs_account = 1 if r_S1AQ13 == 1
replace has_mfs_account = 0 if r_S1AQ13 == 2
label variable has_mfs_account "Has mobile financial service account"

gen internet_used = .
replace internet_used = 1 if r_S1AQ14 == 1
replace internet_used = 0 if r_S1AQ14 == 2
label variable internet_used "Used internet in last 3 months"

* ----------------------------
* Age flags
* ----------------------------

gen age_5plus = age >= 5 if age < .
gen age_15plus = age >= 15 if age < .
gen age_15_64 = age >= 15 & age <= 64 if age < .

label variable age_5plus "Age 5 years and above"
label variable age_15plus "Age 15 years and above"
label variable age_15_64 "Age 15-64 years"

* ----------------------------
* Labour-force variables
* 1 = Yes, 2 = No
* Missing is preserved for children/not applicable.
* ----------------------------

gen worked_livelihood_7d = .
replace worked_livelihood_7d = 1 if lf_S1BQ01 == 1
replace worked_livelihood_7d = 0 if lf_S1BQ01 == 2
label variable worked_livelihood_7d "Worked for livelihood during past 7 days"

gen available_work_7d = .
replace available_work_7d = 1 if lf_S1BQ02 == 1
replace available_work_7d = 0 if lf_S1BQ02 == 2
label variable available_work_7d "Available for work during past 7 days"

gen looking_work_7d = .
replace looking_work_7d = 1 if lf_S1BQ03 == 1
replace looking_work_7d = 0 if lf_S1BQ03 == 2
label variable looking_work_7d "Looked for work during past 7 days"

gen ready_work_7d = .
replace ready_work_7d = 1 if lf_S1BQ04 == 1
replace ready_work_7d = 0 if lf_S1BQ04 == 2
label variable ready_work_7d "Ready to join work if opportunity in past 7 days"

gen looking_work_30d = .
replace looking_work_30d = 1 if lf_S1BQ05 == 1
replace looking_work_30d = 0 if lf_S1BQ05 == 2
label variable looking_work_30d "Looked for work during past 30 days"

gen ready_work_30d = .
replace ready_work_30d = 1 if lf_S1BQ06 == 1
replace ready_work_30d = 0 if lf_S1BQ06 == 2
label variable ready_work_30d "Ready to join work if opportunity in past 30 days"

gen reason_not_working_or_looking = lf_S1BQ07
label variable reason_not_working_or_looking "Reason not available / not looking for work"

gen reason_not_ready_to_work = lf_S1BQ09
label variable reason_not_ready_to_work "Reason not ready to work"

* ----------------------------
* Core LFP outcome
* ----------------------------
use "$STAGE/person_core_step4_lfp.dta", clear

* ============================================================
* STEP 4C: REVISED LFP DEFINITIONS
* ============================================================

* Broad LFP:
* participated if worked, available, or looked for work.
gen lfp_broad = .
replace lfp_broad = 1 if worked_livelihood_7d == 1
replace lfp_broad = 1 if available_work_7d == 1
replace lfp_broad = 1 if looking_work_7d == 1
replace lfp_broad = 1 if looking_work_30d == 1

replace lfp_broad = 0 if worked_livelihood_7d == 0 ///
    & available_work_7d == 0 ///
    & looking_work_7d == 0 ///
    & (looking_work_30d == 0 | missing(looking_work_30d))

label variable lfp_broad "Broad LFP: worked, available, or looked for work"


* Strict LFP:
* participated if worked OR searched and readiness/availability is observed.
gen search_ready_7d = .
replace search_ready_7d = 1 if looking_work_7d == 1 & ready_work_7d == 1
replace search_ready_7d = 0 if looking_work_7d == 0 | ready_work_7d == 0

gen search_ready_30d = .
replace search_ready_30d = 1 if looking_work_30d == 1 & ready_work_30d == 1
replace search_ready_30d = 0 if looking_work_30d == 0 | ready_work_30d == 0

gen lfp_strict = .
replace lfp_strict = 1 if worked_livelihood_7d == 1
replace lfp_strict = 1 if search_ready_7d == 1
replace lfp_strict = 1 if search_ready_30d == 1

replace lfp_strict = 0 if worked_livelihood_7d == 0 ///
    & search_ready_7d == 0 ///
    & search_ready_30d == 0

label variable lfp_strict "Strict LFP: worked or searched and ready"


* Compare old and new outcomes for age 15+
tab labour_force_participation if age_15plus == 1, missing
tab lfp_broad if age_15plus == 1, missing
tab lfp_strict if age_15plus == 1, missing

tab lfp_broad lfp_strict if age_15plus == 1, missing

save "$STAGE/person_core_step4C_lfp_revised.dta", replace






* ============================================================
* STEP 5 CORRECTED: CREATE CLEAN EDUCATION BACKGROUND FILE ONLY
* Source: HH_SEC_2A.dta
* Key: PSU HHID S2AQ00
* Output: $STAGE/eduA.dta
* ============================================================

use "$ROOT/HH_SEC_2A.dta", clear

describe PSU HHID S2AQ00 S2AQ01 S2AQ02 S2AQ03 S2AQ04 S2AQ09 S2AQ10 S2AQ12 S2AQ15 S2AQ16 S2AQ17

count if missing(PSU) | missing(HHID) | missing(S2AQ00)

* Save invalid education rows separately.
preserve
    keep if missing(PSU) | missing(HHID) | missing(S2AQ00)
    save "$OUT/bad_keys_eduA_raw.dta", replace
    export delimited using "$OUT/bad_keys_eduA_raw.csv", replace
restore

* Remove invalid-key rows from education merge file only.
drop if missing(PSU) | missing(HHID) | missing(S2AQ00)

rename PSU psu
rename HHID hhid
rename S2AQ00 pid

count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" & "`v'" != "pid" {
        rename `v' edA_`v'
    }
}

ds psu hhid pid, not
local edvars `r(varlist)'

collapse (firstnm) `edvars', by(psu hhid pid)

count if missing(psu) | missing(hhid) | missing(pid)

isid psu hhid pid

save "$STAGE/eduA.dta", replace

use "$STAGE/eduA.dta", clear

describe psu hhid pid edA_S2AQ01 edA_S2AQ02 edA_S2AQ03 edA_S2AQ04 edA_S2AQ09 edA_S2AQ10 edA_S2AQ12 edA_S2AQ15 edA_S2AQ16 edA_S2AQ17

count







* ============================================================
* STEP 6: MERGE EDUCATION BACKGROUND WITH PERSON CORE
* Input:  $STAGE/person_core_step4C_lfp_revised.dta
* Using:  $STAGE/eduA.dta
* Output: $STAGE/person_core_step6_eduA.dta
* ============================================================

use "$STAGE/person_core_step4C_lfp_revised.dta", clear

isid psu hhid pid

merge 1:1 psu hhid pid using "$STAGE/eduA.dta", gen(_m_eduA)

tab _m_eduA

count

save "$STAGE/person_core_step6_eduA.dta", replace

describe psu hhid pid female age lfp_broad edA_S2AQ01 edA_S2AQ02 edA_S2AQ03 edA_S2AQ04 _m_eduA






* ============================================================
* STEP 7: CREATE EDUCATION ANALYTIC VARIABLES
* Input:  $STAGE/person_core_step6_eduA.dta
* Output: $STAGE/person_core_step7_eduvars.dta
* ============================================================

use "$STAGE/person_core_step6_eduA.dta", clear

* Basic literacy
gen can_read_letter = .
replace can_read_letter = 1 if edA_S2AQ01 == 1
replace can_read_letter = 0 if edA_S2AQ01 == 2
label variable can_read_letter "Can read a letter"

gen can_write_letter = .
replace can_write_letter = 1 if edA_S2AQ02 == 1
replace can_write_letter = 0 if edA_S2AQ02 == 2
label variable can_write_letter "Can write a letter"

gen literate_basic = .
replace literate_basic = 1 if can_read_letter == 1 & can_write_letter == 1
replace literate_basic = 0 if can_read_letter == 0 | can_write_letter == 0
label variable literate_basic "Can both read and write"

* Education attendance and attainment
gen ever_attended_education = .
replace ever_attended_education = 1 if edA_S2AQ03 == 1
replace ever_attended_education = 0 if edA_S2AQ03 == 2
label variable ever_attended_education "Ever attended education"

gen highest_class_completed = edA_S2AQ04
label variable highest_class_completed "Highest class completed"

gen education_place = edA_S2AQ09
label variable education_place "Place of last/current study"

gen school_type_last = edA_S2AQ10
label variable school_type_last "Type of school/institution last/current attended"

gen curriculum = edA_S2AQ12
label variable curriculum "Curriculum"

gen father_education = edA_S2AQ15
label variable father_education "Father highest class completed"

gen mother_education = edA_S2AQ16
label variable mother_education "Mother highest class completed"

gen guardian_education = edA_S2AQ17
label variable guardian_education "Guardian highest class completed"

save "$STAGE/person_core_step7_eduvars.dta", replace

tab literate_basic if age_15plus == 1, missing
tab ever_attended_education if age_15plus == 1, missing
tab highest_class_completed if age_15plus == 1, missing
count






* ============================================================
* STEP 8: CREATE CLEAN CURRENT EDUCATION / ENROLMENT FILE ONLY
* Source: HH_SEC_2B.dta
* Key: PSU HHID S2BQ00
* Output: $STAGE/eduB.dta
* ============================================================

use "$ROOT/HH_SEC_2B.dta", clear

describe PSU HHID S2BQ00 S2BQ01 S2BQ02 S2BQ03 S2BQ04 S2BQ05 S2BQ06 S2BQ07

count if missing(PSU) | missing(HHID) | missing(S2BQ00)

* Save invalid rows separately if any
preserve
    keep if missing(PSU) | missing(HHID) | missing(S2BQ00)
    count
    if r(N) > 0 {
        save "$OUT/bad_keys_eduB_raw.dta", replace
        export delimited using "$OUT/bad_keys_eduB_raw.csv", replace
    }
restore

* Remove invalid-key rows from education-enrolment merge file only
drop if missing(PSU) | missing(HHID) | missing(S2BQ00)

rename PSU psu
rename HHID hhid
rename S2BQ00 pid

count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" & "`v'" != "pid" {
        rename `v' edB_`v'
    }
}

ds psu hhid pid, not
local edbvars `r(varlist)'

collapse (firstnm) `edbvars', by(psu hhid pid)

count if missing(psu) | missing(hhid) | missing(pid)

isid psu hhid pid

save "$STAGE/eduB.dta", replace

use "$STAGE/eduB.dta", clear

describe psu hhid pid edB_S2BQ01 edB_S2BQ02 edB_S2BQ03 edB_S2BQ04 edB_S2BQ05 edB_S2BQ06 edB_S2BQ07

count



use "$STAGE/eduB.dta", clear

count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid

isid psu hhid pid




* ============================================================
* STEP 9: MERGE CURRENT EDUCATION / ENROLMENT FILE
* Input:  $STAGE/person_core_step7_eduvars.dta
* Using:  $STAGE/eduB.dta
* Output: $STAGE/person_core_step9_eduB.dta
* ============================================================

use "$STAGE/person_core_step7_eduvars.dta", clear

isid psu hhid pid

merge 1:1 psu hhid pid using "$STAGE/eduB.dta", gen(_m_eduB)

tab _m_eduB

count

save "$STAGE/person_core_step9_eduB.dta", replace

describe psu hhid pid female age lfp_broad literate_basic edB_S2BQ01 edB_S2BQ03 edB_S2BQ04 _m_eduB



* ============================================================
* STEP 10: CREATE CURRENT EDUCATION / ENROLMENT VARIABLES
* Input:  $STAGE/person_core_step9_eduB.dta
* Output: $STAGE/person_core_step10_edu_current.dta
* ============================================================

use "$STAGE/person_core_step9_eduB.dta", clear

gen currently_attending_school = .
replace currently_attending_school = 1 if edB_S2BQ01 == 1
replace currently_attending_school = 0 if edB_S2BQ01 == 2
label variable currently_attending_school "Currently attending school/educational institution"

gen reason_not_attending_school = edB_S2BQ02
label variable reason_not_attending_school "Reason not currently attending school"

gen current_class = edB_S2BQ03
label variable current_class "Current class attending"

gen receives_scholarship_stipend = .
replace receives_scholarship_stipend = 1 if inlist(edB_S2BQ04, 1, 2, 3)
replace receives_scholarship_stipend = 0 if edB_S2BQ04 == 4
label variable receives_scholarship_stipend "Receives scholarship/stipend"

gen stipend_type = edB_S2BQ05
label variable stipend_type "Type of scholarship/stipend"

gen stipend_amount_12m = edB_S2BQ06
label variable stipend_amount_12m "Scholarship/stipend amount received in last 12 months"

gen free_study_opportunity = .
replace free_study_opportunity = 1 if edB_S2BQ07 == 1
replace free_study_opportunity = 0 if edB_S2BQ07 == 2
label variable free_study_opportunity "Opportunity to study without pay"

save "$STAGE/person_core_step10_edu_current.dta", replace

tab currently_attending_school if age_15plus == 1, missing
tab current_class if age_15plus == 1 & currently_attending_school == 1, missing
tab receives_scholarship_stipend if currently_attending_school == 1, missing
summarize stipend_amount_12m if currently_attending_school == 1
count




* ============================================================
* STEP 11: CREATE CLEAN FUNCTIONAL DIFFICULTY FILE ONLY
* Source: HH_SEC_1A2.dta
* Key: PSU HHID S1AQ0C
* Output: $STAGE/disability.dta
* ============================================================

use "$ROOT/HH_SEC_1A2.dta", clear

describe PSU HHID S1AQ0C S1AQ16 S1AQ17 S1AQ18 S1AQ19 S1AQ20 S1AQ21

count if missing(PSU) | missing(HHID) | missing(S1AQ0C)

preserve
    keep if missing(PSU) | missing(HHID) | missing(S1AQ0C)
    count
    if r(N) > 0 {
        save "$OUT/bad_keys_disability_raw.dta", replace
        export delimited using "$OUT/bad_keys_disability_raw.csv", replace
    }
restore

drop if missing(PSU) | missing(HHID) | missing(S1AQ0C)

rename PSU psu
rename HHID hhid
rename S1AQ0C pid

count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" & "`v'" != "pid" {
        rename `v' fd_`v'
    }
}

ds psu hhid pid, not
local fdvars `r(varlist)'

collapse (firstnm) `fdvars', by(psu hhid pid)

count if missing(psu) | missing(hhid) | missing(pid)

isid psu hhid pid

save "$STAGE/disability.dta", replace

use "$STAGE/disability.dta", clear

describe psu hhid pid fd_S1AQ16 fd_S1AQ17 fd_S1AQ18 fd_S1AQ19 fd_S1AQ20 fd_S1AQ21

count







* ============================================================
* STEP 12: MERGE FUNCTIONAL DIFFICULTY AND CREATE DISABILITY VARIABLES
* Input:  $STAGE/person_core_step10_edu_current.dta
* Using:  $STAGE/disability.dta
* Output: $STAGE/person_core_step12_disability.dta
* ============================================================

use "$STAGE/person_core_step10_edu_current.dta", clear

isid psu hhid pid

merge 1:1 psu hhid pid using "$STAGE/disability.dta", gen(_m_disability)

tab _m_disability

* Functional difficulty coding:
* 1 = no difficulty
* 2 = some difficulty
* 3 = severe difficulty
* 4 = cannot do at all

gen any_functional_difficulty = 0
replace any_functional_difficulty = 1 if inlist(fd_S1AQ16, 2, 3, 4)
replace any_functional_difficulty = 1 if inlist(fd_S1AQ17, 2, 3, 4)
replace any_functional_difficulty = 1 if inlist(fd_S1AQ18, 2, 3, 4)
replace any_functional_difficulty = 1 if inlist(fd_S1AQ19, 2, 3, 4)
replace any_functional_difficulty = 1 if inlist(fd_S1AQ20, 2, 3, 4)
replace any_functional_difficulty = 1 if inlist(fd_S1AQ21, 2, 3, 4)
label variable any_functional_difficulty "Any functional difficulty"

gen severe_functional_difficulty = 0
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ16, 3, 4)
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ17, 3, 4)
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ18, 3, 4)
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ19, 3, 4)
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ20, 3, 4)
replace severe_functional_difficulty = 1 if inlist(fd_S1AQ21, 3, 4)
label variable severe_functional_difficulty "Severe or complete functional difficulty"

save "$STAGE/person_core_step12_disability.dta", replace

tab _m_disability
tab any_functional_difficulty if age_15plus == 1, missing
tab severe_functional_difficulty if age_15plus == 1, missing
count







* ============================================================
* STEP 13: CREATE CLEAN HOUSEHOLD GEO + WEIGHT FILES ONLY
* Sources:
*   HH_SEC_1A.dta
*   weight_final_2022.dta
* Outputs:
*   $STAGE/hh_geo.dta
*   $STAGE/weight.dta
* ============================================================

* ----------------------------
* 13A. Household geography / ID file
* ----------------------------

use "$ROOT/HH_SEC_1A.dta", clear

describe PSU HHID

count if missing(PSU) | missing(HHID)

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

duplicates report psu hhid
isid psu hhid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" {
        rename `v' hh_`v'
    }
}

save "$STAGE/hh_geo.dta", replace

use "$STAGE/hh_geo.dta", clear
describe psu hhid hh_*
count


* ----------------------------
* 13B. Weight file
* ----------------------------

use "$ROOT/weight_final_2022.dta", clear

describe PSU HHID

count if missing(PSU) | missing(HHID)

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

duplicates report psu hhid
isid psu hhid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" {
        rename `v' wt_`v'
    }
}

save "$STAGE/weight.dta", replace

use "$STAGE/weight.dta", clear
describe psu hhid wt_*
count






* ============================================================
* STEP 14: MERGE HOUSEHOLD GEO + WEIGHT INTO PERSON CORE
* Input:  $STAGE/person_core_step12_disability.dta
* Using:  $STAGE/hh_geo.dta and $STAGE/weight.dta
* Output: $STAGE/person_core_step14_geo_weight.dta
* ============================================================

use "$STAGE/person_core_step12_disability.dta", clear

isid psu hhid pid

merge m:1 psu hhid using "$STAGE/hh_geo.dta", gen(_m_hh_geo)

tab _m_hh_geo

merge m:1 psu hhid using "$STAGE/weight.dta", gen(_m_weight)

tab _m_weight

* Create clean geography and survey-design variables
gen division_code = hh_ID_01_CODE
gen district_code = hh_ID_02_CODE
gen rmo_code = hh_ID_09_CODE

gen rural_urban = wt_urbrural
gen stratum16 = wt_stratum16
gen domain16 = wt_domain16
gen household_weight = wt_hh_wgt
gen population_weight = wt_pop_wgt

label variable division_code "Division code"
label variable district_code "District code"
label variable rmo_code "RMO / Rural-urban code from HH_SEC_1A"
label variable rural_urban "Rural/urban from weight file"
label variable stratum16 "Survey stratum"
label variable domain16 "Survey domain"
label variable household_weight "Household survey weight"
label variable population_weight "Population survey weight"

save "$STAGE/person_core_step14_geo_weight.dta", replace

tab _m_hh_geo
tab _m_weight
tab rural_urban, missing
summarize household_weight population_weight
count








* ============================================================
* STEP 15: SAVE FINAL PERSON-CORE FILE FOR LFP MODEL
* Input:  $STAGE/person_core_step14_geo_weight.dta
* Output: $OUT/person_core_lfp_model.dta / .csv
* ============================================================

use "$STAGE/person_core_step14_geo_weight.dta", clear

isid psu hhid pid

order psu hhid pid ///
    female age age_5plus age_15plus age_15_64 ///
    division_code district_code rmo_code rural_urban stratum16 domain16 ///
    household_weight population_weight ///
    relation_head marital_status earner has_mobile has_bank_account ///
    has_mfs_account internet_used ///
    worked_livelihood_7d available_work_7d looking_work_7d ready_work_7d ///
    looking_work_30d ready_work_30d ///
    lfp_broad lfp_strict labour_force_participation ///
    literate_basic can_read_letter can_write_letter ///
    ever_attended_education highest_class_completed ///
    currently_attending_school current_class receives_scholarship_stipend ///
    any_functional_difficulty severe_functional_difficulty ///
    _m_lfs _m_eduA _m_eduB _m_disability _m_hh_geo _m_weight

compress

save "$OUT/person_core_lfp_model.dta", replace
export delimited using "$OUT/person_core_lfp_model.csv", replace

* Final checks
count
count if missing(psu) | missing(hhid) | missing(pid)

tab female, missing
tab age_15plus, missing
tab lfp_broad if age_15plus == 1, missing
tab lfp_broad if age_15plus == 1 & female == 1, missing
tab lfp_broad if age_15plus == 1 & female == 0, missing

summarize age household_weight population_weight






* ============================================================
* STEP 16: CREATE CLEAN SECTION 4A ACTIVITY / EARNINGS FILE
* Source: HH_SEC_4A.dta
* Key: PSU HHID s4aq00
* Activity ID: s4aq0a
* Output: $OUT/activity_4A_earnings_clean.dta / .csv
* ============================================================

use "$ROOT/HH_SEC_4A.dta", clear

describe PSU HHID s4aq00 s4aq0a s4aq01 s4aq01a s4aq01b s4aq02 s4aq03 s4aq04 ///
    s4aq06 s4aq07 s4aq08 s4bq10c s4bq13b s4bq14 s4bq15 s4bq16 s4bq17

count if missing(PSU) | missing(HHID) | missing(s4aq00)

* Save invalid activity rows separately.
preserve
    keep if missing(PSU) | missing(HHID) | missing(s4aq00)
    count
    if r(N) > 0 {
        save "$OUT/bad_keys_activity4A_raw.dta", replace
        export delimited using "$OUT/bad_keys_activity4A_raw.csv", replace
        list PSU HHID s4aq00 s4aq0a, abbrev(24)
    }
restore

* Remove invalid-key rows from activity/earnings construction only.
drop if missing(PSU) | missing(HHID) | missing(s4aq00)

rename PSU psu
rename HHID hhid
rename s4aq00 pid
rename s4aq0a actid

count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid actid

* Work-time variables
gen work_months_12m = s4aq02
gen work_days_month = s4aq03
gen work_hours_day  = s4aq04

gen annual_hours = work_months_12m * work_days_month * work_hours_day ///
    if work_months_12m < . & work_days_month < . & work_hours_day < .

* Earnings variables
gen daily_wage_avg = s4bq10c

gen cash_daily_annual = daily_wage_avg * work_days_month * work_months_12m ///
    if daily_wage_avg < . & work_days_month < . & work_months_12m < .

gen inkind_daily_annual = s4bq13b * work_days_month * work_months_12m ///
    if s4bq13b < . & work_days_month < . & work_months_12m < .

gen gross_monthly_annual = s4bq15 * work_months_12m ///
    if s4bq15 < . & work_months_12m < .

gen net_monthly_annual = s4bq16 * work_months_12m ///
    if s4bq16 < . & work_months_12m < .

gen inkind_other_annual = s4bq17

egen activity_earnings_nonmiss = rownonmiss(cash_daily_annual inkind_daily_annual ///
    gross_monthly_annual net_monthly_annual inkind_other_annual)

egen activity_annual_earnings = rowtotal(cash_daily_annual inkind_daily_annual ///
    gross_monthly_annual net_monthly_annual inkind_other_annual), missing

replace activity_annual_earnings = . if activity_earnings_nonmiss == 0

gen activity_hourly_earnings = activity_annual_earnings / annual_hours ///
    if activity_annual_earnings > 0 & annual_hours > 0

gen log_activity_hourly_earnings = ln(activity_hourly_earnings) ///
    if activity_hourly_earnings > 0

label variable work_months_12m "Months worked in last 12 months"
label variable work_days_month "Days worked per month"
label variable work_hours_day "Hours worked per day"
label variable annual_hours "Constructed annual work hours"
label variable cash_daily_annual "Constructed annual cash daily wage earnings"
label variable gross_monthly_annual "Constructed annual gross monthly salary"
label variable net_monthly_annual "Constructed annual net monthly salary"
label variable inkind_daily_annual "Constructed annual daily in-kind benefit"
label variable inkind_other_annual "Other annual in-kind/benefit"
label variable activity_annual_earnings "Constructed annual activity earnings"
label variable activity_hourly_earnings "Constructed hourly activity earnings"

compress

save "$OUT/activity_4A_earnings_clean.dta", replace
export delimited using "$OUT/activity_4A_earnings_clean.csv", replace

count
count if missing(psu) | missing(hhid) | missing(pid)

summarize work_months_12m work_days_month work_hours_day annual_hours
summarize activity_annual_earnings activity_hourly_earnings

tab s4aq06, missing
tab s4aq07, missing
tab s4aq08, missing





* ============================================================
* STEP 17: COLLAPSE SECTION 4A ACTIVITY EARNINGS TO PERSON LEVEL
* Input:  $OUT/activity_4A_earnings_clean.dta
* Output: $OUT/person_earnings_aggregated.dta / .csv
* ============================================================

use "$OUT/activity_4A_earnings_clean.dta", clear

isid psu hhid pid actid

count if missing(psu) | missing(hhid) | missing(pid)

* Numeric counters for safe collapse
gen byte activity_row = 1
gen byte activity_has_earnings = activity_annual_earnings < .
gen byte activity_has_hours = annual_hours < .

collapse ///
    (sum) earn_annual_total = activity_annual_earnings ///
          earn_annual_hours = annual_hours ///
          earn_cash_daily_annual = cash_daily_annual ///
          earn_inkind_daily_annual = inkind_daily_annual ///
          earn_gross_monthly_annual = gross_monthly_annual ///
          earn_net_monthly_annual = net_monthly_annual ///
          earn_inkind_other_annual = inkind_other_annual ///
          earn_valid_activity_count = activity_has_earnings ///
          earn_valid_hours_count = activity_has_hours ///
          earn_activity_count = activity_row ///
    (firstnm) earn_occ_code = s4aq01a ///
              earn_ind_code = s4aq01b ///
              earn_sector = s4aq06 ///
              earn_emp_status_ag = s4aq07 ///
              earn_emp_status_nonag = s4aq08 ///
              earn_org_type = s4bq14, ///
    by(psu hhid pid)

* Preserve missingness:
* If a person had no valid earnings activity, annual earnings should be missing, not zero.
replace earn_annual_total = . if earn_valid_activity_count == 0
replace earn_cash_daily_annual = . if earn_valid_activity_count == 0
replace earn_inkind_daily_annual = . if earn_valid_activity_count == 0
replace earn_gross_monthly_annual = . if earn_valid_activity_count == 0
replace earn_net_monthly_annual = . if earn_valid_activity_count == 0
replace earn_inkind_other_annual = . if earn_valid_activity_count == 0

replace earn_annual_hours = . if earn_valid_hours_count == 0

gen earn_hourly = earn_annual_total / earn_annual_hours ///
    if earn_annual_total > 0 & earn_annual_hours > 0

gen log_earn_hourly = ln(earn_hourly) if earn_hourly > 0

label variable earn_activity_count "Number of activity rows before person-level collapse"
label variable earn_valid_activity_count "Number of activities with non-missing earnings"
label variable earn_valid_hours_count "Number of activities with non-missing hours"
label variable earn_annual_total "Total annual employment earnings, person-level"
label variable earn_annual_hours "Total annual work hours, person-level"
label variable earn_hourly "Hourly earnings, person-level"
label variable log_earn_hourly "Log hourly earnings, person-level"
label variable earn_occ_code "Main occupation code"
label variable earn_ind_code "Main industry code"
label variable earn_sector "Main economic sector"
label variable earn_emp_status_ag "Employment status, agriculture"
label variable earn_emp_status_nonag "Employment status, non-agriculture"
label variable earn_org_type "Type of organization"

isid psu hhid pid

compress

save "$OUT/person_earnings_aggregated.dta", replace
export delimited using "$OUT/person_earnings_aggregated.csv", replace

count
count if missing(psu) | missing(hhid) | missing(pid)

duplicates report psu hhid pid

summarize earn_activity_count earn_valid_activity_count earn_annual_total earn_annual_hours earn_hourly

tab earn_sector, missing
tab earn_emp_status_ag, missing
tab earn_emp_status_nonag, missing




* ============================================================
* STEP 18: MERGE PERSON-LEVEL EARNINGS INTO PERSON CORE
* Input:  $OUT/person_core_lfp_model.dta
* Using:  $OUT/person_earnings_aggregated.dta
* Output: $OUT/person_core_lfp_earnings_model.dta / .csv
* ============================================================

use "$OUT/person_core_lfp_model.dta", clear

isid psu hhid pid

merge 1:1 psu hhid pid using "$OUT/person_earnings_aggregated.dta", gen(_m_earnings)

tab _m_earnings

count
count if missing(psu) | missing(hhid) | missing(pid)

summarize earn_activity_count earn_valid_activity_count earn_annual_total earn_annual_hours earn_hourly

tab earn_sector, missing
tab earn_emp_status_ag, missing
tab earn_emp_status_nonag, missing

save "$OUT/person_core_lfp_earnings_model.dta", replace
export delimited using "$OUT/person_core_lfp_earnings_model.csv", replace







* ============================================================
* STEP 19: CREATE FINAL ANALYSIS SAMPLE FLAGS
* Input:  $OUT/person_core_lfp_earnings_model.dta
* Output: $OUT/final_hies_lfp_earnings_analysis.dta / .csv
* ============================================================

use "$OUT/person_core_lfp_earnings_model.dta", clear

* Main LFP ML sample: women age 15+ with observed broad LFP
gen sample_lfp_women_15plus = .
replace sample_lfp_women_15plus = 1 if female == 1 & age_15plus == 1 & !missing(lfp_broad)
replace sample_lfp_women_15plus = 0 if !(female == 1 & age_15plus == 1 & !missing(lfp_broad))
label variable sample_lfp_women_15plus "Sample: women age 15+ with observed broad LFP"

* Full gender-comparison LFP sample
gen sample_lfp_all_15plus = .
replace sample_lfp_all_15plus = 1 if age_15plus == 1 & !missing(lfp_broad)
replace sample_lfp_all_15plus = 0 if !(age_15plus == 1 & !missing(lfp_broad))
label variable sample_lfp_all_15plus "Sample: all age 15+ with observed broad LFP"

* Earnings sample: age 15+ workers with positive hourly earnings
gen sample_earnings_15plus = .
replace sample_earnings_15plus = 1 if age_15plus == 1 & earn_hourly > 0 & !missing(earn_hourly)
replace sample_earnings_15plus = 0 if !(age_15plus == 1 & earn_hourly > 0 & !missing(earn_hourly))
label variable sample_earnings_15plus "Sample: age 15+ with positive hourly earnings"

* Women earnings sample
gen sample_earnings_women_15plus = .
replace sample_earnings_women_15plus = 1 if female == 1 & age_15plus == 1 & earn_hourly > 0 & !missing(earn_hourly)
replace sample_earnings_women_15plus = 0 if !(female == 1 & age_15plus == 1 & earn_hourly > 0 & !missing(earn_hourly))
label variable sample_earnings_women_15plus "Sample: women age 15+ with positive hourly earnings"

* Useful log earnings variable is already available: log_earn_hourly

compress

save "$OUT/final_hies_lfp_earnings_analysis.dta", replace
export delimited using "$OUT/final_hies_lfp_earnings_analysis.csv", replace

tab sample_lfp_women_15plus, missing
tab sample_lfp_all_15plus, missing
tab sample_earnings_15plus, missing
tab sample_earnings_women_15plus, missing

summarize lfp_broad earn_annual_total earn_hourly log_earn_hourly if sample_earnings_15plus == 1






* ============================================================
* STEP 20: CREATE HOUSEHOLD FOOD INSECURITY / FIES FILE ONLY
* Source: HH_SEC_10.dta
* Key: PSU HHID
* Output: $STAGE/fies.dta
* ============================================================

use "$ROOT/HH_SEC_10.dta", clear

describe PSU HHID Q10_1 Q10_2 Q10_3 Q10_4 Q10_5 Q10_6 Q10_7 Q10_8

count if missing(PSU) | missing(HHID)

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

duplicates report psu hhid
isid psu hhid

* Convert FIES items:
* 1 = Yes -> 1
* 2 = No  -> 0
* 8/9/missing remain missing for the yes/no score

local fies_yes_vars ""

foreach i of numlist 1/8 {
    gen fies_Q10_`i'_yes = .
    replace fies_Q10_`i'_yes = 1 if Q10_`i' == 1
    replace fies_Q10_`i'_yes = 0 if Q10_`i' == 2

    gen fies_Q10_`i'_dk_refused = .
    replace fies_Q10_`i'_dk_refused = 1 if inlist(Q10_`i', 8, 9)
    replace fies_Q10_`i'_dk_refused = 0 if inlist(Q10_`i', 1, 2)

    local fies_yes_vars "`fies_yes_vars' fies_Q10_`i'_yes"
}

egen fies_valid_items = rownonmiss(`fies_yes_vars')
egen fies_raw_score = rowtotal(`fies_yes_vars')
replace fies_raw_score = . if fies_valid_items < 8

label variable fies_valid_items "Number of valid FIES items"
label variable fies_raw_score "Raw FIES score, 0-8, valid only when all 8 items observed"

keep psu hhid Q10_1 Q10_2 Q10_3 Q10_4 Q10_5 Q10_6 Q10_7 Q10_8 ///
    fies_Q10_* fies_valid_items fies_raw_score

save "$STAGE/fies.dta", replace

count
count if missing(psu) | missing(hhid)
duplicates report psu hhid
summarize fies_raw_score fies_valid_items
tab fies_raw_score, missing






* ============================================================
* STEP 21: MERGE HOUSEHOLD FIES INTO FINAL PERSON ANALYSIS FILE
* Input:  $OUT/final_hies_lfp_earnings_analysis.dta
* Using:  $STAGE/fies.dta
* Output: $OUT/final_hies_lfp_earnings_fies.dta / .csv
* ============================================================

use "$OUT/final_hies_lfp_earnings_analysis.dta", clear

isid psu hhid pid

merge m:1 psu hhid using "$STAGE/fies.dta", gen(_m_fies)

tab _m_fies

count
count if missing(psu) | missing(hhid) | missing(pid)

summarize fies_raw_score fies_valid_items

tab fies_raw_score if sample_lfp_women_15plus == 1, missing

save "$OUT/final_hies_lfp_earnings_fies.dta", replace
export delimited using "$OUT/final_hies_lfp_earnings_fies.csv", replace




* ============================================================
* STEP 22: CREATE CLEAN HOUSEHOLD HOUSING / LIVING FILE ONLY
* Source: HH_SEC_6A.dta
* Key: PSU HHID
* Output: $STAGE/housing_6A.dta
* ============================================================

use "$ROOT/HH_SEC_6A.dta", clear

describe PSU HHID

count if missing(PSU) | missing(HHID)

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

count if missing(psu) | missing(hhid)

duplicates report psu hhid

isid psu hhid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" {
        rename `v' h6A_`v'
    }
}

save "$STAGE/housing_6A.dta", replace

use "$STAGE/housing_6A.dta", clear

count
count if missing(psu) | missing(hhid)
duplicates report psu hhid
describe psu hhid h6A_*







* ============================================================
* STEP 23: MERGE HOUSING/LIVING CONDITIONS INTO FINAL FILE
* Input:  $OUT/final_hies_lfp_earnings_fies.dta
* Using:  $STAGE/housing_6A.dta
* Output: $OUT/final_hies_lfp_earnings_fies_housing.dta / .csv
* ============================================================

use "$OUT/final_hies_lfp_earnings_fies.dta", clear

isid psu hhid pid

merge m:1 psu hhid using "$STAGE/housing_6A.dta", gen(_m_housing)

tab _m_housing

* ----------------------------
* Clean housing/living controls
* ----------------------------

gen hh_rooms = h6A_S6AQ02
label variable hh_rooms "Number of rooms occupied by household"

gen wall_material = h6A_S6AQ04
label variable wall_material "Main wall material"

gen roof_material = h6A_S6AQ05
label variable roof_material "Main roof material"

gen covered_area = h6A_S6AQ06
label variable covered_area "Covered room area"

gen toilet_type = h6A_S6AQ07
label variable toilet_type "Main toilet facility type"

gen shared_toilet = .
replace shared_toilet = 1 if h6A_S6AQ08 == 1
replace shared_toilet = 0 if h6A_S6AQ08 == 2
label variable shared_toilet "Household shares toilet facility"

gen drinking_water_source = h6A_S6AQ09
label variable drinking_water_source "Main drinking water source"

gen other_water_source = h6A_S6AQ13
label variable other_water_source "Main water source for other use"

gen water_collection_time_min = h6A_S6AQ14
label variable water_collection_time_min "Time to collect water and return, minutes"

gen handwashing_place = h6A_S6AQ15
label variable handwashing_place "Usual handwashing place"

gen has_handwashing_material = .
replace has_handwashing_material = 1 if h6A_S6AQ16 == 1
replace has_handwashing_material = 0 if h6A_S6AQ16 == 2
label variable has_handwashing_material "Has soap/detergent/ash/soil/sand for handwashing"

gen light_source = h6A_S6AQ18
label variable light_source "Main light source in house"

gen electricity_hours_24h = h6A_S6AQ19
label variable electricity_hours_24h "Electricity hours in normal 24 hours"

gen cooking_stove_type = h6A_S6AQ20
label variable cooking_stove_type "Main cooking stove type"

gen cooking_fuel = h6A_S6AQ23
label variable cooking_fuel "Main cooking fuel or energy"

gen cooking_location = h6A_S6AQ24
label variable cooking_location "Cooking location"

gen has_landline = .
replace has_landline = 1 if h6A_S6AQ26 == 1
replace has_landline = 0 if h6A_S6AQ26 == 2
label variable has_landline "Household has landline telephone"

gen has_computer = .
replace has_computer = 1 if h6A_S6AQ27 == 1
replace has_computer = 0 if h6A_S6AQ27 == 2
label variable has_computer "Household owns computer/laptop/tablet"

gen house_use = h6A_S6AQ28
label variable house_use "Use of house"

gen house_value = h6A_S6AQ29
label variable house_value "Estimated cost to buy/build current house"

gen slum_household = .
replace slum_household = 1 if h6A_S6AQ30 == 1
replace slum_household = 0 if h6A_S6AQ30 == 2
label variable slum_household "Slum household"

gen all_season_road_2km = .
replace all_season_road_2km = 1 if h6A_S6AQ31 == 1
replace all_season_road_2km = 0 if h6A_S6AQ31 == 2
label variable all_season_road_2km "All-season usable road/dock within 2 km"

gen transport_stop_halfkm = .
replace transport_stop_halfkm = 1 if h6A_S6AQ32 == 1
replace transport_stop_halfkm = 0 if h6A_S6AQ32 == 2
label variable transport_stop_halfkm "Transport stop within half kilometer"

gen transport_stop_time_min = h6A_S6AQ33
label variable transport_stop_time_min "Minutes to nearest transport stop"

gen regular_waste_collection = .
replace regular_waste_collection = 1 if h6A_S6AQ34 == 1
replace regular_waste_collection = 0 if h6A_S6AQ34 == 2
label variable regular_waste_collection "Regular waste collection"

save "$OUT/final_hies_lfp_earnings_fies_housing.dta", replace
export delimited using "$OUT/final_hies_lfp_earnings_fies_housing.csv", replace

tab _m_housing
count
count if missing(psu) | missing(hhid) | missing(pid)

summarize hh_rooms covered_area electricity_hours_24h house_value
tab slum_household, missing
tab has_computer, missing
tab light_source, missing
tab cooking_fuel, missing





* ============================================================
* STEP 24: CREATE CLEAN HOUSEHOLD ASSET FILE ONLY
* Source: HH_SEC_8A.dta
* Key: PSU HHID
* Output: $STAGE/assets_8A.dta
* ============================================================

use "$ROOT/HH_SEC_8A.dta", clear

describe PSU HHID

count if missing(PSU) | missing(HHID)

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

count if missing(psu) | missing(hhid)

duplicates report psu hhid

isid psu hhid

foreach v of varlist _all {
    if "`v'" != "psu" & "`v'" != "hhid" {
        rename `v' asset_`v'
    }
}

save "$STAGE/assets_8A.dta", replace

use "$STAGE/assets_8A.dta", clear

count
count if missing(psu) | missing(hhid)
duplicates report psu hhid
describe psu hhid asset_*



* ============================================================
* STEP 25: MERGE HOUSEHOLD ASSETS INTO FINAL FILE
* Input:  $OUT/final_hies_lfp_earnings_fies_housing.dta
* Using:  $STAGE/assets_8A.dta
* Output: $OUT/final_hies_lfp_earnings_fies_housing_assets.dta / .csv
* ============================================================

use "$OUT/final_hies_lfp_earnings_fies_housing.dta", clear

isid psu hhid pid

merge m:1 psu hhid using "$STAGE/assets_8A.dta", gen(_m_assets)

tab _m_assets

* Clean asset controls
gen owns_land_property_nonoperated = .
replace owns_land_property_nonoperated = 1 if asset_S8AQ01A == 1
replace owns_land_property_nonoperated = 0 if asset_S8AQ01A == 2
label variable owns_land_property_nonoperated "Owns land/property not operated"

gen land_property_value = asset_S8AQ02
label variable land_property_value "Estimated value of owned land/property"

gen purchased_land_12m = .
replace purchased_land_12m = 1 if asset_S8AQ03 == 1
replace purchased_land_12m = 0 if asset_S8AQ03 == 2
label variable purchased_land_12m "Purchased land/property in last 12 months"

gen land_purchase_value_12m = asset_S8AQ04
label variable land_purchase_value_12m "Land/property purchase value in last 12 months"

gen purchased_house_flat_12m = .
replace purchased_house_flat_12m = 1 if asset_S8AQ05 == 1
replace purchased_house_flat_12m = 0 if asset_S8AQ05 == 2
label variable purchased_house_flat_12m "Purchased house/flat in last 12 months"

gen house_flat_purchase_value_12m = asset_S8AQ06
label variable house_flat_purchase_value_12m "House/flat purchase value in last 12 months"

gen owns_other_assets = .
replace owns_other_assets = 1 if asset_S8AQ07 == 1
replace owns_other_assets = 0 if asset_S8AQ07 == 2
label variable owns_other_assets "Owns other financial/physical assets"

gen other_asset_value = asset_S8AQ08
label variable other_asset_value "Value of other assets"

gen purchased_other_assets_12m = .
replace purchased_other_assets_12m = 1 if asset_S8AQ09 == 1
replace purchased_other_assets_12m = 0 if asset_S8AQ09 == 2
label variable purchased_other_assets_12m "Purchased other assets in last 12 months"

gen other_asset_purchase_value_12m = asset_S8AQ10
label variable other_asset_purchase_value_12m "Other asset purchase value in last 12 months"

gen sold_assets_12m = .
replace sold_assets_12m = 1 if asset_S8AQ11 == 1
replace sold_assets_12m = 0 if asset_S8AQ11 == 2
label variable sold_assets_12m "Sold assets in last 12 months"

gen asset_sale_value_12m = asset_S8AQ12
label variable asset_sale_value_12m "Asset sale value in last 12 months"

save "$OUT/final_hies_lfp_earnings_fies_housing_assets.dta", replace
export delimited using "$OUT/final_hies_lfp_earnings_fies_housing_assets.csv", replace

tab _m_assets
count
count if missing(psu) | missing(hhid) | missing(pid)

tab owns_land_property_nonoperated, missing
tab owns_other_assets, missing
summarize land_property_value other_asset_value land_purchase_value_12m other_asset_purchase_value_12m asset_sale_value_12m



* =====================================================* ============================================================
* STEP 26: INSPECT SECTION 9 FILES ONLY
* Purpose: decide whether Section 9 should be used as household
* socioeconomic controls for LFP/earnings analysis.
* ============================================================

local sec9files HH_SEC_9A1 HH_SEC_9B2 HH_SEC_9C HH_SEC_9D1 HH_SEC_9D2 HH_SEC_9E

foreach f of local sec9files {

    display "============================================================"
    display "FILE: `f'"
    display "============================================================"

    use "$ROOT/`f'.dta", clear

    describe, short

    describe PSU HHID, fullnames

    count
    count if missing(PSU) | missing(HHID)

    duplicates report PSU HHID

    lookfor consumption
    lookfor expenditure
    lookfor expense
    lookfor income
    lookfor value
    lookfor total
    lookfor food
    lookfor nonfood
    lookfor item
    lookfor quantity
    lookfor amount
}


* ============================================================
* STEP 27: CREATE SECTION 9 HOUSEHOLD ECONOMIC CONTROLS
* Purpose: household socioeconomic controls for women LFP
*          and earnings inequality analysis
* Output:  $STAGE/sec9_household_controls.dta
* ============================================================


* ----------------------------
* 27A. HH_SEC_9A1: food item value
* ----------------------------

use "$ROOT/HH_SEC_9A1.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen sec9a1_value = S9A2Q08
gen sec9a1_item_row = 1

collapse ///
    (sum) sec9a1_food_value = sec9a1_value ///
          sec9a1_item_count = sec9a1_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9a1_food.dta", replace


* ----------------------------
* 27B. HH_SEC_9B2: food/non-food item value
* ----------------------------

use "$ROOT/HH_SEC_9B2.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen sec9b2_value = S9B2Q08
gen sec9b2_item_row = 1

collapse ///
    (sum) sec9b2_value_total = sec9b2_value ///
          sec9b2_item_count = sec9b2_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9b2_items.dta", replace


* ----------------------------
* 27C. HH_SEC_9C: consumed goods value
* ----------------------------

use "$ROOT/HH_SEC_9C.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen sec9c_cash_credit_value = S9CQ01
gen sec9c_ownprod_received_value = S9CQ02
gen sec9c_item_row = 1

egen sec9c_total_row = rowtotal(sec9c_cash_credit_value sec9c_ownprod_received_value), missing

collapse ///
    (sum) sec9c_cash_credit_value ///
          sec9c_ownprod_received_value ///
          sec9c_total_value = sec9c_total_row ///
          sec9c_item_count = sec9c_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9c_consumed_goods.dta", replace


* ----------------------------
* 27D. HH_SEC_9D1: annual item value
* ----------------------------

use "$ROOT/HH_SEC_9D1.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen sec9d1_value = S9D1Q05
gen sec9d1_amount = S9D1Q04
gen sec9d1_item_row = 1

collapse ///
    (sum) sec9d1_value_total = sec9d1_value ///
          sec9d1_amount_total = sec9d1_amount ///
          sec9d1_item_count = sec9d1_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9d1_annual_items.dta", replace


* ----------------------------
* 27E. HH_SEC_9D2: annual item value
* ----------------------------

use "$ROOT/HH_SEC_9D2.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen sec9d2_value = S9D2Q05
gen sec9d2_amount = S9D2Q04
gen sec9d2_item_row = 1

collapse ///
    (sum) sec9d2_value_total = sec9d2_value ///
          sec9d2_amount_total = sec9d2_amount ///
          sec9d2_item_count = sec9d2_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9d2_annual_items.dta", replace


* ----------------------------
* 27F. HH_SEC_9E: durable asset ownership/value
* ----------------------------

use "$ROOT/HH_SEC_9E.dta", clear

drop if missing(PSU) | missing(HHID)

rename PSU psu
rename HHID hhid

gen durable_owned = .
replace durable_owned = 1 if S9EQ01 == 1
replace durable_owned = 0 if S9EQ01 == 2

gen durable_number = S9EQ02
gen durable_value = S9EQ03
gen durable_item_row = 1

collapse ///
    (sum) durable_owned_count = durable_owned ///
          durable_total_number = durable_number ///
          durable_total_value = durable_value ///
          durable_item_count = durable_item_row, ///
    by(psu hhid)

isid psu hhid

save "$STAGE/sec9e_durable_assets.dta", replace


* ----------------------------
* 27G. Merge all collapsed Section 9 controls
* ----------------------------

use "$STAGE/sec9a1_food.dta", clear

merge 1:1 psu hhid using "$STAGE/sec9b2_items.dta", gen(_m_sec9b2)
merge 1:1 psu hhid using "$STAGE/sec9c_consumed_goods.dta", gen(_m_sec9c)
merge 1:1 psu hhid using "$STAGE/sec9d1_annual_items.dta", gen(_m_sec9d1)
merge 1:1 psu hhid using "$STAGE/sec9d2_annual_items.dta", gen(_m_sec9d2)
merge 1:1 psu hhid using "$STAGE/sec9e_durable_assets.dta", gen(_m_sec9e)

* Combined raw Section 9 socioeconomic indicators
egen sec9_total_observed_value = rowtotal( ///
    sec9a1_food_value ///
    sec9b2_value_total ///
    sec9c_total_value ///
    sec9d1_value_total ///
    sec9d2_value_total), missing

label variable sec9a1_food_value "Section 9A1 total food item value"
label variable sec9b2_value_total "Section 9B2 total item value"
label variable sec9c_total_value "Section 9C total consumed goods value"
label variable sec9d1_value_total "Section 9D1 total annual item value"
label variable sec9d2_value_total "Section 9D2 total annual item value"
label variable durable_owned_count "Number of durable asset types owned"
label variable durable_total_number "Total number of durable items owned"
label variable durable_total_value "Total current value of durable assets"
label variable sec9_total_observed_value "Raw combined Section 9 observed value"

compress

save "$STAGE/sec9_household_controls.dta", replace

count
count if missing(psu) | missing(hhid)
duplicates report psu hhid

summarize sec9a1_food_value sec9b2_value_total sec9c_total_value ///
    sec9d1_value_total sec9d2_value_total durable_owned_count ///
    durable_total_value sec9_total_observed_value

tab _m_sec9b2
tab _m_sec9c
tab _m_sec9d1
tab _m_sec9d2
tab _m_sec9e








* ============================================================
* STEP 28: MERGE SECTION 9 HOUSEHOLD ECONOMIC CONTROLS
* Input:  $OUT/final_hies_lfp_earnings_fies_housing_assets.dta
* Using:  $STAGE/sec9_household_controls.dta
* Output: $OUT/final_hies_women_lfp_earnings_ml_ready.dta / .csv
* ============================================================

use "$OUT/final_hies_lfp_earnings_fies_housing_assets.dta", clear

isid psu hhid pid

merge m:1 psu hhid using "$STAGE/sec9_household_controls.dta", gen(_m_sec9)

tab _m_sec9

count
count if missing(psu) | missing(hhid) | missing(pid)

summarize sec9a1_food_value sec9b2_value_total sec9c_total_value ///
    sec9d1_value_total sec9d2_value_total durable_owned_count ///
    durable_total_value sec9_total_observed_value

save "$OUT/final_hies_women_lfp_earnings_ml_ready.dta", replace
export delimited using "$OUT/final_hies_women_lfp_earnings_ml_ready.csv", replace