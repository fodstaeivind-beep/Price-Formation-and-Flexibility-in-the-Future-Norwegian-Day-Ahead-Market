# ============================================================
# Regression of supply-curve bin volumes on Wind + Hydro
# With time fixed effects and monthly wind coefficients
# ============================================================

# ============================================================
# 0) Load packages
# ============================================================
library(readxl)
library(dplyr)
library(purrr)
library(lubridate)
library(tidyr)
library(fixest)
library(broom)
library(ggplot2)

# ============================================================
# 1) Read and combine monthly Excel files for 2024
#    Folder example: Bins2024/supply_bins_2024_01.xlsx, ..., _12.xlsx
# ============================================================
files <- list.files("Bins2024", full.names = TRUE, pattern = "\\.xlsx$")
df <- map_dfr(files, read_excel)

# ============================================================
# 2) Parse datetime and create time variables
#    parse_date_time handles mixed formats (ymd/dmy and with/without seconds)
# ============================================================
df <- df %>%
  mutate(
    datetime_oslo = if (inherits(datetime_oslo, "POSIXct")) {
      datetime_oslo
    } else {
      parse_date_time(
        as.character(datetime_oslo),
        orders = c("ymd HMS", "dmy HMS", "ymd HM", "dmy HM"),
        tz = "Europe/Oslo"
      )
    }
  )

cat("Failed datetime parses:", sum(is.na(df$datetime_oslo)), "\n")

df <- df %>%
  mutate(
    hour = lubridate::hour(datetime_oslo),
    dow  = lubridate::wday(datetime_oslo, week_start = 1),
    date = as.Date(datetime_oslo),
    month = lubridate::month(datetime_oslo)
  )

# ============================================================
# 3) Rename bin columns to avoid '-' in names (R formula issue)
#    Adjust these names if your Excel column names differ.
# ============================================================
df <- df %>%
  rename(
    bin1 = `bin1_p_lt_-498`,
    bin2 = `bin2_-498_to_-50`,
    bin3 = `bin3_-50_to_0`,
    bin4 = `bin4_0_to_110`,
    bin5 = `bin5_110_to_4000`
  )

# ============================================================
# 4) Column name check for hydro variables
#    IMPORTANT: Adjust these names to match your Excel file.
#
#    Here I assume your dataset contains:
#      - wind_mwh
#      - hydro_res_mwh  (reservoir-based hydro production, MWh)
#      - hydro_ror_mwh  (run-of-river hydro production, MWh)
#
#    If your columns have different names, either rename them here
#    or replace in the regressions below.
# ============================================================
required_cols <- c("wind_mwh", "hydro_res_mwh", "hydro_ror_mwh",
                   "bin1","bin2","bin3","bin4","bin5",
                   "datetime_oslo","hour","dow","date","month")

missing_cols <- setdiff(required_cols, names(df))
if(length(missing_cols) > 0){
  cat("WARNING: Missing columns:\n")
  print(missing_cols)
  cat("\nFix by renaming your hydro columns (or changing names in the script).\n")
}

# ============================================================
# 5) Quick multicollinearity / correlation check
#    (High correlation can make interpretation noisy.)
# ============================================================
if(all(c("wind_mwh","hydro_res_mwh","hydro_ror_mwh") %in% names(df))){
  cat("\nCorrelation matrix (complete cases):\n")
  print(cor(df[, c("wind_mwh","hydro_res_mwh","hydro_ror_mwh")], use="complete.obs"))
}

# ============================================================
# 6) Separate regressions (one model per bin), baseline
#    These are "levels" regressions: bin volume ~ wind + hydro
#    Fixed effects: hour + dow
#    SEs clustered by date
# ============================================================
models_sep <- feols(
  c(bin1, bin2, bin3, bin4, bin5) ~ wind_mwh + hydro_res_mwh + hydro_ror_mwh | hour + dow,
  data = df,
  vcov = ~date
)
etable(models_sep)

# Extract wind coefficients per bin for quick viewing
effects_sep_wind <- data.frame(
  bin  = c("bin1", "bin2", "bin3", "bin4", "bin5"),
  beta_wind = sapply(models_sep, function(m) coef(m)["wind_mwh"])
)
print(effects_sep_wind)

# ============================================================
# 7) Create long ("stacked") dataset for stacked regressions
#    Each hour becomes 5 rows (one per bin): volume + share
# ============================================================
df_long <- df %>%
  pivot_longer(
    cols = starts_with("bin"),
    names_to = "bin",
    values_to = "volume"
  ) %>%
  group_by(datetime_oslo) %>%
  mutate(
    total_volume = sum(volume, na.rm = TRUE),
    share = ifelse(total_volume > 0, volume / total_volume, NA_real_)
  ) %>%
  ungroup() %>%
  mutate(
    bin = factor(bin),
    month = month(datetime_oslo),
    hour  = hour(datetime_oslo),
    dow   = wday(datetime_oslo, week_start = 1),
    date  = as.Date(datetime_oslo)
  )

# bin_month index used to allow separate slopes per bin x month
df_long$bin_month <- interaction(df_long$bin, df_long$month, drop = TRUE)

# ============================================================
# 8) Full-year stacked regression (VOLUMES)
#    Dep var: volume in each bin (stacked)
#
#    i(bin)                  -> bin intercepts
#    i(bin, wind_mwh)         -> wind slope differs by bin
#    i(bin, hydro_res_mwh)    -> reservoir hydro slope differs by bin
#    i(bin, hydro_ror_mwh)    -> run-of-river slope differs by bin
#
#    Fixed effects: hour + dow
# ============================================================
m_year_volume <- feols(
  volume ~
    i(bin) +
    i(bin, wind_mwh) +
    i(bin, hydro_res_mwh) +
    i(bin, hydro_ror_mwh)
  | hour + dow,
  data = df_long,
  vcov = ~date
)
summary(m_year_volume)

# ============================================================
# 9) Full-year stacked regression (SHARES)
#    Dep var: share of total volume in each bin
#
#    Using hour + date FE (strong daily control) like your original.
# ============================================================
m_year_share <- feols(
  share ~
    i(bin) +
    i(bin, wind_mwh) +
    i(bin, hydro_res_mwh) +
    i(bin, hydro_ror_mwh)
  | hour + date,
  data = df_long,
  vcov = ~date
)
summary(m_year_share)

# ============================================================
# 10) Monthly wind coefficients (VOLUMES) + hydro controls
#
#     i(bin_month, wind_mwh)  -> separate wind slope for each (bin x month)
#     i(bin, hydro_*)         -> hydro slopes differ by bin but NOT by month
#
#     This is usually a good bias-variance tradeoff.
# ============================================================
m_bin_month_volume <- feols(
  volume ~
    i(bin) +
    i(bin_month, wind_mwh) +
    i(bin, hydro_res_mwh) +
    i(bin, hydro_ror_mwh)
  | hour + date,
  data = df_long,
  vcov = ~date
)

# Tidy wind coefficients by bin and month
coefs_month_vol <- tidy(m_bin_month_volume) %>%
  filter(grepl("wind_mwh", term)) %>%
  mutate(
    bin   = sub(".*::(bin[^:]+)::.*", "\\1", term),
    month = as.numeric(sub(".*\\.(\\d+)$", "\\1", term)) # fallback parse
  )

# If the month parsing above doesn't work due to term format differences,
# we can parse from df_long$bin_month levels instead. (Ask me if it breaks.)

print(head(coefs_month_vol, 20))

ggplot(coefs_month_vol, aes(x = month, y = estimate, color = bin)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(
    x = "Month",
    y = "Wind effect on volume (per 1 MWh wind)",
    title = "Monthly wind effect on supply volume by price bin (2024), controlling for hydro"
  )

# ============================================================
# 11) Monthly wind coefficients (SHARES) + hydro controls
# ============================================================
m_month_share <- feols(
  share ~
    i(bin) +
    i(bin, wind_mwh, month) +
    i(bin, hydro_res_mwh) +
    i(bin, hydro_ror_mwh)
  | hour + date,
  data = df_long,
  vcov = ~date
)
summary(m_month_share)

coefs_month_share <- tidy(m_month_share) %>%
  filter(grepl("wind_mwh", term)) %>%
  mutate(
    bin   = sub(".*::(bin[^:]+)::.*", "\\1", term),
    month = as.numeric(sub(".*month(\\d+).*", "\\1", term))
  )

ggplot(coefs_month_share, aes(x = month, y = estimate, color = bin)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(
    x = "Month",
    y = "Wind effect on share (per 1 MWh wind)",
    title = "Monthly wind effect on supply curve shares by price bin (2024), controlling for hydro"
  )

# ============================================================
# 12) OPTIONAL: interaction between wind and reservoir hydro
#
#     Rationale: reservoir hydro is flexible and often responds to wind.
#     This lets the wind effect differ depending on reservoir output level.
# ============================================================
m_interaction_volume <- feols(
  volume ~
    i(bin) +
    i(bin, wind_mwh) +
    i(bin, hydro_res_mwh) +
    i(bin, hydro_ror_mwh) +
    i(bin, wind_mwh * hydro_res_mwh)
  | hour + date,
  data = df_long,
  vcov = ~date
)
summary(m_interaction_volume)

# ============================================================
# End
# ============================================================