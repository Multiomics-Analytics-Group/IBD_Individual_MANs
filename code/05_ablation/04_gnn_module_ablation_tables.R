# Metagraph ablation tables: Suppl. Table 4 and paired Wilcoxon tests

.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))
suppressMessages(library(tidyverse))

RESULTS <- RES_ABLATION_GNN
raw <- read_csv(file.path(RESULTS, "cross_evaluation_raw_metrics_per_run_modules.csv"), show_col_types = FALSE)

LABEL <- c(whole = "WG", whole_minus_m1 = "WG - M", m1 = "M only",
           whole_noedges = "WG (no edges)", whole_minus_m1_noedges = "WG - M (no edges)",
           m1_noedges = "M only (no edges)")
TRAIN_ORDER <- c("whole", "whole_minus_m1", "m1", "whole_noedges", "whole_minus_m1_noedges", "m1_noedges")
EVAL_ORDER  <- c("whole_minus_m1", "whole", "m1", "whole_minus_m1_noedges", "whole_noedges", "m1_noedges")
METRICS     <- c(AUPRC = "external_auprc", AUROC = "external_auc")
pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

# Per-run values of one training/evaluation pair, ordered by model run
run_values <- function(train, evaluation, metric) {
  v <- raw %>% filter(train_config == train, eval_set == evaluation) %>% arrange(repeat_idx)
  stopifnot(nrow(v) == 50)
  setNames(v[[METRICS[[metric]]]], v$repeat_idx)
}

# Supplementary table: every combination
# ------------------------------------------------------------------------------
supp <- expand_grid(train = TRAIN_ORDER, evaluation = EVAL_ORDER) %>%
  rowwise() %>%
  mutate(n_runs = 50,
         AUPRC_mean = mean(run_values(train, evaluation, "AUPRC")), AUPRC_sd = pop_sd(run_values(train, evaluation, "AUPRC")),
         AUROC_mean = mean(run_values(train, evaluation, "AUROC")), AUROC_sd = pop_sd(run_values(train, evaluation, "AUROC"))) %>%
  ungroup() %>%
  mutate(training_configuration = LABEL[train], evaluation_set = LABEL[evaluation],
         across(c(AUPRC_mean, AUPRC_sd, AUROC_mean, AUROC_sd), ~ round(.x, 3))) %>%
  select(training_configuration, evaluation_set, n_runs, AUPRC_mean, AUPRC_sd, AUROC_mean, AUROC_sd)
write_csv(supp, file.path(RESULTS, "supplementary_table_ablation_edges.csv"))

# Main table: selected rows and paired comparisons
# ------------------------------------------------------------------------------
main_spec <- tribble(
  ~section,  ~train,                   ~evaluation,              ~ref_train,       ~ref_eval,        ~compared_with,
  "matched", "whole",                  "whole",                  NA,               NA,               "--",
  "matched", "whole_minus_m1",         "whole_minus_m1",         "whole",          "whole",          "WG",
  "matched", "m1",                     "m1",                     "whole",          "whole",          "WG",
  "matched", "whole_noedges",          "whole_noedges",          "whole",          "whole",          "WG",
  "matched", "whole_minus_m1_noedges", "whole_minus_m1_noedges", "whole_minus_m1", "whole_minus_m1", "WG - M",
  "matched", "m1_noedges",             "m1_noedges",             "m1",             "m1",             "M only",
  "whole",   "whole",                  "whole",                  NA,               NA,               "--",
  "whole",   "whole",                  "whole_minus_m1",         "whole",          "whole",          "Evaluation on WG",
  "whole",   "whole",                  "m1",                     "whole",          "whole",          "Evaluation on WG",
  "whole",   "whole",                  "whole_noedges",          "whole",          "whole",          "Evaluation on WG"
)

compare <- function(train, evaluation, ref_train, ref_eval, metric) {
  v <- run_values(train, evaluation, metric)
  out <- tibble(mean = mean(v), sd = pop_sd(v))
  if (is.na(ref_train)) return(out %>% mutate(ref_mean = NA, paired_diff = NA, runs_higher = NA, p = NA))
  ref <- run_values(ref_train, ref_eval, metric)
  stopifnot(identical(names(v), names(ref)))  # paired by model run
  out %>% mutate(ref_mean = mean(ref), paired_diff = mean(v - ref), runs_higher = sum(v > ref),
                 p = wilcox.test(v, ref, paired = TRUE, alternative = "two.sided")$p.value)
}

main <- main_spec %>%
  mutate(row = row_number()) %>%
  crossing(metric = names(METRICS)) %>%
  rowwise() %>%
  mutate(res = list(compare(train, evaluation, ref_train, ref_eval, metric))) %>%
  ungroup() %>%
  unnest(res)

# Benjamini-Hochberg across all comparisons in the table (both metrics together)
tested <- !is.na(main$p)
main$fdr <- NA_real_
main$fdr[tested] <- p.adjust(main$p[tested], method = "BH")
message("Tests: ", sum(tested), " (BH across all of them)")

main_wide <- main %>%
  mutate(training = LABEL[train], evaluation_label = LABEL[evaluation]) %>%
  select(row, section, training, evaluation = evaluation_label, compared_with, metric,
         mean, sd, ref_mean, paired_diff, runs_higher, p, fdr) %>%
  pivot_wider(names_from = metric, values_from = c(mean, sd, ref_mean, paired_diff, runs_higher, p, fdr),
              names_glue = "{metric}_{.value}") %>%
  arrange(row)
write_csv(main_wide %>% select(-row), file.path(RESULTS, "main_table_wilcoxon_tests.csv"))

# LaTeX
# ------------------------------------------------------------------------------
mean_sd <- function(m, s) sprintf("%.3f \\(\\pm\\) %.3f", m, s)
fmt_fdr <- function(q) {
  if (is.na(q)) return("--")
  if (q >= 0.001) return(sprintf("%.3f", q))
  parts <- strsplit(sprintf("%.1e", q), "e")[[1]]
  sprintf("\\(%s{\\times}10^{%d}\\)", parts[1], as.integer(parts[2]))
}
short <- function(x) str_replace(x, " \\(no edges\\)", " (NE)")  # compact labels for the main table

# Compact layout: one configuration column per section (training = evaluation in the first
# section; training = WG in the second), short "no edges" labels, footnotesize
titles <- c(matched = "Trained and evaluated on the same configuration",
            whole = "Trained on WG, evaluated on")
body <- c()
for (sec in names(titles)) {
  if (sec == "whole") body <- c(body, "      \\midrule")
  body <- c(body, sprintf("      \\multicolumn{6}{l}{\\textit{%s}} \\\\", titles[[sec]]))
  sub <- main_wide %>% filter(section == sec)
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    config <- if (sec == "matched") r$training else r$evaluation
    compared <- if (r$compared_with == "Evaluation on WG") "WG" else r$compared_with
    body <- c(body, sprintf("      %s & %s & %s & %s & %s & %s \\\\",
                            short(config), mean_sd(r$AUPRC_mean, r$AUPRC_sd),
                            mean_sd(r$AUROC_mean, r$AUROC_sd), short(compared),
                            fmt_fdr(r$AUPRC_fdr), fmt_fdr(r$AUROC_fdr)))
  }
}

latex <- c(
  "\\begin{table}[htbp]", "  \\centering", "  \\begin{threeparttable}",
  "    \\caption{Performance of GNNs trained and evaluated on ablated graph configurations on the validation dataset (Mean \\(\\pm\\) SD).}",
  "    \\label{tab:ablation_main}", "    \\small",
  "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcclcc}", "      \\toprule",
  "      & & & & \\multicolumn{2}{c}{\\textbf{FDR}\\tnote{c}} \\\\", "      \\cmidrule(lr){5-6}",
  "      \\textbf{Configuration}\\tnote{d} & \\textbf{AUPRC}\\tnote{a} & \\textbf{AUROC}\\tnote{b} & \\textbf{Compared with} & \\textbf{AUPRC} & \\textbf{AUROC} \\\\",
  "      \\midrule",
  body,
  "      \\bottomrule", "    \\end{tabular*}", "    \\begin{tablenotes}", "      \\footnotesize",
  "      \\item WG, whole graph; M, module; NE, no edges. Results for all training and evaluation combinations are shown in Supplementary Table X.",
  "      \\item[a] Baseline AUPRC for random classification equals the proportion of non-IBD individuals in the validation dataset (7 of 25; 0.28).",
  "      \\item[b] Baseline AUROC for random classification is 0.50.",
  "      \\item[c] Two-sided Wilcoxon signed-rank tests paired by model run, adjusted using the Benjamini--Hochberg procedure across all comparisons in the table.",
  "      \\item[d] NE: all associations removed from the graphs; nodes and node attributes are kept.",
  "    \\end{tablenotes}", "  \\end{threeparttable}", "\\end{table}")
cat(latex, sep = "\n")

cat("\n\nPaired comparisons (mean paired difference, runs higher of 50, raw p, FDR):\n")
main_wide %>% filter(!is.na(AUPRC_p)) %>%
  transmute(comparison = paste0(training, " -> ", evaluation, " vs ", compared_with),
            AUPRC = sprintf("%+.3f (%d/50, p=%.2g, FDR=%.2g)", AUPRC_paired_diff, AUPRC_runs_higher, AUPRC_p, AUPRC_fdr),
            AUROC = sprintf("%+.3f (%d/50, p=%.2g, FDR=%.2g)", AUROC_paired_diff, AUROC_runs_higher, AUROC_p, AUROC_fdr)) %>%
  print(n = Inf, width = 200)
