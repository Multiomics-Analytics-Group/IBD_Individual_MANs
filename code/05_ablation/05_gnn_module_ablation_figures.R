# Metagraph ablation figures (Fig. 7)

.root <- getwd(); while (!file.exists(file.path(.root, "code", "common", "paths.R")) && dirname(.root) != .root) .root <- dirname(.root)
source(file.path(.root, "code", "common", "paths.R"))
suppressMessages(library(tidyverse))

RESULTS  <- RES_ABLATION_GNN
FIG_DIR  <- FIG_ABLATION
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
if (!interactive()) pdf(NULL)  # printed plots go nowhere under Rscript

raw   <- read_csv(file.path(RESULTS, "cross_evaluation_raw_metrics_per_run_modules.csv"), show_col_types = FALSE)
tests <- read_csv(file.path(RESULTS, "main_table_wilcoxon_tests.csv"), show_col_types = FALSE)

METRICS  <- c(AUPRC = "external_auprc", AUROC = "external_auc")
BASELINE <- c(AUPRC = 7 / 25, AUROC = 0.5)  # random classification on Hall (7 non-IBD of 25)
LABEL    <- c(whole = "WG", whole_minus_m1 = "WG - M", m1 = "M only")

stars <- function(q) if_else(q < 0.05, "*", "n.s.")  # * = significant (FDR < 0.05)
fdr_of <- function(section_, training_, evaluation_, metric) {
  tests %>% filter(section == section_, training == training_, evaluation == evaluation_) %>%
    pull(paste0(metric, "_fdr"))
}

# Plot style (as in the paper's figure scripts)
theme_paper <- theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.border = element_rect(color = "#494949ff", fill = NA, linewidth = 0.5),
    axis.text = element_text(size = 13, color = "#494949ff"),
    axis.title = element_text(size = 14, color = "#494949ff"),
    legend.title = element_text(size = 12, color = "#494949ff"),
    legend.text = element_text(size = 11, color = "#494949ff")
  )

# Significance bracket between x1 and x2 at height y
bracket <- function(x1, x2, y, label, tick = 0.015) {
  list(
    annotate("segment", x = x1, xend = x2, y = y, yend = y, color = "#494949ff", linewidth = 0.4),
    annotate("segment", x = c(x1, x2), xend = c(x1, x2), y = y, yend = y - tick, color = "#494949ff", linewidth = 0.4),
    annotate("text", x = (x1 + x2) / 2, y = y + if (label == "n.s.") 0.02 else 0.008, label = label,
             size = if (label == "n.s.") 4 else 5.5, color = "#494949ff")
  )
}

# Panel A: whole-graph models, perturbed at evaluation
# ------------------------------------------------------------------------------
PANEL_A_SETS <- c(whole = "WG", whole_minus_m1 = "WG - M", m1 = "M only", whole_noedges = "WG (no edges)")
eval_colors  <- c("WG" = "#8D99AE", "WG - M" = "#FFB703", "M only" = "#0F4C81", "WG (no edges)" = "#B4CBE2")

panel_A <- function(metric) {
  d <- raw %>%
    filter(train_config == "whole", eval_set %in% names(PANEL_A_SETS)) %>%
    transmute(run = repeat_idx, evaluation = factor(PANEL_A_SETS[eval_set], levels = PANEL_A_SETS),
              value = .data[[METRICS[[metric]]]], x = as.numeric(evaluation))
  top <- max(d$value)
  ggplot(d, aes(x = x, y = value)) +
    geom_hline(yintercept = BASELINE[[metric]], linetype = "dashed", color = "#494949ff", linewidth = 0.5) +
    geom_boxplot(aes(group = evaluation, fill = evaluation), outlier.shape = NA, alpha = 0.6,
                 color = "black", width = 0.45) +
    geom_point(aes(fill = evaluation), shape = 21, color = "black", size = 2.2, alpha = 0.8) +
    bracket(1, 2, top + 0.05, stars(fdr_of("whole", "WG", "WG - M", metric))) +
    bracket(1, 3, top + 0.13, stars(fdr_of("whole", "WG", "M only", metric))) +
    bracket(1, 4, top + 0.21, stars(fdr_of("whole", "WG", "WG (no edges)", metric))) +
    scale_fill_manual(values = eval_colors) +
    scale_x_continuous(breaks = 1:4, labels = str_replace(levels(d$evaluation), " \\(no edges\\)", "\n(no edges)"),
                       expand = expansion(add = 0.5)) +
    scale_y_continuous(limits = c(0, top + 0.26), breaks = seq(0, 1, 0.25)) +
    labs(x = "Evaluation graphs (trained on WG)", y = paste(metric, "(validation dataset)")) +
    theme_paper +
    theme(legend.position = "none")
}

# Panel B: with vs without edges, for each node set
# ------------------------------------------------------------------------------
edge_colors <- c("With edges" = "#0F4C81", "Without edges" = "#B4CBE2")

panel_B <- function(metric) {
  d <- bind_rows(
    raw %>% filter(train_config %in% names(LABEL), eval_set == train_config) %>%
      mutate(node_set = train_config, edges = "With edges"),
    raw %>% filter(train_config %in% paste0(names(LABEL), "_noedges"), eval_set == train_config) %>%
      mutate(node_set = str_remove(train_config, "_noedges"), edges = "Without edges")
  ) %>%
    transmute(run = repeat_idx, node_set = factor(LABEL[node_set], levels = c("WG", "WG - M", "M only")),
              edges = factor(edges, levels = names(edge_colors)), value = .data[[METRICS[[metric]]]]) %>%
    mutate(x = as.numeric(node_set) + if_else(edges == "With edges", -0.2, 0.2))
  top <- max(d$value)
  brackets <- map(seq_along(levels(d$node_set)), function(i) {
    ns <- levels(d$node_set)[i]
    bracket(i - 0.2, i + 0.2, top + 0.05,
            stars(fdr_of("matched", paste0(ns, " (no edges)"), paste0(ns, " (no edges)"), metric)))
  })
  ggplot(d, aes(x = x, y = value)) +
    geom_hline(yintercept = BASELINE[[metric]], linetype = "dashed", color = "#494949ff", linewidth = 0.5) +
    geom_boxplot(aes(group = interaction(node_set, edges), fill = edges), outlier.shape = NA, alpha = 0.6,
                 color = "black", width = 0.32) +
    geom_point(aes(fill = edges), shape = 21, color = "black", size = 2.2, alpha = 0.8) +
    brackets +
    scale_fill_manual(values = edge_colors, name = NULL) +
    scale_x_continuous(breaks = 1:3, labels = levels(d$node_set), expand = expansion(add = 0.5)) +
    scale_y_continuous(limits = c(0, top + 0.12), breaks = seq(0, 1, 0.25)) +
    labs(x = "Graphs used for training and evaluation", y = paste(metric, "(validation dataset)")) +
    theme_paper +
    theme(legend.position = "top")
}

# Build, show and save
# ------------------------------------------------------------------------------
plots <- list(
  panel_A_AUPRC = panel_A("AUPRC"), panel_A_AUROC = panel_A("AUROC"),
  panel_B_AUPRC = panel_B("AUPRC"), panel_B_AUROC = panel_B("AUROC")
)

# Align panels A and B of the same metric: identical layout dimensions (plot area, axes,
# titles and legend rows), so that at the same file size their plot areas coincide and the
# two files can be stacked vertically. Panel A gets an empty band where panel B has its legend.
FIG_WIDTH <- 6.5; FIG_HEIGHT <- 5.5
for (metric in names(METRICS)) {
  a <- paste0("panel_A_", metric); b <- paste0("panel_B_", metric)
  dims <- patchwork::get_max_dim(plots[[a]], plots[[b]])
  plots[[a]] <- patchwork::set_dim(plots[[a]], dims)
  plots[[b]] <- patchwork::set_dim(plots[[b]], dims)
}

for (name in names(plots)) {
  print(plots[[name]])
  ggsave(file.path(FIG_DIR, paste0("ablation_", name, ".pdf")), plots[[name]],
         width = FIG_WIDTH, height = FIG_HEIGHT, dpi = 300)
}
message("Saved: ", paste(paste0("ablation_", names(plots), ".pdf"), collapse = ", "), " in ", FIG_DIR)
